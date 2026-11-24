"""Bounded medical volume geometry in RAS world coordinates.

Arrays and shapes use depth, height, width (D,H,W). Each affine maps width,
height, depth (X,Y,Z voxel indices) to RAS coordinates at voxel centres.
"""
import gzip
from itertools import product
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .research_decoder import SOURCE_BYTES_LIMIT, SOURCE_VOXELS_LIMIT, _source_budget, _voxel_budget


_UNIT_IN_MM = {'mm': 1., 'meter': 1e3, 'micron': 1e-3}
_SPATIAL_CODES = frozenset({1, 2, 3, 4, 5})
_TOLERANCE_MM = 1e-3
_TOLERANCE_VOXELS = 1e-3
_RESAMPLE_CHUNK = 64 * 1024


def _grid_shape(grid):
    try:
        shape = tuple(grid['source_shape_dhw'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Medical grid requires a three-dimensional D,H,W shape') from exc
    if len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                              for n in shape):
        raise ValueError('Medical grid requires a three-dimensional integer D,H,W shape')
    shape = tuple(map(int, shape))
    _voxel_budget(shape)
    return shape


def _check_affine(affine):
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.array_equal(affine[3], [0, 0, 0, 1])):
        raise ValueError('Medical volume requires a finite 4×4 spatial affine')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        lengths = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Medical spatial affine requires finite positive axis lengths')
    # Relative singular values expose degenerate directions and also protect
    # the physical coordinate solves from loss of voxel precision.
    if np.linalg.cond(affine[:3, :3]) > 1e12:
        raise ValueError('Medical spatial affine is singular or ill-conditioned')
    return lengths


def affine_mm(grid):
    """Return the voxel-centre affine in RAS millimeters."""
    unit = grid.get('spatial_unit')
    if not isinstance(unit, str) or unit not in _UNIT_IN_MM:
        raise ValueError('Medical volume requires a known spatial unit (mm, meter or micron)')
    try:
        affine = np.array(grid['affine_xyz'], dtype=np.float64, copy=True)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Invalid medical spatial affine') from exc
    if affine.shape != (4, 4):
        raise ValueError('Medical volume requires a 4×4 spatial affine')
    with np.errstate(over='ignore', under='ignore'):
        affine[:3] *= _UNIT_IN_MM[unit]
    _check_affine(affine)
    return affine


def spacing_mm(grid):
    """Return the three axis lengths in D,H,W order."""
    affine = affine_mm(grid)
    return np.linalg.norm(affine[:3, :3], axis=0)[::-1].tolist()


def _read_nifti_header(path, nib):
    # Inspect the raw header before nibabel can repair invalid fields or treat
    # an infinite slope as unscaled. Only read the fixed-size header prefix.
    open_volume = gzip.open if path.name.lower().endswith('.gz') else open
    with open_volume(path, 'rb') as stream:
        prefix = stream.read(nib.Nifti2Header.sizeof_hdr)
    for header_type in (nib.Nifti2Header, nib.Nifti1Header):
        if header_type.may_contain_header(prefix):
            header = header_type(prefix[:header_type.sizeof_hdr], check=False)
            if bytes(header['magic']).rstrip(b'\0') != header_type.single_magic.rstrip(b'\0'):
                continue
            return header
    raise ValueError('Medical volume requires a valid single-file NIfTI header')


def read_nifti(path):
    """Read finite real voxels and geometry from the original NIfTI grid."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ValueError('NIfTI requires nibabel. Install the research input dependencies.') from exc
    from .segmentation_data import _nifti

    path = Path(path)
    if not path.name.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError('Medical volume requires a .nii or .nii.gz file')
    _source_budget(path)
    try:
        header = _read_nifti_header(path, nib)
        slope, intercept = float(header['scl_slope']), float(header['scl_inter'])
        # A zero or undefined (NaN) slope disables NIfTI intensity scaling.
        if math.isinf(slope) or (math.isfinite(slope) and slope != 0 and not math.isfinite(intercept)):
            raise ValueError('Invalid NIfTI intensity scaling')
        offset = float(header['vox_offset'])
        if not math.isfinite(offset) or not offset.is_integer() or not 0 <= offset <= SOURCE_BYTES_LIMIT:
            raise ValueError('Invalid NIfTI voxel offset')
        sform_code, qform_code = int(header['sform_code']), int(header['qform_code'])
        code = sform_code or qform_code
        if code not in _SPATIAL_CODES:
            raise ValueError('NIfTI needs a valid coded spatial affine (sform or qform)')
        # NIfTI treats a zero qfac as +1. Check the active form before relying
        # on nibabel's validation, which can otherwise repair negative pixdims.
        if sform_code == 0 and header['pixdim'][0] == 0:
            header['pixdim'][0] = 1
        try:
            unit = header.get_xyzt_units()[0]
        except KeyError as exc:
            raise ValueError('Invalid NIfTI spatial unit') from exc
        native_affine = header.get_best_affine()
        affine_mm({'affine_xyz': native_affine, 'spatial_unit': unit})
        with np.errstate(over='ignore', invalid='ignore'):
            # Supplying a source guard also detaches voxel data from on-disk
            # memory maps, including when the file contains unscaled integers.
            values, geometry = _nifti(path, source_guard=lambda shape, dtype: _voxel_budget(shape))
        if (geometry['spatial_unit'] != unit or geometry['affine_code'] != code
                or not np.array_equal(geometry['affine_xyz'], native_affine)):
            raise ValueError('NIfTI spatial geometry changed while loading')
    except (OSError, EOFError, nib.filebasedimages.ImageFileError, nib.spatialimages.HeaderDataError) as exc:
        raise ValueError('Cannot read the NIfTI header, spatial affine, or voxel data') from exc
    if not np.isfinite(values).all():
        raise ValueError('NIfTI scaling produced non-finite voxels or the source contains non-finite voxels')
    return values, geometry


def read_ct(path, cancelled=None):
    """Read strictly calibrated HU on a regular CT grid in patient RAS."""
    from .research_decoder import _dicom

    data, metadata = _dicom(Path(path), 'CT', cancelled, strict_ct=True)
    if data.shape[0] < 2:
        raise ValueError('CT volume requires at least two slices')
    if metadata['geometry'] != 'dicom_patient_lps':
        raise ValueError('CT volume requires a patient spatial affine')
    if not np.isfinite(data).all():
        raise ValueError('CT calibration produced non-finite HU')
    first_row, first_column = np.asarray(metadata['orientation_cosines']).reshape(2, 3)
    depth_spacing, row_spacing, column_spacing = metadata['source_spacing_dhw']
    # PS3.3 C.7.6.2 maps column index along the first row direction,
    # and row index along the first column direction. PixelSpacing is row first.
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((first_row * column_spacing, first_column * row_spacing,
                                     np.cross(first_row, first_column) * depth_spacing))
    affine[:3, 3] = metadata['origin_lps_xyz']
    affine[:2] *= -1  # DICOM patient LPS to RAS.
    _check_affine(affine)
    # DICOM position and orientation may have more precision than a NIfTI-1 sform.
    geometry = {'source_shape_dhw': list(data.shape), 'affine_xyz': affine.tolist(),
                'spatial_unit': 'mm', 'affine_code': 1, 'nifti_version': 2}
    return data, geometry


def _corners(shape, *, compare_singleton_axes=False):
    return np.array(list(product(*[(0, max(int(compare_singleton_axes), size - 1)) for size in shape[::-1]])))


def assert_same_grid(reference, labels):
    """Reject a label map whose voxel centres differ from the native reference."""
    shape = _grid_shape(reference)
    if _grid_shape(labels) != shape:
        raise ValueError('Medical image and label grid shapes differ')
    image, mask = affine_mm(reference), affine_mm(labels)
    difference = mask[:3] - image[:3]
    mm_delta = _corners(shape, compare_singleton_axes=True) @ difference[:, :3].T + difference[:, 3]
    voxel_delta = np.linalg.solve(image[:3, :3], mm_delta.T)
    if np.any(np.abs(mm_delta) > _TOLERANCE_MM) or np.any(np.abs(voxel_delta) > _TOLERANCE_VOXELS):
        raise ValueError('Medical label affine does not align with the original reference grid')


def _voxel_transform(source, target):
    """Map target XYZ voxel coordinates to source XYZ without inverting world origins."""
    with np.errstate(over='ignore', invalid='ignore'):
        linear = np.linalg.solve(source[:3, :3], target[:3, :3])
        offset = np.linalg.solve(source[:3, :3], target[:3, 3] - source[:3, 3])
    if not np.isfinite(linear).all() or not np.isfinite(offset).all():
        raise ValueError('Medical affine transform is not finite')
    return linear, offset


def assert_contains(reference, channel):
    """Check the channel's outer voxel faces cover all reference voxel centres."""
    reference_shape, channel_shape = _grid_shape(reference), _grid_shape(channel)
    channel_affine = affine_mm(channel)
    linear, offset = _voxel_transform(channel_affine, affine_mm(reference))
    coordinates = _corners(reference_shape) @ linear.T + offset
    # A coordinate row of the inverse affine gives the distance-to-face scale.
    # Use both tolerances so that tiny voxels cannot hide a sampling shift.
    inverse = np.linalg.inv(channel_affine[:3, :3])
    tolerance = np.minimum(_TOLERANCE_VOXELS, _TOLERANCE_MM * np.linalg.norm(inverse, axis=1))
    upper = np.asarray(channel_shape[::-1]) - 0.5
    if (not np.isfinite(coordinates).all() or np.any(coordinates < -0.5 - tolerance)
            or np.any(coordinates > upper + tolerance)):
        raise ValueError('MRI channel field of view does not cover the reference voxel centres')


def make_model_grid(reference, spacing_dhw_mm):
    """Create a grid whose outer voxel faces cover the centered reference FOV."""
    shape = np.asarray(_grid_shape(reference)[::-1], dtype=np.float64)
    native = affine_mm(reference)
    try:
        spacing = np.asarray(spacing_dhw_mm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Model spacing requires three positive finite millimeter values') from exc
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Model spacing requires three positive finite millimeter values')
    spacing = spacing[::-1]
    native_spacing = np.linalg.norm(native[:3, :3], axis=0)
    with np.errstate(over='ignore', under='ignore'):
        extent = shape * (native_spacing / spacing)
    if not np.isfinite(extent).all() or np.any(extent > SOURCE_VOXELS_LIMIT):
        raise ValueError('Medical model grid exceeds the decoded voxel budget')
    # Round up without admitting an additional voxel from a one-ULP error
    # at an exact integer. A very coarse spacing still leaves one voxel.
    target_shape = tuple(map(int, np.maximum(1, np.ceil(np.nextafter(extent, -np.inf)))))
    _voxel_budget(target_shape)
    model = np.eye(4)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        model[:3, :3] = native[:3, :3] * (spacing / native_spacing)
        native_midpoint = native[:3, :3] @ ((shape - 1) / 2)
        model_midpoint = model[:3, :3] @ ((np.asarray(target_shape) - 1) / 2)
        model[:3, 3] = native[:3, 3] + (native_midpoint - model_midpoint)
    _check_affine(model)
    return {'source_shape_dhw': list(target_shape[::-1]), 'affine_xyz': model.tolist(),
            'spatial_unit': 'mm', 'affine_code': reference['affine_code'],
            'nifti_version': reference['nifti_version']}


def resample_volume(data, source, target, *, order):
    """Pull samples from source voxel centres with clamped edge extension.

    Nearest sampling breaks half-voxel ties toward the larger source index
    and keeps the exact input dtype. Trilinear images are returned as float32.
    Temporary coordinates and interpolation weights are bounded by chunk size.
    """
    source_shape, target_shape = _grid_shape(source), _grid_shape(target)
    values = np.asarray(data)
    if values.shape != source_shape or values.dtype.kind not in 'biuf':
        raise ValueError('Resampled medical volume must be real numeric and match the source grid shape')
    if order not in (0, 1):
        raise ValueError('Medical resampling order must be 0 for labels or 1 for images')
    source_affine, target_affine = affine_mm(source), affine_mm(target)
    dtype = values.dtype if order == 0 else np.float32
    if source_shape == target_shape and np.array_equal(source_affine, target_affine):
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.ascontiguousarray(values, dtype=dtype)
    else:
        linear, offset = _voxel_transform(source_affine, target_affine)
        linear, offset = linear[::-1, ::-1], offset[::-1, None]
        last = np.asarray(source_shape)[:, None] - 1
        out = np.empty(target_shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, out.size, _RESAMPLE_CHUNK):
            stop = min(out.size, start + _RESAMPLE_CHUNK)
            index = np.unravel_index(np.arange(start, stop, dtype=np.intp), target_shape)
            with np.errstate(over='ignore', invalid='ignore'):
                coords = linear @ np.asarray(index, dtype=np.float64) + offset
            if not np.isfinite(coords).all():
                raise ValueError('Medical affine transform generated non-finite voxel coordinates')
            np.clip(coords, 0, last, out=coords)
            if order == 0:
                nearest = np.floor(coords + 0.5).astype(np.intp)
                flat[start:stop] = values[tuple(nearest)]
            else:
                lower = coords.astype(np.intp)
                upper = np.minimum(lower + 1, last)
                weight = coords - lower
                # Accumulate in float64 so interpolation between finite float32
                # extrema does not overflow during subtraction or summation.
                total = np.zeros(stop - start, dtype=np.float64)
                with np.errstate(over='ignore', invalid='ignore'):
                    for corner in product((0, 1), repeat=3):
                        indices = tuple(upper[axis] if high else lower[axis] for axis, high in enumerate(corner))
                        wd, wh, ww = (weight[axis] if high else 1 - weight[axis] for axis, high in enumerate(corner))
                        total += values[indices] * wd * wh * ww
                    flat[start:stop] = total
    if order == 1 and not np.isfinite(out).all():
        raise ValueError('Medical image resampling produced non-finite values')
    return out


def write_native_labels(path, labels, grid):
    """Atomically write integer labels at their exact original voxel centres."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ValueError('NIfTI requires nibabel. Install the research input dependencies.') from exc

    path = Path(path)
    suffix = '.nii.gz' if path.name.lower().endswith('.nii.gz') else '.nii'
    if not path.name.lower().endswith(suffix):
        raise ValueError('Native labels require a .nii or .nii.gz path')
    labels = np.asarray(labels)
    if labels.shape != _grid_shape(grid) or labels.dtype.kind not in 'biu':
        raise ValueError('Native labels must be integers on the original grid shape')
    affine_mm(grid)  # Validate without converting or rounding the stored native affine.
    affine = np.asarray(grid['affine_xyz'], dtype=np.float64)
    code, version = grid.get('affine_code'), grid.get('nifti_version')
    if code not in _SPATIAL_CODES or version not in (1, 2):
        raise ValueError('Native labels require a valid spatial affine code and NIfTI version')
    # An affine decoded from a NIfTI-1 qform can contain more precision than
    # a NIfTI-1 sform can encode. NIfTI-2 preserves those coordinates exactly.
    with np.errstate(over='ignore', under='ignore'):
        needs_double = not np.array_equal(affine, affine.astype(np.float32))
    image_type = nib.Nifti2Image if version == 2 or needs_double else nib.Nifti1Image
    dtype = np.dtype(np.uint8) if labels.dtype.kind == 'b' else labels.dtype
    image = image_type(labels.transpose(2, 1, 0), affine, dtype=dtype)
    image.set_sform(affine, code=code)
    image.header.set_xyzt_units(grid['spatial_unit'])
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=suffix)
    os.close(descriptor)
    try:
        nib.save(image, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
