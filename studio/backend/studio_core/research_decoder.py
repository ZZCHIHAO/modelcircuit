"""Bounded, local decoding for research datasets.

A decoded sample is float32 [C,D,H,W]. Spatial resize samples voxel centres
using linear interpolation. Video D indexes uniformly selected source frames.
No patient names, UIDs, dates, or other source headers enter the metadata.
"""
import io
import math
from pathlib import Path

import numpy as np

from .classification_data import _cancel, _read_source, IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

FORMATS = frozenset({'image', 'gray16', 'dicom', 'nifti', 'video'})
SOURCE_BYTES_LIMIT = 512 * 1024 * 1024
SOURCE_VOXELS_LIMIT = 32 * 1024 * 1024
OUTPUT_VALUES_LIMIT = 32 * 1024 * 1024
MAX_SERIES_FILES = 1024
MAX_VIDEO_FRAMES = 4096
VIDEO_EXTENSIONS = frozenset({'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm'})


def _settings(spec):
    format = spec.get('format')
    if format not in FORMATS:
        raise ValueError('Unknown research image format: ' + str(format))
    size = spec.get('image_size', [64, 64])
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(value) is not int or not 1 <= value <= 512 for value in size)):
        raise ValueError('image_size must be [height,width], each 1..512')
    depth = spec.get('depth', 1)
    channels = spec.get('channels', 1)
    if type(depth) is not int or not 1 <= depth <= 128:
        raise ValueError('depth must be an integer in 1..128')
    if type(channels) is not int or channels not in (1, 3):
        raise ValueError('channels must be 1 or 3')
    if format in ('image', 'gray16') and depth != 1:
        raise ValueError('Static images require depth=1')
    if format in ('gray16', 'dicom', 'nifti') and channels != 1:
        raise ValueError('Medical image decoding requires one grayscale channel')
    if depth * size[0] * size[1] * channels > OUTPUT_VALUES_LIMIT:
        raise ValueError('Decoded sample exceeds the output voxel budget')
    lower, upper = spec.get('intensity_min'), spec.get('intensity_max')
    if (lower, upper) != (None, None):
        if (type(lower) not in (int, float) or type(upper) not in (int, float)
                or not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper
                or abs(lower) > np.finfo(np.float32).max or abs(upper) > np.finfo(np.float32).max):
            raise ValueError('Specify both intensity bounds as finite numbers with min < max')
        if format in ('image', 'video'):
            raise ValueError('Intensity bounds apply only to gray16, DICOM, or NIfTI')
    expected = spec.get('modality', 'auto')
    if expected not in (None, '', 'auto', 'CT', 'MR', 'MRI', 'OCT'):
        raise ValueError('modality must be auto, CT, MR, MRI, or OCT')
    return format, (depth, *size), channels, lower, upper, {'MRI': 'MR'}.get(expected, expected)


def _source_budget(path):
    if not path.is_file() or path.stat().st_size > SOURCE_BYTES_LIMIT:
        raise ValueError('Research data source must be a local file within the byte limit')


def _voxel_budget(shape):
    if any(value < 1 for value in shape) or math.prod(shape) > SOURCE_VOXELS_LIMIT:
        raise ValueError('Research data source exceeds the decoded voxel budget')


def _normalise(array, lower, upper, *, invert=False):
    if not np.isfinite(array).all():
        raise ValueError('Research data contains non-finite intensities')
    source_bounds = [float(array.min()), float(array.max())]
    if lower is None:
        lower, upper = source_bounds
        mode = 'per_sample_minmax'
    else:
        mode = 'fixed_clip'
    if upper == lower:
        output = np.zeros(array.shape, dtype=np.float32)
    else:
        # Difference of two finite float32 endpoints can exceed float32 range.
        scaled = np.clip(array.astype(np.float64), lower, upper)
        scaled -= lower
        scaled /= upper - lower
        output = scaled.astype(np.float32)
    if invert:
        output = 1.0 - output
    return output, {'source_intensity_bounds': source_bounds, 'intensity_bounds': [float(lower), float(upper)],
                    'intensity_mode': mode, 'inverted': invert}


def _resize_axis(array, axis, length):
    count = array.shape[axis]
    if count == length:
        return array
    positions = np.clip((np.arange(length) + 0.5) * count / length - 0.5, 0, count - 1)
    first = positions.astype(np.intp)
    last = np.minimum(first + 1, count - 1)
    view = [1] * array.ndim
    view[axis] = length
    weight = (positions - first).astype(np.float32).reshape(view)
    left = np.take(array, first, axis=axis)
    right = np.take(array, last, axis=axis)
    right -= left
    right *= weight
    left += right
    return left


def _resize_volume(array, target, cancelled):
    # Shrink dimensions first to keep intermediate arrays inside the budget.
    for axis in sorted(range(3), key=lambda i: target[i] / array.shape[i]):
        _cancel(cancelled)
        array = _resize_axis(array, axis, target[axis])
    return array


def _static_image(path, format, target, channels, lower, upper, cancelled):
    from PIL import Image, UnidentifiedImageError

    supported = IMAGE_EXTENSIONS if format == 'image' else {'.png', '.tif', '.tiff'}
    if path.suffix.lower() not in supported:
        raise ValueError(f'Unsupported {format} image extension')
    raw = _read_source(path, MAX_IMAGE_BYTES, cancelled)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, 'n_frames', 1) != 1 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError('Expected a bounded single-frame image')
            if format == 'gray16':
                if not (image.mode.startswith('I;16') or image.mode == 'I'):
                    raise ValueError('gray16 requires an integer PNG or TIFF above 8 bits')
                values = np.asarray(image, dtype=np.float32)
                if float(values.min()) < 0 or float(values.max()) > 65535:
                    raise ValueError('gray16 expects unsigned grayscale values from 0 to 65535')
                source_bounds = [0, 65535] if lower is None else [lower, upper]
                x, metadata = _normalise(values[np.newaxis], *source_bounds)
                if lower is None:
                    metadata['intensity_mode'] = 'uint16_range'
                output = _resize_volume(x, target, cancelled)[np.newaxis]
                metadata.update(source_shape_dhw=[1, image.height, image.width], source_mode=image.mode)
                return output, metadata
            if image.mode.startswith('I;16') or image.mode in ('I', 'F'):
                raise ValueError('High-bit grayscale must use format gray16')
            resized = image.convert('L' if channels == 1 else 'RGB').resize((target[2], target[1]), Image.Resampling.BILINEAR)
            values = np.asarray(resized, dtype=np.float32) / 255.0
            output = values[np.newaxis, np.newaxis] if channels == 1 else values.transpose(2, 0, 1)[:, np.newaxis]
            return output, {'source_shape_dhw': [1, image.height, image.width], 'source_mode': image.mode,
                            'intensity_mode': 'uint8_range', 'intensity_bounds': [0.0, 255.0]}
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError('Cannot decode the static image') from exc


def _nifti(path, cancelled):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ValueError('NIfTI requires nibabel. Install the research input dependencies.') from exc

    if not path.name.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError('NIfTI requires a .nii or .nii.gz file')
    _source_budget(path)
    try:
        source = nib.load(str(path))
        if len(source.shape) != 3:
            raise ValueError('Expected a single 3D NIfTI volume')
        _voxel_budget(source.shape)
        if source.affine is None or not np.isfinite(source.affine).all() or np.linalg.det(source.affine[:3, :3]) == 0:
            raise ValueError('NIfTI affine is invalid')
        _cancel(cancelled)
        volume = nib.as_closest_canonical(source)
        data = volume.get_fdata(dtype=np.float32, caching='unchanged').transpose(2, 1, 0)
        spacing = [float(item) for item in nib.affines.voxel_sizes(volume.affine)[::-1]]
        spatial_unit, _ = volume.header.get_xyzt_units()
        return data, {'orientation': ''.join(nib.aff2axcodes(volume.affine)), 'source_spacing_dhw': spacing,
                      'affine_ras_xyz': volume.affine.tolist(), 'spatial_unit': spatial_unit,
                      'source_shape_dhw': list(data.shape), 'geometry': 'canonical_nifti'}
    except (nib.filebasedimages.ImageFileError, OSError) as exc:
        raise ValueError('Cannot decode the NIfTI volume') from exc


def _numbers(value, count, name):
    try:
        found = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid DICOM {name}') from exc
    if found.shape != (count,) or not np.isfinite(found).all():
        raise ValueError(f'Invalid DICOM {name}')
    return found


def _functional_value(dataset, frame, sequence, field):
    per_frame = dataset.get('PerFrameFunctionalGroupsSequence') or ()
    shared = dataset.get('SharedFunctionalGroupsSequence') or ()
    containers = ([per_frame[frame]] if frame < len(per_frame) else []) + list(shared[:1])
    for container in containers:
        values = container.get(sequence) or ()
        if values and field in values[0]:
            return values[0].get(field)
    return dataset.get(field)


def _ct_rescale(dataset, frame):
    """Return one documented linear HU transform for a classic or enhanced CT frame."""
    from pydicom.uid import CTImageStorage, EnhancedCTImageStorage, LegacyConvertedEnhancedCTImageStorage

    sop_class = dataset.get('SOPClassUID')
    if sop_class != dataset.file_meta.get('MediaStorageSOPClassUID'):
        raise ValueError('HU calibration requires consistent DICOM CT storage metadata')
    classic = sop_class == CTImageStorage
    if not classic and sop_class not in (EnhancedCTImageStorage, LegacyConvertedEnhancedCTImageStorage):
        raise ValueError('HU calibration requires a recognized DICOM CT storage class')
    if dataset.get('ModalityLUTSequence'):
        raise ValueError('CT HU calibration requires linear rescale without a modality LUT')
    transform = dataset
    image_type = dataset.get('ImageType')
    if not classic:
        per_frame = dataset.get('PerFrameFunctionalGroupsSequence') or ()
        shared = dataset.get('SharedFunctionalGroupsSequence') or ()
        if len(shared) > 1:
            raise ValueError('Enhanced CT has ambiguous shared HU calibration')
        groups = [per_frame[frame]] if frame < len(per_frame) else []
        groups.extend(shared)
        transforms = [group.get('PixelValueTransformationSequence') for group in groups
                      if 'PixelValueTransformationSequence' in group]
        if len(transforms) != 1 or not transforms[0] or len(transforms[0]) != 1:
            raise ValueError('Enhanced CT requires one HU rescale per frame')
        transform = transforms[0][0]
        image_type = _functional_value(dataset, frame, 'CTImageFrameTypeSequence', 'FrameType') or image_type

    image_type = tuple(image_type) if image_type and not isinstance(image_type, str) else ()
    if len(image_type) >= 3 and image_type[2] == 'LOCALIZER':
        raise ValueError('CT LOCALIZER images cannot form an HU volume')
    # PS3.3 C.8.2.1 permits an implicit HU unit for classic original,
    # non-localizer, non-multi-energy CT. C.8.15.3.10 requires RescaleType
    # in the enhanced Pixel Value Transformation Macro.
    unit = transform.get('RescaleType')
    implicit_hu = (classic and unit is None and len(image_type) >= 3 and image_type[0] == 'ORIGINAL'
                   and bool(image_type[2]) and dataset.get('MultienergyCTAcquisition') in (None, 'NO'))
    if unit != 'HU' and not implicit_hu:
        raise ValueError('CT rescale units must be documented as HU')
    try:
        slope, intercept = float(transform.get('RescaleSlope')), float(transform.get('RescaleIntercept'))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('CT HU calibration requires both a rescale slope and intercept') from exc
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
        raise ValueError('CT HU rescale must be finite with a nonzero slope')
    return slope, intercept


def _dicom_pixels(dataset, cancelled, *, strict_ct=False):
    from pydicom.pixels import apply_modality_lut

    try:
        data = dataset.pixel_array
    except (RuntimeError, NotImplementedError, ValueError, AttributeError, ImportError) as exc:
        raise ValueError('Cannot decode DICOM pixels. Check the transfer syntax and installed codecs.') from exc
    if data.ndim == 2:
        data = data[np.newaxis]
    frames = int(dataset.get('NumberOfFrames', 1))
    if data.shape != (frames, int(dataset.Rows), int(dataset.Columns)):
        raise ValueError('DICOM requires monochrome 2D slices or a monochrome volume')
    result = np.empty(data.shape, dtype=np.float32)
    for i, frame in enumerate(data):
        _cancel(cancelled)
        if strict_ct:
            slope, intercept = _ct_rescale(dataset, i)
            with np.errstate(over='ignore', invalid='ignore'):
                result[i] = frame.astype(np.float64) * slope + intercept
            continue
        slope = _functional_value(dataset, i, 'PixelValueTransformationSequence', 'RescaleSlope')
        intercept = _functional_value(dataset, i, 'PixelValueTransformationSequence', 'RescaleIntercept')
        if dataset.get('ModalityLUTSequence'):
            result[i] = apply_modality_lut(frame, dataset)
        elif slope is not None or intercept is not None:
            if slope is None or intercept is None or not math.isfinite(float(slope)) or not math.isfinite(float(intercept)) or float(slope) == 0:
                raise ValueError('DICOM intensity rescale slope and intercept must both be finite, with nonzero slope')
            result[i] = frame.astype(np.float64) * float(slope) + float(intercept)
        else:
            result[i] = frame
    return result


def _dicom_geometry(datasets, counts, modality, *, strict_ct=False):
    orientations, positions, spacings = [], [], []
    for dataset, count in zip(datasets, counts):
        per_frame = dataset.get('PerFrameFunctionalGroupsSequence') or ()
        if per_frame and len(per_frame) != count:
            raise ValueError('DICOM per-frame metadata count does not match the pixel frames')
        for i in range(count):
            orientations.append(_functional_value(dataset, i, 'PlaneOrientationSequence', 'ImageOrientationPatient'))
            # A shared origin is valid for a single plane, not all frames.
            positions.append(_functional_value(dataset, i, 'PlanePositionSequence', 'ImagePositionPatient'))
            spacings.append(_functional_value(dataset, i, 'PixelMeasuresSequence', 'PixelSpacing'))
    count = sum(counts)
    complete = all(value is not None for value in orientations + positions + spacings)
    if not complete:
        if len(datasets) > 1 or (count > 1 and modality in ('CT', 'MR')):
            raise ValueError('DICOM volume requires slice positions, orientation, and pixel spacing')
        # OCT files may encode a vendor-defined scan geometry without patient
        # positions. Preserve stored frame order and disclose unavailable geometry.
        return list(range(count)), {'geometry': 'frame_order_only', 'source_spacing_dhw': None}
    direction = _numbers(orientations[0], 6, 'orientation').reshape(2, 3)
    if (not np.allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-3)
            or abs(direction[0] @ direction[1]) > 1e-3):
        raise ValueError('DICOM direction cosines must be orthonormal')
    plane_spacing = _numbers(spacings[0], 2, 'pixel spacing')
    if np.any(plane_spacing <= 0):
        raise ValueError('DICOM pixel spacing must be positive')
    for orient, spacing in zip(orientations[1:], spacings[1:]):
        if (not np.allclose(_numbers(orient, 6, 'orientation'), direction.ravel(), atol=1e-4, rtol=0)
                or not np.allclose(_numbers(spacing, 2, 'pixel spacing'), plane_spacing, rtol=1e-4, atol=1e-4)):
            raise ValueError('DICOM slices disagree on orientation or pixel spacing')
    if strict_ct:
        # Remove small DICOM decimal rounding of the orthonormal axes before
        # projecting positions or deriving the depth column of the affine.
        direction[0] /= np.linalg.norm(direction[0])
        direction[1] -= (direction[0] @ direction[1]) * direction[0]
        direction[1] /= np.linalg.norm(direction[1])
    normal = np.cross(*direction)
    origins = np.asarray([_numbers(item, 3, 'position') for item in positions])
    projections = (origins - origins[0] if strict_ct else origins) @ normal
    order = np.argsort(projections, kind='stable')
    axial_spacing = None
    if count > 1:
        intervals = np.diff(projections[order])
        # Some OCT IODs report a single plane plus OCT-specific geometry.
        if len(datasets) == 1 and modality in ('OCT', 'OPT') and np.all(abs(intervals) < 1e-5):
            return list(range(count)), {'geometry': 'frame_order_only', 'source_spacing_dhw': [None, *plane_spacing.tolist()]}
        axial_spacing = float(np.median(intervals))
        if axial_spacing < 1e-4 or not np.allclose(intervals, axial_spacing, atol=0.05, rtol=0.02):
            raise ValueError('DICOM slice positions are duplicated or not regularly spaced')
        displacements = origins[order] - origins[order[0]]
        depth_displacements = (projections[order] - projections[order[0]])[:, None] * normal
        if not np.allclose(displacements, depth_displacements, atol=0.1, rtol=0):
            raise ValueError('DICOM slice origins have in-plane displacement; tilted volumes need prior resampling')
        if strict_ct:
            axial_spacing = float((projections[order[-1]] - projections[order[0]]) / (count - 1))
            fitted = np.arange(count)[:, None] * axial_spacing * normal
            tolerance = min(1e-3, 1e-3 * min(axial_spacing, *plane_spacing))
            if np.any(intervals <= 0) or np.any(np.linalg.norm(fitted - displacements, axis=1) > tolerance):
                raise ValueError('CT slice displacement cannot be represented on one regular spatial affine')
    return order.tolist(), {'geometry': 'dicom_patient_lps', 'orientation_cosines': direction.ravel().tolist(),
                            'origin_lps_xyz': origins[order[0]].tolist(), 'spatial_unit': 'mm',
                            'source_spacing_dhw': [axial_spacing, *plane_spacing.tolist()]}


def _dicom(path, expected, cancelled, *, strict_ct=False):
    try:
        import pydicom
    except ImportError as exc:
        raise ValueError('DICOM requires pydicom. Install the research input dependencies.') from exc

    paths = [path]
    if path.is_dir():
        paths = []
        for entry in path.iterdir():
            _cancel(cancelled)
            if entry.is_symlink() or not entry.is_file():
                raise ValueError('DICOM series must be a flat directory of local slice files')
            paths.append(entry)
            if len(paths) > MAX_SERIES_FILES:
                raise ValueError('DICOM series exceeds the file count limit')
        paths.sort()
        if not paths:
            raise ValueError('DICOM series contains no slices')
    total_bytes = 0
    for source in paths:
        _source_budget(source)
        total_bytes += source.stat().st_size
        if total_bytes > SOURCE_BYTES_LIMIT:
            raise ValueError('DICOM series exceeds the total byte limit')
    datasets, counts = [], []
    series, modality, size, polarity, frame_of_reference = None, None, None, None, None
    for source in paths:
        _cancel(cancelled)
        try:
            dataset = pydicom.dcmread(str(source))
        except (pydicom.errors.InvalidDicomError, OSError) as exc:
            raise ValueError('Cannot open a DICOM slice in the selected source') from exc
        typ = str(dataset.get('Modality', ''))
        # OPT is the ophthalmic tomography modality code used by DICOM.
        if typ == 'OPT':
            typ = 'OCT'
        if typ not in ('CT', 'MR', 'OCT'):
            raise ValueError('Supported DICOM modalities are OCT, CT, and MR')
        if expected not in (None, '', 'auto') and typ != expected:
            raise ValueError('DICOM modality does not match the expected modality')
        candidate_uid = dataset.get('SeriesInstanceUID')
        geometry_uid = dataset.get('FrameOfReferenceUID')
        if len(paths) > 1 and (not candidate_uid or (series is not None and candidate_uid != series)):
            raise ValueError('DICOM directory must contain exactly one series')
        if datasets and frame_of_reference != geometry_uid:
            raise ValueError('DICOM series contains different frames of reference')
        try:
            dims = int(dataset.Rows), int(dataset.Columns)
            frames = int(dataset.get('NumberOfFrames', 1))
            photometric = str(dataset.PhotometricInterpretation)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError('Missing DICOM pixel dimensions or frame metadata') from exc
        if photometric not in ('MONOCHROME1', 'MONOCHROME2') or int(dataset.get('SamplesPerPixel', 1)) != 1:
            raise ValueError('DICOM research input supports monochrome samples')
        if frames < 1:
            raise ValueError('DICOM frame count must be positive')
        if len(paths) > 1 and frames != 1:
            raise ValueError('A DICOM series directory requires single-frame slices. Use a volume file directly.')
        if dims != (size or dims) or typ != (modality or typ) or photometric != (polarity or photometric):
            raise ValueError('DICOM series slices disagree on shape, modality, or intensity polarity')
        _voxel_budget((sum(counts) + frames, *dims))
        series, modality, size, polarity, frame_of_reference = candidate_uid, typ, dims, photometric, geometry_uid
        datasets.append(dataset)
        counts.append(frames)
    if strict_ct and sum(counts) < 2:
        raise ValueError('CT volume requires at least two slices')
    order, metadata = _dicom_geometry(datasets, counts, modality, strict_ct=strict_ct)
    data = np.concatenate([_dicom_pixels(dataset, cancelled, strict_ct=strict_ct) for dataset in datasets])
    if order != list(range(len(data))):
        data = data[order]
    units = 'HU' if modality == 'CT' else 'modality_intensity'
    metadata.update(modality=modality, intensity_unit=units, source_shape_dhw=list(data.shape),
                    photometric=polarity)
    return data, metadata


def _video(path, target, channels, cancelled):
    try:
        import cv2
    except ImportError as exc:
        raise ValueError('Video requires opencv-python-headless. Install the research input dependencies.') from exc

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError('Unsupported local video extension')
    _source_budget(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open local video with the available codecs')
        values = [capture.get(prop) for prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)]
        if not all(math.isfinite(value) and value >= 1 for value in values):
            raise ValueError('Video must provide a frame count and dimensions')
        count, height, width = (int(value) for value in values)
        if count > MAX_VIDEO_FRAMES:
            raise ValueError(f'Video exceeds {MAX_VIDEO_FRAMES} frames. Trim or segment it before import.')
        _voxel_budget((height, width, 3))
        depth, out_height, out_width = target
        if depth > count:
            raise ValueError('Requested video depth exceeds the number of available frames')
        indexes = [count // 2] if depth == 1 else np.rint(np.linspace(0, count - 1, depth)).astype(int).tolist()
        selected = set(indexes)
        output = np.empty((channels, depth, out_height, out_width), dtype=np.float32)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        written = 0
        # Decode sequentially so results do not depend on backend seek accuracy.
        for i in range(indexes[-1] + 1):
            _cancel(cancelled)
            if not capture.grab():
                raise ValueError('Video ended before all requested frames could be read')
            if i not in selected:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame.shape != (height, width, 3):
                raise ValueError('Invalid frame or variable video frame geometry')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if channels == 1 else cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
            output[:, written] = (frame[np.newaxis] if channels == 1 else frame.transpose(2, 0, 1)) / 255.0
            written += 1
        return output, {'source_shape_dhw': [count, height, width], 'frame_indices': indexes,
                        'fps': fps if math.isfinite(fps) and fps > 0 else None, 'geometry': 'video_frame_order',
                        'intensity_mode': 'uint8_range', 'intensity_bounds': [0.0, 255.0]}
    finally:
        capture.release()


def decode(path, spec, cancelled=None):
    """Decode a research sample and record its geometry and intensity transform."""
    _cancel(cancelled)
    path = Path(path)
    format, target, channels, lower, upper, expected = _settings(spec)
    if format in ('image', 'gray16'):
        output, metadata = _static_image(path, format, target, channels, lower, upper, cancelled)
    elif format == 'video':
        output, metadata = _video(path, target, channels, cancelled)
    else:
        if format == 'dicom':
            volume, metadata = _dicom(path, expected, cancelled)
            invert = metadata['photometric'] == 'MONOCHROME1'
        else:
            volume, metadata = _nifti(path, cancelled)
            invert = False
        volume, intensity = _normalise(volume, lower, upper, invert=invert)
        metadata.update(intensity)
        output = _resize_volume(volume, target, cancelled)[np.newaxis]
    metadata.update(format=format, axis_order=['C', 'D', 'H', 'W'], output_shape=list(output.shape),
                    resampling='linear_voxel_centres' if format in ('gray16', 'dicom', 'nifti') else 'bilinear',
                    depth_axis='time' if format == 'video' else 'slice')
    spacing = metadata.get('source_spacing_dhw')
    if spacing is not None:
        metadata['resampled_spacing_dhw'] = [None if pitch is None else pitch * count / out
                                             for pitch, count, out in zip(spacing, metadata['source_shape_dhw'], target)]
    _cancel(cancelled)
    return np.ascontiguousarray(output, dtype=np.float32), metadata
