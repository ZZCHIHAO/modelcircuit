"""Evaluate medical segmentation on the native reference grid of each case."""
from contextlib import contextmanager
import csv
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np

from .medical_volume_data import read_medical_case
from .segmentation_inference import _taper, _windows
from .segmentation_predictions import _preview


CASE_METRICS_FILENAME = 'medical_case_metrics.csv'
_CASE_FIELDS = ('split', 'sample_id', 'class_index', 'class_value', 'shape_dhw', 'spacing_dhw_mm', 'affine_xyz_mm',
                'target_voxels', 'predicted_voxels', 'intersection', 'dice', 'iou', 'hd95_mm', 'empty_status',
                'mask_file', 'preview_file', 'slice_index', 'slice_count')


def _sliding_case(image, labels, window, classes, overlap, predict, check):
    """Return a stitched hard mask and summed model-grid cross-entropy."""
    shape = image.shape[1:]
    starts = _windows(shape, window, overlap)
    taper = _taper(window)
    slab_depth = min(shape[0], window[0])
    weights = np.zeros((slab_depth, *shape[1:]), dtype=np.float32)
    logits = np.zeros((classes, *weights.shape), dtype=np.float32)
    mask = np.empty(shape, dtype=np.uint8)
    loss_sum = 0.0
    for index, leading in enumerate(starts[0]):
        for others in product(*starts[1:]):
            check()
            offsets = leading, *others
            lengths = tuple(min(width, size - start) for width, size, start in zip(window, shape, offsets))
            local = tuple(slice(0, length) for length in lengths)
            region = tuple(slice(start, start + length) for start, length in zip(offsets, lengths))
            patch = image[(slice(None), *region)]
            padding = tuple((0, width - length) for width, length in zip(window, lengths))
            if any(after for _, after in padding):
                patch = np.pad(patch, ((0, 0), *padding), mode='edge')
            tile = predict(patch)
            slab_region = (local[0], *region[1:])
            with np.errstate(over='raise', invalid='raise'):
                logits[(slice(None), *slab_region)] += tile[(slice(None), *local)] * taper[local]
            weights[slab_region] += taper[local]

        # Subsequent tiles cannot affect rows preceding their first depth start.
        next_start = starts[0][index + 1] if index + 1 < len(starts[0]) else shape[0]
        advance = next_start - leading
        for offset in range(advance):
            check()
            depth = leading + offset
            mask[depth] = logits[:, offset].argmax(axis=0)
            row = logits[:, offset].astype(np.float64)
            row /= weights[offset]
            row -= row.max(axis=0)
            target = np.take_along_axis(row, labels[depth][None], axis=0)[0]
            np.exp(row, out=row)
            loss_sum += float((np.log(row.sum(axis=0)) - target).sum())
        carry = slab_depth - advance
        if carry > 0:
            logits[:, :carry] = logits[:, advance:].copy()
            weights[:carry] = weights[advance:].copy()
        logits[:, carry:].fill(0)
        weights[carry:].fill(0)
    return mask, loss_sum


def _orthogonal_spacing(linear):
    spacing = np.linalg.norm(linear, axis=0)
    unit = linear / spacing
    if np.allclose(unit.T @ unit, np.eye(3), rtol=0, atol=1e-12):
        return spacing[::-1]
    return None


def _hd95_mm(target, prediction, affine_xyz_mm, check):
    """Max directed 95th percentile between 6-neighbour surface voxel centres."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    from scipy.spatial import cKDTree

    check()
    region = []
    for axis in range(3):
        reductions = tuple(other for other in range(3) if other != axis)
        occupied = np.flatnonzero(target.any(axis=reductions) | prediction.any(axis=reductions))
        region.append(slice(occupied[0], occupied[-1] + 1))
    target, prediction = target[tuple(region)], prediction[tuple(region)]
    structure = generate_binary_structure(3, 1)
    target_surface = target & ~binary_erosion(target, structure=structure, border_value=0)
    prediction_surface = prediction & ~binary_erosion(prediction, structure=structure, border_value=0)
    linear = np.asarray(affine_xyz_mm, dtype=np.float64)[:3, :3]
    spacing = _orthogonal_spacing(linear)
    if spacing is not None:
        def directed(source, destination):
            check()
            distances = distance_transform_edt(~destination, sampling=spacing)[source]
            return float(np.percentile(distances, 95, method='linear'))

        return max(directed(target_surface, prediction_surface), directed(prediction_surface, target_surface))

    # A shared translation cancels from all distances. Reversing the indices
    # maps array order (D,H,W) to the affine's voxel index order (X,Y,Z).
    target_points = np.argwhere(target_surface)[:, ::-1] @ linear.T
    prediction_points = np.argwhere(prediction_surface)[:, ::-1] @ linear.T

    def directed_points(source, destination):
        check()
        tree = cKDTree(destination)
        distances = np.empty(len(source), dtype=np.float64)
        for start in range(0, len(source), 65536):
            check()
            block = slice(start, start + 65536)
            distances[block] = tree.query(source[block])[0]
        return float(np.percentile(distances, 95, method='linear'))

    return max(directed_points(target_points, prediction_points), directed_points(prediction_points, target_points))


def _measure_class(labels, mask, index, value, affine_xyz_mm, check):
    target, predicted = labels == index, mask == index
    target_voxels, predicted_voxels = int(np.count_nonzero(target)), int(np.count_nonzero(predicted))
    hd95_mm = None
    if target_voxels and predicted_voxels:
        intersection = int(np.count_nonzero(target & predicted))
        empty = 'both_present'
        hd95_mm = 0.0 if intersection == target_voxels == predicted_voxels else _hd95_mm(target, predicted, affine_xyz_mm, check)
    else:
        intersection = 0
        if target_voxels:
            empty = 'prediction_empty'
        elif predicted_voxels:
            empty = 'target_empty'
        else:
            empty = 'both_empty'
    total = target_voxels + predicted_voxels
    return {'class_index': index, 'class_value': value, 'target_voxels': target_voxels, 'predicted_voxels': predicted_voxels,
            'intersection': intersection, 'dice': 2 * intersection / total if total else None,
            'iou': intersection / (total - intersection) if total else None, 'hd95_mm': hd95_mm, 'empty_status': empty}


def _class_totals(value, index):
    return {'index': index, 'class_value': value, 'target_voxels': 0, 'predicted_voxels': 0, 'intersection': 0,
            'dice': 0.0, 'iou': 0.0, 'hd95_mm': 0.0, 'valid_overlap_cases': 0, 'valid_hd95_cases': 0,
            'both_empty_cases': 0, 'single_empty_cases': 0}


def _add_case_class(total, measured):
    for key in ('target_voxels', 'predicted_voxels', 'intersection'):
        total[key] += measured[key]
    if measured['dice'] is not None:
        total['valid_overlap_cases'] += 1
        total['dice'] += measured['dice']
        total['iou'] += measured['iou']
        if measured['hd95_mm'] is None:
            total['single_empty_cases'] += 1
        else:
            total['valid_hd95_cases'] += 1
            total['hd95_mm'] += measured['hd95_mm']
    else:
        total['both_empty_cases'] += 1


def _final_metrics(totals, count, voxels, loss_sum, loss_voxels):
    metrics = {'count': count, 'voxels': voxels, 'loss': loss_sum / loss_voxels, 'loss_voxels': loss_voxels,
               'space': 'original',
               'original_space': {
                   'version': 1,
                   'grid': 'native_reference',
                   'spatial_unit': 'mm',
                   'aggregation': 'mean_over_defined_case_class_pairs',
                   'hd95': 'max_directed_95th_percentile_of_6_neighbour_surface_voxel_centres',
                   'hd95_quantile': 'linear',
                   'both_empty': 'all_metrics_undefined',
                   'single_empty': 'dice_and_iou_zero_hd95_undefined',
                   'tile_padding': 'edge_after',
                   'loss_grid': 'model',
                   'loss': 'voxel_mean_cross_entropy_after_tapered_sliding_logit_blending',
               }, 'per_class': totals}
    for metric, count_key in (('dice', 'valid_overlap_cases'), ('iou', 'valid_overlap_cases'), ('hd95_mm', 'valid_hd95_cases')):
        denominator = sum(row[count_key] for row in totals[1:])
        metrics['foreground_' + metric] = sum(row[metric] for row in totals[1:]) / denominator if denominator else None
        for row in totals:
            row[metric] = row[metric] / row[count_key] if row[count_key] else None
    return metrics


@contextmanager
def _case_csv(directory):
    if directory is None:
        yield None
        return
    target = directory / CASE_METRICS_FILENAME
    temporary = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', suffix='.tmp',
                                            prefix=target.name + '.', dir=directory, delete=False)
    path = Path(temporary.name)
    try:
        with temporary as stream:
            writer = csv.DictWriter(stream, fieldnames=_CASE_FIELDS)
            writer.writeheader()
            yield writer
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)


def evaluate_medical_split(model, task, manifest, split, device, check, *, output_directory=None, progress=None):
    """Evaluate full held-out cases, retaining one case and one slab of logits at a time."""
    import torch
    from .medical_geometry import affine_mm, resample_volume, write_native_labels

    records = [record for record in manifest['records'] if record['split'] == split]
    if not records:
        raise ValueError(f'Medical-volume {split} split is empty')
    values = manifest['mask_values']
    window = tuple(manifest['input_shape'][2:])
    spec = manifest['data_spec']
    totals = [_class_totals(value, index) for index, value in enumerate(values)]
    size = 0
    loss_voxels = 0
    loss_sum = 0.0
    rows = []
    directory = None if output_directory is None else Path(output_directory)
    stored_values = None
    if directory is not None:
        (directory / 'masks').mkdir(parents=True, exist_ok=True)
        stored_values = np.asarray(values, dtype=np.uint8 if max(values) <= 255 else np.uint16)

    def cancelled():
        check()
        return False

    def predict(patch):
        batch = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).to(device)
        logits = model(**{task.input_port: batch})[task.output_name]
        if logits.dtype != torch.float32 or tuple(logits.shape) != (1, len(values), *window):
            raise ValueError('Runtime mask_logits does not match the medical patch grid or class count')
        result = logits[0].detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite medical segmentation mask_logits')
        return result

    model.eval()
    with torch.no_grad(), _case_csv(directory) as writer:
        for number, record in enumerate(records, 1):
            check()
            case = read_medical_case(record, spec, cancelled=cancelled)
            reference = case['reference_grid']
            image, labels = case['image'], case['labels']
            try:
                model_mask, case_loss = _sliding_case(image, labels, window, len(values), spec['overlap'], predict, check)
            except FloatingPointError as exc:
                raise ValueError(f'Non-finite sliding logits for medical case {record["id"]}') from exc
            loss_voxels += labels.size
            loss_sum += case_loss
            check()
            native_mask = resample_volume(model_mask, case['model_grid'], reference, order=0)
            native_labels = case['native_labels']
            affine = affine_mm(reference)
            size += native_labels.size
            measured = []
            for index, value in enumerate(values):
                check()
                stats = _measure_class(native_labels, native_mask, index, value, affine, check)
                _add_case_class(totals[index], stats)
                measured.append(stats)
            if writer is not None:
                check()
                name = f'{number:06d}'
                mask_file = f'masks/{name}.nii.gz'
                preview_file = f'masks/{name}-preview.png'
                write_native_labels(directory / mask_file, stored_values[native_mask], reference)
                slice_index, slice_count = _preview(directory / preview_file, case['native_image'][:1], native_labels, native_mask)
                row = {'sample_id': record['id'], 'mask_file': mask_file, 'preview_file': preview_file,
                       'slice_index': slice_index, 'slice_count': slice_count}
                rows.append(row)
                geometry = {'shape_dhw': json.dumps(native_labels.shape),
                            'spacing_dhw_mm': json.dumps(np.linalg.norm(affine[:3, :3], axis=0)[::-1].tolist()),
                            'affine_xyz_mm': json.dumps(affine.tolist())}
                writer.writerows({**row, **stats, **geometry, 'split': split} for stats in measured)
            del case, image, labels, model_mask, native_mask, native_labels
            if progress is not None:
                progress(number, len(records))
        check()
    return _final_metrics(totals, len(records), size, loss_sum, loss_voxels), rows
