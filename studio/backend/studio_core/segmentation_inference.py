"""Fixed-grid and overlapping native-grid inference using a saved segmentation run."""
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .classification_data import IMAGE_EXTENSIONS, _read_source
from .classification_runtime import _write_json
from .devices import memory_snapshot, resolve_device
from .experiment_spec import checkpoint_signature, model_digest, validate_supervised_config
from .inference_spec import PENDING_REPORT, PROFILE, REPORT_NAME, validate_inference_config
from .memory_admission import admit_memory
from .schema import prepare_graph
from .segmentation_data import float_image, load_native_image, normalize_image, resize_image, resize_labels
from .segmentation_predictions import _write_native_mask
from .segmentation_task import SegmentationTask

MAX_SPEC_BYTES = 4 * 1024 ** 2
MAX_INFERENCE_FILES = 2048
VIEW_EDGE = 224


def window_starts(length, width, overlap):
    if length <= width:
        return (0,)
    step = max(1, math.floor(width * (1 - overlap)))
    edge = length - width
    starts = tuple(range(0, edge + 1, step))
    return starts if starts[-1] == edge else (*starts, edge)


def _windows(shape, window, overlap):
    return tuple(window_starts(length, width, overlap) for length, width in zip(shape, window))


def _taper(window):
    weights = np.ones(window, dtype=np.float32)
    for axis, width in enumerate(window):
        ramp = np.maximum(.1, 1 - np.abs(np.linspace(-1, 1, width, dtype=np.float32)))
        broadcast = [1] * len(window)
        broadcast[axis] = width
        weights *= ramp.reshape(broadcast)
    return weights


def sliding_mask(image, window, classes, overlap, predict, check, progress=None):
    """Sweep along the first spatial axis and finalize rows no later tile can change."""
    native = image.shape[1:]
    starts = _windows(native, window, overlap)
    count = math.prod(map(len, starts))
    weights = _taper(window)
    slab_depth = min(native[0], window[0])
    slab = np.zeros((classes, slab_depth, *native[1:]), dtype=np.float32)
    mask = np.empty(native, dtype=np.uint8)
    completed = 0
    for index, leading in enumerate(starts[0]):
        for others in product(*starts[1:]):
            check()
            offsets = (leading, *others)
            lengths = tuple(min(width, size - offset) for size, offset, width in zip(native, offsets, window))
            local = tuple(slice(0, length) for length in lengths)
            region = tuple(slice(start, start + length) for start, length in zip(offsets, lengths))
            patch = image[(slice(None), *region)]
            padding = tuple((0, width - length) for length, width in zip(lengths, window))
            if any(after for _, after in padding):
                patch = np.pad(patch, ((0, 0), *padding), mode='edge')
            logits = predict(patch)
            target = (slice(None), local[0], *region[1:])
            with np.errstate(over='raise', invalid='raise'):
                slab[target] += logits[(slice(None), *local)] * weights[local]
            completed += 1
            if progress is not None:
                progress(completed, count)
        next_start = starts[0][index + 1] if index + 1 < len(starts[0]) else native[0]
        advance = next_start - leading
        # A positive weight sum is identical for every class at a voxel, so
        # argmax of the weighted sum equals argmax of the weighted mean.
        for offset in range(advance):
            check()
            mask[leading + offset] = slab[:, offset].argmax(axis=0)
        carry = slab_depth - advance
        if carry:
            slab[:, :carry] = slab[:, advance:].copy()
        slab[:, carry:].fill(0)
    return mask


def _digest_stream(stream, check):
    digest = hashlib.sha256()
    while block := stream.read(1024 ** 2):
        check()
        digest.update(block)
    return digest.hexdigest()


def _host_admission(needed, operation):
    available = memory_snapshot('cpu')['available_bytes']
    if needed > max(0, available - max(available // 10, 64 * 1024 ** 2)):
        raise ValueError(f'Not enough available CPU memory for {operation} ({needed:,} bytes estimated). Close other jobs or use fixed-grid inference.')


def _load_model(config, check):
    import torch
    from .runtime import build_model

    source = Path(config['run_spec']).expanduser().resolve()
    raw = _read_source(source, MAX_SPEC_BYTES, lambda: check() or False)
    try:
        saved = json.loads(raw)
        if not isinstance(saved, dict) or saved.get('version') != 3 or saved.get('profile') != 'segmentation-v1':
            raise ValueError('Select run_spec.json from a saved segmentation training run')
        graph = saved['graph']
        training = validate_supervised_config(saved['config'])
        if training['profile'] != 'segmentation-v1' or saved.get('adapter') != {'id': 'segmentation', 'version': 1}:
            raise ValueError('Run specification does not describe a segmentation model')
        spec = training['data']
        dataset = saved['dataset_digest']
        if not isinstance(dataset, str) or len(dataset) != 64 or any(c not in '0123456789abcdef' for c in dataset):
            raise ValueError('Run specification has an invalid dataset digest')
        shape = (spec['channels'], *([spec['depth']] if spec['format'] == 'nifti' else []), *spec['image_size'])
        model_id = model_digest(graph)
        identity = checkpoint_signature(graph, training, dataset)
        # The data manifest resolves paths. Only the preprocessing fields travel with a model.
        preprocessing_keys = ('format', 'mask_values', 'channels', 'depth', 'image_size')
        effective = saved['data_spec']
        if (model_id != saved['model_digest'] or identity != saved['checkpoint_signature']
                or any(spec[key] != effective[key] for key in preprocessing_keys)):
            raise ValueError('Saved graph, dataset or preprocessing identity mismatch')
        resolved, shapes, parameters = prepare_graph(graph)
        SegmentationTask(training).validate_contract(resolved, shapes, {'input_shape': [None, *shape], 'mask_values': spec['mask_values']})
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError('Invalid segmentation run specification: ' + str(exc)) from exc

    checkpoint = Path(config['checkpoint']).expanduser().resolve() if config['checkpoint'] else source.with_name('best.pt')
    if checkpoint == source or not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
        raise ValueError('Select the matching segmentation checkpoint (.pt). The default is best.pt alongside run_spec.json.')
    device = resolve_device(config['device'])
    memory = admit_memory(resolved, device, batch_size=1, capture=False)
    _host_admission(checkpoint.stat().st_size + parameters * 4, 'checkpoint loading')
    try:
        with checkpoint.open('rb') as stream:
            before = checkpoint.stat()
            digest = _digest_stream(stream, check)
            stream.seek(0)
            state = torch.load(stream, map_location='cpu', weights_only=True)
            after = checkpoint.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError('Checkpoint changed while it was being read')
    except InterruptedError:
        raise
    except Exception as exc:
        raise ValueError('Could not read the segmentation checkpoint: ' + str(exc)) from exc
    if (not isinstance(state, dict) or state.get('version') != 3 or state.get('profile') != 'segmentation-v1'
            or state.get('adapter') != {'id': 'segmentation', 'version': 1}
            or state.get('checkpoint_signature') != identity or state.get('model_digest') != model_id
            or state.get('dataset_digest') != dataset or state.get('mask_values') != spec['mask_values']
            or state.get('input_shape') != [None, *shape] or not isinstance(state.get('model_state'), dict)):
        raise ValueError('Checkpoint identity does not match the saved segmentation run')
    model = build_model(graph, device=device, capture=False, memory_report=memory)
    model.load_state_dict(state['model_state'], strict=True)
    del state
    model.eval()
    provenance = {'model_digest': model_id, 'training_dataset_digest': dataset, 'checkpoint_signature': identity,
                  'checkpoint_sha256': digest, 'run_spec_sha256': hashlib.sha256(raw).hexdigest(),
                  'checkpoint_name': checkpoint.name}
    return model, device, training, shape[1:], provenance, memory['warnings']


def _source_paths(value, fmt):
    path = Path(value).expanduser().resolve()

    def supported(item):
        return item.name.lower().endswith(('.nii', '.nii.gz')) if fmt == 'nifti' else item.suffix.lower() in IMAGE_EXTENSIONS

    if path.is_file() and supported(path):
        return [path]
    if not path.is_dir():
        raise ValueError('Select an image file or directory matching the saved model format: ' + fmt)
    cases = []
    for item in path.iterdir():
        if item.is_file() and not item.is_symlink() and not item.name.startswith('.') and supported(item):
            if not item.resolve().is_relative_to(path):
                raise ValueError('An inference file resolves outside the selected folder')
            cases.append(item)
            if len(cases) > MAX_INFERENCE_FILES:
                raise ValueError(f'Select at most {MAX_INFERENCE_FILES} input files in one inference job')
    if not cases:
        raise ValueError('No files in the selected folder match the saved model format: ' + fmt)
    return sorted(cases, key=lambda item: (item.name.casefold(), item.name))


def _view_shape(shape):
    scale = min(1, VIEW_EDGE / max(shape[-2:]))
    return max(1, round(shape[-2] * scale)), max(1, round(shape[-1] * scale))


def _preview_labels(labels, size):
    if labels.shape == tuple(size):
        return labels
    centres = resize_labels(labels, size)
    edges = [np.arange(width) * source // width for source, width in zip(labels.shape, size)]
    # Protect small foreground regions that otherwise disappear between sample centres.
    pooled = np.maximum.reduceat(labels, edges[0], axis=0)
    pooled = np.maximum.reduceat(pooled, edges[1], axis=1)
    return np.where(centres != 0, centres, pooled)


def _write_views(path, image, mask, check):
    from PIL import Image

    h, w = _view_shape(mask.shape)
    images = image[:, None] if mask.ndim == 2 else image
    masks = mask[None] if mask.ndim == 2 else mask
    # Write frames sequentially so the desktop can memory-map only the requested slice.
    with path.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {'descr': '|u1', 'fortran_order': False,
                                                      'shape': (len(masks), h, w, 4)})
        for index, labels in enumerate(masks):
            check()
            rgb = (images[:, index].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            if rgb.shape[-1] == 1:
                rgb = rgb.repeat(3, axis=2)
            if rgb.shape[:2] != (h, w):
                rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
            frame = np.empty((h, w, 4), dtype=np.uint8)
            frame[..., :3] = rgb
            frame[..., 3] = _preview_labels(labels, (h, w))
            stream.write(memoryview(frame))


def run_segmentation_inference(config, artifact_directory, progress=None, cancelled=None, *, defer_report=False):
    import torch

    settings = validate_inference_config(config)
    started = time.monotonic()

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('Segmentation inference cancelled')
        if time.monotonic() - started > settings['max_runtime_seconds']:
            raise TimeoutError('Segmentation inference exceeded its time limit')

    def report(phase, step, total, tiles=0, tile_total=0):
        if progress is not None:
            progress({'phase': phase, 'step': step, 'total': total, 'batch': tiles, 'batches': tile_total})

    check()
    report('load', 0, 0)
    model, device, training, window, provenance, warnings = _load_model(settings, check)
    spec = training['data']
    classes = len(spec['mask_values'])
    files = _source_paths(settings['source'], spec['format'])
    directory = Path(artifact_directory).resolve()
    masks = directory / 'masks'
    views = directory / 'views'
    try:
        masks.mkdir(parents=True)
        views.mkdir()
    except FileExistsError as exc:
        raise ValueError('Choose a new inference artifact directory for each job') from exc

    def predict(patch):
        with torch.inference_mode():
            argument = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))[None].to(device)
            logits = model(**{training['input_port']: argument})['mask_logits']
            if tuple(logits.shape) != (1, classes, *window) or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                raise ValueError('Model produced invalid segmentation logits for an input window')
            return logits[0].cpu().numpy()

    def guard_decode(shape, dtype):
        if any(value < 1 for value in shape) or dtype.kind not in 'biuf':
            raise ValueError('Input must be a nonempty real-valued volume')
        _host_admission(math.prod(shape) * (max(dtype.itemsize, 8) + 28), 'volume decoding and normalization')

    cases = []
    for index, path in enumerate(files, 1):
        check()
        report('decode', index - 1, len(files))
        before = path.stat()
        image, geometry = load_native_image(path, spec, source_guard=guard_decode)
        with path.open('rb') as stream:
            source_sha256 = _digest_stream(stream, check)
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError('Input changed during decoding: ' + path.name)
        image = normalize_image(float_image(image), spec)
        native = image.shape[1:]
        h, w = _view_shape(native)
        depth = native[0] if len(native) == 3 else 1
        if shutil.disk_usage(directory).free < depth * h * w * 4 + math.prod(native) * 2 + (4 << 20):
            raise ValueError('Insufficient disk space for segmentation predictions and slice previews')
        if settings['mode'] == 'sliding':
            overlap = settings['overlap']
            tile_count = math.prod(map(len, _windows(native, window, overlap)))
            slab_bytes = 4 * classes * min(native[0], window[0]) * math.prod(native[1:])
            _host_admission(2 * slab_bytes + math.prod(native) + 4 * (classes + 2) * math.prod(window), 'sliding window score accumulation')

            def tile_progress(done, total):
                if done == total or done % max(1, total // 64) == 0:
                    report('sliding', index - 1, len(files), done, total)

            prediction = sliding_mask(image, window, classes, overlap, predict, check, tile_progress)
        else:
            tile_count = 1
            report('fixed', index - 1, len(files), 0, 1)
            prediction = resize_labels(predict(resize_image(image, window)).argmax(axis=0).astype(np.uint8), native)
        check()
        suffix = '.nii.gz' if spec['format'] == 'nifti' else '.png'
        mask_file = f'masks/{index:06d}{suffix}'
        view_file = f'views/{index:06d}.npy'
        _write_native_mask(directory / mask_file, prediction, geometry, spec['mask_values'])
        _write_views(directory / view_file, image, prediction, check)
        cases.append({'source_name': path.name, 'source_sha256': source_sha256, 'geometry': geometry, 'source_shape': list(native), 'tile_count': tile_count,
                      'mask_file': mask_file, 'view_file': view_file, 'view_shape': [depth, h, w, 4]})
        report('saved', index, len(files), tile_count, tile_count)
        del image, prediction
    check()
    manifest = directory / REPORT_NAME
    staging = directory / PENDING_REPORT
    _write_json(staging, {'version': 1, 'profile': PROFILE, 'device': str(device), 'mode': settings['mode'],
                           'window': list(window), 'overlap': settings['overlap'] if settings['mode'] == 'sliding' else None,
                           'blend': 'tapered_logits' if settings['mode'] == 'sliding' else None,
                           'source': settings['source'], 'mask_values': spec['mask_values'], 'identity': provenance,
                           'format': spec['format'], 'cases': cases})
    try:
        check()
        if not defer_report:
            staging.replace(manifest)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return {'status': 'succeeded', 'profile': PROFILE, 'device': str(device), 'cases': len(cases), 'mode': settings['mode'],
            'artifacts': {'inference': str(manifest)}, 'warnings': warnings}
