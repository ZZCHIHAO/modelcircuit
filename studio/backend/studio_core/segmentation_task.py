"""Semantic segmentation on the model grid, with independent test-mask export."""


def spatial_axes(shape):
    if len(shape) == 4:
        return ['height', 'width']
    if len(shape) == 5:
        return ['depth', 'height', 'width']
    raise ValueError('Segmentation requires a two- or three-dimensional image tensor')


class SegmentationTask:
    id = 'segmentation'
    version = 1
    paired = False  # The annotation is a target, not a second model input.
    output_name = 'mask_logits'
    prediction_fields = ('sample_id', 'mask_file', 'preview_file', 'slice_index', 'slice_count')

    def __init__(self, config):
        self.input_port = config['input_port']

    def validate_contract(self, resolved, shapes, manifest):
        source_shape = manifest['input_shape']
        axes = spatial_axes(source_shape)
        inputs = resolved['inputs']
        if resolved.get('schema_version') != 2 or len(inputs) != 1 or inputs[0]['id'] != self.input_port:
            raise ValueError('Segmentation requires one named tensor-v1 image input matching the input port')
        contract = resolved['_tensor_specs']
        source = contract['input:' + self.input_port]
        if source['shape'] != source_shape or source['axes'] != ['batch', 'channel', *axes] or source['dtype'] != 'float32':
            raise ValueError(f'Segmentation input requires float32 {source_shape} with channel and spatial axes')
        outputs = {port['id']: port['source'] for port in resolved['outputs']}
        logits = contract.get(outputs.get(self.output_name), {})
        expected = [None, len(manifest['mask_values']), *source_shape[2:]]
        if logits.get('shape') != expected or logits.get('axes') != ['batch', 'class', *axes] or logits.get('dtype') != 'float32':
            raise ValueError(f'Segmentation requires float32 mask_logits {expected} with a class axis and matching spatial grid')
        return outputs

    def bind_batch(self, raw_batch, device):
        import torch

        features, targets, identifiers = raw_batch
        if features.dtype != torch.float32 or targets.dtype != torch.int64 or (len(features), *features.shape[2:]) != tuple(targets.shape):
            raise ValueError('Segmentation batch requires float32 image and an int64 mask on the same grid')
        return {'features': {self.input_port: features.to(device)}, 'targets': targets.to(device), 'sample_ids': list(identifiers)}

    def validate_outputs(self, outputs, batch, manifest):
        import torch

        targets = batch['targets']
        logits = outputs[self.output_name]
        if tuple(logits.shape) != (len(targets), len(manifest['mask_values']), *targets.shape[1:]) or logits.dtype != torch.float32:
            raise ValueError('Runtime mask_logits does not match the target grid, batch or class count')

    def loss(self, outputs, batch):
        from torch.nn.functional import cross_entropy
        return cross_entropy(outputs[self.output_name], batch['targets'])

    def summarize(self, outputs, batch, loss):
        import torch

        targets = batch['targets']
        logits = outputs[self.output_name].detach()
        classes = logits.shape[1]
        predictions = logits.argmax(dim=1)
        intersection = targets[predictions == targets]
        return {'count': len(targets), 'voxels': targets.numel(), 'loss_sum': float(loss.detach()) * targets.numel(),
                'target': torch.bincount(targets.ravel(), minlength=classes).tolist(),
                'prediction': torch.bincount(predictions.ravel(), minlength=classes).tolist(),
                'intersection': torch.bincount(intersection, minlength=classes).tolist()}

    def merge(self, totals, batch):
        if not totals:
            return batch
        merged = {key: totals[key] + batch[key] for key in ('count', 'voxels', 'loss_sum')}
        for key in ('target', 'prediction', 'intersection'):
            merged[key] = [left + right for left, right in zip(totals[key], batch[key])]
        return merged

    def finalize(self, totals):
        per_class = []
        for index, (target, predicted, intersection) in enumerate(zip(totals['target'], totals['prediction'], totals['intersection'])):
            union = target + predicted - intersection
            per_class.append({'index': index, 'target_voxels': target, 'predicted_voxels': predicted, 'intersection': intersection,
                              'dice': 2 * intersection / (target + predicted) if union else None,
                              'iou': intersection / union if union else None})
        foreground = [row for row in per_class[1:] if row['dice'] is not None]
        return {'count': totals['count'], 'voxels': totals['voxels'], 'loss': totals['loss_sum'] / totals['voxels'],
                'foreground_dice': sum(row['dice'] for row in foreground) / len(foreground) if foreground else None,
                'foreground_iou': sum(row['iou'] for row in foreground) / len(foreground) if foreground else None,
                'per_class': per_class}

    def progress_metrics(self, metrics):
        return {key: metrics[key] for key in ('foreground_dice', 'foreground_iou') if metrics[key] is not None}

    def dataset_details(self, manifest):
        return {key: manifest[key] for key in ('mask_values', 'input_shape', 'target_shape')}
