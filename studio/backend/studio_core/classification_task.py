"""Data and output contracts for the existing supervised classification task."""


class ClassificationTask:
    id = 'classification'
    version = 1

    def __init__(self, config):
        self.mapping = config['input_ports'] or {'features': config['input_port']}
        self.paired = config['data']['kind'] == 'paired_csv'

    def validate_contract(self, resolved, shapes, manifest):
        ports = {port['id']: port for port in resolved['inputs']}
        expected = manifest['input_shapes'] if self.paired else {'features': manifest['input_shape']}
        if not self.paired and len(ports) != 1:
            raise ValueError('Classification requires exactly one root input')
        if set(ports) != set(self.mapping.values()):
            raise ValueError('Classification input names must match the dataset mapping')
        for modality, name in self.mapping.items():
            if ports[name]['shape'] != expected[modality] or ports[name]['dtype'] != 'float32':
                raise ValueError(f'Classification {modality} input shape/dtype must match {expected[modality]}')
        sources = {output['id']: output['source'] for output in resolved['outputs']}
        logits = sources.get('logits')
        if shapes.get(logits) != [None, len(manifest['class_names'])]:
            raise ValueError('Classification requires named logits with shape [batch, class_count]')
        if resolved.get('schema_version') == 2:
            spec = resolved['_tensor_specs'][logits]
            if spec['dtype'] != 'float32' or spec['axes'] != ['batch', 'class']:
                raise ValueError('tensor-v1 logits require float32 axes [batch, class]')
        return sources

    def bind_batch(self, raw_batch, device):
        import torch
        features, labels, identifiers = raw_batch
        if self.paired:
            if not isinstance(features, dict) or set(features) != set(self.mapping):
                raise ValueError('Paired batch must contain exactly image and tabular modalities')
        else:
            features = {'features': features}
        inputs = {}
        for key, name in self.mapping.items():
            value = features[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim < 1 or len(value) != len(labels):
                raise ValueError('Input batch size/dtype does not match labels for ' + name)
            inputs[name] = value.to(device)
        return {'features': inputs, 'targets': labels.to(device), 'sample_ids': list(identifiers)}

    def loss(self, outputs, batch):
        from torch.nn.functional import cross_entropy
        return cross_entropy(outputs['logits'], batch['targets'])

    def summarize(self, outputs, batch, loss):
        targets = batch['targets']
        return {'loss_sum': float(loss.detach()) * len(targets),
                'correct': int((outputs['logits'].detach().argmax(1) == targets).sum()), 'count': len(targets)}

    def finalize(self, totals):
        count = totals['count']
        return {'loss': totals['loss_sum'] / count, 'accuracy': totals['correct'] / count, 'count': count}
