"""Data and output contracts for the existing supervised classification task."""


import json


class ClassificationTask:
    id = 'classification'
    version = 1
    output_name = 'logits'
    prediction_fields = ('sample_id', 'label', 'label_name', 'prediction', 'prediction_name', 'probabilities')

    def __init__(self, config):
        kind = config['data']['kind']
        self.multimodal = kind == 'multimodal_csv'
        self.paired = self.multimodal or kind == 'paired_csv'
        self.mapping = config.get('input_ports') or {'features': config['input_port']}

    def validate_contract(self, resolved, shapes, manifest):
        ports = {port['id']: port for port in resolved['inputs']}
        expected = manifest['input_shapes'] if self.paired else {'features': manifest['input_shape']}
        if not self.paired and len(ports) != 1:
            raise ValueError('Classification requires exactly one root input')
        if set(ports) != set(self.mapping.values()):
            raise ValueError('Classification input names must match the dataset mapping')
        if self.multimodal and resolved.get('schema_version') != 2:
            raise ValueError('Multimodal classification requires tensor-v1 input axes')
        for modality, name in self.mapping.items():
            if modality not in expected or ports[name]['shape'] != expected[modality] or ports[name]['dtype'] != 'float32':
                raise ValueError(f'Classification {modality} input shape/dtype must match {expected.get(modality)}')
            if self.multimodal:
                spec = resolved['_tensor_specs']['input:' + name]
                axes = manifest['input_axes'][modality]
                if spec['axes'] != axes or spec['batch_axis'] != 0:
                    raise ValueError(f'Classification {modality} input axes must match {axes}')
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
            if not isinstance(features, dict) or not self.mapping.keys() <= features.keys():
                raise ValueError('Batch is missing a named modality mapped to the graph')
            if not self.multimodal and features.keys() != self.mapping.keys():
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

    def validate_outputs(self, outputs, batch, manifest):
        if outputs['logits'].shape != (len(batch['targets']), len(manifest['class_names'])):
            raise ValueError('Runtime logits shape does not match batch/class count')

    def merge(self, totals, batch):
        return {key: totals.get(key, 0) + value for key, value in batch.items()}

    def progress_metrics(self, metrics):
        return {'accuracy': metrics['accuracy']}

    def prediction_rows(self, outputs, batch, manifest):
        logits = outputs['logits']
        names = manifest['class_names']
        return [dict(sample_id=identifier, label=label, label_name=names[label], prediction=prediction,
                     prediction_name=names[prediction], probabilities=json.dumps(probability, separators=(',', ':'), allow_nan=False))
                for identifier, label, prediction, probability in zip(batch['sample_ids'], batch['targets'].cpu().tolist(),
                                                                     logits.argmax(1).cpu().tolist(), logits.softmax(1).cpu().tolist())]

    def dataset_details(self, manifest):
        return {'class_names': manifest['class_names']}
