"""Single-target float32 regression and sample-weighted error metrics."""
import math


class RegressionTask:
    id = 'regression'
    version = 1
    paired = False
    output_name = 'prediction'
    prediction_fields = ('sample_id', 'target', 'prediction', 'residual')

    def __init__(self, config):
        self.input_port = config['input_port']

    def validate_contract(self, resolved, shapes, manifest):
        ports = {port['id']: port for port in resolved['inputs']}
        if set(ports) != {self.input_port} or ports[self.input_port]['shape'] != manifest['input_shape'] or ports[self.input_port]['dtype'] != 'float32':
            raise ValueError(f"Regression requires one float32 input {self.input_port} with shape {manifest['input_shape']}")
        sources = {output['id']: output['source'] for output in resolved['outputs']}
        prediction = sources.get(self.output_name)
        if shapes.get(prediction) != [None, 1]:
            raise ValueError('Regression requires named prediction with shape [batch, 1]')
        if resolved.get('schema_version') == 2:
            spec = resolved['_tensor_specs'][prediction]
            if spec['dtype'] != 'float32' or spec['axes'] not in (['batch', 'feature'], ['batch', 'target']):
                raise ValueError('tensor-v1 prediction requires float32 axes [batch, feature] or [batch, target]')
        return sources

    def bind_batch(self, raw_batch, device):
        import torch
        features, targets, identifiers = raw_batch
        if features.dtype != torch.float32 or targets.dtype != torch.float32 or targets.shape != (len(features), 1):
            raise ValueError('Regression targets must be float32 [batch, 1] and align with features')
        return {'features': {self.input_port: features.to(device)}, 'targets': targets.to(device), 'sample_ids': list(identifiers)}

    def validate_outputs(self, outputs, batch, manifest):
        prediction = outputs[self.output_name]
        if prediction.shape != batch['targets'].shape or prediction.dtype != batch['targets'].dtype:
            raise ValueError('Runtime prediction must match float32 target shape [batch, 1]')

    def loss(self, outputs, batch):
        from torch.nn.functional import mse_loss
        return mse_loss(outputs[self.output_name], batch['targets'])

    def summarize(self, outputs, batch, loss):
        import torch
        targets = batch['targets'].double()
        residuals = outputs[self.output_name].detach().double() - targets
        variance, mean = torch.var_mean(targets, correction=0)
        return {'count': len(targets), 'abs_sum': float(residuals.abs().sum()),
                'squared_sum': float(residuals.square().sum()), 'target_mean': float(mean),
                'target_m2': float(variance) * len(targets)}

    def merge(self, totals, batch):
        # Parallel variance merge avoids subtracting two large squared sums.
        if not totals:
            return batch
        count = totals['count'] + batch['count']
        delta = batch['target_mean'] - totals['target_mean']
        return {'count': count, 'abs_sum': totals['abs_sum'] + batch['abs_sum'],
                'squared_sum': totals['squared_sum'] + batch['squared_sum'],
                'target_mean': totals['target_mean'] + delta * batch['count'] / count,
                'target_m2': totals['target_m2'] + batch['target_m2'] + delta ** 2 * totals['count'] * batch['count'] / count}

    def finalize(self, totals):
        count = totals['count']
        mse = totals['squared_sum'] / count
        return {'loss': mse, 'mae': totals['abs_sum'] / count, 'rmse': math.sqrt(mse), 'count': count,
                'r2': 1 - totals['squared_sum'] / totals['target_m2'] if totals['target_m2'] > 0 else None}

    def progress_metrics(self, metrics):
        return {name: metrics[name] for name in ('mae', 'rmse', 'r2')}

    def prediction_rows(self, outputs, batch, manifest):
        predictions = outputs[self.output_name].flatten().cpu().tolist()
        targets = batch['targets'].flatten().cpu().tolist()
        return [{'sample_id': identifier, 'target': target, 'prediction': prediction, 'residual': prediction - target}
                for identifier, target, prediction in zip(batch['sample_ids'], targets, predictions)]

    def dataset_details(self, manifest):
        return {'target_column': manifest['target_column']}
