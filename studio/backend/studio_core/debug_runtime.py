"""One immutable forward, optimizer step, or one-parameter comparison."""
from copy import deepcopy
import hashlib
from itertools import islice
import math
from time import perf_counter

import torch

from .debug_spec import numeric_parameters, validate_debug_config
from .devices import resolve_device
from .execution_policy import check_config_batch, hardware_memory_errors
from .experiment_spec import model_digest
from .memory_admission import admit_memory
from .runtime import build_model
from .schema import prepare_graph
from .ports import parse_source

SAMPLE_ELEMENTS = 128
MAX_PARAMETER_ROWS = 64
MAX_OUTPUTS = 12


def _brief(tensor, offset=0):
    """Read a bounded interval in logical order, including noncontiguous views."""
    value = tensor.detach()
    count = max(0, min(value.numel() - offset, SAMPLE_ELEMENTS))
    if count == 0:
        items = []
    else:
        end = offset + count
        if value.is_contiguous():
            sample = value.view(-1)[offset:end]
        else:
            indices = torch.arange(offset, end, device=value.device)
            sample = value[torch.unravel_index(indices, value.shape)]
        items = [item if math.isfinite(item) else None for item in sample.cpu().tolist()]
    finite = [item for item in items if item is not None]
    return {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.'),
            'count': value.numel(), 'sample_size': count, 'offset': offset, 'truncated': count < value.numel(),
            'min': min(finite, default=None), 'max': max(finite, default=None),
            'mean': sum(finite) / len(finite) if finite else None,
            'l2': math.sqrt(sum(item * item for item in finite)), 'values': items}


def _source_instance(graph, ref):
    paths = ref.split('/')
    instance_path = []
    for index, part in enumerate(paths):
        kind, identifier, port = parse_source(part)
        last = index == len(paths) - 1
        if kind == 'input' and last and any(item['id'] == identifier for item in graph['inputs']):
            return None
        call = next((item for item in graph['calls'] if item['id'] == identifier), None) if kind == 'call' else None
        if call is None:
            break
        instance = next(item for item in graph['instances'] if item['id'] == call['instance_id'])
        instance_path.append(instance['id'])
        if last:
            ports = ['out'] if instance['kind'] != 'chip' else [value['id'] for value in instance['_graph']['outputs']]
            if port in ports or port is None and len(ports) == 1:
                return '/'.join(instance_path)
            break
        if instance['kind'] != 'chip':
            break
        graph = instance['_graph']
    raise ValueError('探针端口在当前图中已失效：' + ref)


class _ProbeObserver:
    def __init__(self, sources, graph, gradients, offset):
        self.gradients = gradients
        self.offset = offset
        self.records = {ref: {'instance': _source_instance(graph, ref), 'activation': None,
                              'gradient': None, 'axes': None} for ref in sources}

    def observe(self, source, value, spec):
        info = self.records.get(source)
        if info is None:
            return
        info['activation'] = _brief(value, self.offset)
        info['axes'] = None if spec is None else spec['axes']
        if self.gradients and value.requires_grad:
            value.register_hook(lambda gradient: self._gradient(info, gradient))

    def _gradient(self, info, gradient):
        info['gradient'] = _brief(gradient, self.offset)


def _synthetic_batch(resolved, config, device):
    generator = torch.Generator(device='cpu').manual_seed(config['seed'] + 1)
    features = {}
    for port in resolved['inputs']:
        shape = [config['batch_size'] if dim is None else dim for dim in port['shape']]
        value = (torch.ones(shape, dtype=torch.bool) if port['dtype'] == 'bool' else
                 torch.randn(shape, generator=generator, dtype=torch.float32))
        features[port['id']] = value.to(device)
    return {'features': features, 'sample_ids': [f'synthetic:{i}' for i in range(config['batch_size'])]}, None, None


def _experiment_batch(resolved, shapes, config, device, cancelled):
    from torch.utils.data import DataLoader, Subset
    from .classification_data import build_datasets, inspect_dataset
    from .task_adapters import get_task_adapter
    experiment = config['experiment']
    manifest = inspect_dataset(experiment['data'], experiment['seed'], cancelled)
    if experiment['expected_dataset_digest'] not in (None, manifest['dataset_digest']):
        raise ValueError('数据在检查后已改变，请重新检查再选择样本')
    task_id = 'regression' if experiment['profile'] == 'regression-v1' else 'classification'
    task = get_task_adapter(task_id, 1)(experiment)
    task.validate_contract(resolved, shapes, manifest)
    split = config['split']
    dataset = build_datasets(manifest, cancelled)[split]
    if config['sample_ids']:
        by_id = {record['id']: index for index, record in enumerate(dataset.records)}
        missing = next((item for item in config['sample_ids'] if item not in by_id), None)
        if missing is not None:
            raise ValueError(f'样本 {missing} 不在 {split} 划分内，请重新选择')
        dataset = Subset(dataset, [by_id[item] for item in config['sample_ids']])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    batch = task.bind_batch(next(iter(loader)), device)
    return batch, task, manifest


def _batch_loss(outputs, batch, task, manifest):
    if task is not None:
        task.validate_outputs(outputs, batch, manifest)
        return task.loss(outputs, batch)
    logits = outputs.get('logits')
    if logits is None or logits.ndim != 2 or logits.shape[1] < 2 or not logits.is_floating_point():
        raise ValueError('合成分类单步需要形状为 [N, 类别数]（至少两类）的 logits 输出')
    targets = torch.arange(len(logits), device=logits.device) % logits.shape[1]
    return torch.nn.functional.cross_entropy(logits, targets)


def _output_samples(outputs):
    return {key: _brief(value) for key, value in islice(outputs.items(), MAX_OUTPUTS)}


def _parameter_rows(parameters, before):
    rows = []
    for name, parameter in parameters:
        previous = before[name]
        after = _brief(parameter)
        delta = [right - left for left, right in zip(previous['values'], after['values'])
                 if left is not None and right is not None]
        rows.append({'parameter': name, 'trainable': parameter.requires_grad, 'before': previous, 'after': after,
                     'gradient': None if parameter.grad is None else _brief(parameter.grad),
                     'update_norm': math.sqrt(sum(value * value for value in delta))})
    return rows


def _build(graph, config, device, *, training=False):
    torch.random.default_generator.manual_seed(config['seed'])
    memory = None
    if graph.get('resource_profile') == 'hardware-v5':
        resolved, _, _ = prepare_graph(graph)
        memory = admit_memory(resolved, device, batch_size=config['batch_size'], training=training,
                              optimizer=config['optimizer'], capture=False)
    return build_model(graph, device, capture=False, memory_report=memory)


def _sweep(graph, config, device, batch, task, manifest, check):
    sweep = config['sweep']
    candidate = deepcopy(graph)
    instance = next((item for item in candidate['instances'] if item['id'] == sweep['instance']), None)
    if instance is None or sweep['parameter'] not in numeric_parameters(candidate, instance):
        raise ValueError('请选择主模型上的一个数值参数')
    base = _build(graph, config, device)
    parameter_shapes = {key: tuple(value.shape) for key, value in base.state_dict().items()}
    del base
    comparison = []
    for value in sweep['values']:
        check()
        instance.setdefault('params', {})[sweep['parameter']] = value
        model = _build(candidate, config, device).eval()
        if parameter_shapes != {key: tuple(value.shape) for key, value in model.state_dict().items()}:
            raise ValueError('扫描值改变了模型权重形状，请选择保持权重结构的参数')
        with torch.inference_mode():
            outputs = model(**batch['features'])
            loss = None if task is None else float(_batch_loss(outputs, batch, task, manifest))
            comparison.append({'value': value, 'loss': loss, 'outputs': _output_samples(outputs),
                               'outputs_omitted': max(0, len(outputs) - MAX_OUTPUTS)})
        del model, outputs
    return comparison


@hardware_memory_errors
def run_debug(graph, config, progress=None, cancelled=None):
    """Use one fixed sample batch in a worker and leave the design and checkpoints intact."""
    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('可视化调试已停止')

    check()
    config = validate_debug_config(config)
    check_config_batch(graph, config['batch_size'])
    resolved, shapes, _ = prepare_graph(graph)
    device = resolve_device(config['device'])
    cuda_devices = [] if device.type == 'cpu' else [device.index]
    if graph.get('resource_profile') == 'hardware-v5':
        admit_memory(resolved, device, batch_size=config['batch_size'], training=config['mode'] == 'step',
                     optimizer=config['optimizer'], capture=False)
    started = perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        if cuda_devices:
            torch.cuda.default_generators[device.index].manual_seed(config['seed'])
        if config['experiment'] is None:
            batch, task, manifest = _synthetic_batch(resolved, config, device)
        else:
            batch, task, manifest = _experiment_batch(resolved, shapes, config, device, cancelled)
        check()
        mode = config['mode']
        result = {'mode': mode, 'device': str(device), 'model_digest': model_digest(graph),
                  'batch_kind': 'synthetic' if manifest is None else manifest['kind'], 'seed': config['seed'],
                  'split': config['split'] if manifest is not None else None, 'sample_ids': batch['sample_ids'],
                  'input_shapes': {key: list(value.shape) for key, value in batch['features'].items()},
                  'input_dtypes': {key: str(value.dtype) for key, value in batch['features'].items()},
                  'task_profile': config['experiment']['profile'] if task is not None else None,
                  'optimizer_settings': {key: config[key] for key in ('optimizer', 'lr')} if mode == 'step' else None,
                  'probe_offset': config['probe_offset'], 'sample_count': len(batch['sample_ids']),
                  'optimizer_steps': 0, 'loss': None,
                  'dataset_digest': None if manifest is None else manifest['dataset_digest'], 'outputs': {},
                  'batch_digest': hashlib.sha256('\n'.join(batch['sample_ids']).encode('utf-8')).hexdigest(),
                  'probes': {}, 'parameters': [], 'sweep': []}
        if mode == 'sweep':
            result['sweep'] = _sweep(graph, config, device, batch, task, manifest, check)
        else:
            step = mode == 'step'
            probes = _ProbeObserver(config['sources'], resolved, step, config['probe_offset'])
            model = _build(graph, config, device, training=step).train(step)
            if step:
                learnable = [value for value in model.parameters() if value.requires_grad]
                if not learnable:
                    raise ValueError('训练一步需要至少一个可训练参数')
                optimizer_cls = torch.optim.SGD if config['optimizer'] == 'sgd' else torch.optim.AdamW
                optimizer = optimizer_cls(learnable, lr=config['lr'])
                named = list(model.named_parameters())
                prefixes = tuple('components.' + record['instance'].replace('/', '.components.') + '.'
                                 for record in probes.records.values() if record['instance'] is not None)
                measured = sorted(named, key=lambda pair: not pair[0].startswith(prefixes))[:MAX_PARAMETER_ROWS]
                before = {name: _brief(value) for name, value in measured}
                outputs = model.forward_observed(batch['features'], probes)
                loss = _batch_loss(outputs, batch, task, manifest)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('损失为非有限值')
                loss.backward()
                check()
                if any(value.grad is not None and not bool(torch.isfinite(value.grad).all()) for value in learnable):
                    raise ValueError('梯度含非有限值')
                optimizer.step()
                result.update(loss=float(loss.detach()), optimizer_steps=1, parameters=_parameter_rows(measured, before),
                              parameters_omitted=len(named) - len(measured))
            else:
                with torch.inference_mode():
                    outputs = model.forward_observed(batch['features'], probes)
                    if task is not None:
                        result['loss'] = float(_classification_loss(outputs, batch, task, manifest))
            result.update(outputs=_output_samples(outputs), outputs_omitted=max(0, len(outputs) - MAX_OUTPUTS),
                          probes=probes.records)
        check()
        result['elapsed_ms'] = (perf_counter() - started) * 1000
        return result
