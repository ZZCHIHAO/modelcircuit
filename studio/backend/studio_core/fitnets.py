"""FitNets hint training, student distillation, and weights-only artifacts."""
from copy import deepcopy
import csv
import hashlib
import io
import math
import platform
from pathlib import Path
import time

from .classification_data import build_datasets, inspect_dataset
from .classification_runtime import _atomic_write, _cpu_tree, _probe, _write_json
from .classification_task import ClassificationTask
from .execution_policy import check_config_batch, hardware_memory_errors
from .experiment_spec import model_digest, validate_supervised_config
from .ports import call_source_refs, parse_source
from .schema import prepare_graph

_ROLES = ('teacher', 'student', 'projector')
_ROLE_OUTPUTS = {'teacher': {'hint', 'logits'}, 'student': {'guided', 'logits'}}


def _checked(condition, message):
    if not condition:
        raise ValueError(message)


def validate_fitnets_graph(graph, config=None, manifest=None):
    """Validate the paired branches and the feature projection without Torch."""
    resolved, shapes, count = prepare_graph(graph)
    instances = {instance['id']: instance for instance in resolved['instances']}
    _checked(set(instances) == set(_ROLES) and all(i['kind'] == 'chip' for i in instances.values()),
             'FitNets requires the root chips teacher, student and projector')
    _checked(not instances['student'].get('frozen') and not instances['projector'].get('frozen'),
             'FitNets student and projector chips must be trainable')
    calls = {call['instance_id']: call for call in resolved['calls']}
    _checked(len(resolved['calls']) == 3 and set(calls) == set(_ROLES),
             'FitNets requires one call for each of the three root chips')
    _checked(len(resolved['inputs']) == 1, 'FitNets requires exactly one shared input')
    input_ref = 'input:' + resolved['inputs'][0]['id']
    branch_inputs = {}
    for role in _ROLES:
        ports = instances[role]['_graph']['inputs']
        _checked(len(ports) == 1, f'FitNets {role} chip requires exactly one input')
        branch_inputs[role] = ports[0]['id']
    for role in ('teacher', 'student'):
        expected = _ROLE_OUTPUTS[role]
        _checked({p['id'] for p in instances[role]['_graph']['outputs']} == expected,
                 f'FitNets {role} chip requires ports {", ".join(sorted(expected))}')
        _checked(calls[role]['inputs'] == {branch_inputs[role]: input_ref},
                 f'FitNets {role} must connect to the shared input')
    teacher = 'call:' + calls['teacher']['id']
    student = 'call:' + calls['student']['id']
    projector_ports = [port['id'] for port in instances['projector']['_graph']['outputs']]
    _checked(len(projector_ports) == 1, 'FitNets projector requires exactly one output')
    _checked(calls['projector']['inputs'] == {branch_inputs['projector']: student + ':guided'},
             'FitNets projector input must connect to student:guided')
    sources = {output['id']: output['source'] for output in resolved['outputs']}
    expected = {'logits': student + ':logits', 'teacher_logits': teacher + ':logits',
                'teacher_hint': teacher + ':hint', 'student_guided': student + ':guided'}
    for key, ref in expected.items():
        _checked(sources.get(key) == ref, f'FitNets root output {key} must connect to {ref}')
    _checked(sources.get('student_projection') in call_source_refs(calls['projector']['id'], projector_ports),
             'FitNets root output student_projection must connect to the projector')
    _checked(sources.keys() == (expected.keys() | {'student_projection'}),
             'FitNets root outputs must be logits, teacher_logits, teacher_hint, student_guided and student_projection')
    _checked(shapes[sources['teacher_hint']] == shapes[sources['student_projection']],
             'Teacher hint and student projection must have matching shapes')
    _checked(shapes[sources['teacher_logits']] == shapes[sources['logits']],
             'Teacher and student logits must have matching shapes and class count')
    _checked(len(shapes[sources['logits']]) == 2 and shapes[sources['logits']][0] is None,
             'FitNets logits require [batch, class_count]')
    if config is not None and manifest is not None:
        ClassificationTask(config).validate_contract(resolved, shapes, manifest)
    return resolved, shapes, count


def _standalone(branch, root, original_output, output_name):
    """Convert a resolved chip into a clean inference graph with the same keys."""
    definitions = {item['id']: item for item in root.get('definitions', [])}
    ext = {}

    def collect(scope):
        for instance in scope['instances']:
            if instance['kind'] == 'chip':
                identifier = instance['definition_id']
                if identifier not in ext:
                    ext[identifier] = deepcopy(definitions[identifier])
                    collect(ext[identifier]['graph'])

    collect(branch)
    graph = {key: deepcopy(branch[key]) for key in ('inputs', 'instances', 'calls', 'outputs')}
    # Resolved root call arguments may bind public chip parameters. Only the
    # resolved primitive values belong in the standalone graph.
    for instance in graph['instances']:
        instance.pop('_graph', None)
        instance.pop('_output_mode', None)
    graph['outputs'] = [{'id': output_name, 'source': next(
        output['source'] for output in branch['outputs'] if output['id'] == original_output)}]
    graph.update({key: root[key] for key in ('schema_version', 'capability_profile', 'resource_profile',
                                           'resource_batch_bound') if key in root})
    graph.update(id='fitnets_student', revision=1, definitions=list(ext.values()))
    return graph


def fitnets_teacher_graph(graph):
    """Return the teacher as a standalone supervised classifier."""
    resolved, _, _ = validate_fitnets_graph(graph)
    teacher = next(item for item in resolved['instances'] if item['id'] == 'teacher')
    result = _standalone(teacher['_graph'], graph, 'logits', 'logits')
    result.update(id='fitnets_teacher', name='FitNets Teacher')
    prepare_graph(result)
    return result


def _selected_calls(graph, output_ids):
    calls = {item['id']: item for item in graph['_ordered_calls']}
    needed = set()
    pending = [output['source'] for output in graph['outputs'] if output['id'] in output_ids]
    while pending:
        kind, identifier, _ = parse_source(pending.pop())
        if kind == 'call' and identifier not in needed:
            needed.add(identifier)
            pending.extend(calls[identifier]['inputs'].values())
    return [item for item in graph['_ordered_calls'] if item['id'] in needed]


def _with_scope_outputs(module, inputs, *output_ids):
    """Run only ancestors of selected scope outputs using the original parameters."""
    original = module.graph
    narrowed = dict(original, _ordered_calls=_selected_calls(original, output_ids),
                    outputs=[port for port in original['outputs'] if port['id'] in output_ids])
    module.graph = narrowed
    try:
        return module(**inputs)
    finally:
        module.graph = original


def _teacher_checkpoint(module, path, classes, check):
    import torch

    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as file:
        stat = file.seek(0, io.SEEK_END)
        _checked(stat > 0, 'Teacher checkpoint is empty')
        file.seek(0)
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            check()
            sha.update(chunk)
        file.seek(0)
        checkpoint = torch.load(file, weights_only=True, map_location='cpu')
    _checked(isinstance(checkpoint, dict), 'Teacher checkpoint must contain a state dictionary')
    _checked(checkpoint.get('class_names') == classes,
             'Teacher checkpoint class order must match the dataset class order')
    state = checkpoint.get('model_state')
    template = module.state_dict()
    _checked(isinstance(state, dict) and state.keys() == template.keys(),
             'Teacher checkpoint state keys do not match the teacher chip')
    for name, tensor in state.items():
        check()
        expected = template[name]
        _checked(isinstance(tensor, torch.Tensor) and tensor.layout == expected.layout and
                 tensor.shape == expected.shape and tensor.dtype == expected.dtype,
                 f'Teacher checkpoint tensor is incompatible: {name}')
        _checked(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
                 f'Teacher checkpoint tensor contains non-finite values: {name}')
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False).eval()
    return sha.hexdigest()


def _changed_instances(previous, current):
    import torch

    return sorted({name.split('.')[1] for name, tensor in current.items()
                   if not torch.equal(previous[name], tensor)})


def _finite(tensor, label):
    import torch

    _checked(bool(torch.isfinite(tensor).all()), f'Non-finite FitNets {label}')
    return tensor


@hardware_memory_errors
def run_fitnets(graph, config, artifact_directory, progress=None, cancelled=None):
    """Fit hints, then minimize supervised and temperature-scaled student losses."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from .devices import resolve_device
    from .runtime import build_model

    config = validate_supervised_config(config)
    _checked(config['profile'] == 'fitnets-v1', 'FitNets requires the fitnets-v1 task profile')
    _checked(not config.get('attention_probe_requests'), 'FitNets uses the hint probes for this task')
    check_config_batch(graph, config['batch_size'])
    settings = config['fitnets']
    started = time.monotonic()
    total_steps = 0
    step = 0

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('FitNets experiment cancelled')
        if time.monotonic() - started > config['max_runtime_seconds']:
            raise TimeoutError('FitNets experiment exceeded max_runtime_seconds')
        return False

    def emit(phase, stage, epoch=0, **values):
        if progress is not None:
            progress({'phase': phase, 'stage': stage, 'epoch': epoch,
                      'epochs': settings['hint_epochs'] + config['epochs'],
                      'step': step, 'total': total_steps, **values})

    check()
    emit('inspect', 'prepare')
    manifest = inspect_dataset(config['data'], seed=config['seed'], cancelled=check)
    _checked(config['expected_dataset_digest'] in (None, manifest['dataset_digest']),
             'Dataset digest changed since inspection')
    resolved, _, parameter_count = validate_fitnets_graph(graph, config, manifest)
    device = resolve_device(config['device'])
    datasets = build_datasets(manifest, cancelled=check)
    total_steps = math.ceil(len(datasets['train']) / config['batch_size']) * (settings['hint_epochs'] + config['epochs'])
    task = ClassificationTask(config)
    memory_report = None
    if graph.get('resource_profile') == 'hardware-v5':
        from .memory_admission import admit_memory
        batch = min(config['batch_size'], max(manifest['counts'].values()))
        memory_report = admit_memory(resolved, device, batch_size=batch, training=True,
                                     capture=False, optimizer=config['optimizer'], workflow='classification')
    paths = {key: Path(artifact_directory).resolve() / name for key, name in {
        'manifest': 'manifest.json', 'run_spec': 'run_spec.json', 'metrics': 'metrics.json',
        'hint_checkpoint': 'hint.pt', 'student_checkpoint': 'student-best.pt',
        'last_checkpoint': 'student-last.pt', 'predictions': 'predictions.csv'}.items()}
    paths['manifest'].parent.mkdir(parents=True, exist_ok=True)
    cuda_devices = [device.index] if device.type == 'cuda' else []
    optimizer_type = torch.optim.AdamW if config['optimizer'] == 'adamw' else torch.optim.SGD

    def batches(split, shuffle=False):
        # A dedicated generator avoids dependence on the global RNG used by Dropout.
        generator = train_rng if shuffle else torch.Generator().manual_seed(config['seed'] + 1)
        return DataLoader(datasets[split], batch_size=config['batch_size'], shuffle=shuffle,
                          generator=generator, num_workers=0)

    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(config['seed'])
        if cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config['seed'])
        train_rng = torch.Generator(device='cpu').manual_seed(config['seed'])
        model = build_model(graph, device=device, capture=False, memory_report=memory_report)
        teacher, student, projector = (model.components[role] for role in _ROLES)
        branch_ports = {role: model.components[role].graph['inputs'][0]['id'] for role in _ROLES}
        projector_output = projector.graph['outputs'][0]['id']
        teacher_sha = _teacher_checkpoint(teacher, settings['teacher_checkpoint'], manifest['class_names'], check)
        student_graph = _standalone(student.graph, graph, 'logits', 'logits')
        prepare_graph(student_graph)
        old_teacher = _cpu_tree(teacher.state_dict())
        initial_student = _cpu_tree(student.state_dict())
        initial_projector = _cpu_tree(projector.state_dict())
        prefix = {call['instance_id'] for call in _selected_calls(student.graph, ('guided',))}
        hint_parameters = [param for key in prefix for param in student.components[key].parameters() if param.requires_grad]
        projection_parameters = [param for param in projector.parameters() if param.requires_grad]
        student_parameters = [param for param in student.parameters() if param.requires_grad]
        _checked(bool(hint_parameters) and bool(projection_parameters),
                 'FitNets requires a trainable student prefix and projector')
        optimizer = optimizer_type(hint_parameters + projection_parameters, lr=config['lr'])
        _write_json(paths['manifest'], manifest)
        environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': str(device),
                       'cuda': torch.version.cuda if cuda_devices else None}
        _write_json(paths['run_spec'], {'graph': graph, 'config': config, 'environment': environment,
                                          'graph_digest': model_digest(graph), 'dataset_digest': manifest['dataset_digest'],
                                          'teacher_checkpoint_sha256': teacher_sha})

        def fetch(raw_batch):
            batch = task.bind_batch(raw_batch, device)
            return batch, batch['features'][config['input_port']]

        def student_logits(x):
            return _finite(_with_scope_outputs(student, {branch_ports['student']: x}, 'logits')['logits'], 'student logits')

        def save(path, item):
            check()
            _atomic_write(path, lambda file: torch.save(item, file))

        def optimize(loss):
            _finite(loss, 'loss')
            _checked(loss.requires_grad, 'FitNets loss must depend on trainable parameters')
            loss.backward()
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        _finite(param.grad, 'gradient')
            optimizer.step()
            for group in optimizer.param_groups:
                for param in group['params']:
                    _finite(param, 'parameter')

        history = []
        for epoch in range(1, settings['hint_epochs'] + 1):
            student.train()
            projector.train()
            loss_sum, seen = 0.0, 0
            loader = batches('train', shuffle=True)
            for batch_index, raw in enumerate(loader, 1):
                check()
                _, x = fetch(raw)
                with torch.no_grad():
                    hint = _finite(_with_scope_outputs(teacher, {branch_ports['teacher']: x}, 'hint')['hint'], 'teacher hint')
                guided = _with_scope_outputs(student, {branch_ports['student']: x}, 'guided')['guided']
                projected = projector(**{branch_ports['projector']: guided})[projector_output]
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(projected, hint)
                optimize(loss)
                loss_sum += float(loss.detach()) * len(x)
                seen += len(x)
                step += 1
                emit('train', 'hint', epoch, batch=batch_index, batches=len(loader), loss=float(loss.detach()))
            train_loss = loss_sum / seen
            history.append({'stage': 'hint', 'epoch': epoch, 'train_loss': train_loss})
            emit('epoch', 'hint', epoch, train_loss=train_loss)
        hint_student = _cpu_tree(student.state_dict())
        hint_projector = _cpu_tree(projector.state_dict())
        save(paths['hint_checkpoint'], {'format': 'fitnets-hint-v1', 'student_state': hint_student,
                                           'projector_state': hint_projector, 'teacher_checkpoint_sha256': teacher_sha,
                                           'hint_epochs': settings['hint_epochs'], 'dataset_digest': manifest['dataset_digest']})
        projector.eval()
        del optimizer
        optimizer = optimizer_type(student_parameters, lr=config['lr'])
        temperature, weight = settings['temperature'], settings['distill_weight']
        best_score, best_epoch = math.inf, None
        best_metrics = None
        best_step = None
        grad_norms = {}

        def evaluate(split, epoch, predictions=None):
            student.eval()
            seen = {}
            with torch.no_grad():
                for raw in batches(split):
                    check()
                    batch, x = fetch(raw)
                    output = {'logits': student_logits(x)}
                    loss = _finite(task.loss(output, batch), 'evaluation loss')
                    seen = task.merge(seen, task.summarize(output, batch, loss))
                    if predictions is not None:
                        predictions.extend(task.prediction_rows(output, batch, manifest))
            emit(split, 'distill', epoch)
            return task.finalize(seen)

        for distill_epoch in range(1, config['epochs'] + 1):
            epoch = settings['hint_epochs'] + distill_epoch
            student.train()
            seen = {}
            hard_sum, soft_sum = 0.0, 0.0
            loader = batches('train', shuffle=True)
            for batch_index, raw in enumerate(loader, 1):
                check()
                batch, x = fetch(raw)
                with torch.no_grad():
                    teaching = _finite(_with_scope_outputs(teacher, {branch_ports['teacher']: x}, 'logits')['logits'], 'teacher logits')
                logits = student_logits(x)
                hard = F.cross_entropy(logits, batch['targets'])
                soft = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                                F.softmax(teaching / temperature, dim=-1), reduction='batchmean') * temperature ** 2
                loss = (1.0 - weight) * hard + weight * soft
                optimizer.zero_grad(set_to_none=True)
                optimize(loss)
                seen = task.merge(seen, task.summarize({'logits': logits}, batch, loss))
                hard_sum += float(hard.detach()) * len(x)
                soft_sum += float(soft.detach()) * len(x)
                step += 1
                emit('train', 'distill', epoch, batch=batch_index, batches=len(loader), loss=float(loss.detach()))
            train_metrics = task.finalize(seen)
            val_metrics = evaluate('val', epoch)
            count = train_metrics['count']
            point = {'stage': 'distill', 'epoch': epoch, 'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss'],
                     'train_accuracy': train_metrics['accuracy'], 'val_accuracy': val_metrics['accuracy'],
                     'hard_loss': hard_sum / count, 'soft_loss': soft_sum / count}
            history.append(point)
            snapshot = {'format': 'fitnets-student-v1', 'graph': student_graph, 'model_state': _cpu_tree(student.state_dict()),
                        'class_names': manifest['class_names'], 'preprocessing': manifest['preprocessing'],
                        'dataset_digest': manifest['dataset_digest'], 'teacher_checkpoint_sha256': teacher_sha,
                        'epoch': epoch, 'distill_epoch': distill_epoch, 'metrics': {'train': train_metrics, 'val': val_metrics}}
            save(paths['last_checkpoint'], snapshot)
            if val_metrics['loss'] < best_score:
                best_score, best_epoch, best_step = val_metrics['loss'], epoch, step
                best_metrics = snapshot['metrics']
                grad_norms = {name: float(p.grad.detach().double().norm()) if p.grad is not None else None
                              for name, p in student.named_parameters()}
                save(paths['student_checkpoint'], snapshot)
            emit('epoch', 'distill', epoch, **{key: value for key, value in point.items() if key not in ('epoch', 'stage')})

        teacher_changed = _changed_instances(old_teacher, teacher.state_dict()) if device.type == 'cpu' else _changed_instances(old_teacher, _cpu_tree(teacher.state_dict()))
        _checked(not teacher_changed, 'FitNets teacher state changed while training')
        final_student = _cpu_tree(student.state_dict())
        phase_summary = {'teacher_unchanged': True,
                         'hint_changed_instances': {'student': _changed_instances(initial_student, hint_student),
                                                    'projector': _changed_instances(initial_projector, hint_projector)},
                         'distill_changed_instances': {'student': _changed_instances(hint_student, final_student),
                                                       'projector': _changed_instances(hint_projector, _cpu_tree(projector.state_dict()))}}
        checkpoint = torch.load(paths['student_checkpoint'], weights_only=True, map_location='cpu')
        student.load_state_dict(checkpoint['model_state'], strict=True)
        predictions = []
        metrics = {**best_metrics, 'test': evaluate('test', best_epoch, predictions)}
        table = io.StringIO(newline='')
        writer = csv.DictWriter(table, fieldnames=task.prediction_fields)
        writer.writeheader()
        writer.writerows(predictions)
        _atomic_write(paths['predictions'], lambda file: file.write(table.getvalue().encode('utf-8')))
        _write_json(paths['metrics'], {'history': history, 'metrics': metrics, 'phase_summary': phase_summary,
                                        'best_epoch': best_epoch, 'best_step': best_step})
        model.eval()
        check()
        val_batch, val_x = fetch(next(iter(batches('val'))))
        with torch.no_grad():
            outputs = model(**val_batch['features'])
        source_map = {output['id']: output['source'] for output in resolved['outputs']}
        output_metadata = {}
        previews = {}
        probes = {}
        for label, value in outputs.items():
            _finite(value, label)
            measured = _probe(value, 64)
            probes[source_map[label]] = measured
            previews[label] = measured['values']
            output_metadata[label] = {'shape': measured['shape'], 'count': measured['count'],
                                      'preview_count': measured['sample_size'], 'truncated': measured['count'] > measured['sample_size'],
                                      'source': 'measured', 'tensor_source': source_map[label]}
        measurements = []
        for role, module in (('student', student), ('teacher', teacher), ('projector', projector)):
            for name, parameter in module.named_parameters():
                before = initial_student.get(name) if role == 'student' else initial_projector.get(name) if role == 'projector' else None
                delta = float((parameter.detach().cpu().double() - before.double()).norm()) if before is not None else 0.0
                measurements.append({'name': f'components.{role}.{name}', 'shape': list(parameter.shape),
                                     'trainable': parameter.requires_grad, 'grad_norm': grad_norms.get(name) if role == 'student' else None,
                                     'update_norm': delta})
        result = {'status': 'succeeded', 'profile': 'fitnets-v1', 'device': str(device), 'seed': config['seed'],
                  'data_source': manifest['kind'], 'history': history, 'metrics': metrics, 'global_step': step,
                  'best_epoch': best_epoch, 'completed_epochs': settings['hint_epochs'] + config['epochs'],
                  'phase_summary': phase_summary, 'parameter_count': parameter_count,
                  'probes': probes, 'parameters': measurements[:512], 'shared_instances': [],
                  'outputs': previews, 'output_metadata': output_metadata,
                  'probe_context': {'split': 'val', 'epoch': best_epoch, 'device': str(device), 'mode': 'eval',
                                    'sample_ids': val_batch['sample_ids'], 'gradient_source': 'student validation-selected distillation epoch',
                                    'parameter_update_source': 'student best checkpoint, final hint projector, frozen teacher'},
                  'dataset': {'dataset_digest': manifest['dataset_digest'], 'class_names': manifest['class_names'],
                              'counts': manifest['counts'], 'manifest_path': str(paths['manifest'])},
                  'artifacts': {key: str(path) for key, path in paths.items()},
                  'warnings': [] if memory_report is None else memory_report['warnings']}
        check()
        return result
