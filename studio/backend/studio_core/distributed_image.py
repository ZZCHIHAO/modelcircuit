"""Train an ImageFolder research project under a single-host torchrun job."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import timedelta
import io
import os
from pathlib import Path
import random
import time

from .classification_data import _json_digest, build_datasets
from .classification_runtime import _atomic_write, _cpu_tree, _write_json
from .classification_task import ClassificationTask
from .execution_policy import check_config_batch
from .experiment_spec import checkpoint_signature, validate_supervised_config
from .project_runner import _new_run, _record_project, _utc, _write_result, evaluation_dataset_digest
from .research_project import bind_project, read_project
from .schema import prepare_graph


CHECKPOINT_VERSION = 1


def epoch_shard(count, seed, epoch, rank, world_size, per_device_batch, accumulation):
    """Divide each shuffled full update across ranks without padding or duplication."""
    import torch

    global_micro_batch = world_size * per_device_batch
    effective_batch = global_micro_batch * accumulation
    usable = count - count % effective_batch
    if not usable:
        raise ValueError(f'Training split has {count} images and needs at least one effective global batch of {effective_batch}')
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch - 1)).tolist()
    offset = rank * per_device_batch
    return [index for start in range(offset, usable, global_micro_batch)
            for index in permutation[start:start + per_device_batch]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('project', 'data', 'out', 'progress'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Directory containing an atomically committed last.pt')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--per-device-batch', type=int, required=True)
    parser.add_argument('--accumulation', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.per_device_batch <= 64:
        parser.error('--per-device-batch must be between 1 and 64')
    if not 1 <= args.accumulation <= 32:
        parser.error('--accumulation must be between 1 and 32')
    if not 1 <= args.epochs <= 1000:
        parser.error('--epochs must be between 1 and 1000')
    return args


def _runtime(kind):
    import torch
    import torch.distributed as dist

    try:
        rank, world, local = (int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
        local_world = int(os.environ['LOCAL_WORLD_SIZE'])
    except (KeyError, ValueError) as exc:
        raise ValueError('Start distributed classification with single-host torchrun') from exc
    if world < 1 or rank != local or local_world != world or not 0 <= rank < world:
        raise ValueError('Distributed classification currently supports only a single host')
    if kind == 'cuda':
        if not dist.is_nccl_available() or torch.cuda.device_count() != world:
            raise ValueError('CUDA DDP requires NCCL and a CUDA_VISIBLE_DEVICES set containing exactly one GPU per rank')
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    # Separate workers have their own graph and data loader. Avoid competing for
    # every CPU core when the same host runs multiple ranks.
    torch.set_num_threads(1)
    return rank, world, device, backend


def _loader(dataset, indices, batch, seed):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch, sampler=indices, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def _finite(tensor):
    import torch

    return not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())


def _checked_loss(model, task, manifest, batch):
    outputs = model(**batch['features'])
    task.validate_outputs(outputs, batch, manifest)
    loss = task.loss(outputs, batch)
    if not _finite(loss):
        raise ValueError('Non-finite distributed classification loss')
    return outputs, loss


def _metrics(task, totals, device, *, distributed):
    import torch
    import torch.distributed as dist

    keys = ('loss_sum', 'correct', 'count')
    values = torch.tensor([totals.get(key, 0) for key in keys], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    loss, correct, count = values.tolist()
    if count < 1:
        raise ValueError('Evaluation split is empty')
    return task.finalize(dict(loss_sum=loss, correct=int(correct), count=int(count)))


def _setup_model(graph, resolved, config, device, *, resume):
    import torch
    from .runtime import build_model

    seed = config['seed']
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    options = {}
    if graph.get('resource_profile') == 'hardware-v5':
        from .memory_admission import admit_memory

        options['memory_report'] = admit_memory(resolved, device, batch_size=config['batch_size'],
                                                 training=True, optimizer=config['optimizer'], capture=False,
                                                 workflow='classification', resume=resume)
    model = build_model(graph, device=device, capture=False, **options)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('Distributed classification requires at least one trainable parameter')
    optimizer_class = torch.optim.SGD if config['optimizer'] == 'sgd' else torch.optim.AdamW
    return model, optimizer_class(trainable, lr=config['lr'])


def _train_epoch(model, task, manifest, raw_batches, optimizer, accumulation, device, deadline):
    from torch.nn.utils import clip_grad_norm_

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    for index, raw in enumerate(raw_batches, 1):
        _check_deadline(deadline)
        batch = task.bind_batch(raw, device)
        synchronize = index % accumulation == 0
        with nullcontext() if synchronize else model.no_sync():
            outputs, loss = _checked_loss(model, task, manifest, batch)
            (loss / accumulation).backward()
        totals = task.merge(totals, task.summarize(outputs, batch, loss))
        if synchronize:
            # This check keeps the optimizer from committing a non-finite
            # update. Infinite norm makes the clipping coefficient exactly 1.
            clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if any(not _finite(parameter) for parameter in model.parameters()):
                raise ValueError('Non-finite parameter after distributed optimizer step')
    return _metrics(task, totals, device, distributed=True)


def _evaluate(model, task, manifest, batches, device, deadline, *, distributed, writer=None):
    import torch

    model.eval()
    totals = {}
    with torch.inference_mode():
        for raw in batches:
            _check_deadline(deadline)
            batch = task.bind_batch(raw, device)
            outputs, loss = _checked_loss(model, task, manifest, batch)
            totals = task.merge(totals, task.summarize(outputs, batch, loss))
            if writer is not None:
                writer.writerows(task.prediction_rows(outputs, batch, manifest))
    return _metrics(task, totals, device, distributed=distributed)


def _broadcast_buffers(model):
    import torch
    import torch.distributed as dist

    # Evaluate a single model state across all ranks without synchronized
    # forward calls. Graphs may have BatchNorm buffers that differ by rank.
    with torch.no_grad():
        for buffer in model.buffers():
            if buffer.numel():
                dist.broadcast(buffer, src=0)


def _check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Distributed training exceeded max_runtime_seconds')


def _identity(project, config, execution):
    return _json_digest({'training': checkpoint_signature(project['graph'], config, project['dataset']['digest']),
                         'execution': execution, 'version': CHECKPOINT_VERSION})


def _read_checkpoint(path, identity, epochs, world):
    import torch

    state = torch.load(path / 'last.pt', weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or state.get('ddp_resume_version') != CHECKPOINT_VERSION or state.get('ddp_signature') != identity:
        raise ValueError('Resume requires the same model, locked data, optimizer and distributed execution settings')
    completed = state.get('completed_epochs')
    rng_states = state.get('rank_rng')
    if (type(completed) is not int or not 1 <= completed < epochs or not isinstance(rng_states, list) or
            len(rng_states) != world):
        raise ValueError('Resume requires a complete checkpoint and a target epoch after the saved epoch')
    return state


def _random_state(device):
    import torch

    return {'python': random.getstate(), 'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _restore_random_state(state, device):
    import torch

    random.setstate(state['python'])
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def _save_checkpoint(path, payload):
    import torch

    _atomic_write(path, lambda stream: torch.save(payload, stream))


def _seed_rank(seed, rank, device):
    import torch

    random.seed(seed + rank)
    torch.random.default_generator.manual_seed(seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed + rank)


def _measure_devices(world, rank, device):
    import torch
    import torch.distributed as dist

    device_info = {'rank': rank, 'visible_device_index': device.index, 'type': device.type}
    if device.type == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        device_info['name'] = torch.cuda.get_device_name(device)
        if len(visible) == world:
            device_info['device_selector'] = visible[rank]
    devices = [None] * world
    dist.all_gather_object(devices, device_info)
    return devices


def _progress(path, log, **event):
    record = {'time': _utc(), **event}
    _write_json(path, record)
    log.append(record)


def _train(args, rank, world, device, backend, directory, report, log):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from .python_io import export_python

    project = read_project(args.project)
    if project['config']['profile'] != 'classification-v2' or project['dataset']['kind'] != 'image_folder':
        raise ValueError('Distributed execution supports ImageFolder research projects')
    if project['graph_error'] is not None:
        raise ValueError(project['graph_error'])
    deadline = time.monotonic() + project['config']['max_runtime_seconds']

    def check():
        _check_deadline(deadline)
        return False

    config, manifest = bind_project(project, args.data, cancelled=check)
    config = validate_supervised_config({**config, 'device': str(device), 'epochs': args.epochs,
                                         'batch_size': args.per_device_batch})
    graph = project['graph']
    check_config_batch(graph, args.per_device_batch)
    resolved, shapes, _ = prepare_graph(graph)
    task = ClassificationTask(config)
    task.validate_contract(resolved, shapes, manifest)
    datasets = build_datasets(manifest, cancelled=check)
    seed = config['seed']
    effective_batch = args.per_device_batch * world * args.accumulation
    available = len(datasets['train'])
    dropped = available % effective_batch
    first_indices = epoch_shard(available, seed, 1, rank, world, args.per_device_batch, args.accumulation)
    execution = {'strategy': 'ddp', 'backend': backend, 'device_type': device.type, 'world_size': world,
                 'per_device_batch': args.per_device_batch, 'accumulation': args.accumulation,
                 'effective_global_batch': effective_batch, 'train_samples_available': available,
                 'train_samples_dropped_per_epoch': dropped,
                 'train_samples_per_rank': len(first_indices), 'devices': _measure_devices(world, rank, device)}
    signature = _identity(project, config, execution)
    # Ranks must agree on identities before entering synchronized forward/backward.
    identities = [None] * world
    dist.all_gather_object(identities, signature)
    if any(other != signature for other in identities):
        raise ValueError('The distributed workers see different models, data or execution settings')

    model, optimizer = _setup_model(graph, resolved, config, device, resume=args.resume is not None)
    completed, updates, history, best = 0, 0, [], None
    saved_random = None
    if args.resume:
        state = _read_checkpoint(args.resume, signature, args.epochs, world)
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        completed, updates, history, best = (state[key] for key in ('completed_epochs', 'global_step', 'history', 'best'))
        saved_random = state['rank_rng'][rank]
        if rank == 0:
            _save_checkpoint(directory / 'last.pt', state)
        del state
    # Return only logits to DDP. Auxiliary graph outputs can have branches with
    # parameters outside the loss and should not participate in its reducer.
    class ClassificationModel(torch.nn.Module):
        def __init__(self, graph_model):
            super().__init__()
            self.graph_model = graph_model

        def forward(self, **features):
            return {'logits': self.graph_model(**features)['logits']}

    wrapper = ClassificationModel(model)
    distributed_model = DistributedDataParallel(wrapper, device_ids=[device.index] if device.type == 'cuda' else None,
                                                 find_unused_parameters=True, gradient_as_bucket_view=True)
    if saved_random is not None:
        _restore_random_state(saved_random, device)
    else:
        _seed_rank(seed, rank, device)

    if rank == 0:
        _record_project(report, project, config['profile'], str(device))
        report.update(seed=seed, local_dataset_digest=manifest['dataset_digest'], execution=execution,
                      evaluation=config['profile'] + '-test-v1', evaluation_dataset_digest=evaluation_dataset_digest(project['dataset']),
                      epochs=args.epochs, batch_size=args.per_device_batch, resumed_from_epoch=completed)
        _progress(args.progress, log, phase='ready', epoch=completed, epochs=args.epochs, updates=updates,
                  resume_available=bool(completed))
    eval_indices = range(rank, len(datasets['val']), world)
    val_loader = _loader(datasets['val'], eval_indices, args.per_device_batch, seed + 1)
    identity = {'version': 3 if graph.get('schema_version') == 2 else 2, 'profile': config['profile'],
                'checkpoint_signature': checkpoint_signature(graph, config, manifest['dataset_digest']),
                'model_digest': project['model_digest'], 'dataset_digest': manifest['dataset_digest']}
    checkpoint_metadata = {**identity, 'class_names': manifest['class_names'], 'input_shape': manifest['input_shape'],
                           'preprocessing': manifest['preprocessing'], 'config': config}
    if graph.get('schema_version') == 2:
        checkpoint_metadata['adapter'] = {'id': 'classification', 'version': 1}
    for epoch in range(completed + 1, args.epochs + 1):
        train_indices = epoch_shard(available, seed, epoch, rank, world, args.per_device_batch, args.accumulation)
        train_loader = _loader(datasets['train'], train_indices, args.per_device_batch, seed + epoch)
        train = _train_epoch(distributed_model, task, manifest, train_loader, optimizer, args.accumulation, device, deadline)
        _broadcast_buffers(model)
        val = _evaluate(wrapper, task, manifest, val_loader, device, deadline, distributed=True)
        updates += len(train_loader) // args.accumulation
        history.append({'epoch': epoch, 'step': updates, 'loss': train['loss'], 'train_loss': train['loss'],
                        'val_loss': val['loss'], 'train_accuracy': train['accuracy'], 'val_accuracy': val['accuracy'],
                        'train_count': train['count'], 'val_count': val['count']})
        rng = [None] * world
        dist.all_gather_object(rng, _random_state(device))
        if rank == 0:
            weights = _cpu_tree(model.state_dict())
            if best is None or val['loss'] < best['metrics']['val']['loss']:
                best = {**checkpoint_metadata, 'completed_epochs': epoch, 'global_step': updates,
                        'model_state': weights, 'metrics': {'train': train, 'val': val}}
            state = {**checkpoint_metadata, 'ddp_resume_version': CHECKPOINT_VERSION, 'ddp_signature': signature,
                     'rank_rng': rng, 'history': history, 'completed_epochs': epoch, 'global_step': updates,
                     'model_state': weights, 'optimizer_state': _cpu_tree(optimizer.state_dict()), 'best': best}
            _save_checkpoint(directory / 'last.pt', state)
            del state, weights
            _progress(args.progress, log, phase='epoch', epoch=epoch, epochs=args.epochs, updates=updates,
                      train_loss=train['loss'], val_loss=val['loss'], resume_available=True)
        dist.barrier()

    if rank == 0:
        model.load_state_dict(best['model_state'], strict=True)
        # Test the selected checkpoint once and write rows in manifest order.
        with io.StringIO(newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=task.prediction_fields)
            writer.writeheader()
            test_loader = _loader(datasets['test'], range(len(datasets['test'])), args.per_device_batch, seed + 2)
            test = _evaluate(wrapper, task, manifest, test_loader, device, deadline, distributed=False, writer=writer)
            _atomic_write(directory / 'predictions.csv', lambda file: file.write(stream.getvalue().encode('utf-8')))
        _save_checkpoint(directory / 'best.pt', best)
        metrics = {**best['metrics'], 'test': test}
        _write_json(directory / 'metrics.json', {'metrics': metrics, 'history': history, 'global_step': updates,
                                                 'completed_epochs': args.epochs, 'best_epoch': best['completed_epochs']})
        _write_json(directory / 'manifest.json', manifest)
        _write_json(directory / 'run_spec.json', {**identity, 'graph': graph, 'config': config, 'execution': execution,
                                                  'data_spec': manifest['data_spec']})
        (directory / 'model_snapshot.py').write_text(export_python(graph), encoding='utf-8', newline='\n')
        report.update(metrics=metrics, history=history, best_epoch=best['completed_epochs'], global_step=updates)
        artifacts = {'best_checkpoint': 'best.pt', 'last_checkpoint': 'last.pt', 'predictions': 'predictions.csv',
                     'metrics': 'metrics.json', 'run_spec': 'run_spec.json', 'model_snapshot': 'model_snapshot.py',
                     'manifest': 'manifest.json'}
        _progress(args.progress, log, phase='test', epoch=args.epochs, epochs=args.epochs, updates=updates,
                  resume_available=True)
        _write_result(directory, report, 'succeeded', log, {key: directory / name for key, name in artifacts.items()})


def main(argv=None):
    import torch.distributed as dist

    args = parse_args(argv)
    rank, world, device, backend = _runtime(args.device)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
    directory, report, log = None, None, []
    try:
        if rank == 0:
            directory, report = _new_run(args.out, 'train', {'project': str(args.project), 'data_path': str(args.data),
                                                             'device': str(device), 'strategy': 'ddp'})
            _progress(args.progress, log, phase='bind', epoch=0, epochs=args.epochs, updates=0, resume_available=False)
        _train(args, rank, world, device, backend, directory, report, log)
    except (Exception, KeyboardInterrupt) as error:
        if directory is not None:
            _write_result(directory, report, 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', log, {}, error)
        raise
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
