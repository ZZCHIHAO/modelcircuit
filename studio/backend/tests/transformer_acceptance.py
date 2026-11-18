"""Acceptance for expanded Transformer graphs against independent Torch models."""
import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import torch

from studio_core.runtime import build_model
from studio_core.schema import validate_graph
from studio_core.transformer_templates import make_attention, make_encoder, make_small_vit
from transformer_reference import AttentionReference, EncoderReference, SmallViTReference


def bundle_graph(bundle):
    definition = next(item for item in bundle.definitions if item['id'] == bundle.entry['definition_id'])
    scope = definition['graph']
    return deepcopy({
        **{key: value for key, value in bundle.requirements.items() if key != 'operator_set'},
        'id': 'transformer_fixture', 'revision': 1, 'inputs': scope['inputs'], 'definitions': bundle.definitions,
        'instances': [bundle.entry],
        'calls': [dict(id='entry', instance_id=bundle.entry['id'],
                       inputs={port['id']: 'input:' + port['id'] for port in scope['inputs']})],
        'outputs': [dict(id=port['id'], source='call:entry:' + port['id']) for port in scope['outputs']],
    })


def _layers(root, reference, names):
    return {f'{root}components.{actual}.{key}': f'{reference}{expected}.{key}'
            for actual, expected in names.items() for key in ('weight', 'bias')}


def _attention_state(root='components.entry.', reference=''):
    return _layers(root, reference, {'q_proj': 'query', 'k_proj': 'key', 'v_proj': 'value', 'out_proj': 'output'})


def encoder_state_map(depth, root='components.entry.', reference=''):
    state = _layers(root, reference, {'final_norm': 'norm'})
    for i in range(depth):
        prefix, target = f'{root}components.block{i}.', f'{reference}blocks.{i}.'
        state.update(_layers(prefix, target, {'attention_norm': 'attn_norm', 'mlp_norm': 'mlp_norm',
                                             'mlp_up': 'up', 'mlp_down': 'down'}))
        state.update(_attention_state(prefix + 'components.self_attention.', target + 'self_attn.'))
    return state


def _vit_state(depth):
    state = _layers('components.entry.', '', {'patch_embed': 'patch', 'classifier': 'head'})
    state.update(encoder_state_map(depth, 'components.entry.components.encoder.', 'encoder.'))
    state['components.entry.components.cls_token.value'] = 'class_token'
    state['components.entry.components.pos_token.value'] = 'position'
    return state


def compare_reference(graph, reference, state_map, x, *, keep=None, dropout=0.0):
    device = x.device
    tolerances = {'output': (1e-6, 1e-5), 'gradient': (2e-6, 1e-4)} if device.type == 'cpu' else {
        'output': (1e-5, 1e-4), 'gradient': (2e-5, 2e-4)}
    errors = {}

    def close(actual, expected, group, name, tolerance='output'):
        if actual.dtype == torch.bool:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
            return
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), f'{name} is non-finite'
        atol, rtol = (0.0, 0.0) if tolerance == 'exact' else tolerances[tolerance]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                                   msg=lambda message: f'{name}: {message}')
        delta = (actual.detach() - expected.detach()).abs()
        expected = expected.detach().abs()
        relative = (delta.double() / expected.double()).masked_fill(expected == 0, 0)
        summary = errors.setdefault(group, {'max_abs': 0.0, 'max_rel_nonzero': 0.0})
        summary['max_abs'] = max(summary['max_abs'], float(delta.max()))
        summary['max_rel_nonzero'] = max(summary['max_rel_nonzero'], float(relative.max()))

    model = build_model(graph, device=device, capture=False)
    reference = reference.to(device)
    state = model.state_dict()
    counterpart = reference.state_dict()
    # Known paths are listed above. A missing, extra, renamed or aliased key fails before copying.
    assert set(state) == set(state_map), f'Graph state keys differ: {set(state) ^ set(state_map)}'
    assert len(state_map) == len(set(state_map.values())), 'Reference state mapping contains duplicate destinations'
    assert set(counterpart) == set(state_map.values()), f'Reference state keys differ: {set(counterpart) ^ set(state_map.values())}'
    for actual_key, expected_key in state_map.items():
        a, b = state[actual_key], counterpart[expected_key]
        assert a.shape == b.shape and a.dtype == b.dtype, f'State shape/dtype differs: {actual_key} and {expected_key}'
        if a.dtype == torch.bool:
            close(a, b, 'before', actual_key)
    model.load_state_dict({actual: counterpart[expected] for actual, expected in state_map.items()}, strict=True)
    parameters, ref_parameters = dict(model.named_parameters()), dict(reference.named_parameters())
    assert len({id(param) for param in parameters.values()}) == len(parameters), 'Duplicate platform parameters'
    assert set(ref_parameters) == {state_map[name] for name in parameters}, 'Trainable parameter paths differ'
    count = sum(param.numel() for param in parameters.values())
    assert validate_graph(graph)['parameter_count'] == count

    def inputs(value):
        return {'x': value} if keep is None else {'x': value, 'keep': keep}

    def rng():
        return torch.get_rng_state() if device.type == 'cpu' else torch.cuda.get_rng_state(device)

    a = x.detach().clone().requires_grad_(True)
    b = x.detach().clone().requires_grad_(True)
    torch.manual_seed(613)
    actual = model(**inputs(a))
    platform_rng = rng()
    torch.manual_seed(613)
    expected = reference(**inputs(b))
    assert torch.equal(platform_rng, rng()), 'Forward consumed different RNG state'
    assert set(actual) == set(expected) and actual, 'Output ports differ'
    for name, value in actual.items():
        close(value, expected[name], 'train_output', name)
    if 'attention' in actual:
        weights = actual['attention']
        close(weights.sum(-1), torch.ones_like(weights[..., 0]), 'weights_sum', 'normalized pre-dropout weights')
        if keep is not None:
            assert torch.count_nonzero(weights.masked_select(~keep[:, None, None, :])) == 0

    # Asymmetric readout avoids nearly constant norms of final LayerNorm activations.
    main = 'logits' if 'logits' in actual else 'features'
    coefficient = torch.linspace(-0.25, 0.75, actual[main].numel(), device=device).reshape_as(actual[main])
    (actual[main] * coefficient).mean().backward()
    (expected[main] * coefficient).mean().backward()
    close(a.grad, b.grad, 'input_grad', 'input x', 'gradient')
    for name, value in parameters.items():
        grad, ref_grad = value.grad, ref_parameters[state_map[name]].grad
        assert grad is not None and ref_grad is not None, f'Missing gradient for {name}'
        close(grad, ref_grad, 'parameter_grad', name, 'gradient')
    before = {name: param.detach().clone() for name, param in parameters.items()}
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    assert any(not value.detach().equal(before[name]) for name, value in parameters.items()), 'SGD did not update weights'
    state, counterpart = model.state_dict(), reference.state_dict()
    for actual_key, expected_key in state_map.items():
        close(state[actual_key], counterpart[expected_key], 'state_after_sgd', actual_key, 'gradient')

    fresh = build_model(graph, device=device, capture=False)
    fresh.load_state_dict(state, strict=True)
    model.eval()
    reference.eval()
    fresh.eval()
    with torch.no_grad():
        if dropout:
            model.train()
            torch.manual_seed(614)
            another = model(**inputs(x))[main]
            torch.manual_seed(615)
            assert not torch.equal(another, model(**inputs(x))[main]), 'Training dropout ignores seed'
            model.eval()
        eval_rng = rng()
        actual = model(**inputs(x))
        expected = reference(**inputs(x))
        reloaded = fresh(**inputs(x))
        assert torch.equal(eval_rng, rng()), 'Eval forward consumed RNG state'
        for name, value in actual.items():
            close(value, expected[name], 'eval_output', name)
            close(value, reloaded[name], 'reloaded_eval', name, 'exact')
    return dict(parameter_count=count, state_map=state_map, tolerances=tolerances, errors=errors, passed=True)


def run_attention_case(*, tokens, heads, mode, batch, dropout=0.0, device='cpu'):
    torch.manual_seed(479)
    settings = dict(sequence_length=tokens, hidden_dim=8, num_heads=heads, dropout=dropout,
                    masked=mode == 'external', causal=mode == 'causal', return_attention=True)
    graph = bundle_graph(make_attention(**settings))
    reference = AttentionReference(8, heads, dropout, causal_length=tokens if mode == 'causal' else None)
    state = _attention_state()
    keep = None
    if mode == 'external':
        keep = (torch.arange(tokens, device=device)[None, :] + torch.arange(batch, device=device)[:, None]) % 3 != 1
        if tokens == 1:
            keep.fill_(True)
    if mode == 'causal':
        state['components.entry.components.causal_mask.keep'] = 'causal_keep'
    x = torch.randn(batch, tokens, 8, device=device)
    return dict(model='attention', settings=settings, batch=batch,
                **compare_reference(graph, reference, state, x, keep=keep, dropout=dropout))


def run_encoder_case(*, device='cpu', dropout=0.25, attention_dropout=0.15):
    torch.manual_seed(479)
    settings = dict(sequence_length=5, hidden_dim=8, num_heads=2, mlp_dim=11, depth=2,
                    dropout=dropout, attention_dropout=attention_dropout)
    graph = bundle_graph(make_encoder(**settings))
    reference = EncoderReference(8, 2, 11, 2, dropout, attention_dropout)
    x = torch.randn(2, 5, 8, device=device)
    return dict(model='encoder', settings=settings,
                **compare_reference(graph, reference, encoder_state_map(2), x, dropout=dropout or attention_dropout))


def run_vit_case(*, device='cpu', settings=None):
    torch.manual_seed(479)
    settings = {**dict(image_size=32, patch_size=4, channels=3, hidden_dim=64, num_heads=4, mlp_dim=128,
                       depth=2, num_classes=2, dropout=0.0, attention_dropout=0.0), **(settings or {})}
    graph = bundle_graph(make_small_vit(**settings))
    reference = SmallViTReference(**settings)
    x = torch.randn(2, settings['channels'], settings['image_size'], settings['image_size'], device=device)
    return dict(model='vit', settings=settings,
                **compare_reference(graph, reference, _vit_state(settings['depth']), x,
                                    dropout=settings['dropout'] or settings['attention_dropout']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--section', choices=('attention', 'vit', 'all'), default='all')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'cpu|cuda:[0-9]+', args.device):
        parser.error('--device must be cpu or an explicit cuda:N')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    report = dict(device=args.device, interpreter=sys.executable, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  section=args.section, cases=[], status='failed')
    try:
        if device.type == 'cuda':
            if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                report['status'] = 'blocked'
                raise RuntimeError(f'{device} is unavailable')
            report['device_name'] = torch.cuda.get_device_name(device)
        cases = report['cases']
        if args.section in ('all', 'attention'):
            for tokens in (1, 5):
                for heads in (1, 2, 4):
                    for mode in ('none', 'causal', 'external'):
                        for batch in (1, 2):
                            cases.append(run_attention_case(tokens=tokens, heads=heads, mode=mode, batch=batch, device=device))
            cases.append(run_attention_case(tokens=5, heads=2, mode='external', batch=2, dropout=0.35, device=device))
        if args.section in ('all', 'vit'):
            cases.append(run_encoder_case(device=device, dropout=0.0, attention_dropout=0.0))
            cases.append(run_encoder_case(device=device))
            cases.append(run_vit_case(device=device))
            cases.append(run_vit_case(device=device, settings=dict(image_size=8, patch_size=2, channels=1, hidden_dim=8,
                         num_heads=2, mlp_dim=12, depth=1, num_classes=3, dropout=0.2, attention_dropout=0.15)))
        report['status'] = 'passed'
    except (AssertionError, ValueError, RuntimeError) as exc:
        report['error'] = str(exc)
    report['passed_count'] = len(report['cases'])
    report['cases_sha256'] = hashlib.sha256(json.dumps(report['cases'], sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{args.device} {args.section}: {report["status"]}, {report["passed_count"]} cases; {args.output}')
    if 'error' in report:
        print(report['error'])
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
