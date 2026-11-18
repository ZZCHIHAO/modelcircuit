"""Bounded device acceptance against independent references and Python export.

Run with backend and backend/tests on PYTHONPATH. This is a test utility, not a
model runtime or benchmark. It never downloads data, dependencies, or weights.
"""
import argparse
import json

import torch

from component_fixture import bundle_graph
from model_family_reference import make_reference
from model_state_mapping import canonical_parameters, canonical_state_dict, copy_platform_state
from studio_core.model_registry import build_registered_model
from studio_core.runtime import build_model
from studio_core.python_io import export_python


REPRESENTATIVES = ('model.resnet18_classifier', 'backbone.vgg11', 'model.squeezenet1_1_classifier')


def _absolute_error(actual, expected):
    return float((actual.detach() - expected.detach()).abs().max().item()) if actual.numel() else 0.0


def run_model_acceptance(model_id, *, device='cpu', side=32, num_classes=7, rtol=2e-5, atol=2e-6,
                         resource_profile='hardware-v5'):
    """Compare batch2 train output/input+parameter gradients/SGD/state/export.

    Copies every named state explicitly. Dropout uses identical per-forward
    seed. Tolerances target ordinary float32 rounding; any backend that exceeds
    them fails with the actual discrepancy instead of silently relaxing bounds.
    """
    device = torch.device(device)
    config = {'input_shape': [None, 3, side, side], 'resource_profile': resource_profile}
    classes = num_classes if model_id.startswith('model.') else None
    if classes is not None:
        config['num_classes'] = classes
    bundle = build_registered_model(model_id, config)
    graph = bundle_graph(bundle, {'x': config['input_shape']},
                         {key: value for key, value in bundle.requirements.items() if key != 'operator_set'})
    model = build_model(graph, device=device, capture=False).train()
    reference = make_reference(model_id, num_classes=classes).to(device).train()
    copy_platform_state(reference, model, model_id)
    namespace = {}
    exec(export_python(graph), namespace)
    standalone = namespace['Model']().to(device).train()
    standalone.load_state_dict(model.state_dict(), strict=True)
    torch.manual_seed(1729)
    sample = torch.randn(2, 3, side, side, device=device)
    outputs, inputs = [], []
    for candidate in (model, reference, standalone):
        x = sample.detach().clone().requires_grad_()
        torch.manual_seed(932)
        output = candidate(x) if candidate is reference else candidate(x=x)
        selected = {port: output[port] for port in ('logits',) if port in output} if classes is not None else {'features': output['features']}
        sum(value.square().mean() for value in selected.values()).backward()
        outputs.append(selected)
        inputs.append(x)
    output_error = input_error = parameter_error = state_error = 0.0
    for output in outputs[1:]:
        assert set(output) == set(outputs[0]), 'Output port mismatch'
        for name, value in output.items():
            torch.testing.assert_close(value, outputs[0][name], rtol=rtol, atol=atol)
            output_error = max(output_error, _absolute_error(value, outputs[0][name]))
    for x in inputs[1:]:
        torch.testing.assert_close(x.grad, inputs[0].grad, rtol=rtol, atol=atol)
        input_error = max(input_error, _absolute_error(x.grad, inputs[0].grad))
    assert torch.count_nonzero(inputs[0].grad).item() > 0, 'Trivial input gradient'
    canonical = canonical_parameters(model, model_id)
    expected = dict(reference.named_parameters())
    assert set(canonical) == set(expected), 'Reference parameter key mismatch'
    for name, value in canonical.items():
        assert value.grad is not None and expected[name].grad is not None, name
        torch.testing.assert_close(value.grad, expected[name].grad, rtol=rtol, atol=atol, msg=name)
        parameter_error = max(parameter_error, _absolute_error(value.grad, expected[name].grad))
    platform = dict(model.named_parameters())
    assert set(platform) == set(dict(standalone.named_parameters())), 'Standalone parameter key mismatch'
    for name, value in standalone.named_parameters():
        torch.testing.assert_close(value.grad, platform[name].grad, rtol=rtol, atol=atol, msg=name)
    before = {name: value.detach().clone() for name, value in canonical.items()}
    for candidate in (model, reference, standalone):
        torch.optim.SGD(candidate.parameters(), lr=.03).step()
    assert any(not torch.equal(value, before[name]) for name, value in canonical.items()), 'No parameter update'
    state = canonical_state_dict(model, model_id)
    reference_state = reference.state_dict()
    assert set(state) == set(reference_state), 'Reference state key mismatch'
    for name, value in state.items():
        torch.testing.assert_close(value, reference_state[name], rtol=rtol, atol=atol, msg=name)
        state_error = max(state_error, _absolute_error(value, reference_state[name]))
    for name, value in standalone.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=rtol, atol=atol, msg=name)
    bn_counters = [int(value.item()) for name, value in state.items() if name.endswith('num_batches_tracked')]
    expected_bn = 53 if 'resnet50' in model_id else 36 if 'resnet34' in model_id else 20 if 'resnet18' in model_id else 0
    assert bn_counters == [1] * expected_bn, 'BN counter mismatch'
    return {'model_id': model_id, 'device': str(device), 'input_shape': [2, 3, side, side], 'resource_profile': resource_profile,
            'parameter_count': sum(p.numel() for p in model.parameters()), 'state_keys': len(state),
            'bn_counters': bn_counters, 'rtol': rtol, 'atol': atol, 'dropout_seed': 932,
            'max_output_error': output_error, 'max_input_gradient_error': input_error,
            'max_parameter_gradient_error': parameter_error, 'max_reference_state_error': state_error,
            'standalone_strict_state': True, 'sgd_updated': True, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', action='append')
    parser.add_argument('--side', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--resource-profile', choices=('hardware-v5', 'vision-v4'), default='hardware-v5')
    args = parser.parse_args()
    torch.set_num_threads(1)
    # A reproducible acceptance process; no performance claim is made.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps([run_model_acceptance(identifier, device=args.device, side=args.side, num_classes=args.num_classes,
                                            resource_profile=args.resource_profile)
                      for identifier in args.model or REPRESENTATIVES], indent=2))


if __name__ == '__main__':
    main()
