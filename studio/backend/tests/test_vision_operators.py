"""A1 operator contracts, independent numerical references and state isolation."""
import copy
import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from studio_core import runtime, schema
from studio_core.python_io import export_python


NEW_KINDS = ("BatchNorm2d", "Identity", "Sigmoid", "Multiply", "Scale", "OneMinus")


def api(module):
    try:
        return importlib.import_module("studio_core." + module)
    except ModuleNotFoundError as error:
        raise AssertionError("A1 operator module is not implemented: " + module) from error


def graph_for(kind, params=None, shape=None, second_shape=None):
    shape = [None, 3, 2, 2] if shape is None else shape
    binary = kind in ("Add", "Concat", "Multiply")
    inputs = [{"id": "x", "shape": shape, "dtype": "float32"}]
    if binary:
        inputs.append({"id": "y", "shape": shape if second_shape is None else second_shape, "dtype": "float32"})
    return {"schema_version": 1, "id": "operator_test", "revision": 1,
            "inputs": inputs,
            "instances": [{"id": "operation", "kind": kind, "params": params or {}}],
            "calls": [{"id": "apply_op", "instance_id": "operation",
                       "inputs": {"a": "input:x", "b": "input:y"} if binary else {"x": "input:x"}}],
            "outputs": [{"id": "out", "source": "call:apply_op"}], "definitions": []}


def bn_state(module):
    if module.affine:
        with torch.no_grad():
            module.weight.copy_(torch.tensor([0.75, -1.25, 2.0]))
            module.bias.copy_(torch.tensor([0.2, -0.3, 0.4]))
    if module.track_running_stats:
        module.running_mean.copy_(torch.tensor([0.4, -0.2, 0.7]))
        module.running_var.copy_(torch.tensor([1.2, 0.8, 1.5]))
        module.num_batches_tracked.fill_(3)


def unused_bn_graph(channels=3, track=True):
    graph = graph_for("BatchNorm2d", {"num_features": channels, "affine": False, "track_running_stats": track}, [1, 1])
    graph["calls"] = []
    graph["outputs"][0]["source"] = "input:x"
    return graph


class OperatorDeclarationTests(unittest.TestCase):
    def test_registered_specs_keep_constructor_and_editor_defaults_distinct(self):
        operators = api("operators")
        expected = {"Linear", "ReLU", "GELU", "Add", "Concat", "LayerNorm", "Dropout",
                    "Conv2d", "MaxPool2d", "AdaptiveAvgPool2d", "Flatten", "Conv3d", "MaxPool3d", "AdaptiveAvgPool3d", "Upsample",
                    "SVC", "SVR", "RandomForestClassifier", "RandomForestRegressor", "PCA", "KMeans", *NEW_KINDS}
        self.assertEqual(set(operators.operator_kinds()), expected)
        self.assertIsInstance(operators.operator_kinds(), tuple)
        self.assertEqual(operators.operator_spec("Dropout").defaults["p"], 0.5)
        self.assertEqual(operators.operator_defaults("Dropout"), {"p": 0.1})
        self.assertEqual(operators.operator_spec("Multiply").input_ports, ("a", "b"))
        self.assertEqual(operators.operator_spec("BatchNorm2d").required, ("num_features",))
        self.assertEqual(operators.operator_spec("BatchNorm2d").state_kind, "parameters_and_buffers")
        self.assertIsNone(operators.operator_spec("Scale").python_constructor)

    def test_returned_specs_defaults_normalization_and_shapes_are_isolated(self):
        operators, rules = api("operators"), api("operator_rules")
        original = {"normalized_shape": [3]}
        result = rules.normalize_operator("LayerNorm", original)
        result["normalized_shape"][0] = 999
        self.assertEqual(original, {"normalized_shape": [3]})
        defaults = operators.operator_defaults("AdaptiveAvgPool2d")
        defaults["output_size"][0] = 999
        self.assertEqual(operators.operator_defaults("AdaptiveAvgPool2d"), {"output_size": [1, 1]})
        spec = operators.operator_spec("BatchNorm2d")
        spec.defaults["eps"] = 999
        self.assertEqual(operators.operator_spec("BatchNorm2d").defaults["eps"], 1e-5)
        spec.editor_defaults["num_features"] = 999
        self.assertNotEqual(operators.operator_defaults("BatchNorm2d")["num_features"], 999)
        shape = [None, 3, 2, 2]
        output, count = rules.infer_operator("BatchNorm2d", {"num_features": 3}, {"x": shape})
        output[-1] = 999
        self.assertEqual(shape, [None, 3, 2, 2])
        self.assertEqual(count, 6)

    def test_descriptors_and_rules_import_without_torch_or_gui(self):
        script = """import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'dearpygui'):
            raise RuntimeError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from studio_core.operators import operator_defaults
from studio_core.operator_rules import infer_operator
assert operator_defaults('Scale') == {'factor': 1.0}
assert infer_operator('BatchNorm2d', {'num_features': 3}, {'x': [None,3,2,2]}) == ([None,3,2,2],6)
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_omissions_and_supported_parameter_sets_are_preserved(self):
        rules = api("operator_rules")
        for kind, params in (("ReLU", {}), ("Concat", {}), ("Linear", {"in_features": 2, "out_features": 3})):
            self.assertEqual(rules.normalize_operator(kind, params), params)
        self.assertEqual(rules.normalize_operator("Conv2d", {"in_channels": 1, "out_channels": 2, "kernel_size": 3})["padding"], 0)
        for kind, params in (("ReLU", {"inplace": False}), ("Linear", {"in_features": 2, "out_features": 3, "bias": False}), ("Dropout", {})):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                rules.normalize_operator(kind, params)

    def test_new_constructor_defaults_are_mathematical_not_editor_presets(self):
        rules = api("operator_rules")
        self.assertEqual(rules.normalize_operator("Scale", {}), {"factor": 1.0})
        self.assertEqual(rules.normalize_operator("BatchNorm2d", {"num_features": 3}),
                         {"num_features": 3, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True})
        self.assertEqual(rules.infer_operator("Multiply", {}, {"a": [None, 3], "b": [None, 3]}), ([None, 3], 0))
        with self.assertRaises(ValueError):
            rules.infer_operator("Identity", {}, {"wrong": [None, 3]})
        with self.assertRaises(ValueError):
            rules.normalize_operator("__import__", {})

    def test_invalid_parameters_are_rejected_before_construction(self):
        rules = api("operator_rules")
        cases = [("Scale", {"factor": value}) for value in (True, float("nan"), float("inf"), -float("inf"), "1", None)]
        cases += [("BatchNorm2d", {"num_features": 3, name: value}) for name, values in (
            ("num_features", (0, -1, True, 3.0)), ("eps", (0, -1, True, float("inf"), float("nan"))),
            ("momentum", (-0.1, 1.1, True, float("inf"), float("nan"))),
            ("affine", (0, 1, "true")), ("track_running_stats", (0, 1, "false"))) for value in values]
        cases += [(kind, {"inplace": False}) for kind in ("Identity", "Sigmoid", "OneMinus")]
        for kind, params in cases:
            with self.subTest(kind=kind, params=params), self.assertRaises(ValueError):
                rules.normalize_operator(kind, params)
        for momentum in (None, 0, 1, 0.25):
            self.assertEqual(rules.normalize_operator("BatchNorm2d", {"num_features": 3, "momentum": momentum})["momentum"], momentum)


class VisionSchemaTests(unittest.TestCase):
    def test_unused_bn_buffers_fail_preflight_before_any_constructor(self):
        direct = unused_bn_graph(10**12)
        nested = graph_for("Identity", shape=[1, 1])
        nested["instances"] = [{"id": "entry", "kind": "chip", "definition_id": "block", "params": {}}]
        nested["calls"] = []
        nested["outputs"][0]["source"] = "input:x"
        nested["definitions"] = [{"id": "block", "public_parameters": [], "bindings": {}, "graph": direct}]
        unused_definition = graph_for("Identity", shape=[1, 1])
        unused_definition["definitions"] = copy.deepcopy(nested["definitions"])
        for name, graph in (("unused_instance", direct), ("nested_instance", nested), ("unused_definition", unused_definition)):
            with self.subTest(name=name):
                with patch("studio_core.runtime.make_operator", side_effect=AssertionError("constructor reached before buffer budget")):
                    with self.assertRaisesRegex(ValueError, "buffer.*budget"):
                        runtime.build_model(graph)
                report = schema.validate_graph(graph)
                self.assertFalse(report["valid"])
                self.assertEqual(report["errors"][0]["field"], "resource_limits")
        # Width alone allocates no storage when both affine and tracking are off.
        report = schema.validate_graph(unused_bn_graph(10**12, track=False))
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["parameter_count"], 0)

    def test_bn_buffers_obey_per_tensor_and_separate_total_limits(self):
        with patch("studio_core.schema.resource_limits", return_value=schema.ResourceLimits(100, 10, 100)):
            self.assertFalse(schema.validate_graph(unused_bn_graph(11))["valid"])
            valid = schema.validate_graph(unused_bn_graph(10))
            self.assertTrue(valid["valid"], valid["errors"])
            self.assertEqual(valid["parameter_count"], 0)
        graph = unused_bn_graph(3)
        graph["instances"].append({"id": "other", "kind": "BatchNorm2d", "params": {"num_features": 3, "affine": False}})
        # 2 * (3 running means + 3 variances + 1 counter) = 14 buffers;
        # the sole live tensor has one element and parameters remain zero.
        with patch("studio_core.schema.resource_limits", return_value=schema.ResourceLimits(100, 100, 13)):
            report = schema.validate_graph(graph)
            self.assertFalse(report["valid"])
            self.assertIn("buffer", report["errors"][0]["message"])
        with patch("studio_core.schema.resource_limits", return_value=schema.ResourceLimits(100, 100, 14)):
            report = schema.validate_graph(graph)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["parameter_count"], 0)

    def test_buffer_budget_counts_shared_chip_instance_once_and_independent_twice(self):
        child = unused_bn_graph(3)
        for key in ("schema_version", "revision", "id", "definitions"): child.pop(key)
        graph = graph_for("Identity", shape=[1, 1])
        graph["definitions"] = [{"id": "block", "public_parameters": [], "bindings": {}, "graph": child}]
        graph["instances"] = [{"id": "shared", "kind": "chip", "definition_id": "block", "params": {}}]
        graph["calls"] = [{"id": key, "instance_id": "shared", "inputs": {"x": "input:x"}} for key in ("first", "second")]
        graph["outputs"][0]["source"] = "call:second"
        with patch("studio_core.schema.resource_limits", return_value=schema.ResourceLimits(100, 100, 10)):
            report = schema.validate_graph(graph)
            self.assertTrue(report["valid"], report["errors"])
            model = runtime.build_model(graph, capture=False)
            self.assertEqual(sum(b.numel() for b in model.buffers()), 7)
            graph["instances"].append({"id": "independent", "kind": "chip", "definition_id": "block", "params": {}})
            graph["calls"][1]["instance_id"] = "independent"
            with patch("studio_core.runtime.make_operator", side_effect=AssertionError("allocated before aggregate buffer preflight")):
                with self.assertRaisesRegex(ValueError, "buffer.*budget"):
                    runtime.build_model(graph)

    def test_old_diagnostics_keep_parameter_field_locations(self):
        cases = [("Linear", {"in_features": 0, "out_features": 3}, [None, 2], "in_features"),
                 ("Linear", {"in_features": 3, "out_features": 3}, [None, 2], "in_features"),
                 ("Concat", {"dim": 1.5}, [None, 2], "dim"),
                 ("Concat", {"dim": 4}, [None, 2], "dim"),
                 ("LayerNorm", {"normalized_shape": 0}, [None, 2], "normalized_shape"),
                 ("LayerNorm", {"normalized_shape": 3}, [None, 2], "normalized_shape"),
                 ("Dropout", {"p": 2}, [None, 2], "p")]
        for kind, params, shape, expected in cases:
            with self.subTest(kind=kind, params=params):
                report = schema.validate_graph(graph_for(kind, params, shape))
                self.assertFalse(report["valid"])
                self.assertEqual(report["errors"][0]["field"], expected)

    def test_new_shapes_and_bn_parameter_counts_exclude_buffers(self):
        for kind in NEW_KINDS:
            for affine in (False, True):
                params = {"num_features": 3, "affine": affine} if kind == "BatchNorm2d" else {}
                with self.subTest(kind=kind, affine=affine):
                    graph = graph_for(kind, params)
                    before = copy.deepcopy(graph)
                    report = schema.validate_graph(graph)
                    self.assertTrue(report["valid"], report["errors"])
                    self.assertEqual(report["shapes"]["call:apply_op"], [None, 3, 2, 2])
                    self.assertEqual(report["parameter_count"], 6 if kind == "BatchNorm2d" and affine else 0)
                    self.assertEqual(graph, before)

    def test_static_broadcast_bad_bn_rank_channels_and_ports_are_rejected(self):
        cases = [graph_for("Multiply", shape=[None, 3], second_shape=[None, 1]),
                 graph_for("BatchNorm2d", {"num_features": 2}),
                 graph_for("BatchNorm2d", {"num_features": 3}, [None, 3]),
                 graph_for("Scale", {"factor": True})]
        bad_ports = graph_for("Multiply")
        bad_ports["calls"][0]["inputs"] = {"x": "input:x"}
        cases.append(bad_ports)
        for graph in cases:
            with self.subTest(graph=graph):
                self.assertFalse(schema.validate_graph(graph)["valid"])

    def test_new_parameter_and_tensor_budgets_fail_before_module_allocation(self):
        # Use a lower parameter budget while keeping the input tensor permitted.
        small = schema.ResourceLimits(5, 1000, 10000)
        with patch("studio_core.runtime.make_operator", side_effect=AssertionError("module allocated before graph validation")):
            with patch("studio_core.schema.resource_limits", return_value=small):
                with self.assertRaisesRegex(ValueError, "parameter budget"):
                    runtime.build_model(graph_for("BatchNorm2d", {"num_features": 3}))
            with self.assertRaisesRegex(ValueError, "Tensor element budget"):
                runtime.build_model(graph_for("Scale", shape=[None, 20_000]))

    def test_python_v4_supports_each_new_kind_after_a2(self):
        from studio_core.python_io import import_python, python_profile
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                graph = graph_for(kind, {"num_features": 3} if kind == "BatchNorm2d" else {})
                self.assertEqual(python_profile(graph), 'vision-pytorch-v4')
                restored = import_python(export_python(graph))
                self.assertEqual(restored['instances'][0]['kind'], kind)
                self.assertTrue(schema.validate_graph(restored)['valid'])


class VisionNumericalTests(unittest.TestCase):
    def test_large_finite_integer_scale_factors_execute_like_floats(self):
        make_operator = api("operator_runtime").make_operator
        for factor in (10**20, -(10**20)):
            with self.subTest(factor=factor):
                x = torch.tensor([[1e-20]], requires_grad=True)
                expected = torch.tensor([[1.0 if factor > 0 else -1.0]])
                direct = make_operator("Scale", {"factor": factor})(x)
                torch.testing.assert_close(direct, expected)
                torch.testing.assert_close(direct, make_operator("Scale", {"factor": float(factor)})(x))
                direct.sum().backward()
                torch.testing.assert_close(x.grad, torch.tensor([[float(factor)]]))
                graph = graph_for("Scale", {"factor": factor}, [None, 1])
                torch.testing.assert_close(runtime.build_model(graph)(x=x.detach())["out"], expected)
                self.assertIs(type(graph["instances"][0]["params"]["factor"]), int)

    def test_scale_multiply_and_complement_have_real_gradients(self):
        make_operator = api("operator_runtime").make_operator
        x = torch.tensor([[2., -1.]], requires_grad=True)
        y = torch.tensor([[3., 4.]], requires_grad=True)
        product = make_operator("Multiply", {})(x, y)
        value = make_operator("Scale", {"factor": 0.25})(product)
        torch.testing.assert_close(make_operator("OneMinus", {})(x), torch.tensor([[-1., 2.]]))
        value.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[0.75, 1.]]))
        torch.testing.assert_close(y.grad, torch.tensor([[0.5, -0.25]]))
        self.assertEqual(list(make_operator("Scale", {"factor": 0.25}).parameters()), [])

    def test_same_shape_operators_graph_outputs_and_derivatives_match_formula(self):
        values = torch.tensor([[-2., 0.25, 1.5]])
        for kind, params, formula in (("Identity", {}, lambda x: x),
                                      ("Sigmoid", {}, lambda x: 1 / (1 + torch.exp(-x))),
                                      ("Scale", {"factor": -1.25}, lambda x: -1.25 * x),
                                      ("OneMinus", {}, lambda x: 1 - x)):
            with self.subTest(kind=kind):
                model = runtime.build_model(graph_for(kind, params, [None, 3]))
                x, ref = values.clone().requires_grad_(), values.clone().requires_grad_()
                actual, expected = model(x=x)["out"], formula(ref)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward(); expected.square().sum().backward()
                torch.testing.assert_close(x.grad, ref.grad)
                self.assertEqual(list(model.parameters()), [])

    def test_multiply_runtime_rejects_broadcast_and_graph_matches_hand_gradient(self):
        make_operator = api("operator_runtime").make_operator
        with self.assertRaisesRegex(ValueError, "equal.*shape"):
            make_operator("Multiply", {})(torch.ones(2, 3), torch.ones(2, 1))
        graph = graph_for("Multiply", shape=[None, 3])
        model = runtime.build_model(graph)
        with self.assertRaisesRegex(ValueError, "equal.*shape"):
            model(x=torch.ones(2, 3), y=torch.ones(1, 3))
        x, y = torch.tensor([[1., 2., 3.]], requires_grad=True), torch.tensor([[4., 5., 6.]], requires_grad=True)
        out = model(x=x, y=y)["out"]
        torch.testing.assert_close(out, torch.tensor([[4., 10., 18.]]))
        out.sum().backward()
        torch.testing.assert_close(x.grad, y.detach()); torch.testing.assert_close(y.grad, x.detach())

    def test_each_new_output_is_budget_checked_before_forward(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                model = runtime.build_model(graph_for(kind, {"num_features": 3} if kind == "BatchNorm2d" else {}), capture=False)
                # Inputs fit; the call's additional output would exceed the total.
                model.limits = schema.ResourceLimits(100, 1000, 30 if kind != "Multiply" else 50)
                entered = []
                model.components["operation"].register_forward_pre_hook(lambda *args: entered.append(True))
                inputs = {"x": torch.ones(2, 3, 2, 2)}
                if kind == "Multiply": inputs["y"] = torch.ones(2, 3, 2, 2)
                with self.assertRaisesRegex(ValueError, "cumulative.*budget"):
                    model(**inputs)
                self.assertEqual(entered, [])

    def test_new_operators_preserve_capture_switch_and_device_input_checks(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                model = runtime.build_model(graph_for(kind, {"num_features": 3} if kind == "BatchNorm2d" else {}), capture=False)
                inputs = {"x": torch.ones(2, 3, 2, 2)}
                if kind == "Multiply": inputs["y"] = inputs["x"].clone()
                model(**inputs)
                self.assertEqual(model.last_values, {})
                model.set_capture(True)
                actual = model(**inputs)["out"]
                snapshot = model.last_values["call:apply_op"].clone()
                with torch.no_grad(): actual.add_(10)
                torch.testing.assert_close(model.last_values["call:apply_op"], snapshot)
                wrong = dict(inputs); wrong["x"] = torch.empty(2, 3, 2, 2, device="meta")
                with self.assertRaisesRegex(ValueError, "float32 tensor on cpu"):
                    model(**wrong)


class BatchNormStateTests(unittest.TestCase):
    def test_all_affine_tracking_combinations_match_native_train_eval_gradients_and_step(self):
        data = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4) / 13 - 1
        for affine in (False, True):
            for track in (False, True):
                with self.subTest(affine=affine, track=track):
                    params = {"num_features": 3, "eps": 0.001, "momentum": 0.25,
                              "affine": affine, "track_running_stats": track}
                    model = runtime.build_model(graph_for("BatchNorm2d", params, [None, 3, 2, 4]))
                    module, reference = model.components["operation"], nn.BatchNorm2d(**params)
                    bn_state(reference); module.load_state_dict(reference.state_dict(), strict=True)
                    x, ref_x = data.clone().requires_grad_(), data.clone().requires_grad_()
                    actual, expected = model(x=x)["out"], reference(ref_x)
                    torch.testing.assert_close(actual, expected)
                    actual.square().mean().backward(); expected.square().mean().backward()
                    torch.testing.assert_close(x.grad, ref_x.grad)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 6 if affine else 0)
                    if affine:
                        for name, parameter in module.named_parameters():
                            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
                        torch.optim.SGD(model.parameters(), lr=0.05).step()
                        torch.optim.SGD(reference.parameters(), lr=0.05).step()
                    self.assertEqual(set(module.state_dict()), set(reference.state_dict()))
                    for name, state in module.state_dict().items():
                        torch.testing.assert_close(state, reference.state_dict()[name])
                    model.eval(); reference.eval()
                    eval_data = data * 2 + 1
                    torch.testing.assert_close(model(x=eval_data)["out"], reference(eval_data))
                    if track:
                        self.assertEqual(module.num_batches_tracked.item(), 4)
                        self.assertEqual(set(dict(module.named_buffers())), {"running_mean", "running_var", "num_batches_tracked"})
                    else:
                        self.assertFalse(any("running_" in name or "num_batches" in name for name in model.state_dict()))
                        expected_batch = (eval_data - eval_data.mean((0, 2, 3), keepdim=True)) / torch.sqrt(eval_data.var((0, 2, 3), unbiased=False, keepdim=True) + params["eps"])
                        if affine: expected_batch = expected_batch * module.weight[None, :, None, None] + module.bias[None, :, None, None]
                        torch.testing.assert_close(model(x=eval_data)["out"], expected_batch, rtol=1e-5, atol=2e-6)

    def test_cumulative_momentum_none_and_strict_state_roundtrip_match_reference(self):
        params = {"num_features": 3, "momentum": None}
        graph = graph_for("BatchNorm2d", params)
        model, reference = runtime.build_model(graph), nn.BatchNorm2d(**params)
        bn_state(reference); model.components["operation"].load_state_dict(reference.state_dict())
        for offset in (-1., 2.):
            x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 7 + offset
            torch.testing.assert_close(model(x=x)["out"], reference(x))
        restored = runtime.build_model(graph)
        restored.load_state_dict(model.state_dict(), strict=True)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(restored.components["operation"].state_dict()[key], value)
        missing = dict(model.state_dict()); del missing["components.operation.running_mean"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            restored.load_state_dict(missing, strict=True)
        no_track = runtime.build_model(graph_for("BatchNorm2d", {"num_features": 3, "track_running_stats": False}))
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            no_track.load_state_dict(model.state_dict(), strict=True)

    def test_singleton_fails_exactly_when_batch_statistics_are_required(self):
        for track in (False, True):
            for training in (False, True):
                for batch in (0, 1):
                    with self.subTest(track=track, training=training, batch=batch):
                        graph = graph_for("BatchNorm2d", {"num_features": 3, "track_running_stats": track}, [None, 3, 1, 1])
                        model = runtime.build_model(graph).train(training)
                        if training or not track:
                            with self.assertRaisesRegex(ValueError, "[Bb]atch|[Mm]ore than 1") as caught:
                                model(x=torch.ones(batch, 3, 1, 1))
                            self.assertIn("call:apply_op", str(caught.exception))
                        else:
                            self.assertEqual(list(model(x=torch.ones(batch, 3, 1, 1))["out"].shape), [batch, 3, 1, 1])
                        self.assertEqual(model.training, training)
                        self.assertEqual(model.components["operation"].training, training)

    def test_direct_bn_constructor_enforces_channels_even_without_affine_or_buffers(self):
        make_operator = api("operator_runtime").make_operator
        for affine in (False, True):
            for track in (False, True):
                with self.subTest(affine=affine, track=track):
                    module = make_operator("BatchNorm2d", {"num_features": 3, "affine": affine, "track_running_stats": track})
                    with self.assertRaisesRegex(ValueError, "channels.*num_features"):
                        module(torch.ones(2, 4, 2, 2))

    def test_frozen_bn_stops_affine_gradients_but_updates_train_buffers(self):
        for affine in (False, True):
            for track in (False, True):
                with self.subTest(affine=affine, track=track):
                    params = {"num_features": 3, "affine": affine, "track_running_stats": track}
                    graph = graph_for("BatchNorm2d", params)
                    graph["instances"][0]["frozen"] = True
                    model, reference = runtime.build_model(graph), nn.BatchNorm2d(**params).requires_grad_(False)
                    bn_state(reference); model.components["operation"].load_state_dict(reference.state_dict())
                    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2).requires_grad_()
                    ref_x = x.detach().clone().requires_grad_()
                    actual, expected = model(x=x)["out"], reference(ref_x)
                    actual.square().sum().backward(); expected.square().sum().backward()
                    torch.testing.assert_close(actual, expected); torch.testing.assert_close(x.grad, ref_x.grad)
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                    for key, value in model.components["operation"].state_dict().items():
                        torch.testing.assert_close(value, reference.state_dict()[key])
                    if track: self.assertEqual(model.components["operation"].num_batches_tracked.item(), 4)

    def test_shared_bn_calls_update_one_buffer_twice_and_count_parameters_once(self):
        graph = graph_for("BatchNorm2d", {"num_features": 3})
        graph["calls"].append({"id": "apply_again", "instance_id": "operation", "inputs": {"x": "call:apply_op"}})
        graph["outputs"][0]["source"] = "call:apply_again"
        self.assertEqual(schema.prepare_graph(graph)[2], 6)
        model, reference = runtime.build_model(graph), nn.BatchNorm2d(3)
        bn_state(reference); model.components["operation"].load_state_dict(reference.state_dict())
        x = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 7).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        actual, expected = model(x=x)["out"], reference(reference(ref_x))
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward(); expected.square().mean().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        for name, parameter in model.components["operation"].named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
        torch.optim.SGD(model.parameters(), lr=0.05).step()
        torch.optim.SGD(reference.parameters(), lr=0.05).step()
        self.assertEqual(model.components["operation"].num_batches_tracked.item(), 5)
        for key, value in model.components["operation"].state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key])

    def test_same_definition_distinct_instances_keep_bn_state_independent(self):
        child = graph_for("BatchNorm2d", {"num_features": 3})
        for key in ("schema_version", "revision", "id", "definitions"): child.pop(key)
        graph = graph_for("Identity")
        graph["instances"] = [{"id": key, "kind": "chip", "definition_id": "block", "params": {}} for key in ("left", "right")]
        graph["calls"] = [{"id": key, "instance_id": key, "inputs": {"x": source}} for key, source in (("left", "input:x"), ("right", "call:left"))]
        graph["outputs"][0]["source"] = "call:right"
        graph["definitions"] = [{"id": "block", "public_parameters": [], "bindings": {}, "graph": child}]
        model = runtime.build_model(graph)
        self.assertEqual(schema.prepare_graph(graph)[2], 12)
        left, right = [model.components[key].components["operation"] for key in ("left", "right")]
        self.assertNotEqual(left.running_mean.data_ptr(), right.running_mean.data_ptr())
        self.assertNotEqual(left.weight.data_ptr(), right.weight.data_ptr())
        first, second = nn.BatchNorm2d(3), nn.BatchNorm2d(3)
        first.load_state_dict(left.state_dict()); second.load_state_dict(right.state_dict())
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        actual, expected = model(x=x)["out"], second(first(ref_x))
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward(); expected.square().mean().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        for module, reference in ((left, first), (right, second)):
            for name, parameter in module.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
        torch.optim.SGD(model.parameters(), lr=0.05).step()
        torch.optim.SGD(list(first.parameters()) + list(second.parameters()), lr=0.05).step()
        for actual, reference in ((left, first), (right, second)):
            self.assertEqual(actual.num_batches_tracked.item(), 1)
            for key, value in actual.state_dict().items(): torch.testing.assert_close(value, reference.state_dict()[key])
        self.assertFalse(torch.equal(left.running_mean, right.running_mean))


if __name__ == "__main__":
    unittest.main()
