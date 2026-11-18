"""Independent CNN arithmetic, execution and static-source bridge regressions."""
import unittest
from unittest.mock import patch

import torch

from studio_core import python_io, runtime, schema


def cnn_graph():
    kinds = [("conv", "Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": [3, 2], "stride": [2, 1], "padding": [1, 0], "groups": 2, "bias": False}),
             ("pool", "MaxPool2d", {"kernel_size": [2, 3], "stride": [2, 2], "ceil_mode": True}),
             ("average", "AdaptiveAvgPool2d", {"output_size": [1, 2]}),
             ("flatten", "Flatten", {}),
             ("head", "Linear", {"in_features": 8, "out_features": 3})]
    calls, source = [], "input:x"
    for identifier, kind, params in kinds:
        calls.append({"id": identifier + "_call", "instance_id": identifier, "inputs": {"x": source}})
        source = "call:" + identifier + "_call"
    return {"schema_version": 1, "id": "cnn", "revision": 1, "resource_profile": "classification-v2",
            "inputs": [{"id": "x", "shape": [None, 2, 9, 10], "dtype": "float32"}],
            "instances": [{"id": i, "kind": k, "params": p} for i, k, p in kinds],
            "calls": calls, "outputs": [{"id": "logits", "source": source}], "definitions": []}


def one_op(kind, params, shape):
    graph = cnn_graph()
    graph["inputs"][0]["shape"] = shape
    graph["instances"] = [{"id": "operator", "kind": kind, "params": params}]
    graph["calls"] = [{"id": "apply_op", "instance_id": "operator", "inputs": {"x": "input:x"}}]
    graph["outputs"] = [{"id": "out", "source": "call:apply_op"}]
    return graph


class CnnSchemaTests(unittest.TestCase):
    def test_channel_group_dilation_pool_ceil_and_flatten_shapes_are_exact(self):
        report = schema.validate_graph(cnn_graph())
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["shapes"]["call:conv_call"], [None, 4, 5, 9])
        self.assertEqual(report["shapes"]["call:pool_call"], [None, 4, 3, 4])
        self.assertEqual(report["shapes"]["call:average_call"], [None, 4, 1, 2])
        self.assertEqual(report["shapes"]["call:flatten_call"], [None, 8])
        self.assertEqual(report["parameter_count"], 51)
        for kind, params, shape, expected in (
            ("Conv2d", {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "dilation": 2}, [None, 3, 11, 12], [None, 6, 7, 8]),
            ("MaxPool2d", {"kernel_size": 2, "stride": 3, "padding": 1, "ceil_mode": True}, [None, 2, 3, 4], [None, 2, 2, 2]),
            ("AdaptiveAvgPool2d", {"output_size": [None, 2]}, [None, 3, 7, 8], [None, 3, 7, 2]),
            ("Flatten", {"start_dim": -3, "end_dim": -2}, [None, 2, 3, 4], [None, 6, 4]),
        ):
            with self.subTest(kind=kind):
                result = schema.validate_graph(one_op(kind, params, shape))
                self.assertTrue(result["valid"], result["errors"])
                self.assertEqual(result["shapes"]["call:apply_op"], expected)

    def test_cnn_parameters_rank_channels_and_empty_outputs_are_strict(self):
        invalid = [
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "groups": 3}, [None, 2, 8, 8]),
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": 3}, [None, 3, 8, 8]),
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": 9}, [None, 2, 8, 8]),
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": True}, [None, 2, 8, 8]),
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "bias": 1}, [None, 2, 8, 8]),
            ("Conv2d", {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "padding_mode": "reflect"}, [None, 2, 8, 8]),
            ("MaxPool2d", {"kernel_size": 2, "return_indices": True}, [None, 2, 8, 8]),
            ("MaxPool2d", {"kernel_size": 2, "padding": 2}, [None, 2, 8, 8]),
            ("MaxPool2d", {"kernel_size": 2, "stride": 0}, [None, 2, 8, 8]),
            ("MaxPool2d", {"kernel_size": 2}, [None, 8]),
            ("AdaptiveAvgPool2d", {"output_size": [0, 1]}, [None, 2, 8, 8]),
            ("Flatten", {"start_dim": 3, "end_dim": 1}, [None, 2, 8, 8]),
            ("Flatten", {"unknown": 1}, [None, 2, 8, 8]),
        ]
        for kind, params, shape in invalid:
            with self.subTest(kind=kind, params=params):
                self.assertFalse(schema.validate_graph(one_op(kind, params, shape))["valid"])

    def test_explicit_profile_is_larger_but_default_s1_and_parameter_limits_remain_bounded(self):
        graph = one_op("ReLU", {}, [None, 3, 224, 224])
        self.assertTrue(schema.validate_graph(graph)["valid"])
        del graph["resource_profile"]
        self.assertFalse(schema.validate_graph(graph)["valid"])
        graph["resource_profile"] = "unbounded"
        self.assertFalse(schema.validate_graph(graph)["valid"])
        graph = one_op("Conv2d", {"in_channels": 1024, "out_channels": 2048, "kernel_size": 3}, [None, 1024, 3, 3])
        self.assertTrue(schema.validate_graph(graph)["valid"])
        graph["instances"][0]["params"]["out_channels"] = 2200
        with patch.object(torch.nn, "Conv2d", side_effect=AssertionError("weight allocation must not run")):
            with self.assertRaisesRegex(ValueError, "budget"):
                runtime.build_model(graph)

    def test_root_resource_profile_reaches_nested_definitions_without_changing_s1_globals(self):
        child = one_op("ReLU", {}, [None, 3, 224, 224])
        for key in ("schema_version", "id", "revision", "resource_profile", "definitions"):
            child.pop(key, None)
        graph = one_op("ReLU", {}, [None, 3, 224, 224])
        graph["instances"] = [{"id": "nested", "kind": "chip", "definition_id": "image_chip", "params": {}}]
        graph["calls"][0]["instance_id"] = "nested"
        graph["definitions"] = [{"id": "image_chip", "public_parameters": [], "bindings": {}, "graph": child}]
        self.assertTrue(schema.validate_graph(graph)["valid"])
        model = runtime.build_model(graph, capture=False)
        self.assertEqual(list(model(x=torch.zeros(1, 3, 224, 224))["out"].shape), [1, 3, 224, 224])
        graph.pop("resource_profile")
        self.assertFalse(schema.validate_graph(graph)["valid"])


class CnnRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_gradients_and_optimizer_update_match_independent_handwritten_cnn(self):
        model = runtime.build_model(cnn_graph(), device="cpu", capture=False)
        reference = torch.nn.Sequential(torch.nn.Conv2d(2, 4, (3, 2), (2, 1), (1, 0), groups=2, bias=False),
                                        torch.nn.MaxPool2d((2, 3), (2, 2), ceil_mode=True),
                                        torch.nn.AdaptiveAvgPool2d((1, 2)), torch.nn.Flatten(), torch.nn.Linear(8, 3))
        reference[0].load_state_dict(model.components["conv"].state_dict())
        reference[4].load_state_dict(model.components["head"].state_dict())
        x = torch.linspace(-1.2, 1.9, 540).reshape(3, 2, 9, 10).requires_grad_(True)
        reference_x = x.detach().clone().requires_grad_(True)
        actual, expected = model(x=x)["logits"], reference(reference_x)
        torch.testing.assert_close(actual, expected)
        labels = torch.tensor([0, 2, 1])
        torch.nn.functional.cross_entropy(actual, labels).backward()
        torch.nn.functional.cross_entropy(expected, labels).backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        torch.optim.SGD(model.parameters(), lr=0.03).step()
        torch.optim.SGD(reference.parameters(), lr=0.03).step()
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right)
        self.assertEqual(model.last_values, {})
        model.set_capture(True)
        model(x=x)
        self.assertEqual(model.last_values["call:conv_call"].shape, (3, 4, 5, 9))

    def test_capture_flag_is_recursive_and_preserves_nested_shared_cnn_gradients(self):
        graph = cnn_graph()
        child = one_op("Conv2d", {"in_channels": 2, "out_channels": 2, "kernel_size": 1}, [None, 2, 9, 10])
        for key in ("schema_version", "id", "revision", "resource_profile", "definitions"):
            child.pop(key, None)
        graph["definitions"] = [{"id": "encoder", "public_parameters": [], "bindings": {}, "graph": child}]
        graph["instances"] = [{"id": "shared", "kind": "chip", "definition_id": "encoder", "params": {}}, {"id": "sum_op", "kind": "Add", "params": {}}]
        graph["calls"] = [{"id": "left", "instance_id": "shared", "inputs": {"x": "input:x"}}, {"id": "right", "instance_id": "shared", "inputs": {"x": "input:x"}}, {"id": "sum_call", "instance_id": "sum_op", "inputs": {"a": "call:left", "b": "call:right"}}]
        graph["outputs"] = [{"id": "out", "source": "call:sum_call"}]
        model = runtime.build_model(graph)
        x = torch.ones(2, 2, 9, 10)
        model(x=x)["out"].sum().backward()
        self.assertTrue(model.components["shared"].last_values)
        self.assertEqual(model.components["shared"].components["operator"].weight.grad[0, 0, 0, 0].item(), 360)
        model.set_capture(False)
        model(x=x)
        self.assertEqual(model.last_values, {})
        self.assertEqual(model.components["shared"].last_values, {})
        self.assertEqual(len(list(model.parameters())), 2)

    def test_capture_disabled_still_blocks_dynamic_conv_before_allocation(self):
        graph = one_op("Conv2d", {"in_channels": 1, "out_channels": 32, "kernel_size": 1}, [None, 1, 8, 8])
        model = runtime.build_model(graph, capture=False)
        with patch.object(model.components["operator"], "forward", side_effect=AssertionError("output allocation must not run")):
            with self.assertRaisesRegex(ValueError, "budget"):
                model(x=torch.zeros(9000, 1, 8, 8))

    def test_capture_disabled_still_blocks_dynamic_adaptive_pool_before_allocation(self):
        graph = one_op("AdaptiveAvgPool2d", {"output_size": [512, 512]}, [None, 1, 1, 1])
        model = runtime.build_model(graph, capture=False)
        with patch.object(model.components["operator"], "forward", side_effect=AssertionError("pool allocation must not run")):
            with self.assertRaisesRegex(ValueError, "budget"):
                model(x=torch.zeros(65, 1, 1, 1))

    def test_nested_cumulative_runtime_budget_remains_enabled_without_capture(self):
        child = {"inputs": [{"id": "x", "shape": [None, 1], "dtype": "float32"}],
                 "instances": [{"id": "activation", "kind": "ReLU", "params": {}}],
                 "calls": [{"id": "activate", "instance_id": "activation", "inputs": {"x": "input:x"}}],
                 "outputs": [{"id": "out", "source": "call:activate"}]}
        graph = {"schema_version": 1, "id": "nested_budget", "revision": 1,
                 "inputs": child["inputs"], "instances": [{"id": "shared", "kind": "chip", "definition_id": "block", "params": {}}],
                 "calls": [{"id": "call_" + str(i), "instance_id": "shared", "inputs": {"x": "input:x"}} for i in range(7)],
                 "outputs": [{"id": "out", "source": "call:call_6"}],
                 "definitions": [{"id": "block", "public_parameters": [], "bindings": {}, "graph": child}]}
        model = runtime.build_model(graph, capture=False)
        with self.assertRaisesRegex(ValueError, "cumulative.*budget"):
            model(x=torch.zeros(800_000, 1))
        self.assertEqual(model.last_values, {})
        self.assertEqual(model.components["shared"].last_values, {})


class CnnBridgeTests(unittest.TestCase):
    def test_versioned_cnn_source_tuple_list_and_metadata_deletion_round_trip(self):
        source = python_io.export_python(cnn_graph())
        self.assertIn("classification-pytorch-v2", source)
        edited = source.replace("kernel_size=[3, 2]", "kernel_size=(3, 2)").replace("stride=[2, 1]", "stride=(2, 1)")
        edited = "\n".join(line for line in edited.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(edited)
        self.assertEqual(restored["resource_profile"], "classification-v2")
        self.assertEqual(restored["instances"][0]["params"]["kernel_size"], [3, 2])
        self.assertTrue(schema.validate_graph(restored)["valid"])

    def test_current_constructor_edits_override_visual_metadata_and_export_state_keys_match(self):
        graph = cnn_graph()
        source = python_io.export_python(graph).replace("bias=False", "bias=True", 1)
        restored = python_io.import_python(source)
        self.assertTrue(restored["instances"][0]["params"]["bias"])
        self.assertEqual(schema.validate_graph(restored)["parameter_count"], 55)
        namespace = {}
        exec(compile(source, "<trusted-test-export>", "exec"), namespace)
        model = runtime.build_model(restored)
        standalone = namespace["Model"]()
        self.assertEqual(set(model.state_dict()), set(standalone.state_dict()))
        standalone.load_state_dict(model.state_dict(), strict=True)
        x = torch.linspace(0, 1, 360).reshape(2, 2, 9, 10)
        torch.testing.assert_close(model(x=x)["logits"], standalone(x=x)["logits"])
        self.assertFalse(any("anchor" in key for key in model.state_dict()))

    def test_unsupported_pool_tuple_output_and_arbitrary_source_remain_rejected(self):
        source = python_io.export_python(cnn_graph())
        tuple_source = source.replace("ceil_mode=True", "ceil_mode=True, return_indices=True", 1)
        self.assertNotEqual(tuple_source, source)
        for modified in (tuple_source, source + "\nraise RuntimeError('executed')\n"):
            with self.assertRaises(ValueError):
                python_io.import_python(modified)

    def test_nested_shared_cnn_public_tuple_binding_survives_source_without_metadata(self):
        graph = cnn_graph()
        child = {key: graph[key] for key in ("inputs", "instances", "calls", "outputs")}
        graph = {"schema_version": 1, "id": "shared_cnn", "revision": 1, "resource_profile": "classification-v2",
                 "inputs": child["inputs"],
                 "instances": [{"id": "encoder", "kind": "chip", "definition_id": "image_block", "params": {"kernel": [3, 2]}},
                               {"id": "sum_op", "kind": "Add", "params": {}}],
                 "calls": [{"id": "left", "instance_id": "encoder", "inputs": {"x": "input:x"}},
                           {"id": "right", "instance_id": "encoder", "inputs": {"x": "input:x"}},
                           {"id": "sum_call", "instance_id": "sum_op", "inputs": {"a": "call:left", "b": "call:right"}}],
                 "outputs": [{"id": "logits", "source": "call:sum_call"}],
                 "definitions": [{"id": "image_block", "public_parameters": [{"id": "kernel", "default": [3, 2]}],
                                  "bindings": {"kernel": ["conv.kernel_size"]}, "graph": child}]}
        source = python_io.export_python(graph).replace("kernel=[3, 2]", "kernel=(3, 2)")
        source = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(source)
        self.assertEqual(restored["definitions"][0]["public_parameters"][0]["default"], [3, 2])
        self.assertEqual(restored["definitions"][0]["bindings"], {"kernel": ["conv.kernel_size"]})
        self.assertEqual(schema.validate_graph(restored)["parameter_count"], 51)
        model = runtime.build_model(restored, capture=False)
        namespace = {}
        exec(compile(source, "<trusted-nested-test-export>", "exec"), namespace)
        standalone = namespace["Model"]()
        standalone.load_state_dict(model.state_dict(), strict=True)
        x = torch.linspace(-1, 1, 360).reshape(2, 2, 9, 10)
        actual, expected = model(x=x)["logits"], standalone(x=x)["logits"]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        for left, right in zip(model.parameters(), standalone.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        self.assertEqual(len(runtime._shared_instances(model)), 1)


if __name__ == "__main__":
    unittest.main()
