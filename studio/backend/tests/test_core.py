"""Behavioral tests for S1 graph validation and real CPU execution."""
import copy
import importlib
import json
import math
import unittest


def core(name):
    try:
        return importlib.import_module("studio_core." + name)
    except ModuleNotFoundError as exc:
        raise AssertionError("S1 core module is not implemented: " + name) from exc


class SchemaTests(unittest.TestCase):
    def example(self):
        return core("examples").example_graph()

    def test_example_is_fresh_and_has_exact_shared_shapes_and_count(self):
        graph = self.example()
        result = core("schema").validate_graph(graph)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["parameter_count"], 58)
        self.assertEqual(result["shapes"]["call:fuse"], [None, 8])
        graph["instances"][0]["params"]["hidden_features"] = 99
        self.assertEqual(self.example()["instances"][0]["params"]["hidden_features"], 8)

    def test_invalid_connections_cycles_shapes_parameters_and_recursive_chips(self):
        invalid = []
        graph = self.example(); graph["calls"][0]["inputs"]["x"] = "call:absent"; invalid.append(graph)
        graph = self.example(); graph["calls"][0]["inputs"]["x"] = "call:classify"; invalid.append(graph)
        graph = self.example(); graph["inputs"][1]["shape"] = [None, 5]; invalid.append(graph)
        graph = self.example(); graph["instances"][2]["params"]["out_features"] = 0; invalid.append(graph)
        graph = self.example(); graph["calls"][0]["inputs"]["wrong"] = graph["calls"][0]["inputs"].pop("x"); invalid.append(graph)
        graph = self.example(); graph["instances"][0]["params"]["typo"] = 2; invalid.append(graph)
        graph = self.example(); graph["definitions"][0]["graph"]["instances"][0] = copy.deepcopy(graph["instances"][0]); invalid.append(graph)
        graph = self.example(); graph["definitions"][0]["bindings"]["hidden_features"] = ["missing.out_features"]; invalid.append(graph)
        for graph in invalid:
            with self.subTest(graph=graph):
                report = core("schema").validate_graph(graph)
                self.assertFalse(report["valid"])
                self.assertTrue(report["errors"])
                self.assertIsNone(report["parameter_count"])

    def test_public_binding_changes_copy_and_nested_chip_shapes(self):
        graph = self.example()
        graph["instances"][0]["params"] = {"in_features": 6, "hidden_features": 10}
        for port in graph["inputs"]: port["shape"][-1] = 6
        graph["instances"][2]["params"]["in_features"] = 10
        inner = copy.deepcopy(graph["definitions"][0])
        inner["id"] = "inner"
        outer = graph["definitions"][0]
        outer["graph"]["instances"][0] = {"id": "dense", "kind": "chip", "definition_id": "inner", "params": {"in_features": 4, "hidden_features": 8}}
        outer["bindings"] = {"in_features": ["dense.in_features"], "hidden_features": ["dense.hidden_features"]}
        graph["definitions"].append(inner)
        before = copy.deepcopy(graph)
        report = core("schema").validate_graph(graph)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["shapes"]["call:encode_a"], [None, 10])
        self.assertEqual(report["parameter_count"], 92)
        self.assertEqual(graph, before)

    def test_validator_handles_malformed_json_values_without_crashing(self):
        for graph in (None, [], {}, {"inputs": None}, {"schema_version": 1, "instances": [None]}):
            with self.subTest(graph=graph):
                report = core("schema").validate_graph(graph)
                self.assertFalse(report["valid"])
                self.assertTrue(report["errors"])

    def test_concat_normalization_and_dropout_have_explicit_shape_rules(self):
        graph = self.example()
        graph["instances"][1] = {"id": "fusion", "kind": "Concat", "params": {"dim": -1}}
        graph["instances"][2] = {"id": "head", "kind": "LayerNorm", "params": {"normalized_shape": [16]}}
        report = core("schema").validate_graph(graph)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["shapes"]["call:classify"], [None, 16])
        self.assertEqual(report["parameter_count"], 72)
        for kind, params in (("Concat", {"dim": 5}), ("LayerNorm", {"normalized_shape": 7}), ("Dropout", {"p": float("nan")})):
            graph["instances"][2] = {"id": "head", "kind": kind, "params": params}
            report = core("schema").validate_graph(graph)
            self.assertFalse(report["valid"])

    def test_module_attribute_identifiers_are_rejected_before_pytorch_construction(self):
        for identifier in ("__class__", "_modules", "apply", "named_parameters"):
            graph = self.example()
            graph["instances"][0]["id"] = identifier
            for call in graph["calls"]:
                if call["instance_id"] == "encoder": call["instance_id"] = identifier
            report = core("schema").validate_graph(graph)
            self.assertFalse(report["valid"], identifier)
            self.assertEqual(report["errors"][0]["field"], "id")

    def test_malformed_required_fields_return_diagnostics_instead_of_python_exceptions(self):
        replacements = (None, True, 3, [], {}, "")
        paths = (
            ("inputs",), ("instances",), ("calls",), ("outputs",), ("definitions",),
            ("inputs", 0, "shape"), ("inputs", 0, "dtype"),
            ("instances", 0, "kind"), ("instances", 0, "params"),
            ("calls", 0, "instance_id"), ("calls", 0, "inputs"),
            ("calls", 0, "inputs", "x"), ("outputs", 0, "source"),
            ("definitions", 0, "graph"), ("definitions", 0, "bindings"),
        )
        for path in paths:
            for replacement in replacements:
                graph = self.example()
                target = graph
                for key in path[:-1]: target = target[key]
                target[path[-1]] = replacement
                with self.subTest(path=path, replacement=replacement):
                    report = core("schema").validate_graph(graph)
                    self.assertIn("valid", report)
                    if not report["valid"]: self.assertTrue(report["errors"])

    def test_resource_limits_reject_huge_models_documents_and_chip_expansion(self):
        graphs = []
        graph = self.example()
        graph["instances"][2]["params"] = {"in_features": 2048, "out_features": 2048}
        graph["instances"][0]["params"]["hidden_features"] = 2048
        graphs.append(graph)
        graph = self.example(); graph["inputs"][0]["shape"] = [None, 4096, 4096]; graphs.append(graph)
        graph = self.example(); graph["name"] = "x" * 4097; graphs.append(graph)
        graph = self.example(); graph["calls"] = [{"id": "node_" + str(i), "instance_id": "head", "inputs": {"x": "input:a"}} for i in range(600)]; graphs.append(graph)
        for expansion in (False, True):
            graph = {"schema_version": 1, "id": "nested", "revision": 1,
                     "inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                     "instances": [{"id": "chip", "kind": "chip", "definition_id": "level_0", "params": {}}],
                     "calls": [{"id": "run", "instance_id": "chip", "inputs": {"x": "input:x"}}],
                     "outputs": [{"id": "out", "source": "call:run"}], "definitions": []}
            for depth in range(6 if expansion else 8):
                is_last = depth == (5 if expansion else 7)
                instances = [{"id": "child_" + str(i), "kind": "ReLU" if is_last else "chip", "params": {}}
                             for i in range(4 if expansion else 1)]
                if not is_last:
                    for instance in instances: instance["definition_id"] = "level_" + str(depth + 1)
                nested = {"inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}], "instances": instances,
                          "calls": [{"id": "run_" + str(i), "instance_id": instance["id"], "inputs": {"x": "input:x"}}
                                    for i, instance in enumerate(instances)],
                          "outputs": [{"id": "out", "source": "call:run_0"}]}
                graph["definitions"].append({"id": "level_" + str(depth), "public_parameters": [], "bindings": {}, "graph": nested})
            graphs.append(graph)
        for index, graph in enumerate(graphs):
            with self.subTest(case=index, definitions=len(graph["definitions"])):
                result = core("schema").validate_graph(graph)
                self.assertFalse(result["valid"])
                self.assertRegex(result["errors"][0]["message"].lower(), "limit|budget|maximum")

    def test_extreme_numeric_values_are_diagnostics_instead_of_overflow_exceptions(self):
        graph = self.example()
        graph["instances"][2] = {"id": "head", "kind": "Dropout", "params": {"p": 10**400}}
        self.assertFalse(core("schema").validate_graph(graph)["valid"])
        graph = self.example()
        graph["calls"][0]["position"] = {"x": 10**400, "y": 0}
        self.assertFalse(core("schema").validate_graph(graph)["valid"])

    def test_nonfinite_numbers_anywhere_in_document_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for location in ("name", "nested_metadata", "definition_metadata"):
                graph = self.example()
                if location == "name": graph["name"] = value
                elif location == "nested_metadata": graph["metadata"] = {"nested": [1, {"value": value}]}
                else: graph["definitions"][0]["metadata"] = [value]
                with self.subTest(value=value, location=location):
                    report = core("schema").validate_graph(graph)
                    self.assertFalse(report["valid"])
                    self.assertRegex(report["errors"][0]["message"].lower(), "finite")


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is not installed yet")
        cls.torch = torch
        torch.set_num_threads(1)

    def test_shared_forward_backward_update_matches_handwritten_reference(self):
        torch = self.torch
        graph = core("examples").example_graph()
        model = core("runtime").build_model(graph)
        encoder = torch.nn.Linear(4, 8)
        head = torch.nn.Linear(8, 2)
        encoder.load_state_dict(model.components["encoder"].components["dense"].state_dict())
        head.load_state_dict(model.components["head"].state_dict())
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        b = torch.flip(a, (0,)) - 0.7
        actual = model(a=a, b=b)
        features = torch.relu(encoder(a)) + torch.relu(encoder(b))
        reference = head(features)
        torch.testing.assert_close(actual["features"], features)
        torch.testing.assert_close(actual["logits"], reference)
        labels = torch.tensor([0, 1, 0])
        loss = torch.nn.functional.cross_entropy(actual["logits"], labels)
        ref_loss = torch.nn.functional.cross_entropy(reference, labels)
        loss.backward(); ref_loss.backward()
        torch.testing.assert_close(model.components["encoder"].components["dense"].weight.grad, encoder.weight.grad)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.1).step()
        torch.testing.assert_close(model.components["encoder"].components["dense"].weight, encoder.weight)
        self.assertEqual(len(list(model.parameters())), 4)
        self.assertIs(model.components["encoder"], model.components["encoder"])
        with self.assertRaises(ValueError): model(a=a, b=torch.ones(3, 5))

    def test_real_experiment_reports_measurements_and_is_reproducible(self):
        graph = core("examples").example_graph()
        events = []
        config = {"steps": 8, "lr": 0.1, "seed": 7, "batch_size": 12}
        result = core("runtime").run_experiment(graph, config, events.append)
        again = core("runtime").run_experiment(graph, config)
        self.assertEqual(result, again)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["data_source"], "synthetic_s1")
        self.assertEqual(len(result["history"]), 8)
        self.assertEqual([e["step"] for e in events], list(range(1, 9)))
        self.assertEqual(result["parameter_count"], 58)
        self.assertEqual(result["shared_instances"], [{"instance_id": "encoder", "call_ids": ["encode_a", "encode_b"]}])
        self.assertLess(result["loss_after"], result["loss_before"])
        self.assertTrue(any(p["grad_norm"] > 0 and p["update_norm"] > 0 for p in result["parameters"]))
        probe = result["probes"]["call:fuse"]
        self.assertEqual(probe["shape"], [12, 8])
        self.assertEqual(probe["source"], "measured")
        self.assertEqual(probe["sample_size"], 96)
        values = self.torch.tensor(probe["values"])
        self.assertAlmostEqual(values.mean().item(), probe["mean"], places=6)
        self.assertEqual(set(result["outputs"]), {"logits", "features"})

    def test_frozen_parameters_cancellation_and_invalid_training_contract(self):
        graph = core("examples").example_graph()
        graph["instances"][0]["frozen"] = True
        result = core("runtime").run_experiment(graph, {"steps": 2})
        frozen = [p for p in result["parameters"] if "encoder" in p["name"]]
        self.assertTrue(frozen)
        self.assertTrue(all(not p["trainable"] and p["update_norm"] == 0 and p["grad_norm"] in (None, 0) for p in frozen))
        with self.assertRaises(InterruptedError): core("runtime").run_experiment(graph, {}, cancelled=lambda: True)
        for config in ({"steps": 0}, {"lr": 5}, {"lr": 10**400}, {"seed": -1}, {"batch_size": True}, {"unknown": 2}):
            with self.assertRaises(ValueError): core("runtime").run_experiment(graph, config)
        graph["instances"][2]["params"]["out_features"] = 3
        with self.assertRaises(ValueError): core("runtime").run_experiment(graph, {})

    def test_nested_chip_runtime_uses_bound_dimensions_and_named_outputs(self):
        graph = core("examples").example_graph()
        original = copy.deepcopy(graph["definitions"][0])
        original["id"] = "inner"
        outer = graph["definitions"][0]
        outer["graph"]["instances"][0] = {"id": "dense", "kind": "chip", "definition_id": "inner", "params": {"in_features": 4, "hidden_features": 8}}
        outer["bindings"] = {"in_features": ["dense.in_features"], "hidden_features": ["dense.hidden_features"]}
        graph["definitions"].append(original)
        graph["instances"][0]["params"]["hidden_features"] = 5
        graph["instances"][2]["params"]["in_features"] = 5
        model = core("runtime").build_model(graph)
        torch = self.torch
        a, b = torch.ones(2, 4), torch.zeros(2, 4)
        result = model(a=a, b=b)
        projection = model.components["encoder"].components["dense"].components["dense"]
        self.assertEqual(list(projection.weight.shape), [5, 4])
        expected = torch.relu(projection(a)) + torch.relu(projection(b))
        torch.testing.assert_close(result["features"], expected)
        self.assertEqual(set(result), {"logits", "features"})

    def test_probe_sampling_reports_full_count_and_bounded_values(self):
        graph = core("examples").example_graph()
        result = core("runtime").run_experiment(graph, {"steps": 1, "batch_size": 64})
        probe = result["probes"]["call:fuse"]
        self.assertEqual(probe["count"], 512)
        self.assertEqual(probe["sample_size"], 256)
        self.assertEqual(len(probe["values"]), 256)
        self.assertEqual(probe["finite_count"], 512)

    def test_training_rejects_logits_disconnected_from_trainable_parameters(self):
        graph = core("examples").example_graph()
        graph["inputs"][0]["shape"] = [None, 2]
        graph["calls"] = []
        graph["outputs"] = [{"id": "logits", "source": "input:a"}]
        with self.assertRaisesRegex(ValueError, "trainable"):
            core("runtime").run_experiment(graph, {"steps": 1})

    def test_unbatched_concat_is_valid_but_dynamic_add_never_broadcasts(self):
        graph = {"schema_version": 1, "id": "vectors", "revision": 1,
                 "inputs": [{"id": "a", "shape": [4], "dtype": "float32"}, {"id": "b", "shape": [5], "dtype": "float32"}],
                 "instances": [{"id": "join", "kind": "Concat", "params": {"dim": 0}}],
                 "calls": [{"id": "joined", "instance_id": "join", "inputs": {"a": "input:a", "b": "input:b"}}],
                 "outputs": [{"id": "vector", "source": "call:joined"}], "definitions": []}
        model = core("runtime").build_model(graph)
        result = model(a=self.torch.ones(4), b=self.torch.zeros(5))["vector"]
        self.assertEqual(list(result.shape), [9])
        graph["inputs"] = [{"id": "a", "shape": [None, 4], "dtype": "float32"}, {"id": "b", "shape": [None, 4], "dtype": "float32"}]
        graph["instances"] = [{"id": "join", "kind": "Add", "params": {}}]
        model = core("runtime").build_model(graph)
        with self.assertRaises(ValueError): model(a=self.torch.ones(1, 4), b=self.torch.ones(3, 4))

    def test_oversized_model_is_rejected_before_torch_allocates_weights(self):
        from unittest.mock import patch
        graph = core("examples").example_graph()
        graph["instances"][0]["params"]["hidden_features"] = 2048
        graph["instances"][2]["params"] = {"in_features": 2048, "out_features": 2048}
        with patch.object(self.torch.nn, "Linear", side_effect=AssertionError("weight allocation must not run")):
            with self.assertRaises(ValueError): core("runtime").build_model(graph)

    def test_dynamic_concat_expansion_is_bounded_before_large_linear_output_allocation(self):
        from unittest.mock import patch
        calls, previous = [], "input:x"
        for index in range(9):
            calls.append({"id": "double_" + str(index), "instance_id": "duplicate", "inputs": {"a": previous, "b": previous}})
            previous = "call:double_" + str(index)
        calls.append({"id": "project", "instance_id": "projection", "inputs": {"x": previous}})
        graph = {"schema_version": 1, "id": "expanding", "revision": 1,
                 "inputs": [{"id": "x", "shape": [None, 1], "dtype": "float32"}],
                 "instances": [{"id": "duplicate", "kind": "Concat", "params": {"dim": 0}},
                               {"id": "projection", "kind": "Linear", "params": {"in_features": 1, "out_features": 1000}}],
                 "calls": calls, "outputs": [{"id": "out", "source": "call:project"}], "definitions": []}
        model = core("runtime").build_model(graph)
        with patch.object(model.components["projection"], "forward", side_effect=AssertionError("large output allocation must not run")):
            with self.assertRaisesRegex(ValueError, "budget"):
                model(x=self.torch.ones(64, 1))

    def test_dropout_calls_preserve_declared_effect_order(self):
        torch = self.torch
        graph = {"schema_version": 1, "id": "dropout_order", "revision": 1,
                 "inputs": [{"id": name, "shape": [None, 4], "dtype": "float32"} for name in ("a", "b")],
                 "instances": [{"id": "drop", "kind": "Dropout", "params": {"p": 0.5}}],
                 "calls": [{"id": "first", "instance_id": "drop", "inputs": {"x": "input:a"}},
                           {"id": "second", "instance_id": "drop", "inputs": {"x": "call:first"}},
                           {"id": "third", "instance_id": "drop", "inputs": {"x": "input:b"}}],
                 "outputs": [{"id": name, "source": "call:" + name} for name in ("first", "second", "third")], "definitions": []}
        model = core("runtime").build_model(graph)
        a = torch.arange(1, 257, dtype=torch.float32).reshape(64, 4)
        b = a + 300
        torch.manual_seed(77)
        first = torch.nn.functional.dropout(a, p=0.5, training=True)
        second = torch.nn.functional.dropout(first, p=0.5, training=True)
        third = torch.nn.functional.dropout(b, p=0.5, training=True)
        torch.manual_seed(77)
        actual = model(a=a, b=b)
        for name, expected in (("first", first), ("second", second), ("third", third)):
            torch.testing.assert_close(actual[name], expected)

    def test_experiment_rejects_nonfinite_extra_output_with_its_source(self):
        graph = core("examples").example_graph()
        graph["instances"].append({"id": "amplify", "kind": "Add", "params": {}})
        previous = "input:a"
        for index in range(140):
            graph["calls"].append({"id": "amplified_" + str(index), "instance_id": "amplify", "inputs": {"a": previous, "b": previous}})
            previous = "call:amplified_" + str(index)
        graph["outputs"].append({"id": "overflow", "source": previous})
        with self.assertRaisesRegex(ValueError, r"[Nn]on.?finite.*overflow.*call:amplified_139"):
            core("runtime").run_experiment(graph, {"steps": 1, "batch_size": 2})

    def test_many_output_aliases_have_honest_bounded_measured_previews(self):
        graph = core("examples").example_graph()
        for port in graph["inputs"]: port["shape"][-1] = 1000
        graph["instances"][0]["params"]["in_features"] = 1000
        graph["outputs"] = [{"id": "logits", "source": "call:classify"}] + [
            {"id": "alias_" + str(index), "source": "input:a"} for index in range(511)]
        config = {"steps": 1, "batch_size": 2, "seed": 11}
        result = core("runtime").run_experiment(graph, config)

        def scalars(value):
            return sum(scalars(child) for child in value) if isinstance(value, list) else 1

        preview_counts = {key: scalars(value) for key, value in result["outputs"].items()}
        self.assertLessEqual(sum(preview_counts.values()), 4096)
        self.assertTrue(all(count <= 256 for count in preview_counts.values()))
        generator = self.torch.Generator().manual_seed(11)
        expected = self.torch.randn(2, 1000, generator=generator).reshape(-1)
        for key, value in result["outputs"].items():
            metadata = result["output_metadata"][key]
            self.assertEqual(metadata["preview_count"], preview_counts[key])
            self.assertEqual(metadata["source"], "measured")
            if key.startswith("alias_"):
                self.assertEqual(metadata["shape"], [2, 1000])
                self.assertEqual(metadata["count"], 2000)
                self.assertTrue(metadata["truncated"])
                self.assertEqual(metadata["tensor_source"], "input:a")
                self.torch.testing.assert_close(self.torch.tensor(value).reshape(-1), expected[:metadata["preview_count"]])
        json.dumps(result, allow_nan=False)
        model = core("runtime").build_model(graph)
        inputs = {"a": self.torch.ones(2, 1000), "b": self.torch.zeros(2, 1000)}
        full = model(**inputs)
        self.assertEqual(list(full["alias_510"].shape), [2, 1000])

    def test_large_finite_probe_statistics_remain_strict_json(self):
        graph = core("examples").example_graph()
        graph["instances"].append({"id": "amplify", "kind": "Add", "params": {}})
        previous = "input:a"
        for index in range(120):
            graph["calls"].append({"id": "amplified_" + str(index), "instance_id": "amplify", "inputs": {"a": previous, "b": previous}})
            previous = "call:amplified_" + str(index)
        result = core("runtime").run_experiment(graph, {"steps": 1, "batch_size": 2})
        probe = result["probes"][previous]
        self.assertEqual(probe["inf_count"], 0)
        self.assertTrue(math.isfinite(probe["std"]))
        self.assertTrue(math.isfinite(probe["mean"]))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
