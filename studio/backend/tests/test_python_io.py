"""The S1 bridge reads source, never executes uploaded Python."""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from studio_core import python_io
except ImportError:
    python_io = None


def fixture_graph():
    return {
        "schema_version": 1, "id": "shared", "name": "Shared model", "revision": 3,
        "inputs": [{"id": "a", "name": "Left", "shape": [None, 4], "dtype": "float32"},
                   {"id": "b", "name": "Right", "shape": [None, 4], "dtype": "float32"}],
        "instances": [
            {"id": "encoder", "kind": "chip", "definition_id": "encoder_chip", "name": "Encoder", "params": {"in_features": 4, "hidden_features": 8}, "frozen": False},
            {"id": "fusion", "kind": "Add", "name": "Fusion", "params": {}},
            {"id": "head", "kind": "Linear", "name": "Head", "params": {"in_features": 8, "out_features": 2}, "frozen": False}],
        "calls": [
            {"id": "encode_a", "instance_id": "encoder", "inputs": {"x": "input:a"}, "position": {"x": 250, "y": 90}},
            {"id": "encode_b", "instance_id": "encoder", "inputs": {"x": "input:b"}, "position": {"x": 250, "y": 300}},
            {"id": "fuse", "instance_id": "fusion", "inputs": {"a": "call:encode_a", "b": "call:encode_b"}},
            {"id": "classify", "instance_id": "head", "inputs": {"x": "call:fuse"}}],
        "outputs": [{"id": "logits", "source": "call:classify"}, {"id": "features", "source": "call:fuse"}],
        "definitions": [{
            "id": "encoder_chip", "name": "Encoder chip", "version": "0.1.0",
            "public_parameters": [{"id": "in_features", "name": "Input size", "default": 4}, {"id": "hidden_features", "name": "Hidden size", "default": 8}],
            "bindings": {"in_features": ["dense.in_features"], "hidden_features": ["dense.out_features"]},
            "graph": {"inputs": [{"id": "x", "name": "Features", "shape": [None, 4], "dtype": "float32"}],
                      "instances": [{"id": "dense", "kind": "Linear", "name": "Dense", "params": {"in_features": 4, "out_features": 8}}, {"id": "activation", "kind": "ReLU", "params": {}}],
                      "calls": [{"id": "project", "instance_id": "dense", "inputs": {"x": "input:x"}}, {"id": "activate", "instance_id": "activation", "inputs": {"x": "call:project"}}],
                      "outputs": [{"id": "out", "source": "call:activate"}]}}]}


class PythonBridgeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(python_io, "The Python bridge is not implemented")
        self.graph = fixture_graph()

    def code(self):
        return python_io.export_python(self.graph)

    def test_round_trip_sharing_bindings_names_and_positions(self):
        restored = python_io.import_python(self.code())
        self.assertEqual([c["instance_id"] for c in restored["calls"][:2]], ["encoder", "encoder"])
        self.assertEqual(restored["definitions"][0]["bindings"], self.graph["definitions"][0]["bindings"])
        self.assertEqual(restored["name"], "Shared model")
        self.assertEqual(restored["calls"][0]["position"], {"x": 250, "y": 90})
        self.assertEqual(restored["outputs"], self.graph["outputs"])
        self.assertEqual(restored["inputs"][0]["shape"], [None, 4])

    def test_input_and_output_positions_round_trip_in_root_and_nested_scopes(self):
        for scope_index, scope in enumerate((self.graph, self.graph["definitions"][0]["graph"])):
            for group_index, group in enumerate(("inputs", "outputs")):
                for item_index, item in enumerate(scope[group]):
                    item["position"] = {"x": 30 + 700 * group_index, "y": 80 + 140 * item_index + 25 * scope_index}
        restored = python_io.import_python(self.code())
        for original, imported in zip((self.graph, self.graph["definitions"][0]["graph"]), (restored, restored["definitions"][0]["graph"])):
            for group in ("inputs", "outputs"):
                with self.subTest(group=group, scope=original["inputs"][0]["id"]):
                    self.assertEqual({item["id"]: item.get("position") for item in imported[group]}, {item["id"]: item["position"] for item in original[group]})

    def test_port_positions_do_not_restore_stale_input_shape_or_output_source(self):
        self.graph["inputs"][0]["position"] = {"x": 35, "y": 120}
        self.graph["outputs"][1]["position"] = {"x": 900, "y": 310}
        inner = self.graph["definitions"][0]["graph"]
        inner["inputs"][0]["position"] = {"x": 40, "y": 100}
        inner["outputs"][0]["position"] = {"x": 720, "y": 115}
        source = self.code().replace("'float32[?,4]'", "'float32[?,3,4]'")
        source = source.replace("'features': call_fuse", "'features': call_encode_a").replace("        return call_activate", "        return call_project")
        restored = python_io.import_python(source)
        self.assertEqual(restored["inputs"][0]["shape"], [None, 3, 4])
        self.assertEqual(restored["outputs"][1]["source"], "call:encode_a")
        self.assertEqual(restored["inputs"][0].get("position"), {"x": 35, "y": 120})
        self.assertEqual(restored["outputs"][1].get("position"), {"x": 900, "y": 310})
        imported_inner = restored["definitions"][0]["graph"]
        self.assertEqual(imported_inner["inputs"][0]["shape"], [None, 3, 4])
        self.assertEqual(imported_inner["outputs"][0]["source"], "call:project")
        self.assertEqual(imported_inner["inputs"][0].get("position"), {"x": 40, "y": 100})
        self.assertEqual(imported_inner["outputs"][0].get("position"), {"x": 720, "y": 115})

    def test_recursive_unfreeze_over_frozen_descendant_is_rejected_without_silent_translation(self):
        for nested in (False, True):
            graph = fixture_graph()
            graph["definitions"][0]["graph"]["instances"][0]["frozen"] = True
            if nested:
                graph["definitions"].append({"id": "outer", "public_parameters": [], "bindings": {}, "graph": {
                    "inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                    "instances": [{"id": "inner", "kind": "chip", "definition_id": "encoder_chip", "params": {}}],
                    "calls": [{"id": "encode", "instance_id": "inner", "inputs": {"x": "input:x"}}],
                    "outputs": [{"id": "out", "source": "call:encode"}]}})
                graph["instances"][0].update(definition_id="outer", params={})
            source = python_io.export_python(graph)
            edited = source.replace("    def forward(self, a:", "        self.components['encoder'].requires_grad_(True)\n\n    def forward(self, a:", 1)
            self.assertNotEqual(edited, source)
            with self.subTest(nested=nested), self.assertRaisesRegex(ValueError, "[Rr]ecursive unfreeze|frozen descendant"):
                python_io.import_python(edited)

    def test_representable_unfreeze_and_later_explicit_freeze_follow_current_source(self):
        source = self.code()
        directives = "        self.components['head'].requires_grad_(True)\n        self.components['encoder'].requires_grad_(True)\n\n"
        edited = source.replace("    def forward(self, a:", directives + "    def forward(self, a:", 1)
        restored = python_io.import_python(edited)
        self.assertIs(restored["instances"][0]["frozen"], False)
        self.assertIs(restored["instances"][2]["frozen"], False)
        self.graph["definitions"][0]["graph"]["instances"][0]["frozen"] = True
        source = self.code()
        directives = "        self.components['encoder'].requires_grad_(True)\n        self.components['encoder'].requires_grad_(False)\n\n"
        restored = python_io.import_python(source.replace("    def forward(self, a:", directives + "    def forward(self, a:", 1))
        self.assertIs(restored["instances"][0]["frozen"], True)

    def test_source_edit_of_linear_dimensions_overrides_metadata(self):
        source = self.code().replace("in_features=8, out_features=2", "in_features=8, out_features=3")
        self.assertNotEqual(source, self.code())
        restored = python_io.import_python(source)
        self.assertEqual(next(x for x in restored["instances"] if x["id"] == "head")["params"]["out_features"], 3)

    def test_source_edit_of_connection_overrides_metadata(self):
        source = self.code().replace("b=call_encode_b", "b=call_encode_a")
        self.assertNotEqual(source, self.code())
        restored = python_io.import_python(source)
        self.assertEqual(restored["calls"][2]["inputs"]["b"], "call:encode_a")

    def test_without_metadata_source_reconstructs_structure(self):
        source = "\n".join(line for line in self.code().splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(source)
        self.assertEqual(len(restored["definitions"]), 1)
        self.assertEqual(restored["calls"][0]["instance_id"], restored["calls"][1]["instance_id"])
        self.assertEqual(restored["inputs"][0]["shape"], [None, 4])

    def test_public_constructor_edits_update_parameters_and_binding_defaults(self):
        source = self.code().replace("in_features=4, hidden_features=8", "in_features=4, hidden_features=12")
        source = source.replace("in_features=8, out_features=2", "in_features=12, out_features=2")
        restored = python_io.import_python(source)
        self.assertEqual(restored["instances"][0]["params"]["hidden_features"], 12)

    def test_input_shape_is_source_declared_without_visual_metadata(self):
        self.graph["inputs"][0]["shape"] = [None, 3, 4]
        self.graph["inputs"][1]["shape"] = [None, 3, 4]
        self.graph["definitions"][0]["graph"]["inputs"][0]["shape"] = [None, 3, 4]
        source = "\n".join(line for line in self.code().splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(source)
        self.assertEqual(restored["inputs"][0]["shape"], [None, 3, 4])

    def test_stale_input_shape_metadata_cannot_override_current_source(self):
        source = self.code()
        tree = ast.parse(source)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign))
        visual = ast.literal_eval(assignment.value)
        visual["root"]["inputs"]["a"]["shape"] = [None, 99, 4]
        visual["root"]["inputs"]["a"]["dtype"] = "float64"
        lines = source.splitlines()
        lines[assignment.lineno - 1] = "STUDIO_VISUAL_METADATA = " + repr(visual)
        restored = python_io.import_python("\n".join(lines))
        self.assertEqual(restored["inputs"][0]["shape"], [None, 4])
        self.assertEqual(restored["inputs"][0]["dtype"], "float32")

    def test_unsafe_annotation_is_rejected_without_execution(self):
        source = self.code().replace("a: 'float32[?,4]'", "a: __import__('os').system('echo bad')")
        self.assertNotEqual(source, self.code())
        with self.assertRaisesRegex(ValueError, "Unsupported|unsupported"):
            python_io.import_python(source)

    def test_alias_imports_and_local_value_aliases_are_static(self):
        source = self.code().replace("from torch import nn", "from torch import nn as layers").replace("nn.", "layers.")
        source = source.replace("        call_fuse =", "        shared_value = call_encode_a\n        call_fuse =").replace("a=call_encode_a", "a=shared_value")
        self.assertEqual(python_io.import_python(source)["calls"][2]["inputs"]["a"], "call:encode_a")

    def test_common_torch_module_import_aliases_are_supported(self):
        source = self.code().replace("import torch\n", "import torch as pt\n").replace("torch.cat", "pt.cat")
        source = source.replace("from torch import nn", "import torch.nn as layers").replace("nn.", "layers.")
        self.assertEqual(python_io.import_python(source)["instances"][2]["kind"], "Linear")

    def test_native_pytorch_input_keyword_maps_to_graph_x_port(self):
        source = self.code().replace("](x=call_fuse)", "](input=call_fuse)")
        restored = python_io.import_python(source)
        self.assertEqual(restored["calls"][3]["inputs"], {"x": "call:fuse"})

    def test_missing_imports_are_not_mistaken_for_valid_standalone_source(self):
        source = self.code().replace("import torch\n", "").replace("from torch import nn\n", "")
        with self.assertRaisesRegex(ValueError, "import|Import"):
            python_io.import_python(source)

    def test_primitive_parameter_edits_are_read_from_source(self):
        self.graph["instances"][1] = {"id": "fusion", "kind": "Concat", "params": {"dim": -1}}
        self.graph["instances"][2]["params"]["in_features"] = 16
        self.graph["instances"].append({"id": "drop", "kind": "Dropout", "params": {"p": 0.25}})
        self.graph["calls"].append({"id": "drop_out", "instance_id": "drop", "inputs": {"x": "call:classify"}})
        source = self.code().replace("Concat(dim=-1)", "Concat(dim=1)").replace("p=0.25", "p=0.4")
        restored = python_io.import_python(source)
        self.assertEqual(restored["instances"][1]["params"]["dim"], 1)
        self.assertEqual(restored["instances"][3]["params"]["p"], 0.4)

    def test_layernorm_tuple_syntax_imports_as_schema_compatible_shape(self):
        self.graph["instances"].append({"id": "norm", "kind": "LayerNorm", "params": {"normalized_shape": [8]}})
        self.graph["calls"].append({"id": "normalized", "instance_id": "norm", "inputs": {"x": "call:fuse"}})
        source = self.code().replace("normalized_shape=[8]", "normalized_shape=(8,)")
        restored = python_io.import_python(source)
        self.assertEqual(restored["instances"][3]["params"]["normalized_shape"], [8])

    def test_nested_chip_hierarchy_has_scoped_instance_ids(self):
        inner = copy.deepcopy(self.graph["definitions"][0])
        outer = copy.deepcopy(inner)
        outer["id"] = "outer_chip"
        outer["public_parameters"] = []
        outer["bindings"] = {}
        outer["graph"]["instances"] = [{"id": "encoder", "kind": "chip", "definition_id": "encoder_chip", "params": {"in_features": 4, "hidden_features": 8}}]
        outer["graph"]["calls"] = [{"id": "encode", "instance_id": "encoder", "inputs": {"x": "input:x"}}]
        outer["graph"]["outputs"] = [{"id": "out", "source": "call:encode"}]
        self.graph["definitions"] = [inner, outer]
        self.graph["instances"][0]["definition_id"] = "outer_chip"
        self.graph["instances"][0]["params"] = {}
        restored = python_io.import_python(self.code())
        outer_restored = next(d for d in restored["definitions"] if d["id"] == "outer_chip")
        self.assertEqual(outer_restored["graph"]["instances"][0]["id"], "encoder")
        self.assertEqual(outer_restored["graph"]["instances"][0]["definition_id"], "encoder_chip")

    def test_unsupported_code_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            malicious = self.code() + "\nopen(" + repr(str(marker)) + ", 'w').write('bad')\n"
            with self.assertRaisesRegex(ValueError, "Unsupported|unsupported"):
                python_io.import_python(malicious)
            self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "Unsupported|unsupported"):
            python_io.import_python(self.code().replace("        call_fuse =", "        if a.sum() > 0:\n            a = a + 1\n        call_fuse ="))

    def test_invalid_and_unrecognized_python_raise_value_error(self):
        for source in ("def broken(:", "x = 1", "import os\nos.system('echo bad')"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                python_io.import_python(source)

    def test_dropout_call_order_preserves_ready_source_sequence(self):
        graph = fixture_graph()
        graph["inputs"] = [graph["inputs"][0]]
        graph["definitions"] = []
        graph["instances"] = [{"id": name, "kind": "Dropout", "params": {"p": 0.5}} for name in ("drop_one", "drop_two", "drop_three")]
        graph["calls"] = [
            {"id": "first", "instance_id": "drop_one", "inputs": {"x": "input:a"}},
            {"id": "second", "instance_id": "drop_two", "inputs": {"x": "call:first"}},
            {"id": "third", "instance_id": "drop_three", "inputs": {"x": "input:a"}},
        ]
        graph["outputs"] = [{"id": name, "source": "call:" + name} for name in ("first", "second", "third")]
        source = python_io.export_python(graph)
        restored = python_io.import_python(source)
        self.assertEqual([call["id"] for call in restored["calls"]], ["first", "second", "third"])
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "dropout_order.py"
            script.write_text(source + '''
a = torch.ones(16, 4)
model = Model().train()
torch.manual_seed(732)
first = torch.nn.functional.dropout(a, p=0.5, training=True)
second = torch.nn.functional.dropout(first, p=0.5, training=True)
third = torch.nn.functional.dropout(a, p=0.5, training=True)
torch.manual_seed(732)
actual = model(a=a)
for name, expected in (("first", first), ("second", second), ("third", third)):
    torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
''', encoding="utf-8")
            process = subprocess.run([sys.executable, "-I", str(script)], cwd=directory, text=True, capture_output=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_unused_chip_component_library_survives_round_trip(self):
        unused = copy.deepcopy(self.graph["definitions"][0])
        unused["id"] = "unused_encoder"
        self.graph["definitions"].append(unused)
        source = self.code()
        compile(source, "exported_model.py", "exec")
        restored = python_io.import_python(source)
        self.assertEqual({item["id"] for item in restored["definitions"]}, {"encoder_chip", "unused_encoder"})
        self.assertEqual(restored["outputs"], self.graph["outputs"])
        without_visuals = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        self.assertEqual(len(python_io.import_python(without_visuals)["definitions"]), 2)
        unrelated = source[source.index("class Model(nn.Module):"):].replace("class Model(nn.Module):", "class AnotherModel(nn.Module):", 1)
        with self.assertRaisesRegex(ValueError, "root|unrelated"):
            python_io.import_python(source + "\n" + unrelated)

    def test_public_binding_supplies_missing_primitive_constructor_parameter(self):
        from studio_core.schema import validate_graph
        del self.graph["definitions"][0]["graph"]["instances"][0]["params"]["out_features"]
        self.assertTrue(validate_graph(self.graph)["valid"])
        source = self.code()
        compile(source, "exported_model.py", "exec")
        restored = python_io.import_python(source)
        definition = restored["definitions"][0]
        self.assertEqual(definition["graph"]["instances"][0]["params"]["out_features"], 8)
        self.assertEqual(definition["bindings"]["hidden_features"], ["dense.out_features"])
        self.assertTrue(validate_graph(restored)["valid"])

    def test_declared_input_shape_conflict_is_rejected_instead_of_rewritten(self):
        source = self.code().replace("a: 'float32[?,4]'", "a: 'float32[?,99]'")
        self.assertNotEqual(source, self.code())
        with self.assertRaisesRegex(ValueError, "Invalid imported graph|input|shape"):
            python_io.import_python(source)

    def test_in_features_edits_require_matching_explicit_input_annotations(self):
        source = self.code().replace("in_features=4, hidden_features=8", "in_features=5, hidden_features=8")
        with self.assertRaisesRegex(ValueError, "Invalid imported graph|input|shape"):
            python_io.import_python(source)
        consistent = source.replace("'float32[?,4]'", "'float32[?,5]'")
        restored = python_io.import_python(consistent)
        self.assertEqual(restored["inputs"][0]["shape"], [None, 5])
        self.assertEqual(restored["instances"][0]["params"]["in_features"], 5)

    def test_public_parameters_cannot_shadow_exporter_runtime_names(self):
        for reserved in ("self", "nn", "super"):
            with self.subTest(name=reserved):
                graph = fixture_graph()
                graph["definitions"][0]["public_parameters"].append({"id": reserved, "default": 1})
                with self.assertRaisesRegex(ValueError, "reserved|Reserved|shadow"):
                    python_io.export_python(graph)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
    def test_export_is_standalone_and_state_keys_match_runtime(self):
        from studio_core.runtime import build_model
        expected_keys = sorted(build_model(self.graph).state_dict())
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "standalone.py"
            script.write_text(self.code() + "\nimport json\nm = Model()\noutputs = m(torch.ones(3, 4), torch.ones(3, 4))\nprint(json.dumps({'keys': sorted(m.state_dict()), 'shape': list(outputs['logits'].shape)}))\n", encoding="utf-8")
            result = subprocess.run([sys.executable, "-I", str(script)], cwd=directory, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        measured = json.loads(result.stdout)
        self.assertEqual(measured["keys"], expected_keys)
        self.assertEqual(measured["shape"], [3, 2])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
    def test_nested_shared_export_matches_runtime_outputs_and_gradients(self):
        import torch
        from studio_core.runtime import build_model

        graph = fixture_graph()
        inner = graph["definitions"][0]
        graph["definitions"].append({
            "id": "outer_chip", "name": "Outer chip", "version": "0.1.0",
            "public_parameters": copy.deepcopy(inner["public_parameters"]),
            "bindings": {"in_features": ["encoder.in_features"], "hidden_features": ["encoder.hidden_features"]},
            "graph": {
                "inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                "instances": [{"id": "encoder", "kind": "chip", "definition_id": "encoder_chip", "params": {"in_features": 4, "hidden_features": 8}}],
                "calls": [{"id": "encode", "instance_id": "encoder", "inputs": {"x": "input:x"}}],
                "outputs": [{"id": "out", "source": "call:encode"}],
            },
        })
        graph["instances"][0]["definition_id"] = "outer_chip"
        graph["instances"][0]["params"] = {"in_features": 5, "hidden_features": 6}
        graph["instances"][2]["params"]["in_features"] = 6
        graph["instances"][2]["frozen"] = True
        for item in graph["inputs"]:
            item["shape"] = [None, 5]
        torch.manual_seed(812)
        reference = build_model(graph).eval()
        dense_name = "components.encoder.components.encoder.components.dense.weight"
        self.assertEqual(list(reference.state_dict()[dense_name].shape), [6, 5])
        a = torch.linspace(-1.5, 2.0, 15).reshape(3, 5).requires_grad_(True)
        b = torch.linspace(0.8, -1.0, 15).reshape(3, 5).requires_grad_(True)
        expected_outputs = reference(a=a, b=b)
        expected_loss = expected_outputs["logits"].square().mean() + 0.01 * expected_outputs["features"].square().mean()
        expected_loss.backward()
        expected_gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
        self.assertIsNone(expected_gradients["components.head.weight"])
        self.assertGreater(expected_gradients[dense_name].abs().sum().item(), 0)
        source = python_io.export_python(graph)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "standalone_parity.py"
            payload = Path(directory) / "runtime_state.pt"
            measured_path = Path(directory) / "measured.pt"
            torch.save({"state": reference.state_dict(), "a": a.detach(), "b": b.detach()}, payload)
            script.write_text(source + '''
import sys
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
model = Model().eval()
model.load_state_dict(payload["state"], strict=True)
a = payload["a"].clone().requires_grad_(True)
b = payload["b"].clone().requires_grad_(True)
outputs = model(a=a, b=b)
loss = outputs["logits"].square().mean() + 0.01 * outputs["features"].square().mean()
loss.backward()
torch.save({
    "outputs": {name: value.detach() for name, value in outputs.items()},
    "gradients": {name: parameter.grad for name, parameter in model.named_parameters()},
    "a_grad": a.grad,
    "b_grad": b.grad,
    "trainable": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
}, sys.argv[2])
''', encoding="utf-8")
            process = subprocess.run([sys.executable, "-I", str(script), str(payload), str(measured_path)], cwd=directory, text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            measured = torch.load(measured_path, map_location="cpu", weights_only=True)
        self.assertEqual(set(measured["outputs"]), {"logits", "features"})
        for name, value in expected_outputs.items():
            torch.testing.assert_close(measured["outputs"][name], value.detach(), rtol=1e-6, atol=1e-7)
        self.assertEqual(set(measured["gradients"]), set(expected_gradients))
        for name, gradient in expected_gradients.items():
            with self.subTest(parameter=name):
                if gradient is None:
                    self.assertIsNone(measured["gradients"][name])
                else:
                    torch.testing.assert_close(measured["gradients"][name], gradient, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(measured["a_grad"], a.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(measured["b_grad"], b.grad, rtol=1e-6, atol=1e-7)
        self.assertFalse(measured["trainable"]["components.head.weight"])
        self.assertTrue(measured["trainable"][dense_name])


if __name__ == "__main__":
    unittest.main()
