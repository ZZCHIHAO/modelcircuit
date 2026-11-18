"""Public constructor types and omitted overrides are source-visible contracts."""
import copy
import unittest

from studio_core import python_io, schema
from studio_core.examples import example_graph
from studio_core.experiment_spec import checkpoint_signature, default_classification_config, model_digest


def typed_graph(nested=False):
    graph = example_graph()
    graph["instances"][0]["params"] = {}
    inner = graph["definitions"][0]
    for parameter in inner["public_parameters"]:
        parameter["type"] = "int"
    if nested:
        outer = {"id": "outer", "output_mode": "named", "public_parameters": [{"id": "width", "type": "int", "default": 8}],
                 "bindings": {"width": ["inner_block.hidden_features"]}, "graph": {
                     "inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                     "instances": [{"id": "inner_block", "kind": "chip", "definition_id": inner["id"], "params": {}}],
                     "calls": [{"id": "encode", "instance_id": "inner_block", "inputs": {"x": "input:x"}}],
                     "outputs": [{"id": "out", "source": "call:encode"}]}}
        graph["definitions"].append(outer)
        graph["instances"][0]["definition_id"] = "outer"
    return graph


class PublicConstructorTypeTests(unittest.TestCase):
    def test_metadata_free_annotations_preserve_declared_types_and_omitted_overrides(self):
        graph = typed_graph()
        source = python_io.export_python(graph)
        self.assertIn("hidden_features: 'int'=8", source)
        without = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(without)
        self.assertEqual([p["type"] for p in restored["definitions"][0]["public_parameters"]], ["int", "int"])
        self.assertEqual(restored["instances"][0]["params"], {})
        restored["definitions"][0]["public_parameters"][1]["default"] = 6
        restored["instances"][2]["params"]["in_features"] = 6
        report = schema.validate_graph(restored)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["shapes"]["call:fuse"], [None, 6])

    def test_current_annotation_and_explicit_constructor_override_edits_win(self):
        graph = typed_graph()
        graph["instances"][0]["params"] = {"hidden_features": 5}
        graph["instances"][2]["params"]["in_features"] = 5
        source = python_io.export_python(graph)
        self.assertIn("hidden_features: 'int'=8", source)
        edited = source.replace("hidden_features: 'int'=8", "hidden_features: 'auto'=8")
        edited = edited.replace("hidden_features=5", "hidden_features=6").replace("in_features=5, out_features=2", "in_features=6, out_features=2")
        restored = python_io.import_python(edited)
        self.assertEqual(restored["definitions"][0]["public_parameters"][1]["type"], "auto")
        self.assertEqual(restored["instances"][0]["params"], {"hidden_features": 6})

    def test_unknown_or_executable_constructor_annotations_are_rejected_statically(self):
        source = python_io.export_python(typed_graph())
        self.assertIn("hidden_features: 'int'=8", source)
        for annotation in ("int", "'float32'", "__import__('os').system('bad')", "['int']"):
            changed = source.replace("hidden_features: 'int'=8", "hidden_features: " + annotation + "=8")
            with self.subTest(annotation=annotation), self.assertRaises(ValueError): python_io.import_python(changed)

    def test_no_edit_model_and_checkpoint_identity_include_nested_parent_bindings(self):
        config = default_classification_config()
        for nested in (False, True):
            graph = typed_graph(nested)
            restored = python_io.import_python(python_io.export_python(graph))
            with self.subTest(nested=nested):
                self.assertEqual(restored["instances"][0]["params"], {})
                self.assertEqual(model_digest(graph), model_digest(restored))
                self.assertEqual(checkpoint_signature(graph, config, "a" * 64), checkpoint_signature(restored, config, "a" * 64))

    def test_untyped_s1_constructor_stays_unannotated_and_without_type_fields(self):
        graph = example_graph(); source = python_io.export_python(graph)
        self.assertNotIn("hidden_features:", source)
        self.assertEqual(python_io.python_profile(graph), "s1-pytorch-v1")
        restored = python_io.import_python(source)
        self.assertTrue(all("type" not in parameter for parameter in restored["definitions"][0]["public_parameters"]))

    def test_s2_false_presence_hint_preserves_identity_and_cannot_override_current_freeze(self):
        graph = example_graph(); source = python_io.export_python(graph)
        restored = python_io.import_python(source)
        self.assertEqual(model_digest(graph), model_digest(restored))
        self.assertIs(restored["instances"][0]["frozen"], False)
        edited = source.replace("    def forward(self, a:", "        self.components['encoder'].requires_grad_(False)\n\n    def forward(self, a:", 1)
        self.assertNotEqual(edited, source)
        changed = python_io.import_python(edited)
        self.assertIs(changed["instances"][0]["frozen"], True)
        forged = source.replace("'frozen': False", "'frozen': True")
        self.assertNotEqual(forged, source)
        self.assertNotIn("frozen", python_io.import_python(forged)["instances"][0])
        explicit_unfreeze = forged.replace("    def forward(self, a:", "        self.components['encoder'].requires_grad_(True)\n\n    def forward(self, a:", 1)
        self.assertIs(python_io.import_python(explicit_unfreeze)["instances"][0]["frozen"], False)

    def test_constructor_omissions_and_cnn_optional_defaults_do_not_change_model_identity(self):
        from studio.backend.tests.test_cnn_model import cnn_graph
        graph = cnn_graph()
        restored = python_io.import_python(python_io.export_python(graph))
        self.assertEqual(model_digest(graph), model_digest(restored))

    def test_declared_type_default_conflict_from_current_source_is_rejected(self):
        graph = PublicTypeValidationTests().graph("float", 0.5)
        source = python_io.export_python(graph)
        self.assertIn("alpha: 'float'=0.5", source)
        modified = source.replace("alpha: 'float'=0.5", "alpha: 'int'=0.5")
        with self.assertRaisesRegex(ValueError, "[Pp]ublic.*type"):
            python_io.import_python(modified)


class PublicTypeValidationTests(unittest.TestCase):
    def graph(self, declared_type, default, override=None, nested=False):
        definition = {"id": "typed_drop", "public_parameters": [{"id": "alpha", "type": declared_type, "default": default}],
                      "bindings": {"alpha": ["drop.p"]}, "graph": {
                          "inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                          "instances": [{"id": "drop", "kind": "Dropout", "params": {"p": 0.5}}],
                          "calls": [{"id": "noise", "instance_id": "drop", "inputs": {"x": "input:x"}}],
                          "outputs": [{"id": "out", "source": "call:noise"}]}}
        graph = {"schema_version": 1, "id": "public_type", "revision": 1, "inputs": copy.deepcopy(definition["graph"]["inputs"]),
                 "instances": [{"id": "block", "kind": "chip", "definition_id": definition["id"], "params": {} if override is None else {"alpha": override}}],
                 "calls": [{"id": "apply_block", "instance_id": "block", "inputs": {"x": "input:x"}}],
                 "outputs": [{"id": "out", "source": "call:apply_block"}], "definitions": [definition]}
        if nested:
            outer = {"id": "parent", "public_parameters": [{"id": "value", "default": override}], "bindings": {"value": ["child.alpha"]},
                     "graph": {"inputs": copy.deepcopy(graph["inputs"]), "instances": [{"id": "child", "kind": "chip", "definition_id": "typed_drop", "params": {}}],
                               "calls": [{"id": "nested", "instance_id": "child", "inputs": {"x": "input:x"}}], "outputs": [{"id": "out", "source": "call:nested"}]}}
            graph["definitions"].append(outer); graph["instances"][0].update(definition_id="parent", params={})
        return graph

    def test_default_instance_override_and_nested_binding_must_obey_declared_type(self):
        cases = [("int", 0.5, None, False), ("int", 1, 0.5, False), ("int", 1, 0.5, True),
                 ("bool", 1, None, False), ("float", True, None, False), ("shape", 0.5, None, False), ("unknown", 0.5, None, False)]
        for declared, default, override, nested in cases:
            with self.subTest(declared=declared, override=override, nested=nested):
                result = schema.validate_graph(self.graph(declared, default, override, nested))
                self.assertFalse(result["valid"], result)
                self.assertRegex(result["errors"][0]["message"].lower(), "public|type|default")
        for declared, value in (("auto", 0.5), ("float", 0.5), ("int", 1)):
            self.assertTrue(schema.validate_graph(self.graph(declared, value))["valid"])

    def test_all_public_type_rules_apply_even_without_a_primitive_binding(self):
        for declared, value in (("auto", {"value": 1}), ("int", -3), ("float", 0.75), ("bool", True), ("shape", [2, 3]), ("shape", 4)):
            graph = self.graph(declared, value)
            graph["definitions"][0]["bindings"] = {}
            with self.subTest(declared=declared, value=value):
                self.assertTrue(schema.validate_graph(graph)["valid"])


if __name__ == "__main__": unittest.main()
