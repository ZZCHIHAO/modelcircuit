"""Native editor document behavior, independent of DearPyGui."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from studio_core.examples import example_graph


def document(graph=None):
    try:
        cls = importlib.import_module("studio.native.document").Document
    except ModuleNotFoundError as exc:
        raise AssertionError("Native Document is not implemented") from exc
    return cls(graph)


class DocumentTests(unittest.TestCase):
    def test_default_graph_is_fresh_and_selection_is_explicit(self):
        first, second = document(), document()
        first.graph["name"] = "changed"
        self.assertNotEqual(first.graph["name"], second.graph["name"])
        self.assertTrue(first.validation()["valid"])
        self.assertIsNone(first.scope_id)
        first.select("call:encode_b")
        self.assertEqual(first.selected_call["id"], "encode_b")
        self.assertEqual(first.selected_instance["id"], "encoder")
        first.select("input:a")
        self.assertIsNone(first.selected_call)
        self.assertIsNone(first.selected_instance)
        with self.assertRaises(ValueError): first.select("call:missing")
        self.assertEqual(first.selected, "input:a")

    def test_navigation_tracks_exact_call_paths_through_nested_shared_chips(self):
        graph = example_graph()
        graph["definitions"].append({"id": "wrapper", "name": "Wrapper", "version": "0.1.0", "public_parameters": [], "bindings": {},
                                     "graph": {"inputs": [{"id": "x", "shape": [None, 4], "dtype": "float32"}],
                                               "instances": [{"id": "inner", "kind": "chip", "definition_id": "encoder_chip", "params": {"in_features": 4, "hidden_features": 8}}],
                                               "calls": [{"id": "inside", "instance_id": "inner", "inputs": {"x": "input:x"}}],
                                               "outputs": [{"id": "out", "source": "call:inside"}]}})
        graph["instances"][0]["definition_id"] = "wrapper"
        graph["instances"][0]["params"] = {}
        doc = document(graph)
        doc.enter("encode_a")
        doc.enter("inside")
        doc.select("call:project")
        self.assertEqual(doc.scope_id, "encoder_chip")
        self.assertEqual(doc.trail, [{"id": "wrapper", "call_id": "encode_a"}, {"id": "encoder_chip", "call_id": "inside"}])
        self.assertEqual(doc.probe_source(), "call:encode_a/call:inside/call:project")
        self.assertEqual(doc.probe_source("output:out"), "call:encode_a/call:inside/call:activate")
        doc.back()
        self.assertEqual(doc.scope_id, "wrapper")
        doc.root(); doc.enter("encode_b"); doc.enter("inside")
        self.assertEqual(doc.probe_source("input:x"), "call:encode_b/call:inside/input:x")
        self.assertEqual(doc.graph, graph)

    def test_semantic_edits_are_visible_invalid_and_never_rewrite_other_parameters(self):
        doc = document()
        revision, fingerprint = doc.graph["revision"], doc.semantic_key()
        doc.set_param("encoder", "hidden_features", 10)
        self.assertEqual(doc.graph["revision"], revision + 1)
        self.assertNotEqual(doc.semantic_key(), fingerprint)
        self.assertEqual(doc.graph["instances"][2]["params"]["in_features"], 8)
        self.assertFalse(doc.validation()["valid"])
        doc.set_param("head", "in_features", 10)
        self.assertTrue(doc.validation()["valid"])
        doc.set_input("encode_a", "x", "call:classify")
        self.assertFalse(doc.validation()["valid"])
        doc.set_input("encode_a", "x", None)
        self.assertNotIn("x", doc.graph["calls"][0]["inputs"])
        self.assertFalse(doc.validation()["valid"])

    def test_movement_changes_only_position_and_preserves_semantic_identity(self):
        doc = document()
        before = copy.deepcopy(doc.graph)
        fingerprint = doc.semantic_key()
        doc.move("call:encode_a", 123.5, 98)
        self.assertEqual(doc.semantic_key(), fingerprint)
        self.assertEqual(doc.graph["revision"], before["revision"])
        self.assertEqual(doc.graph["calls"][0]["position"], {"x": 123.5, "y": 98})
        before["calls"][0]["position"] = {"x": 123.5, "y": 98}
        self.assertEqual(doc.graph, before)
        with self.assertRaises(ValueError): doc.move("call:encode_a", float("inf"), 3)

    def test_parameter_edits_preserve_numeric_types_that_compare_equal(self):
        doc = document()
        revision = doc.graph["revision"]
        doc.set_param("encoder", "hidden_features", 8.0)
        self.assertIs(type(doc.graph["instances"][0]["params"]["hidden_features"]), float)
        self.assertEqual(doc.graph["revision"], revision + 1)
        self.assertFalse(doc.validation()["valid"])
        doc.set_param("encoder", "hidden_features", 8)
        self.assertIs(type(doc.graph["instances"][0]["params"]["hidden_features"]), int)
        self.assertTrue(doc.validation()["valid"])

    def test_output_can_be_disconnected_without_removing_its_editable_port(self):
        doc = document()
        doc.select("output:features")
        revision = doc.graph["revision"]
        doc.set_output("features", None)
        output = next(item for item in doc.scope["outputs"] if item["id"] == "features")
        self.assertNotIn("source", output)
        self.assertEqual(doc.selected, "output:features")
        self.assertIsNone(doc.probe_source())
        self.assertEqual(doc.graph["revision"], revision + 1)
        self.assertFalse(doc.validation()["valid"])
        self.assertTrue(doc.validation()["errors"])
        doc.set_output("output:features", None)
        self.assertEqual(doc.graph["revision"], revision + 1)
        doc.set_output("features", "call:fuse")
        self.assertEqual(doc.graph["revision"], revision + 2)
        self.assertTrue(doc.validation()["valid"])

    def test_definition_edit_applies_to_all_independent_instances(self):
        doc = document()
        new_ref = doc.add("chip", "encoder_chip")
        new_id = doc.selected_instance["id"]
        doc.set_input(new_ref, "x", "input:a")
        doc.enter("encode_a")
        doc.set_frozen("dense", True)
        self.assertTrue(doc.scope["instances"][0]["frozen"])
        doc.root()
        self.assertTrue(doc.validation()["valid"])
        from studio_core.runtime import build_model
        model = build_model(doc.graph)
        self.assertFalse(model.components["encoder"].components["dense"].weight.requires_grad)
        self.assertFalse(model.components[new_id].components["dense"].weight.requires_grad)

    def test_add_share_and_remove_preserve_instances_and_clear_dependents(self):
        doc = document()
        original = doc.add("Linear")
        self.assertFalse(doc.validation()["valid"])
        instance_id = doc.selected_instance["id"]
        doc.set_input(original, "x", "input:a")
        shared = doc.share(original)
        self.assertEqual(doc.selected_instance["id"], instance_id)
        self.assertEqual(sum(i["id"] == instance_id for i in doc.scope["instances"]), 1)
        downstream = doc.add("ReLU")
        doc.set_input(downstream, "x", original)
        doc.set_output("features", original)
        revision = doc.graph["revision"]
        doc.remove(original)
        self.assertEqual(doc.graph["revision"], revision + 1)
        self.assertIn(instance_id, [i["id"] for i in doc.scope["instances"]])
        doc.select(shared)
        self.assertEqual(doc.selected_instance["id"], instance_id)
        doc.select(downstream)
        self.assertEqual(doc.selected_call["inputs"], {})
        self.assertNotIn("source", next(o for o in doc.scope["outputs"] if o["id"] == "features"))
        doc.remove(shared)
        self.assertNotIn(instance_id, [i["id"] for i in doc.scope["instances"]])

    def test_encapsulation_copies_boundaries_sharing_and_nested_references(self):
        doc = document()
        before = copy.deepcopy(doc.graph)
        ref = doc.encapsulate(["encode_a", "encode_b", "fuse"], "Shared fusion", "fusion_chip", "fuse")
        self.assertEqual(doc.graph["calls"][:-1], before["calls"])
        self.assertEqual(doc.graph["instances"][:-1], before["instances"])
        self.assertEqual(doc.graph["definitions"][:-1], before["definitions"])
        self.assertEqual(doc.graph["revision"], before["revision"] + 1)
        definition = doc.graph["definitions"][-1]
        self.assertEqual(definition["id"], "fusion_chip")
        self.assertEqual(len(definition["graph"]["instances"]), 2)
        encoder = next(i for i in definition["graph"]["instances"] if i["kind"] == "chip")
        self.assertEqual(encoder["definition_id"], "encoder_chip")
        calls = definition["graph"]["calls"]
        self.assertEqual(calls[0]["instance_id"], calls[1]["instance_id"])
        self.assertEqual(set(doc.selected_call["inputs"].values()), {"input:a", "input:b"})
        self.assertEqual([p["shape"] for p in definition["graph"]["inputs"]], [[None, 4], [None, 4]])
        self.assertEqual(doc.selected, ref)
        self.assertTrue(doc.validation()["valid"], doc.validation()["errors"])
        self.assertEqual(doc.validation()["parameter_count"], 98)
        from studio_core.python_io import import_python
        from studio_core.runtime import build_model
        from studio_core.schema import validate_graph
        import torch
        restored = import_python(doc.export_python())
        self.assertEqual(validate_graph(restored)["parameter_count"], 98)
        torch.manual_seed(19); original_model = build_model(before)
        torch.manual_seed(19); edited_model = build_model(doc.graph)
        inputs = {"a": torch.ones(2, 4), "b": torch.zeros(2, 4)}
        for key, expected in original_model(**inputs).items():
            torch.testing.assert_close(edited_model(**inputs)[key], expected)
        self.assertIsNot(edited_model.components[doc.selected_instance["id"]].components["encoder"], edited_model.components["encoder"])

    def test_encapsulation_keeps_external_call_boundary_and_fails_atomically(self):
        doc = document()
        doc.encapsulate(["encode_a", "fuse"], "Partial", "partial_chip", "fuse")
        self.assertEqual(set(doc.selected_call["inputs"].values()), {"input:a", "call:encode_b"})
        self.assertTrue(doc.validation()["valid"])
        before = copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError): doc.encapsulate(["encode_a"], "Bad", "bad_chip", "classify")
        self.assertEqual(doc.graph, before)
        with self.assertRaises(ValueError): doc.encapsulate(["encode_a"], "Bad", "partial_chip", "encode_a")
        self.assertEqual(doc.graph, before)

    def test_template_and_current_source_import_reset_navigation_and_increment_revision(self):
        doc = document()
        ref = doc.add_template("New encoder", "new_encoder")
        self.assertFalse(doc.validation()["valid"])
        doc.set_input(ref, "x", "input:a")
        self.assertTrue(doc.validation()["valid"], doc.validation()["errors"])
        code = doc.export_python()
        self.assertIn("out_features=2", code)
        doc.enter("encode_a")
        revision = doc.graph["revision"]
        doc.import_python(code.replace("out_features=2", "out_features=3"))
        self.assertIsNone(doc.scope_id)
        self.assertEqual(doc.trail, [])
        self.assertIsNone(doc.selected)
        self.assertEqual(doc.graph["revision"], revision + 1)
        self.assertEqual(next(i for i in doc.graph["instances"] if i["id"] == "head")["params"]["out_features"], 3)

    def test_json_loading_is_transactional_and_invalid_drafts_can_be_saved(self):
        doc = document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            doc.set_input("encode_a", "x", None)
            doc.save_json(path)
            restored = document()
            restored.load_json(path)
            self.assertEqual(restored.graph, doc.graph)
            self.assertFalse(restored.validation()["valid"])
            restored.enter("encode_b")
            before, trail = copy.deepcopy(restored.graph), copy.deepcopy(restored.trail)
            bad_position = example_graph()
            bad_position["calls"][0]["position"] = []
            bad_public = example_graph()
            bad_public["definitions"][0]["public_parameters"] = ["wrong"]
            for bad in ([], {"inputs": "bad"}, {**example_graph(), "instances": ["wrong"]}, bad_position, bad_public):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError): restored.load_json(path)
                self.assertEqual(restored.graph, before)
                self.assertEqual(restored.trail, trail)
            path.write_text('{"name":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError): restored.load_json(path)
            self.assertEqual(restored.graph, before)


if __name__ == "__main__":
    unittest.main()
