import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from native.document import Document


class WorkspaceStateTests(unittest.TestCase):
    def test_paired_workspace_and_fusion_starter_preserve_named_mapping(self):
        from studio_core.experiment_spec import default_classification_config
        from native.workspace_state import save_workspace,read_workspace,starter_graph,compatibility_issue
        manifest={'kind':'paired_csv','input_shapes':{'image':[None,3,16,16],'tabular':[None,2]},'class_names':['a','b']}
        config=default_classification_config('paired_csv')
        doc=Document(starter_graph(manifest))
        self.assertTrue(doc.validation()['valid'],doc.validation())
        self.assertEqual({item['id'] for item in doc.graph['inputs']},{'image','clinical'})
        self.assertEqual(compatibility_issue(doc,manifest,input_ports=config['input_ports']),'')
        self.assertTrue(compatibility_issue(doc,manifest,input_ports={'image':'clinical','tabular':'image'}))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'workspace.json';save_workspace(path,doc,config)
            restored,experiment=read_workspace(path)
            self.assertEqual(experiment,config)
            self.assertEqual(restored.graph,doc.graph)

    def test_workspace_roundtrip_and_legacy_graph(self):
        from native.workspace_state import read_workspace, save_workspace, synthetic_config
        document = Document()
        experiment = {"profile": "classification-v2", "data": {"kind": "csv", "path": "D:/测试.csv", "target_column": "label", "feature_columns": ["a", "b"]}, "epochs": 8, "device": "cpu"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workspace.json"
            save_workspace(path, document, experiment)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["workspace_version"], 2)
            restored, settings = read_workspace(path)
            self.assertEqual(restored.graph, document.graph)
            self.assertEqual(settings, experiment)
            document.save_json(path)
            restored, settings = read_workspace(path)
            self.assertEqual(restored.graph, document.graph)
            self.assertEqual(settings, synthetic_config())

    def test_identity_includes_settings_and_data_digest_but_not_positions(self):
        from native.workspace_state import composite_key, synthetic_config
        document = Document()
        config = synthetic_config()
        before = composite_key(document, config, None)
        document.move("call:encode_a", 100, 200)
        self.assertEqual(composite_key(document, config, None), before)
        changed = deepcopy(config); changed["lr"] = 0.01
        self.assertNotEqual(composite_key(document, changed, None), before)
        self.assertNotEqual(composite_key(document, config, {"dataset_digest": "a"}), before)
        self.assertNotEqual(composite_key(document, config, {"dataset_digest": "a"}), composite_key(document, config, {"dataset_digest": "b"}))

    def test_invalid_envelope_does_not_replace_existing_file(self):
        from native.workspace_state import read_workspace, save_workspace
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workspace.json"
            path.write_text('{"workspace_version":99}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "workspace_version"):
                read_workspace(path)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                save_workspace(path, Document(), {"lr": float("nan")})
            self.assertEqual(path.read_bytes(), original)

    def test_editable_experiment_rejects_malformed_controls_but_allows_unconfigured_data(self):
        from native.workspace_state import read_workspace, save_workspace
        malformed = [
            {"profile": "classification-v2", "data": None},
            {"profile": "unknown"},
            {"profile": "classification-v2", "data": {"kind": "video"}},
            {"profile": "classification-v2", "data": {"kind": "csv"}, "epochs": "ten"},
            {"profile": "classification-v2", "data": {"kind": "csv", "feature_columns": "f1"}},
            {"profile": "classification-v2", "data": {"kind": "image_folder", "image_size": [32]}},
            {"steps": None},
            {"steps": 10**400},
            {"lr": 10**400},
            {"profile": "classification-v2", "data": {"kind": "csv"}, "epochs": 10**400},
            {"profile": "classification-v2", "data": {"kind": "csv"}, "lr": 10**400},
            {"profile": "classification-v2", "data": {"kind": "csv", "validation_fraction": 10**400}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workspace.json"
            for experiment in malformed:
                payload = {"workspace_version": 2, "model": Document().graph, "experiment": experiment}
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                    read_workspace(path)
            editable = {"profile": "classification-v2", "data": {"kind": "csv", "path": "", "feature_columns": []}}
            save_workspace(path, Document(), editable)
            self.assertEqual(read_workspace(path)[1], editable)

    def test_csv_starter_has_one_real_input_and_matching_classes(self):
        from native.workspace_state import starter_graph
        manifest = {"kind": "csv", "input_shape": [None, 5], "class_names": ["a", "b", "c"]}
        document = Document(starter_graph(manifest))
        self.assertTrue(document.validation()["valid"], document.validation())
        self.assertEqual(document.graph["inputs"][0]["shape"], [None, 5])
        self.assertEqual(len(document.graph["inputs"]), 1)
        self.assertEqual(document.validation()["shapes"]["call:logits"], [None, 3])

    def test_data_compatibility_reports_two_input_model_and_wrong_classes(self):
        from native.workspace_state import compatibility_issue, starter_graph
        manifest = {"kind": "csv", "input_shape": [None, 5], "class_names": ["a", "b", "c"]}
        self.assertIn("一个", compatibility_issue(Document(), manifest, "x"))
        document = Document(starter_graph(manifest))
        self.assertEqual(compatibility_issue(document, manifest, "x"), "")
        document.set_param("logits", "out_features", 2)
        self.assertIn("logits", compatibility_issue(document, manifest, "x"))


if __name__ == "__main__":
    unittest.main()
