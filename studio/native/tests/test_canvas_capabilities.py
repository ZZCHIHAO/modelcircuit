"""D0 checks use real DPG items; no desktop input or selected-state doubles."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg


class CanvasCapabilitiesTests(unittest.TestCase):
    def diagnostic_type(self):
        # A missing diagnostic is an expected behavior failure in the RED run.
        try:
            module = importlib.import_module("native.canvas_capabilities")
        except ModuleNotFoundError:
            self.fail("The independent real-DPG canvas diagnostic is not implemented")
        return module.CanvasCapabilitiesDiagnostic

    def test_routes_real_ports_and_rejects_cross_editor_links_and_delinks(self):
        with self.diagnostic_type()() as diagnostic:
            root, chip = diagnostic.views["root"], diagnostic.views["chip"]
            self.assertNotEqual(root["nodes"]["call:shared"], chip["nodes"]["call:shared"])
            for view in (root, chip):
                self.assertIsInstance(view["source"], int, "Native link callbacks carry numeric attribute IDs")
                self.assertIsInstance(view["target"], int)
                try:
                    dpg.get_item_callback(view["editor"])(view["editor"], [view["source"], view["target"]],
                                                         dpg.get_item_user_data(view["editor"]))
                except ValueError as error:
                    self.fail("A registered editor alias sender must resolve to its native owner: " + str(error))
                diagnostic.invoke(view["editor"], [view["source"], view["target"]])
                self.assertEqual(len(view["links"]), 1)
                link = next(iter(view["links"]))
                parent = dpg.get_item_info(link)["parent"]
                parent_id = dpg.get_alias_id(parent) if isinstance(parent, str) else parent
                self.assertEqual(parent_id, dpg.get_alias_id(view["editor"]))
            root_link, chip_link = next(iter(root["links"])), next(iter(chip["links"]))
            with self.assertRaises(ValueError):
                diagnostic.invoke(root["editor"], [root["source"], chip["target"]])
            with self.assertRaises(ValueError):
                diagnostic.invoke(root["editor"], [root["target"], root["source"]])
            with self.assertRaises(ValueError):
                diagnostic.invoke(root["editor"], chip_link, configuration_key="delink_callback")
            self.assertTrue(dpg.does_item_exist(chip_link))
            diagnostic.invoke(root["editor"], root_link, configuration_key="delink_callback")
            self.assertFalse(dpg.does_item_exist(root_link))
            self.assertTrue(dpg.does_item_exist(chip_link))

    def test_logical_selection_preserves_other_view_and_does_not_fake_native_selection(self):
        with self.diagnostic_type()() as diagnostic:
            diagnostic.select("root", ["call:shared"], intentional=True)
            diagnostic.select("chip", ["call:next"], intentional=True)
            diagnostic.select("root", [], intentional=False)
            self.assertEqual(diagnostic.views["root"]["selection"], ["call:shared"])
            self.assertEqual(diagnostic.views["chip"]["selection"], ["call:next"])
            # Public getters are called directly; no injected selected IDs.
            self.assertEqual(dpg.get_selected_nodes(diagnostic.views["root"]["editor"]), [])
            self.assertEqual(dpg.get_selected_nodes(diagnostic.views["chip"]["editor"]), [])
            diagnostic.invoke("d0_root_clear")
            self.assertEqual(diagnostic.views["root"]["selection"], [])
            self.assertEqual(diagnostic.views["chip"]["selection"], ["call:next"])
            self.assertEqual(dpg.get_selected_links(diagnostic.views["root"]["editor"]), [])
            with self.assertRaises(ValueError):
                diagnostic.select("root", ["call:missing"], intentional=True)

    def test_native_editor_node_port_link_ids_survive_hiding_splits_and_resize(self):
        with self.diagnostic_type()() as diagnostic:
            for view in diagnostic.views.values():
                diagnostic.invoke(view["editor"], [view["source"], view["target"]])
            before = diagnostic.identities()
            self.assertGreater(before["root"]["editor"], 0)
            self.assertNotEqual(before["root"]["nodes"], before["chip"]["nodes"])
            for layout in ("single", "columns", "rows", "single"):
                diagnostic.invoke("d0_layout_" + layout)
                diagnostic.frames()
                self.assertEqual(diagnostic.identities(), before)
            diagnostic.resize(1100, 780)
            self.assertEqual(diagnostic.identities(), before)
            self.assertFalse(dpg.is_item_shown(diagnostic.views["chip"]["editor"]))
            diagnostic.invoke("d0_layout_columns")
            diagnostic.frames()
            self.assertTrue(dpg.is_item_shown(diagnostic.views["chip"]["editor"]))
            self.assertEqual(diagnostic.identities(), before)
            self.assertTrue(dpg.get_item_configuration(diagnostic.views["root"]["editor"])["minimap"])

    def test_position_commit_and_navigation_callbacks_keep_model_refs_scoped(self):
        with self.diagnostic_type()() as diagnostic:
            self.assertTrue(hasattr(diagnostic, "document"), "Position callbacks need a real Document fixture")
            semantic_before = diagnostic.document.semantic_key()
            revision_before = diagnostic.document.graph["revision"]
            before = dict(diagnostic.views["root"]["positions"])
            diagnostic.invoke("d0_layout_columns")
            dpg.set_item_pos(diagnostic.views["chip"]["nodes"]["call:shared"], (87, 134))
            diagnostic.invoke("d0_chip_commit")
            self.assertEqual(diagnostic.views["chip"]["positions"]["call:shared"], [87, 134])
            self.assertEqual(diagnostic.views["root"]["positions"], before)
            self.assertEqual(diagnostic.document.graph["calls"][0]["position"], {"x": 30, "y": 40})
            self.assertEqual(diagnostic.document.graph["definitions"][0]["graph"]["calls"][0]["position"], {"x": 87, "y": 134})
            self.assertEqual(diagnostic.document.semantic_key(), semantic_before)
            self.assertEqual(diagnostic.document.graph["revision"], revision_before)
            self.assertEqual(diagnostic.document.trail, [])
            diagnostic.invoke("d0_chip_open")
            self.assertEqual(diagnostic.opened, {"view_id": "chip", "ref": "call:shared"})
            self.assertEqual(diagnostic.views["root"]["positions"], before)

    def test_report_has_real_framebuffers_raw_api_evidence_and_cleanup(self):
        diagnostic_type = self.diagnostic_type()
        with tempfile.TemporaryDirectory() as temporary:
            diagnostic = diagnostic_type()
            with diagnostic:
                report = diagnostic.run(Path(temporary))
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["identity_before"], report["identity_after"])
                self.assertEqual(report["capabilities"]["exact_pan_zoom"], "unsupported-public-api")
                self.assertIn("add_node_editor", report["api"]["signatures"])
                self.assertIn("get_selected_nodes", report["api"]["signatures"])
                self.assertIn("real_mouse_drag", report["unverified"])
                for name in ("single.png", "columns.png", "rows.png"):
                    data = (Path(temporary) / name).read_bytes()
                    self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertGreater(len(data), 1000)
            self.assertTrue(diagnostic.context_destroyed)
            raw = json.loads((Path(temporary) / "canvas-capabilities.json").read_text(encoding="utf-8"))
            self.assertTrue(raw["lifecycle"]["context_destroyed"])
            self.assertTrue((Path(temporary) / "canvas-capabilities.log").is_file())

    def test_destroy_failure_records_failure_and_preserves_the_original_exception(self):
        real_destroy = dpg.destroy_context
        for body_fails, report_write_fails in ((False, False), (True, False), (False, True)):
            with self.subTest(body_fails=body_fails, report_write_fails=report_write_fails):
                with tempfile.TemporaryDirectory() as temporary:
                    diagnostic = self.diagnostic_type()()
                    cleanup_error = RuntimeError("injected destroy failure")
                    body_error = ValueError("original body failure") if body_fails else None
                    expected_error = body_error or cleanup_error
                    try:
                        # Only this public function is replaced. Real DPG editors,
                        # viewport, callbacks and framebuffers run normally.
                        with self.assertRaises(type(expected_error)) as caught:
                            with patch.object(dpg, "destroy_context", side_effect=cleanup_error):
                                with diagnostic:
                                    diagnostic.run(Path(temporary))
                                    if report_write_fails:
                                        invalid_output = Path(temporary) / "not-a-directory"
                                        invalid_output.write_text("file", encoding="utf-8")
                                        diagnostic.output = invalid_output
                                    if body_error is not None:
                                        raise body_error
                        self.assertIs(caught.exception, expected_error)
                        self.assertFalse(diagnostic.context_destroyed)
                        self.assertFalse(diagnostic.report["lifecycle"]["context_destroyed"])
                        self.assertEqual(diagnostic.report["status"], "failed")
                        self.assertIn("injected destroy failure", diagnostic.report["lifecycle"]["cleanup_error"])
                        self.assertNotIn("context_destroyed", [event["event"] for event in diagnostic.events])
                        if body_error is not None:
                            self.assertTrue(any("DPG cleanup failed" in note for note in body_error.__notes__))
                        if report_write_fails:
                            self.assertTrue(any("report" in note for note in cleanup_error.__notes__))
                        else:
                            raw = json.loads((Path(temporary) / "canvas-capabilities.json").read_text(encoding="utf-8"))
                            self.assertFalse(raw["lifecycle"]["context_destroyed"])
                            self.assertEqual(raw["status"], "failed")
                            self.assertEqual(raw["events"][-1]["event"], "context_destroy_failed")
                    finally:
                        # The injected failure leaves the real context alive.
                        # Always release it with the original public DPG function.
                        real_destroy()


if __name__ == "__main__":
    unittest.main()
