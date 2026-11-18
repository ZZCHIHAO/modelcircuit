"""Bounded S4 D0 experiment, independent of the production workbench.

Only public Dear PyGui APIs are used. Callback invocation checks application
routing, not the delivery or semantics of physical mouse gestures.
"""
import argparse
from copy import deepcopy
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import shutil
import tempfile

import dearpygui.dearpygui as dpg

from .document import Document


def _document_fixture():
    scope = {"inputs": [{"id": "x", "shape": [None, 2], "dtype": "float32"}],
             "instances": [{"id": "one", "kind": "ReLU", "params": {}},
                           {"id": "two", "kind": "ReLU", "params": {}}],
             "calls": [{"id": "shared", "instance_id": "one", "inputs": {"x": "input:x"}, "position": {"x": 30, "y": 40}},
                       {"id": "next", "instance_id": "two", "inputs": {"x": "call:shared"}, "position": {"x": 250, "y": 40}}],
             "outputs": [{"id": "out", "source": "call:next"}]}
    graph = deepcopy(scope)
    graph.update({"schema_version": 1, "id": "d0_canvas_fixture", "revision": 1,
                  "definitions": [{"id": "d0_chip", "version": "1.0.0", "public_parameters": [],
                                   "bindings": {}, "graph": deepcopy(scope)}]})
    graph["instances"].append({"id": "chip_instance", "kind": "chip", "definition_id": "d0_chip", "params": {}})
    graph["calls"].append({"id": "open_chip", "instance_id": "chip_instance", "inputs": {"x": "input:x"}})
    return Document(graph)


class CanvasCapabilitiesDiagnostic:
    """Own one real DPG context/viewport and two persistent node editors."""

    def __init__(self):
        self.document = _document_fixture()
        self.views = {}
        self.events = []
        self.opened = None
        self.layout = "single"
        self.context_destroyed = False
        self.report = None
        self.output = None

    def __enter__(self):
        dpg.create_context()
        self.events.append({"event": "context_created"})
        try:
            dpg.configure_app(manual_callback_management=True)
            dpg.create_viewport(title="ML Forge - S4 D0 canvas diagnostic", width=1240, height=860)
            with dpg.window(tag="d0_main", no_scrollbar=True):
                dpg.add_text("S4 D0 | real DPG editors | application callbacks only")
                with dpg.group(horizontal=True):
                    for layout in ("single", "columns", "rows"):
                        dpg.add_button(label=layout, tag="d0_layout_" + layout,
                                       callback=self._layout_callback, user_data=layout)
                for view_id in ("root", "chip"):
                    with dpg.child_window(tag="d0_" + view_id + "_pane", no_scrollbar=True):
                        dpg.add_text(view_id + " scope | same refs, distinct native IDs")
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Commit node positions", tag="d0_" + view_id + "_commit",
                                           callback=self._commit_positions, user_data=view_id)
                            dpg.add_button(label="Clear logical + native selection", tag="d0_" + view_id + "_clear",
                                           callback=self._clear, user_data=view_id)
                        self._build_editor(view_id)
            with dpg.theme() as selected_theme:
                with dpg.theme_component(dpg.mvNode):
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (65, 116, 160), category=dpg.mvThemeCat_Nodes)
            self.selected_theme = selected_theme
            dpg.setup_dearpygui()
            dpg.set_primary_window("d0_main", True)
            dpg.set_viewport_resize_callback(self._resized)
            dpg.show_viewport()
            self.set_layout("single")
            self.frames()
            return self
        except BaseException:
            dpg.destroy_context()
            self.context_destroyed = True
            raise

    def __exit__(self, exception_type, exception, traceback):
        cleanup_error = None
        try:
            dpg.destroy_context()
        except BaseException as error:
            cleanup_error = error
            self.context_destroyed = False
            self.events.append({"event": "context_destroy_failed", "error": type(error).__name__ + ": " + str(error)})
        else:
            self.context_destroyed = True
            self.events.append({"event": "context_destroyed"})
        if self.report is not None:
            self.report["lifecycle"]["context_destroyed"] = self.context_destroyed
            if cleanup_error is not None:
                self.report["status"] = "failed"
                self.report["lifecycle"]["cleanup_error"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
            self.report["events"] = list(self.events)
            try:
                self._write_report()
            except BaseException as report_error:
                original = exception if exception is not None else cleanup_error
                if original is None:
                    raise
                original.add_note("Failed to write cleanup report: " + type(report_error).__name__ + ": " + str(report_error))
        if cleanup_error is not None:
            if exception is not None:
                exception.add_note("DPG cleanup failed: " + type(cleanup_error).__name__ + ": " + str(cleanup_error))
                return False
            raise cleanup_error
        return False

    def _build_editor(self, view_id):
        editor = "d0_" + view_id + "_editor"
        view = {"editor": editor, "nodes": {}, "node_tags": {}, "links": {},
                "selection": [], "positions": {"call:shared": [30, 40], "call:next": [250, 40]}}
        self.views[view_id] = view
        with dpg.node_editor(tag=editor, minimap=True, callback=self._connect,
                             delink_callback=self._disconnect, user_data=view_id):
            for index, ref in enumerate(("call:shared", "call:next")):
                node_tag = "d0_" + view_id + "_node_" + str(index)
                with dpg.node(label=ref, tag=node_tag, pos=view["positions"][ref]) as node:
                    if index == 0:
                        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                            dpg.add_button(label="Open shared reference", tag="d0_" + view_id + "_open",
                                           callback=self._open, user_data={"view_id": view_id, "ref": ref})
                        with dpg.node_attribute(tag="d0_" + view_id + "_source", attribute_type=dpg.mvNode_Attr_Output) as attribute:
                            dpg.add_text("output")
                        view["source"] = dpg.get_alias_id(attribute)
                    else:
                        with dpg.node_attribute(tag="d0_" + view_id + "_target", attribute_type=dpg.mvNode_Attr_Input) as attribute:
                            dpg.add_text("input")
                        view["target"] = dpg.get_alias_id(attribute)
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                        dpg.add_text("scope=" + view_id)
                view["nodes"][ref] = dpg.get_alias_id(node_tag)
                view["node_tags"][ref] = node_tag

    def invoke(self, tag, app_data=None, *, configuration_key="callback"):
        """Invoke the callback registered on a real item, with its real userdata."""
        callback = (dpg.get_item_callback(tag) if configuration_key == "callback"
                    else dpg.get_item_configuration(tag)[configuration_key])
        if callback is None:
            raise ValueError("No registered callback: " + str(tag))
        sender = dpg.get_alias_id(tag) if isinstance(tag, str) else tag
        self.events.append({"event": "invoke_registered_callback", "sender": sender,
                            "kind": configuration_key, "app_data": app_data})
        callback(sender, app_data, dpg.get_item_user_data(tag))

    def _owned_view(self, sender, view_id):
        view = self.views[view_id]
        sender_id = dpg.get_alias_id(sender) if isinstance(sender, str) else sender
        if sender_id != dpg.get_alias_id(view["editor"]):
            raise ValueError("Sender does not own this editor")
        return view

    def _connect(self, sender, app_data, user_data):
        view = self._owned_view(sender, user_data)
        source, target = app_data
        if source != view["source"] or target != view["target"]:
            self.events.append({"event": "link_rejected", "view_id": user_data,
                                "source": source, "target": target})
            raise ValueError("Ports must have matching direction and belong to this editor")
        if (source, target) in view["links"].values():
            return
        link = dpg.add_node_link(source, target, parent=view["editor"])
        view["links"][link] = (source, target)
        self.events.append({"event": "link_created", "view_id": user_data, "native_id": link})

    def _disconnect(self, sender, app_data, user_data):
        view = self._owned_view(sender, user_data)
        if app_data not in view["links"]:
            self.events.append({"event": "delink_rejected", "view_id": user_data, "native_id": app_data})
            raise ValueError("Link belongs to another editor or no longer exists")
        dpg.delete_item(app_data)
        del view["links"][app_data]
        self.events.append({"event": "link_deleted", "view_id": user_data, "native_id": app_data})

    def select(self, view_id, refs, *, intentional):
        view = self.views[view_id]
        if any(ref not in view["nodes"] for ref in refs):
            raise ValueError("Unknown model reference")
        if not refs and not intentional:
            return
        view["selection"] = list(refs)
        for ref, node in view["nodes"].items():
            dpg.bind_item_theme(node, self.selected_theme if ref in refs else 0)
        self.events.append({"event": "logical_selection", "view_id": view_id, "refs": list(refs)})

    def _clear(self, sender, app_data, user_data):
        self.select(user_data, [], intentional=True)
        dpg.clear_selected_nodes(self.views[user_data]["editor"])
        dpg.clear_selected_links(self.views[user_data]["editor"])

    def _commit_positions(self, sender, app_data, user_data):
        view = self.views[user_data]
        previous_path = [entry["call_id"] for entry in self.document.trail]
        previous_selection = self.document.selected
        try:
            self.document.root()
            if user_data == "chip":
                self.document.enter("open_chip")
            for ref, node in view["nodes"].items():
                position = list(dpg.get_item_pos(node))
                self.document.move(ref, *position)
                view["positions"][ref] = position
        finally:
            self.document.root()
            for call_id in previous_path:
                self.document.enter(call_id)
            self.document.select(previous_selection)
        self.events.append({"event": "positions_committed", "view_id": user_data,
                            "positions": dict(view["positions"])})

    def _open(self, sender, app_data, user_data):
        self.opened = dict(user_data)
        self.events.append({"event": "reference_opened", **self.opened})

    def _layout_callback(self, sender, app_data, user_data):
        self.set_layout(user_data)

    def _resized(self, sender, app_data, user_data):
        self.set_layout(self.layout)
        self.events.append({"event": "viewport_resize_callback",
                            "client_size": [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()]})

    def set_layout(self, mode):
        if mode not in ("single", "columns", "rows"):
            raise ValueError("Unsupported layout")
        self.layout = mode
        width, height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        width, height = max(500, width - 20), max(300, height - 80)
        sizes = [(width, height), (width, height)]
        positions = [(8, 65), (8, 65)]
        if mode == "columns":
            sizes = [((width - 8) // 2, height)] * 2
            positions[1] = (16 + sizes[0][0], 65)
        elif mode == "rows":
            sizes = [(width, (height - 8) // 2)] * 2
            positions[1] = (8, 73 + sizes[0][1])
        for index, view_id in enumerate(("root", "chip")):
            shown = index == 0 or mode != "single"
            pane = "d0_" + view_id + "_pane"
            w, h = sizes[index]
            dpg.configure_item(pane, pos=positions[index], width=w, height=h, show=shown)
            dpg.configure_item(self.views[view_id]["editor"], width=w - 18, height=h - 65, show=shown)
        self.events.append({"event": "layout", "mode": mode, "client_size": [width + 20, height + 80]})

    def resize(self, width, height):
        dpg.set_viewport_width(width)
        dpg.set_viewport_height(height)
        self.frames()
        self.set_layout(self.layout)
        self.frames()

    def frames(self, count=6):
        for _ in range(count):
            dpg.run_callbacks(dpg.get_callback_queue())
            dpg.render_dearpygui_frame()

    def identities(self):
        return {view_id: {"editor": dpg.get_alias_id(view["editor"]),
                          "nodes": {ref: dpg.get_alias_id(tag) for ref, tag in view["node_tags"].items()},
                          "ports": [dpg.get_alias_id("d0_" + view_id + "_source"),
                                    dpg.get_alias_id("d0_" + view_id + "_target")],
                          "links": sorted(view["links"])}
                for view_id, view in self.views.items()}

    def _capture(self, target):
        # DPG's encoder uses ASCII temp paths; copy its untouched framebuffer.
        with tempfile.TemporaryDirectory(prefix="mlforge-d0-") as temporary:
            source = Path(temporary) / "frame.png"
            self.frames()
            dpg.output_frame_buffer(str(source))
            for _ in range(60):
                self.frames(1)
                if source.is_file() and source.stat().st_size:
                    shutil.copy2(source, target)
                    return
        raise RuntimeError("Framebuffer export did not complete")

    def _api_evidence(self):
        names = ("add_node_editor", "add_node", "add_node_attribute", "add_node_link",
                 "get_selected_nodes", "get_selected_links", "clear_selected_nodes", "clear_selected_links",
                 "get_item_pos", "set_item_pos", "get_alias_id", "get_item_info",
                 "get_item_configuration", "configure_item", "set_viewport_resize_callback", "output_frame_buffer")
        public = sorted(name for name in dir(dpg) if not name.startswith("_"))
        configuration = dpg.get_item_configuration(self.views["root"]["editor"])
        return {"version": version("dearpygui"),
                "signatures": {name: str(inspect.signature(getattr(dpg, name))) for name in names},
                "public_names_containing_pan_zoom_select": [name for name in public if any(word in name.lower() for word in ("pan", "zoom", "select"))],
                "editor_configuration_keys": sorted(configuration),
                "editor_minimap": configuration["minimap"],
                "pan_zoom_configuration_keys": [name for name in configuration if any(word in name.lower() for word in ("pan", "zoom"))]}

    def run(self, output_directory):
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        for view in self.views.values():
            self.invoke(view["editor"], [view["source"], view["target"]])
        before = self.identities()
        root, chip = self.views["root"], self.views["chip"]
        root_link, chip_link = next(iter(root["links"])), next(iter(chip["links"]))
        rejected = []
        for payload, kind in (([root["source"], chip["target"]], "callback"),
                              ([root["target"], root["source"]], "callback"),
                              (chip_link, "delink_callback")):
            try:
                self.invoke(root["editor"], payload, configuration_key=kind)
            except ValueError as error:
                rejected.append(str(error))
            else:
                raise AssertionError("Invalid cross-editor/direction callback was accepted")
        assert len(root["links"]) == len(chip["links"]) == 1
        self.select("root", ["call:shared"], intentional=True)
        self.select("chip", ["call:next"], intentional=True)
        self.select("root", [], intentional=False)
        selection = {key: {"logical": list(view["selection"]),
                           "native_nodes": dpg.get_selected_nodes(view["editor"]),
                           "native_links": dpg.get_selected_links(view["editor"])}
                     for key, view in self.views.items()}
        assert selection["root"]["logical"] == ["call:shared"]
        self.invoke("d0_root_clear")
        assert chip["selection"] == ["call:next"] and root["selection"] == []
        native_positions_before = {key: {ref: list(dpg.get_item_pos(node)) for ref, node in view["nodes"].items()}
                                   for key, view in self.views.items()}
        root_positions = dict(root["positions"])
        semantic_before = self.document.semantic_key()
        revision_before = self.document.graph["revision"]
        dpg.set_item_pos(chip["nodes"]["call:shared"], (87, 134))
        self.invoke("d0_chip_commit")
        assert chip["positions"]["call:shared"] == [87, 134]
        assert root["positions"] == root_positions
        assert self.document.graph["calls"][0]["position"] == {"x": 30, "y": 40}
        assert self.document.graph["definitions"][0]["graph"]["calls"][0]["position"] == {"x": 87, "y": 134}
        assert self.document.semantic_key() == semantic_before
        assert self.document.graph["revision"] == revision_before
        self.invoke("d0_chip_open")
        assert self.opened == {"view_id": "chip", "ref": "call:shared"}
        stages = []
        for mode in ("single", "columns", "rows"):
            self.invoke("d0_layout_" + mode)
            self.frames()
            assert self.identities() == before
            stages.append({"mode": mode, "identities": self.identities(),
                           "editor_rects": {key: {"size": list(dpg.get_item_rect_size(view["editor"])),
                                                   "shown": dpg.is_item_shown(view["editor"])}
                                            for key, view in self.views.items()}})
            self._capture(self.output / (mode + ".png"))
        self.resize(1100, 780)
        after = self.identities()
        assert before == after
        for key, view in self.views.items():
            for ref, node in view["nodes"].items():
                expected = [87, 134] if key == "chip" and ref == "call:shared" else native_positions_before[key][ref]
                assert list(dpg.get_item_pos(node)) == expected
        raw_link_parent = {key: dpg.get_item_info(next(iter(view["links"])))["parent"] for key, view in self.views.items()}
        link_parent = {key: dpg.get_alias_id(parent) if isinstance(parent, str) else parent
                       for key, parent in raw_link_parent.items()}
        assert link_parent == {key: value["editor"] for key, value in before.items()}
        self.invoke(root["editor"], root_link, configuration_key="delink_callback")
        assert not dpg.does_item_exist(root_link) and dpg.does_item_exist(chip_link)
        self.report = {"status": "passed", "kind": "bounded-real-dpg-canvas-diagnostic",
                       "api": self._api_evidence(), "identity_before": before, "identity_after": after,
                       "identity_comparison_before_delink": True, "link_native_parents": link_parent,
                       "link_raw_parent_values": raw_link_parent,
                       "rejected_callbacks": rejected, "selection_observation": selection,
                       "layout_stages": stages, "positions_after_callback": {key: view["positions"] for key, view in self.views.items()},
                       "opened_reference": self.opened,
                       "document_layout_evidence": {"root_shared": self.document.graph["calls"][0]["position"],
                                                    "chip_shared": self.document.graph["definitions"][0]["graph"]["calls"][0]["position"],
                                                    "semantic_key_unchanged": self.document.semantic_key() == semantic_before,
                                                    "revision_unchanged": self.document.graph["revision"] == revision_before,
                                                    "cursor_restored": self.document.trail == []},
                       "capabilities": {"two_live_editors": "verified", "scoped_link_callbacks": "verified",
                                        "native_identity_retention": "verified", "logical_selection_isolation": "verified",
                                        "native_selection_get_clear": "verified-empty-state-only",
                                        "native_selection_restore": "unsupported-public-api",
                                        "exact_pan_zoom": "unsupported-public-api", "minimap": "configured-and-rendered"},
                       "unverified": ["real_mouse_drag", "real_mouse_link_creation", "positive_native_selection",
                                      "real_double_click_delivery_and_hit_testing", "minimap_mouse_navigation",
                                      "precise_native_pan_zoom_retention", "production_CanvasHost_integration"],
                       "screenshots": ["single.png", "columns.png", "rows.png"],
                       "lifecycle": {"context_created": True, "viewport_shown": True, "context_destroyed": False},
                       "events": list(self.events)}
        self._write_report()
        return self.report

    def _write_report(self):
        (self.output / "canvas-capabilities.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output / "canvas-capabilities.log").write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in self.events) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/native-s4-d0/canvas"))
    arguments = parser.parse_args()
    with CanvasCapabilitiesDiagnostic() as diagnostic:
        report = diagnostic.run(arguments.output)
    print(json.dumps({"status": report["status"], "output": str(arguments.output),
                      "context_destroyed": diagnostic.context_destroyed}))


if __name__ == "__main__":
    main()
