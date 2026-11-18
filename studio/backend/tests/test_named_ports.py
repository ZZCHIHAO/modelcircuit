"""Named chip outputs retain graph, tensor and static Python semantics."""
import copy
import unittest

import torch

from studio_core import python_io, runtime, schema


def named_graph():
    image = {"id": "image", "shape": [None, 2, 3, 3], "dtype": "float32"}
    table = {"id": "table", "shape": [None, 3], "dtype": "float32"}
    inner = {"id": "inner", "output_mode": "named", "public_parameters": [], "bindings": {}, "graph": {
        "inputs": [image, table],
        "instances": [{"id": "conv", "kind": "Conv2d", "params": {"in_channels": 2, "out_channels": 2, "kernel_size": 1, "bias": False}},
                      {"id": "average", "kind": "AdaptiveAvgPool2d", "params": {"output_size": [1, 1]}},
                      {"id": "flat", "kind": "Flatten", "params": {}},
                      {"id": "clinical", "kind": "Linear", "params": {"in_features": 3, "out_features": 2}},
                      {"id": "fusion", "kind": "Add", "params": {}}],
        "calls": [{"id": "convolve", "instance_id": "conv", "inputs": {"x": "input:image"}},
                  {"id": "pool", "instance_id": "average", "inputs": {"x": "call:convolve:out"}},
                  {"id": "flatten", "instance_id": "flat", "inputs": {"x": "call:pool"}},
                  {"id": "project", "instance_id": "clinical", "inputs": {"x": "input:table"}},
                  {"id": "fuse", "instance_id": "fusion", "inputs": {"a": "call:flatten", "b": "call:project"}}],
        "outputs": [{"id": "features", "source": "call:fuse"}, {"id": "clinical", "source": "call:project:out"}]}}
    outer = {"id": "outer", "output_mode": "named", "public_parameters": [], "bindings": {}, "graph": {
        "inputs": [copy.deepcopy(image), copy.deepcopy(table)],
        "instances": [{"id": "inner_block", "kind": "chip", "definition_id": "inner", "params": {}}],
        "calls": [{"id": "encode", "instance_id": "inner_block", "inputs": {"image": "input:image", "table": "input:table"}}],
        "outputs": [{"id": "main", "source": "call:encode:features"}, {"id": "aux", "source": "call:encode:clinical"}]}}
    inputs = [{**copy.deepcopy(port), "id": port["id"] + suffix} for suffix in ("_a", "_b") for port in (image, table)]
    return {"schema_version": 1, "id": "named_shared", "revision": 1, "resource_profile": "classification-v2", "inputs": inputs,
            "instances": [{"id": "encoder", "kind": "chip", "definition_id": "outer", "params": {}},
                          {"id": "sum_op", "kind": "Add", "params": {}},
                          {"id": "head", "kind": "Linear", "params": {"in_features": 2, "out_features": 2}}],
            "calls": [{"id": "left", "instance_id": "encoder", "inputs": {"image": "input:image_a", "table": "input:table_a"}},
                      {"id": "right", "instance_id": "encoder", "inputs": {"image": "input:image_b", "table": "input:table_b"}},
                      {"id": "sum_features", "instance_id": "sum_op", "inputs": {"a": "call:left:main", "b": "call:right:main"}},
                      {"id": "sum_aux", "instance_id": "sum_op", "inputs": {"a": "call:left:aux", "b": "call:right:aux"}},
                      {"id": "classify", "instance_id": "head", "inputs": {"x": "call:sum_features:out"}}],
            "outputs": [{"id": "logits", "source": "call:classify"}, {"id": "aux", "source": "call:sum_aux"}], "definitions": [inner, outer]}


class NamedPortSchemaTests(unittest.TestCase):
    def test_public_reference_helpers_parse_exactly_and_accept_definition_list_or_registry(self):
        from studio_core.ports import parse_source, output_port_ids
        self.assertEqual(parse_source("input:image"), ("input", "image", None))
        self.assertEqual(parse_source("call:left:main"), ("call", "left", "main"))
        self.assertEqual(parse_source("call:left"), ("call", "left", None))
        for source in (None, "input:x:out", "call:x:out:bad", "call:x:", "output:x", "call:bad-id"):
            with self.subTest(source=source), self.assertRaises(ValueError): parse_source(source)
        graph = named_graph(); instance = graph["instances"][0]
        self.assertEqual(output_port_ids(instance, graph["definitions"]), ["main", "aux"])
        self.assertEqual(output_port_ids(instance, {d["id"]: d for d in graph["definitions"]}), ["main", "aux"])
        self.assertEqual(output_port_ids(graph["instances"][1], []), ["out"])

    def test_shapes_and_unique_parameters_cover_all_qualified_outputs(self):
        report = schema.validate_graph(named_graph())
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["parameter_count"], 18)
        self.assertEqual(report["shapes"]["call:left:main"], [None, 2])
        self.assertEqual(report["shapes"]["call:left:aux"], [None, 2])
        self.assertNotIn("call:left", report["shapes"])
        self.assertEqual(report["shapes"]["call:classify:out"], report["shapes"]["call:classify"])

    def test_unknown_ambiguous_ports_modes_and_port_cycles_fail_with_diagnostics(self):
        for source, message in (("call:left", "ambiguous"), ("call:left:missing", "port"), ("call:classify:wrong", "port"), ("call:classify:out", "cycle")):
            graph = named_graph(); graph["calls"][2]["inputs"]["a"] = source
            with self.subTest(source=source):
                result = schema.validate_graph(graph)
                self.assertFalse(result["valid"])
                self.assertIn(message, result["errors"][0]["message"].lower())
        for mode in ("tensor", "invalid", None):
            graph = named_graph(); graph["definitions"][0]["output_mode"] = mode
            with self.subTest(mode=mode): self.assertFalse(schema.validate_graph(graph)["valid"])

    def test_multi_input_public_in_features_never_changes_first_input_shape(self):
        graph = named_graph()
        inner = graph["definitions"][0]
        inner["public_parameters"] = [{"id": "in_features", "default": 3}]
        inner["bindings"] = {"in_features": ["clinical.in_features"]}
        before = copy.deepcopy(graph)
        self.assertTrue(schema.validate_graph(graph)["valid"])
        self.assertEqual(graph, before)

    def test_output_reordering_keeps_connections_and_inferred_named_mode(self):
        graph = named_graph()
        for definition in graph["definitions"]:
            definition.pop("output_mode")
            definition["graph"]["outputs"].reverse()
        report = schema.validate_graph(graph)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["shapes"]["call:left:main"], [None, 2])


class NamedPortRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_shared_nested_forward_gradients_and_update_match_independent_reference(self):
        model = runtime.build_model(named_graph(), capture=False)
        conv = torch.nn.Conv2d(2, 2, 1, bias=False); clinical = torch.nn.Linear(3, 2); head = torch.nn.Linear(2, 2)
        internal = model.components["encoder"].components["inner_block"]
        conv.load_state_dict(internal.components["conv"].state_dict())
        clinical.load_state_dict(internal.components["clinical"].state_dict()); head.load_state_dict(model.components["head"].state_dict())
        values = {"image_a": torch.linspace(-1, 1, 54).reshape(3, 2, 3, 3), "image_b": torch.linspace(1, -0.6, 54).reshape(3, 2, 3, 3),
                  "table_a": torch.linspace(-0.5, 0.8, 9).reshape(3, 3), "table_b": torch.linspace(1.1, -0.8, 9).reshape(3, 3)}
        actual_inputs = {key: value.clone().requires_grad_(True) for key, value in values.items()}
        reference_inputs = {key: value.clone().requires_grad_(True) for key, value in values.items()}
        def encode(image, table):
            tabular = clinical(table)
            return conv(image).mean((2, 3)) + tabular, tabular
        left, left_aux = encode(reference_inputs["image_a"], reference_inputs["table_a"])
        right, right_aux = encode(reference_inputs["image_b"], reference_inputs["table_b"])
        expected = {"logits": head(left + right), "aux": left_aux + right_aux}
        actual = model(**actual_inputs)
        labels = torch.tensor([0, 1, 0])
        for key in expected: torch.testing.assert_close(actual[key], expected[key])
        loss = lambda outputs: torch.nn.functional.cross_entropy(outputs["logits"], labels) + 0.03 * outputs["aux"].square().mean()
        loss(actual).backward(); loss(expected).backward()
        for key in values: torch.testing.assert_close(actual_inputs[key].grad, reference_inputs[key].grad)
        reference_parameters = list(conv.parameters()) + list(clinical.parameters()) + list(head.parameters())
        for first, second in zip(model.parameters(), reference_parameters): torch.testing.assert_close(first.grad, second.grad)
        torch.optim.SGD(model.parameters(), lr=0.04).step(); torch.optim.SGD(reference_parameters, lr=0.04).step()
        for first, second in zip(model.parameters(), reference_parameters): torch.testing.assert_close(first, second)
        self.assertEqual(len(list(model.parameters())), 5)
        self.assertEqual(runtime._shared_instances(model)[0]["call_ids"], ["left", "right"])
        self.assertEqual(model.last_values, {})
        model.set_capture(True); model(**actual_inputs)
        self.assertIn("call:left:main", model.last_values)
        self.assertIn("call:left/call:encode:features", model.last_values)
        self.assertIn("call:left/call:encode/call:convolve:out", model.last_values)

    def test_many_ports_aliasing_one_tensor_do_not_duplicate_budget_or_snapshot(self):
        ports = [{"id": "port_" + str(i), "source": "input:x"} for i in range(32)]
        graph = {"schema_version": 1, "id": "alias_budget", "revision": 1, "inputs": [{"id": "x", "shape": [None, 12500], "dtype": "float32"}],
                 "instances": [{"id": "chip", "kind": "chip", "definition_id": "aliases", "params": {}}],
                 "calls": [{"id": "bundle", "instance_id": "chip", "inputs": {"x": "input:x"}}],
                 "outputs": [{"id": "first", "source": "call:bundle:port_0"}, {"id": "last", "source": "call:bundle:port_31"}],
                 "definitions": [{"id": "aliases", "output_mode": "named", "graph": {"inputs": [{"id": "x", "shape": [None, 12500], "dtype": "float32"}], "instances": [], "calls": [], "outputs": ports}}]}
        model = runtime.build_model(graph, capture=False)
        actual = model(x=torch.zeros(64, 12500))
        self.assertIs(actual["first"], actual["last"])
        model.set_capture(True); model(x=torch.zeros(2, 12500))
        self.assertIs(model.last_values["call:bundle:port_0"], model.last_values["call:bundle:port_31"])

    def test_two_selected_ports_do_not_repeat_a_stochastic_chip_call(self):
        chip = {"id": "stochastic", "output_mode": "named", "graph": {
            "inputs": [{"id": "x", "shape": [None, 8], "dtype": "float32"}],
            "instances": [{"id": "drop", "kind": "Dropout", "params": {"p": 0.5}}],
            "calls": [{"id": "mask", "instance_id": "drop", "inputs": {"x": "input:x"}}],
            "outputs": [{"id": "left", "source": "call:mask"}, {"id": "right", "source": "call:mask"}]}}
        graph = {"schema_version": 1, "id": "drop_ports", "revision": 1, "inputs": copy.deepcopy(chip["graph"]["inputs"]),
                 "definitions": [chip], "instances": [{"id": "block", "kind": "chip", "definition_id": "stochastic", "params": {}}, {"id": "sum_op", "kind": "Add", "params": {}}],
                 "calls": [{"id": "sample", "instance_id": "block", "inputs": {"x": "input:x"}}, {"id": "sum_call", "instance_id": "sum_op", "inputs": {"a": "call:sample:left", "b": "call:sample:right"}}],
                 "outputs": [{"id": "out", "source": "call:sum_call"}]}
        model = runtime.build_model(graph, capture=False).train()
        namespace = {}; exec(python_io.export_python(graph), namespace)
        standalone = namespace["Model"]().train()
        x = torch.ones(4, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17); expected = 2 * torch.nn.functional.dropout(x, p=0.5, training=True)
            expected_rng = torch.get_rng_state().clone()
            for candidate in (model, standalone):
                torch.manual_seed(17)
                torch.testing.assert_close(candidate(x=x)["out"], expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_parent_output_capture_survives_selectively_disabled_child_capture(self):
        graph = named_graph(); model = runtime.build_model(graph, capture=True)
        model.components["encoder"].set_capture(False)
        inputs = {port["id"]: torch.ones(*[2 if dim is None else dim for dim in port["shape"]]) for port in graph["inputs"]}
        result = model(**inputs)
        self.assertEqual(tuple(result["logits"].shape), (2, 2))
        self.assertIn("call:left:main", model.last_values)
        self.assertEqual(model.components["encoder"].last_values, {})


class NamedPortBridgeTests(unittest.TestCase):
    def test_metadata_free_named_graph_preserves_hierarchy_sharing_and_selected_source(self):
        graph = named_graph(); source = python_io.export_python(graph)
        self.assertEqual(python_io.python_profile(graph), "ports-pytorch-v3")
        self.assertIn("ports-pytorch-v3", source)
        without = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
        restored = python_io.import_python(without)
        self.assertTrue(schema.validate_graph(restored)["valid"])
        self.assertEqual(restored["calls"][2]["inputs"], {"a": "call:left:main", "b": "call:right:main"})
        self.assertEqual([d["output_mode"] for d in restored["definitions"]], ["named", "named"])
        model = runtime.build_model(restored, capture=False)
        namespace = {}; exec(compile(source, "<trusted-export>", "exec"), namespace)
        standalone = namespace["Model"](); standalone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(standalone.state_dict()))
        inputs = {port["id"]: torch.ones(*[2 if dim is None else dim for dim in port["shape"]]) for port in graph["inputs"]}
        actual, expected = model(**inputs), standalone(**inputs)
        for key in actual: torch.testing.assert_close(actual[key], expected[key])
        actual["logits"].sum().backward(); expected["logits"].sum().backward()
        for first, second in zip(model.parameters(), standalone.parameters()): torch.testing.assert_close(first.grad, second.grad)

    def test_current_named_selector_and_alias_edits_override_stale_visual_metadata(self):
        source = python_io.export_python(named_graph())
        edited = source.replace("a=call_left['main']", "a=call_left['aux']")
        self.assertNotEqual(source, edited)
        restored = python_io.import_python(edited)
        self.assertEqual(restored["calls"][2]["inputs"]["a"], "call:left:aux")
        aliased = edited.replace("        call_sum_features =", "        selected = call_left['aux']\n        call_sum_features =").replace("a=call_left['aux']", "a=selected")
        self.assertEqual(python_io.import_python(aliased)["calls"][2]["inputs"]["a"], "call:left:aux")

    def test_single_named_dict_and_old_meaningful_output_id_survive_without_metadata(self):
        from studio_core.examples import example_graph
        for explicit_mode in (False, True):
            graph = example_graph(); definition = graph["definitions"][0]
            definition["graph"]["outputs"][0]["id"] = "features"
            if explicit_mode: definition["output_mode"] = "named"
            source = python_io.export_python(graph)
            source = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = "))
            restored = python_io.import_python(source)
            self.assertEqual(restored["definitions"][0]["graph"]["outputs"][0]["id"], "features")
            self.assertEqual(restored["definitions"][0]["output_mode"], "named")
            self.assertTrue(schema.validate_graph(restored)["valid"])

    def test_dynamic_unknown_selectors_and_bundles_used_as_tensors_are_rejected(self):
        source = python_io.export_python(named_graph())
        for expression in ("call_left[table_a]", "call_left['absent']", "call_left[0]", "call_left", "table_a['main']"):
            modified = source.replace("a=call_left['main']", "a=" + expression)
            with self.subTest(expression=expression), self.assertRaises(ValueError): python_io.import_python(modified)

    def test_alias_cannot_rebind_self_and_change_actual_component_lookup_semantics(self):
        source = python_io.export_python(named_graph())
        modified = source.replace("        call_sum_features =", "        self = call_left\n        call_sum_features =")
        with self.assertRaisesRegex(ValueError, "reassignment|self|reserved"):
            python_io.import_python(modified)


if __name__ == "__main__": unittest.main()
