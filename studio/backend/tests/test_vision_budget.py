"""D0 characterization and independent arithmetic; no production ResNet builder.

The audit fixture is deliberately not a platform graph or an executed model.
Literal expectations below come from ResNet18 convolution/BN formulas and the
current schema traversal, rather than from the fixture's implementation.
"""
import copy
import unittest

from studio_core import schema


def relu_graph(shape, profile="classification-v2"):
    graph = {
        "schema_version": 1, "id": "budget_case", "revision": 1,
        "inputs": [{"id": "x", "shape": shape, "dtype": "float32"}],
        "instances": [{"id": "activation", "kind": "ReLU", "params": {}}],
        "calls": [{"id": "activate", "instance_id": "activation",
                   "inputs": {"x": "input:x"}}],
        "outputs": [{"id": "out", "source": "call:activate"}],
        "definitions": [],
    }
    if profile is not None:
        graph["resource_profile"] = profile
    return graph


class LegacyBudgetCharacterizationTests(unittest.TestCase):
    def test_old_null_batch_still_estimates_64_and_rejects_large_stem(self):
        self.assertEqual(schema._elements([None, 64, 112, 112]), 51_380_224)
        report = schema.validate_graph(relu_graph([None, 64, 112, 112]))
        self.assertFalse(report["valid"])
        self.assertEqual(report["errors"][0]["field"], "resource_limits")

    def test_old_single_tensor_boundary_is_enforced_without_allocation(self):
        limits = schema.resource_limits("classification-v2")
        self.assertEqual(schema._shape([None, 262_144], "x", limits), [None, 262_144])
        with self.assertRaises(schema.GraphError):
            schema._shape([None, 262_145], "x", limits)

    def test_old_cumulative_boundary_is_enforced_without_allocation(self):
        budget = schema._Budget(schema.resource_limits("classification-v2"))
        for _ in range(8):
            budget.tensor([64, 262_144], "x")
        self.assertEqual(budget.inferred_elements, 134_217_728)
        with self.assertRaises(schema.GraphError):
            budget.tensor([1], "over")

    def test_legacy_parameter_boundaries_remain_distinct(self):
        for profile, in_features, out_features, expected in (
            (None, 999, 2000, 2_000_000),
            ("classification-v2", 3999, 5000, 20_000_000),
        ):
            with self.subTest(profile=profile):
                graph = relu_graph([None, in_features], profile)
                graph["instances"][0].update(
                    kind="Linear", params={"in_features": in_features, "out_features": out_features})
                result = schema.validate_graph(graph)
                self.assertTrue(result["valid"], result["errors"])
                self.assertEqual(result["parameter_count"], expected)
                graph["instances"][0]["params"]["out_features"] += 1
                self.assertFalse(schema.validate_graph(graph)["valid"])

    def test_nonbatch_unknown_and_excess_rank_still_fail(self):
        for shape in ([None, None, 4, 4], [None] * 9):
            with self.subTest(shape=shape):
                self.assertFalse(schema.validate_graph(relu_graph(shape))["valid"])

    def test_actual_batch_65_is_allowed_for_tiny_legacy_models(self):
        import torch
        from studio_core.runtime import build_model
        for profile in (None, "classification-v2"):
            with self.subTest(profile=profile):
                model = build_model(relu_graph([None, 2], profile), capture=False)
                actual = model(x=torch.ones(65, 2))["out"]
                self.assertEqual(list(actual.shape), [65, 2])
                self.assertTrue(torch.equal(actual, torch.ones(65, 2)))

    def test_nested_call_is_charged_for_instance_inference_and_call_inference(self):
        child = relu_graph([None, 4])
        child = {key: child[key] for key in ("inputs", "instances", "calls", "outputs")}
        graph = relu_graph([None, 4])
        graph["instances"] = [{"id": "nested", "kind": "chip", "definition_id": "unit", "params": {}}]
        graph["calls"][0]["instance_id"] = "nested"
        registry = {"unit": {"id": "unit", "graph": child}}
        budget = schema._Budget(schema.resource_limits("classification-v2"))
        resolved = schema._resolve_scope(graph, registry, budget=budget)
        schema._infer(resolved, budget)
        # Root input + two complete child passes + root chip output = 6 * 256.
        self.assertEqual(budget.inferred_elements, 1536)
        self.assertEqual(budget.resolved_nodes, 8)
        self.assertEqual(budget.inference_nodes, 12)

    def test_v1_chip_bundle_has_strict_fields_and_cannot_declare_new_policy(self):
        from native.chip_library import validate_bundle
        graph = relu_graph([None, 4])
        definition = {"id": "unit", "version": "1.0.0", "public_parameters": [], "bindings": {},
                      "graph": {key: graph[key] for key in ("inputs", "instances", "calls", "outputs")}}
        bundle = {"chip_format_version": 1, "definition_id": "unit", "definitions": [definition]}
        self.assertEqual(validate_bundle(bundle)[1], ["unit"])
        changed = copy.deepcopy(bundle)
        changed["requirements"] = {"resource_profile": "vision-v4", "resource_batch_bound": 2}
        with self.assertRaises(ValueError):
            validate_bundle(changed)

    def test_vision_profile_requires_explicit_validated_batch_bound(self):
        graph = relu_graph([None, 3, 32, 32])
        graph["resource_profile"] = "vision-v4"
        self.assertFalse(schema.validate_graph(graph)["valid"])
        graph['resource_batch_bound'] = 2
        self.assertTrue(schema.validate_graph(graph)['valid'])
        # D0's original unsupported-profile/BN assertions were characterization.
        # A1/A2 now implement them; preserve the explicit policy requirement.

    def test_python_profile_remains_legacy_compatible(self):
        from studio_core.python_io import python_profile
        self.assertEqual(python_profile(relu_graph([None, 4], None)), "s1-pytorch-v1")
        self.assertEqual(python_profile(relu_graph([None, 4])), "classification-pytorch-v2")


class IndependentResNetAuditTests(unittest.TestCase):
    def audit(self, **kwargs):
        from vision_budget_audit import audit_resnet18
        return audit_resnet18(**kwargs)

    def test_candidate_only_replaces_unknown_batch_and_keeps_hard_limits(self):
        from vision_budget_audit import candidate_elements
        self.assertEqual(candidate_elements([None, 64, 112, 112], 2), 1_605_632)
        self.assertEqual(candidate_elements([None, 64, 112, 112], 64), 51_380_224)
        self.assertEqual(candidate_elements([65, 2], 2), 130)
        for invalid in ([None, None, 4], [None, 0], [True, 3]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                candidate_elements(invalid, 2)
        for bound in (0, 65, True, 2.0):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                candidate_elements([None, 3], bound)
        result = self.audit()
        self.assertEqual(result["limits"], {
            "max_parameters": 20_000_000, "max_tensor_elements": 16_777_216,
            "max_inferred_elements": 134_217_728})

    def test_standard_stage_shapes_and_independent_parameter_counts(self):
        result = self.audit()
        self.assertEqual(result["stem_conv_shape"], [2, 64, 112, 112])
        self.assertEqual(result["stage_shapes"], [
            [2, 64, 56, 56], [2, 128, 28, 28], [2, 256, 14, 14], [2, 512, 7, 7]])
        self.assertEqual(result["features_shape"], [2, 512])
        self.assertEqual(result["logits_shape"], [2, 1000])
        self.assertEqual(result["stage_parameters"], [147_968, 525_568, 2_099_712, 8_393_728])
        self.assertEqual(result["stem_parameters"], 9536)
        self.assertEqual(result["backbone_parameters"], 11_176_512)
        self.assertEqual(result["parameters"], 11_689_512)

    def test_bn_buffers_are_separate_and_every_state_instance_is_named(self):
        result = self.audit()
        self.assertEqual(result["bn_instances"], 20)
        self.assertEqual(result["bn_affine_parameters"], 9600)
        self.assertEqual(result["bn_float_buffer_elements"], 9600)
        self.assertEqual(result["bn_int64_buffer_elements"], 20)
        self.assertEqual(result["parameter_and_persistent_buffer_bytes"], 46_796_608)
        paths = result["bn_state_instances"]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("entry.stem.bn", paths)
        self.assertIn("entry.stage4.block0.projection.bn", paths)
        self.assertIn("entry.stage4.block1.bn2", paths)

    def test_nested_traversal_is_not_replaced_by_leaf_only_estimate(self):
        result = self.audit()
        self.assertEqual(result["flat_leaf_elements"], 16_762_832)
        self.assertEqual(result["scope_elements"]["stem"], 5_519_360)
        self.assertEqual(result["scope_elements"]["stage1"], 14_049_280)
        self.assertEqual(result["scope_elements"]["stage2"], 11_239_424)
        self.assertEqual(result["scope_elements"]["stage3"], 5_619_712)
        self.assertEqual(result["scope_elements"]["stage4"], 2_809_856)
        self.assertEqual(result["root_inference_elements"], 160_171_888)
        self.assertEqual(result["definition_inference_elements"], 136_131_536)
        self.assertEqual(result["total_inference_elements"], 296_303_424)
        self.assertEqual(result["largest_tensor_elements"], 1_605_632)
        self.assertFalse(result["cumulative_fits"])
        self.assertTrue(result["parameters_fit"])
        self.assertTrue(result["single_tensor_fits"])

    def test_hierarchy_estimate_has_finite_independent_instance_paths(self):
        result = self.audit()
        self.assertEqual(result["definitions"], 17)
        # Four chip edges; the root graph itself is not a chip instance.
        self.assertEqual(result["chip_depth"], 4)
        self.assertEqual(result["primitive_instances"], 69)
        self.assertEqual(result["chip_instances"], 17)
        self.assertEqual(result["block_instances"], ["entry.stage%d.block%d" % (stage, block)
                                                for stage in range(1, 5) for block in range(2)])
        self.assertLess(result["estimated_storage"]["schema_values_including_keys"], 20_000)
        self.assertLess(result["estimated_storage"]["max_json_depth"], 32)
        self.assertLess(result["estimated_storage"]["utf8_compact_bytes"], 4 * 1024 * 1024)

    def test_batch_one_and_two_separate_execution_from_compiler_work(self):
        for batch, runtime, root_static, defaults, total, output in (
            (1, 12_020_176, 80_085_944, 68_065_768, 148_151_712, 1000),
            (2, 24_040_352, 160_171_888, 136_131_536, 296_303_424, 2000),
        ):
            with self.subTest(batch=batch):
                result = self.audit(batch_bound=batch)
                self.assertEqual(result["root_output_elements"], output)
                self.assertEqual(result["runtime_counting_estimate_elements"], runtime)
                self.assertEqual(result["root_inference_elements"], root_static)
                self.assertEqual(result["definition_inference_elements"], defaults)
                self.assertEqual(result["total_inference_elements"], total)
                self.assertFalse(result["cumulative_fits"])
                self.assertEqual(result["resolved_nodes_including_definition_checks"], 819)
                self.assertEqual(result["inference_nodes_including_definition_checks"], 2864)

    def test_one_channel_variant_is_not_mislabeled_as_standard_three_channels(self):
        result = self.audit(input_channels=1, num_classes=None)
        self.assertEqual(result["parameters"], 11_170_240)
        self.assertIsNone(result["logits_shape"])
        self.assertEqual(result["features_shape"], [2, 512])


if __name__ == "__main__":
    unittest.main()
