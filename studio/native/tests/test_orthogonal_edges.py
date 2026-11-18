"""Orthogonal routing and a public-DPG, per-canvas drawing adapter."""
import importlib
import unittest

try:
    edges = importlib.import_module('native.orthogonal_edges')
except ModuleNotFoundError:
    edges = None


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(edges, 'The orthogonal_edges module must exist')

    def assertOrthogonal(self, points):
        self.assertGreaterEqual(len(points), 2)
        for a, b in zip(points, points[1:]):
            self.assertNotEqual(a, b)
            self.assertTrue(a[0] == b[0] or a[1] == b[1], (a, b))

    def test_same_level_forward_link_is_one_horizontal_segment(self):
        self.assertEqual(edges.route_orthogonal((10, 30), (250, 30)), ((10.0, 30.0), (250.0, 30.0)))

    def test_forward_route_exits_right_and_approaches_input_from_left(self):
        points = edges.route_orthogonal((10, 30), (250, 100))
        self.assertOrthogonal(points)
        self.assertGreater(points[1][0], points[0][0])
        self.assertGreater(points[-1][0], points[-2][0])

    def test_reversed_route_goes_around_source_and_target_bodies(self):
        bodies = ((100, 0, 180, 90), (0, 0, 60, 90))
        points = edges.route_orthogonal((180, 45), (0, 45), source_rect=bodies[0], target_rect=bodies[1])
        self.assertOrthogonal(points)
        self.assertGreater(points[1][0], 180)
        self.assertLess(points[-2][0], 0)
        for segment in zip(points, points[1:]):
            self.assertEqual(edges.clip_polyline(segment, (-100, -100, 300, 200), obstacles=bodies), (segment,))

    def test_route_avoids_intervening_body(self):
        body = (80, 10, 150, 70)
        points = edges.route_orthogonal((10, 40), (240, 40), obstacles=(body,))
        self.assertOrthogonal(points)
        for segment in zip(points, points[1:]):
            self.assertEqual(edges.clip_polyline(segment, (-100, -100, 400, 300), obstacles=(body,)), (segment,))

    def test_fanout_routes_are_stable_and_can_use_distinct_channels(self):
        first = edges.route_orthogonal((10, 30), (250, 130), lane=0)
        second = edges.route_orthogonal((10, 30), (250, 130), lane=1)
        self.assertEqual(first, edges.route_orthogonal((10, 30), (250, 130), lane=0))
        self.assertNotEqual(first, second)

    def test_clipping_does_not_draw_or_hit_across_pane(self):
        actual = edges.clip_polyline(((-20, 30), (120, 30)), (0, 0, 100, 100))
        self.assertEqual(actual, (((0.0, 30.0), (100.0, 30.0)),))
        self.assertIsNone(edges.hit_test_segments((110, 30), actual, tolerance=3))
        self.assertEqual(edges.hit_test_segments((50, 32), actual, tolerance=3), 2)

    def test_node_body_occludes_foreground_wire_and_hit_area(self):
        actual = edges.clip_polyline(((0, 30), (120, 30)), (0, 0, 150, 100), obstacles=((40, 10, 70, 50),))
        self.assertEqual(actual, (((0.0, 30.0), (40.0, 30.0)), ((70.0, 30.0), (120.0, 30.0))))
        self.assertIsNone(edges.hit_test_segments((55, 30), actual, tolerance=4))

    def test_diagonal_input_rejected(self):
        with self.assertRaises(ValueError):
            edges.clip_polyline(((0, 0), (20, 20)), (0, 0, 100, 100))

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            edges.route_orthogonal((float('nan'), 0), (100, 10))


class RealDpgOverlayTests(unittest.TestCase):
    def test_real_draw_items_keep_native_ids_and_use_visible_segment_hit_testing(self):
        self.assertIsNotNone(edges, 'The orthogonal_edges module must exist')
        import dearpygui.dearpygui as dpg
        dpg.create_context()
        overlay = None
        try:
            dpg.create_viewport(title='Orthogonal bounded frame test', width=500, height=300)
            with dpg.window(tag='test_main'):
                with dpg.node_editor(tag='test_editor'):
                    with dpg.node(label='Source', pos=(20, 30)):
                        with dpg.node_attribute(tag='test_source', attribute_type=dpg.mvNode_Attr_Output):
                            dpg.add_text('output')
                    with dpg.node(label='Target', pos=(240, 130)):
                        with dpg.node_attribute(tag='test_target', attribute_type=dpg.mvNode_Attr_Input):
                            dpg.add_text('input')
                    native_link = dpg.add_node_link('test_source', 'test_target')
            dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window('test_main', True)
            for _ in range(3):
                dpg.render_dearpygui_frame()
            overlay = edges.OrthogonalEdgeOverlay()
            overlay.update({native_link: ((60, 70), (280, 170))}, (10, 10, 400, 250))
            dpg.render_dearpygui_frame()
            self.assertTrue(dpg.does_item_exist(native_link))
            self.assertTrue(overlay.segments[native_link])
            a, b = overlay.segments[native_link][0]
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            self.assertEqual(overlay.hit_test(middle), native_link)
            self.assertIsNone(overlay.hit_test((450, 170)))
            self.assertTrue(all(dpg.does_item_exist(item) for item in overlay.draw_items[native_link]))
            ids = dict(overlay.draw_items)
            overlay.update({native_link: ((60, 70), (280, 170))}, (10, 10, 400, 250), selected=(native_link,))
            self.assertEqual(ids, overlay.draw_items, 'Unchanged geometry should retain draw items')
            from unittest.mock import patch
            with patch.object(dpg, 'configure_item', wraps=dpg.configure_item) as configured:
                overlay.update({native_link: ((60, 70), (280, 170))}, (10, 10, 400, 250), selected=(native_link,))
                configured.assert_not_called()
            overlay.clear()
            self.assertEqual(overlay.segments, {})
            self.assertIsNone(overlay.hit_test(middle), 'Clearing removes stale selection hit targets')
            self.assertTrue(dpg.does_item_exist(native_link), 'Adapter owns drawings, never native graph items')
        finally:
            if overlay is not None:
                overlay.dispose()
            dpg.destroy_context()


class RoutingRegressionTests(unittest.TestCase):
    def test_reversed_route_keeps_horizontal_socket_direction_without_bodies(self):
        points = edges.route_orthogonal((180, 45), (0, 45))
        self.assertGreater(points[1][0], points[0][0])
        self.assertGreater(points[-1][0], points[-2][0])
        self.assertTrue(any(point[1] != 45 for point in points))

    def test_obstacle_close_to_socket_shortens_departure_channel(self):
        body = (20, 10, 80, 60)
        points = edges.route_orthogonal((10, 30), (160, 30), obstacles=(body,))
        self.assertGreater(points[1][0], 10)
        for segment in zip(points, points[1:]):
            self.assertEqual(edges.clip_polyline(segment, (0, -100, 200, 150), obstacles=(body,)), (segment,))

class WorkBudgetTests(unittest.TestCase):
    def test_explicit_zero_search_budget_uses_deterministic_orthogonal_fallback(self):
        actual = edges.route_orthogonal((180, 45), (0, 45), max_expansions=0)
        self.assertEqual(actual, ((180.0, 45.0), (198.0, 45.0), (198.0, 13.0), (-18.0, 13.0), (-18.0, 45.0), (0.0, 45.0)))
        self.assertEqual(actual, edges.route_orthogonal((180, 45), (0, 45), max_expansions=0))


class PaneRoutingTests(unittest.TestCase):
    def test_reversed_wire_uses_available_channel_inside_view(self):
        box = (500, 50, 980, 450)
        bodies = ((800, 190, 920, 270), (520, 320, 640, 400))
        route = edges.route_orthogonal((920, 230), (520, 360), obstacles=bodies, bounds=box, lane=1)
        self.assertTrue(all(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3] for p in route))
        self.assertEqual(edges.clip_polyline(route, box, obstacles=bodies), tuple(zip(route, route[1:])))


if __name__ == '__main__':
    unittest.main()
