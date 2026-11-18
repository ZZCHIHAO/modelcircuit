"""Screen-space Manhattan wires for native node editors.

Routing is independent of DPG. The adapter owns only foreground draw items:
callers retain native nodes, attributes, link IDs and all document callbacks.
Hit tests use exactly the clipped, visible wire segments. Call ``update`` on
the GUI thread after geometry is available and ``clear`` for hidden canvases.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import hypot, isfinite

Point = tuple[float, float]
Rect = tuple[float, float, float, float]
Segment = tuple[Point, Point]
MAX_ROUTE_EXPANSIONS = 4096
MAX_CANVAS_SEARCH_EXPANSIONS = 8192


def _point(value) -> Point:
    point = (float(value[0]), float(value[1]))
    if not all(isfinite(number) for number in point):
        raise ValueError('Wire coordinates must be finite')
    return point


def _rect(value) -> Rect:
    left, top = _point(value[:2])
    right, bottom = _point(value[2:])
    if right < left or bottom < top:
        raise ValueError('Rectangle bounds are reversed')
    return left, top, right, bottom


def _compact(points):
    result = []
    for raw in points:
        point = _point(raw)
        if result and result[-1] == point:
            continue
        while len(result) > 1 and ((result[-2][0] == result[-1][0] == point[0])
                                  or (result[-2][1] == result[-1][1] == point[1])):
            result.pop()
        result.append(point)
    return tuple(result)


def _crosses_body(a, b, rect):
    left, top, right, bottom = rect
    if a[1] == b[1]:
        return top < a[1] < bottom and max(min(a[0], b[0]), left) < min(max(a[0], b[0]), right)
    return left < a[0] < right and max(min(a[1], b[1]), top) < min(max(a[1], b[1]), bottom)


def _clear(points, bodies):
    return all(not _crosses_body(a, b, body) for a, b in zip(points, points[1:]) for body in bodies)


def route_orthogonal(source, target, *, source_rect=None, target_rect=None,
                     obstacles=(), lane=0, bounds=None,
                     max_expansions=MAX_ROUTE_EXPANSIONS) -> tuple[Point, ...]:
    """Return deterministic right-exit/left-entry, axis-aligned vertices.

    Same-level unobstructed forward wires stay completely straight. Ordinary
    forward wires use a stable vertical channel. Reversed or obstructed wires
    use a rectilinear visibility grid with bend penalties. For exceptionally
    dense canvases the search uses the nearest 64 obstacles and at most
    ``max_expansions`` state expansions (default 4096); the renderer
    still occludes against every body, so drawing never covers a node.
    Overlapping nodes may have no complete feasible route.
    """
    start, finish = _point(source), _point(target)
    if start == finish:
        return (start,)
    bodies = tuple(dict.fromkeys(_rect(item) for item in
                                 (*obstacles, *((source_rect,) if source_rect else ()),
                                  *((target_rect,) if target_rect else ()))))
    gap = finish[0] - start[0]
    lane = max(0, int(lane)) % 8
    stub = min(18 + lane * 5, gap / 3) if gap > 0 else 18 + lane * 5
    routing_bounds = _rect(bounds) if bounds is not None else None
    if routing_bounds is not None and not all(routing_bounds[0] <= p[0] <= routing_bounds[2]
                                               and routing_bounds[1] <= p[1] <= routing_bounds[3]
                                               for p in (start, finish)):
        routing_bounds = None  # Offscreen endpoints are clipped after routing.
    if routing_bounds is not None:
        stub = min(stub, max(0, routing_bounds[2] - start[0]) / 2,
                   max(0, finish[0] - routing_bounds[0]) / 2)
    # A nearby body can require turning before the usual 18-pixel exit.
    for left, top, right, bottom in bodies:
        if top < start[1] < bottom and left > start[0]:
            stub = min(stub, (left - start[0]) / 2)
        if top < finish[1] < bottom and right < finish[0]:
            stub = min(stub, (finish[0] - right) / 2)
    if gap > 0 and start[1] == finish[1] and _clear((start, finish), bodies):
        return (start, finish)
    if gap > 2 * stub:
        channel = max(start[0] + stub, min(finish[0] - stub, (start[0] + finish[0]) / 2 + lane * 8))
        candidates = [channel]
        candidates.extend(sorted({body[0] - 8 for body in bodies} | {body[2] + 8 for body in bodies},
                                 key=lambda x: (abs(x - channel), x))[:16])
        for x in candidates:
            if start[0] + stub <= x <= finish[0] - stub:
                points = _compact((start, (x, start[1]), (x, finish[1]), finish))
                if _clear(points, bodies):
                    return points
    # Preserve horizontal departure and arrival even when the target is left
    # of its source. Coordinates on body boundaries are allowed, interiors not.
    departure = (start[0] + stub, start[1])
    approach = (finish[0] - stub, finish[1])
    nearest = sorted(bodies, key=lambda r: (min(abs((r[0] + r[2]) / 2 - start[0]) + abs((r[1] + r[3]) / 2 - start[1]),
                                               abs((r[0] + r[2]) / 2 - finish[0]) + abs((r[1] + r[3]) / 2 - finish[1])), r))[:64]
    xs = sorted({departure[0], approach[0], *(v for body in nearest for v in (body[0] - 8, body[2] + 8))})
    ys = sorted({start[1], finish[1], min(start[1], finish[1]) - 32 - lane * 6,
                 max(start[1], finish[1]) + 32 + lane * 6,
                 *(v for body in nearest for v in (body[1] - 8 - lane * 2, body[3] + 8 + lane * 2))})
    if routing_bounds is not None:
        xs = sorted({x for x in xs if routing_bounds[0] <= x <= routing_bounds[2]}
                    | {routing_bounds[0] + 2, routing_bounds[2] - 2})
        ys = sorted({y for y in ys if routing_bounds[1] <= y <= routing_bounds[3]}
                    | {routing_bounds[1] + 2, routing_bounds[3] - 2})
    origin = (xs.index(departure[0]), ys.index(departure[1]), 0)
    goal = (xs.index(approach[0]), ys.index(approach[1]))
    queue, cost, previous = [], {origin: 0.0}, {}
    heappush(queue, (0.0, 0.0, origin))
    terminal = None
    expansions = 0
    while queue and expansions < max(0, min(MAX_ROUTE_EXPANSIONS, int(max_expansions))):
        _, distance, state = heappop(queue)
        if distance != cost.get(state):
            continue
        expansions += 1
        x, y, direction = state
        if (x, y) == goal:
            terminal = state
            break
        for xx, yy, next_direction in ((x - 1, y, 0), (x + 1, y, 0), (x, y - 1, 1), (x, y + 1, 1)):
            if not (0 <= xx < len(xs) and 0 <= yy < len(ys)):
                continue
            if state == origin and xx < x:
                continue
            if (xx, yy) == goal and xx < x:
                continue
            a, b = (xs[x], ys[y]), (xs[xx], ys[yy])
            if not _clear((a, b), nearest):
                continue
            next_state = (xx, yy, next_direction)
            next_cost = distance + abs(a[0] - b[0]) + abs(a[1] - b[1]) + (12 if direction != next_direction else 0)
            if next_cost < cost.get(next_state, float('inf')):
                cost[next_state] = next_cost
                previous[next_state] = state
                heuristic = abs(b[0] - approach[0]) + abs(b[1] - approach[1])
                heappush(queue, (next_cost + heuristic, next_cost, next_state))
    if terminal is not None:
        route = []
        while True:
            route.append((xs[terminal[0]], ys[terminal[1]]))
            if terminal == origin:
                break
            terminal = previous[terminal]
        return _compact((start, *reversed(route), finish))
    # Degenerate overlapping geometry: keep topology orthogonal; clipping makes
    # the unavoidable hidden portions explicit instead of crossing node bodies.
    detour = min(ys)
    return _compact((start, departure, (departure[0], detour), (approach[0], detour), approach, finish))


def clip_polyline(points, clip_rect, *, obstacles=()) -> tuple[Segment, ...]:
    """Clip axis-aligned segments to a canvas and subtract node interiors."""
    points = tuple(_point(point) for point in points)
    clip = _rect(clip_rect)
    bodies = tuple(_rect(body) for body in obstacles)
    result = []
    for a, b in zip(points, points[1:]):
        if a == b:
            continue
        horizontal = a[1] == b[1]
        if not horizontal and a[0] != b[0]:
            raise ValueError('Wire segments must be horizontal or vertical')
        axis = 0 if horizontal else 1
        fixed = a[1 - axis]
        if not clip[1 - axis] <= fixed <= clip[3 - axis]:
            continue
        low, high = max(min(a[axis], b[axis]), clip[axis]), min(max(a[axis], b[axis]), clip[axis + 2])
        intervals = [(low, high)] if low < high else []
        for body in bodies:
            if not body[1 - axis] < fixed < body[3 - axis]:
                continue
            next_intervals = []
            for lo, hi in intervals:
                if body[axis] >= hi or body[axis + 2] <= lo:
                    next_intervals.append((lo, hi))
                else:
                    if lo < body[axis]:
                        next_intervals.append((lo, body[axis]))
                    if body[axis + 2] < hi:
                        next_intervals.append((body[axis + 2], hi))
            intervals = next_intervals
        if a[axis] > b[axis]:
            intervals = [(hi, lo) for lo, hi in reversed(intervals)]
        for lo, hi in intervals:
            result.append((((lo, fixed), (hi, fixed)) if horizontal else ((fixed, lo), (fixed, hi))))
    return tuple(result)


def hit_test_segments(point, segments, *, tolerance=6):
    """Return nearest visible-segment distance, or None outside tolerance."""
    point = _point(point)
    nearest = float('inf')
    for a, b in segments:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx * dx + dy * dy
        if not length_squared:
            continue
        fraction = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared))
        nearest = min(nearest, hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy))
    return nearest if nearest <= tolerance else None


class OrthogonalEdgeOverlay:
    """One noninteractive foreground drawlist and hit map per canvas.

    Native Bezier visibility and input dispatch remain caller responsibilities.
    No native node/link selection getter is used: visible-wire selection is an
    explicit set of native link IDs owned by the canvas controller.
    """
    def __init__(self, dpg=None):
        if dpg is None:
            import dearpygui.dearpygui as dpg
        self.dpg = dpg
        self.drawlist = dpg.add_viewport_drawlist(front=True)
        self.routes = {}
        self.segments = {}
        self.draw_items = {}
        self.clip_rect = (0, 0, 0, 0)
        self.node_rects = ()
        self._geometry = None
        self._style = None
        self._disposed = False

    def update(self, edges, clip_rect, node_rects=(), *, selected=(), hovered=None):
        clip = _rect(clip_rect)
        bodies = tuple(_rect(rect) for rect in node_rects)
        normalized = tuple((key, _point(value[0]), _point(value[1])) for key, value in sorted(edges.items(), key=lambda item: str(item[0])))
        geometry = (normalized, clip, bodies)
        if self._disposed:
            raise RuntimeError('Cannot update a disposed wire overlay')
        if geometry != self._geometry:
            self.clear()
            self.clip_rect, self.node_rects = clip, bodies
            fanout = {}
            # Divide the total rerender search allowance deterministically;
            # ordinary unobstructed wires take the constant-size fast path.
            allowance = min(MAX_ROUTE_EXPANSIONS, MAX_CANVAS_SEARCH_EXPANSIONS // max(1, len(normalized)))
            for key, source, target in normalized:
                lane = fanout.get(source, 0)
                fanout[source] = lane + 1
                route = route_orthogonal(source, target, obstacles=bodies, lane=lane,
                                         bounds=clip,
                                         max_expansions=allowance)
                segments = clip_polyline(route, clip, obstacles=bodies)
                self.routes[key], self.segments[key] = route, segments
                self.draw_items[key] = tuple(self.dpg.draw_line(a, b, parent=self.drawlist, thickness=2.0,
                                                               color=(116, 162, 176, 255)) for a, b in segments)
            self._geometry = geometry
        selected = frozenset(selected)
        style = (selected, hovered)
        if style == self._style:
            return self.routes
        for key, items in self.draw_items.items():
            color = (243, 194, 105, 255) if key in selected else (171, 224, 230, 255) if key == hovered else (116, 162, 176, 255)
            for item in items:
                self.dpg.configure_item(item, color=color, thickness=2.8 if key in selected else 2.0)
        self._style = style
        return self.routes

    def hit_test(self, point, *, tolerance=6):
        point = _point(point)
        left, top, right, bottom = self.clip_rect
        if not left <= point[0] <= right or not top <= point[1] <= bottom:
            return None
        if any(rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3] for rect in self.node_rects):
            return None
        candidates = []
        for key, segments in self.segments.items():
            distance = hit_test_segments(point, segments, tolerance=tolerance)
            if distance is not None:
                candidates.append((distance, str(key), key))
        return min(candidates)[2] if candidates else None

    def clear(self):
        if not self._disposed and self.dpg.does_item_exist(self.drawlist):
            self.dpg.delete_item(self.drawlist, children_only=True)
        self.routes.clear()
        self.segments.clear()
        self.draw_items.clear()
        self._geometry = None
        self._style = None
        self.clip_rect = (0, 0, 0, 0)
        self.node_rects = ()

    def dispose(self):
        self.clear()
        if not self._disposed and self.dpg.does_item_exist(self.drawlist):
            self.dpg.delete_item(self.drawlist)
        self._disposed = True
