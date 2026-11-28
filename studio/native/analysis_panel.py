"""Projection and cluster results in the native workbench."""
from collections import defaultdict
from math import isfinite

import dearpygui.dearpygui as dpg


def render_analysis(parent, analysis):
    if not isinstance(analysis, dict):
        return
    variance = analysis.get('explained_variance_ratio')
    if isinstance(variance, list) and all(type(value) in (int, float) and isfinite(value) for value in variance) and variance:
        detail = '   '.join(f'PC{i} {value:.1%}' for i, value in enumerate(variance[:8], 1))
        dpg.add_text(f'主成分解释方差  ·  累计 {sum(variance):.1%}\n{detail}', parent=parent, wrap=640)
    counts = analysis.get('cluster_counts')
    if isinstance(counts, list) and all(type(value) is int and value >= 0 for value in counts) and counts:
        detail = '   '.join(f'{i} · {value} 个样本' for i, value in enumerate(counts[:12]))
        dpg.add_text(f'{len(counts)} 簇  ·  {detail}', parent=parent, wrap=640)
    points, axes = analysis.get('points'), analysis.get('axes')
    if (not isinstance(points, list) or not isinstance(axes, list) or len(axes) != 2
            or not all(isinstance(axis, str) for axis in axes)):
        return
    groups = defaultdict(lambda: ([], []))
    for row in points[:2048]:
        if not isinstance(row, list) or len(row) != 3:
            continue
        x, y, cluster = row
        if not all(type(value) in (float, int) and isfinite(value) for value in (x, y)):
            continue
        label = ('投影' if cluster is None else f'簇 {cluster}' if type(cluster) is int and 0 <= cluster < 24 else '其他簇')
        group = groups[label]
        group[0].append(x)
        group[1].append(y)
    if not groups:
        return
    dpg.add_text(f"显示 {sum(len(group[0]) for group in groups.values())} / {analysis.get('rows', '—')} 个样本", parent=parent)
    with dpg.plot(parent=parent, height=250, width=-1):
        dpg.add_plot_legend()
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=axes[0][:100])
        with dpg.plot_axis(dpg.mvYAxis, label=axes[1][:100]) as y_axis:
            for label, (x, y) in groups.items():
                dpg.add_scatter_series(x, y, label=label)
        dpg.fit_axis_data(x_axis)
        dpg.fit_axis_data(y_axis)
