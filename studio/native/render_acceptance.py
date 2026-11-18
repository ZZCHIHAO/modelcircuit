"""In-process DPG rendering/callback checks; no desktop input automation."""
import time
import tempfile
import shutil
from pathlib import Path

import dearpygui.dearpygui as dpg

from .document import Document
from .ui import NativeWorkbench


def _frames(ui, count=6):
    for _ in range(count):
        ui.process_callbacks()
        ui.tick()
        dpg.render_dearpygui_frame()


def _capture(ui, target):
    from PIL import Image
    # The DPG mvBuffer NumPy buffer interface reports an invalid length in
    # 2.3.1. Use its own PNG encoder, then copy the unaltered render to the
    # requested Unicode destination using Python filesystem APIs.
    with tempfile.TemporaryDirectory(prefix='mlforge-frame-') as temporary:
        image_path = Path(temporary) / 'frame.png'
        _frames(ui)
        dpg.output_frame_buffer(str(image_path))
        for _ in range(30):
            _frames(ui, 1)
            if image_path.is_file() and image_path.stat().st_size:
                with Image.open(image_path) as rendered:
                    rendered.verify()
                shutil.copy2(image_path, target)
                return
    raise RuntimeError('DPG framebuffer capture did not complete')


def _wait_ui(ui, timeout=100):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _frames(ui, 1)
        if ui.run_record and ui.run_record['status'] in {'succeeded', 'failed', 'cancelled'}:
            return ui.run_record
    raise TimeoutError('Native UI worker deadline exceeded')


def check_render(manager, directory, resources, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    document = Document()
    ui = NativeWorkbench(document, manager, directory, resources)
    try:
        ui.build()
        _frames(ui)
        bounds = []
        for ref, item in ui.ref_nodes.items():
            x, y = dpg.get_item_pos(item)
            width, height = dpg.get_item_rect_size(item)
            assert width > 0 and height > 0
            bounds.append((ref, x, y, width, height))
        for index, first in enumerate(bounds):
            for second in bounds[index + 1:]:
                _, x, y, width, height = first
                _, xx, yy, ww, hh = second
                assert not (min(x + width, xx + ww) > max(x, xx) and min(y + height, yy + hh) > max(y, yy)), (first, second)
        assert len(bounds) == 8
        ui._start_run()
        run = _wait_ui(ui)
        if run['status'] != 'succeeded':
            raise RuntimeError(str(run.get('error')))
        assert ui.result['parameter_count'] == 58
        assert len(ui.result['history']) == 20
        _capture(ui, output / 'native-root.png')
        ui._enter_root('encode_b')
        document.select('call:project')
        ui._render_inspector(); ui._render_probe()
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id('native_probe_tab'))
        _frames(ui)
        assert ui.snapshot()['probe_source'] == 'call:encode_b/call:project'
        assert ui.result['probes'][ui.snapshot()['probe_source']]['source'] == 'measured'
        _capture(ui, output / 'native-chip.png')
        ui._root()
        ui._open_python()
        original = dpg.get_value('native_python_code')
        before = document.semantic_key()
        invalid = original + '\nraise RuntimeError("not_executed")\n'
        dpg.set_value('native_python_code', invalid)
        ui._import_python()
        assert document.semantic_key() == before
        assert dpg.get_value('native_python_error')
        ui._python_hidden()
        ui._open_python()
        assert dpg.get_value('native_python_code') == invalid
        _capture(ui, output / 'native-python-error.png')
        ui._generate_python()
        ui._python_hidden()
        dpg.set_value('native_steps', 200)
        ui._start_run()
        ui._cancel_run()
        cancelled = _wait_ui(ui)
        assert cancelled['status'] == 'cancelled'
        return {'status': 'passed', 'kind': 'native-render-callbacks', 'node_count': len(bounds),
                'root_node_rectangles_do_not_overlap': True, 'run_id': run['run_id'],
                'cancel_run_id': cancelled['run_id'], 'measured_probe': 'call:encode_b/call:project',
                'failed_python_draft_preserved': True, 'screenshots': ['native-root.png', 'native-chip.png', 'native-python-error.png']}
    finally:
        ui.shutdown()
