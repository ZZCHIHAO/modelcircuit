"""Desktop orchestration. DPG is imported only by the main GUI process."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .paths import create_runtime_paths, resolve_data_directory, resource_root


def preserve_failed_workspace(source, error):
    """Keep a failed-read source even if its best-effort backup is denied."""
    source = Path(source)
    backup = source.with_name(source.stem + '-invalid-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
    try:
        shutil.copy2(source, backup)
        message = f'工作区读取失败，已备份至 {backup}；自动保存将保留原件：{error}'
    except OSError as backup_error:
        message = f'工作区读取及备份失败，自动保存将保留原件 {source}：{error}；{backup_error}'
    logging.exception(message)
    return message


def load_saved_workspace_full(paths):
    from .document import Document
    from .workspace_state import read_workspace_full, synthetic_config
    protected, warnings = set(), []
    loaded = None
    for source in (paths['data'] / 'workspace.json', paths['draft']):
        try:
            if not source.is_file():
                continue
            loaded = read_workspace_full(source)
            break
        except Exception as error:
            protected.add(source.resolve())
            warnings.append(preserve_failed_workspace(source, error))
    if loaded is None:
        loaded = Document(), synthetic_config(), None
    # Runtime recovery metadata is deliberately outside the model payload.
    loaded[0].workspace_protected_paths = protected | getattr(loaded[0], 'workspace_protected_paths', set())
    loaded[0].workspace_load_warnings = warnings + getattr(loaded[0], 'workspace_load_warnings', [])
    return loaded


def load_saved_workspace(paths):
    return load_saved_workspace_full(paths)[:2]


def close_workers(manager):
    close_error = None
    try:
        manager.close()
    except Exception as error:
        close_error = error
    # The supervisor normally confirms cancellation; bound the final fallback
    # too, so a closing desktop window never leaves an owned process behind.
    for record in list(manager.runs.values()):
        process = record.get('_process')
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=3)
        monitor = record.get('_monitor')
        if monitor and monitor.is_alive():
            monitor.join(timeout=3)
    alive = [child.pid for child in multiprocessing.active_children() if child.is_alive()]
    if alive:
        raise RuntimeError('Owned workers did not stop: ' + str(alive))
    if close_error is not None:
        raise close_error


def main(argv=None):
    parser = argparse.ArgumentParser(description='ModelCircuit Studio native DearPyGui desktop')
    parser.add_argument('--data-dir', type=Path, help='Override the user-owned state directory')
    parser.add_argument('--self-test', type=Path, help='Run the real worker diagnostic and write its JSON report')
    parser.add_argument('--render-check', type=Path, help='Run in-process native render/callback diagnostics and write JSON')
    parser.add_argument('--classification-check', type=Path, help='Run real CSV/image classification worker diagnostics')
    parser.add_argument('--classification-render-check', type=Path, help='Run native real-data callback/render diagnostics')
    parser.add_argument('--check-device', default='cpu', help='Explicit diagnostic device, cpu or cuda:N')
    parser.add_argument('--s3-check', type=Path, help='Run paired data and named chip worker diagnostics')
    parser.add_argument('--s3-render-check', type=Path, help='Run paired data and chip editing native callback diagnostics')
    parser.add_argument('--s4-check', type=Path, help='Run S4 catalog, large-model worker and Python diagnostics')
    parser.add_argument('--s4-render-check', type=Path, help='Run S4 native catalog, chip and workspace callback diagnostics')
    parser.add_argument('--s6-render-check', type=Path, help='Run S6 regression through native controls and real workers')
    parser.add_argument("--s7-render-check", type=Path, help="Verify grouped 3D data and the native FitNets workflow")
    args = parser.parse_args(argv)
    diagnostics = [args.self_test, args.render_check, args.classification_check, args.classification_render_check,
                   args.s3_check, args.s3_render_check, args.s4_check, args.s4_render_check, args.s6_render_check, args.s7_render_check]
    if sum(bool(item) for item in diagnostics) > 1:
        parser.error('Choose only one diagnostic mode')
    output_report = next((item for item in diagnostics if item), None)
    paths, handler, manager, report = None, None, None, None
    return_code = 1
    try:
        paths = create_runtime_paths(resolve_data_directory(args.data_dir))
        handler = RotatingFileHandler(paths['logs'] / 'desktop.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s %(levelname)s %(message)s', force=True)
        backend = Path(__file__).resolve().parents[1] / 'backend'
        sys.path.insert(0, str(backend))
        from server import RunManager
        manager = RunManager(paths['runs'])
        logging.info('Native application starting; frozen=%s; data=%s', bool(getattr(sys, 'frozen', False)), paths['data'])
        if args.self_test:
            from .acceptance import check_runtime
            report = check_runtime(manager)
        elif args.render_check:
            from .render_acceptance import check_render
            report = check_render(manager, paths['data'], resource_root(), args.render_check.parent)
        elif args.classification_check:
            from .classification_acceptance import check_classification_runtime
            report = check_classification_runtime(manager, paths['data'], device=args.check_device)
        elif args.classification_render_check:
            from .classification_render_acceptance import check_classification_render
            report = check_classification_render(manager, paths['data'], resource_root(), args.classification_render_check.parent, device=args.check_device)
        elif args.s3_check:
            from .s3_acceptance import check_s3_runtime
            report = check_s3_runtime(manager,paths['data'],device=args.check_device)
        elif args.s3_render_check:
            from .s3_render_acceptance import check_s3_render
            report = check_s3_render(manager,paths['data'],resource_root(),args.s3_render_check.parent,device=args.check_device)
        elif args.s4_check:
            from .s4_acceptance import check_s4_runtime
            report = check_s4_runtime(manager, paths['data'], device=args.check_device)
        elif args.s4_render_check:
            from .s4_render_acceptance import check_s4_render
            report = check_s4_render(manager, paths['data'], resource_root(), args.s4_render_check.parent, device=args.check_device)
        elif args.s6_render_check:
            from .s6_acceptance import check_s6
            report = check_s6(manager, paths['data'], resource_root(), args.s6_render_check.parent, args.check_device)
        elif args.s7_render_check:
            from .s7_acceptance import check_s7
            report = check_s7(manager, paths['data'], resource_root(), args.s7_render_check.parent, args.check_device)
        else:
            document, experiment, workbench_payload = load_saved_workspace_full(paths)
            from .ui import NativeWorkbench
            NativeWorkbench(document, manager, paths['data'], resource_root(), experiment=experiment,
                            workbench_payload=workbench_payload).run()
        return_code = 0
    except Exception as error:
        logging.exception('Native application failed')
        report = {'status': 'failed', 'error': str(error)}
        if not output_report:
            import ctypes
            log_hint = str(paths['logs'] / 'desktop.log') if paths else '数据目录无法初始化，请选择可写目录。'
            ctypes.windll.user32.MessageBoxW(None, 'ModelCircuit Studio 启动或运行失败。\n' + str(error) + '\n日志：' + log_hint, 'ModelCircuit Studio', 0x10)
        return_code = 1
    finally:
        try:
            if manager is not None:
                close_workers(manager)
            if report is not None:
                report['workers_stopped'] = True
        except Exception as error:
            logging.exception('Worker shutdown failed')
            report = {'status': 'failed', 'error': str(error), 'workers_stopped': False}
            return_code = 1
        if output_report:
            try:
                output_report.parent.mkdir(parents=True, exist_ok=True)
                output_report.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
            except OSError:
                logging.exception('Diagnostic report could not be saved')
                return_code = 1
        logging.info('Native application exiting')
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
    return return_code
