"""Local coordinator for immutable experiments and independent workers."""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import math
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from studio_core.classical_task import classical_contract, contains_classical
from studio_core.examples import example_graph
from studio_core.python_io import export_python, import_python, python_profile
from studio_core.schema import validate_graph
from studio_core.tensor_specs import require_executable_profile
from studio_core.experiment_spec import SUPERVISED_PROFILES, validate_supervised_config
from studio_core.inference_spec import PENDING_REPORT, PROFILE as INFERENCE_PROFILE, REPORT_NAME
from worker import experiment_worker

TERMINAL = {"succeeded", "failed", "cancelled"}
MAX_BODY = 1_000_000
MAX_RESULT_BYTES = 2_000_000
MAX_STORED_BYTES = 8_000_000


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _wire_json(value, limit, label):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError(label + " must contain finite JSON values") from error
    if len(encoded) > limit:
        raise ValueError(label + " exceeds the serialized byte budget")
    return encoded


class RunManager:
    """A run owns its process, queue, cancellation event and immutable snapshot."""

    def __init__(self, directory, worker_target=experiment_worker, max_active=2, timeout=120):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker_target = worker_target
        self.max_active = max_active
        self.timeout = timeout
        self.context = mp.get_context("spawn")
        self.lock = threading.RLock()
        self.runs = {}
        self.closed = False

    def _public(self, record):
        return copy.deepcopy({key: value for key, value in record.items() if not key.startswith("_") and key not in {"graph", "config"}})

    def _save(self, record):
        persisted = {key: value for key, value in record.items() if not key.startswith("_")}
        _wire_json(persisted, MAX_STORED_BYTES, "Run artifact")
        target = self.directory / (record["run_id"] + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(persisted, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        # Windows readers/virus scanners can briefly deny delete-sharing on
        # an otherwise writable record. Keep atomic replacement and retry only
        # that transient permission failure within a bounded interval.
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.03 * (2 ** attempt))

    def _event(self, record, kind, **values):
        record["_seq"] += 1
        record["events"].append({"run_id": record["run_id"], "revision": record["revision"],
                                  "seq": record["_seq"], "type": kind, **values})
        record["events"] = record["events"][-512:]

    def submit(self, graph, config):
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError("; ".join(issue["message"] for issue in report["errors"]))
        _wire_json({"graph": graph, "config": config}, MAX_BODY, "Run snapshot")
        # Validation never imports torch or scans a dataset on the UI thread.
        supervised = isinstance(config.get("profile"), str) and config["profile"] in SUPERVISED_PROFILES
        typed = graph.get('schema_version') == 2
        classical = contains_classical(graph)
        if classical and not supervised:
            classical_contract(graph, config)  # Reports the required CSV task.
        if not supervised:
            require_executable_profile(graph)
        elif 'attention_probe_requests' in config and not typed:
            raise ValueError('Attention probes require tensor-v1')
        if supervised:
            config = validate_supervised_config(config)
            if classical:
                classical_contract(graph, config)
            elif config['data']['kind'] == 'table_csv':
                raise ValueError('table_csv requires a classical estimator with a single prediction output')
        else:
            allowed = {"steps", "lr", "seed", "batch_size"}
            if not config.keys() <= allowed:
                raise ValueError("Unsupported experiment configuration key")
            for name, default, low, high in (("steps", 20, 1, 200), ("seed", 42, 0, 2**31-1), ("batch_size", 16, 2, 64)):
                value = config.get(name, default)
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(name + " is outside the supported S1 range")
            lr = config.get("lr", 0.05)
            if type(lr) not in (int, float) or not 0.001 <= lr <= 1:
                raise ValueError("lr must be between 0.001 and 1")
        return self._start_worker(graph, config, managed=supervised, classical=classical)

    def submit_debug(self, graph, config):
        from studio_core.debug_spec import validate_debug_config
        config = validate_debug_config(config)
        _wire_json({'graph': graph, 'config': config}, MAX_BODY, 'Debug snapshot')
        report = validate_graph(graph)
        if not report['valid']:
            raise ValueError(report['errors'][0]['message'])
        if contains_classical(graph):
            raise ValueError('传统机器学习模型请在实验页使用拟合和评估。调试单步用于 PyTorch 模型。')
        require_executable_profile(graph)
        from studio_core.execution_policy import check_config_batch
        check_config_batch(graph, config['batch_size'])
        return self._start_worker(graph, config, managed=False, classical=False)

    def submit_segmentation_inference(self, config):
        from studio_core.inference_spec import validate_inference_config
        config = validate_inference_config(config)
        _wire_json(config, MAX_BODY, 'Inference settings')
        return self._start_worker(None, config, managed=True, classical=False)

    def submit_project(self, config):
        from studio_core.project_runner import validate_project_job
        config = validate_project_job(config)
        _wire_json(config, MAX_BODY, 'Project job settings')
        return self._start_worker(None, config, managed=True, classical=False)

    def _start_worker(self, graph, config, *, managed, classical):
        with self.lock:
            if self.closed:
                raise ValueError("Coordinator is shutting down")
            device = config.get("device", "cpu")
            if device.startswith("cuda:") and any(
                record["status"] not in TERMINAL and record["config"].get("device") == device
                for record in self.runs.values()
            ):
                raise ValueError(device + " 已被本工作台的另一个实验占用，请等待或停止该实验")
            if sum(record["status"] not in TERMINAL for record in self.runs.values()) >= self.max_active:
                raise ValueError("本地最多同时运行两个实验，请等待或停止现有实验")
            # Keep live records bounded; completed JSON artifacts remain on disk.
            for old_id in list(self.runs):
                if len(self.runs) < 64:
                    break
                if self.runs[old_id]["status"] in TERMINAL:
                    self.runs.pop(old_id)
            run_id = uuid.uuid4().hex
            messages, stop = self.context.Queue(), self.context.Event()
            record = {"run_id": run_id, "status": "queued", "revision": graph.get("revision", 1) if graph is not None else None,
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "profile": config.get("profile", "s1-pytorch-v1"),
                      **({"execution": "classical"} if classical else {}),
                      "progress": {"step": 0, "total": 0 if managed else config.get("steps", 20), "loss": None},
                      "result": None, "error": None, "events": [],
                      "graph": copy.deepcopy(graph), "config": copy.deepcopy(config),
                      "_queue": messages, "_stop": stop, "_seq": 0, "_cancel_at": None,
                      "_started": time.monotonic(),
                      "_time_limit": config["max_runtime_seconds"] if managed else self.timeout}
            worker_options = {"artifact_directory": str(self.directory / run_id)} if managed else {}
            process = self.context.Process(target=self.worker_target,
                                           args=(record["graph"], record["config"], messages, stop),
                                           kwargs=worker_options,
                                           daemon=True, name="mlforge-" + run_id[:8])
            record["_process"] = process
            self.runs[run_id] = record
            self._event(record, "status", status="queued")
            try:
                self._save(record)
            except Exception:
                self.runs.pop(run_id, None)
                messages.close()
                raise
            try:
                process.start()
            except Exception as error:
                record["status"], record["error"] = "failed", str(error)
                self._event(record, "error", message=str(error))
                self._save(record)
                raise ValueError("Unable to start worker: " + str(error)) from error
            monitor = threading.Thread(target=self._monitor, args=(record,), daemon=True,
                                       name="supervisor-" + run_id[:8])
            record["_monitor"] = monitor
            monitor.start()
            return self._public(record)

    def _receive(self, record, event):
        try:
            _wire_json(event, MAX_RESULT_BYTES if event.get("type") == "result" else 32_000, "Worker result" if event.get("type") == "result" else "Worker event")
        except (ValueError, AttributeError) as error:
            record["_pending_error"] = str(error)
            record.pop("_pending_result", None)
            return
        kind = event.get("type")
        if kind == "status" and record["_cancel_at"] is None:
            if event.get("status") not in {"preparing", "running"}:
                record["_pending_error"] = "Worker cannot declare a terminal state before process exit"
                return
            record["status"] = event["status"]
            self._event(record, "status", status=event["status"])
        elif kind == "progress":
            allowed = {"step", "total", "loss", "phase", "epoch", "epochs", "batch", "batches",
                       "train_loss", "val_loss", "train_accuracy", "val_accuracy",
                       "stage", "train_mae", "val_mae", "train_rmse", "val_rmse", "train_r2", "val_r2"}
            progress = {name: value for name, value in event.items() if name in allowed}
            if any(type(value) not in (int, float, str, type(None)) for value in progress.values()):
                record["_pending_error"] = "Worker progress must contain scalar values"
                return
            record["progress"] = {"step": 0, "total": 0, "loss": None, **progress}
            self._event(record, "progress", **record["progress"])
        elif kind == "result":
            record["_pending_result"] = event["result"]
        elif kind == "error":
            record["_pending_error"] = event.get("error", "Worker error")
            record["worker_traceback"] = event.get("traceback", "")
        elif kind == "cancelled":
            record["_worker_cancelled"] = True

    def _finish_managed_report(self, record):
        from studio_core.project_runner import PROFILE as PROJECT_PROFILE, finalize_project_report
        if record['profile'] == PROJECT_PROFILE:
            finalize_project_report(self.directory / record['run_id'], record)
            return
        if record['profile'] != INFERENCE_PROFILE:
            return
        directory = self.directory / record['run_id']
        if record['status'] == 'succeeded':
            (directory / PENDING_REPORT).replace(directory / REPORT_NAME)
        else:
            (directory / PENDING_REPORT).unlink(missing_ok=True)
            (directory / REPORT_NAME).unlink(missing_ok=True)

    def _monitor(self, record):
        process, messages = record["_process"], record["_queue"]
        try:
            while True:
                try:
                    event = messages.get(timeout=0.05)
                    with self.lock:
                        self._receive(record, event)
                except queue.Empty:
                    pass
                now = time.monotonic()
                if record["_cancel_at"] is not None and now - record["_cancel_at"] > 2 and process.is_alive():
                    process.terminate()
                if now - record["_started"] > record["_time_limit"] and process.is_alive():
                    record["_pending_error"] = "Worker exceeded the local execution time limit"
                    process.terminate()
                if not process.is_alive():
                    process.join(timeout=1)
                    while True:
                        try:
                            event = messages.get_nowait()
                            with self.lock:
                                self._receive(record, event)
                        except queue.Empty:
                            break
                    with self.lock:
                        if record["_cancel_at"] is not None or record.get("_worker_cancelled"):
                            record["status"] = "cancelled"
                        elif record.get("_pending_error"):
                            record["status"], record["error"] = "failed", record["_pending_error"]
                        elif "_pending_result" in record and process.exitcode == 0:
                            record["status"], record["result"] = "succeeded", record["_pending_result"]
                        else:
                            record["status"] = "failed"
                            record["error"] = "Worker exited without a result (exit code " + str(process.exitcode) + ")"
                        record["exit_code"] = process.exitcode
                        self._finish_managed_report(record)
                        self._event(record, "status", status=record["status"], error=record["error"])
                        self._save(record)
                    break
        except Exception as error:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            with self.lock:
                record["status"], record["error"] = "failed", "Supervisor error: " + str(error)
                record["result"] = None
                record.pop("_pending_result", None)
                self._event(record, "error", message=record["error"])
                try:
                    self._finish_managed_report(record)
                    self._save(record)
                except (OSError, ValueError, TypeError) as persistence_error:
                    record["persistence_error"] = str(persistence_error)
        finally:
            messages.close()

    def get(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError("Run not found")
            return self._public(self.runs[run_id])

    def cancel(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError("Run not found")
            record = self.runs[run_id]
            if record["status"] not in TERMINAL and record["_cancel_at"] is None:
                record["_cancel_at"] = time.monotonic()
                record["_stop"].set()
                record["status"] = "cancel_requested"
                self._event(record, "status", status="cancel_requested")
                self._save(record)
            return self._public(record)

    def close(self):
        with self.lock:
            self.closed = True
            records = list(self.runs.values())
        for record in records:
            if record["status"] not in TERMINAL:
                self.cancel(record["run_id"])
        for record in records:
            monitor = record.get("_monitor")
            if monitor:
                monitor.join(timeout=5)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "MLForgeStudio/0.1"

    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        try:
            raw = _wire_json(payload, MAX_STORED_BYTES, "API response")
        except ValueError:
            status, raw = 500, b'{"error":"Response could not be represented within the finite JSON budget"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if not 0 < length <= MAX_BODY:
            raise ValueError("Request body must be between 1 and 1000000 bytes")
        self.connection.settimeout(10)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"), parse_float=_finite_float, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/health":
            available = importlib.util.find_spec("torch") is not None
            version = importlib.metadata.version("torch") if available else None
            return self._json(200, {"ok": True, "version": "0.1.0", "torch_available": available,
                                    "torch_version": version, "capabilities": ["s1-pytorch-v1", "classification-pytorch-v2", "classification-v2", "regression-v1", "segmentation-v1", "local-cpu", "hierarchical-chips", "measured-probes"]})
        if route == "/api/example":
            return self._json(200, {"graph": example_graph()})
        if route.startswith("/api/runs/"):
            try:
                return self._json(200, self.server.manager.get(route[len("/api/runs/"):]))
            except KeyError:
                return self._json(404, {"error": "Run not found"})
        self._json(404, {"error": "Unknown API route"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed = {"http://127.0.0.1:4173", "http://localhost:4173", "http://127.0.0.1:" + str(self.server.server_port)}
        if origin and origin not in allowed:
            return self._json(403, {"error": "Cross-site requests are not allowed"})
        try:
            body = self._body()
        except (ValueError, OSError) as error:
            return self._json(400, {"error": str(error)})
        route = urlsplit(self.path).path
        try:
            if route == "/api/validate":
                return self._json(200, validate_graph(body.get("graph")))
            if route == "/api/export":
                graph = body.get('graph')
                code = export_python(graph)
                return self._json(200, {"code": code, "profile": python_profile(graph)})
            if route == "/api/import":
                code = body.get("code")
                if not isinstance(code, str) or len(code) > 200_000:
                    raise ValueError("Python source must be a string of at most 200000 characters")
                graph = import_python(code)
                return self._json(200, {"graph": graph, "profile": python_profile(graph)})
            if route == "/api/runs":
                return self._json(201, self.server.manager.submit(body.get("graph"), body.get("config", {})))
            if route.startswith("/api/runs/") and route.endswith("/cancel"):
                run_id = route[len("/api/runs/"):-len("/cancel")]
                return self._json(200, self.server.manager.cancel(run_id))
            return self._json(404, {"error": "Unknown API route"})
        except KeyError:
            self._json(404, {"error": "Run not found"})
        except (ValueError, TypeError, SyntaxError) as error:
            self._json(422, {"error": str(error)})
        except Exception:
            self._json(500, {"error": "Local service failed to process this request"})


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        self.manager = manager
        super().__init__(address, RequestHandler)


def main():
    parser = argparse.ArgumentParser(description="ML Forge Studio local CPU coordinator")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manager = RunManager(Path(__file__).resolve().parents[1] / "runs")
    server = StudioServer(("127.0.0.1", args.port), manager)
    print("ML Forge Studio API listening on http://127.0.0.1:" + str(args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()

