"""HTTP contracts and actual process isolation, without a training dependency."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from studio_core.examples import example_graph

SERVER_SPEC = importlib.util.find_spec("server")
if SERVER_SPEC:
    from server import RunManager, StudioServer


def controlled_worker(graph, config, queue, stop):
    queue.put({"type": "status", "status": "running"})
    if config.get("steps", 1) == 200:
        while not stop.wait(0.01):
            pass
        queue.put({"type": "cancelled"})
        return
    queue.put({"type": "progress", "step": 1, "total": 1, "loss": 0.5})
    queue.put({"type": "result", "result": {"status": "succeeded", "input_id": graph["id"]}})


def crashing_worker(graph, config, queue, stop):
    os._exit(17)


def nonfinite_worker(graph, config, queue, stop):
    queue.put({"type": "result", "result": {"status": "succeeded", "unsafe": float("inf")}})


def oversized_worker(graph, config, queue, stop):
    queue.put({"type": "result", "result": {"status": "succeeded", "data": "x" * 2_000_001}})


class ServerContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SERVER_SPEC, "server.py must provide the HTTP coordinator")
        self.temp = tempfile.TemporaryDirectory()
        self.manager = RunManager(Path(self.temp.name), worker_target=controlled_worker)
        self.server = StudioServer(("127.0.0.1", 0), self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.manager.close()
            self.temp.cleanup()

    def request(self, path, data=None, raw=None, headers=None):
        body = raw if raw is not None else json.dumps(data).encode() if data is not None else None
        request = Request(self.url + path, data=body,
                          headers=headers or ({"Content-Type": "application/json"} if body is not None else {}))
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def terminal(self, run_id):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            record = self.manager.get(run_id)
            if record["status"] in {"succeeded", "failed", "cancelled"}:
                return record
            time.sleep(0.025)
        self.fail("Worker did not reach a confirmed terminal state")

    def test_example_and_validation(self):
        status, payload = self.request("/api/example")
        self.assertEqual(status, 200)
        status, report = self.request("/api/validate", {"graph": payload["graph"]})
        self.assertEqual(status, 200)
        self.assertTrue(report["valid"])

    def test_round_trip_uses_current_code(self):
        _, exported = self.request("/api/export", {"graph": example_graph()})
        self.assertIn("class ", exported["code"])
        status, imported = self.request("/api/import", {"code": exported["code"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(imported["graph"]["calls"]), 4)
        calls = imported["graph"]["calls"]
        self.assertEqual(calls[0]["instance_id"], calls[1]["instance_id"])

    def test_export_import_report_the_actual_classification_model_profile(self):
        graph=example_graph();graph['resource_profile']='classification-v2'
        status,exported=self.request('/api/export',{'graph':graph})
        self.assertEqual(status,200)
        self.assertEqual(exported['profile'],'classification-pytorch-v2')
        status,imported=self.request('/api/import',{'code':exported['code']})
        self.assertEqual(status,200)
        self.assertEqual(imported['profile'],'classification-pytorch-v2')

    def test_invalid_json_and_unsupported_code_fail(self):
        self.assertEqual(self.request("/api/import", raw=b"{broken", headers={"Content-Type": "application/json"})[0], 400)
        status, error = self.request("/api/import", {"code": "import os\nos.system('echo SHOULD_NOT_RUN')"})
        self.assertEqual(status, 422)
        self.assertIn("error", error)

    def test_invalid_graph_does_not_launch(self):
        graph = example_graph()
        graph["calls"][0]["inputs"]["x"] = "call:missing"
        status, payload = self.request("/api/runs", {"graph": graph, "config": {}})
        self.assertEqual(status, 422)
        self.assertIn("error", payload)

    def test_missing_run_and_unknown_route(self):
        self.assertEqual(self.request("/api/runs/not-found")[0], 404)
        self.assertEqual(self.request("/api/not-found")[0], 404)

    def test_run_snapshot_and_persisted_result(self):
        graph = example_graph()
        status, run = self.request("/api/runs", {"graph": graph, "config": {"steps": 1}})
        self.assertEqual(status, 201)
        graph["id"] = "changed_after_submit"
        record = self.terminal(run["run_id"])
        self.assertEqual(record["status"], "succeeded")
        self.assertNotEqual(record["result"]["input_id"], graph["id"])
        saved = json.loads((Path(self.temp.name) / (run["run_id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "succeeded")
        self.assertEqual(saved["revision"], run["revision"])
        self.assertTrue(all(event["run_id"] == run["run_id"] for event in record["events"]))

    def test_cancel_is_isolated_and_confirmed(self):
        first = self.manager.submit(example_graph(), {"steps": 200})
        second = self.manager.submit(example_graph(), {"steps": 1})
        self.manager.cancel(first["run_id"])
        cancelled = self.terminal(first["run_id"])
        succeeded = self.terminal(second["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["result"])
        self.assertEqual(succeeded["status"], "succeeded")

    def test_cross_site_origin_rejected(self):
        status, _ = self.request("/api/validate", {"graph": example_graph()},
                                 headers={"Content-Type": "application/json", "Origin": "https://unrelated.example"})
        self.assertEqual(status, 403)

    def test_exponent_overflow_is_rejected_at_json_boundary_and_does_not_fill_slots(self):
        graph = example_graph()
        graph["name"] = "EXPONENT_OVERFLOW"
        raw = json.dumps({"graph": graph, "config": {}}).replace('"EXPONENT_OVERFLOW"', '1e400').encode()
        for _ in range(2):
            status, _ = self.request("/api/runs", raw=raw, headers={"Content-Type": "application/json"})
            self.assertEqual(status, 400)
        self.assertFalse(self.manager.runs)
        status, accepted = self.request("/api/runs", {"graph": example_graph(), "config": {"steps": 1}})
        self.assertEqual(status, 201)
        self.assertEqual(self.terminal(accepted["run_id"])["status"], "succeeded")

    def test_prepare_persistence_failure_rolls_back_slot_before_start(self):
        with patch.object(self.manager, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.manager.submit(example_graph(), {})
        self.assertFalse(self.manager.runs)
        accepted = self.manager.submit(example_graph(), {"steps": 1})
        self.assertEqual(self.terminal(accepted["run_id"])["status"], "succeeded")


class ProcessExitContracts(unittest.TestCase):
    def assert_bad_result_reliably_failed(self, target, expected_text):
        with tempfile.TemporaryDirectory() as temp:
            manager = RunManager(Path(temp), worker_target=target)
            try:
                run = manager.submit(example_graph(), {})
                until = time.monotonic() + 8
                while time.monotonic() < until:
                    record = manager.get(run["run_id"])
                    if record["status"] in {"succeeded", "failed"}:
                        self.assertEqual(record["status"], "failed")
                        self.assertIn(expected_text, record["error"])
                        self.assertIsNone(record["result"])
                        json.dumps(record, allow_nan=False)
                        saved = json.loads((Path(temp) / (run["run_id"] + ".json")).read_text(encoding="utf-8"))
                        self.assertEqual(saved["status"], "failed")
                        return
                    time.sleep(0.025)
                self.fail("Invalid worker result did not reach a durable terminal failure")
            finally:
                manager.close()

    def test_nonfinite_worker_result_does_not_break_polling_or_persistence(self):
        self.assertIsNotNone(SERVER_SPEC, "server.py is required")
        self.assert_bad_result_reliably_failed(nonfinite_worker, "JSON")

    def test_oversized_worker_result_is_rejected_before_attaching_to_public_record(self):
        self.assertIsNotNone(SERVER_SPEC, "server.py is required")
        self.assert_bad_result_reliably_failed(oversized_worker, "budget")

    def test_crash_is_failed_not_stuck_running(self):
        self.assertIsNotNone(SERVER_SPEC, "server.py must provide the process supervisor")
        with tempfile.TemporaryDirectory() as temp:
            manager = RunManager(Path(temp), worker_target=crashing_worker)
            try:
                run = manager.submit(example_graph(), {})
                until = time.monotonic() + 8
                while time.monotonic() < until:
                    record = manager.get(run["run_id"])
                    if record["status"] == "failed":
                        self.assertIn("17", record["error"])
                        return
                    time.sleep(0.025)
                self.fail("Crashed worker was not supervised")
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
