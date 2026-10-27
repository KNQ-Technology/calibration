"""Headless regression tests for measurement liveness and retry handling.

Extract the production methods from the AST so importing the Tk application
does not create a window or perform its import-time environment setup.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import MethodType, SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"
METHODS = (
    "_measurement_task_active",
    "_finish_measurement_if_done",
    "_poll_measurement",
    "_force_stop_measurement",
    "_confirm_measurement_retry",
    "_run_measurement_worker",
)


def _load_methods(messagebox, *, popen=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    assert len(methods) == len(METHODS), "A measurement retry method is missing"
    namespace = {
        "messagebox": messagebox,
        "os": os,
        "sys": sys,
        "subprocess": (subprocess if popen is None else SimpleNamespace(
            Popen=popen, PIPE=subprocess.PIPE)),
        "threading": threading,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class _Worker:
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


class _Process:
    def __init__(self, live=True):
        self.live = live
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None if self.live else 0

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1
        self.live = False


class _App:
    def __init__(self, *, confirm=False):
        self.prompts = []
        self.infos = []
        self.after_calls = []
        self.idle_calls = []
        self.logs = []
        self.status_values = []
        self.refreshes = []
        self._measurement_run_id = 1
        self._measurement_running = False
        self._measurement_worker = None
        self._measurement_process = None
        self._measurement_cancel = None
        self._measurement_result = None
        self._measurement_restart_pending = False
        self._measurement_log_window = None
        self._measurement_log_text = None
        self.status = SimpleNamespace(set=self.status_values.append)
        self.save_and_run_measurement = lambda: self.refreshes.append("restart")

        def askyesno(title, text):
            self.prompts.append((title, text))
            if hasattr(self, "on_prompt"):
                self.on_prompt()
            return confirm

        messagebox = SimpleNamespace(
            askyesno=askyesno,
            showinfo=lambda title, text: self.infos.append((title, text)),
        )
        for name, method in _load_methods(messagebox).items():
            if name in METHODS:
                setattr(self, name, MethodType(method, self))

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))

    def after_idle(self, callback):
        self.idle_calls.append(callback)

    def _append_measurement_log(self, _log_text, output):
        self.logs.append(output)

    def _refresh_step_list(self):
        self.refreshes.append("steps")

    def _refresh_finish_summary(self):
        self.refreshes.append("summary")


class MeasurementRetryTests(unittest.TestCase):
    def test_stale_running_flag_asks_before_retry_and_cancel_does_not_restart(self):
        app = _App(confirm=False)
        app._measurement_running = True

        self.assertFalse(app._confirm_measurement_retry())
        self.assertFalse(app._measurement_running)
        self.assertEqual(len(app.prompts), 1)
        self.assertIn("上一次测量已经结束", app.prompts[0][1])
        self.assertEqual(app.status_values[-1], "测量异常结束")
        self.assertEqual(app.idle_calls, [])

    def test_stale_running_flag_can_be_confirmed_for_new_measurement(self):
        app = _App(confirm=True)
        app._measurement_running = True

        self.assertTrue(app._confirm_measurement_retry())
        self.assertFalse(app._measurement_running)
        self.assertEqual(len(app.prompts), 1)
        self.assertFalse(app._measurement_restart_pending)

    def test_live_measurement_cancel_keeps_original_task_running(self):
        app = _App(confirm=False)
        app._measurement_running = True
        app._measurement_worker = _Worker(True)
        app._measurement_process = _Process()
        app._measurement_cancel = threading.Event()

        self.assertFalse(app._confirm_measurement_retry())
        self.assertTrue(app._measurement_running)
        self.assertTrue(app._measurement_task_active())
        self.assertFalse(app._measurement_cancel.is_set())
        self.assertEqual(app._measurement_process.terminated, 0)
        self.assertEqual(app.idle_calls, [])
        self.assertIn("当前测量仍在运行", app.prompts[0][1])

    def test_live_measurement_confirmation_waits_for_old_worker_to_exit(self):
        app = _App(confirm=True)
        app._measurement_running = True
        worker = app._measurement_worker = _Worker(True)
        process = app._measurement_process = _Process()
        app._measurement_cancel = threading.Event()

        self.assertFalse(app._confirm_measurement_retry())
        self.assertTrue(app._measurement_cancel.is_set())
        self.assertEqual(process.terminated, 1)
        self.assertTrue(app._measurement_restart_pending)
        self.assertEqual(app.idle_calls, [])  # no overlapping second measurement
        self.assertFalse(app._finish_measurement_if_done(1))
        self.assertEqual(app._confirm_measurement_retry(), False)
        self.assertEqual(len(app.infos), 1)
        self.assertEqual(len(app.prompts), 1)

        worker.alive = False
        process.live = False
        app._measurement_result = (1, "已停止\n", "测量已取消")
        self.assertTrue(app._finish_measurement_if_done(1))
        self.assertFalse(app._measurement_running)
        self.assertFalse(app._measurement_restart_pending)
        self.assertEqual(len(app.idle_calls), 1)
        app.idle_calls[0]()
        self.assertIn("restart", app.refreshes)

    def test_worker_finishes_during_confirmation_and_retries_without_cancel(self):
        app = _App(confirm=True)
        app._measurement_running = True
        worker = app._measurement_worker = _Worker(True)
        process = app._measurement_process = _Process()
        app._measurement_cancel = threading.Event()

        def finish_while_prompt_is_open():
            worker.alive = False
            process.live = False
            app._measurement_result = (1, "旧任务完成\n", "测量完成")

        app.on_prompt = finish_while_prompt_is_open
        self.assertTrue(app._confirm_measurement_retry())
        self.assertFalse(app._measurement_running)
        self.assertFalse(app._measurement_restart_pending)
        self.assertEqual(process.terminated, 0)
        self.assertEqual(app.after_calls, [])
        self.assertEqual(app.idle_calls, [])
        self.assertEqual(app.status_values[-1], "测量完成")

    def test_poll_reconciles_completed_worker_and_process(self):
        app = _App()
        app._measurement_running = True
        worker = app._measurement_worker = _Worker(True)
        process = app._measurement_process = _Process()
        app._measurement_log_window = SimpleNamespace(winfo_exists=lambda: True)
        app._measurement_log_text = object()

        app._poll_measurement(1)
        self.assertEqual(len(app.after_calls), 1)
        self.assertEqual(app.after_calls[0][0], 200)
        worker.alive = False
        process.live = False
        app._measurement_result = (1, "报告生成完成\n", "测量完成")
        app.after_calls[0][1]()

        self.assertFalse(app._measurement_running)
        self.assertEqual(app.status_values[-1], "测量完成")
        self.assertEqual(app.logs, ["报告生成完成\n"])
        self.assertEqual(app.after_calls[1:], [])
        self.assertIn("summary", app.refreshes)

    def test_old_poll_and_completion_cannot_reset_new_run(self):
        app = _App()
        app._measurement_run_id = 2
        app._measurement_running = True
        app._measurement_worker = _Worker(True)

        app._poll_measurement(1)
        self.assertFalse(app._finish_measurement_if_done(1))
        self.assertTrue(app._measurement_running)
        self.assertEqual(app.after_calls, [])
        self.assertEqual(app.status_values, [])

    def test_force_stop_kills_only_current_pending_retry(self):
        app = _App()
        app._measurement_process = _Process()
        app._measurement_restart_pending = True

        app._force_stop_measurement(0)
        self.assertEqual(app._measurement_process.killed, 0)
        app._force_stop_measurement(1)
        self.assertEqual(app._measurement_process.killed, 1)

    def test_worker_records_subprocess_success_and_failure(self):
        for returncode, expected_status in ((0, "测量完成"), (7, "测量失败")):
            with self.subTest(returncode=returncode):
                calls = []

                class FakeProcess:
                    def __init__(self):
                        self.returncode = returncode

                    def poll(self):
                        return returncode

                    def communicate(self, **kwargs):
                        calls.append(("communicate", kwargs))
                        return "计算输出\n", "错误详情\n" if returncode else ""

                process = FakeProcess()

                def fake_popen(*args, **kwargs):
                    calls.append(("popen", args, kwargs))
                    return process

                worker = _load_methods(SimpleNamespace(), popen=fake_popen)[
                    "_run_measurement_worker"]
                app = SimpleNamespace(
                    project_dir=Path("/calibration"),
                    _measurement_process=None,
                    _measurement_result=None,
                    _manual_dir=lambda: Path("/calibration/store/manual/camera1"),
                )
                worker(app, 3, threading.Event())

                self.assertIs(app._measurement_process, process)
                self.assertEqual(calls[0][0], "popen")
                self.assertEqual(calls[1], ("communicate", {"input": "y\n"}))
                self.assertEqual(app._measurement_result[0], 3)
                self.assertEqual(app._measurement_result[2], expected_status)
                self.assertIn("计算输出", app._measurement_result[1])
                if returncode:
                    self.assertIn("错误详情", app._measurement_result[1])
                    self.assertIn("退出码：7", app._measurement_result[1])
                else:
                    self.assertIn("结果目录", app._measurement_result[1])

    def test_worker_skips_spawn_when_cancelled_before_launch(self):
        def forbidden_popen(*_args, **_kwargs):
            self.fail("cancelled worker must not start subprocess")

        worker = _load_methods(SimpleNamespace(), popen=forbidden_popen)[
            "_run_measurement_worker"]
        app = SimpleNamespace(_measurement_process=None, _measurement_result=None)
        cancelled = threading.Event()
        cancelled.set()
        worker(app, 4, cancelled)

        self.assertIsNone(app._measurement_process)
        self.assertEqual(app._measurement_result[2], "测量已取消")

    def test_worker_cancels_process_if_retry_requested_during_launch(self):
        cancelled = threading.Event()

        class FakeProcess:
            returncode = -15

            def __init__(self):
                self.terminated = 0

            def poll(self):
                return None

            def terminate(self):
                self.terminated += 1

            def communicate(self, **_kwargs):
                return "部分输出\n", ""

        process = FakeProcess()

        def fake_popen(*_args, **_kwargs):
            cancelled.set()
            return process

        worker = _load_methods(SimpleNamespace(), popen=fake_popen)[
            "_run_measurement_worker"]
        app = SimpleNamespace(project_dir=Path("/calibration"),
                              _measurement_process=None, _measurement_result=None)
        worker(app, 5, cancelled)

        self.assertEqual(process.terminated, 1)
        self.assertEqual(app._measurement_result[2], "测量已取消")
        self.assertIn("准备重新测量", app._measurement_result[1])


if __name__ == "__main__":
    unittest.main()
