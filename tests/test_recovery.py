"""Deterministic orchestration tests; synthetic files live only in temp folders.

No Monte Carlo panels or official RNG streams are produced by this test suite.
The separate, bounded real-input replay is recorded outside the code package.
"""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import recovery
import START_HERE
import run_simulation
from simulation.storage import load, save, sha


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "MOCK_INTERRUPTED_RUN"
        self.study = self.run / "main_study"
        self.partial = self.study / "panels/h9/FORMAL_001479"
        self.log = self.study / "logs/h9/FORMAL_001479.log"
        self.area = self.study / "recovery_disk_001"
        self.values = [{"fixed_fixture": name} for name in recovery.INPUT_NAMES]
        self.hashes = {}
        for name, value in zip(recovery.INPUT_NAMES, self.values):
            path = self.root / "expected_bytes" / name
            save(path, value)
            self.hashes[name] = sha(path)
            if name != "uniforms.json.gz":
                save(self.partial / name, value)
        (self.partial / "uniforms.json.gz.pending").touch()
        save(self.partial / "RESOURCE_RECORD.json", {
            "elapsed_seconds": 1.25, "peak_sampled_rss_bytes": 100,
            "returncode": -9, "failure": "Available disk below 4 GiB"})
        self.log.parent.mkdir(parents=True)
        self.log.write_text("original disk-stop log\n")
        save(self.study / "REPRODUCTION_IDENTITY.json", {"fixture": True})
        save(self.run / "PIPELINE_STOPPED.json", {"fixture": "original failure"})
        paths = [*self.partial.iterdir(), self.log,
                 self.study / "REPRODUCTION_IDENTITY.json", self.run / "PIPELINE_STOPPED.json"]
        self.approval = {"run_name": self.run.name, "h": 9, "panel": "FORMAL_001479",
                         "completed_counts": {"3": 5000, "6": 5000, "9": 1479},
                         "evidence": {str(p.relative_to(self.run)): {
                             "sha256": sha(p), "bytes": p.stat().st_size} for p in paths}}
        approval_path = self.root / "approval.json"
        save(approval_path, self.approval)
        fake_package = self.root / "mock_package"
        save(fake_package / "expected/original_scientific_hashes.json.gz", {
            "studies": {"step5": {"h9/FORMAL_001479": self.hashes}}})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(recovery, "APPROVAL", approval_path))
        self.stack.enter_context(patch.object(recovery, "PACKAGE", fake_package))
        self.generator = self.stack.enter_context(patch(
            "simulation.main_study.generate", return_value=tuple(self.values)))
        self.stack.enter_context(patch.object(recovery, "frozen_seed", return_value="SAME_FROZEN_SEED"))

    def replay(self):
        with redirect_stdout(StringIO()):
            recovery.reconstruct_input(self.run, self.area / "prepared")

    def test_same_address_seed_and_atomic_input_commit(self):
        self.replay()
        args, kw = self.generator.call_args
        self.assertEqual(args[:4], (9, "FORMAL_001479", "SAME_FROZEN_SEED", recovery.RUN))
        self.assertEqual(kw, {"stage": "formal"})
        commit = load(self.area / "prepared/INPUT_COMMIT.json")
        self.assertEqual(commit["files"], self.hashes)
        self.assertEqual(commit["identity"], {"study": "step5", "h": 9, "panel": "FORMAL_001479"})
        self.assertEqual(list((self.area / "prepared").glob("*.pending")), [])
        self.assertFalse((self.area / "prepared/prediction.json.gz").exists())

    def test_existing_saved_inputs_do_not_change(self):
        self.replay()
        recovery.verify_evidence(self.run)
        for name in recovery.INPUT_NAMES[:-1]:
            self.assertEqual(sha(self.partial / name), self.hashes[name])

    def test_replay_mismatch_stops_before_commit_or_archive(self):
        self.generator.return_value = ({"wrong": 1}, *self.values[1:])
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            self.replay()
        self.assertFalse((self.area / "prepared/INPUT_COMMIT.json").exists())
        self.assertFalse((self.area / "archived").exists())
        recovery.verify_evidence(self.run)

    def test_changed_saved_input_fails_before_rng(self):
        (self.partial / "visible.json.gz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Changed interruption evidence"):
            self.replay()
        self.generator.assert_not_called()

    def test_changed_pending_file_fails_before_rng(self):
        (self.partial / "uniforms.json.gz.pending").write_bytes(b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "Changed interruption evidence"):
            self.replay()
        self.generator.assert_not_called()

    def test_changed_identity_fails_before_rng(self):
        (self.study / "REPRODUCTION_IDENTITY.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Changed interruption evidence"):
            self.replay()
        self.generator.assert_not_called()

    def test_wrong_run_fails(self):
        with self.assertRaisesRegex(ValueError, "Not the approved"):
            recovery.verify_evidence(self.root / "OTHER_RUN")

    def test_failed_task_is_not_replaced(self):
        save(self.partial / "FAILED.json", {"error": "scientific failure"})
        with self.assertRaisesRegex(ValueError, "Unexpected partial"):
            self.replay()
        self.generator.assert_not_called()

    def test_existing_prediction_is_not_replaced(self):
        save(self.partial / "prediction.json.gz", {"existing": True})
        with self.assertRaisesRegex(ValueError, "Unexpected partial"):
            self.replay()
        self.generator.assert_not_called()

    def test_existing_replay_destination_blocks_second_attempt(self):
        self.replay()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.replay()
        self.assertEqual(self.generator.call_count, 1)

    def test_archive_preserves_all_original_bytes_and_log(self):
        self.replay()
        recovery.publish_prepared(self.run, self.area)
        receipt = load(self.area / "RECOVERY_RECEIPT.json")
        for relative, row in receipt["preserved_evidence"].items():
            self.assertEqual(sha(self.run / relative), row["sha256"])
        self.assertTrue((self.area / "archived/panel/uniforms.json.gz.pending").exists())
        self.assertEqual((self.area / "archived/worker.log").read_text(), "original disk-stop log\n")
        self.assertFalse(self.log.exists())
        self.assertTrue((self.partial / "INPUT_COMMIT.json").exists())
        self.assertEqual(list(self.partial.glob("*.pending")), [])
        self.assertEqual(receipt["completed_tasks_recomputed"], 0)

    def test_tampered_prepared_input_fails_before_archive(self):
        self.replay()
        (self.area / "prepared/uniforms.json.gz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Committed file changed"):
            recovery.publish_prepared(self.run, self.area)
        recovery.verify_evidence(self.run)

    def test_prepared_commit_with_wrong_h_fails(self):
        self.replay()
        path = self.area / "prepared/INPUT_COMMIT.json"
        commit = load(path)
        commit["identity"]["h"] = 6
        from simulation.storage import canon
        path.write_text(canon(commit))
        with self.assertRaisesRegex(ValueError, "identity"):
            recovery.publish_prepared(self.run, self.area)
        recovery.verify_evidence(self.run)

    def test_prepared_commit_with_wrong_panel_fails(self):
        self.replay()
        path = self.area / "prepared/INPUT_COMMIT.json"
        commit = load(path)
        commit["identity"]["panel"] = "FORMAL_001480"
        from simulation.storage import canon
        path.write_text(canon(commit))
        with self.assertRaisesRegex(ValueError, "identity"):
            recovery.publish_prepared(self.run, self.area)

    def test_budget_keeps_failed_attempt_and_replay_cost(self):
        before = run_simulation.consumed_seconds(self.study)
        self.replay()
        save(self.area / "replay/RESOURCE_RECORD.json", {"elapsed_seconds": 2.5})
        recovery.publish_prepared(self.run, self.area)
        self.assertEqual(before, 1.25)
        self.assertEqual(run_simulation.consumed_seconds(self.study), before + 2.5)
        save(self.partial / "RESOURCE_RECORD.json", {"elapsed_seconds": 3.0})
        self.assertEqual(run_simulation.consumed_seconds(self.study), before + 2.5 + 3.0)

    def test_preflight_failure_prevents_worker_and_archive(self):
        with patch.object(recovery, "inspect_run", side_effect=ValueError("8 GiB")), \
                patch.object(run_simulation, "run_task") as worker:
            with self.assertRaisesRegex(ValueError, "8 GiB"):
                recovery.prepare_resume(self.run)
            worker.assert_not_called()
        self.assertFalse(self.area.exists())

    def test_replay_watchdog_failure_prevents_publication(self):
        with patch.object(recovery, "inspect_run", return_value={"remaining_main_compute_seconds": 50}), \
                patch.object(run_simulation, "run_task", side_effect=ValueError("worker limit")), \
                patch.object(recovery, "publish_prepared") as publish:
            with self.assertRaisesRegex(ValueError, "worker limit"):
                recovery.prepare_resume(self.run)
            publish.assert_not_called()
        recovery.verify_evidence(self.run)

    def test_preflight_passes_remaining_budget_to_watchdog(self):
        with patch.object(recovery, "inspect_run", return_value={"remaining_main_compute_seconds": 42}), \
                patch.object(run_simulation, "run_task") as worker, \
                patch.object(recovery, "publish_prepared"):
            with redirect_stdout(StringIO()):
                recovery.prepare_resume(self.run)
            self.assertEqual(worker.call_args.args[2], 42)


class ResumeLauncherTests(unittest.TestCase):
    def test_resume_is_main_then_unchanged_full_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            calls = []
            with patch.object(sys, "argv", ["START_HERE.py", "--resume", "--out", str(run)]), \
                    patch.object(START_HERE, "check_environment"), \
                    patch.object(recovery, "prepare_resume") as prepare, \
                    patch.object(START_HERE.subprocess, "run", side_effect=lambda cmd, **kw: calls.append(cmd)), \
                    redirect_stdout(StringIO()):
                START_HERE.main()
            prepare.assert_called_once_with(run.resolve())
            self.assertEqual(len(calls), 8)
            self.assertIn("--resume", calls[0])
            self.assertNotIn("--resume", calls[2])
            self.assertIn(str(run / "main_study/panels"), calls[2])
            self.assertIn("verify_reproduction.py", calls[3][2])
            self.assertIn("render_paper.py", calls[7][2])

    def test_resume_failure_keeps_original_stop_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            original = run / "PIPELINE_STOPPED.json"
            original.write_text("original evidence")
            with patch.object(sys, "argv", ["START_HERE.py", "--resume", "--out", str(run)]), \
                    patch.object(START_HERE, "check_environment"), \
                    patch.object(recovery, "prepare_resume"), \
                    patch.object(START_HERE.subprocess, "run", side_effect=ValueError("QA stop")), \
                    redirect_stdout(StringIO()):
                with self.assertRaisesRegex(ValueError, "QA stop"):
                    START_HERE.main()
            self.assertEqual(original.read_text(), "original evidence")
            self.assertEqual(len(list(run.glob("PIPELINE_STOPPED_*.json"))), 1)
            self.assertFalse((run / "FULL_REPRODUCTION_VERIFIED.json").exists())

    def test_resume_requires_explicit_output(self):
        with patch.object(sys, "argv", ["START_HERE.py", "--resume"]), \
                patch.object(START_HERE, "check_environment"), \
                patch.object(recovery, "prepare_resume") as prepare:
            with self.assertRaisesRegex(ValueError, "existing --out"):
                START_HERE.main()
            prepare.assert_not_called()

    def test_completed_task_is_verified_without_launching_worker(self):
        """One synthetic DONE record tests the real main-loop skip branch."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / "MOCK_COMPLETED"
            folder = out / "panels/h3/FORMAL_000000"
            save(folder / "scientific_fixture.json", {"not_a_simulation": True})
            save(folder / "DONE.json", {
                "identity": {"study": "step5", "h": 3, "panel": "FORMAL_000000"},
                "files": {"scientific_fixture.json": sha(folder / "scientific_fixture.json")}})
            save(folder / "RESOURCE_RECORD.json", {"elapsed_seconds": 12})
            save(out / "recovery_disk_001/archived/panel/RESOURCE_RECORD.json", {"elapsed_seconds": 1})
            save(out / "recovery_disk_001/replay/RESOURCE_RECORD.json", {"elapsed_seconds": 2})
            package = run_simulation.PACKAGE
            save(out / "REPRODUCTION_IDENTITY.json", {
                "study": "step5", "seed_manifest_sha256": sha(package / "protocol/step5_seed.json"),
                "code_sha256": {str(p.relative_to(package)): sha(p)
                                for p in sorted((package / "simulation").rglob("*.py"))}})
            checksum = sha(folder / "DONE.json")
            with patch.object(sys, "argv", ["run_simulation.py", "--study", "main_study",
                    "--out", str(out), "--resume", "--run-full-study"]), \
                    patch.object(run_simulation, "HS", (3,)), \
                    patch.object(run_simulation, "REPLICATES", 1), \
                    patch.object(run_simulation, "run_task") as worker, redirect_stdout(StringIO()):
                run_simulation.main()
            worker.assert_not_called()
            self.assertEqual(sha(folder / "DONE.json"), checksum)
            self.assertEqual(load(out / "COMPLETE.json")["elapsed_seconds"], 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
