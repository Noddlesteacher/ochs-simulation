"""Test stage wiring without executing the long simulation."""

import importlib.metadata
from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import START_HERE


class EntryPointChecks(unittest.TestCase):
    def test_full_stage_order_and_new_data_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            out = (Path(directory) / "MOCK_ONLY").resolve()
            calls = []
            with patch.object(
                sys, "argv", ["START_HERE.py", "--out", str(out)]
            ), patch.object(START_HERE, "check_environment"), patch.object(
                START_HERE.subprocess,
                "run",
                side_effect=lambda cmd, **kw: calls.append(cmd),
            ):
                with redirect_stdout(StringIO()):
                    START_HERE.main()
            stages = [Path(cmd[2]).name if cmd[2] != "-m" else cmd[3] for cmd in calls]
            self.assertEqual(
                stages,
                [
                    "run_simulation.py",
                    "simulation.build_templates",
                    "run_simulation.py",
                    "verify_reproduction.py",
                    "collect_reproduction.py",
                    "collect_reproduction.py",
                    "verify_reproduction.py",
                    "render_paper.py",
                ],
            )
            self.assertIn("main_study", calls[0])
            self.assertIn("fixed_history_study", calls[2])
            self.assertIn(str(out / "main_study/panels"), calls[2])
            self.assertIn(str(out / "templates"), calls[2])
            self.assertIn(str(out / "main_study_summary/tables"), calls[-1])
            self.assertIn(str(out / "fixed_history_summary/tables"), calls[-1])
            self.assertIn(
                str(out / "fixed_history_summary/RETAINED_PATHS_AND_COUNTS.json"),
                calls[-1],
            )
            for command in calls:
                self.assertNotIn("step5", command)
                self.assertNotIn("step6", command)
            # The MOCK success marker is contained in this temporary test directory.
            self.assertTrue((out / "FULL_REPRODUCTION_VERIFIED.json").exists())

    def test_failure_stops_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            out = (Path(directory) / "MOCK_FAILURE_ONLY").resolve()
            with patch.object(
                sys, "argv", ["START_HERE.py", "--out", str(out)]
            ), patch.object(START_HERE, "check_environment"), patch.object(
                START_HERE.subprocess,
                "run",
                side_effect=RuntimeError("deliberate QA failure"),
            ) as run:
                with self.assertRaises(RuntimeError):
                    with redirect_stdout(StringIO()):
                        START_HERE.main()
            self.assertEqual(run.call_count, 1)
            self.assertFalse((out / "FULL_REPRODUCTION_VERIFIED.json").exists())
            self.assertTrue((out / "PIPELINE_STOPPED.json").exists())

    def test_disk_gate_does_not_launch_computation(self):
        from collections import namedtuple

        usage = namedtuple("usage", "total used free")(8 * 2**30, 7 * 2**30, 1 * 2**30)
        with patch.object(START_HERE.shutil, "disk_usage", return_value=usage), patch(
            "verify_package.main"
        ), patch.object(START_HERE.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "8 GiB"):
                START_HERE.check_environment()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
