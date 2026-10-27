"""Test public names without launching any simulation worker.

Temporary MOCK_ONLY folders contain synthetic orchestration records, never
scientific results. The one-case loop below is a test fixture only; the
production configuration remains three horizons and 5,000 panels per horizon.
"""

from collections import namedtuple
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_simulation
from simulation.storage import load, save, sha
from study_names import LEGACY_STUDY_IDS, STUDY_TITLES


class StopBeforeWorker(RuntimeError):
    """An intentional boundary: never generate or score a panel in these tests."""


class StudyNameChecks(unittest.TestCase):
    def test_mapping_retains_original_identifiers(self):
        self.assertEqual(
            LEGACY_STUDY_IDS,
            {"main_study": "step5", "fixed_history_study": "step6"},
        )
        self.assertEqual(set(LEGACY_STUDY_IDS), set(STUDY_TITLES))

    def test_help_uses_public_names(self):
        output = StringIO()
        with patch.object(sys, "argv", ["run_simulation.py", "--help"]):
            with self.assertRaises(SystemExit) as stopped:
                with redirect_stdout(output):
                    run_simulation.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("{main_study,fixed_history_study}", output.getvalue())
        self.assertNotIn("step5", output.getvalue().lower())
        self.assertNotIn("step6", output.getvalue().lower())

    def test_old_labels_are_not_public_cli_choices(self):
        for old_name in LEGACY_STUDY_IDS.values():
            with self.subTest(old_name=old_name):
                with patch.object(
                    sys,
                    "argv",
                    ["run_simulation.py", "--study", old_name, "--out", "UNUSED"],
                ), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        run_simulation.main()
                self.assertEqual(stopped.exception.code, 2)

    def test_no_launch_without_full_study_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in LEGACY_STUDY_IDS:
                out = Path(directory) / name
                with patch.object(
                    sys,
                    "argv",
                    ["run_simulation.py", "--study", name, "--out", str(out)],
                ), patch.object(run_simulation, "run_task") as worker:
                    with self.assertRaisesRegex(Exception, "No run started"):
                        run_simulation.main()
                    worker.assert_not_called()
                    self.assertFalse(out.exists())

    def check_launch_boundary(self, public_name):
        """Check the real launcher's mapping, paths, seed hash and budget."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve() / "MOCK_ONLY"
            root.mkdir()
            out = root / public_name
            main_inputs = root / "main_study/panels"
            templates = root / "templates"
            if public_name == "fixed_history_study":
                # A name-only mock pool entry satisfies the inventory fixture.
                save(main_inputs / "h3/FORMAL_000000/DONE.json", {"mock_only": True})
            arguments = [
                "run_simulation.py",
                "--study",
                public_name,
                "--out",
                str(out),
                "--run-full-study",
            ]
            if public_name == "fixed_history_study":
                arguments += [
                    "--main-inputs",
                    str(main_inputs),
                    "--templates",
                    str(templates),
                ]
            stack.enter_context(patch.object(sys, "argv", arguments))
            stack.enter_context(patch.object(run_simulation, "HS", (3,)))
            stack.enter_context(patch.object(run_simulation, "REPLICATES", 1))
            usage = namedtuple("Usage", "total used free")(
                40 * 2**30, 10 * 2**30, 30 * 2**30
            )
            stack.enter_context(
                patch.object(run_simulation.shutil, "disk_usage", return_value=usage)
            )
            stack.enter_context(
                patch.object(
                    run_simulation,
                    "template_hashes",
                    return_value={"MOCK_ONLY": "no-scientific-data"},
                )
            )
            worker = stack.enter_context(
                patch.object(run_simulation, "run_task", side_effect=StopBeforeWorker)
            )
            spawn = stack.enter_context(
                patch.object(
                    run_simulation.subprocess,
                    "Popen",
                    side_effect=AssertionError("No worker may be launched"),
                )
            )
            printed = StringIO()
            with self.assertRaises(StopBeforeWorker), redirect_stdout(printed):
                run_simulation.main()
            spawn.assert_not_called()
            worker.assert_called_once()
            command, folder, budget, run_root = worker.call_args.args
            legacy = LEGACY_STUDY_IDS[public_name]
            self.assertEqual(command[command.index("--study") + 1], legacy)
            self.assertEqual(folder, out / "panels/h3/FORMAL_000000")
            self.assertEqual(run_root, out)
            self.assertEqual(budget, (24 if public_name == "main_study" else 8) * 3600)
            if public_name == "fixed_history_study":
                self.assertEqual(
                    command[command.index("--main-inputs") + 1], str(main_inputs)
                )
                self.assertEqual(
                    command[command.index("--templates") + 1], str(templates)
                )
            identity = load(out / "REPRODUCTION_IDENTITY.json")
            self.assertEqual(identity["study"], legacy)
            self.assertEqual(identity["panels_per_h"], 5000)
            self.assertEqual(identity["h"], [3, 6, 9])
            self.assertEqual(
                identity["seed_manifest_sha256"],
                sha(run_simulation.PACKAGE / "protocol" / f"{legacy}_seed.json"),
            )
            self.assertIn(STUDY_TITLES[public_name], printed.getvalue())
            self.assertFalse((out / "panels").exists())
            self.assertFalse((out / "COMPLETE.json").exists())

    def test_main_study_keeps_original_worker_identity_and_seed(self):
        self.check_launch_boundary("main_study")

    def test_fixed_history_study_keeps_original_worker_identity_and_seed(self):
        self.check_launch_boundary("fixed_history_study")


if __name__ == "__main__":
    unittest.main(verbosity=2)
