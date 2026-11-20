"""Check that live progress comes only from the active isolated job."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from generic_runner import ACTIVE_PROFILES, job_progress


class GenericProgressTests(unittest.TestCase):
    def test_reads_latest_complete_event_and_ignores_old_or_invalid_links(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            job = SimpleNamespace(directory=base / "job")
            attempt = job.directory / "attempts" / "infer-new" / "output"
            attempt.mkdir(parents=True)
            run_name = "run-" + "1" * 32
            sandbox = base / "Packages" / ACTIVE_PROFILES["infer"] / "AC" / "Temp" / run_name
            sandbox.mkdir(parents=True)
            descriptor = attempt / "live.json"
            descriptor.write_text(json.dumps({"stage": "infer", "run_folder": run_name}), encoding="utf-8")
            event = {"event": "progress", "stage": "dino", "completed": 2, "total": 3}
            (sandbox / "worker.log").write_text(json.dumps(event) + '\n{"partial":', encoding="utf-8")
            with patch.dict("os.environ", LOCALAPPDATA=str(base)):
                self.assertEqual(job_progress(job, "infer", 0), "MM 找候選框 · 2／3 張")
                self.assertIsNone(job_progress(job, "infer", descriptor.stat().st_mtime_ns + 1))
                descriptor.write_text(json.dumps({"stage": "infer", "run_folder": "../other"}), encoding="utf-8")
                self.assertIsNone(job_progress(job, "infer", 0))


if __name__ == "__main__":
    unittest.main()
