import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from probe_capabilities import run  # noqa: E402


class ProbeProcessTest(unittest.TestCase):
    def test_success_keeps_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run([sys.executable, "-c", "print('evidence')"], root, 2)
            self.assertEqual(result["status"], "passed")
            self.assertEqual((root / "stdout.txt").read_text(), "evidence\n")

    def test_failure_is_not_classified_as_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(
                [sys.executable, "-c", "raise RuntimeError('compiler failed')"],
                Path(directory),
                2,
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn(
                "compiler failed", (Path(directory) / "stderr.txt").read_text()
            )

    def test_timeout_kills_descendant_holding_output_pipe(self):
        command = (
            "import subprocess,sys,time;"
            " subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']);"
            " print('started',flush=True); time.sleep(10)"
        )
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            result = run([sys.executable, "-c", command], Path(directory), 0.3)
            self.assertEqual(result["status"], "unknown")
            self.assertLess(time.monotonic() - start, 3)
            self.assertIn("started", (Path(directory) / "stdout.txt").read_text())


if __name__ == "__main__":
    unittest.main()
