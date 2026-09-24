"""Generated files stay readable while independent codegen processes publish them."""

import json
import multiprocessing
from pathlib import Path

from flagfft_codegen.artifacts import write_text_atomic


def _publish_versions(path: str, content: str, start: multiprocessing.Event) -> None:
    start.wait()
    for _ in range(8):
        write_text_atomic(Path(path), content)


def test_concurrent_artifact_publication(tmp_path):
    artifact = tmp_path / "kernel.json"
    versions = [json.dumps({"version": i, "source": str(i) * 500_000}) for i in range(4)]
    write_text_atomic(artifact, versions[0])
    start = multiprocessing.Event()
    workers = [
        multiprocessing.Process(target=_publish_versions, args=(str(artifact), value, start))
        for value in versions
    ]
    for worker in workers:
        worker.start()
    start.set()
    try:
        while any(worker.is_alive() for worker in workers):
            content = artifact.read_text()
            assert content in versions
            json.loads(content)
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
    assert all(worker.exitcode == 0 for worker in workers)
    assert not list(tmp_path.glob(".kernel.json.*"))
