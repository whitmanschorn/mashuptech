import time
from pathlib import Path

from mashuptech.state import DONE, FAILED, PROCESSING, QUEUED, JobStore


def test_add_dedupes_active_jobs(tmp_path):
    store = JobStore(tmp_path / "state.json")
    j1 = store.add(Path("/inbox/a.mp3"))
    assert j1 is not None
    assert store.add(Path("/inbox/a.mp3")) is None
    store.finish(j1.id, output_dir="/out/a")
    # Once finished, the same path can be queued again (e.g. re-dropped file).
    assert store.add(Path("/inbox/a.mp3")) is not None


def test_lifecycle_and_persistence(tmp_path):
    path = tmp_path / "state.json"
    store = JobStore(path)
    a = store.add(Path("/inbox/a.mp3"))
    b = store.add(Path("/inbox/b.mp3"))
    assert store.next_queued().id == a.id
    store.start(a.id, "mps")
    store.stage(a.id, "separate")
    store.progress(a.id, 0.5)
    assert store.get(a.id).status == PROCESSING
    assert store.next_queued().id == b.id

    # Reload mid-flight: processing job goes back to the queue, order preserved.
    store2 = JobStore(path)
    jobs = store2.jobs()
    assert [j.id for j in jobs] == [a.id, b.id]
    assert jobs[0].status == QUEUED and jobs[0].progress == 0.0


def test_fail_retry_and_snapshot(tmp_path):
    store = JobStore(tmp_path / "state.json")
    a = store.add(Path("/inbox/a.mp3"))
    store.start(a.id, "cpu")
    store.fail(a.id, "boom", source="/inbox/failed/a.mp3")
    snap = store.snapshot()
    assert snap["stats"]["failed"] == 1
    assert snap["jobs"][0]["error"] == "boom"
    store.requeue(a.id, Path("/inbox/a.mp3"))
    assert store.get(a.id).status == QUEUED and store.get(a.id).error is None


def test_eta_uses_average_of_done_jobs(tmp_path):
    store = JobStore(tmp_path / "state.json")
    a = store.add(Path("/inbox/a.mp3"))
    store.start(a.id, "cpu")
    store.update(a.id, started_at=time.time() - 100)
    store.finish(a.id, output_dir="/out/a")
    b = store.add(Path("/inbox/b.mp3"))
    c = store.add(Path("/inbox/c.mp3"))
    snap = store.snapshot()
    assert 99 <= snap["stats"]["avg_seconds"] <= 101
    assert 198 <= snap["stats"]["eta_seconds"] <= 202


def test_subscribers_receive_snapshots(tmp_path):
    store = JobStore(tmp_path / "state.json")
    q = store.subscribe()
    store.log("hello")
    snap = q.get(timeout=1)
    assert snap["log"][-1]["msg"] == "hello"
    store.unsubscribe(q)
