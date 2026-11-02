import json
import time

import pytest

from pqfl.train.worker import claim_next, execute, prepare, serve, submit


def test_submit_validates_and_refuses_duplicates(tmp_path):
    path = submit(tmp_path, "job-1", "pqfl.train.jobs:sleep_job", {"seconds": 0})
    assert json.loads(path.read_text())["function"] == "pqfl.train.jobs:sleep_job"
    with pytest.raises(FileExistsError):
        submit(tmp_path, "job-1", "pqfl.train.jobs:sleep_job")
    for bad_id, bad_fn in (("bad id", "a:b"), ("ok", "no_colon")):
        with pytest.raises(ValueError):
            submit(tmp_path, bad_id, bad_fn)


def test_jobs_run_in_order_and_failures_do_not_stop_the_worker(tmp_path):
    submit(tmp_path, "a-first", "pqfl.train.jobs:sleep_job", {"seconds": 0})
    time.sleep(0.02)
    submit(tmp_path, "b-fails", "pqfl.train.jobs:sleep_job", {"fail": True})
    time.sleep(0.02)
    submit(tmp_path, "c-last", "pqfl.train.jobs:sleep_job", {"seconds": 0})
    messages = []
    reason = serve(tmp_path, poll_seconds=0.01, idle_exit_seconds=0.2, log=messages.append)
    assert reason == "idle timeout"
    assert [m.split()[-1] for m in messages if "running" in m] == ["a-first.json", "b-fails.json", "c-last.json"]
    assert sorted(p.name for p in (tmp_path / "done").iterdir()) == ["a-first.json", "c-last.json"]
    failed = json.loads((tmp_path / "failed" / "b-fails.json").read_text())
    assert "requested failure" in failed["error"] and "Traceback" in failed["traceback"]
    assert "requested failure" in (tmp_path / "logs" / "b-fails.log").read_text()
    assert not list((tmp_path / "pending").iterdir()) and not list((tmp_path / "running").iterdir())


def test_stop_file_ends_the_loop_and_claims_are_exclusive(tmp_path):
    prepare(tmp_path)
    (tmp_path / "STOP").write_text("")
    assert serve(tmp_path, poll_seconds=0.01, idle_exit_seconds=10, log=lambda m: None) == "stop file"
    (tmp_path / "STOP").unlink()
    submit(tmp_path, "only", "pqfl.train.jobs:sleep_job")
    first = claim_next(tmp_path)
    assert first is not None and claim_next(tmp_path) is None
    assert execute(tmp_path, first) is True


def test_code_edits_apply_to_the_next_job_without_restarting(tmp_path, monkeypatch):
    package = tmp_path / "pkgs"
    (package / "pqfl").mkdir(parents=True)
    (package / "pqfl" / "__init__.py").write_text("")
    module = package / "pqfl" / "demo.py"
    module.write_text("def value():\n    return 1\n")
    monkeypatch.syspath_prepend(str(package))
    queue = tmp_path / "queue"
    submit(queue, "one", "pqfl.demo:value")
    serve(queue, poll_seconds=0.01, idle_exit_seconds=0.1, log=lambda m: None)
    module.write_text("def value():\n    return 2\n")
    submit(queue, "two", "pqfl.demo:value")
    serve(queue, poll_seconds=0.01, idle_exit_seconds=0.1, log=lambda m: None)
    results = {p.stem: json.loads(p.read_text())["result"] for p in (queue / "done").iterdir()}
    assert results == {"one": 1, "two": 2}
