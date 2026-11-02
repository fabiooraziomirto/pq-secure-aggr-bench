"""File-based job queue for a long-lived GPU worker (pure Python, no torch).

Jobs are JSON files in `pending/`; the worker claims the oldest by renaming it into `running/`,
imports `module:function`, calls it with the job's args and writes `done/` or `failed/`. All
`pqfl` modules are dropped from `sys.modules` before every job, so code edits apply without
restarting the worker (and therefore without giving up the reserved GPU memory).
A file named STOP in the queue directory ends the loop; so does a long enough idle period.
"""
import contextlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

STATES = ("pending", "running", "done", "failed")


def prepare(queue: Path) -> Path:
    queue = Path(queue)
    for state in STATES:
        (queue / state).mkdir(parents=True, exist_ok=True)
    return queue


def submit(queue, job_id: str, function: str, args: dict | None = None) -> Path:
    """Write a job atomically; `function` is `package.module:callable`."""
    queue = prepare(queue)
    if ":" not in function or not job_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("job_id must be alphanumeric (with - and _) and function 'module:callable'")
    final = queue / "pending" / f"{job_id}.json"
    if any((queue / state / f"{job_id}.json").exists() for state in STATES):
        raise FileExistsError(f"Job {job_id} already exists")
    temporary = final.with_suffix(".tmp")
    temporary.write_text(json.dumps({"id": job_id, "function": function, "args": args or {}}))
    os.replace(temporary, final)
    return final


def claim_next(queue: Path) -> Path | None:
    for path in sorted((queue / "pending").glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name)):
        target = queue / "running" / path.name
        try:
            os.rename(path, target)
        except FileNotFoundError:
            continue
        return target
    return None


def _fresh_import(function: str):
    module_name, attribute = function.split(":", 1)
    for name in [m for m in sys.modules if m == "pqfl" or m.startswith("pqfl.")]:
        del sys.modules[name]
    importlib.invalidate_caches()
    return getattr(importlib.import_module(module_name), attribute)


def execute(queue: Path, claimed: Path) -> bool:
    """Run one claimed job; stdout and stderr go to `<queue>/logs/<id>.log`."""
    job = json.loads(claimed.read_text())
    log_dir = queue / "logs"
    log_dir.mkdir(exist_ok=True)
    started = time.time()
    outcome = {"id": job["id"], "function": job["function"], "args": job["args"], "started": started}
    ok = True
    with open(log_dir / f"{job['id']}.log", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            outcome["result"] = _fresh_import(job["function"])(**job["args"])
        except BaseException as exc:  # a failed job must never take the worker (and its memory) down
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            ok = False
            outcome["error"] = repr(exc)
            outcome["traceback"] = traceback.format_exc()
            traceback.print_exc()
    outcome["seconds"] = time.time() - started
    destination = queue / ("done" if ok else "failed") / claimed.name
    destination.write_text(json.dumps(outcome, default=str))
    claimed.unlink()
    return ok


def serve(queue, *, poll_seconds: float, idle_exit_seconds: float, log=print) -> str:
    """Process jobs until STOP appears or the queue stays idle; returns the reason."""
    queue = prepare(queue)
    last_activity = time.time()
    # Cached bytecode is validated only by source mtime (seconds) and size, so an edit of the same
    # length made within the same second would run stale code: never write bytecode in this loop.
    previous_flag, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        while True:
            if (queue / "STOP").exists():
                return "stop file"
            claimed = claim_next(queue)
            if claimed is not None:
                log(f"{time.strftime('%H:%M:%S')} running {claimed.name}")
                ok = execute(queue, claimed)
                log(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {claimed.name}")
                last_activity = time.time()
                continue
            if time.time() - last_activity > idle_exit_seconds:
                return "idle timeout"
            time.sleep(poll_seconds)
    finally:
        sys.dont_write_bytecode = previous_flag
