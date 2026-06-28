"""In-memory async backtest jobs with live progress for the UI."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from services.sensex_dhan_backtest import BacktestParams, BacktestProgress

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_TTL_SEC = 3600


def _progress_to_dict(p: BacktestProgress) -> Dict[str, Any]:
    d = asdict(p)
    total = int(d.get("total") or 0)
    current = int(d.get("current") or 0)
    d["percent"] = round(100.0 * current / total, 1) if total > 0 else 0.0
    return d


def _set_progress(job_id: str, progress: BacktestProgress) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job and job.get("status") == "running":
            job["progress"] = _progress_to_dict(progress)
            job["updated_at"] = time.time()


def start_backtest_job(
    params: BacktestParams,
    *,
    segment: str,
    runner: Callable[..., Dict[str, Any]],
) -> str:
    job_id = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "segment": segment,
            "progress": _progress_to_dict(
                BacktestProgress(phase="init", message="Starting backtest…")
            ),
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    def _run() -> None:
        try:
            def progress_cb(p: BacktestProgress) -> None:
                _set_progress(job_id, p)

            result = runner(params, progress_cb=progress_cb)
            with _lock:
                job = _jobs.get(job_id)
                if job:
                    job["status"] = "completed"
                    job["result"] = result
                    job["progress"] = _progress_to_dict(
                        BacktestProgress(phase="done", message="Backtest complete")
                    )
                    job["updated_at"] = time.time()
        except Exception as exc:
            with _lock:
                job = _jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["progress"] = _progress_to_dict(
                        BacktestProgress(phase="error", message=str(exc))
                    )
                    job["updated_at"] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get_backtest_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "segment": job.get("segment"),
            "progress": job.get("progress"),
            "result": job.get("result"),
            "error": job.get("error"),
        }


def prune_old_jobs() -> None:
    cutoff = time.time() - _TTL_SEC
    with _lock:
        stale = [jid for jid, j in _jobs.items() if float(j.get("updated_at") or 0) < cutoff]
        for jid in stale:
            _jobs.pop(jid, None)
