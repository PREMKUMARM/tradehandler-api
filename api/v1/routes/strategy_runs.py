"""CRUD-style API over strategy_definitions / strategy_runs (metadata + run tracking)."""
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from zoneinfo import ZoneInfo

from core.exceptions import NotFoundError, ValidationError
from database.connection import get_database
from schemas.support_ops import RunStatusPatch, StrategyDefinitionIn, StrategyRunIn

router = APIRouter(prefix="/strategy-runs", tags=["Strategy runs"])


def _row(r) -> Dict[str, Any]:
    return {k: r[k] for k in r.keys()}


@router.post("/definitions")
def create_definition(body: StrategyDefinitionIn):
    did = (body.id or "").strip() or str(uuid.uuid4())
    if not body.name.strip():
        raise ValidationError(message="name required", field="name")
    now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO strategy_definitions (id, name, spec_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (did, body.name.strip(), json.dumps(body.spec, default=str), now),
    )
    conn.commit()
    cur = conn.execute("SELECT * FROM strategy_definitions WHERE id = ?", (did,))
    row = cur.fetchone()
    return {"data": _row(row)}


@router.get("/definitions")
def list_definitions():
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("SELECT * FROM strategy_definitions ORDER BY created_at DESC")
    return {"data": [_row(r) for r in cur.fetchall()]}


@router.post("")
def create_run(body: StrategyRunIn):
    rid = str(uuid.uuid4())
    now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    mode = (body.mode or "paper").strip()
    if mode not in ("paper", "live", "backtest"):
        raise ValidationError(message="mode must be paper, live, or backtest", field="mode")

    db = get_database()
    conn = db.get_connection()
    if body.definition_id:
        cur = conn.execute(
            "SELECT 1 FROM strategy_definitions WHERE id = ?", (body.definition_id,)
        )
        if not cur.fetchone():
            raise NotFoundError(resource="strategy_definition", identifier=body.definition_id)

    conn.execute(
        """
        INSERT INTO strategy_runs (id, definition_id, mode, status, started_at, ended_at, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rid,
            body.definition_id,
            mode,
            "running",
            now,
            None,
            json.dumps(body.meta, default=str),
        ),
    )
    conn.commit()
    cur = conn.execute("SELECT * FROM strategy_runs WHERE id = ?", (rid,))
    return {"data": _row(cur.fetchone())}


@router.get("")
def list_runs(status: Optional[str] = Query(None)):
    db = get_database()
    conn = db.get_connection()
    if status:
        cur = conn.execute(
            "SELECT * FROM strategy_runs WHERE status = ? ORDER BY started_at DESC",
            (status,),
        )
    else:
        cur = conn.execute("SELECT * FROM strategy_runs ORDER BY started_at DESC LIMIT 200")
    return {"data": [_row(r) for r in cur.fetchall()]}


@router.get("/{run_id}")
def get_run(run_id: str):
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("SELECT * FROM strategy_runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    if not row:
        raise NotFoundError(resource="strategy_run", identifier=run_id)
    fills = conn.execute(
        "SELECT * FROM strategy_fills WHERE run_id = ? ORDER BY id ASC", (run_id,)
    ).fetchall()
    return {"data": {"run": _row(row), "fills": [_row(f) for f in fills]}}


@router.patch("/{run_id}/status")
def patch_run_status(run_id: str, status: str, ended: bool = True):
    st = status.strip().lower()
    if st not in ("running", "completed", "failed", "cancelled"):
        raise ValidationError(message="invalid status", field="status")
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("SELECT id FROM strategy_runs WHERE id = ?", (run_id,))
    if not cur.fetchone():
        raise NotFoundError(resource="strategy_run", identifier=run_id)
    now_iso = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    if body.ended:
        conn.execute(
            "UPDATE strategy_runs SET status = ?, ended_at = ? WHERE id = ?",
            (st, now_iso, run_id),
        )
    else:
        conn.execute("UPDATE strategy_runs SET status = ? WHERE id = ?", (st, run_id))
    conn.commit()
    cur = conn.execute("SELECT * FROM strategy_runs WHERE id = ?", (run_id,))
    return {"data": _row(cur.fetchone())}
