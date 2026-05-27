"""Human-readable autonomous watch status for V2 / commodity UI."""
from __future__ import annotations

from typing import Any, Dict, Optional


def describe_autonomous_setup(
    plan: Optional[Dict[str, Any]],
    *,
    min_score: int,
    guard_message: Optional[str] = None,
    enforce_score: bool = True,
) -> Dict[str, Any]:
    """
    Map live trade plan to UI phase + detail.

    phase: waiting | in_progress | confirmed
    """
    plan = plan or {}
    score = int(plan.get("entry_confirmation_score") or 0)
    min_score = max(40, min(100, int(min_score or 65)))

    if not plan.get("entry_ready"):
        detail = plan.get("entry_block_reason") or "Entry not confirmed by indicators"
        return {
            "setup_phase": "waiting",
            "setup_detail": detail,
            "entry_confirmation_score": score,
            "autonomous_eligible": False,
        }

    score_ok = score >= min_score
    if score_ok or (not enforce_score and not guard_message):
        detail = (
            f"Entry confirmed — score {score} ≥ {min_score} "
            f"(autonomous will place when guard checks pass)"
            if enforce_score and score_ok
            else f"Checklist complete — score {score} (ready for autonomous)"
        )
        return {
            "setup_phase": "confirmed",
            "setup_detail": detail,
            "entry_confirmation_score": score,
            "autonomous_eligible": True,
        }

    ind = plan.get("indicators") or {}
    strat = str(plan.get("strategy_name") or "").lower()
    kind = str(plan.get("option_type") or "CE").upper()
    last_5m = ind.get("last_5m_close")

    if "pdh" in strat or "pdl" in strat:
        if kind == "PE":
            pdl = ind.get("pdl")
            if pdl:
                pdl_f = float(pdl)
                last_s = f"{float(last_5m):.0f}" if last_5m is not None else "—"
                return {
                    "setup_phase": "in_progress",
                    "setup_detail": (
                        f"PDL break in progress — spot below PDL {pdl_f:.0f} but "
                        f"last 5m close {last_s} not below level yet "
                        f"(score {score}/{min_score})"
                    ),
                    "entry_confirmation_score": score,
                    "autonomous_eligible": False,
                }
        if kind == "CE":
            pdh = ind.get("pdh")
            if pdh:
                pdh_f = float(pdh)
                last_s = f"{float(last_5m):.0f}" if last_5m is not None else "—"
                return {
                    "setup_phase": "in_progress",
                    "setup_detail": (
                        f"PDH break in progress — spot above PDH {pdh_f:.0f} but "
                        f"last 5m close {last_s} not above level yet "
                        f"(score {score}/{min_score})"
                    ),
                    "entry_confirmation_score": score,
                    "autonomous_eligible": False,
                }

    if "opening range" in strat or "orb" in strat:
        return {
            "setup_phase": "in_progress",
            "setup_detail": (
                f"ORB break in progress — waiting for 5m close beyond range "
                f"(score {score}/{min_score})"
            ),
            "entry_confirmation_score": score,
            "autonomous_eligible": False,
        }

    fallback = guard_message or (
        f"Confirmation score {score} below minimum {min_score} for autonomous entry"
    )
    return {
        "setup_phase": "in_progress",
        "setup_detail": fallback,
        "entry_confirmation_score": score,
        "autonomous_eligible": False,
    }
