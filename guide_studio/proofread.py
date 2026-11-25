from __future__ import annotations

from typing import Any, Dict, List

from .model import evaluate_reference


def proofread_manuscript(bundle: Dict[str, Any], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for line in sorted(bundle["manuscript"], key=lambda x: (x["script_id"], x["line_order"], x["line_id"])):
        ev = evaluate_reference(
            bundle,
            model,
            language=line["language"],
            wording=line["wording"],
            intended_person_id=line.get("intended_person_id", ""),
            site_id=line["site_id"],
            last_person_id=line.get("last_person_id", ""),
        )
        suggestion = ""
        if not ev["approved"] and line.get("intended_person_id"):
            # Prefer a full appellation from the alias table.
            for a in bundle["appellations"]:
                if (
                    a["language"] == line["language"]
                    and a.get("kind") == "full"
                    and a.get("person_id") == line["intended_person_id"]
                ):
                    suggestion = a["appellation"]
                    break
        rows.append(
            {
                "line_id": line["line_id"],
                "script_id": line["script_id"],
                "language": line["language"],
                "line_order": line["line_order"],
                "site_id": line["site_id"],
                "last_person_id": line["last_person_id"],
                "wording": line["wording"],
                "text": line.get("text", ""),
                "intended_person_id": line.get("intended_person_id", ""),
                "approved": ev["approved"],
                "decision": "approve" if ev["approved"] else "replace_short_form",
                "suggested_full_appellation": suggestion,
                "top_person_id": ev["top_person_id"],
                "intended_rank": ev["intended_rank"],
                "intended_probability": ev["intended_probability"],
                "threshold": ev["threshold"],
                "reason": ev["reason"],
                "ranking": "|".join(ev["ranking"]),
                "distribution_json": ev["distribution"],
            }
        )
    return rows
