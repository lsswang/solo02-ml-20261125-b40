"""File-backed session/page persistence.

The store is deliberately small and deterministic.  A session id is the only
key needed after closing: reopening restores the exact last physical page.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                self.data = json.load(fh)
        else:
            self.data = {"schema_version": 1, "sessions": {}}
        self.data.setdefault("sessions", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".sessions-", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def get(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.data["sessions"]:
            raise KeyError(f"未知会话号: {session_id}")
        return self.data["sessions"][session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self.data["sessions"]

    def create(
        self,
        session_id: str,
        scene: str = "home",
        start_page: int = 1,
        route: str = "fixed_preset",
        layout_id: str | None = None,
    ) -> dict[str, Any]:
        if session_id in self.data["sessions"]:
            return self.get(session_id)
        now = datetime.now(timezone.utc).isoformat()
        session = {
            "session_id": session_id,
            "scene": scene,
            "page": int(start_page),  # 1-based physical page
            "last_word": None,
            "route": route,
            "layout_id": layout_id,
            "closed": False,
            "created_at": now,
            "updated_at": now,
            "events": [],
        }
        self.data["sessions"][session_id] = session
        self.save()
        return session

    def ensure(
        self,
        session_id: str,
        scene: str = "home",
        start_page: int = 1,
        route: str = "fixed_preset",
    ) -> dict[str, Any]:
        if self.exists(session_id):
            return self.get(session_id)
        return self.create(session_id, scene, start_page, route)

    def switch_scene(self, session_id: str, new_scene: str, entry_event: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self.get(session_id)
        old_scene = session["scene"]
        old_page = session["page"]
        session["scene"] = new_scene
        event = {
            "type": "switch_scene",
            "from_scene": old_scene,
            "to_scene": new_scene,
            "retained_page": old_page,
        }
        if entry_event:
            event.update(entry_event)
        session["events"].append(event)
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return session

    def record_word(self, session_id: str, step: dict[str, Any]) -> dict[str, Any]:
        session = self.get(session_id)
        session["page"] = int(step["target_page"])
        session["last_word"] = step["target_word"]
        session["events"].append({"type": "select_word", **step})
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return session

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        session = self.get(session_id)
        session["events"].append(event)
        self.save()

    def close(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session["closed"] = True
        session["events"].append({"type": "close", "restored_page": session["page"]})
        self.save()
        return session

    def reopen(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session["closed"] = False
        session["events"].append({"type": "reopen", "restored_page": session["page"]})
        self.save()
        return session
