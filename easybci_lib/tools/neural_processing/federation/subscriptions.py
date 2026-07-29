"""Federation subscriptions store + TOFU public-key fingerprints."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from easybci_lib.constants import get_easybci_home

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    source_id: str
    url: str
    transport: str  # "git" | "http"
    public_key_fingerprint: str
    added_at: str


class SubscriptionStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (get_easybci_home() / "federation" / "subscriptions.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, rows: List[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, sub: Subscription) -> None:
        rows = [r for r in self._load() if r["source_id"] != sub.source_id]
        rows.append(asdict(sub))
        self._save(rows)

    def remove(self, source_id: str) -> bool:
        rows = self._load()
        new_rows = [r for r in rows if r["source_id"] != source_id]
        if len(new_rows) == len(rows):
            return False
        self._save(new_rows)
        return True

    def list_all(self) -> List[Subscription]:
        return [Subscription(**r) for r in self._load()]

    def get(self, source_id: str) -> Optional[Subscription]:
        for r in self._load():
            if r["source_id"] == source_id:
                return Subscription(**r)
        return None
