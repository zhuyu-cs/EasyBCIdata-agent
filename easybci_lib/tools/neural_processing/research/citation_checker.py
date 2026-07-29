"""Detect retracted / withdrawn citations in the parameter registry.

Three data sources, each with its own check_* method. Network failures are
ALWAYS converted to status="unreachable" — never raised. Callers (CLI / banner)
treat unreachable as "couldn't verify, leave the default alone".
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Literal

from .parameter_registry import load_registry

logger = logging.getLogger(__name__)

_TIMEOUT_S = 5


@dataclass
class CitationCheckResult:
    citation_id: str
    status: Literal["ok", "retracted", "withdrawn", "revised", "unreachable", "unknown"]
    checked_at: str
    detail: str
    source: Literal["retraction_watch", "arxiv", "crossref", "manual"]


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _make_id(*parts: str) -> str:
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class CitationChecker:
    def __init__(self, *, timeout_s: float = _TIMEOUT_S) -> None:
        self._timeout = timeout_s

    # ----------------------------------------------- DOI via Crossref

    def check_doi(self, doi: str) -> CitationCheckResult:
        cid = _make_id("doi", doi)
        url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='/')}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            msg = data.get("message", {})
            update_to = msg.get("update-to", []) or []
            for u in update_to:
                if u.get("type") == "retraction-of":
                    return CitationCheckResult(
                        citation_id=cid, status="retracted",
                        checked_at=_now_iso(),
                        detail=f"retraction-of: {u.get('DOI', '')}",
                        source="crossref",
                    )
            return CitationCheckResult(
                citation_id=cid, status="ok",
                checked_at=_now_iso(),
                detail=msg.get("title", [""])[0] if msg.get("title") else "",
                source="crossref",
            )
        except urllib.error.HTTPError as exc:
            return CitationCheckResult(
                citation_id=cid,
                status="unknown" if exc.code in (404, 410) else "unreachable",
                checked_at=_now_iso(),
                detail=f"HTTP {exc.code}",
                source="crossref",
            )
        except Exception as exc:  # noqa: BLE001
            return CitationCheckResult(
                citation_id=cid, status="unreachable",
                checked_at=_now_iso(),
                detail=str(exc),
                source="crossref",
            )

    # ----------------------------------------------- arXiv withdrawn

    def check_arxiv(self, arxiv_id: str) -> CitationCheckResult:
        cid = _make_id("arxiv", arxiv_id)
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
            if "withdrawn" in body.lower():
                return CitationCheckResult(
                    citation_id=cid, status="withdrawn",
                    checked_at=_now_iso(),
                    detail="arxiv:withdrawn marker present in response",
                    source="arxiv",
                )
            return CitationCheckResult(
                citation_id=cid, status="ok",
                checked_at=_now_iso(),
                detail="no withdrawn marker",
                source="arxiv",
            )
        except urllib.error.HTTPError as exc:
            return CitationCheckResult(
                citation_id=cid,
                status="unknown" if exc.code == 404 else "unreachable",
                checked_at=_now_iso(),
                detail=f"HTTP {exc.code}",
                source="arxiv",
            )
        except Exception as exc:  # noqa: BLE001
            return CitationCheckResult(
                citation_id=cid, status="unreachable",
                checked_at=_now_iso(),
                detail=str(exc),
                source="arxiv",
            )

    # ----------------------------------------------- generic URL reachability

    def check_url(self, url: str) -> CitationCheckResult:
        cid = _make_id("url", url)
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            return CitationCheckResult(
                citation_id=cid, status="ok",
                checked_at=_now_iso(),
                detail="HEAD succeeded",
                source="manual",
            )
        except urllib.error.HTTPError as exc:
            try:
                with urllib.request.urlopen(url, timeout=self._timeout):
                    pass
                return CitationCheckResult(
                    citation_id=cid, status="ok",
                    checked_at=_now_iso(),
                    detail="GET succeeded after HEAD failure",
                    source="manual",
                )
            except Exception as exc2:  # noqa: BLE001
                return CitationCheckResult(
                    citation_id=cid, status="unreachable",
                    checked_at=_now_iso(),
                    detail=f"HEAD {exc.code} / GET {exc2}",
                    source="manual",
                )
        except Exception as exc:  # noqa: BLE001
            return CitationCheckResult(
                citation_id=cid, status="unreachable",
                checked_at=_now_iso(),
                detail=str(exc),
                source="manual",
            )

    # ----------------------------------------------- whole registry sweep

    def check_registry(self) -> List[CitationCheckResult]:
        registry = load_registry(force=True)
        results: List[CitationCheckResult] = []
        for op_name, schema in registry.items():
            for param_name, param in (getattr(schema, "parameters", {}) or {}).items():
                for paradigm_name, entry in (getattr(param, "empirical_defaults", {}) or {}).items():
                    if entry.origin_doi:
                        results.append(self.check_doi(entry.origin_doi))
                    elif entry.origin_arxiv_id:
                        results.append(self.check_arxiv(entry.origin_arxiv_id))
                    elif entry.origin_url:
                        results.append(self.check_url(entry.origin_url))
        return results
