"""Federation transport abstraction.

GitChannel is the default. HttpManifestChannel is intentionally not implemented
— only the interface — so this module can land without running a second
transport mechanism.
"""
from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .keys import (
    load_lab_keypair,
    public_key_fingerprint,
    sign_bytes,
    verify_bytes,
)
from .redact import redact_frontmatter
from .subscriptions import SubscriptionStore

logger = logging.getLogger(__name__)


class FederationChannel(ABC):
    @abstractmethod
    def push(self, *, proven_dir: Path) -> int:
        """Push local proven pipelines to remote. Returns number of entries pushed."""

    @abstractmethod
    def pull(self, *, into_dir: Path) -> int:
        """Pull from remote into ``into_dir``. Returns number of entries accepted."""

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Diff summary: local vs remote."""


class GitChannel(FederationChannel):
    def __init__(
        self,
        *,
        remote_url: str,
        local_clone_dir: Path,
        source_id: Optional[str] = None,
        subscription_store: Optional[SubscriptionStore] = None,
    ) -> None:
        self.remote_url = remote_url
        self.local_clone_dir = Path(local_clone_dir)
        # Optional — only required for TOFU fingerprint pinning across
        # multiple pulls.  When None, ``pull`` performs first-use trust
        # without persisting and ``status`` reports the bare commit diff.
        self.source_id = source_id
        self._subscriptions = subscription_store

    def _ensure_clone(self):
        try:
            import git
        except ImportError as exc:
            raise RuntimeError(
                "gitpython not installed; pip install 'gitpython==3.1.50'"
            ) from exc
        if not self.local_clone_dir.exists():
            git.Repo.clone_from(self.remote_url, self.local_clone_dir)
            return git.Repo(self.local_clone_dir)
        # Existing clone — fast-forward to latest remote so subsequent
        # signature checks are against the up-to-date corpus.
        repo = git.Repo(self.local_clone_dir)
        try:
            repo.remote("origin").pull()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "git fetch failed for %s; using existing clone state: %s",
                self.remote_url, exc,
            )
        return repo

    def push(self, *, proven_dir: Path) -> int:
        repo = self._ensure_clone()
        private, _public = load_lab_keypair()

        n_pushed = 0
        for md in sorted(proven_dir.glob("**/*.md")):
            if md.name in ("DESCRIPTION.md", "README.md"):
                continue
            content = md.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            end = content.find("---", 3)
            if end < 0:
                continue
            fm = yaml.safe_load(content[3:end]) or {}
            body = content[end + 3:]
            fm = redact_frontmatter(fm)
            sig = sign_bytes(private, body.encode("utf-8"))
            fm["signature"] = sig
            out = "---\n" + yaml.safe_dump(fm, allow_unicode=True) + "---" + body
            target = self.local_clone_dir / md.name
            target.write_text(out, encoding="utf-8")
            repo.index.add([str(target.relative_to(self.local_clone_dir))])
            n_pushed += 1

        if n_pushed > 0:
            repo.index.commit(f"federation push: {n_pushed} entries")
            try:
                repo.remote("origin").push()
            except Exception as exc:  # noqa: BLE001
                logger.warning("git push failed (commits remain local): %s", exc)
        return n_pushed

    # ----- pull/status helpers -----------------------------------------

    def _read_remote_public_key(self, repo) -> Optional[bytes]:
        """Return the raw bytes of ``lab_public_key.pem`` at repo root, or
        None when missing.  The federation contract requires every pushing
        lab to commit its public key alongside its entries.
        """
        pem_path = self.local_clone_dir / "lab_public_key.pem"
        if not pem_path.exists():
            return None
        try:
            return pem_path.read_bytes()
        except OSError:
            return None

    def _quarantine_dir(self) -> Path:
        from easybci_lib.constants import get_easybci_home
        base = get_easybci_home() / "federation" / "quarantine"
        if self.source_id:
            base = base / self.source_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _verify_entry(self, content: str, public_pem: bytes) -> bool:
        """Verify the ``signature`` frontmatter field against the body.

        Returns True iff the frontmatter parses, contains a non-empty
        ``signature``, and ``verify_bytes`` accepts the body.  Any other
        outcome (malformed frontmatter, missing signature, verification
        failure) returns False.
        """
        if not content.startswith("---"):
            return False
        end = content.find("---", 3)
        if end < 0:
            return False
        try:
            fm = yaml.safe_load(content[3:end]) or {}
        except yaml.YAMLError:
            return False
        sig = fm.get("signature")
        if not sig:
            return False
        body = content[end + 3:]
        try:
            return verify_bytes(public_pem, body.encode("utf-8"), sig)
        except Exception:  # noqa: BLE001
            return False

    # ----- pull/status -------------------------------------------------

    def pull(self, *, into_dir: Path) -> int:
        """TOFU + signature-verifying pull.

        On first pull for a given ``source_id``, the public key fingerprint
        from the repo root is recorded into SubscriptionStore (TOFU).
        Subsequent pulls reject the entire batch when the fingerprint has
        changed — the user must explicitly re-subscribe to accept the new
        identity.

        Each ``*.md`` (excluding DESCRIPTION/README) is signature-verified
        against the lab's public key.  Entries that pass land in
        ``into_dir``; entries that fail are copied to
        ``~/.easybci/federation/quarantine/<source_id>/`` so the user can
        review them manually before promoting.

        Returns the number of entries accepted into ``into_dir``.
        """
        into_dir = Path(into_dir)
        into_dir.mkdir(parents=True, exist_ok=True)
        repo = self._ensure_clone()

        public_pem = self._read_remote_public_key(repo)
        if public_pem is None:
            raise RuntimeError(
                f"federation pull: remote {self.remote_url} is missing "
                "lab_public_key.pem at the repo root; cannot verify signatures."
            )
        fingerprint = public_key_fingerprint(public_pem)

        # TOFU pinning — only enforced when both source_id and a store are
        # available.  CLI dispatch wires both; programmatic callers can
        # opt out by leaving them unset.
        store = self._subscriptions or (
            SubscriptionStore() if self.source_id else None
        )
        if self.source_id and store is not None:
            sub = store.get(self.source_id)
            if sub is not None and sub.public_key_fingerprint:
                if sub.public_key_fingerprint != fingerprint:
                    raise RuntimeError(
                        f"federation pull: public-key fingerprint for "
                        f"source_id={self.source_id!r} changed "
                        f"({sub.public_key_fingerprint!r} → {fingerprint!r}); "
                        "remote identity may have rotated or been compromised. "
                        "Re-subscribe explicitly to accept the new key."
                    )
            elif sub is not None:
                # First-pull TOFU: record the fingerprint so future pulls
                # detect rotation.
                from .subscriptions import Subscription as _Sub
                store.add(_Sub(
                    source_id=sub.source_id,
                    url=sub.url,
                    transport=sub.transport,
                    public_key_fingerprint=fingerprint,
                    added_at=sub.added_at,
                ))

        n_accepted = 0
        n_quarantined = 0
        for md in sorted(self.local_clone_dir.glob("**/*.md")):
            if md.name in ("DESCRIPTION.md", "README.md"):
                continue
            try:
                content = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            target_name = md.name
            if self._verify_entry(content, public_pem):
                target = into_dir / target_name
                target.write_text(content, encoding="utf-8")
                n_accepted += 1
            else:
                qpath = self._quarantine_dir() / target_name
                try:
                    qpath.write_text(content, encoding="utf-8")
                except OSError as exc:
                    logger.debug("quarantine write failed for %s: %s", md, exc)
                n_quarantined += 1
                logger.warning(
                    "federation pull: signature failed for %s; quarantined", md.name,
                )

        if n_quarantined:
            logger.info(
                "federation pull: %d accepted, %d quarantined for review at %s",
                n_accepted, n_quarantined, self._quarantine_dir(),
            )
        return n_accepted

    def status(self) -> Dict[str, Any]:
        """Diff summary between the local clone and remote.

        Returns a dict with:
          - ``remote_url``
          - ``local_commit`` / ``remote_commit`` (None when the clone or
            remote is unreachable)
          - ``ahead`` / ``behind`` commit counts
          - ``quarantined`` count for ``source_id`` (when known)
          - ``public_key_fingerprint`` (None when no key file is present)
        """
        try:
            import git
        except ImportError:
            return {
                "remote_url": self.remote_url,
                "error": "gitpython not installed",
            }

        out: Dict[str, Any] = {
            "remote_url": self.remote_url,
            "source_id": self.source_id,
        }
        if not self.local_clone_dir.exists():
            out["local_commit"] = None
            out["remote_commit"] = None
            out["ahead"] = 0
            out["behind"] = 0
            out["note"] = "no local clone yet — run pull to bootstrap"
            return out

        try:
            repo = git.Repo(self.local_clone_dir)
            try:
                repo.remote("origin").fetch()
            except Exception as exc:  # noqa: BLE001
                logger.debug("fetch failed during status: %s", exc)
            local = repo.head.commit.hexsha
            try:
                remote_ref = repo.remote("origin").refs[repo.active_branch.name]
                remote = remote_ref.commit.hexsha
            except (IndexError, AttributeError, ValueError):
                remote = None
            out["local_commit"] = local
            out["remote_commit"] = remote
            if remote and remote != local:
                # Use rev-list counts so ahead/behind are exact rather than
                # eyeballed from the SHA pair.
                try:
                    ahead = sum(1 for _ in repo.iter_commits(f"{remote}..{local}"))
                    behind = sum(1 for _ in repo.iter_commits(f"{local}..{remote}"))
                except Exception:  # noqa: BLE001
                    ahead = behind = 0
                out["ahead"] = ahead
                out["behind"] = behind
            else:
                out["ahead"] = 0
                out["behind"] = 0
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"git inspection failed: {exc}"

        # Public key fingerprint — visible so the user can compare against
        # whatever was published out-of-band before subscribing.
        public_pem = None
        pem_path = self.local_clone_dir / "lab_public_key.pem"
        if pem_path.exists():
            try:
                public_pem = pem_path.read_bytes()
            except OSError:
                public_pem = None
        out["public_key_fingerprint"] = (
            public_key_fingerprint(public_pem) if public_pem else None
        )

        # Quarantine count — best-effort directory listing.
        if self.source_id:
            from easybci_lib.constants import get_easybci_home
            qdir = get_easybci_home() / "federation" / "quarantine" / self.source_id
            try:
                out["quarantined"] = sum(1 for p in qdir.glob("*.md")) if qdir.exists() else 0
            except OSError:
                out["quarantined"] = 0

        return out

    def _purge_local_clone(self) -> None:
        """Test/maintenance helper — remove the local clone so the next
        ``pull`` performs a fresh clone (and re-runs TOFU pinning if a
        subscription is registered).  Safe to call when the directory does
        not exist.
        """
        if self.local_clone_dir.exists():
            shutil.rmtree(self.local_clone_dir, ignore_errors=True)
