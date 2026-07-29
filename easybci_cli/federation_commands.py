"""easybci federation: init / subscribe / unsubscribe / list-sources / pull / push / status."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from easybci_cli import cli_output
from easybci_lib.constants import get_easybci_home
from easybci_lib.tools.neural_processing.federation.channel import GitChannel
from easybci_lib.tools.neural_processing.federation.keys import generate_lab_keypair
from easybci_lib.tools.neural_processing.federation.subscriptions import (
    Subscription,
    SubscriptionStore,
)


def _ensure_federation_deps() -> bool:
    """Check that ``cryptography`` and ``gitpython`` are importable; if not,
    try a one-shot lazy install via :data:`LAZY_DEPS["federation"]` and fall
    back to a friendly user message.  Returns True on success, False when
    dependencies remain missing — callers should ``return 1`` on False.
    """
    missing: list[str] = []
    try:  # pragma: no cover — environment-dependent
        import cryptography  # noqa: F401
    except ImportError:
        missing.append("cryptography")
    try:  # pragma: no cover — environment-dependent
        import git  # noqa: F401
    except ImportError:
        missing.append("gitpython")
    if not missing:
        return True
    cli_output.print_warning(
        f"federation requires optional dependencies: {', '.join(missing)}."
    )
    try:
        from easybci_lib.tools.lazy_deps import ensure
        cli_output.print_info("Attempting lazy install via `pip` …")
        ensure("federation", prompt=False)
    except Exception as exc:  # noqa: BLE001
        cli_output.print_warning(
            f"lazy install failed: {exc}. "
            "Install manually with `pip install -e .[federation]` "
            "or `uv sync --extra federation`."
        )
        return False
    # Re-check after install — the import path may have cached the original
    # failure, so use importlib invalidation as belt-and-braces.
    try:
        import importlib
        importlib.invalidate_caches()
        import cryptography  # noqa: F401
        import git  # noqa: F401
    except ImportError as exc:
        cli_output.print_warning(
            f"federation deps still missing after install: {exc}. "
            "Install manually with `pip install -e .[federation]`."
        )
        return False
    cli_output.print_info("federation deps installed; continuing.")
    return True


def cmd_federation_init(args: Any) -> int:
    if not _ensure_federation_deps():
        return 1
    lab_id = getattr(args, "lab_id", "") or "local"
    priv, pub = generate_lab_keypair(lab_id=lab_id)
    cli_output.print_info(f"Generated keypair for lab '{lab_id}'.")
    cli_output.print_info(f"  Private: {priv}")
    cli_output.print_info(f"  Public:  {pub}")
    cli_output.print_warning("Keep the private key SAFE. Loss = inability to sign new contributions.")
    return 0


def cmd_federation_subscribe(args: Any) -> int:
    store = SubscriptionStore()
    sub = Subscription(
        source_id=args.source_id,
        url=args.url,
        transport=args.transport or "git",
        public_key_fingerprint="",
        added_at=_dt.datetime.utcnow().isoformat() + "Z",
    )
    store.add(sub)
    cli_output.print_info(f"Subscribed to {args.source_id} ({args.url}).")
    cli_output.print_info("First `easybci federation pull` will register the source's public key (TOFU).")
    return 0


def cmd_federation_unsubscribe(args: Any) -> int:
    ok = SubscriptionStore().remove(args.source_id)
    if not ok:
        cli_output.print_warning(f"no subscription {args.source_id}")
        return 1
    cli_output.print_info(f"Unsubscribed from {args.source_id}.")
    return 0


def cmd_federation_list_sources(args: Any) -> int:
    subs = SubscriptionStore().list_all()
    if getattr(args, "json", False):
        print(json.dumps([s.__dict__ for s in subs], ensure_ascii=False, indent=2))
        return 0
    if not subs:
        cli_output.print_info("(no subscriptions)")
        return 0
    cli_output.print_info(f"{'SOURCE_ID':<20} {'TRANSPORT':<6} {'URL':<60} FP")
    for s in subs:
        fp = s.public_key_fingerprint or "(unverified)"
        cli_output.print_info(f"{s.source_id:<20} {s.transport:<6} {s.url:<60} {fp}")
    return 0


def cmd_federation_pull(args: Any) -> int:
    """T1.2 — TOFU pull: clone, signature-verify each entry, copy passes
    into the local proven-pipelines library, quarantine failures."""
    if not _ensure_federation_deps():
        return 1
    if not args.source_id:
        cli_output.print_warning("specify --source-id NAME to pull from")
        return 1
    store = SubscriptionStore()
    sub = store.get(args.source_id)
    if sub is None:
        cli_output.print_warning(
            f"no subscription {args.source_id}; "
            f"run `easybci federation subscribe --source-id {args.source_id} --url <git-url>` first"
        )
        return 1
    proven_dir = get_easybci_home() / "skills" / "proven-pipelines"
    proven_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = get_easybci_home() / "federation" / "clones" / args.source_id
    channel = GitChannel(
        remote_url=sub.url,
        local_clone_dir=clone_dir,
        source_id=args.source_id,
        subscription_store=store,
    )
    try:
        n = channel.pull(into_dir=proven_dir)
    except RuntimeError as exc:
        cli_output.print_warning(f"federation pull rejected: {exc}")
        return 1
    cli_output.print_info(
        f"Pulled {n} entries from {sub.url} into {proven_dir}."
    )
    qdir = get_easybci_home() / "federation" / "quarantine" / args.source_id
    if qdir.exists():
        n_q = sum(1 for _ in qdir.glob("*.md"))
        if n_q:
            cli_output.print_warning(
                f"{n_q} entries quarantined; review at {qdir} before promoting."
            )
    return 0


def cmd_federation_push(args: Any) -> int:
    if not _ensure_federation_deps():
        return 1
    if not args.source_id:
        cli_output.print_warning("specify --source-id NAME for the remote to push to")
        return 1
    sub = SubscriptionStore().get(args.source_id)
    if sub is None:
        cli_output.print_warning(f"no subscription {args.source_id}")
        return 1
    proven_dir = get_easybci_home() / "skills" / "proven-pipelines"
    if not proven_dir.is_dir():
        cli_output.print_warning(f"no proven pipelines under {proven_dir}")
        return 1
    clone_dir = get_easybci_home() / "federation" / "clones" / args.source_id
    # Construct the channel with the subscription store so push runs with the
    # same TOFU bookkeeping as pull/status — keeps a consistent audit trail
    # across the three operations.
    channel = GitChannel(
        remote_url=sub.url,
        local_clone_dir=clone_dir,
        source_id=args.source_id,
        subscription_store=SubscriptionStore(),
    )
    n = channel.push(proven_dir=proven_dir)
    cli_output.print_info(f"Pushed {n} entries to {sub.url}.")
    return 0


def cmd_federation_status(args: Any) -> int:
    """T1.2 — Per-source diff summary: local vs remote commits + quarantine
    count + public key fingerprint."""
    if not _ensure_federation_deps():
        return 1
    if not args.source_id:
        cli_output.print_warning("specify --source-id NAME for the remote to inspect")
        return 1
    store = SubscriptionStore()
    sub = store.get(args.source_id)
    if sub is None:
        cli_output.print_warning(f"no subscription {args.source_id}")
        return 1
    clone_dir = get_easybci_home() / "federation" / "clones" / args.source_id
    channel = GitChannel(
        remote_url=sub.url,
        local_clone_dir=clone_dir,
        source_id=args.source_id,
        subscription_store=store,
    )
    info = channel.status()
    if getattr(args, "json", False):
        cli_output.print_info(json.dumps(info, indent=2, default=str))
        return 0
    cli_output.print_info(f"source: {args.source_id}")
    cli_output.print_info(f"remote: {info.get('remote_url')}")
    cli_output.print_info(f"local commit:  {info.get('local_commit')}")
    cli_output.print_info(f"remote commit: {info.get('remote_commit')}")
    cli_output.print_info(
        f"ahead: {info.get('ahead', 0)}  behind: {info.get('behind', 0)}"
    )
    cli_output.print_info(f"quarantined: {info.get('quarantined', 0)}")
    cli_output.print_info(
        f"public key fingerprint: {info.get('public_key_fingerprint') or '(missing)'}"
    )
    if info.get("error"):
        cli_output.print_warning(info["error"])
        return 1
    return 0
