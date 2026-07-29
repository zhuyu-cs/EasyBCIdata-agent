"""Ed25519 keypair management for federation.

Heavy `cryptography` imports are deferred to first call so a fresh
`pip install -e .` can load the federation module surface without
mandating the optional dependency on every user.  Run
`pip install -e .[federation]` (or let `lazy_deps.ensure("federation")`
pull it in on demand) to enable the real signing/verifying paths.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Tuple

from easybci_lib.constants import get_easybci_home


def _crypto():
    """Return (serialization, Ed25519PrivateKey, InvalidSignature) — raises a
    user-friendly ImportError when ``cryptography`` is absent."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:  # pragma: no cover — import-time guard
        raise ImportError(
            "federation requires the optional `cryptography` dependency. "
            "Install via `pip install -e .[federation]` or "
            "`uv sync --extra federation`."
        ) from exc
    return serialization, Ed25519PrivateKey, InvalidSignature


def _identity_dir() -> Path:
    return get_easybci_home() / "lab_identity"


def generate_lab_keypair(*, lab_id: str) -> Tuple[Path, Path]:
    serialization, Ed25519PrivateKey, _ = _crypto()
    d = _identity_dir()
    d.mkdir(parents=True, exist_ok=True)

    private = Ed25519PrivateKey.generate()
    public = private.public_key()

    private_path = d / "private.pem"
    public_path = d / "public.pem"

    private_path.write_bytes(private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    public_path.write_bytes(public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))

    os.chmod(private_path, 0o600)

    (d / "lab_id.txt").write_text(lab_id, encoding="utf-8")
    return private_path, public_path


def load_lab_keypair():
    serialization, _, _ = _crypto()
    d = _identity_dir()
    private_pem = (d / "private.pem").read_bytes()
    public_pem = (d / "public.pem").read_bytes()
    private = serialization.load_pem_private_key(private_pem, password=None)
    public = serialization.load_pem_public_key(public_pem)
    return private, public


def export_public_key_pem(public_key) -> bytes:
    serialization, _, _ = _crypto()
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def sign_bytes(private_key, payload: bytes) -> str:
    sig = private_key.sign(payload)
    return base64.b64encode(sig).decode("ascii")


def verify_bytes(public_key_pem: bytes, payload: bytes, signature_b64: str) -> bool:
    serialization, _, InvalidSignature = _crypto()
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        public_key.verify(base64.b64decode(signature_b64), payload)
        return True
    except (InvalidSignature, Exception):  # noqa: BLE001
        return False


def public_key_fingerprint(public_key_pem: bytes) -> str:
    """SHA256 of public-key PEM, hex-truncated to 16 chars — for TOFU display."""
    return hashlib.sha256(public_key_pem).hexdigest()[:16]
