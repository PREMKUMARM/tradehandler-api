"""
Password hashing utilities (stdlib only — PBKDF2-HMAC-SHA256).
"""
import hashlib
import hmac
import secrets
from typing import Optional

_PBKDF2_ITERATIONS = 260_000
_SALT_BYTES = 16


def hash_password(plain_password: str) -> str:
    """Return `pbkdf2_sha256$<salt>$<hash>` for storage."""
    if not plain_password:
        raise ValueError("Password must not be empty")
    salt = secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(plain_password: str, stored_hash: Optional[str]) -> bool:
    """Constant-time verify against stored hash."""
    if not plain_password or not stored_hash:
        return False
    try:
        scheme, salt_hex, digest_hex = stored_hash.split("$", 2)
        if scheme != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            _PBKDF2_ITERATIONS,
        )
        return hmac.compare_digest(expected, actual)
    except (ValueError, TypeError):
        return False
