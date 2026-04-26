#!/usr/bin/env python3
"""
Verify Firebase/FCM service account JSON can obtain an access token.

Usage:
  export FCM_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
  export FCM_PROJECT_ID=your-firebase-project-id   # optional; else read from JSON
  python3 scripts/verify_fcm_service_account.py

  python3 scripts/verify_fcm_service_account.py /path/to/key.json [project_id]

Exit codes: 0 = OAuth refresh OK, 1 = refresh failed, 2 = bad args / file / JSON shape.

When refresh fails with:
  invalid_grant: Invalid JWT Signature
after local PEM checks pass, Google IAM is not accepting this key_id + private_key pair
anymore (revoked/deleted key, or a JSON that was edited so key_id no longer matches the PEM).
Fix: Firebase Console → Project settings → Service accounts → Generate new private key,
then replace the JSON on the server (and delete old keys in GCP IAM if listed as unused).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _openssl_rsa_check(pem: str) -> str | None:
    """Return openssl stderr+stdout if openssl exists; None if openssl missing."""
    try:
        r = subprocess.run(
            ["openssl", "rsa", "-noout", "-check"],
            input=pem.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
        out = (r.stdout or b"").decode() + (r.stderr or b"").decode()
        return out.strip() or None
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        return f"(openssl error: {e})"


def _cryptography_load(pem: str) -> str | None:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        serialization.load_pem_private_key(pem.encode("utf-8"), password=None, backend=default_backend())
    except ImportError:
        return "(install cryptography for PEM validation: pip install cryptography)"
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def main() -> int:
    path = os.environ.get("FCM_SERVICE_ACCOUNT_JSON", "").strip()
    project = os.environ.get("FCM_PROJECT_ID", "").strip()

    if len(sys.argv) >= 2:
        path = sys.argv[1]
    if len(sys.argv) >= 3:
        project = sys.argv[2]

    if not path:
        print("ERROR: set FCM_SERVICE_ACCOUNT_JSON or pass path as first argument.", file=sys.stderr)
        return 2

    p = Path(path)
    if not p.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    raw = p.read_bytes()
    print("=== diagnostics (no secrets) ===")
    print(f"path: {p.resolve()}")
    print(f"bytes: {len(raw)} sha256: {hashlib.sha256(raw).hexdigest()}")
    print(f"utf-8 BOM: {raw.startswith(bytes([0xEF, 0xBB, 0xBF]))}")

    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}", file=sys.stderr)
        return 2

    for key in ("type", "project_id", "private_key_id", "private_key", "client_email", "token_uri"):
        if key not in data:
            print(f"ERROR: missing '{key}' — not a service account JSON?", file=sys.stderr)
            return 2

    if data.get("type") != "service_account":
        print("ERROR: JSON type must be 'service_account'.", file=sys.stderr)
        return 2

    pk = data.get("private_key", "")
    print(f"type: {data.get('type')} project_id: {data.get('project_id')}")
    print(f"client_email: {data.get('client_email')}")
    print(f"private_key_id: {data.get('private_key_id')}")
    print(f"token_uri: {data.get('token_uri')}")
    print(f"private_key chars: {len(pk)} embedded CR: {pk.count(chr(13))}")

    ossl = _openssl_rsa_check(pk)
    if ossl:
        print(f"openssl rsa -check: {ossl}")
    cr = _cryptography_load(pk)
    if cr:
        print(f"cryptography load_pem_private_key: FAIL {cr}")
    else:
        print("cryptography load_pem_private_key: OK")

    if not project:
        project = str(data.get("project_id", "")).strip()
    if not project:
        print("ERROR: FCM_PROJECT_ID not set and JSON has no project_id.", file=sys.stderr)
        return 2

    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account
    except ImportError:
        print("ERROR: install deps: pip install google-auth", file=sys.stderr)
        return 2

    print("")
    print("=== OAuth refresh (what FCM uses) ===")
    scopes = [
        "https://www.googleapis.com/auth/firebase.messaging",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    last_err: Exception | None = None
    for scope in scopes:
        try:
            creds = service_account.Credentials.from_service_account_info(data, scopes=[scope])
            creds.refresh(GoogleAuthRequest())
            if creds.token:
                print(f"OK with scope {scope!r}; access token length {len(creds.token)}")
                print("")
                print("FCM HTTP v1 calls can authenticate with this file.")
                return 0
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"FAIL scope {scope!r}: {e!r}")

    sys.stdout.flush()
    print("", file=sys.stderr)
    print("FAILED: Google rejected the service-account JWT (see errors above).", file=sys.stderr)
    if last_err and "Invalid JWT Signature" in repr(last_err):
        print(
            "Hint: PEM parses locally, but oauth2.googleapis.com still returns Invalid JWT Signature — "
            "this almost always means the private_key_id is not a currently valid key for that "
            "service account in Google Cloud IAM (key deleted/revoked), or the JSON was edited so "
            "private_key and private_key_id no longer match. Generate a NEW key JSON in Firebase/GCP; "
            "editing deploy scripts or app code cannot repair an IAM-revoked key.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
