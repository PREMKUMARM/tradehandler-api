"""
Ensure default admin user exists (credentials from environment only).
"""
import os
import uuid
from datetime import datetime

from database.models import User
from database.user_repository import get_user_repository
from utils.logger import log_info, log_warning
from utils.password_utils import hash_password


def ensure_default_admin_user() -> None:
    """
    Create or update the default admin user when DEFAULT_ADMIN_EMAIL and
    DEFAULT_ADMIN_PASSWORD are set in the environment.
    """
    email = (os.getenv("DEFAULT_ADMIN_EMAIL") or "").strip().lower()
    password = os.getenv("DEFAULT_ADMIN_PASSWORD") or ""
    name = (os.getenv("DEFAULT_ADMIN_NAME") or "Admin").strip() or "Admin"

    if not email or not password:
        log_warning(
            "[Auth] DEFAULT_ADMIN_EMAIL / DEFAULT_ADMIN_PASSWORD not set — "
            "email/password sign-in disabled until configured"
        )
        return

    repo = get_user_repository()
    existing = repo.get_by_email(email)
    password_hash = hash_password(password)

    if existing:
        if not repo.get_password_hash(existing.user_id):
            repo.set_password_hash(existing.user_id, password_hash)
            log_info(f"[Auth] Set password for existing admin user {email}")
        return

    user_id = str(uuid.uuid4())
    user = User(
        user_id=user_id,
        email=email,
        name=name,
        picture=None,
        google_id=None,
        created_at=datetime.now(),
        last_login=None,
        is_active=True,
    )
    repo.save(user)
    repo.set_password_hash(user_id, password_hash)
    log_info(f"[Auth] Created default admin user {email}")
