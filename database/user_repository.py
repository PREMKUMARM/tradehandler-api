"""
User repository for database operations
"""
import json
from datetime import datetime
from typing import Optional, List
from database.connection import get_database
from database.models import User


class UserRepository:
    """Repository for user operations"""

    def save(self, user: User) -> bool:
        """Save or update a user"""
        try:
            db = get_database()
            query = '''
                INSERT OR REPLACE INTO users
                (user_id, email, name, picture, google_id, created_at, last_login, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                user.user_id,
                user.email,
                user.name,
                user.picture,
                user.google_id,
                user.created_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
                1 if user.is_active else 0
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None

    def get_by_google_id(self, google_id: str) -> Optional[User]:
        """Get user by Google ID"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM users WHERE google_id = ?",
                (google_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            print(f"Error getting user by google_id: {e}")
            return None

    def get_by_user_id(self, user_id: str) -> Optional[User]:
        """Get user by user_id"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            print(f"Error getting user by user_id: {e}")
            return None

    def update_last_login(self, user_id: str) -> bool:
        """Update last login timestamp"""
        try:
            db = get_database()
            db.execute_query(
                "UPDATE users SET last_login = ? WHERE user_id = ?",
                (datetime.now().isoformat(), user_id)
            )
            db.commit()
            return True
        except Exception as e:
            print(f"Error updating last login: {e}")
            return False

    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row['user_id'],
            email=row['email'],
            name=row.get('name'),
            picture=row.get('picture'),
            google_id=row.get('google_id'),
            created_at=datetime.fromisoformat(row['created_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row.get('last_login') else None,
            is_active=bool(row.get('is_active', 1))
        )


# Global repository instance
_user_repo: Optional[UserRepository] = None


def get_user_repository() -> UserRepository:
    """Get user repository instance"""
    global _user_repo
    if _user_repo is None:
        _user_repo = UserRepository()
    return _user_repo

