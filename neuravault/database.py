"""
database.py - SQLite User Database for NeuraVault

Implements secure user storage with parameterized queries to prevent SQL injection.
All queries use parameter binding (?) instead of string formatting.

Features:
    - User registration with hashed passwords
    - User lookup by username
    - Role-based user management (admin / user)
    - SQL injection prevention via parameterized queries
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "neuravault_users.db"


def _get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with safe defaults."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    # Enforce foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    """Context manager that yields a DB connection and commits on success."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the users table if it does not exist."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL UNIQUE,
                email       TEXT    NOT NULL UNIQUE,
                hashed_pw   TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'user',
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
    logger.info("User database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# CRUD helpers – ALL use parameterized queries (SQL-injection safe)
# ---------------------------------------------------------------------------

def create_user(
    username: str,
    email: str,
    hashed_password: str,
    role: str = "user",
) -> Dict[str, Any]:
    """Insert a new user. Returns the created user dict."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, email, hashed_pw, role)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, hashed_password, role),
        )
        user_id = cursor.lastrowid

    return get_user_by_id(user_id)  # type: ignore[arg-type]


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Look up a user by username (parameterized – SQL-injection safe)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Look up a user by email (parameterized – SQL-injection safe)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Look up a user by ID (parameterized – SQL-injection safe)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def update_user_role(user_id: int, new_role: str) -> bool:
    """Update a user's role. Returns True if a row was updated."""
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE users SET role = ? WHERE id = ?", (new_role, user_id)
        )
    return cursor.rowcount > 0


def deactivate_user(user_id: int) -> bool:
    """Soft-delete a user by setting is_active = 0."""
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE users SET is_active = 0 WHERE id = ?", (user_id,)
        )
    return cursor.rowcount > 0
