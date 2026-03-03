"""
security.py - Security Module for NeuraVault

Implements all security best practices:
    1. Rate Limiting        - per-IP request throttling via slowapi
    2. Authentication       - JWT bearer tokens (access + refresh)
    3. Authorization        - role-based access control (admin / user)
    4. Input Validation     - centralised sanitisation helpers
    5. CORS Configuration   - strict origin allowlist
    6. Security Headers     - OWASP-recommended HTTP headers
    7. SQL Injection Prev.  - enforced in database.py (parameterized queries)
    8. XSS Prevention       - HTML entity escaping + CSP header
    9. CSRF Protection      - double-submit cookie pattern
   10. Password Hashing     - bcrypt via passlib
"""

import os
import re
import html
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import Request, HTTPException, Depends, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator, EmailStr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv("NV_SECRET_KEY", secrets.token_urlsafe(64))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("NV_ACCESS_TOKEN_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("NV_REFRESH_TOKEN_DAYS", "7"))

ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Roles
ROLE_ADMIN = "admin"
ROLE_USER = "user"

# ---------------------------------------------------------------------------
# 10. Password Hashing (bcrypt)
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# 2. Authentication – JWT helpers
# ---------------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a signed JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Dependency: get current user from Authorization header
# ---------------------------------------------------------------------------
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """FastAPI dependency – extracts and validates the JWT bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Access token required.",
        )
    username: Optional[str] = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject."
        )

    from neuravault.database import get_user_by_username

    user = get_user_by_username(username)
    if user is None or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return user


# ---------------------------------------------------------------------------
# 3. Authorization – role checker dependency
# ---------------------------------------------------------------------------
class RoleChecker:
    """FastAPI dependency that verifies the user has the required role."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    async def __call__(self, user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource.",
            )
        return user


require_admin = RoleChecker([ROLE_ADMIN])
require_user = RoleChecker([ROLE_ADMIN, ROLE_USER])


# ---------------------------------------------------------------------------
# 4. Input Validation – Pydantic models with strict validators
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username must be alphanumeric (underscores allowed).")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=30)
    password: str = Field(..., min_length=1, max_length=128)


class SecureQueryRequest(BaseModel):
    """Query with XSS-safe validation."""

    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        # Strip HTML tags and escape entities (XSS prevention)
        v = sanitize_input(v)
        if not v.strip():
            raise ValueError("Question cannot be empty after sanitisation.")
        return v


# ---------------------------------------------------------------------------
# 8. XSS Prevention helpers
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_input(text: str) -> str:
    """Remove HTML tags, escape entities, and strip dangerous patterns."""
    text = _TAG_RE.sub("", text)
    text = html.escape(text, quote=True)
    # Remove null bytes
    text = text.replace("\x00", "")
    return text.strip()


def sanitize_output(text: str) -> str:
    """Escape output for safe rendering (double-defence)."""
    return html.escape(text, quote=True)


# ---------------------------------------------------------------------------
# 6. Security Headers middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds OWASP-recommended security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:*; "
            "frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

        return response


# ---------------------------------------------------------------------------
# 9. CSRF Protection middleware (double-submit cookie pattern)
# ---------------------------------------------------------------------------
CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "x-csrf-token"
CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Double-submit cookie CSRF protection.

    • On every response a random csrf_token cookie is set (if absent).
    • Mutating requests (POST/PUT/DELETE/PATCH) must include an
      X-CSRF-Token header whose value matches the cookie.
    """

    async def dispatch(self, request: Request, call_next):
        # --- Check on mutating methods ---------------------------------
        if request.method not in CSRF_SAFE_METHODS:
            # Skip CSRF for auth endpoints (no cookie yet) and health
            path = request.url.path
            csrf_exempt = (
                path.startswith("/api/auth/")
                or path == "/api/health"
                or path == "/docs"
                or path == "/openapi.json"
            )

            if not csrf_exempt:
                cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
                header_token = request.headers.get(CSRF_HEADER_NAME)

                if not cookie_token or not header_token:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="CSRF token missing.",
                    )
                if not secrets.compare_digest(cookie_token, header_token):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="CSRF token mismatch.",
                    )

        response: Response = await call_next(request)

        # --- Set cookie if not present ---------------------------------
        if CSRF_COOKIE_NAME not in request.cookies:
            token = secrets.token_urlsafe(32)
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=token,
                httponly=False,       # JS must read it to send in header
                samesite="strict",
                secure=False,         # Set True in production with HTTPS
                max_age=3600,
            )

        return response


# ---------------------------------------------------------------------------
# 1. Rate Limiting configuration (used via slowapi in api.py)
# ---------------------------------------------------------------------------
def get_remote_address(request: Request) -> str:
    """Extract client IP for rate-limit keying."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
