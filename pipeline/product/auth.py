"""
Kullanıcı yönetimi, parola hashleme, API anahtarı üretimi.
- PostgreSQL (SQLAlchemy 2.0) — Render.com uyumlu
- PBKDF2-SHA256 ile parola hashleme (100.000 iterasyon)
- Timing attack'lara karşı hmac.compare_digest koruması
"""
import hashlib
import hmac
import os
import uuid
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Veritabanı Bağlantısı
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable tanımlı değil. "
        "Lokal geliştirme için .env dosyası oluşturun, production için Render dashboard'dan ayarlayın."
    )

# Render.com "postgres://" ile başlatır, SQLAlchemy "postgresql+psycopg2://" bekler
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(254), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    api_key = Column(String(64), unique=True, nullable=False, index=True)
    verification_credits = Column(Integer, default=3, nullable=False)
    tier = Column(String(16), default="free", nullable=False)
    invite_code = Column(String(16), unique=True, nullable=False, index=True)
    referred_by = Column(String(16), nullable=True)
    has_made_action = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Interest(Base):
    __tablename__ = "interests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(254), nullable=False)
    package = Column(String(64), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class UsageLog(Base):
    __tablename__ = "usage_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(String(64), nullable=False)
    input_text = Column(Text, nullable=True)
    result_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    usage_log_id = Column(Integer, ForeignKey("usage_log.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # +1 / -1
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# ---------------------------------------------------------------------------
# Güvenlik ve Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------
_PBKDF2_ITER = 100_000
_SALT_BYTES = 16
_tables_initialized = False

def init_db() -> None:
    """Tabloları ilk DB bağlantısında oluşturur (import sırasında değil)."""
    global _tables_initialized
    if not _tables_initialized:
        Base.metadata.create_all(bind=engine)
        _tables_initialized = True

def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = os.urandom(_SALT_BYTES)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITER)
    return f"{salt.hex()}:{pwd_hash.hex()}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Timing attack'a dayanıklı parola doğrulaması."""
    if not stored_password or not provided_password:
        return False
    try:
        salt_hex, hash_hex = stored_password.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected_hash = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, _PBKDF2_ITER)
        actual_hash = bytes.fromhex(hash_hex)
        return hmac.compare_digest(expected_hash, actual_hash)
    except (ValueError, AttributeError):
        return False

def generate_unique_key() -> str:
    return uuid.uuid4().hex

@contextmanager
def get_connection() -> Iterator[Session]:
    """Veritabanı oturum yöneticisi."""
    init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except IntegrityError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
