"""
Database models for CodeForensics API.
Uses SQLAlchemy with SQLite (local) / PostgreSQL (production).
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Analysis(Base):
    """Stores commit analysis results."""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    repo_url = Column(String, index=True)
    commit_sha = Column(String, index=True)
    
    # Prediction results
    risk_score = Column(Float)
    risk_level = Column(String)
    
    # Features used
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    files_changed = Column(Integer)
    lines_added = Column(Integer)
    lines_deleted = Column(Integer)
    complexity_delta = Column(Float)
    avg_file_churn = Column(Float)
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)


class TrackedRepo(Base):
    """Repositories being tracked."""
    __tablename__ = "tracked_repos"
    
    id = Column(Integer, primary_key=True, index=True)
    repo_url = Column(String, unique=True, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    last_analyzed = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized")


def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
