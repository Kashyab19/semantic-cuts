import hashlib
import os
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Table, create_engine, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# --- CONFIG ---
# This matches the DATABASE_URL in your docker-compose
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://user:password@localhost:5432/semantic_cuts"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---

# 1. The Link Table (Many-to-Many)
# "User X has Video Y in their library"
user_library = Table(
    "user_library",
    Base.metadata,
    Column("user_id", String, primary_key=True),  # The User ID (e.g. "tenant-alpha")
    Column("video_id", String, ForeignKey("videos.id"), primary_key=True),
    Column("added_at", DateTime, default=func.now()),
)


# 2. The Content Table
# "The physical video asset"
class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    url_hash = Column(String, unique=True, index=True, nullable=False)  # SHA256 of URL
    url = Column(String, nullable=False)
    title = Column(String)
    status = Column(String, default="queued")  # queued, processing, completed, error
    created_at = Column(DateTime, default=func.now())


# --- HELPERS ---


def init_db():
    """Creates the tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_url_hash(url: str) -> str:
    """Consistent hashing to find duplicates."""
    return hashlib.sha256(url.strip().encode()).hexdigest()


def update_status(job_id: str, status: str):
    """Used by Manager/Worker to update status."""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == job_id).first()
        if video:
            video.status = status
            db.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        db.close()


def add_video_to_library(db, url: str, title: str, user_id: str):
    """
    The Smart Ingest Logic:
    - If video exists: Link it to user. Return (video, is_new=False).
    - If video new: Create it, Link it. Return (video, is_new=True).
    """
    url_hash = get_url_hash(url)

    # 1. Check if the video physically exists
    video = db.query(Video).filter(Video.url_hash == url_hash).first()
    is_new = False

    if not video:
        is_new = True
        video = Video(id=uuid.uuid4().hex, url_hash=url_hash, url=url, title=title)
        db.add(video)
        db.commit()
        db.refresh(video)

    # 2. Link to the User (Access Control)
    # Check if this specific user already has it
    link_exists = (
        db.query(user_library).filter_by(user_id=user_id, video_id=video.id).first()
    )

    if not link_exists:
        stmt = user_library.insert().values(user_id=user_id, video_id=video.id)
        db.execute(stmt)
        db.commit()

    return video, is_new


def get_user_video_ids(db, user_id: str):
    """Get the list of video IDs this user is allowed to search."""
    results = db.query(user_library.c.video_id).filter_by(user_id=user_id).all()
    # Results is list of tuples [('id1',), ('id2',)], flatten it:
    return [r[0] for r in results]
