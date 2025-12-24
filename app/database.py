import datetime
import os
import sqlite3

# CRITICAL: Point to the shared volume directory
# If we don't do this, API and Manager will have different databases!
DB_FOLDER = "/app/assets"
DB_NAME = os.path.join(DB_FOLDER, "vidsearch.db")


def init_db():
    """Creates the table if it doesn't exist"""
    # Ensure the directory exists first
    os.makedirs(DB_FOLDER, exist_ok=True)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            status TEXT,
            duration REAL,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def add_video(job_id, url, title="Unknown Video"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO videos (id, title, url, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, title, url, "queued", datetime.datetime.now()),
    )
    conn.commit()
    conn.close()


def update_status(job_id, status):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE videos SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()
    conn.close()


def get_all_videos():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM videos ORDER BY created_at DESC")
    return c.fetchall()


# Initialize immediately
init_db()
