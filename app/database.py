import datetime
import sqlite3

DB_NAME = "vidsearch.db"


def init_db():
    """Creates the table if it doesn't exist"""
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
        (job_id, title, url, "processing", datetime.datetime.now()),
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
    # Return results as a dictionary-like object
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM videos ORDER BY created_at DESC")
    return c.fetchall()


# Initialize immediately when this module is imported
init_db()
