import json
import os

import cv2
import database
import redis
import yt_dlp
from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
from kafka.admin import NewTopic

# --- CONFIG ---
INGEST_TOPIC = "video_ingestion"
CHUNK_TOPIC = "chunk_processing"
KAFKA_BROKER = "127.0.0.1:9094"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

CHUNK_DURATION = 5  # Seconds per minion
NUM_PARTITIONS = 4

# --- INFRASTRUCTURE ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def setup_topic():
    """Ensures the topic exists and has enough partitions for parallel processing"""
    admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)

    existing_topics = admin_client.list_topics()

    if CHUNK_TOPIC not in existing_topics:
        print(f"Creating topic '{CHUNK_TOPIC}' with {NUM_PARTITIONS} partitions...")
        topic_list = [
            NewTopic(
                name=CHUNK_TOPIC, num_partitions=NUM_PARTITIONS, replication_factor=1
            )
        ]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    else:
        # Optional: You could check partition count here and increase it,
        print(f"Topic '{CHUNK_TOPIC}' exists.")


def download(url, job_id):
    """Downloads video to a shared folder accessible by all workers"""
    output_path = f"assets/videos/{job_id}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f" Video already exists, skipping download.")
        return output_path

    print(f" Downloading {url}...")
    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,  # Fix for 403 errors
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f" Download failed: {e}")
        return None


def dispatch_job(job):
    job_id = job["job_id"]
    url = job["url"]

    print(f" Manager: Received Job {job_id}")

    database.add_video(job_id, url, title=f"Video {job_id[:8]}")
    # 1. Download (The "Shared Asset")
    video_path = download(url, job_id)
    if not video_path:
        return

    # 2. Probe (Get Duration)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    print(f" Duration: {duration_sec:.2f}s | FPS: {fps}")

    # 3. Split (The Logic)
    # We chop the video into 30-second tasks
    chunks = []
    for start in range(0, int(duration_sec), CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration_sec)
        chunks.append((start, end))

    total_chunks = len(chunks)
    print(f" Slicing into {total_chunks} chunks...")

    # 4. Update Scoreboard (Redis)
    # We set the "Remaining Count" to the number of chunks
    redis_key = f"job:{job_id}:pending"
    redis_client.set(redis_key, total_chunks)

    # 5. Dispatch (Fan-Out)
    for i, (start, end) in enumerate(chunks):
        chunk_payload = {
            "job_id": job_id,
            "video_path": video_path,  # Minions read this local file
            "start_time": start,
            "end_time": end,
            "chunk_index": i,
            "total_chunks": total_chunks,
        }
        producer.send(CHUNK_TOPIC, chunk_payload)

    database.update_status(job_id, "processing")

    producer.flush()
    print(f" Dispatched {total_chunks} tasks to '{CHUNK_TOPIC}'")


def start_manager():
    print(f" Manager listening on '{INGEST_TOPIC}'...")

    setup_topic()

    consumer = KafkaConsumer(
        INGEST_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="manager_group",
    )

    for message in consumer:
        dispatch_job(message.value)


if __name__ == "__main__":
    start_manager()
