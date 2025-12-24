import json
import os

import cv2
import redis
import yt_dlp
from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
from kafka.admin import NewTopic

# --- INTERNAL IMPORTS ---
from app import database  # <--- FIXED IMPORT

# --- CONFIG ---
INGEST_TOPIC = "video_ingestion"
CHUNK_TOPIC = "chunk_processing"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

CHUNK_DURATION = 30  # Increased to 30s for better context
NUM_PARTITIONS = 4

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)


def setup_topic():
    admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    if CHUNK_TOPIC not in admin_client.list_topics():
        print(f"Creating topic '{CHUNK_TOPIC}'...")
        admin_client.create_topics(
            [
                NewTopic(
                    name=CHUNK_TOPIC,
                    num_partitions=NUM_PARTITIONS,
                    replication_factor=1,
                )
            ]
        )


def download(url, job_id):
    # DOWNLOAD TO SHARED VOLUME
    output_path = f"/app/assets/{job_id}.mp4"

    if os.path.exists(output_path):
        return output_path

    print(f"Downloading {url}...")
    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def dispatch_job(job):
    job_id = job["job_id"]
    url = job["url"]
    print(f"Manager: Processing {job_id}")

    # 1. Download
    video_path = download(url, job_id)
    if not video_path:
        return

    # 2. Update Status
    database.update_status(job_id, "processing")

    # 3. Probe Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    # 4. Split
    chunks = []
    for start in range(0, int(duration_sec), CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration_sec)
        chunks.append((start, end))

    redis_client.set(f"job:{job_id}:pending", len(chunks))

    # 5. Fan-out
    for i, (start, end) in enumerate(chunks):
        payload = {
            "job_id": job_id,
            "video_path": video_path,  # Sends absolute path (/app/assets/...)
            "start_time": start,
            "end_time": end,
            "chunk_index": i,
        }
        producer.send(CHUNK_TOPIC, payload)

    producer.flush()
    print(f"Dispatched {len(chunks)} chunks.")


def start_manager():
    setup_topic()
    consumer = KafkaConsumer(
        INGEST_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="manager_group",
    )
    print("Manager listening...")
    for message in consumer:
        dispatch_job(message.value)


if __name__ == "__main__":
    start_manager()
