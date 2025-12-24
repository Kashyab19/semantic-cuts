import json
import logging
import os
import uuid

import cv2
import redis
import requests
from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.audio import transcribe_chunk

# --- INTERNAL IMPORTS ---
from app.scene_detect import SceneDetector

# --- CONFIG ---
CHUNK_TOPIC = "chunk_processing"
# Use environment variable for flexibility, default to internal docker network alias
INFERENCE_URL = os.getenv("INFERENCE_API_URL", "http://api:8000/embed")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Minion")

logger.info(f"Minion reporting. Brain at: {INFERENCE_URL}")

# --- INFRASTRUCTURE CONNECTIONS ---
try:
    qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
except Exception as e:
    logger.critical(f"Failed to connect to infrastructure: {e}")
    exit(1)

# Ensure Collections
try:
    for name, dim in [("video_frames", 512), ("audio_transcripts", 384)]:
        if not qdrant.collection_exists(name):
            qdrant.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {name}")
except Exception as e:
    logger.error(f"Failed to ensure collections: {e}")


def get_embedding(frame):
    """
    Sends a frame to the API for embedding.
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    try:
        res = requests.post(INFERENCE_URL, files={"file": buffer.tobytes()})
        if res.status_code == 200:
            return res.json()["vector"]
        else:
            logger.error(f"Embedding failed: {res.status_code} - {res.text}")
            return None
    except Exception as e:
        logger.error(f"Embedding API Error: {e}")
        return None


def process_chunk(task):
    job_id = task["job_id"]
    video_path = task["video_path"]  # Uses /app/assets/...
    chunk_index = task["chunk_index"]

    logger.info(f"--- Processing Chunk #{chunk_index} for Job {job_id} ---")

    # 1. GHOST FILE CHECK
    if not os.path.exists(video_path):
        logger.critical(f"FILE MISSING: Worker cannot see {video_path}")
        # List directory to help debug volume mounting issues
        base_dir = os.path.dirname(video_path)
        if os.path.exists(base_dir):
            logger.info(f"Contents of {base_dir}: {os.listdir(base_dir)}")
        else:
            logger.error(f"Base directory {base_dir} does not exist!")
        return

    # 2. VISUAL PROCESSING
    try:
        # Lower threshold to 0.6 to be more sensitive
        detector = SceneDetector(threshold=0.6)
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logger.error(f"Video FPS is 0. Corrupt file? {video_path}")
            cap.release()
            return

        # Jump to start of chunk
        start_frame = int(task["start_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        visual_points = []
        current_frame = start_frame
        max_frame = int(task["end_time"] * fps)

        logger.info(f"Scanning frames {start_frame} to {max_frame}...")

        while cap.isOpened() and current_frame < max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if detector.process_frame(frame):
                timestamp = current_frame / fps
                logger.info(f"Cut detected at {timestamp:.2f}s")

                vec = get_embedding(frame)
                if vec:
                    visual_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "video_id": job_id,
                                "timestamp": timestamp,
                                "type": "scene_cut",
                            },
                        )
                    )
            current_frame += 1
        cap.release()

        if visual_points:
            qdrant.upsert("video_frames", visual_points)
            logger.info(f"Saved {len(visual_points)} visual vectors.")
        else:
            logger.info("No scene cuts detected in this chunk.")

    except Exception as e:
        logger.error(f"Visual processing failed: {e}")

    # 3. AUDIO PROCESSING
    try:
        audio_data = transcribe_chunk(
            video_path, job_id, task["start_time"], task["end_time"]
        )
        if audio_data:
            points = [
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in audio_data
            ]
            qdrant.upsert("audio_transcripts", points)
            logger.info(f"Saved {len(points)} audio segments.")
        else:
            logger.info("No audio data extracted.")

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")

    # 4. REDUCE (Track Completion)
    try:
        remaining = redis_client.decr(f"job:{job_id}:pending")
        if remaining <= 0:
            logger.info(f"JOB {job_id} COMPLETE!")
            # Optional: Update status in DB to 'completed' here
    except Exception as e:
        logger.error(f"Redis update failed: {e}")


def start():
    logger.info("Waiting for tasks...")
    consumer = KafkaConsumer(
        CHUNK_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="minion_group",
    )
    for msg in consumer:
        process_chunk(msg.value)


if __name__ == "__main__":
    start()
