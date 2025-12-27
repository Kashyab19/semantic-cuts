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

# --- INTERNAL IMPORTS ---
from app.audio import transcribe_chunk

# CHANGE: Importing your new isolated class
from app.utils.structural_scene_detector import StructuralSceneDetector

# --- CONFIG ---
CHUNK_TOPIC = "chunk_processing"
INFERENCE_URL = os.getenv("INFERENCE_API_URL", "http://api:8000/embed")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Minion")

# --- INFRASTRUCTURE ---
try:
    qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
except Exception as e:
    logger.critical(f"Infra connection failed: {e}")
    exit(1)

# Ensure Collections
try:
    for name, dim in [("video_frames", 512), ("audio_transcripts", 384)]:
        if not qdrant.collection_exists(name):
            qdrant.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
except Exception as e:
    logger.error(f"Collection check failed: {e}")


def get_embedding(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    try:
        res = requests.post(INFERENCE_URL, files={"file": buffer.tobytes()})
        return res.json()["vector"] if res.status_code == 200 else None
    except Exception:
        return None


def process_chunk(task):
    job_id = task["job_id"]
    video_path = task["video_path"]
    user_id = task.get("user_id", "demo_user")

    # 1. GHOST CHECK
    if not os.path.exists(video_path):
        logger.critical(f"FILE MISSING: {video_path}")
        return

    # 2. VISUAL PROCESSING
    try:
        # Initialize your new class
        detector = StructuralSceneDetector(threshold=12)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        start_frame = int(task["start_time"] * fps)
        max_frame = int(task["end_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        visual_points = []
        current_frame = start_frame
        last_cut_frame = start_frame  # Track for heartbeat

        # PARAM: Force a cut every 60 seconds if scene is static
        # This prevents "vector starvation" on long slides
        HEARTBEAT_FRAMES = 60 * fps

        while cap.isOpened() and current_frame < max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Check for scene change OR heartbeat timeout
            is_scene_change = detector.process_frame(frame)
            is_heartbeat = (current_frame - last_cut_frame) > HEARTBEAT_FRAMES

            if is_scene_change or is_heartbeat:
                timestamp = current_frame / fps
                cut_type = "scene_cut" if is_scene_change else "heartbeat"

                logger.info(f"Visual event ({cut_type}) at {timestamp:.2f}s")

                vec = get_embedding(frame)
                if vec:
                    visual_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "video_id": job_id,
                                "timestamp": timestamp,
                                "type": cut_type,
                                "user_id": user_id,
                                "method": "dhash",
                            },
                        )
                    )
                # Reset the heartbeat timer
                last_cut_frame = current_frame

            current_frame += 1
        cap.release()

        if visual_points:
            qdrant.upsert("video_frames", visual_points)
            logger.info(f"Saved {len(visual_points)} visual vectors.")

    except Exception as e:
        logger.error(f"Visual processing failed: {e}")

    # 3. AUDIO PROCESSING (Standard)
    try:
        audio_data = transcribe_chunk(
            video_path, job_id, task["start_time"], task["end_time"]
        )
        if audio_data:
            points = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload={**p["payload"], "user_id": user_id},
                )
                for p in audio_data
            ]
            qdrant.upsert("audio_transcripts", points)
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")

    # 4. REDUCE
    try:
        if redis_client.decr(f"job:{job_id}:pending") <= 0:
            logger.info(f"JOB {job_id} COMPLETE!")
    except Exception:
        pass


def start():
    logger.info("Minion Ready.")
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
