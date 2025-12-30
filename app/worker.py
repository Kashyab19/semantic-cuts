import json
import logging
import os
import threading
import time
import uuid

import cv2
import redis
import requests
from kafka import KafkaConsumer
from kafka.errors import CommitFailedError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# --- INTERNAL IMPORTS ---
from app.audio import transcribe_chunk
from app.database import index as database
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
logger = logging.getLogger("Worker")

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
    """
    Standard processing logic. No Kafka specific code here.
    """
    job_id = task["job_id"]
    video_path = task["video_path"]
    user_id = task.get("user_id", "demo_user")

    if not os.path.exists(video_path):
        logger.critical(f"FILE MISSING: {video_path}")
        return

    # --- VISUAL PROCESSING ---
    try:
        detector = StructuralSceneDetector(threshold=12)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        start_frame = int(task["start_time"] * fps)
        max_frame = int(task["end_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        visual_points = []
        current_frame = start_frame
        last_cut_frame = start_frame
        HEARTBEAT_FRAMES = 60 * fps

        while cap.isOpened() and current_frame < max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            is_scene_change = detector.process_frame(frame)
            is_heartbeat = (current_frame - last_cut_frame) > HEARTBEAT_FRAMES

            if is_scene_change or is_heartbeat:
                timestamp = current_frame / fps
                cut_type = "scene_cut" if is_scene_change else "heartbeat"

                # Log sparsely to avoid spamming production logs
                logger.debug(f"Visual event ({cut_type}) at {timestamp:.2f}s")

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
                last_cut_frame = current_frame
            current_frame += 1
        cap.release()

        if visual_points:
            qdrant.upsert("video_frames", visual_points)
            logger.info(f"Saved {len(visual_points)} visual vectors.")

    except Exception as e:
        logger.error(f"Visual processing failed: {e}")
        raise e  # Re-raise to signal failure to the thread wrapper

    # --- AUDIO PROCESSING ---
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
        # Audio failure might be non-critical, decide if you want to raise e

    # --- REDUCE STEP ---
    try:
        pending_key = f"job:{job_id}:pending"
        remaining = redis_client.decr(pending_key)
        if remaining <= 0:
            logger.info(f"JOB {job_id} COMPLETE!")
            # Update database status
            database.update_status(job_id, "completed")
            redis_client.set(f"job:{job_id}:status", "completed")
    except Exception as e:
        logger.error(f"Error updating job status: {e}")


def start():
    logger.info("Worker Ready (Threaded Mode).")

    consumer = KafkaConsumer(
        CHUNK_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="minion_group",
        max_poll_records=1,  # Fetch only one
        enable_auto_commit=False,  # We commit manually
        max_poll_interval_ms=600000,  # 10m safety net (backup for thread loop)
    )

    for msg in consumer:
        task = msg.value
        logger.info(
            f"Processing chunk {task.get('chunk_index')} for job {task.get('job_id')}"
        )

        # 1. PAUSE: Stop fetching to prevent rebalance during processing
        partitions = consumer.assignment()
        consumer.pause(*partitions)  # <--- FIXED: Added * to unpack

        # 2. THREAD: Run processing in background
        thread_exc = []

        def worker():
            try:
                process_chunk(task)
            except Exception as e:
                thread_exc.append(e)

        t = threading.Thread(target=worker)
        t.start()

        # 3. HEARTBEAT: Keep the session alive while thread works
        while t.is_alive():
            consumer.poll(0)  # Sends heartbeat
            time.sleep(1)

        t.join()  # Ensure finish

        # 4. RESUME & COMMIT
        consumer.resume(*partitions)  # <--- FIXED: Added * to unpack

        if thread_exc:
            logger.error(f"Chunk failed: {thread_exc[0]}")
            # OPTIONAL: Send to Dead Letter Queue (DLQ) here
            # For now, we commit to avoid the Infinite Loop (Poison Pill)
            consumer.commit()
        else:
            try:
                consumer.commit()
                logger.info("Chunk committed successfully.")
            except CommitFailedError:
                logger.critical(
                    "Commit failed! Processing took too long even with heartbeats."
                )


if __name__ == "__main__":
    # Initialize database
    database.init_db()
    start()
