import json
import os
import uuid

import cv2
import redis
import requests
import logging 

from app.scene_detector import SceneDetector
from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

# --- CONFIG ---
CHUNK_TOPIC = "chunk_processing"

# The Minion now talks to the "Inference Service" instead of loading the model itself
INFERENCE_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8001/embed")
INFERENCE_BATCH_URL = INFERENCE_URL.replace("/embed", "/embed_batch")
BATCH_SIZE = 16
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"

# --- INFRASTRUCTURE ---
logger.info(f"Minion reporting for duty. Inference Server: {INFERENCE_URL}")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Ensure Collection Exists
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
logger.info("Ready to work.")


def get_embedding_from_server(frame):
    """
    Sends the raw image frame to the Embedding API (server.py)
    and receives the 512-float vector.
    """
    # Encode frame to JPEG in memory
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None

    try:
        response = requests.post(INFERENCE_URL, files={"file": buffer.tobytes()})
        if response.status_code == 200:
            return response.json()["vector"]
        else:
            logger.error(f"Error from Embedding Server: {response.text}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Connection Error to Embedding Server: {e}", exc_info=True)
        return None


def get_embeddings_batch(frames):
    """
    Sends multiple frames in a single request to the Embedding API.
    Returns a list of 512-float vectors (same order as input frames).
    """
    files = []
    for frame in frames:
        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            files.append(("files", ("frame.jpg", buffer.tobytes(), "image/jpeg")))

    if not files:
        return []

    try:
        response = requests.post(INFERENCE_BATCH_URL, files=files)
        if response.status_code == 200:
            return response.json()["vectors"]
        else:
            logger.error(f"Error from Embedding Server (batch): {response.text}", exc_info=True)
            return []
    except Exception as e:
        logger.error(f"Connection Error to Embedding Server (batch): {e}", exc_info=True)
        return []


def process_chunk(task):
    job_id = task["job_id"]
    video_path = task["video_path"]
    start_time = task["start_time"]
    end_time = task["end_time"]
    chunk_index = task["chunk_index"]

    logger.info(f"Processing Chunk #{chunk_index} ({start_time}s - {end_time}s)")

    # Instantiate Detector FRESH for every chunk
    detector = SceneDetector(threshold=0.7)

    # --- STUTTER FIX: DEBOUNCE LOGIC ---
    last_cut_timestamp = -1.0
    MIN_SCENE_DURATION = 1.0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # JUMP to the start time
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Build the video serving URL so the frontend can play it
    video_serve_url = f"/videos/{job_id}.mp4"

    # --- PHASE 1: Detect scenes and collect frames ---
    scene_frames = []   # raw frames to embed
    scene_metadata = []  # corresponding metadata for each frame
    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / fps

        if timestamp > end_time:
            break

        # 1. COOLDOWN CHECK
        if (
            last_cut_timestamp != -1.0
            and (timestamp - last_cut_timestamp) < MIN_SCENE_DURATION
        ):
            current_frame += 1
            continue

        # 2. SCENE DETECTION
        is_new_scene = detector.process_frame(frame)

        if is_new_scene:
            scene_frames.append(frame)
            scene_metadata.append({
                "video_id": job_id,
                "url": video_serve_url,
                "source_url": task.get("url", ""),
                "timestamp": timestamp,
                "second_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                "type": "scene_cut",
            })
            last_cut_timestamp = timestamp

        current_frame += 1

    cap.release()

    # --- PHASE 2: Batch embed all scene frames ---
    points_to_upload = []

    for i in range(0, len(scene_frames), BATCH_SIZE):
        batch_frames = scene_frames[i:i + BATCH_SIZE]
        batch_meta = scene_metadata[i:i + BATCH_SIZE]

        vectors = get_embeddings_batch(batch_frames)

        for vector, meta in zip(vectors, batch_meta):
            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=meta,
                )
            )

        logger.info(f"Batch embedded {len(vectors)}/{len(batch_frames)} frames for Chunk #{chunk_index}")

    # --- PHASE 3: Upload to Qdrant ---
    if points_to_upload:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)
        logger.info(f"Saved {len(points_to_upload)} vectors for Chunk #{chunk_index}.")
    else:
        logger.info(f"No scenes detected in Chunk #{chunk_index} (Static video?)")

    # --- PHASE 4: Update job progress ---
    redis_key = f"job:{job_id}:pending"
    remaining = int(redis_client.decr(redis_key))

    if remaining <= 0:
        from app import database
        database.update_status(job_id, "completed")
        logger.info(f"JOB {job_id} COMPLETE! (I was the last one)")
    else:
        logger.info(f"There are {remaining} chunks left for other minions.")


def start_minion():
    logger.info(f"Minion listening on '{CHUNK_TOPIC}'...")
    consumer = KafkaConsumer(
        CHUNK_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="minion_group",
        auto_offset_reset="earliest",
    )

    for message in consumer:
        try:
            process_chunk(message.value)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)


if __name__ == "__main__":
    start_minion()
