import json
import os
import uuid

import cv2
import redis
import requests
from app.scene_detect import SceneDetector
from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# --- CONFIG ---
CHUNK_TOPIC = "chunk_processing"

# The Minion now talks to the "Inference Service" instead of loading the model itself
INFERENCE_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8001/embed")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"

# --- INFRASTRUCTURE ---
print(f"Minion reporting for duty. Inference Server: {INFERENCE_URL}")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Ensure Collection Exists
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
print("Ready to work.")


def get_embedding_from_server(frame):
    """
    Sends the raw image frame to the Inference API (server.py)
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
            print(f"Error from Inference Server: {response.text}")
            return None
    except Exception as e:
        print(f"Connection Error to Inference Server: {e}")
        return None


def process_chunk(task):
    job_id = task["job_id"]
    video_path = task["video_path"]
    start_time = task["start_time"]
    end_time = task["end_time"]
    chunk_index = task["chunk_index"]

    print(f"Processing Chunk #{chunk_index} ({start_time}s - {end_time}s)")

    # Instantiate Detector FRESH for every chunk
    detector = SceneDetector(threshold=0.7)

    # --- STUTTER FIX: DEBOUNCE LOGIC ---
    # We will ignore any cuts that happen less than 1.0 second after the previous one
    last_cut_timestamp = -1.0
    MIN_SCENE_DURATION = 1.0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # JUMP to the start time
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    points_to_upload = []
    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / fps

        # STOP if we reached the end of our assigned chunk
        if timestamp > end_time:
            break

        # 1. COOLDOWN CHECK (Fixes the "stutter" bursts)
        # If we just cut recently, skip the heavy processing
        if (
            last_cut_timestamp != -1.0
            and (timestamp - last_cut_timestamp) < MIN_SCENE_DURATION
        ):
            current_frame += 1
            continue

        # 2. SCENE DETECTION
        is_new_scene = detector.process_frame(frame)

        if is_new_scene:
            # 3. REMOTE INFERENCE (Call the separate Brain service)
            vector = get_embedding_from_server(frame)

            if vector:
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "video_id": job_id,
                            "url": task.get("url", "local_file"),
                            "timestamp": timestamp,
                            "second_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                            "type": "scene_cut",
                        },
                    )
                )
                print(f"Scene Change at {timestamp:.1f}s -> Encoded.")

                # Update the last cut time so we enforce the cooldown
                last_cut_timestamp = timestamp

        current_frame += 1

    cap.release()

    # Upload to DB
    if points_to_upload:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)
        print(f"Saved {len(points_to_upload)} vectors for Chunk #{chunk_index}.")
    else:
        print(f"No scenes detected in Chunk #{chunk_index} (Static video?)")

    # --- THE REDUCE STEP ---
    redis_key = f"job:{job_id}:pending"
    # Force cast to int for type safety
    remaining = int(redis_client.decr(redis_key))

    if remaining <= 0:
        print(f"JOB {job_id} COMPLETE! (I was the last one)")
    else:
        print(f"There are {remaining} chunks left for other minions.")


def start_minion():
    print(f"Minion listening on '{CHUNK_TOPIC}'...")
    consumer = KafkaConsumer(
        CHUNK_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="minion_group",
        auto_offset_reset="latest",
    )

    for message in consumer:
        try:
            process_chunk(message.value)
        except Exception as e:
            print(f"Error processing chunk: {e}")


if __name__ == "__main__":
    start_minion()
