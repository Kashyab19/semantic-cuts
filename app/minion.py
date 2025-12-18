import json
import os
import sys
import uuid

import cv2
import redis
import torch
from kafka import KafkaConsumer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from scene_detector import SceneDetector
from transformers import CLIPModel, CLIPProcessor

# --- CONFIG ---
CHUNK_TOPIC = "chunk_processing"
KAFKA_BROKER = "127.0.0.1:9094"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"

# --- INFRASTRUCTURE ---
print("Minion reporting to duty... Loading CLIP...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on: {DEVICE}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Ensure Collection Exists
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
print("Ready to work.")


def process_chunk(task):
    job_id = task["job_id"]
    video_path = task["video_path"]
    start_time = task["start_time"]
    end_time = task["end_time"]
    chunk_index = task["chunk_index"]

    print(f" Processing Chunk #{chunk_index} ({start_time}s - {end_time}s)")

    # Instantiate Detector FRESH for every chunk so it doesn't remember old videos
    detector = SceneDetector(threshold=0.7)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # JUMP to the start time (The magic of Seek)
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    points_to_upload = []
    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate timestamp
        timestamp = current_frame / fps

        # STOP if we reached the end of our assigned chunk
        if timestamp > end_time:
            break

        # Ask the detector if this frame is worth processing
        is_new_scene = detector.process_frame(frame)

        if is_new_scene:
            # AI Processing (Only runs when the scene actually changes!)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            inputs = processor(images=pil_image, return_tensors="pt", padding=True).to(
                DEVICE
            )

            # no gradient means that we are asking clip not to train but to inference
            # this saves computation
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)

            # Normalize
            embedding = outputs[0].cpu().numpy()
            embedding = embedding / torch.norm(torch.tensor(embedding)).item()

            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "video_id": job_id,
                        "url": task.get("url", "local_file"),
                        "timestamp": timestamp,
                        "second_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                        "type": "scene_cut",
                    },
                )
            )
            # Log only on cuts
            print(f"Scene Change at {timestamp:.1f}s -> Encoded.")

        current_frame += 1

    cap.release()

    # Upload to DB (Batch upload is more efficient than one-by-one)
    if points_to_upload:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)
        print(f" Saved {len(points_to_upload)} vectors for Chunk #{chunk_index}.")
    else:
        print(f" No scenes detected in Chunk #{chunk_index} (Static video?)")

    # --- THE REDUCE STEP ---
    # Decrement the pending counter in Redis
    redis_key = f"job:{job_id}:pending"
    remaining = int(redis_client.decr(redis_key))

    if remaining <= 0:
        print(f"JOB {job_id} COMPLETE! (I was the last one)")
        # Optional: Add logic here to update SQLite status to 'completed'
    else:
        print(f"   There are {remaining} chunks left for other minions.")


def start_minion():
    print(f"Minion listening on '{CHUNK_TOPIC}'...")
    consumer = KafkaConsumer(
        CHUNK_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="minion_group",  # All minions share this group to load-balance
        auto_offset_reset="latest",
    )

    for message in consumer:
        try:
            process_chunk(message.value)
        except Exception as e:
            print(f"❌ Error processing chunk: {e}")
            # In a real production system, you would send this to a Dead Letter Queue (DLQ)


if __name__ == "__main__":
    start_minion()
