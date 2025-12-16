import json
import os
import shutil
import uuid

import cv2  # OpenCV
import torch
import yt_dlp
from kafka import KafkaConsumer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

# KAFKA CONFIG:s
KAFKA_TOPIC = "video_ingestion"
KAFKA_BROKER = "127.0.0.1:9094"

# QDRANT:
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "video_frames"
VECTOR_SIZE = 512

print("Loading CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("Model Loaded.")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure Collection Exists
if not qdrant.collection_exists(QDRANT_COLLECTION_NAME):
    print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def download(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading video from {url} to {output_path}")

    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "logtostderr": False,
        "source_address": "0.0.0.0",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def process(job):
    print(f"Processing job: {job}")

    video_url = job["url"]
    job_id = job["job_id"]

    video_path = f"assets/videos/video_{job_id}.mp4"

    try:
        download(video_url, video_path)
    except Exception as e:
        print(f"Error downloading video: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Fallback to prevent division by zero

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    points_to_upload = []

    current_frame_idx = 0

    print(f"Processing {video_path} ({fps} FPS, {total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optimize: Only process 1 frame every 2 seconds
        if current_frame_idx % (fps * 2) == 0:
            timestamp = current_frame_idx / fps

            # Convert OpenCV (BGR) to PIL (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Generate Embedding
            inputs = processor(images=pil_image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)

            # Normalize vector
            embedding = outputs[0].cpu().numpy()
            embedding = embedding / torch.norm(torch.tensor(embedding)).item()

            # Prepare for Qdrant
            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "video_id": job_id,
                        "url": video_url,
                        "timestamp": timestamp,
                        "second_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                    },
                )
            )
            print(f"Processed frame at {timestamp:.1f}s")

        current_frame_idx += 1  # Increment current frame

    # FIX 5: Release and Upload MUST be outside the while loop
    cap.release()

    if points_to_upload:
        print(f"Uploading {len(points_to_upload)} frames to DB for job {job_id}...")
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION_NAME,  # FIX 6: Use correct global var
            points=points_to_upload,
        )
        print("Indexing Complete!")

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)


def start_worker():
    print(f"Worker started. Listening on '{KAFKA_TOPIC}'...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="video_processor_group",
    )

    for message in consumer:
        print("-" * 30)
        print(f"Received Job: {message.value.get('job_id', 'Unknown')}")
        process(message.value)


if __name__ == "__main__":
    start_worker()
