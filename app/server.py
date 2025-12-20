import io
import json
import logging
import os
import tempfile

import cv2
import numpy as np
import redis
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from kafka import KafkaProducer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import CLIPModel, CLIPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Cuts - Inference Engine")

# --- CONFIGURATION ---

# Detect if running inside Docker or Local
IN_DOCKER = os.getenv("KUBERNETES_SERVICE_HOST") or os.path.exists("/.dockerenv")

# Define Hostnames
QDRANT_HOST = "qdrant" if IN_DOCKER else "localhost"
REDIS_HOST = "redis" if IN_DOCKER else "localhost"
REDPANDA_HOST = "redpanda" if IN_DOCKER else "localhost"
REDPANDA_PORT = 9092 if IN_DOCKER else 9094

COLLECTION_NAME = "video_frames"
VECTOR_SIZE = 512

# --- INFRASTRUCTURE CONNECTIONS ---

# Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    redis_client.ping()
    logger.info(f"Redis connected at {REDIS_HOST}:6379")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Qdrant
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
    logger.info(f"Qdrant connected at {QDRANT_HOST}:6333")
except Exception as e:
    logger.error(f"Qdrant connection failed: {e}")
    qdrant_client = None

# Redpanda (Kafka)
try:
    producer = KafkaProducer(
        bootstrap_servers=f"{REDPANDA_HOST}:{REDPANDA_PORT}",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    logger.info(f"Redpanda connected at {REDPANDA_HOST}:{REDPANDA_PORT}")
except Exception as e:
    logger.error(f"Redpanda connection failed: {e}")
    producer = None

# --- ML MODEL ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading CLIP model on {DEVICE}...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logger.info("Model loaded successfully.")


@app.on_event("startup")
def startup_event():
    """Initialize Qdrant collection on server startup."""
    if not qdrant_client:
        logger.warning("Qdrant not connected, skipping initialization.")
        return

    try:
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")


def process_video(file_path: str, video_id: str):
    """
    Background task: Extracts frames, embeds them, and saves to Qdrant.
    TODO: Incorporate smart scene detection from yesterday to prevent memory usage.
    """
    logger.info(f"Starting processing for video: {video_id}")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {file_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback if FPS detection fails

    frame_interval = int(fps)  # Process 1 frame per second

    current_frame = 0
    points_to_upload = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if current_frame % frame_interval == 0:
            try:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Inference
                inputs = processor(
                    images=pil_image, return_tensors="pt", padding=True
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)

                # Normalize
                embedding = outputs[0].cpu().numpy()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Prepare Qdrant Point
                timestamp = current_frame / fps
                point_id = f"{video_id}_{int(timestamp)}"

                # Create a point UUID deterministically if needed, or let Qdrant handle it if using integers.
                # Here we use a string ID logic or hash it if Qdrant requires UUIDs.
                # For simplicity in this example, we assume Qdrant supports string IDs or we use a UUID helper.
                import uuid

                final_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

                points_to_upload.append(
                    models.PointStruct(
                        id=final_id,
                        vector=embedding.tolist(),
                        payload={
                            "video_id": video_id,
                            "timestamp": timestamp,
                            "frame_index": current_frame,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Error processing frame {current_frame}: {e}")

        current_frame += 1

    cap.release()

    # Bulk upload to Qdrant
    if points_to_upload and qdrant_client:
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME, points=points_to_upload
            )
            logger.info(
                f"Uploaded {len(points_to_upload)} frames to Qdrant for {video_id}."
            )
        except Exception as e:
            logger.error(f"Failed to upload to Qdrant: {e}")

    # Cleanup temp file
    try:
        os.remove(file_path)
    except OSError:
        pass

    # Notify Redpanda
    if producer:
        try:
            producer.send(
                "video_processed",
                {
                    "video_id": video_id,
                    "status": "completed",
                    "frames_processed": len(points_to_upload),
                },
            )
            producer.flush()
        except Exception as e:
            logger.error(f"Failed to send Kafka message: {e}")

    logger.info(f"Finished processing video: {video_id}")


@app.get("/health")
def health():
    return {
        "status": "operational",
        "device": DEVICE,
        "infrastructure": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "redpanda": "connected" if producer else "disconnected",
            "redis": "connected" if redis_client else "disconnected",
        },
    }


@app.post("/ingest")
async def ingest_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a video, saves it temporarily, and triggers background processing.
    """
    try:
        # Create a temp file to store the upload
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        video_id = os.path.splitext(file.filename)[0]

        # Trigger background task
        background_tasks.add_task(process_video, tmp_path, video_id)

        return {
            "status": "processing_started",
            "video_id": video_id,
            "message": "Video is being processed in the background.",
        }

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return {"error": str(e)}


@app.get("/search")
def search_video(query: str, limit: int = 5):
    """
    Search functionality using the modern Qdrant 'query_points' API.
    """
    if not qdrant_client:
        return {"error": "Qdrant is not connected"}

    try:
        # 1. Vectorize Text (Using your Torch logic)
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # 2. Normalize (Using Torch as per your snippet)
        text_vector = text_features[0]
        text_vector = text_vector / text_vector.norm(p=2, dim=-1, keepdim=True)
        text_vector_list = text_vector.cpu().numpy().tolist()

        # 3. Search Qdrant (Using query_points)
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=text_vector_list,
            limit=limit,
            with_payload=True,
        )

        # Access points from the result
        hits = search_result.points

        # 4. Format Results
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "video_id": payload.get("video_id", "unknown"),
                    # We save 'timestamp' as a float (e.g., 12.0), so we read that back
                    "timestamp": payload.get("timestamp", 0.0),
                    "frame_index": payload.get("frame_index"),
                }
            )

        return {"query": query, "results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
