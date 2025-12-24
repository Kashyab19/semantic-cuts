import io
import json
import logging
import os
import uuid
from typing import List, Optional

import redis
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastembed import TextEmbedding
from kafka import KafkaProducer
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

from app import database

# --- CONFIG ---
# Detect if running in Docker or Local
IN_DOCKER = os.path.exists("/.dockerenv")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

INGEST_TOPIC = "video_ingestion"
AUDIO_COLLECTION = "audio_transcripts"
VIDEO_COLLECTION = "video_frames"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Semantic Cuts")

# --- INFRASTRUCTURE ---
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )
    logger.info(f"Kafka connected at {KAFKA_BROKER}")
except Exception as e:
    logger.critical(f"Kafka Failed: {e}")
    producer = None

try:
    qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
    logger.info(f"Qdrant connected at {QDRANT_HOST}")
except Exception as e:
    logger.critical(f"Qdrant Failed: {e}")

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# --- MODELS ---
DEVICE = "cpu"  # Force CPU to avoid CUDA complications in Docker for now
logger.info(f"Loading CLIP on {DEVICE}...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

logger.info("Loading Text Embedding...")
text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


class VideoRequest(BaseModel):
    url: str
    title: Optional[str] = "Untitled"


class SearchResult(BaseModel):
    video_id: str
    timestamp: float
    end_timestamp: Optional[float] = None
    text: Optional[str] = None
    score: float
    type: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/video")
async def queue_video(payload: VideoRequest):
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka unavailable")

    job_id = uuid.uuid4().hex
    database.add_video(job_id, payload.url, payload.title)

    task_payload = {
        "job_id": job_id,
        "url": payload.url,
        "title": payload.title,
        "status": "queued",
    }

    producer.send(INGEST_TOPIC, task_payload)
    producer.flush()
    redis_client.set(f"job:{job_id}:status", "queued")

    logger.info(f"Queued Job {job_id}")
    return {"job_id": job_id, "status": "queued"}


@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        vector = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return {"vector": vector.cpu().numpy().tolist()[0]}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=List[SearchResult])
async def search(query: str, limit: int = 5):
    results = []
    logger.info(f"Searching for: {query}")

    # 1. AUDIO SEARCH
    try:
        text_vec = list(text_model.embed([query]))[0]
        audio_hits = qdrant.query_points(
            collection_name=AUDIO_COLLECTION,
            query=text_vec,
            limit=limit,
            with_payload=True,
        ).points
        for hit in audio_hits:
            results.append(
                {
                    "video_id": hit.payload.get("video_id"),
                    "timestamp": hit.payload.get("timestamp", 0.0),
                    "end_timestamp": hit.payload.get("end_timestamp"),
                    "text": hit.payload.get("text"),
                    "score": hit.score,
                    "type": "audio",
                }
            )
    except Exception as e:
        logger.error(f"Audio Search Failed: {e}")
        # Don't crash, just log

    # 2. VISUAL SEARCH
    try:
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(
            DEVICE
        )
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        clip_vec = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        video_hits = qdrant.query_points(
            collection_name=VIDEO_COLLECTION,
            query=clip_vec.cpu().numpy().tolist()[0],
            limit=limit,
            with_payload=True,
        ).points

        for hit in video_hits:
            results.append(
                {
                    "video_id": hit.payload.get("video_id"),
                    "timestamp": hit.payload.get("timestamp", 0.0),
                    "text": f"[Visual] {query}",
                    "score": hit.score,
                    "type": "visual",
                }
            )
    except Exception as e:
        logger.error(f"Visual Search Failed: {e}")
        # Don't crash, just log

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
