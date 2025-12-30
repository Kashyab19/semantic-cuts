import io
import json
import logging
import os
import uuid
from typing import List, Optional

import redis
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastembed import TextEmbedding
from kafka import KafkaProducer
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

from app.database import index
from app.utils.index import deduplicate_results, reciprocal_rank_fusion

# Initialize database on startup
index.init_db()

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
async def queue_video(
    payload: VideoRequest,
    x_user_id: str = Header(default="demo_user", alias="x-user-id"),
):
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka Instance Unavailable")

    if not x_user_id:
        raise HTTPException(status_code=400, detail="User ID is missing")

    # Add video to database (smart ingest: deduplicates by URL hash)
    db = index.SessionLocal()
    try:
        video, is_new = index.add_video_to_library(
            db, payload.url, payload.title or "Untitled", x_user_id
        )
        job_id = video.id

        if is_new:
            logger.info(f"New video added: {job_id}")
        else:
            logger.info(f"Existing video linked to user: {job_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        db.close()

    task_payload = {
        "job_id": job_id,
        "url": payload.url,
        "title": payload.title,
        "user_id": x_user_id,
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
async def search(query: str, limit: int = 10):
    audio_results = []
    video_results = []

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
            audio_results.append(
                {
                    "video_id": hit.payload.get("video_id"),
                    "timestamp": hit.payload.get("timestamp"),
                    "text": hit.payload.get("text"),
                    "type": "audio",
                    "raw_score": hit.score,
                }
            )
    except Exception as e:
        logger.error(f"Audio Search Failed: {e}")

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
            video_results.append(
                {
                    "video_id": hit.payload.get("video_id"),
                    "timestamp": hit.payload.get("timestamp"),
                    "text": "[Visual Match]",
                    "type": "visual",
                    "raw_score": hit.score,
                }
            )
    except Exception as e:
        logger.error(f"Visual Search Failed: {e}")

    # 3. MERGE & CLEAN
    # Rank them fairly
    fused_results = reciprocal_rank_fusion(audio_results, video_results)

    # Remove clusters of results (e.g. 10s, 11s, 12s -> just 10s)
    final_results = deduplicate_results(fused_results)

    return final_results[:limit]
