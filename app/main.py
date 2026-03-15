import asyncio
import json
import os
import uuid
import logging

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from kafka import KafkaProducer
from pydantic import BaseModel

from app.database import get_all_videos, get_video

logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9094")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


class VideoRequest(BaseModel):
    url: str  # could be a YT video
    user_id: str = uuid.uuid4().hex


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/video")
async def video(payload: VideoRequest):
    job_id = str(uuid.uuid4().hex)

    task_payload = {
        "job_id": job_id,
        "user_id": payload.user_id,
        "url": payload.url,
        "status": "queued",
    }

    try:
        producer.send("video_ingestion", task_payload)
        producer.flush()  # WHY?

        logger.info(f"Job {job_id} queued")

        return {
            "message": "Job queued successfully",
            "job_id": job_id,
            "status": "started_processing",
        }

    except Exception as e:
        logger.error(f"Kafka error for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to queue video for processing")


@app.get("/video/{job_id}/progress")
async def video_progress(job_id: str):
    video = get_video(job_id)
    if not video:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        while True:
            video = get_video(job_id)
            status = video["status"] if video else "unknown"

            total = redis_client.get(f"job:{job_id}:total")
            pending = redis_client.get(f"job:{job_id}:pending")

            if total is not None and pending is not None:
                total = int(total)
                pending = int(pending)
                progress = round(((total - pending) / total) * 100, 1) if total > 0 else 0
            else:
                total = 0
                pending = 0
                progress = 100.0 if status == "completed" else 0.0

            event = json.dumps({
                "status": status,
                "pending": pending,
                "total": total,
                "progress": progress,
            })
            yield f"data: {event}\n\n"

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/videos")
async def list_videos():
    rows = get_all_videos()
    videos = [
        {
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "status": row["status"],
            "duration": row["duration"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"videos": videos}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
