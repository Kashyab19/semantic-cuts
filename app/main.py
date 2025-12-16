import json
import uuid

from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel

app = FastAPI()

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9094",
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

        print(f"Job {job_id} queued")

        return {
            "message": "Job queued successfully",
            "job_id": job_id,
            "status": "started_processing",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[Kafka Error]{str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
