import logging
import os
import uuid

import ffmpeg
from fastembed import TextEmbedding
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models
try:
    # Use 'tiny' or 'base' for speed during debugging
    WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    logger.critical(f"Failed to load Audio models: {e}")


def transcribe_chunk(
    video_path: str, video_id: str, start_time: float, end_time: float
):
    duration = end_time - start_time
    temp_audio = f"/tmp/temp_{video_id}_{int(start_time)}.wav"
    points = []

    # 1. DEBUG: Check if file exists before asking FFmpeg to touch it
    if not os.path.exists(video_path):
        logger.error(f"AUDIO ERROR: File not found at {video_path}")
        return []

    try:
        # 2. Extract Audio
        (
            ffmpeg.input(video_path, ss=start_time, t=duration)
            .output(temp_audio, ac=1, ar=16000, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 3. Transcribe
        segments, _ = WHISPER.transcribe(temp_audio)
        valid_segments = [s for s in segments if len(s.text.strip()) > 5]

        # 4. Embed & Package
        if valid_segments:
            texts = [s.text.strip() for s in valid_segments]
            embeddings = list(EMBEDDER.embed(texts))

            for i, seg in enumerate(valid_segments):
                points.append(
                    {
                        "id": str(uuid.uuid4()),
                        "vector": embeddings[i].tolist(),
                        "payload": {
                            "video_id": video_id,
                            "text": seg.text.strip(),
                            "timestamp": start_time + seg.start,
                            "end_timestamp": start_time + seg.end,
                            "type": "audio_transcript",
                        },
                    }
                )

        return points

    except ffmpeg.Error as e:
        # CRITICAL: Print the actual FFmpeg error
        error_message = e.stderr.decode("utf8")
        logger.error(f"FFMPEG FAILED: {error_message}")
        return []

    except Exception as e:
        logger.error(f"General Audio Error: {e}")
        return []

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
