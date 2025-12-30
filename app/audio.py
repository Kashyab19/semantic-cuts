import logging
import os
import uuid

import ffmpeg
import numpy as np
from fastembed import TextEmbedding
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LOAD MODELS ---
try:
    WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    logger.critical(f"Failed to load Audio models: {e}")


def transcribe_chunk(
    video_path: str, video_id: str, start_time: float, end_time: float
):
    """
    1. Pipes audio from FFmpeg directly to memory (No Disk I/O).
    2. Uses VAD to skip silence (Saves CPU).
    3. Merges short segments into semantic blocks (Increases Accuracy).
    """
    duration = end_time - start_time

    # 1. GHOST CHECK
    if not os.path.exists(video_path):
        logger.error(f"AUDIO ERROR: File not found at {video_path}")
        return []

    try:
        # 2. EXTRACT AUDIO TO MEMORY (Zero-Copy)
        # We request raw PCM data (s16le) at 16khz mono
        out, _ = (
            ffmpeg.input(video_path, ss=start_time, t=duration)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Convert raw bytes to float32 numpy array (Required by Whisper)
        # normalized between -1 and 1
        audio_array = (
            np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        )

        # 3. TRANSCRIBE WITH VAD
        # vad_filter=True stops the model from hallucinating on silence
        segments, _ = WHISPER.transcribe(audio_array, vad_filter=True)

        # 4. SEMANTIC MERGING
        # Group segments until they form a coherent thought (~200 chars)
        points = []
        buffer_text = []
        buffer_start = None
        buffer_end = 0.0

        # Generator to list so we can iterate
        segments = list(segments)

        for i, seg in enumerate(segments):
            if not buffer_text:
                buffer_start = seg.start

            buffer_text.append(seg.text.strip())
            buffer_end = seg.end

            # Current semantic block
            full_text = " ".join(buffer_text)

            # Heuristic: If block is long enough OR it's the last segment
            if len(full_text) >= 256 or i == len(segments) - 1:
                # Embed the GROUPS, not the fragments
                vector_gen = EMBEDDER.embed([full_text])
                vector = list(vector_gen)[0].tolist()  # Unpack generator

                points.append(
                    {
                        "id": str(uuid.uuid4()),
                        "vector": vector,
                        "payload": {
                            "video_id": video_id,
                            "text": full_text,
                            # Map relative whisper time back to absolute video time
                            "timestamp": start_time + buffer_start,
                            "end_timestamp": start_time + buffer_end,
                            "type": "audio_transcript",
                            "strategy": "semantic_merge_256",
                        },
                    }
                )

                # Reset buffer
                buffer_text = []
                buffer_start = None

        return points

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
        logger.error(f"FFMPEG MEMORY FAIL: {error_msg}")
        return []

    except Exception as e:
        logger.error(f"General Audio Error: {e}")
        return []
