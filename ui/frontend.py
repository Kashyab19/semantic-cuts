import time

import requests
import streamlit as st
import torch
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

# --- CONFIGURATION ---
st.set_page_config(page_title="Semantic Cuts", page_icon="🎬", layout="wide")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"
API_URL = "http://localhost:8000/video"


# --- CACHED RESOURCES ---
@st.cache_resource
def load_models():
    # Only load these once to save memory/time
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return model, processor, client


model, processor, client = load_models()

# --- SIDEBAR: ADD VIDEO ---
with st.sidebar:
    st.header("Add Video")
    video_url = st.text_input(
        "Video URL (MP4)", placeholder="https://example.com/video.mp4"
    )

    if st.button("Index Video"):
        if not video_url:
            st.error("Please enter a URL")
        else:
            try:
                # Call our FastAPI backend
                response = requests.post(
                    API_URL, json={"url": video_url, "user_id": "streamlit_user"}
                )

                if response.status_code == 200:
                    st.success(" Job Queued!")
                    st.info(f"Job ID: {response.json().get('job_id')}")
                    st.markdown("Check your **Worker Terminal** to see progress.")
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"API Error: {e}")
                st.warning("Is 'uvicorn app.main:app' running?")

    st.divider()
    st.markdown("### System Status")
    try:
        # Simple health check
        info = client.get_collection(COLLECTION_NAME)
        st.success(f" Database Connected")
        st.caption(f"Indexed Vectors: {info.points_count}")
    except:
        st.error("Qdrant Offline")

# --- MAIN PAGE: SEARCH ---
st.title(" Semantic Cuts")
st.markdown("### Search your video library with AI")

query = st.text_input(
    "Describe a moment:",
    placeholder="e.g. a red car, people laughing, a explosion",
    help="Hit Enter to search",
)

if query:
    with st.spinner(f"Searching for '{query}'..."):
        # 1. Vectorize
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_vector = model.get_text_features(**inputs)[0]
            text_vector = (text_vector / text_vector.norm()).tolist()

        # 2. Search
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_vector,
                limit=4,
                with_payload=True,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # 3. Render
    if not results.points:
        st.warning("No matches found.")
    else:
        st.subheader("Top Matches")
        cols = st.columns(2)

        for i, hit in enumerate(results.points):
            score = hit.score
            payload = hit.payload or {}
            timestamp = payload.get("second_formatted", "N/A")
            start_time = int(payload.get("timestamp", 0))
            video_url = payload.get("url", "")

            with cols[i % 2]:
                st.video(video_url, start_time=start_time)
                st.caption(f"**#{i + 1}** | Score: `{score:.2f}` | Time: `{timestamp}`")
                st.divider()
