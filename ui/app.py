import streamlit as st
import torch
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

# --- CONFIG ---
st.set_page_config(page_title="Semantic Cuts", page_icon="", layout="wide")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"


@st.cache_resource
def load_models():
    print("Loading CLIP & Qdrant...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Ready.")
    return model, processor, client


model, processor, client = load_models()

# --- HEADER ---
st.title("Semantic Cuts")
st.caption("Multimodal Video Search Engine")

# --- SEARCH BAR ---
col_search, col_btn = st.columns([4, 1])
with col_search:
    query = st.text_input(
        "Search",
        placeholder="Describe a moment (e.g. 'a red car', 'people laughing')...",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Search", type="primary", use_container_width=True)

if query or search_btn:
    if not query:
        st.warning("Please enter a search term.")
        st.stop()

    with st.spinner(f"Scanning library for '{query}'..."):
        # 1. Vectorize Text
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_vector = model.get_text_features(**inputs)[0]
            text_vector = (text_vector / text_vector.norm()).tolist()

        # 2. Search Database
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_vector,
                limit=8,  # Top 8 results
                with_payload=True,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # 3. Display Results
    if not results.points:
        st.warning("No matching moments found.")
    else:
        st.markdown(f"**Found {len(results.points)} relevant moments:**")

        # 4-Column Grid
        cols = st.columns(4)
        for i, hit in enumerate(results.points):
            payload = hit.payload or {}
            score = hit.score
            time_str = payload.get("second_formatted", "0:00")
            timestamp = int(payload.get("timestamp", 0))
            url = payload.get("url", "")

            with cols[i % 4]:
                st.video(url, start_time=timestamp)
                st.caption(f"**{time_str}** • Match: `{int(score * 100)}%`")
