import os
import sys
import time

# Add project root to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import properly
from app.database import index as database

# 4. Now you can imacport from the 'app' folder
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VidSearch Admin", layout="wide")
API_URL = "http://localhost:8000/video"

st.title("Admin Console")
st.markdown("### Content Ingestion Pipeline")

col1, col2 = st.columns([1, 2])

# --- LEFT: UPLOAD ---
with col1:
    st.container(border=True)
    st.subheader("1. Ingest Video")
    with st.form("ingest_form"):
        video_url = st.text_input(
            "Video URL (MP4)", placeholder="https://example.com/video.mp4"
        )
        user_id = st.text_input("Uploader ID", value="admin")
        submitted = st.form_submit_button("Dispatch Job", type="primary")

    if submitted and video_url:
        try:
            response = requests.post(
                API_URL, json={"url": video_url, "user_id": user_id}
            )
            if response.status_code == 200:
                st.toast("Job Queued Successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- RIGHT: LIBRARY ---
with col2:
    st.subheader("2. Video Library")
    # Initialize database if needed
    database.init_db()
    videos = database.get_all_videos()

    if not videos:
        st.info("No videos found. Upload one to start.")
    else:
        # Format for display
        data = []
        for v in videos:
            data.append(
                {
                    "Job ID": v["id"][:8],
                    "URL": v["url"],
                    "Status": v["status"],
                    "Ingested At": v["created_at"],
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Video Source"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )
