import os
import sys
import time

# 1. Get the path to the current file's directory (ui/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the 'app' directory (sibling to ui/)
#    Logic: Go up one level (..), then down into 'app'
app_dir = os.path.join(os.path.dirname(current_dir), "app")

# 3. Add the 'app' directory explicitly to Python's search path
sys.path.append(app_dir)

# 4. Now import 'database' directly (since we are technically "inside" the app folder now)
import database

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
