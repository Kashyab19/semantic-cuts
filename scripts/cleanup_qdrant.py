from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

collections = ["video_frames", "audio_transcripts"]

print("WARNING: Wiping Database locally..")

for name in collections:
    if client.collection_exists(name):
        client.delete_collection(name)
        print(f"Deleted collection: {name}")
    else:
        print(f"Collection not found: {name}")

print("\nDatabase is clean. Please restart the Server/Worker to recreate them.")
