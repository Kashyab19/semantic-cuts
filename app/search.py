import torch
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

# --- CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_frames"

print("Loading CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("Model Loaded.")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def search_video(query_text):
    print(f"\nSearching for: '{query_text}'...")

    # 1. Vectorize Text
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # 2. Normalize
    text_vector = text_features[0]
    text_vector = text_vector / text_vector.norm(p=2, dim=-1, keepdim=True)
    text_vector = text_vector.cpu().numpy().tolist()

    # 3. Search Qdrant (FIXED)
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=text_vector,
            limit=3,
            with_payload=True,
        )
        hits = search_result.points
    except Exception as e:
        print(f"Search Error: {e}")
        return

    # 4. Display Results
    print(f"found {len(hits)} hits:")
    for i, hit in enumerate(hits):
        score = hit.score
        # Safety check: payload might be empty if something went wrong
        payload = hit.payload or {}
        timestamp = payload.get("second_formatted", "N/A")
        print(f" #{i + 1} | Score: {score:.3f} | Time: {timestamp}")


if __name__ == "__main__":
    while True:
        query = input("\nEnter search query (or 'q' to quit): ")
        if query.lower() == "q":
            break
        search_video(query)
