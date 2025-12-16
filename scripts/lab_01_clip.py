import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def run_clip_demo():
    model_id = "openai/clip-vit-base-patch32"

    # --- MAKE SURE THESE TWO LINES ARE HERE ---
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # This is the standard cats on a pink couch
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"Downloading image from the URL: {url}")
    image = Image.open(requests.get(url, stream=True).raw)

    texts = ["a photo of a dog", "a photo of a cat", "a photo of a truck"]
    print(f"Asking model if this is: {texts}")

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # No gradient means less memory usage - HOW?
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    print(f"Label probabilities: {probs}")

    print("\n Results:")

    for text, prob in zip(texts, probs[0]):
        print(f"{text}: {prob.item():.4f}")


if __name__ == "__main__":
    run_clip_demo()
