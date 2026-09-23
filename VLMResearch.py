import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("mps")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

paths = ["dog.jpg", "cutecat.jpg"]
images = [Image.open(p).convert("RGB") for p in paths]
labels = ["a dog", "a cat", "a bird", "a car"]

inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
inputs = {k: v.to("mps") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)  # shape: (3 images, 4 labels)

# --- New: find the highest score for each image and say what it was ---
all_rows = probs.tolist()

for i in range(len(paths)):
    path = paths[i]
    row = all_rows[i]

    # Build the label -> score dictionary for this image
    scores = {}
    for j in range(len(labels)):
        scores[labels[j]] = row[j]

    # Search the dictionary for the largest value
    best_label = ""
    best_score = 0.0
    for label in scores:
        score = scores[label]
        if score > best_score:
            best_score = score
            best_label = label

    print(path + " was " + best_label)