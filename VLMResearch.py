import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("mps")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cutecat.jpg")
labels = ["a dog", "a cat"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to("mps") for k, v in inputs.items()}

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
print(dict(zip(labels, probs[0].tolist())))