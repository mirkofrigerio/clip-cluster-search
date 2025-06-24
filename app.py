import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import os

# === Config ===
IMAGE_ROOT = r'data\abo-images-small\images\small'
INDEX_PATH = "embeddings/faiss_index.index"
FILENAMES_CSV = "embeddings/image_filenames.csv"
MODEL_NAME = "openai/clip-vit-large-patch14"
TOP_K = 20

# === Load everything once ===
@st.cache_resource
def load_model_and_index():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH) 
    filenames = pd.read_csv(FILENAMES_CSV)["filename"].tolist()
    return model, processor, index, filenames

model, processor, index, filenames = load_model_and_index()

# === Streamlit UI ===
st.title("🖼️ Semantic Image Search with CLIP + FAISS")
query = st.text_input("Enter a description", "a pink cotton summer dress")

if query:
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    D, I = index.search(text_features.cpu().numpy(), TOP_K)

    st.subheader(f"Top {TOP_K} Matches")
    images_per_row = 4  # Number of images per row

    for i in range(0, TOP_K, images_per_row):
        cols = st.columns(images_per_row)
        for j, idx in enumerate(I[0][i:i+images_per_row]):
            path = os.path.join(IMAGE_ROOT, filenames[idx][:2], filenames[idx])
            try:
                img = Image.open(path)
                cols[j].image(img, caption=filenames[idx].split("/")[-1], use_container_width=True)
            except Exception as e:
                cols[j].write(f"⚠️ Couldn't load image: {filenames[idx]}")

