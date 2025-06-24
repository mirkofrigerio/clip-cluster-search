import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
import faiss

from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# === Config ===
image_folder = "data/abo-images-small/images/small/" 
output_dir = "embeddings/"
batch_size = 64 # Number of images to process in each batch
use_faiss = True  # Set to False if you don't want to create a FAISS index
percent_of_dataset = 0.3  # Percentage of the total dataset we sample. Lower = Faster/Cheaper when testing.

os.makedirs(output_dir, exist_ok=True)

# === Load image paths ===
path = "C:/Users/mirko/Documents/Coding/transformers/data/abo-images-small/images/metadata/images.csv.gz"
df = pd.read_csv(path)
df = df.sample(frac=percent_of_dataset)

image_paths = [os.path.join(image_folder, f) for f in df['path']
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# === Load model & processor ===
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# === Batch embedding creation ===
all_filenames = []

for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i+batch_size]
    images = []
    batch_filenames = []
    for p in batch_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            batch_filenames.append(os.path.basename(p))
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    if not images:
        continue
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v for k, v in inputs.items()}

    # torch.no_grad because we are just interested in the embeddings created by a forward pass, we don't care about gradients and don't want to carry them around
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        # Normalize the outputs to unit length - this is important for cosine similarity
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        batch_emb = outputs.cpu().numpy()
        # Save batch embeddings
        np.save(os.path.join(output_dir, f"embeddings_batch_{i//batch_size:05d}.npy"), batch_emb)
        all_filenames.extend(batch_filenames)
 
# Save all filenames
pd.DataFrame({"filename": all_filenames}).to_csv(
    os.path.join(output_dir, "image_filenames.csv"), index_label="index"
)

# === Save as index for faiss search ===

# === FAISS indexing ===
if use_faiss:
    print("Creating FAISS index...")
    # Load all batch files and concatenate embeddings
    batch_files = sorted(glob.glob(os.path.join(output_dir, "embeddings_batch_*.npy")))
    all_embeddings = np.concatenate([np.load(f) for f in batch_files], axis=0)    
    
    index = faiss.IndexFlatIP(all_embeddings.shape[1])  # since we normalized the embeddings, we can use Inner Product (IP) for cosine similarity
    index.add(all_embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.index"))
    print(f"✅ FAISS index saved with {index.ntotal} vectors")