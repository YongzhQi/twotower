import os
import numpy as np
import faiss
import torch
import torch.nn.functional as F

def build_faiss_ip_index(item_embeddings: torch.Tensor) -> faiss.Index:
    # Normalize for cosine similarity via inner product
    emb = F.normalize(item_embeddings, dim=-1).cpu().numpy().astype("float32")
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index

def save_faiss_index(index: faiss.Index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)
