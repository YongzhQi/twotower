import os
import json
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.model import TwoTower
from src.index_faiss import load_faiss_index

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts/movielens_small_twotower")

class RecItem(BaseModel):
    movieId: int
    title: str
    score: float

class RecResponse(BaseModel):
    user_id: int
    items: List[RecItem]

def load_artifacts(art_dir: str):
    # Load mappings, model, FAISS, movies meta
    with open(os.path.join(art_dir, "user_id_map.json"), "r") as f:
        user2idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(art_dir, "item_id_map.json"), "r") as f:
        item2idx = {int(k): int(v) for k, v in json.load(f).items()}
    idx2item = {v: k for k, v in item2idx.items()}

    ckpt = torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu")
    model = TwoTower(ckpt["num_users"], ckpt["num_items"], ckpt["embedding_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    index = load_faiss_index(os.path.join(art_dir, "index.faiss"))
    movies = pd.read_csv(os.path.join(art_dir, "movies.csv"))

    return user2idx, idx2item, model, index, movies

user2idx, idx2item, model, index, movies = load_artifacts(ART_DIR)
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/recommend", response_model=RecResponse)
def recommend(user_id: int = Query(..., description="MovieLens userId"), k: int = Query(10, ge=1, le=100)):
    if user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found in training data")
    uid = user2idx[user_id]
    with torch.no_grad():
        uvec = model.user_emb.weight[uid]
        uvec = F.normalize(uvec, dim=-1).unsqueeze(0).numpy().astype("float32")
    scores, idxs = index.search(uvec, k)
    idxs = idxs[0]
    scores = scores[0]
    items = []
    for r, s in zip(idxs, scores):
        movie_id = idx2item[r]
        row = movies[movies["movieId"] == movie_id]
        title = row["title"].iloc[0] if not row.empty else str(movie_id)
        items.append({"movieId": int(movie_id), "title": title, "score": float(s)})
    return {"user_id": user_id, "items": items}
