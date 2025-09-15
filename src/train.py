import os
import json
import argparse
from typing import Dict

import yaml
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from src.data import load_movielens, build_id_mappings, make_dataloader
from src.model import TwoTower
from src.index_faiss import build_faiss_ip_index, save_faiss_index

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for u, i in loader:
        u = u.to(device)
        i = i.to(device)
        optimizer.zero_grad()
        u_emb, i_emb = model(u, i)
        # In-batch negatives: scores = U * I^T
        logits = (u_emb @ i_emb.T)  # [B, B]
        targets = torch.arange(u.shape[0], device=device)
        loss_u = ce(logits, targets)
        loss_i = ce(logits.T, targets)  # symmetric
        loss = (loss_u + loss_i) * 0.5
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.shape[0]
    return total_loss / (len(loader.dataset) // loader.batch_size * loader.batch_size + 1e-8)

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    art_cfg = cfg["artifacts"]

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cpu"))

    # Load data
    ratings, movies = load_movielens(data_cfg["root_dir"])
    df_ui, user2idx, item2idx = build_id_mappings(ratings, rating_threshold=data_cfg.get("rating_threshold", 0.0))
    num_users = len(user2idx)
    num_items = len(item2idx)

    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(df_ui)}")

    loader = make_dataloader(df_ui, batch_size=train_cfg["batch_size"], shuffle=True)

    # Model
    model = TwoTower(num_users=num_users, num_items=num_items, embedding_dim=train_cfg["embedding_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # Train
    for epoch in range(1, train_cfg["epochs"] + 1):
        avg_loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}/{train_cfg['epochs']} - loss: {avg_loss:.4f}")

    # Prepare artifacts
    art_dir = art_cfg["dir"]
    os.makedirs(art_dir, exist_ok=True)

    # Save mappings
    user_map_path = os.path.join(art_dir, art_cfg["user_map"])
    item_map_path = os.path.join(art_dir, art_cfg["item_map"])
    movies_meta_path = os.path.join(art_dir, art_cfg["movies_meta"])
    with open(user_map_path, "w") as f:
        json.dump(user2idx, f)
    with open(item_map_path, "w") as f:
        json.dump(item2idx, f)
    movies.to_csv(movies_meta_path, index=False)

    # Save model
    model_path = os.path.join(art_dir, art_cfg["model_file"])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": train_cfg["embedding_dim"],
        },
        model_path,
    )

    # Build and save FAISS index
    item_emb = model.item_emb.weight.detach()
    index = build_faiss_ip_index(item_emb)
    faiss_path = os.path.join(art_dir, art_cfg["faiss_index"])
    save_faiss_index(index, faiss_path)

    # Save a copy of the config used
    with open(os.path.join(art_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Artifacts saved in: {art_dir}")
    print(f"- Model:        {model_path}")
    print(f"- FAISS index:  {faiss_path}")
    print(f"- User map:     {user_map_path}")
    print(f"- Item map:     {item_map_path}")
    print(f"- Movies meta:  {movies_meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/movielens-small.yaml")
    args = parser.parse_args()
    main(args.config)
