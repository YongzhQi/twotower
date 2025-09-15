import os
import json
import argparse
from typing import Dict, List, Tuple, Set

import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from src.data import ensure_movielens_small
from src.model import TwoTower
from src.index_faiss import build_faiss_ip_index, save_faiss_index

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_movielens_frames(root_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_dir = ensure_movielens_small(root_dir)
    ratings = pd.read_csv(os.path.join(ds_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(ds_dir, "movies.csv"))
    return ratings, movies

def split_train_val_by_time(
    ratings: pd.DataFrame,
    rating_threshold: float = 0.0,
    holdout_per_user: int = 1,
    min_interactions: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train_df and val_df with columns [userId, movieId, timestamp]."""
    df = ratings.copy()
    if rating_threshold > 0:
        df = df[df["rating"] >= rating_threshold].copy()

    # Ensure timestamp exists
    if "timestamp" not in df.columns:
        raise ValueError("ratings.csv missing 'timestamp' column required for time-based split.")

    df = df[["userId", "movieId", "timestamp"]].drop_duplicates()

    train_parts = []
    val_parts = []

    for uid, g in df.groupby("userId"):
        g = g.sort_values("timestamp")
        if len(g) < max(min_interactions, holdout_per_user + 1):
            # skip users without enough interactions
            continue
        val = g.tail(holdout_per_user)
        train = g.iloc[:-holdout_per_user]
        train_parts.append(train)
        val_parts.append(val)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    return train_df, val_df

def build_id_mappings_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build user/item contiguous ids from union of train and val users/items."""
    users = pd.concat([train_df["userId"], val_df["userId"]]).drop_duplicates().astype(int).tolist()
    items = pd.concat([train_df["movieId"], val_df["movieId"]]).drop_duplicates().astype(int).tolist()
    user2idx = {int(u): i for i, u in enumerate(sorted(users))}
    item2idx = {int(iid): i for i, iid in enumerate(sorted(items))}
    return user2idx, item2idx

def df_to_uid_iid(df: pd.DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["uid"] = out["userId"].map(user2idx)
    out["iid"] = out["movieId"].map(item2idx)
    out = out.dropna(subset=["uid", "iid"]).astype({"uid": int, "iid": int})
    return out[["uid", "iid", "userId", "movieId", "timestamp"]]

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, uids: torch.Tensor, iids: torch.Tensor):
        self.uids = uids.long()
        self.iids = iids.long()
    def __len__(self):
        return self.uids.shape[0]
    def __getitem__(self, idx):
        return self.uids[idx], self.iids[idx]

def make_dataloader(df_ui: pd.DataFrame, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    u = torch.from_numpy(df_ui["uid"].to_numpy())
    i = torch.from_numpy(df_ui["iid"].to_numpy())
    ds = InteractionDataset(u, i)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    for u, i in loader:
        u = u.to(device)
        i = i.to(device)
        optimizer.zero_grad()
        u_emb, i_emb = model(u, i)
        logits = (u_emb @ i_emb.T)  # [B, B]
        targets = torch.arange(u.shape[0], device=device)
        loss_u = ce(logits, targets)
        loss_i = ce(logits.T, targets)
        loss = 0.5 * (loss_u + loss_i)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.shape[0]
        total_examples += u.shape[0]
    return total_loss / max(total_examples, 1)

def build_train_seen_map(train_df_ui: pd.DataFrame) -> Dict[int, Set[int]]:
    """Map uid -> set(iid) seen in train."""
    seen: Dict[int, Set[int]] = {}
    for uid, group in train_df_ui.groupby("uid"):
        seen[int(uid)] = set(int(i) for i in group["iid"].tolist())
    return seen

def topk_unseen_for_user(
    uid: int,
    model: TwoTower,
    index,
    train_seen: Dict[int, Set[int]],
    k: int,
    extra: int = 100,
) -> List[int]:
    """Return top-K item iids not seen in train."""
    with torch.no_grad():
        uvec = model.user_emb.weight[uid]
        uvec = F.normalize(uvec, dim=-1).unsqueeze(0).cpu().numpy().astype("float32")
    # Ask for more to allow filtering seen items
    ask = k + len(train_seen.get(uid, set())) + max(extra, 0)
    scores, idxs = index.search(uvec, ask)
    preds = []
    seen_set = train_seen.get(uid, set())
    for iid in idxs[0]:
        if int(iid) in seen_set:
            continue
        preds.append(int(iid))
        if len(preds) >= k:
            break
    return preds

def eval_metrics(
    model: TwoTower,
    index,
    val_df_ui: pd.DataFrame,
    train_df_ui: pd.DataFrame,
    k: int,
) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K, HitRate@K across users with val data."""
    train_seen = build_train_seen_map(train_df_ui)
    # Build user -> list of validation item iids
    val_by_user: Dict[int, List[int]] = {}
    for uid, g in val_df_ui.groupby("uid"):
        val_by_user[int(uid)] = [int(i) for i in g["iid"].tolist()]

    users = list(val_by_user.keys())
    if not users:
        return {"users_evaluated": 0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0}

    recalls = []
    ndcgs = []
    hits = []

    for uid in tqdm(users, desc=f"Evaluating @K={k}", leave=False):
        positives = set(val_by_user[uid])
        if not positives:
            continue
        preds = topk_unseen_for_user(uid, model, index, train_seen, k)

        # Recall@K
        hit_count = len(positives.intersection(preds))
        recall = hit_count / len(positives)
        recalls.append(recall)

        # HitRate@K (at least one positive in top-K)
        hits.append(1.0 if hit_count > 0 else 0.0)

        # NDCG@K (binary relevance)
        dcg = 0.0
        for rank, iid in enumerate(preds, start=1):
            if iid in positives:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_hits = min(len(positives), k)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1)) if ideal_hits > 0 else 1.0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    metrics = {
        "users_evaluated": int(len(recalls)),
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
    }
    return metrics

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["eval"]
    art_cfg = cfg["artifacts"]

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cpu"))

    # Load and split
    ratings, _ = load_movielens_frames(data_cfg["root_dir"])
    train_df, val_df = split_train_val_by_time(
        ratings,
        rating_threshold=data_cfg.get("rating_threshold", 0.0),
        holdout_per_user=split_cfg.get("holdout_per_user", 1),
        min_interactions=split_cfg.get("min_interactions", 2),
    )

    print(f"Users with train/val: {train_df['userId'].nunique()} "
          f"(train rows={len(train_df)}, val rows={len(val_df)})")

    # Build mappings from union
    user2idx, item2idx = build_id_mappings_from_splits(train_df, val_df)
    num_users = len(user2idx)
    num_items = len(item2idx)

    # Map to uids/iids
    train_ui = df_to_uid_iid(train_df, user2idx, item2idx)
    val_ui = df_to_uid_iid(val_df, user2idx, item2idx)

    # DataLoader
    loader = make_dataloader(train_ui, batch_size=train_cfg["batch_size"], shuffle=True)

    # Model + Optim
    model = TwoTower(num_users=num_users, num_items=num_items, embedding_dim=train_cfg["embedding_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # Train
    for epoch in range(1, train_cfg["epochs"] + 1):
        avg_loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}/{train_cfg['epochs']} - loss: {avg_loss:.4f}")

    # Build FAISS index
    item_emb = model.item_emb.weight.detach()
    index = build_faiss_ip_index(item_emb)

    # Evaluate
    k = int(eval_cfg.get("k", 10))
    metrics = eval_metrics(model, index, val_ui, train_ui, k=k)
    print(f"Evaluation @K={k} — users={metrics['users_evaluated']}, "
          f"Recall@K={metrics['recall_at_k']:.4f}, "
          f"NDCG@K={metrics['ndcg_at_k']:.4f}, "
          f"HitRate@K={metrics['hit_rate_at_k']:.4f}")

    # Save artifacts + metrics
    art_dir = art_cfg["dir"]
    os.makedirs(art_dir, exist_ok=True)

    with open(os.path.join(art_dir, art_cfg["user_map"]), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(art_dir, art_cfg["item_map"]), "w") as f:
        json.dump(item2idx, f)

    # Save model and index to allow serving if you want
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": train_cfg["embedding_dim"],
        },
        os.path.join(art_dir, art_cfg["model_file"]),
    )
    save_faiss_index(index, os.path.join(art_dir, art_cfg["faiss_index"]))

    # Save metrics
    metrics_path = os.path.join(art_dir, art_cfg.get("metrics_file", "metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump({"k": k, **metrics}, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/movielens-small-eval.yaml")
    args = parser.parse_args()
    main(args.config)
