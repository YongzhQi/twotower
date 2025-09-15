import os
import io
import json
import zipfile
import urllib.request
from typing import Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def ensure_movielens_small(root_dir: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    dest_zip = os.path.join(root_dir, "ml-latest-small.zip")
    dest_dir = os.path.join(root_dir, "ml-latest-small")
    if os.path.isdir(dest_dir) and os.path.isfile(os.path.join(dest_dir, "ratings.csv")):
        return dest_dir
    if not os.path.exists(dest_zip):
        print(f"Downloading MovieLens small to {dest_zip} ...")
        urllib.request.urlretrieve(ML_SMALL_URL, dest_zip)
    print("Extracting zip...")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(root_dir)
    return dest_dir

def load_movielens(root_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_dir = ensure_movielens_small(root_dir)
    ratings = pd.read_csv(os.path.join(ds_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(ds_dir, "movies.csv"))
    return ratings, movies

def build_id_mappings(ratings: pd.DataFrame, rating_threshold: float = 0.0):
    df = ratings.copy()
    if rating_threshold > 0:
        df = df[df["rating"] >= rating_threshold].copy()
    # Keep only userId, movieId
    df = df[["userId", "movieId"]].drop_duplicates().reset_index(drop=True)

    # Map to contiguous ids
    unique_users = df["userId"].unique()
    unique_items = df["movieId"].unique()
    user2idx = {int(u): i for i, u in enumerate(sorted(unique_users))}
    item2idx = {int(iid): i for i, iid in enumerate(sorted(unique_items))}

    df["uid"] = df["userId"].map(user2idx)
    df["iid"] = df["movieId"].map(item2idx)

    return df[["uid", "iid"]], user2idx, item2idx

class InteractionDataset(Dataset):
    def __init__(self, uids: torch.Tensor, iids: torch.Tensor):
        self.uids = uids.long()
        self.iids = iids.long()

    def __len__(self):
        return self.uids.shape[0]

    def __getitem__(self, idx):
        return self.uids[idx], self.iids[idx]

def make_dataloader(df_ui: pd.DataFrame, batch_size: int, shuffle: bool = True) -> DataLoader:
    u = torch.from_numpy(df_ui["uid"].to_numpy())
    i = torch.from_numpy(df_ui["iid"].to_numpy())
    ds = InteractionDataset(u, i)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
