import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, u_ids: torch.Tensor, i_ids: torch.Tensor):
        u = self.user_emb(u_ids)
        i = self.item_emb(i_ids)
        return u, i

    @torch.no_grad()
    def user_vector(self, u_id: int):
        vec = self.user_emb.weight[u_id]
        return F.normalize(vec, dim=-1)
