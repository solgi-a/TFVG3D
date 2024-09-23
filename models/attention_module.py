import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#-------------------------------------------------------------------------------------------------------------------------------

class MultiheadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1, ret_att=False):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        self.ret_att = ret_att

    def attention(self, queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        headdim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(headdim)  # B x H x S x L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values, scores  # B x H x S x headdim

    def forward(self, query, key, value, mask=None, sum_seq=False):
        """
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x S x L, 1 for heads
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]  # B x H x S|L x head_dim
        result, att = self.attention(queries, keys, values, mask, self.dropout)  # B x H x S x headdim
        if sum_seq:
            result = result.sum(2, keepdim=True)  # B x H x 1 x headdim
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        # B x S x D / (if sum_seq)B x 1 x D
        if self.ret_att:
            return self.linears[-1](result), att.mean(dim=1)
        else:
            return self.linears[-1](result)

#-------------------------------------------------------------------------------------------------------------------------------

class AttentionModule(nn.Module):
    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout, ret_att=False):
        super().__init__()
        self.dim = dim
        self.msa = MultiheadAttn(dim, n_head, dropout=msa_dropout, ret_att=ret_att)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )
        self.ret_att = ret_att

    def forward(self, q, k, v, mask, add_q=True):
        if self.ret_att:
            msa, att = self.msa(q, k, v, mask)
        else:
            msa = self.msa(q, k, v, mask)
            att = None
        if add_q:
            x = self.norm1(q + msa)
        else:
            x = self.norm1(v + msa)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x, att
