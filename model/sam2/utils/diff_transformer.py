import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class MultiheadDiff1(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        depth: int = 0.5,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.internal_dim = embedding_dim // downsample_rate
        self.head_dim = embedding_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5
        self.n_rep = 1

        self.head_dim = embedding_dim // num_heads // 2

        self.q_proj = nn.Linear(embedding_dim, embedding_dim * 2)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim * 2)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.dropout_p = dropout

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        bsz, tgt_len, embed_dim = q.shape
        src_len = v.shape[1]
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads * 2)
        k = self._separate_heads(k, self.num_heads * 2)
        v = self._separate_heads(v, self.num_heads)

        q = q.view(bsz, self.num_heads, 2, q.shape[-2], q.shape[-1])
        k = k.view(bsz, self.num_heads, 2, k.shape[-2], k.shape[-1])

        q1, q2 = q[:, :, 0, :, :], q[:, :, 1, :, :]
        k1, k2 = k[:, :, 0, :, :], k[:, :, 1, :, :]

        attn_weights1 = F.softmax(torch.matmul(q1, k1.transpose(-1, -2)), dim=-1)
        attn_weights2 = F.softmax(torch.matmul(q2, k2.transpose(-1, -2)), dim=-1)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = attn_weights1 - lambda_full * attn_weights2

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn

        # q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        # k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        # v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        # k = repeat_kv(k.transpose(1, 2), 1)
        # v = repeat_kv(v.transpose(1, 2), 1)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        attn_mask = torch.triu(
            torch.zeros([tgt_len, src_len])
            .float()
            .fill_(float("-inf"))
            .type_as(attn_weights),
            1 + offset,
        )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, 1, -1)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine=True,
        memory_efficient=False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


if __name__ == "__main__":
    transformer = MultiheadDiff1(256, 8, downsample_rate=2)
    q = torch.rand(8, 1, 256)
    k = torch.rand(8, 256, 256)
    v = torch.rand(8, 256, 256)
    out = transformer(q, k, v)
    print(out.shape)
