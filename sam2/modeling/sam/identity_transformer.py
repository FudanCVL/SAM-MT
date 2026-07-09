# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Tuple, Type, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from sam2.modeling.sam2_utils import MLP

class IdentityTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        
        # Identity transformer
        for i in range(depth):
            self.layers.append(
                IdentityAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
    
    def create_target_attention_mask(self, queries: Tensor, keys: Tensor):
        num_objects = queries.shape[1]
        num_frames = int(keys.shape[1] / (queries.shape[1])) 
        per_frame_tokens = num_objects
        assert queries.shape[0] == keys.shape[0]
        attn_mask = torch.zeros((queries.shape[0], queries.shape[1], keys.shape[1]), dtype=torch.bool, device=queries.device)
        for obj_idx in range(num_objects):
            for frame_idx in range(num_frames):
                start_idx = per_frame_tokens * frame_idx + obj_idx
                end_idx = per_frame_tokens * frame_idx + obj_idx + 1
                attn_mask[:, obj_idx:obj_idx+1, start_idx:end_idx] = True
        return attn_mask

    def forward(
        self,
        query,
        pixel,
        query_emb,
        pixel_pe,
    ) -> Tuple[Tensor, Tensor]:
        
        attn_mask = self.create_target_attention_mask(query, pixel)
        
        for idx, layer in enumerate(self.layers):
            query = layer(
                queries=query,
                keys=pixel, 
                query_pe=query_emb, 
                key_pe=pixel_pe, 
                original_keys=pixel,
                idx = idx,
                attn_mask=attn_mask,
            )
        return query


class IdentityAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__() 
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=4, kv_in_dim=256)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe
        self.embedding_dim = embedding_dim
        
    
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, original_keys: Tensor, idx: int, attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        q = queries
        k = keys + key_pe
        
        # Cross attention block
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, attn_mask=attn_mask)
        queries = queries + attn_out
        queries = self.norm1(queries)
        
        # Self attention block
        q = queries
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        return queries


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, attn_mask=attn_mask)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out