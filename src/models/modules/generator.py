import math

import torch
import torch.nn as nn

class MoEGeneratorBlock(nn.Module):
    def __init__(self, hidden_size, vlm_hidden_size, num_heads, mlp_ratio=4.0, causal=True):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x, vlm_feat):
        T_img = x.shape[1]
        T_vlm = vlm_feat.shape[1]

        full_mask = None
        if self.causal:
            full_mask = torch.zeros((T_img, T_img + T_vlm), device=x.device, dtype=x.dtype)
            causal_mask = torch.triu(torch.ones((T_img, T_img), device=x.device, dtype=torch.bool), diagonal=1)
            full_mask[:, :T_img].masked_fill_(causal_mask, float('-inf'))
        
        x_norm = self.norm1(x)
        v_feat = self.vlm_proj(vlm_feat)
        
        kv = torch.cat([x_norm, v_feat], dim=1)
        attn_out, _ = self.attn(query=x_norm, key=kv, value=kv, attn_mask=full_mask)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(dtype=self.mlp[0].weight.dtype))


class EmuTokenClassificationGenerator(nn.Module):
    """
    Autoregressive Transformer for Image Generation using MoE-like Layer-wise Cross Attention
    """
    def __init__(self, vocab_size, vlm_hidden_size, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        self.blocks = nn.ModuleList([
            MoEGeneratorBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, causal=True) for _ in range(depth)
        ])
        
        self.norm_final = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids, vlm_hidden_states):
        x = self.token_emb(input_ids)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        
        hidden_states = []
        for block, vlm_state in zip(self.blocks, relevant_vlm_states):
            vlm_state = vlm_state.to(dtype=x.dtype)
            x = block(x, vlm_state)
            hidden_states.append(x)
            
        x = self.norm_final(x)
        logits = self.head(x)
        
        return logits, hidden_states


class DINOFeatureFlowGenerator(nn.Module):
    """
    Flow-matching Transformer over DINO patch features with layer-wise VLM cross attention.
    """
    def __init__(
        self,
        feature_dim,
        vlm_hidden_size,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        max_seq_len=1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        self.blocks = nn.ModuleList([
            MoEGeneratorBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, causal=False)
            for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, feature_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, noisy_features, timesteps, vlm_hidden_states):
        x = self.input_proj(noisy_features.to(dtype=self.input_proj.weight.dtype))
        if x.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(
                f"DINO feature token length {x.shape[1]} exceeds generator max_seq_len "
                f"{self.pos_embed.shape[1]}."
            )
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = x + self.t_embedder(timesteps).unsqueeze(1)

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]

        hidden_states = []
        for block, vlm_state in zip(self.blocks, relevant_vlm_states):
            vlm_state = vlm_state.to(dtype=x.dtype)
            x = block(x, vlm_state)
            hidden_states.append(x)

        pred_velocity = self.head(self.norm_final(x))
        return pred_velocity, hidden_states
