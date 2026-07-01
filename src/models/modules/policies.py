import math

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# ----------------------------- Shared Components -----------------------------
# -----------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def build_causal_action_mask(history_len, action_len, device, kv_extra_len=0):
    total_len = history_len + action_len
    mask = torch.zeros(total_len, total_len + kv_extra_len, device=device, dtype=torch.bool)

    if history_len > 0:
        mask[:history_len, history_len:history_len + action_len] = True

    future_mask = torch.ones(action_len, action_len, device=device, dtype=torch.bool).triu(1)
    mask[history_len:, history_len:history_len + action_len] = future_mask
    return mask

class MetaQueryBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MoEBlock(nn.Module):
    def __init__(self, hidden_size, vlm_hidden_size, num_heads, mlp_ratio=4.0, gen_hidden_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.gen_proj = nn.Linear(gen_hidden_size, hidden_size) if gen_hidden_size is not None else None

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, vlm_feat, gen_feat=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        v_feat = self.vlm_proj(vlm_feat)
        kv_list = [x_norm, v_feat]
        if gen_feat is not None and self.gen_proj is not None:
            kv_list.append(self.gen_proj(gen_feat))
        kv = torch.cat(kv_list, dim=1)

        attn_out, _ = self.attn(query=x_norm, key=kv, value=kv, attn_mask=attn_mask)
        
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -----------------------------------------------------------------------------
# ----------------------------- Diffusion Policies ----------------------------
# -----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class ActionDiffusionTransformerMetaquery(nn.Module):
    def __init__(self, action_dim, condition_dim, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj = nn.Linear(condition_dim, hidden_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))
        
        self.blocks = nn.ModuleList([
            MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_action, timestep, condition, history_actions=None):
        noisy_action = noisy_action.to(dtype=self.input_proj.weight.dtype)
        condition = condition.to(dtype=self.cond_proj.weight.dtype)
        
        if history_actions is not None:
            history_actions = history_actions.to(dtype=self.input_proj.weight.dtype)
            x_input = torch.cat([history_actions, noisy_action], dim=1)
        else:
            x_input = noisy_action

        x = self.input_proj(x_input) 
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        t = self.t_embedder(timestep) 
        c = self.cond_proj(condition) + t 
        
        for block in self.blocks:
            x = block(x, c)
            
        output = self.final_layer(x, c)
        
        if history_actions is not None:
            output = output[:, -noisy_action.shape[1]:, :]
            
        return output


class ActionDiffusionTransformerMoE(nn.Module):
    def __init__(self, action_dim, vlm_hidden_size, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0, gen_hidden_size=None):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_action, timestep, vlm_hidden_states, history_actions=None, gen_hidden_states=None):
        noisy_action = noisy_action.to(dtype=self.input_proj.weight.dtype)

        if history_actions is not None:
            history_actions = history_actions.to(dtype=self.input_proj.weight.dtype)
            x_input = torch.cat([history_actions, noisy_action], dim=1)
        else:
            x_input = noisy_action

        x = self.input_proj(x_input)
        x = x + self.pos_embed[:, :x.shape[1], :]
        t = self.t_embedder(timestep)

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks):]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states):
            vlm_state = vlm_state.to(dtype=x.dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=x.dtype)
            x = block(x, t, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, t)
        
        if history_actions is not None:
            output = output[:, -noisy_action.shape[1]:, :]
            
        return output

# -----------------------------------------------------------------------------
# ----------------------------- Regression Policies ---------------------------
# -----------------------------------------------------------------------------
class ActionRegressionTransformerMetaquery(nn.Module):
    def __init__(self, action_dim, condition_dim, num_actions=1, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(action_dim, hidden_size)
        
        self.query_embed = nn.Parameter(torch.zeros(1, num_actions, hidden_size))
        
        self.cond_proj = nn.Linear(condition_dim, hidden_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))
        
        self.blocks = nn.ModuleList([
            MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None):
        B = condition.shape[0]
        dtype = self.input_proj.weight.dtype
        condition = condition.to(dtype=dtype)
        
        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)
        
        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries
            
        x = x + self.pos_embed[:, :x.shape[1], :]
        c = self.cond_proj(condition)
        
        for block in self.blocks:
            x = block(x, c)
            
        output = self.final_layer(x, c)
        output = output[:, -self.num_actions:, :]
        
        return output

class ActionRegressionTransformerMoE(nn.Module):
    def __init__(self, action_dim, vlm_hidden_size, num_actions=1, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0, gen_hidden_size=None):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, num_actions, hidden_size))

        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None, gen_hidden_states=None):
        vlm_hidden_states = condition

        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        final_state = final_state.to(dtype=dtype)

        c_emb = final_state.mean(dim=1)
        c = self.cond_proj(c_emb)

        B = c.shape[0]

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, :x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks):]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states):
            vlm_state = vlm_state.to(dtype=dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=dtype)
            x = block(x, c, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, c)
        output = output[:, -self.num_actions:, :]

        return output

# -----------------------------------------------------------------------------
# --------------------------- Classification Policies -------------------------
# -----------------------------------------------------------------------------
class ActionClassificationTransformerMetaquery(nn.Module):
    def __init__(self, action_dim, condition_dim, num_actions=1, num_bins=256,
                 hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 fast_mode=False, fast_expected_seq_len=64, fast_vocab_size=2048):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.fast_mode = fast_mode
        self.fast_vocab_size = fast_vocab_size
        self.pose_dim = action_dim - 1

        if fast_mode:
            self.total_queries = fast_expected_seq_len
            self.per_dim_classes = fast_vocab_size
        else:
            self.dim_per_action = action_dim
            self.total_queries = num_actions * self.dim_per_action
            self.per_dim_classes = num_bins
        
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, self.total_queries, hidden_size))
        self.cond_proj = nn.Linear(condition_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))
        
        self.blocks = nn.ModuleList([
            MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None):
        B = condition.shape[0]
        dtype = self.input_proj.weight.dtype
        condition = condition.to(dtype=dtype)
        
        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)
        
        if history_actions is not None:
            hist_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([hist_emb, queries], dim=1)
        else:
            x = queries
        
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        c = self.cond_proj(condition)
        
        for block in self.blocks:
            x = block(x, c)
            
        output = self.final_layer(x, c)
        output = output[:, -self.total_queries:, :]  
        
        if self.fast_mode:
            return output  # (B, fast_expected_seq_len, fast_vocab_size)
        else:
            return output.view(B, self.num_actions, self.action_dim, self.per_dim_classes)

class ActionClassificationTransformerMoE(nn.Module):
    def __init__(self, action_dim, vlm_hidden_size, num_actions=1, num_bins=256,
                 hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 fast_mode=False, fast_expected_seq_len=64, fast_vocab_size=2048,
                 gen_hidden_size=None):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.fast_mode = fast_mode
        self.fast_vocab_size = fast_vocab_size

        if fast_mode:
            self.total_queries = fast_expected_seq_len
            self.per_dim_classes = fast_vocab_size
        else:
            self.dim_per_action = action_dim
            self.total_queries = num_actions * self.dim_per_action
            self.per_dim_classes = num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, self.total_queries, hidden_size))

        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None, gen_hidden_states=None):
        vlm_hidden_states = condition

        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        final_state = final_state.to(dtype=dtype)

        c_emb = final_state.mean(dim=1)
        c = self.cond_proj(c_emb)

        B = c.shape[0]

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, :x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks):]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states):
            vlm_state = vlm_state.to(dtype=dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=dtype)
            x = block(x, c, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, c)
        output = output[:, -self.total_queries:, :]

        if self.fast_mode:
            return output  # (B, fast_expected_seq_len, fast_vocab_size)
        else:
            return output.view(B, self.num_actions, self.action_dim, self.per_dim_classes)


class _AutoregressiveClassificationMixin:
    def _setup_ar_tokens(self, hidden_size, total_queries, per_dim_classes):
        self.total_queries = total_queries
        self.per_dim_classes = per_dim_classes
        self.token_embed = nn.Embedding(per_dim_classes, hidden_size)
        self.bos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def _init_ar_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.bos_embed, std=0.02)

    def _build_ar_inputs(self, batch_size, dtype, device, target_ids=None, generated_ids=None):
        if target_ids is not None and generated_ids is not None:
            raise ValueError("Provide either target_ids or generated_ids, not both.")

        if target_ids is not None:
            target_ids = target_ids.to(device=device, dtype=torch.long)
            action_len = target_ids.shape[1]
            prev_ids = target_ids[:, :-1]
        elif generated_ids is not None:
            prev_ids = generated_ids.to(device=device, dtype=torch.long)
            action_len = prev_ids.shape[1] + 1
        else:
            prev_ids = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            action_len = 1

        if action_len > self.total_queries:
            raise ValueError(f"AR sequence length {action_len} exceeds max length {self.total_queries}.")

        invalid = (prev_ids < 0) | (prev_ids >= self.per_dim_classes)
        prev_ids = prev_ids.masked_fill(invalid, self.per_dim_classes - 1)
        bos = self.bos_embed.expand(batch_size, -1, -1).to(dtype=dtype)
        prev_emb = self.token_embed(prev_ids).to(dtype=dtype)
        return torch.cat([bos, prev_emb], dim=1), action_len


class ActionClassificationTransformerMetaqueryAutoregressive(_AutoregressiveClassificationMixin, nn.Module):
    def __init__(self, action_dim, condition_dim, num_actions=1, num_bins=256,
                 hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 fast_mode=False, fast_expected_seq_len=64, fast_vocab_size=2048):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.fast_mode = fast_mode
        self.fast_vocab_size = fast_vocab_size

        total_queries = fast_expected_seq_len if fast_mode else num_actions * action_dim
        per_dim_classes = fast_vocab_size if fast_mode else num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self._setup_ar_tokens(hidden_size, total_queries, per_dim_classes)
        self.cond_proj = nn.Linear(condition_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList([
            MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        self._init_ar_weights()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, target_ids=None, generated_ids=None, history_actions=None):
        B = condition.shape[0]
        dtype = self.input_proj.weight.dtype
        device = condition.device
        condition = condition.to(dtype=dtype)

        action_emb, action_len = self._build_ar_inputs(
            B, dtype, device, target_ids=target_ids, generated_ids=generated_ids
        )
        history_len = 0
        if history_actions is not None:
            hist_emb = self.input_proj(history_actions.to(device=device, dtype=dtype))
            history_len = hist_emb.shape[1]
            x = torch.cat([hist_emb, action_emb], dim=1)
        else:
            x = action_emb

        x = x + self.pos_embed[:, :x.shape[1], :]
        c = self.cond_proj(condition)
        attn_mask = build_causal_action_mask(history_len, action_len, x.device)

        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)

        output = self.final_layer(x, c)
        return output[:, -action_len:, :]


class ActionClassificationTransformerMoEAutoregressive(_AutoregressiveClassificationMixin, nn.Module):
    def __init__(self, action_dim, vlm_hidden_size, num_actions=1, num_bins=256,
                 hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 fast_mode=False, fast_expected_seq_len=64, fast_vocab_size=2048,
                 gen_hidden_size=None):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.fast_mode = fast_mode
        self.fast_vocab_size = fast_vocab_size

        total_queries = fast_expected_seq_len if fast_mode else num_actions * action_dim
        per_dim_classes = fast_vocab_size if fast_mode else num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self._setup_ar_tokens(hidden_size, total_queries, per_dim_classes)
        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        self._init_ar_weights()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, target_ids=None, generated_ids=None, history_actions=None, gen_hidden_states=None):
        vlm_hidden_states = condition
        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        device = final_state.device
        final_state = final_state.to(dtype=dtype)

        c = self.cond_proj(final_state.mean(dim=1))
        B = c.shape[0]
        action_emb, action_len = self._build_ar_inputs(
            B, dtype, device, target_ids=target_ids, generated_ids=generated_ids
        )

        history_len = 0
        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(device=device, dtype=dtype))
            history_len = history_emb.shape[1]
            x = torch.cat([history_emb, action_emb], dim=1)
        else:
            x = action_emb

        x = x + self.pos_embed[:, :x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks):]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states):
            vlm_state = vlm_state.to(dtype=dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=dtype)
            uses_gen = gen_state is not None and block.gen_proj is not None
            kv_extra_len = vlm_state.shape[1] + (gen_state.shape[1] if uses_gen else 0)
            attn_mask = build_causal_action_mask(history_len, action_len, x.device, kv_extra_len=kv_extra_len)
            x = block(x, c, vlm_state, gen_feat=gen_state, attn_mask=attn_mask)

        output = self.final_layer(x, c)
        return output[:, -action_len:, :]
