import sys
import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor, AutoTokenizer,
    SiglipVisionModel, SiglipImageProcessor, LlamaForCausalLM, 
    PaliGemmaForConditionalGeneration, 
    Qwen3VLForConditionalGeneration
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from .policies import (
    ActionDiffusionTransformerMetaquery, ActionDiffusionTransformerMoE,
    ActionRegressionTransformerMetaquery, ActionRegressionTransformerMoE,
    ActionClassificationTransformerMetaquery, ActionClassificationTransformerMoE, ActionVQVAE
)
from .generator import ImageGeneratorTransformer
from .encoder import ActionTransformerProjector
from .connector import ConnectorTransformer

try:
    from .Emu3_5_VisionTokenizer.modeling_emu3p5visionvq import Emu3p5VisionVQModel
except ImportError:
    # Fallback for directory with dot in name (Emu3.5_VisionTokenizer) which is not a valid package name
    sys.path.append(os.path.join(os.path.dirname(__file__), "Emu3.5_VisionTokenizer"))
    from modeling_emu3p5visionvq import Emu3p5VisionVQModel



class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

class VLANeXt(nn.Module):
    def __init__(
        self, 
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7,
        num_actions=1,
        num_queries=16,
        num_history=0,
        loss_type="diffusion", # Options: "diffusion", "regression", "classification"
        future_image_loss_weight=0.0,
        num_train_timesteps=1000,
        num_inference_timesteps=10,
        scheduler_type="ddim", # Options: "ddim", "flow_match"
        condition_type="loose", # Options: "loose", "tight", "soft"
        policy_hidden_size=1024,
        policy_depth=24,
        policy_num_heads=16,
        policy_mlp_ratio=4.0,
        use_proprio_input_vlm=True,
        use_action_input_policy=False,
        use_transformer_proprio_projector=True,
        projector_depth=2,
        projector_num_heads=4,
        use_transformer_connector=True,
        connector_depth=2,
        connector_num_heads=4,
        backbone_mode="finetune", # Options: "frozen", "finetune"
        gradient_checkpointing=True,
        num_bins=256,
        action_vqvae=None,

        generator_hidden_size=768,
        generator_depth=12,
        generator_num_heads=12,
        generator_mlp_ratio=4.0,
        attn_implementation="flash_attention_2",
        dct_loss_weight=0.1,
        dct_low_freq_weight=1.0,
        dct_high_freq_weight=3.0,
        dct_freq_split=0.5,
        dct_similarity_type="mse",  # Options: "mse", "mae", "cosine"
    ):
        super().__init__()
        
        print(f"Initializing VLM {lmm_path} with attn_implementation: {attn_implementation}")
        if "paligemma" in lmm_path.lower():
            self.model_family = "paligemma"
            self.lmm = PaliGemmaForConditionalGeneration.from_pretrained(
                lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        elif "llama" in lmm_path.lower():
            self.model_family = "llama"
            self.lmm = LlamaForCausalLM.from_pretrained(
                lmm_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                vision_encoder_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            tokenizer = AutoTokenizer.from_pretrained(lmm_path)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
            self.processor = LlamaProcessorWrapper(tokenizer, image_processor)
            self.hidden_size = self.lmm.config.hidden_size
            self.vision_projector = nn.Sequential(
                nn.Linear(self.vision_encoder.config.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        elif "qwen" in lmm_path.lower():
            self.model_family = "qwen"
            self.lmm = Qwen3VLForConditionalGeneration.from_pretrained(
                lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        
        if backbone_mode == "frozen":
            self.lmm.requires_grad_(False)
            if self.model_family == "llama":
                self.vision_encoder.requires_grad_(False)
        elif backbone_mode == "finetune":
            self.lmm.requires_grad_(True)
            if self.model_family == "llama":
                self.vision_encoder.requires_grad_(True)
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        if gradient_checkpointing:
            model_to_configure = self.lmm
            if hasattr(model_to_configure, "gradient_checkpointing_enable"):
                model_to_configure.gradient_checkpointing_enable()
            if hasattr(self.lmm, "enable_input_require_grads"):
                self.lmm.enable_input_require_grads()
            config = self.lmm.config
            if hasattr(config, "use_cache"):
                config.use_cache = False
            if self.model_family == "llama":
                 if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                    self.vision_encoder.gradient_checkpointing_enable()

        self.num_queries = num_queries
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.condition_type = condition_type
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.future_image_loss_weight = future_image_loss_weight
        self.enable_future_image_loss = (future_image_loss_weight > 0)
        self.dct_loss_weight = dct_loss_weight
        self.dct_low_freq_weight = dct_low_freq_weight
        self.dct_high_freq_weight = dct_high_freq_weight
        self.dct_freq_split = dct_freq_split
        self.dct_similarity_type = dct_similarity_type
        
        self.action_vqvae_config = action_vqvae
        if self.action_vqvae_config.get('enabled', False):
            self.action_vqvae = ActionVQVAE(
                action_dim=action_dim,
                latent_codes_per_step=3, 
                codebook_size=self.action_vqvae_config.get('codebook_size', 1024),
                hidden_size=self.action_vqvae_config.get('hidden_size', 256),
                depth=self.action_vqvae_config.get('depth', 2),
                num_heads=self.action_vqvae_config.get('num_heads', 4)
            )
        else:
            self.action_vqvae = None


        if self.enable_future_image_loss:
            print("Initializing Future Image Generator Components...")
            self.vq_model = Emu3p5VisionVQModel.from_pretrained("BAAI/Emu3.5-VisionTokenizer", trust_remote_code=True)
            self.vq_model.requires_grad_(False)
            self.vq_codebook_size = self.vq_model.config.codebook_size
            
            self.generator = ImageGeneratorTransformer(
                vocab_size=self.vq_codebook_size,
                vlm_hidden_size=self.hidden_size,
                hidden_size=generator_hidden_size,
                depth=generator_depth,
                num_heads=generator_num_heads,
                mlp_ratio=generator_mlp_ratio
            )
        else:
            self.vq_model = None
            self.generator = None

        if self.use_proprio_input_vlm:
            projector_input_dim = action_dim
            if use_transformer_proprio_projector:
                self.action_projector = ActionTransformerProjector(
                    action_dim=projector_input_dim,
                    hidden_size=self.hidden_size,
                    depth=projector_depth,
                    num_heads=projector_num_heads
                )
            else:
                self.action_projector = nn.Linear(projector_input_dim, self.hidden_size)
        else:
            self.action_projector = None
        
        self.meta_queries = nn.Parameter(
            torch.randn(num_queries, self.hidden_size)
        )
        if self.condition_type == "loose":
            if use_transformer_connector:
                self.connector = ConnectorTransformer(
                    input_dim=self.hidden_size,
                    output_dim=self.hidden_size,
                    depth=connector_depth,
                    num_heads=connector_num_heads
                )
            else:
                self.connector = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size) # Project to diffusion cond dim
                )
        else:
            self.connector = None

        gen_hidden_dim = generator_hidden_size if self.enable_future_image_loss else None
        if loss_type == "regression":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionRegressionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim
                )
            elif condition_type == "loose":
                self.action_head = ActionRegressionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for regression: {condition_type}")
            self.noise_scheduler = None
        elif loss_type == "classification":
            is_vqvae = (self.action_vqvae is not None)

            if condition_type == "loose":
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes
                    )
                else:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False
                    )
            elif condition_type in ["tight", "soft"]:
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes,
                        gen_hidden_size=gen_hidden_dim
                    )
                else:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False,
                        gen_hidden_size=gen_hidden_dim
                    )
            else:
                raise NotImplementedError(f"Classification policy does not support {condition_type}.")
            self.noise_scheduler = None
        elif loss_type == "diffusion":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionDiffusionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim
                )
            elif condition_type == "loose":
                self.action_head = ActionDiffusionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for diffusion: {condition_type}")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if loss_type == "diffusion":
            if scheduler_type == "ddim":
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_train_timesteps,
                    clip_sample=False,
                    prediction_type="epsilon"
                )
            elif scheduler_type == "flow_match": 
                self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def forward_action_vqvae_pretrain(self, actions):
        if self.action_vqvae is None:
            raise RuntimeError("Action VQ-VAE not initialized.")
            
        actions = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
        loss = self.action_vqvae(actions)
        return loss

    def get_vlm_condition(self, input_ids, attention_mask, proprioception=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, token_type_ids=None):
        if self.model_family == "paligemma":
            return self._get_vlm_condition_paligemma(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, token_type_ids=token_type_ids)
        elif self.model_family == "llama":
            return self._get_vlm_condition_llama(input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask)
        elif self.model_family == "qwen":
            return self._get_vlm_condition_qwen(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)

    def _get_vlm_condition_qwen(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw):
        B = input_ids.shape[0]
        
        backbone = self.lmm.model
        lmm_config = self.lmm.config
        pad_token_id = getattr(lmm_config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        inputs_embeds = backbone.get_input_embeddings()(input_ids)
        
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            proprio_ids = torch.full((B, proprioception.shape[1]), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([proprio_ids, input_ids], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            queries_ids = torch.full((B, self.num_queries), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            extended_input_ids = torch.cat([input_ids, queries_ids], dim=1)
        else:
            extended_input_ids = input_ids

        rope_kwargs = {
            "input_ids": extended_input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask
        }

        position_ids, _ = backbone.get_rope_index(**rope_kwargs)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"])
        forward_kwargs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "output_hidden_states": output_hidden_states_flag,
        }
        outputs = backbone(**forward_kwargs)
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)
            
        return connector_out, hidden_states

    def _get_vlm_condition_llama(self, input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask):
        B = input_ids.shape[0]
        pixel_values = pixel_values.to(dtype=self.vision_encoder.dtype)
        
        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_feats = vision_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_feats)

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)
        
        text_embeds = self.lmm.model.embed_tokens(input_ids)

        proprio_embeds = None
        if self.use_proprio_input_vlm and proprioception is not None:
             proprio_embeds = self.action_projector(proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype))

        embeds_list = [image_embeds]
        image_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list = [image_mask]

        if proprio_embeds is not None:
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                p_mask = torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                mask_list.append(p_mask)
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(text_embeds.dtype)
            embeds_list.append(queries_embeds)
            queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            mask_list.append(queries_mask)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        combined_attention_mask = torch.cat(mask_list, dim=1)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"])
        outputs = self.lmm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=output_hidden_states_flag
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _get_vlm_condition_paligemma(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, token_type_ids=None):
        from transformers.models.paligemma.modeling_paligemma import create_causal_mask_mapping

        B = input_ids.shape[0]

        backbone = self.lmm.model

        inputs_embeds = backbone.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = backbone.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = backbone.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            # Proprio tokens are prefix context — token_type_ids=0 (bidirectional)
            if token_type_ids is not None:
                proprio_type_ids = torch.zeros(B, proprioception.shape[1], device=token_type_ids.device, dtype=token_type_ids.dtype)
                token_type_ids = torch.cat([proprio_type_ids, token_type_ids], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            # Query tokens are suffix — token_type_ids=1 (causal)
            if token_type_ids is not None:
                queries_type_ids = torch.ones(B, self.num_queries, device=token_type_ids.device, dtype=token_type_ids.dtype)
                token_type_ids = torch.cat([token_type_ids, queries_type_ids], dim=1)

        # Build the proper PaliGemma causal mask with bidirectional attention on prefix/image tokens
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0) + 1  # PaliGemma positions are 1-indexed
        causal_mask_mapping = create_causal_mask_mapping(
            backbone.config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            is_training=self.training,
        )

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"] )
        outputs = backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states_flag,
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _compute_gen_loss_and_feats(self, future_images, vlm_hidden_states):
        with torch.no_grad():
            future_images = future_images.to(device=self.vq_model.device, dtype=self.vq_model.dtype)
            _, _, (_, _, token_ids) = self.vq_model.encode(future_images)
            B = future_images.shape[0]
            token_ids = token_ids.view(B, -1)
        
        sos_token = torch.zeros((B, 1), dtype=token_ids.dtype, device=token_ids.device)
        gen_input = torch.cat([sos_token, token_ids[:, :-1]], dim=1)
        
        gen_logits, gen_hidden_states = self.generator(gen_input, vlm_hidden_states)
        loss_img = F.cross_entropy(gen_logits.reshape(-1, self.vq_codebook_size), token_ids.reshape(-1))
        
        return loss_img, gen_hidden_states

    def _compute_dct_loss(self, pred, target):
        B, T, D = pred.shape

        if not hasattr(self, '_dct_matrix') or self._dct_matrix.shape[0] != T or self._dct_matrix.device != pred.device:
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            
            dct_m[0, :] *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            
            self._dct_matrix = dct_m

        split_idx = max(1, int(T * self.dct_freq_split))
        freq_weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        freq_weights[:split_idx] = self.dct_low_freq_weight
        freq_weights[split_idx:] = self.dct_high_freq_weight
        freq_weights = freq_weights.view(1, T, 1)

        pred_perm = pred.permute(0, 2, 1)
        pred_dct = torch.matmul(pred_perm, self._dct_matrix.t())
        pred_dct = pred_dct.permute(0, 2, 1)

        target_perm = target.permute(0, 2, 1)
        target_dct = torch.matmul(target_perm, self._dct_matrix.t())
        target_dct = target_dct.permute(0, 2, 1)

        sim_type = self.dct_similarity_type
        if sim_type == "mse":
            diff = (pred_dct - target_dct) ** 2
            return (diff * freq_weights).mean()
        elif sim_type == "mae":
            diff = (pred_dct - target_dct).abs()
            return (diff * freq_weights).mean()
        elif sim_type == "cosine":
            pred_norm = torch.nn.functional.normalize(pred_dct, dim=-1)
            target_norm = torch.nn.functional.normalize(target_dct, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1, keepdim=True)
            cos_dist = 1.0 - cos_sim
            return (cos_dist * freq_weights).mean()
        else:
            raise ValueError(f"Unknown dct_similarity_type: {sim_type!r}. "
                             f"Options are: 'mse', 'mae', 'cosine'.")

    def forward(self, input_ids=None, attention_mask=None, actions=None, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, future_images=None, task=None, token_type_ids=None):
        if task == "action_vqvae_pretrain":
            return self.forward_action_vqvae_pretrain(actions)
            
        if self.loss_type == "regression":
            return self._forward_regression(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, token_type_ids=token_type_ids
            )
        elif self.loss_type == "classification":
            return self._forward_classification(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, token_type_ids=token_type_ids
            )
        elif self.loss_type == "diffusion":
            return self._forward_diffusion(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, token_type_ids=token_type_ids
            )

    def _forward_classification(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, token_type_ids=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids
        )
        
        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)

        policy_history = history_actions if self.use_action_input_policy else None
        
        if self.condition_type in ["tight", "soft"]:
            if self.enable_future_image_loss:
                pred_logits = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred_logits = self.action_head(hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred_logits = self.action_head(cond_input, history_actions=policy_history)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)

        pred_action_continuous = None
        loss = 0.0

        if self.action_vqvae is not None:
             with torch.no_grad():
                 self.action_vqvae.eval()
                 actions_input = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
                 _, indices, _ = self.action_vqvae.encode(actions_input)
             
             loss = F.cross_entropy(
                 pred_logits.reshape(-1, self.action_vqvae.codebook_size), 
                 indices.reshape(-1)
             )

             if self.dct_loss_weight > 0:
                 probs = F.softmax(pred_logits, dim=-1)
                 pred_action_continuous = self.action_vqvae.decode_probs(probs)
        else:
            logits = pred_logits
            pose_logits = logits[:, :, :self.action_dim - 1, :]
            gripper_logits = logits[:, :, -1:, :2]
            
            gt_pose = torch.clamp(actions[:, :, :6], -1, 1)
            gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()
            
            gt_gripper = torch.clamp(actions[:, :, 6:7], -1, 1)
            gt_gripper_idx = ((gt_gripper + 1) / 2).round().long() # 0 or 1

            loss_pose = F.cross_entropy(pose_logits.reshape(-1, self.num_bins), gt_pose_idx.reshape(-1))
            loss_gripper = F.cross_entropy(gripper_logits.reshape(-1, 2), gt_gripper_idx.reshape(-1))
            
            loss = (loss_pose + loss_gripper) / 2.0

            if self.dct_loss_weight > 0:
                pose_probs = F.softmax(pose_logits, dim=-1)
                bin_centers = torch.linspace(-1, 1, self.num_bins, device=actions.device, dtype=pose_probs.dtype)
                pred_pose = torch.sum(pose_probs * bin_centers, dim=-1)

                gripper_probs = F.softmax(gripper_logits, dim=-1)
                p1 = gripper_probs[..., 1]
                pred_gripper = -1.0 + 2.0 * p1
                
                pred_action_continuous = torch.cat([pred_pose, pred_gripper], dim=-1)

        if self.dct_loss_weight > 0 and pred_action_continuous is not None:
             loss_dct = self._compute_dct_loss(pred_action_continuous.float(), actions.float())
             loss = loss + self.dct_loss_weight * loss_dct
        
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        return loss

    def _forward_regression(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, token_type_ids=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids
        )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)
        
        policy_history = history_actions if self.use_action_input_policy else None
        
        if self.condition_type in ["tight", "soft"]:
             if self.enable_future_image_loss:
                 pred_actions = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
             else:
                 pred_actions = self.action_head(hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
             cond_input = connector_out.mean(dim=1)
             pred_actions = self.action_head(cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")

        if actions.ndim == 2: actions = actions.unsqueeze(1)
        loss = F.mse_loss(pred_actions, actions)

        if self.dct_loss_weight > 0:
            loss_dct = self._compute_dct_loss(pred_actions.float(), actions.float())
            loss = loss + self.dct_loss_weight * loss_dct
            
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        return loss

    def _forward_diffusion(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, token_type_ids=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids
        )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)
        noise = torch.randn_like(actions)
        B = actions.shape[0]
        
        if self.scheduler_type == "flow_match":
            sigmas = torch.rand((B,), device=actions.device)
            sigmas_expanded = sigmas.view(B, *([1] * (actions.ndim - 1)))
            noisy_actions = (1.0 - sigmas_expanded) * actions + sigmas_expanded * noise
            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            timesteps = sigmas * self.noise_scheduler.config.num_train_timesteps
            target = noise - actions
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            target = noise
            
        policy_history = history_actions if self.use_action_input_policy else None

        if self.condition_type in ["tight", "soft"]:
            if self.enable_future_image_loss:
                pred = self.action_head(noisy_actions, timesteps, hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred = self.action_head(noisy_actions, timesteps, hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred = self.action_head(noisy_actions, timesteps, cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        loss = F.mse_loss(pred, target)

        if self.dct_loss_weight > 0:
            pred_x_start = None
            if self.scheduler_type == "flow_match":
                 pred_x_start = noisy_actions - sigmas_expanded * pred
            elif self.scheduler_type == "ddim":
                 def view_right(t):
                    while t.ndim < pred.ndim:
                        t = t.unsqueeze(-1)
                    return t
                 alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=pred.device, dtype=pred.dtype)
                 alpha_prod_t = alphas_cumprod[timesteps]
                 pred_x_start = (noisy_actions - view_right((1 - alpha_prod_t).sqrt()) * pred) / view_right(alpha_prod_t.sqrt())

            if pred_x_start is not None:
                 loss_dct = self._compute_dct_loss(pred_x_start.float(), actions.float())
                 loss = loss + self.dct_loss_weight * loss_dct
            
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        return loss

    @torch.no_grad()
    def predict_action(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, token_type_ids=None):
        B = input_ids.shape[0]

        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids
        )
        
        policy_history = history_actions if self.use_action_input_policy else None
        gen_hidden_states = None
        if self.enable_future_image_loss and self.condition_type in ["tight", "soft"]:
             num_img_tokens = 256 
             curr_ids = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
             gen_context = hidden_states
             for _ in range(num_img_tokens):
                 logits, _ = self.generator(curr_ids, gen_context)
                 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                 curr_ids = torch.cat([curr_ids, next_token], dim=1)
             gen_input = curr_ids[:, :-1]
             _, gen_hidden_states = self.generator(gen_input, gen_context)

        if self.loss_type == "regression":
            if self.condition_type in ["tight", "soft"]:
                if self.enable_future_image_loss:
                    action = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                else:
                    action = self.action_head(hidden_states, history_actions=policy_history)
            elif self.condition_type == "loose":
                cond_input = connector_out.mean(dim=1)
                action = self.action_head(cond_input, history_actions=policy_history)
            if action.ndim == 2 and self.num_actions > 1:
                action = action.view(action.shape[0], self.num_actions, self.action_dim)
            
            return action.to(dtype=self.lmm.dtype)

        elif self.loss_type == "classification":
            if self.condition_type in ["tight", "soft"]:
                if self.enable_future_image_loss:
                    logits = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                else:
                    logits = self.action_head(hidden_states, history_actions=policy_history)
            else:
                cond_input = connector_out.mean(dim=1)
                logits = self.action_head(cond_input, history_actions=policy_history)

            if self.action_vqvae is not None:
                indices = torch.argmax(logits, dim=-1)  # (B, T, Latent_Codes)
                action = self.action_vqvae.decode_indices(indices)
                return action.to(dtype=self.lmm.dtype)

            else:
                pose_logits = logits[:, :, :self.action_dim - 1, :]
                gripper_logits = logits[:, :, -1:, :2]
                pose_idx = torch.argmax(pose_logits, dim=-1)
                gripper_idx = torch.argmax(gripper_logits, dim=-1)
                pose_pred = (pose_idx.float() / (self.num_bins - 1)) * 2 - 1
                gripper_pred = gripper_idx.float() * 2 - 1
                action = torch.cat([pose_pred, gripper_pred], dim=-1).to(dtype=self.lmm.dtype)
                return action

        elif self.loss_type == "diffusion":
            action = torch.randn(B, self.num_actions, self.action_dim, device=input_ids.device).to(self.lmm.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            
            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), t, device=input_ids.device)
                if self.scheduler_type != "flow_match": timesteps = timesteps.long()
                if self.condition_type in ["tight", "soft"]:
                    if self.enable_future_image_loss:
                        output = self.action_head(action, timesteps, hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                    else:
                        output = self.action_head(action, timesteps, hidden_states, history_actions=policy_history)
                else:
                    cond_input = connector_out.mean(dim=1)
                    output = self.action_head(action, timesteps, cond_input, history_actions=policy_history)
                
                action = self.noise_scheduler.step(output, t, action).prev_sample
                action = action.to(dtype=self.lmm.dtype)
            
            return action
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    @torch.no_grad()
    def predict_image(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, max_new_tokens=1024, token_type_ids=None):
        _, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids
        )
        gen_vlm_ctx = hidden_states
        
        curr_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            logits, _ = self.generator(curr_ids, gen_vlm_ctx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        generated_tokens = curr_ids[:, 1:]
        H_latent = int(generated_tokens.shape[1]**0.5)
        decoded_images = self.vq_model.decode_code(generated_tokens, shape=(input_ids.shape[0], H_latent, H_latent))
        return decoded_images

if __name__ == "__main__":
    print("Testing VLANeXt Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Initialize Model (Minimal Config)
    model = VLANeXt(
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        action_dim=7, num_actions=4, num_history=2,
        backbone_mode="finetune", gradient_checkpointing=False
    ).to(device, dtype)
    processor = model.processor

    def run_test(modality="image"):
        print(f"\n=== Testing {modality.capitalize()} ===")
        B = 2
        # Dummy Data
        img = Image.new('RGB', (64, 64), color='red')
        media = [img] * B if modality == "image" else [[img]*8] * B
        content_key = "image" if modality == "image" else "video"
        
        # Process
        msgs = [[{"role": "user", "content": [{"type": content_key, content_key: m}, {"type": "text", "text": "Task."}]}] for m in media]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, **{f"{modality}s": media}, padding=True, return_tensors="pt")
        
        # Move to device & cast
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in ["pixel_values", "pixel_values_videos"]:
            if k in inputs: inputs[k] = inputs[k].to(dtype)
            
        # Filter valid args for forward
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        fwd_args = {k: v for k, v in inputs.items() if k in valid_keys}

        # Tensors
        act_gt = torch.randn(B, 4, 7, device=device, dtype=dtype)
        proprio = torch.randn(B, 2, 7, device=device, dtype=dtype)
        hist_act = torch.randn(B, 2, 7, device=device, dtype=dtype)

        # Tests
        print(f"Action Gen Loss: {model(actions=act_gt, proprioception=proprio, history_actions=hist_act, **fwd_args).item():.4f}")
        print(f"Action Pred Shape: {model.predict_action(proprioception=proprio, history_actions=hist_act, **fwd_args).shape}")

    run_test("image")
    run_test("video")
    print("\nTest Passed!")