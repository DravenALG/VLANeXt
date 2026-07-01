import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer,
    SiglipVisionModel, SiglipImageProcessor, LlamaForCausalLM,
    PaliGemmaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from src.models.modules.policies import (
    ActionDiffusionTransformerMetaquery, ActionDiffusionTransformerMoE,
    ActionRegressionTransformerMetaquery, ActionRegressionTransformerMoE,
    ActionClassificationTransformerMetaquery, ActionClassificationTransformerMoE,
    ActionClassificationTransformerMetaqueryAutoregressive,
    ActionClassificationTransformerMoEAutoregressive
)
from src.models.modules.encoder import ActionTransformerProjector
from src.models.modules.connector import ConnectorTransformer
from src.models.modules.generator import DINOFeatureFlowGenerator, EmuTokenClassificationGenerator
from src.models.emu_vision_tokenizer.modeling_emu3p5visionvq import Emu3p5VisionVQModel
from src.models.wan_backbone import WanBackbone


class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

def _load_backbone(model_cls, model_path, use_pretrained_backbone=True, **kwargs):
    if use_pretrained_backbone:
        return model_cls.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    attn_implementation = kwargs.get("_attn_implementation") or kwargs.get("attn_implementation")
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    return model_cls(config).to(dtype=torch.bfloat16)

class VLANeXt(nn.Module):

    # -----------------------------------------------------------------------------
    # ------------------------------ Initialization -------------------------------
    # -----------------------------------------------------------------------------
    def __init__(
        self, 
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        use_pretrained_backbone=True,
        action_dim=7,
        num_actions=1,
        num_queries=16,
        num_history=0,
        loss_type="diffusion", # Options: "diffusion", "regression", "classification"
        classification_type="parallel", # Options: "parallel", "autoregressive"
        future_image_loss_weight=0.0,
        num_train_timesteps=1000,
        num_inference_timesteps=10,
        scheduler_type="ddim", # Options: "ddim", "flow_match"
        diffusion_loss_domain="noise", # Options: "noise" (predicts ε for DDIM, velocity for flow_match), "x0" (predicts clean action directly)
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
        train_text_embedding=None,
        train_vision_encoder=None,
        gradient_checkpointing=True,
        num_bins=256,
        fast_action_tokenizer=None,
        generator_hidden_size=768,
        generator_depth=12,
        generator_num_heads=12,
        generator_mlp_ratio=4.0,
        generator_max_seq_len=1024,
        future_image_num_tokens=256,
        future_image_prediction_type="emu_token",
        future_image_dino_model_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
        future_image_dino_image_size=256,
        future_image_flow_num_inference_timesteps=10,
        attn_implementation="flash_attention_2",
        language_action_loss_weight=0.0,
        policy_gradient_stop_for_vlm=False,
        dct_loss_weight=0.1,
        dct_low_freq_weight=1.0,
        dct_high_freq_weight=3.0,
        dct_freq_split=0.5,
        dct_similarity_type="mse",  # Options: "mse", "mae", "cosine"
        video_generation_loss_weight=1.0,
        wan_action_condition_mode="fast",
        wan_flow_shift=5.0,
        wan_text_len=512,
        wan_tokenizer_model_id="google/umt5-xxl",
        future_video_downsample=1,
    ):
        super().__init__()
        
        print(f"Initializing VLM {lmm_path} with pretrained_backbone={use_pretrained_backbone}, attn_implementation: {attn_implementation}")
        if "wan" in lmm_path.lower():
            self.model_family = "wan"
            self.lmm = WanBackbone(
                model_id=lmm_path,
                tokenizer_model_id=wan_tokenizer_model_id,
                text_len=wan_text_len,
                torch_dtype=torch.bfloat16,
                device="cpu",
            )
            self.processor = self.lmm.processor
            self.hidden_size = self.lmm.hidden_size
        elif "paligemma" in lmm_path.lower():
            self.model_family = "paligemma"
            self.lmm = _load_backbone(
                PaliGemmaForConditionalGeneration,
                lmm_path,
                use_pretrained_backbone,
                _attn_implementation=attn_implementation,
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        elif "llama" in lmm_path.lower():
            self.model_family = "llama"
            self.lmm = _load_backbone(
                LlamaForCausalLM,
                lmm_path,
                use_pretrained_backbone,
                attn_implementation=attn_implementation,
            )
            self.vision_encoder = _load_backbone(
                SiglipVisionModel,
                vision_encoder_path,
                use_pretrained_backbone,
                attn_implementation=attn_implementation,
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
        elif "qwen3_5" in lmm_path.lower() or "qwen3.5" in lmm_path.lower():
            self.model_family = "qwen"
            self.lmm = _load_backbone(
                Qwen3_5ForConditionalGeneration,
                lmm_path,
                use_pretrained_backbone,
                _attn_implementation=attn_implementation,
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        elif "qwen3-vl" in lmm_path.lower():
            self.model_family = "qwen"
            self.lmm = _load_backbone(
                Qwen3VLForConditionalGeneration,
                lmm_path,
                use_pretrained_backbone,
                _attn_implementation=attn_implementation,
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        
        self.configure_backbone_trainability(
            backbone_mode,
            train_text_embedding=train_text_embedding,
            train_vision_encoder=train_vision_encoder,
        )

        if gradient_checkpointing:
            if self.model_family == "wan":
                if hasattr(self.lmm.dit, "gradient_checkpointing_enable"):
                    self.lmm.dit.gradient_checkpointing_enable()
            else:
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
        self.classification_type = str(classification_type or "parallel").lower()
        if self.classification_type not in ("parallel", "autoregressive"):
            raise ValueError(
                f"Unknown classification_type: {self.classification_type}. "
                "Options: 'parallel', 'autoregressive'."
            )
        self.scheduler_type = scheduler_type
        self.diffusion_loss_domain = diffusion_loss_domain
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.condition_type = condition_type
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.is_video_generation_backbone = self.model_family == "wan"
        self.video_generation_loss_weight = float(video_generation_loss_weight)
        self.enable_video_generation_loss = self.is_video_generation_backbone
        self.wan_action_condition_mode = str(wan_action_condition_mode or "fast").lower()
        if self.wan_action_condition_mode not in ("fast", "joint"):
            raise ValueError("wan_action_condition_mode must be one of: 'fast', 'joint'.")
        self.wan_flow_shift = float(wan_flow_shift)
        self.future_video_downsample = int(future_video_downsample)
        if self.future_video_downsample <= 0:
            raise ValueError(f"future_video_downsample must be positive, got {future_video_downsample}.")
        if self.is_video_generation_backbone:
            if loss_type != "diffusion":
                raise ValueError("WAN video-generation backbone requires loss_type='diffusion'.")
            if scheduler_type != "flow_match":
                raise ValueError("WAN video-generation backbone requires scheduler_type='flow_match'.")
            if condition_type != "tight":
                raise ValueError("WAN video-generation backbone only supports condition_type='tight'.")
            if float(future_image_loss_weight) > 0:
                print("WAN backbone ignores future_image_loss_weight; disabling future image loss.")
            if float(language_action_loss_weight) > 0:
                print("WAN backbone ignores language_action_loss_weight; disabling language-action loss.")
            future_image_loss_weight = 0.0
            language_action_loss_weight = 0.0
        self.future_image_loss_weight = float(future_image_loss_weight)
        self.enable_future_image_loss = self.future_image_loss_weight > 0
        self.future_image_num_tokens = int(future_image_num_tokens)
        self.future_image_prediction_type = future_image_prediction_type
        self.future_image_dino_image_size = int(future_image_dino_image_size)
        self.future_image_flow_num_inference_timesteps = int(future_image_flow_num_inference_timesteps)
        self.language_action_loss_weight = float(language_action_loss_weight)
        self.enable_language_action_loss = self.language_action_loss_weight > 0
        self.policy_gradient_stop_for_vlm = bool(policy_gradient_stop_for_vlm)
        self.dct_loss_weight = dct_loss_weight
        self.dct_low_freq_weight = dct_low_freq_weight
        self.dct_high_freq_weight = dct_high_freq_weight
        self.dct_freq_split = dct_freq_split
        self.dct_similarity_type = dct_similarity_type
        if self.future_image_prediction_type not in ("emu_token", "dinov3_flow"):
            raise ValueError(
                f"Unknown future_image_prediction_type: {self.future_image_prediction_type}. "
                "Options: 'emu_token', 'dinov3_flow'."
            )
        

        self.fast_tokenizer = None
        self.fast_vocab_size = None
        self.fast_expected_seq_len = None
        self.fast_eos_id = None
        self.fast_pad_id = None
        _fast_cfg = fast_action_tokenizer or {}
        if _fast_cfg.get('enabled', False):
            import json as _json
            from transformers import PreTrainedTokenizerFast as _PTTFast
            from .fast_tokenizer.processing_action_tokenizer import UniversalActionProcessor
            _custom_path = _fast_cfg.get('tokenizer_path', '')
            _fast_dir = (
                _custom_path if _custom_path and os.path.isdir(_custom_path)
                else os.path.join(os.path.dirname(__file__), "FAST_ActionTokenizer")
            )
            # ProcessorMixin.save_pretrained may put the tokenizer in a
            # `bpe_tokenizer/` subdirectory; fall back to that if needed.
            _bpe_subdir = os.path.join(_fast_dir, "bpe_tokenizer")
            _bpe_load_dir = _bpe_subdir if os.path.isdir(_bpe_subdir) else _fast_dir
            print(f"Loading FAST tokenizer from: {_bpe_load_dir}")
            _bpe = _PTTFast.from_pretrained(_bpe_load_dir)
            with open(os.path.join(_fast_dir, "processor_config.json")) as _f:
                _proc_cfg = _json.load(_f)
            _vocab_size = _proc_cfg.get('vocab_size', 2048)
            self.fast_tokenizer = UniversalActionProcessor(
                bpe_tokenizer=_bpe,
                scale=_proc_cfg.get('scale', 10),
                vocab_size=_vocab_size,
                min_token=_proc_cfg.get('min_token', 0),
                action_dim=action_dim,
                time_horizon=num_actions,
            )
            self.fast_eos_id = _vocab_size
            self.fast_vocab_size = _vocab_size + 1  # FAST vocab plus EOS class
            self.fast_expected_seq_len = _fast_cfg.get('expected_seq_len', 64)
            self.fast_pad_id = _vocab_size + 1  # out-of-range ID used as padding sentinel

        self.vq_model = None
        self.vq_codebook_size = None
        self.future_image_processor = None
        self.dino_model = None
        self.dino_num_register_tokens = 0
        self.future_image_flow_scheduler = None
        self.generator = None

        if self.enable_future_image_loss and self.future_image_prediction_type == "emu_token":
            print("Initializing Emu future image token generator components...")
            self.vq_model = Emu3p5VisionVQModel.from_pretrained(
                "BAAI/Emu3.5-VisionTokenizer",
                trust_remote_code=True,
            )
            self.vq_model.requires_grad_(False)
            self.vq_model.eval()
            self.vq_codebook_size = self.vq_model.config.codebook_size
            self.generator = EmuTokenClassificationGenerator(
                vocab_size=self.vq_codebook_size,
                vlm_hidden_size=self.hidden_size,
                hidden_size=generator_hidden_size,
                depth=generator_depth,
                num_heads=generator_num_heads,
                mlp_ratio=generator_mlp_ratio,
                max_seq_len=generator_max_seq_len,
            )
        elif self.enable_future_image_loss and self.future_image_prediction_type == "dinov3_flow":
            print(f"Initializing DINOv3 future feature flow generator: {future_image_dino_model_path}")
            size = {"height": self.future_image_dino_image_size, "width": self.future_image_dino_image_size}
            self.future_image_processor = AutoImageProcessor.from_pretrained(
                future_image_dino_model_path,
                size=size,
                trust_remote_code=True,
            )
            self.dino_model = AutoModel.from_pretrained(
                future_image_dino_model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.dino_model.requires_grad_(False)
            self.dino_model.eval()
            self.dino_num_register_tokens = int(getattr(self.dino_model.config, "num_register_tokens", 0) or 0)
            dino_hidden_size = int(getattr(self.dino_model.config, "hidden_size", 0) or 0)
            if dino_hidden_size <= 0:
                raise ValueError("DINOv3 flow mode requires a ViT-style model config with hidden_size.")
            self.generator = DINOFeatureFlowGenerator(
                feature_dim=dino_hidden_size,
                vlm_hidden_size=self.hidden_size,
                hidden_size=generator_hidden_size,
                depth=generator_depth,
                num_heads=generator_num_heads,
                mlp_ratio=generator_mlp_ratio,
                max_seq_len=generator_max_seq_len,
            )
            self.future_image_flow_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps
            )

        gen_hidden_dim = generator_hidden_size if self.enable_future_image_loss else None

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
        self.wan_proprio_projector = None
        if self.model_family == "wan" and self.use_proprio_input_vlm:
            self.wan_proprio_projector = nn.Linear(action_dim, int(self.lmm.dit.text_dim))
        
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
                    gen_hidden_size=gen_hidden_dim,
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
            is_fast = (self.fast_tokenizer is not None)
            head_kwargs = dict(
                action_dim=action_dim,
                num_actions=num_actions,
                hidden_size=policy_hidden_size,
                depth=policy_depth,
                num_heads=policy_num_heads,
                mlp_ratio=policy_mlp_ratio,
            )
            if is_fast:
                head_kwargs.update(
                    fast_mode=True,
                    fast_expected_seq_len=self.fast_expected_seq_len,
                    fast_vocab_size=self.fast_vocab_size,
                )
            else:
                head_kwargs["num_bins"] = num_bins

            if self.classification_type == "parallel":
                metaquery_cls = ActionClassificationTransformerMetaquery
                moe_cls = ActionClassificationTransformerMoE
            else:
                metaquery_cls = ActionClassificationTransformerMetaqueryAutoregressive
                moe_cls = ActionClassificationTransformerMoEAutoregressive

            if condition_type == "loose":
                self.action_head = metaquery_cls(
                    condition_dim=self.hidden_size,
                    **head_kwargs,
                )
            elif condition_type in ["tight", "soft"]:
                self.action_head = moe_cls(
                    vlm_hidden_size=self.hidden_size,
                    gen_hidden_size=gen_hidden_dim,
                    **head_kwargs,
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
                    gen_hidden_size=gen_hidden_dim,
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

        if (
            not self.use_action_input_policy
            and self.loss_type in ("classification", "regression")
            and hasattr(self.action_head, "input_proj")
        ):
            self.action_head.input_proj.requires_grad_(False)

        if loss_type == "diffusion":
            if diffusion_loss_domain not in ("noise", "x0"):
                raise ValueError(f"Unknown diffusion_loss_domain: {diffusion_loss_domain}. Options: 'noise', 'x0'.")
            if scheduler_type == "ddim":
                ddim_prediction_type = "sample" if diffusion_loss_domain == "x0" else "epsilon"
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_train_timesteps,
                    clip_sample=False,
                    prediction_type=ddim_prediction_type
                )
            elif scheduler_type == "flow_match":
                self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Utilities for accessing model components
    def get_text_embedding(self):
        if hasattr(self.lmm, "get_input_embeddings"):
            return self.lmm.get_input_embeddings()
        backbone = getattr(self.lmm, "model", None)
        if backbone is not None and hasattr(backbone, "get_input_embeddings"):
            return backbone.get_input_embeddings()
        return None

    def get_vision_encoder(self):
        if self.model_family == "llama":
            return getattr(self, "vision_encoder", None)
        backbone = getattr(self.lmm, "model", None)
        if backbone is None:
            return None
        if self.model_family == "qwen":
            return getattr(backbone, "visual", None)
        if self.model_family == "paligemma":
            return getattr(backbone, "vision_tower", None)
        return None

    # Configure the trainability of the backbone and its components based on the specified mode
    def configure_backbone_trainability(
        self,
        backbone_mode,
        train_text_embedding=None,
        train_vision_encoder=None,
    ):
        if backbone_mode == "frozen":
            backbone_trainable = False
        elif backbone_mode == "finetune":
            backbone_trainable = True
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        self.lmm.requires_grad_(backbone_trainable)
        if self.model_family == "wan":
            self.lmm.dit.requires_grad_(backbone_trainable)
            self.lmm.text_encoder_model.requires_grad_(False)
            self.lmm.vae_model.requires_grad_(False)
            self.lmm.text_encoder_model.eval()
            self.lmm.vae_model.eval()
        if self.model_family == "llama":
            self.vision_encoder.requires_grad_(backbone_trainable)

        text_embedding = self.get_text_embedding()
        if train_text_embedding is not None and text_embedding is not None:
            text_embedding.requires_grad_(bool(train_text_embedding))

        vision_encoder = self.get_vision_encoder()
        if train_vision_encoder is not None and vision_encoder is not None:
            vision_encoder.requires_grad_(bool(train_vision_encoder))

        if (
            hasattr(self, "action_head")
            and not self.use_action_input_policy
            and self.loss_type in ("classification", "regression")
            and hasattr(self.action_head, "input_proj")
        ):
            self.action_head.input_proj.requires_grad_(False)

    # -----------------------------------------------------------------------------
    # ------------------------ Conditions from Backbone ---------------------------
    # -----------------------------------------------------------------------------
    # Utilities for generating VLM conditions based on the model family
    def get_vlm_condition(self, input_ids=None, attention_mask=None, proprioception=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, token_type_ids=None, language_action_labels=None, append_meta_queries=True, return_last_hidden_state=False, prompt_texts=None, current_images=None, future_videos=None, noisy_video_latents=None, timesteps=None):
        if self.model_family == "paligemma":
            return self._get_vlm_condition_paligemma(
                input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values,
                token_type_ids=token_type_ids, language_action_labels=language_action_labels,
                append_meta_queries=append_meta_queries,
                return_last_hidden_state=return_last_hidden_state,
            )
        elif self.model_family == "llama":
            return self._get_vlm_condition_llama(
                input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask,
                language_action_labels=language_action_labels,
                append_meta_queries=append_meta_queries,
                return_last_hidden_state=return_last_hidden_state,
            )
        elif self.model_family == "qwen":
            return self._get_vlm_condition_qwen(
                input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values,
                pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids,
                language_action_labels=language_action_labels,
                append_meta_queries=append_meta_queries,
                return_last_hidden_state=return_last_hidden_state,
            )
        elif self.model_family == "wan":
            return self._get_vlm_condition_wam(
                prompt_texts=prompt_texts,
                current_images=current_images,
                future_videos=future_videos,
                noisy_video_latents=noisy_video_latents,
                timesteps=timesteps,
                proprioception=proprioception,
                append_meta_queries=append_meta_queries,
                return_last_hidden_state=return_last_hidden_state,
            )

    def _wan_extra_context(self, proprioception):
        if not self.use_proprio_input_vlm or proprioception is None:
            return None
        projector = self.wan_proprio_projector or self.action_projector
        if projector is None:
            return None
        param = next(projector.parameters())
        return projector(proprioception.to(device=param.device, dtype=param.dtype))

    def _wan_context(self, prompt_texts, proprioception=None):
        if prompt_texts is None:
            raise ValueError("prompt_texts must be provided when using WAN backbone.")
        extra_context = self._wan_extra_context(proprioception)
        return self.lmm.encode_prompts(prompt_texts, extra_context=extra_context)

    def _wan_current_video(self, current_images=None, future_videos=None):
        if current_images is not None:
            if current_images.ndim == 4:
                return current_images.unsqueeze(2)
            if current_images.ndim == 5 and current_images.shape[2] == 1:
                return current_images
            raise ValueError(
                "current_images must have shape [B, C, H, W] or [B, C, 1, H, W], "
                f"got {tuple(current_images.shape)}."
            )
        if future_videos is None:
            raise ValueError("Either current_images or future_videos must be provided for WAN conditioning.")
        if future_videos.ndim != 5:
            raise ValueError(f"future_videos must have shape [B, C, T, H, W], got {tuple(future_videos.shape)}.")
        return future_videos[:, :, :1]

    def _sample_wan_flow_timesteps(self, batch_size, device, dtype):
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigmas = self.lmm.shifted_sigma(u, self.wan_flow_shift).to(device=device, dtype=dtype)
        timesteps = sigmas.float() * float(self.num_train_timesteps)
        return timesteps, sigmas

    def _wan_add_flow_noise(self, clean, noise, sigmas):
        sigmas_expanded = sigmas.view(clean.shape[0], *([1] * (clean.ndim - 1)))
        return (1.0 - sigmas_expanded) * clean + sigmas_expanded * noise

    def _wan_video_token_layout(self, video_latents):
        if video_latents.ndim != 5:
            raise ValueError(f"video_latents must be [B, C, T, H, W], got {tuple(video_latents.shape)}.")
        patch_t, patch_h, patch_w = self.lmm.dit.patch_size
        if int(patch_t) != 1:
            raise ValueError(f"WAN first-frame conditioning requires temporal patch size 1, got {patch_t}.")
        batch_size, _, frames, height, width = video_latents.shape
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                "WAN latent spatial dims must be divisible by patch size, "
                f"got H={height}, W={width}, patch=({patch_h}, {patch_w})."
            )
        tokens_per_frame = (height // patch_h) * (width // patch_w)
        seq_len = int(frames * tokens_per_frame)
        return batch_size, int(tokens_per_frame), seq_len

    def _wan_first_frame_token_timesteps(self, video_latents, timesteps):
        batch_size, first_frame_tokens, seq_len = self._wan_video_token_layout(video_latents)
        token_timesteps = timesteps.to(device=video_latents.device, dtype=video_latents.dtype)
        if token_timesteps.ndim == 1:
            if token_timesteps.shape[0] != batch_size:
                raise ValueError(
                    f"timesteps batch {token_timesteps.shape[0]} does not match video batch {batch_size}."
                )
            token_timesteps = token_timesteps[:, None].expand(batch_size, seq_len).clone()
        elif token_timesteps.ndim == 2:
            if token_timesteps.shape != (batch_size, seq_len):
                raise ValueError(
                    f"token timesteps must be {(batch_size, seq_len)}, got {tuple(token_timesteps.shape)}."
                )
            token_timesteps = token_timesteps.clone()
        else:
            raise ValueError(f"timesteps must be [B] or [B, seq_len], got {tuple(token_timesteps.shape)}.")
        token_timesteps[:, :first_frame_tokens] = 0
        return token_timesteps

    def _wan_fast_video_attention_inputs(self, video_latents, timesteps):
        _, first_frame_tokens, seq_len = self._wan_video_token_layout(video_latents)

        self_attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=video_latents.device)
        self_attn_mask[:first_frame_tokens, first_frame_tokens:] = False

        token_timesteps = self._wan_first_frame_token_timesteps(video_latents, timesteps)
        return token_timesteps, self_attn_mask

    def _get_vlm_condition_wam(
        self,
        prompt_texts=None,
        current_images=None,
        future_videos=None,
        noisy_video_latents=None,
        timesteps=None,
        proprioception=None,
        append_meta_queries=True,
        return_last_hidden_state=False,
        return_video_pred=False,
    ):
        if self.condition_type != "tight":
            raise ValueError("WAN backbone only supports tight conditioning; soft/loose are not implemented.")
        del append_meta_queries
        if noisy_video_latents is None:
            current_video = self._wan_current_video(current_images=current_images, future_videos=future_videos)
            latents = self.lmm.encode_videos(current_video.to(device=self.lmm.device, dtype=self.lmm.dtype))
        else:
            latents = noisy_video_latents.to(device=self.lmm.device, dtype=self.lmm.dtype)
        batch_size = latents.shape[0]
        if timesteps is None:
            timesteps = torch.zeros((batch_size,), device=latents.device, dtype=latents.dtype)
        else:
            timesteps = timesteps.to(device=latents.device, dtype=latents.dtype)
        if latents.shape[2] >= 1:
            timesteps = self._wan_first_frame_token_timesteps(latents, timesteps)
        context = self._wan_context(prompt_texts, proprioception=proprioception)
        pred_video, hidden_states = self.lmm.forward_latents(
            latents,
            timesteps,
            context,
            return_hidden_states=True,
        )
        if self.policy_gradient_stop_for_vlm and hidden_states is not None:
            hidden_states = tuple(state.detach() for state in hidden_states)
        connector_out = None
        loss_language_action = None
        result = (connector_out, hidden_states, loss_language_action)
        if return_video_pred:
            result = result + (pred_video,)
        if return_last_hidden_state:
            result = result + (hidden_states[-1],)
        return result

    def _select_wan_policy_hidden_states(self, hidden_states, video_latents):
        if hidden_states is None:
            raise ValueError("WAN policy conditioning requires hidden states from the video DiT forward.")
        if self.wan_action_condition_mode == "joint":
            return hidden_states
        if self.wan_action_condition_mode != "fast":
            raise ValueError(f"Unknown WAN action condition mode: {self.wan_action_condition_mode}")
        if video_latents is None or video_latents.ndim != 5:
            raise ValueError("video_latents must be [B, C, T, H, W] for WAN fast conditioning.")
        _, patch_h, patch_w = self.lmm.dit.patch_size
        height, width = video_latents.shape[-2:]
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                "WAN latent spatial dims must be divisible by patch size, "
                f"got H={height}, W={width}, patch=({patch_h}, {patch_w})."
            )
        first_frame_tokens = (height // patch_h) * (width // patch_w)
        return tuple(state[:, :first_frame_tokens, :] for state in hidden_states)

    def _compute_video_generation_loss(
        self,
        future_videos,
        timesteps,
        sigmas,
        prompt_texts,
        proprioception=None,
        return_hidden_states=True,
    ):
        if future_videos is None:
            raise ValueError("future_videos must be provided when using WAN video-generation backbone.")
        if future_videos.ndim != 5:
            raise ValueError(f"future_videos must be [B, C, T, H, W], got {tuple(future_videos.shape)}.")
        expected_frames = self.num_actions // self.future_video_downsample + 1
        if self.num_actions % self.future_video_downsample != 0:
            raise ValueError("num_actions must be divisible by future_video_downsample for WAN video learning.")
        if future_videos.shape[2] != expected_frames:
            raise ValueError(
                f"future_videos has T={future_videos.shape[2]}, expected {expected_frames} "
                f"from num_actions={self.num_actions} and future_video_downsample={self.future_video_downsample}."
            )
        if future_videos.shape[2] % 4 != 1:
            raise ValueError(f"WAN future video length must satisfy T % 4 == 1, got {future_videos.shape[2]}.")

        clean_latents = self.lmm.encode_videos(future_videos.to(device=self.lmm.device, dtype=self.lmm.dtype))
        noise = torch.randn_like(clean_latents)
        noisy_latents = self._wan_add_flow_noise(clean_latents, noise, sigmas.to(clean_latents.dtype))
        noisy_latents[:, :, :1] = clean_latents[:, :, :1]
        target_velocity = noise - clean_latents

        context = self._wan_context(prompt_texts, proprioception=proprioception)
        video_timesteps = timesteps.to(device=clean_latents.device, dtype=clean_latents.dtype)
        self_attn_mask = None
        if self.wan_action_condition_mode == "fast":
            video_timesteps, self_attn_mask = self._wan_fast_video_attention_inputs(
                noisy_latents,
                video_timesteps,
            )
        else:
            video_timesteps = self._wan_first_frame_token_timesteps(noisy_latents, video_timesteps)
        pred_velocity, hidden_states = self.lmm.forward_latents(
            noisy_latents,
            video_timesteps,
            context,
            return_hidden_states=return_hidden_states,
            self_attn_mask=self_attn_mask,
        )
        pred_tail = pred_velocity[:, :, 1:]
        target_tail = target_velocity[:, :, 1:]
        loss_video = F.mse_loss(pred_tail.float(), target_tail.float())
        if self.policy_gradient_stop_for_vlm and hidden_states is not None:
            hidden_states = tuple(state.detach() for state in hidden_states)
        return loss_video, hidden_states, noisy_latents

    # Utilities for generating VLM conditions from Qwen
    def _get_vlm_condition_qwen(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=None, language_action_labels=None, append_meta_queries=True, return_last_hidden_state=False):
        B = input_ids.shape[0]

        backbone = self.lmm.model
        lmm_config = self.lmm.config
        pad_token_id = getattr(lmm_config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        inputs_embeds = backbone.get_input_embeddings()(input_ids)
        if language_action_labels is not None:
            language_action_labels = language_action_labels.to(device=input_ids.device)

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
            if language_action_labels is not None:
                language_action_labels = torch.cat([self._ignore_label_block(language_action_labels, proprioception.shape[1]), language_action_labels], dim=1)
            if mm_token_type_ids is not None:
                proprio_type_ids = torch.zeros(B, proprioception.shape[1], dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
                mm_token_type_ids = torch.cat([proprio_type_ids, mm_token_type_ids], dim=1)

        if append_meta_queries and self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            queries_ids = torch.full((B, self.num_queries), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            extended_input_ids = torch.cat([input_ids, queries_ids], dim=1)
            if language_action_labels is not None:
                language_action_labels = torch.cat([language_action_labels, self._ignore_label_block(language_action_labels, self.num_queries)], dim=1)
            if mm_token_type_ids is not None:
                queries_type_ids = torch.zeros(B, self.num_queries, dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
                mm_token_type_ids = torch.cat([mm_token_type_ids, queries_type_ids], dim=1)
        else:
            extended_input_ids = input_ids

        rope_kwargs = {
            "input_ids": extended_input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask
        }

        position_ids, _ = backbone.get_rope_index(**rope_kwargs)

        output_hidden_states_flag = append_meta_queries and (
            self.enable_future_image_loss or self.condition_type in ["tight", "soft"]
        )
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
        loss_language_action = self._compute_shared_language_action_loss(outputs.last_hidden_state, language_action_labels)
        if self.policy_gradient_stop_for_vlm and self.condition_type in ["tight", "soft"] and hidden_states is not None:
            hidden_states = tuple(state.detach() for state in hidden_states)
        connector_out = None
        if append_meta_queries and self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            if self.policy_gradient_stop_for_vlm:
                query_outputs = query_outputs.detach()
            connector_out = self.connector(query_outputs)
        result = (connector_out, hidden_states, loss_language_action)
        return result + (outputs.last_hidden_state,) if return_last_hidden_state else result

    # Utilities for generating VLM conditions from LLama
    def _get_vlm_condition_llama(self, input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask, language_action_labels=None, append_meta_queries=True, return_last_hidden_state=False):
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
        if attention_mask is None:
            attention_mask = torch.ones(B, input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        if language_action_labels is not None:
            language_action_labels = language_action_labels.to(device=input_ids.device)

        proprio_embeds = None
        if self.use_proprio_input_vlm and proprioception is not None:
             proprio_embeds = self.action_projector(proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype))

        embeds_list = [image_embeds]
        image_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list = [image_mask]
        label_parts = []
        if language_action_labels is not None:
            label_parts.append(self._ignore_label_block(language_action_labels, image_embeds.shape[1]))

        if proprio_embeds is not None:
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                p_mask = torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                mask_list.append(p_mask)
            if language_action_labels is not None:
                label_parts.append(self._ignore_label_block(language_action_labels, proprio_embeds.shape[1]))
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)
        if language_action_labels is not None:
            label_parts.append(language_action_labels)

        if append_meta_queries and self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(text_embeds.dtype)
            embeds_list.append(queries_embeds)
            queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            mask_list.append(queries_mask)
            if language_action_labels is not None:
                label_parts.append(self._ignore_label_block(language_action_labels, self.num_queries))

        inputs_embeds = torch.cat(embeds_list, dim=1)
        combined_attention_mask = torch.cat(mask_list, dim=1)
        if language_action_labels is not None:
            language_action_labels = torch.cat(label_parts, dim=1)

        output_hidden_states_flag = append_meta_queries and (
            self.enable_future_image_loss or self.condition_type in ["tight", "soft"]
        )
        outputs = self.lmm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=output_hidden_states_flag
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        loss_language_action = self._compute_shared_language_action_loss(outputs.last_hidden_state, language_action_labels)
        if self.policy_gradient_stop_for_vlm and self.condition_type in ["tight", "soft"] and hidden_states is not None:
            hidden_states = tuple(state.detach() for state in hidden_states)
        connector_out = None
        if append_meta_queries and self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            if self.policy_gradient_stop_for_vlm:
                query_outputs = query_outputs.detach()
            connector_out = self.connector(query_outputs)
        result = (connector_out, hidden_states, loss_language_action)
        return result + (outputs.last_hidden_state,) if return_last_hidden_state else result

    # Utilities for generating VLM conditions from Paligemma
    def _get_vlm_condition_paligemma(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, token_type_ids=None, language_action_labels=None, append_meta_queries=True, return_last_hidden_state=False):
        B = input_ids.shape[0]

        backbone = self.lmm.model

        inputs_embeds = backbone.get_input_embeddings()(input_ids)
        if language_action_labels is not None:
            language_action_labels = language_action_labels.to(device=input_ids.device)
            token_type_ids = (language_action_labels != -100).to(device=input_ids.device, dtype=torch.long)

        if pixel_values is not None:
            image_outputs = backbone.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = backbone.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Prepend proprioception (prefix / bidirectional → token_type_ids = 0)
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            if token_type_ids is not None:
                proprio_type_ids = torch.zeros(B, proprioception.shape[1], dtype=token_type_ids.dtype, device=token_type_ids.device)
                token_type_ids = torch.cat([proprio_type_ids, token_type_ids], dim=1)
            if language_action_labels is not None:
                language_action_labels = torch.cat([self._ignore_label_block(language_action_labels, proprioception.shape[1]), language_action_labels], dim=1)

        # Append meta-queries (suffix / causal → token_type_ids = 1)
        if append_meta_queries and self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            if token_type_ids is not None:
                queries_type_ids = torch.ones(B, self.num_queries, dtype=token_type_ids.dtype, device=token_type_ids.device)
                token_type_ids = torch.cat([token_type_ids, queries_type_ids], dim=1)
            if language_action_labels is not None:
                language_action_labels = torch.cat([language_action_labels, self._ignore_label_block(language_action_labels, self.num_queries)], dim=1)

        # PaliGemma uses 1-indexed position_ids
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0) + 1

        # Build proper causal mask: bidirectional for prefix (token_type_ids==0),
        # causal for suffix (token_type_ids==1), via PaliGemma's mask creation.
        causal_mask_mapping = self.lmm.create_masks_for_generate(
            config=self.lmm.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            is_first_iteration=True,
        )

        output_hidden_states_flag = append_meta_queries and (
            self.enable_future_image_loss or self.condition_type in ["tight", "soft"]
        )
        outputs = backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states_flag,
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        loss_language_action = self._compute_shared_language_action_loss(outputs.last_hidden_state, language_action_labels)
        if self.policy_gradient_stop_for_vlm and self.condition_type in ["tight", "soft"] and hidden_states is not None:
            hidden_states = tuple(state.detach() for state in hidden_states)
        connector_out = None
        if append_meta_queries and self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            if self.policy_gradient_stop_for_vlm:
                query_outputs = query_outputs.detach()
            connector_out = self.connector(query_outputs)
        result = (connector_out, hidden_states, loss_language_action)
        return result + (outputs.last_hidden_state,) if return_last_hidden_state else result

    # Utilities for language action loss computation inside the backbone and conditions
    @staticmethod
    def _ignore_label_block(language_action_labels, length):
        if language_action_labels is None or length <= 0:
            return None
        return torch.full(
            (language_action_labels.shape[0], int(length)),
            -100,
            dtype=language_action_labels.dtype,
            device=language_action_labels.device,
        )

    def _compute_shared_language_action_loss(self, last_hidden_state, language_action_labels):
        if language_action_labels is None or not self.enable_language_action_loss:
            return None
        logits = self._project_lm_logits(last_hidden_state)
        return self._causal_lm_loss(logits, language_action_labels)

    def _project_lm_logits(self, hidden_states):
        output_embeddings = None
        if hasattr(self.lmm, "get_output_embeddings"):
            output_embeddings = self.lmm.get_output_embeddings()
        if output_embeddings is not None:
            return output_embeddings(hidden_states)
        if hasattr(self.lmm, "lm_head"):
            return self.lmm.lm_head(hidden_states)
        language_model = getattr(self.lmm, "language_model", None)
        if language_model is not None and hasattr(language_model, "lm_head"):
            return language_model.lm_head(hidden_states)
        if hasattr(self.lmm, "model") and hasattr(self.lmm.model, "lm_head"):
            return self.lmm.model.lm_head(hidden_states)
        raise AttributeError("Could not find an LM head for language-action loss.")

    @staticmethod
    def _causal_lm_loss(logits, language_action_labels):
        language_action_labels = language_action_labels.to(device=logits.device)
        shift_logits = logits[:, :-1, :]
        shift_labels = language_action_labels[:, 1:]
        valid = shift_labels != -100
        if not torch.any(valid):
            return logits.sum() * 0.0
        shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.reshape(-1)
        valid = valid.reshape(-1)
        return F.cross_entropy(shift_logits[valid].float(), shift_labels[valid])

    # -----------------------------------------------------------------------------
    # ----------------------------- Policy Loss  ----------------------------------
    # -----------------------------------------------------------------------------
    # Forward function, the main entry for the forward process and loss computation
    def forward(self, input_ids=None, attention_mask=None, actions=None, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, token_type_ids=None, language_action_labels=None, future_images=None, future_videos=None, prompt_texts=None, current_images=None, return_loss_components=False):
        if self.enable_language_action_loss and language_action_labels is None:
            raise ValueError("language_action_labels must be provided when language_action_loss_weight > 0.")
        if self.loss_type == "regression":
            return self._forward_regression(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids, token_type_ids,
                language_action_labels=language_action_labels,
                future_images=future_images,
                return_loss_components=return_loss_components,
            )
        elif self.loss_type == "classification":
            return self._forward_classification(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids, token_type_ids,
                language_action_labels=language_action_labels,
                future_images=future_images,
                return_loss_components=return_loss_components,
            )
        elif self.loss_type == "diffusion":
            return self._forward_diffusion(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids, token_type_ids,
                language_action_labels=language_action_labels,
                future_images=future_images,
                future_videos=future_videos,
                prompt_texts=prompt_texts,
                current_images=current_images,
                return_loss_components=return_loss_components,
            )

    def _format_loss_output(self, total_loss, loss_components, weighted_loss_components, return_loss_components):
        if not return_loss_components:
            return total_loss
        return {
            "loss": total_loss,
            "loss_components": {k: v for k, v in loss_components.items() if v is not None},
            "weighted_loss_components": {k: v for k, v in weighted_loss_components.items() if v is not None},
        }

    # Forward for classification loss
    def _forward_classification(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=None, token_type_ids=None, language_action_labels=None, future_images=None, return_loss_components=False):
        connector_out, hidden_states, loss_language_action = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, token_type_ids=token_type_ids,
            language_action_labels=language_action_labels,
        )

        loss_img = None
        gen_hidden_states = None
        if self.enable_future_image_loss:
            loss_img, gen_hidden_states = self._compute_future_image_loss_and_feats(future_images, hidden_states)

        policy_history = history_actions if self.use_action_input_policy else None
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)

        target_ids = None
        if self.classification_type == "autoregressive":
            target_ids = self._classification_target_ids(actions)

        pred_logits = self._run_classification_head(
            connector_out,
            hidden_states,
            policy_history,
            gen_hidden_states=gen_hidden_states,
            target_ids=target_ids,
        )
        action_loss, pred_action_continuous = self._classification_loss_and_prediction(
            pred_logits,
            actions,
            target_ids=target_ids,
        )
        total_loss = action_loss
        loss_components = {"action": action_loss}
        weighted_loss_components = {"action": action_loss}

        if self.dct_loss_weight > 0 and pred_action_continuous is not None:
            loss_dct = self._compute_dct_loss(pred_action_continuous.float(), actions.float())
            weighted_loss_dct = self.dct_loss_weight * loss_dct
            total_loss = total_loss + weighted_loss_dct
            loss_components["dct"] = loss_dct
            weighted_loss_components["dct"] = weighted_loss_dct

        if loss_img is not None:
            weighted_loss_img = self.future_image_loss_weight * loss_img
            total_loss = total_loss + weighted_loss_img
            loss_components["future_image"] = loss_img
            weighted_loss_components["future_image"] = weighted_loss_img

        if loss_language_action is not None:
            weighted_loss_language_action = self.language_action_loss_weight * loss_language_action
            total_loss = total_loss + weighted_loss_language_action
            loss_components["language_action"] = loss_language_action
            weighted_loss_components["language_action"] = weighted_loss_language_action

        return self._format_loss_output(
            total_loss,
            loss_components,
            weighted_loss_components,
            return_loss_components,
        )

    # Utils for classification target construction
    def _tokenize_actions_fast(self, actions):
        """Convert (B, T, D) actions to padded FAST BPE token IDs plus EOS."""
        B, T, D = actions.shape
        actions_np = actions.detach().float().cpu().numpy()
        token_sequences = self.fast_tokenizer(actions_np)  # list[list[int]], len=B
        B = actions_np.shape[0]
        max_len = self.fast_expected_seq_len
        if max_len < 1:
            raise ValueError(f"fast_expected_seq_len must be >= 1, got {max_len}")
        padded = torch.full((B, max_len), self.fast_pad_id, dtype=torch.long, device=actions.device)
        for i, seq in enumerate(token_sequences):
            # Reserve one slot for EOS so inference can ignore predictions after it.
            seq_len = min(len(seq), max_len - 1)
            if seq_len > 0:
                padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long, device=actions.device)
            padded[i, seq_len] = self.fast_eos_id
        return padded

    # Shared utils for classification training and inference
    def _run_classification_head(
        self,
        connector_out,
        hidden_states,
        policy_history,
        gen_hidden_states=None,
        target_ids=None,
        generated_ids=None,
    ):
        ar_kwargs = {}
        if self.classification_type == "autoregressive":
            ar_kwargs["target_ids"] = target_ids
            ar_kwargs["generated_ids"] = generated_ids

        if self.condition_type in ["tight", "soft"]:
            return self.action_head(
                hidden_states,
                history_actions=policy_history,
                gen_hidden_states=gen_hidden_states,
                **ar_kwargs,
            )
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            return self.action_head(cond_input, history_actions=policy_history, **ar_kwargs)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")

    def _classification_target_ids(self, actions):
        if self.fast_tokenizer is not None:
            return self._tokenize_actions_fast(actions)

        pose_dim = self.action_dim - 1
        gt_pose = torch.clamp(actions[:, :, :pose_dim], -1, 1)
        gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()

        gt_gripper = torch.clamp(actions[:, :, pose_dim:pose_dim + 1], -1, 1)
        gt_gripper_idx = ((gt_gripper + 1) / 2).round().long()

        return torch.cat([gt_pose_idx, gt_gripper_idx], dim=-1).reshape(actions.shape[0], -1)

    def _classification_loss_and_prediction(self, pred_logits, actions, target_ids=None):
        if self.fast_tokenizer is not None:
            if target_ids is None:
                target_ids = self._tokenize_actions_fast(actions)
            loss = F.cross_entropy(
                pred_logits.reshape(-1, self.fast_vocab_size),
                target_ids.reshape(-1),
                ignore_index=self.fast_pad_id,
            )
            return loss, None

        logits = pred_logits
        if self.classification_type == "autoregressive":
            logits = logits.reshape(actions.shape[0], actions.shape[1], self.action_dim, self.num_bins)

        pose_dim = self.action_dim - 1
        pose_logits = logits[:, :, :pose_dim, :]
        gripper_logits = logits[:, :, pose_dim:pose_dim + 1, :2]

        gt_pose = torch.clamp(actions[:, :, :pose_dim], -1, 1)
        gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()

        gt_gripper = torch.clamp(actions[:, :, pose_dim:pose_dim + 1], -1, 1)
        gt_gripper_idx = ((gt_gripper + 1) / 2).round().long()

        loss_pose = F.cross_entropy(pose_logits.reshape(-1, self.num_bins), gt_pose_idx.reshape(-1))
        loss_gripper = F.cross_entropy(gripper_logits.reshape(-1, 2), gt_gripper_idx.reshape(-1))
        loss = (loss_pose + loss_gripper) / 2.0

        pred_action_continuous = None
        if self.dct_loss_weight > 0:
            pose_probs = F.softmax(pose_logits, dim=-1)
            bin_centers = torch.linspace(-1, 1, self.num_bins, device=actions.device, dtype=pose_probs.dtype)
            pred_pose = torch.sum(pose_probs * bin_centers, dim=-1)

            gripper_probs = F.softmax(gripper_logits, dim=-1)
            pred_gripper = -1.0 + 2.0 * gripper_probs[..., 1]
            pred_action_continuous = torch.cat([pred_pose, pred_gripper], dim=-1)

        return loss, pred_action_continuous

    # forward for regression loss
    def _forward_regression(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=None, token_type_ids=None, language_action_labels=None, future_images=None, return_loss_components=False):
        connector_out, hidden_states, loss_language_action = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, token_type_ids=token_type_ids,
            language_action_labels=language_action_labels,
        )

        loss_img = None
        gen_hidden_states = None
        if self.enable_future_image_loss:
            loss_img, gen_hidden_states = self._compute_future_image_loss_and_feats(future_images, hidden_states)

        policy_history = history_actions if self.use_action_input_policy else None

        if self.condition_type in ["tight", "soft"]:
             pred_actions = self.action_head(
                 hidden_states,
                 history_actions=policy_history,
                 gen_hidden_states=gen_hidden_states,
             )
        elif self.condition_type == "loose":
             cond_input = connector_out.mean(dim=1)
             pred_actions = self.action_head(cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")

        if actions.ndim == 2: actions = actions.unsqueeze(1)
        action_loss = F.mse_loss(pred_actions, actions)
        total_loss = action_loss
        loss_components = {"action": action_loss}
        weighted_loss_components = {"action": action_loss}

        if self.dct_loss_weight > 0:
            loss_dct = self._compute_dct_loss(pred_actions.float(), actions.float())
            weighted_loss_dct = self.dct_loss_weight * loss_dct
            total_loss = total_loss + weighted_loss_dct
            loss_components["dct"] = loss_dct
            weighted_loss_components["dct"] = weighted_loss_dct

        if loss_img is not None:
            weighted_loss_img = self.future_image_loss_weight * loss_img
            total_loss = total_loss + weighted_loss_img
            loss_components["future_image"] = loss_img
            weighted_loss_components["future_image"] = weighted_loss_img

        if loss_language_action is not None:
            weighted_loss_language_action = self.language_action_loss_weight * loss_language_action
            total_loss = total_loss + weighted_loss_language_action
            loss_components["language_action"] = loss_language_action
            weighted_loss_components["language_action"] = weighted_loss_language_action

        return self._format_loss_output(
            total_loss,
            loss_components,
            weighted_loss_components,
            return_loss_components,
        )

    # forward for diffusion loss
    def _forward_diffusion(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=None, token_type_ids=None, language_action_labels=None, future_images=None, future_videos=None, prompt_texts=None, current_images=None, return_loss_components=False):
        if self.model_family == "wan":
            return self._forward_video_generation_diffusion(
                actions=actions,
                proprioception=proprioception,
                history_actions=history_actions,
                future_videos=future_videos,
                prompt_texts=prompt_texts,
                current_images=current_images,
                return_loss_components=return_loss_components,
            )

        connector_out, hidden_states, loss_language_action = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, token_type_ids=token_type_ids,
            language_action_labels=language_action_labels,
        )

        loss_img = None
        gen_hidden_states = None
        if self.enable_future_image_loss:
            loss_img, gen_hidden_states = self._compute_future_image_loss_and_feats(future_images, hidden_states)

        if actions.ndim == 2: actions = actions.unsqueeze(1)
        noise = torch.randn_like(actions)
        B = actions.shape[0]
        
        if self.scheduler_type == "flow_match":
            sigmas = torch.rand((B,), device=actions.device)
            sigmas_expanded = sigmas.view(B, *([1] * (actions.ndim - 1)))
            noisy_actions = (1.0 - sigmas_expanded) * actions + sigmas_expanded * noise
            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            timesteps = sigmas * self.noise_scheduler.config.num_train_timesteps
            target = actions if self.diffusion_loss_domain == "x0" else (noise - actions)
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            target = actions if self.diffusion_loss_domain == "x0" else noise
            
        policy_history = history_actions if self.use_action_input_policy else None

        if self.condition_type in ["tight", "soft"]:
            pred = self.action_head(
                noisy_actions,
                timesteps,
                hidden_states,
                history_actions=policy_history,
                gen_hidden_states=gen_hidden_states,
            )
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred = self.action_head(noisy_actions, timesteps, cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        action_loss = F.mse_loss(pred, target)
        total_loss = action_loss
        loss_components = {"action": action_loss}
        weighted_loss_components = {"action": action_loss}

        if self.dct_loss_weight > 0:
            pred_x_start = None
            if self.diffusion_loss_domain == "x0":
                pred_x_start = pred
            else:
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
                 weighted_loss_dct = self.dct_loss_weight * loss_dct
                 total_loss = total_loss + weighted_loss_dct
                 loss_components["dct"] = loss_dct
                 weighted_loss_components["dct"] = weighted_loss_dct

        if loss_img is not None:
            weighted_loss_img = self.future_image_loss_weight * loss_img
            total_loss = total_loss + weighted_loss_img
            loss_components["future_image"] = loss_img
            weighted_loss_components["future_image"] = weighted_loss_img

        if loss_language_action is not None:
            weighted_loss_language_action = self.language_action_loss_weight * loss_language_action
            total_loss = total_loss + weighted_loss_language_action
            loss_components["language_action"] = loss_language_action
            weighted_loss_components["language_action"] = weighted_loss_language_action

        return self._format_loss_output(
            total_loss,
            loss_components,
            weighted_loss_components,
            return_loss_components,
        )
    
    # Utils for computing the diffusion loss for video generation together with action generation
    def _forward_video_generation_diffusion(self, actions, proprioception=None, history_actions=None, future_videos=None, prompt_texts=None, current_images=None, return_loss_components=False):
        if actions is None:
            raise ValueError("actions must be provided for WAN diffusion training.")
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions.to(device=self.lmm.device, dtype=self.lmm.dtype)
        batch_size = actions.shape[0]
        timesteps, sigmas = self._sample_wan_flow_timesteps(batch_size, actions.device, actions.dtype)
        action_noise = torch.randn_like(actions)
        noisy_actions = self._wan_add_flow_noise(actions, action_noise, sigmas)
        action_target = actions if self.diffusion_loss_domain == "x0" else (action_noise - actions)

        loss_video, video_hidden_states, noisy_video_latents = self._compute_video_generation_loss(
            future_videos=future_videos,
            timesteps=timesteps,
            sigmas=sigmas,
            prompt_texts=prompt_texts,
            proprioception=proprioception,
            return_hidden_states=True,
        )
        hidden_states = self._select_wan_policy_hidden_states(video_hidden_states, noisy_video_latents)

        policy_history = history_actions.to(device=actions.device, dtype=actions.dtype) if (self.use_action_input_policy and history_actions is not None) else None
        pred = self.action_head(
            noisy_actions,
            timesteps,
            hidden_states,
            history_actions=policy_history,
        )
        action_loss = F.mse_loss(pred.float(), action_target.float())
        total_loss = action_loss
        loss_components = {"action": action_loss}
        weighted_loss_components = {"action": action_loss}

        if self.dct_loss_weight > 0:
            if self.diffusion_loss_domain == "x0":
                pred_x_start = pred
            else:
                sigmas_expanded = sigmas.view(batch_size, *([1] * (pred.ndim - 1)))
                pred_x_start = noisy_actions - sigmas_expanded * pred
            loss_dct = self._compute_dct_loss(pred_x_start.float(), actions.float())
            weighted_loss_dct = self.dct_loss_weight * loss_dct
            total_loss = total_loss + weighted_loss_dct
            loss_components["dct"] = loss_dct
            weighted_loss_components["dct"] = weighted_loss_dct

        weighted_loss_video = self.video_generation_loss_weight * loss_video
        total_loss = total_loss + weighted_loss_video
        loss_components["video_generation"] = loss_video
        weighted_loss_components["video_generation"] = weighted_loss_video

        return self._format_loss_output(
            total_loss,
            loss_components,
            weighted_loss_components,
            return_loss_components,
        )
    
    # Utils for computing future image loss inside the policy loss computation
    def _compute_future_image_loss_and_feats(self, future_images, vlm_hidden_states):
        if self.future_image_prediction_type == "emu_token":
            return self._compute_emu_token_loss_and_feats(future_images, vlm_hidden_states)
        if self.future_image_prediction_type == "dinov3_flow":
            return self._compute_dino_flow_loss_and_feats(future_images, vlm_hidden_states)
        raise ValueError(f"Unknown future_image_prediction_type: {self.future_image_prediction_type}")

    def _compute_emu_token_loss_and_feats(self, future_images, vlm_hidden_states):
        if future_images is None:
            raise ValueError("future_images must be provided when future_image_loss_weight > 0.")
        if vlm_hidden_states is None:
            raise ValueError("VLM hidden states are required for future image generation.")

        with torch.no_grad():
            self.vq_model.eval()
            future_images = future_images.to(device=self.vq_model.device, dtype=self.vq_model.dtype)
            _, _, (_, _, token_ids) = self.vq_model.encode(future_images)
            B = future_images.shape[0]
            token_ids = token_ids.view(B, -1).long()

        if token_ids.shape[1] > self.generator.pos_embed.shape[1]:
            raise ValueError(
                f"Future image token length {token_ids.shape[1]} exceeds "
                f"generator max_seq_len {self.generator.pos_embed.shape[1]}."
            )

        sos_token = torch.zeros((token_ids.shape[0], 1), dtype=token_ids.dtype, device=token_ids.device)
        gen_input = torch.cat([sos_token, token_ids[:, :-1]], dim=1)
        gen_logits, gen_hidden_states = self.generator(gen_input, vlm_hidden_states)
        loss_img = F.cross_entropy(gen_logits.reshape(-1, self.vq_codebook_size), token_ids.reshape(-1))

        return loss_img, gen_hidden_states

    def _compute_dino_flow_loss_and_feats(self, future_images, vlm_hidden_states):
        if vlm_hidden_states is None:
            raise ValueError("VLM hidden states are required for future image generation.")

        with torch.no_grad():
            target_features = self._extract_dino_patch_features(future_images)

        if target_features.shape[1] > self.generator.pos_embed.shape[1]:
            raise ValueError(
                f"DINO feature token length {target_features.shape[1]} exceeds "
                f"generator max_seq_len {self.generator.pos_embed.shape[1]}."
            )

        B = target_features.shape[0]
        noise = torch.randn_like(target_features)
        sigmas = torch.rand((B,), device=target_features.device, dtype=target_features.dtype)
        sigmas_expanded = sigmas.view(B, 1, 1)
        noisy_features = (1.0 - sigmas_expanded) * target_features + sigmas_expanded * noise
        timesteps = sigmas.float() * self.future_image_flow_scheduler.config.num_train_timesteps
        target_velocity = noise - target_features

        pred_velocity, gen_hidden_states = self.generator(noisy_features, timesteps, vlm_hidden_states)
        loss_img = F.mse_loss(pred_velocity.float(), target_velocity.float())
        return loss_img, gen_hidden_states
    
    def _extract_dino_patch_features(self, future_images):
        if future_images is None:
            raise ValueError("future_images must be provided when future_image_loss_weight > 0.")
        if self.dino_model is None:
            raise ValueError("DINOv3 model is not initialized.")

        self.dino_model.eval()
        dino_param = next(self.dino_model.parameters())
        future_images = future_images.to(device=dino_param.device, dtype=dino_param.dtype)
        outputs = self.dino_model(pixel_values=future_images)
        features = outputs.last_hidden_state
        if features.ndim != 3:
            raise ValueError(
                "DINOv3 flow mode requires ViT-style token features from last_hidden_state; "
                f"got shape {tuple(features.shape)}."
            )
        patch_start = 1 + self.dino_num_register_tokens
        if features.shape[1] <= patch_start:
            raise ValueError(
                f"DINOv3 output has too few tokens ({features.shape[1]}) for "
                f"1 CLS + {self.dino_num_register_tokens} register tokens."
            )
        return features[:, patch_start:, :]

    # Utils for computing the dct loss inside the policy loss computation
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

    # -----------------------------------------------------------------------------
    # ------------------------------- Inference -----------------------------------
    # -----------------------------------------------------------------------------
    # Generate actions in inference
    @torch.no_grad()
    def predict_action(self, input_ids=None, attention_mask=None, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, token_type_ids=None, return_image=False, image_max_new_tokens=None, generate_language_action=False, language_action_max_new_tokens=256, prompt_texts=None, current_images=None, return_video=False):
        if self.model_family == "wan":
            return self.predict_video_and_action(
                prompt_texts=prompt_texts,
                current_images=current_images,
                proprioception=proprioception,
                history_actions=history_actions,
                return_video=return_video,
            )
        
        if generate_language_action:
            input_ids, attention_mask, token_type_ids, mm_token_type_ids = self._generate_language_action_inputs(
                input_ids,
                attention_mask,
                proprioception=proprioception,
                proprio_attention_mask=proprio_attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                token_type_ids=token_type_ids,
                max_new_tokens=language_action_max_new_tokens,
            )
        B = input_ids.shape[0]

        connector_out, hidden_states, _ = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            token_type_ids=token_type_ids
        )

        policy_history = history_actions if self.use_action_input_policy else None
        generated_image_tokens = None
        gen_hidden_states = None
        if self.enable_future_image_loss and (self.condition_type in ["tight", "soft"] or return_image):
            if return_image and self.future_image_prediction_type != "emu_token":
                raise ValueError("return_image=True is only supported for future_image_prediction_type='emu_token'.")
            if self.future_image_prediction_type == "emu_token":
                generated_image_tokens, generated_gen_hidden_states = self._generate_future_image_tokens_and_hidden_states_emu(
                    B,
                    hidden_states,
                    input_ids.device,
                    max_new_tokens=image_max_new_tokens,
                )
            elif self.future_image_prediction_type == "dinov3_flow":
                _, generated_gen_hidden_states = self._generate_future_image_features_and_hidden_states_dino(
                    B,
                    hidden_states,
                    input_ids.device,
                )
            else:
                raise ValueError(f"Unknown future_image_prediction_type: {self.future_image_prediction_type}")
            if self.condition_type in ["tight", "soft"]:
                gen_hidden_states = generated_gen_hidden_states
        elif return_image:
            raise ValueError("return_image=True requires future_image_loss_weight > 0.")

        decoded_images = None
        if return_image:
            decoded_images = self._decode_future_image_tokens(generated_image_tokens)

        def pack_action(action):
            action = action.to(dtype=self.lmm.dtype)
            return (action, decoded_images) if return_image else action

        if self.loss_type == "regression":
            if self.condition_type in ["tight", "soft"]:
                action = self.action_head(
                    hidden_states,
                    history_actions=policy_history,
                    gen_hidden_states=gen_hidden_states,
                )
            elif self.condition_type == "loose":
                cond_input = connector_out.mean(dim=1)
                action = self.action_head(cond_input, history_actions=policy_history)
            if action.ndim == 2 and self.num_actions > 1:
                action = action.view(action.shape[0], self.num_actions, self.action_dim)
            
            return pack_action(action)

        elif self.loss_type == "classification":
            if self.classification_type == "autoregressive":
                pred_ids = self._generate_autoregressive_classification_ids(
                    B,
                    connector_out,
                    hidden_states,
                    policy_history,
                    gen_hidden_states,
                    input_ids.device,
                )
                if self.fast_tokenizer is not None:
                    action = self._decode_fast_action_ids(pred_ids, B, input_ids.device, self.lmm.dtype)
                else:
                    action = self._decode_bin_action_ids(pred_ids, self.lmm.dtype)
                return pack_action(action)

            logits = self._run_classification_head(
                connector_out,
                hidden_states,
                policy_history,
                gen_hidden_states=gen_hidden_states,
            )

            if self.fast_tokenizer is not None:
                pred_ids = logits.argmax(dim=-1)
                action = self._decode_fast_action_ids(pred_ids, B, input_ids.device, self.lmm.dtype)
                return pack_action(action)

            pose_dim = self.action_dim - 1
            pose_logits = logits[:, :, :pose_dim, :]
            gripper_logits = logits[:, :, pose_dim:pose_dim + 1, :2]
            pose_idx = torch.argmax(pose_logits, dim=-1)
            gripper_idx = torch.argmax(gripper_logits, dim=-1)
            pred_ids = torch.cat([pose_idx, gripper_idx], dim=-1).reshape(B, -1)
            action = self._decode_bin_action_ids(pred_ids, self.lmm.dtype)
            return pack_action(action)

        elif self.loss_type == "diffusion":
            action = torch.randn(B, self.num_actions, self.action_dim, device=input_ids.device).to(self.lmm.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            
            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), t, device=input_ids.device)
                if self.scheduler_type != "flow_match": timesteps = timesteps.long()
                if self.condition_type in ["tight", "soft"]:
                    output = self.action_head(
                        action,
                        timesteps,
                        hidden_states,
                        history_actions=policy_history,
                        gen_hidden_states=gen_hidden_states,
                    )
                else:
                    cond_input = connector_out.mean(dim=1)
                    output = self.action_head(action, timesteps, cond_input, history_actions=policy_history)

                # FlowMatchEulerDiscreteScheduler only consumes velocity, so convert x0→velocity here.
                # DDIM in x0 mode is handled by prediction_type="sample" at scheduler construction.
                if self.scheduler_type == "flow_match" and self.diffusion_loss_domain == "x0":
                    sigma_t = max(float(t) / self.noise_scheduler.config.num_train_timesteps, 1e-6)
                    output = (action - output) / sigma_t

                action = self.noise_scheduler.step(output, t, action).prev_sample
                action = action.to(dtype=self.lmm.dtype)
            
            return pack_action(action)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    # Utils for classification inference
    def _generate_autoregressive_classification_ids(self, B, connector_out, hidden_states, policy_history, gen_hidden_states, device):
        max_len = self.fast_expected_seq_len if self.fast_tokenizer is not None else self.num_actions * self.action_dim
        generated_ids = torch.empty(B, 0, device=device, dtype=torch.long)
        finished = torch.zeros(B, device=device, dtype=torch.bool)

        for step in range(max_len):
            logits = self._run_classification_head(
                connector_out,
                hidden_states,
                policy_history,
                gen_hidden_states=gen_hidden_states,
                generated_ids=generated_ids,
            )
            next_logits = logits[:, -1, :]
            if self.fast_tokenizer is not None:
                next_ids = next_logits.argmax(dim=-1)
                next_ids = torch.where(finished, torch.full_like(next_ids, self.fast_eos_id), next_ids)
                finished = finished | (next_ids == self.fast_eos_id)
            else:
                dim_idx = step % self.action_dim
                if dim_idx == self.action_dim - 1:
                    next_ids = next_logits[:, :2].argmax(dim=-1)
                else:
                    next_ids = next_logits.argmax(dim=-1)

            generated_ids = torch.cat([generated_ids, next_ids.unsqueeze(1)], dim=1)
            if self.fast_tokenizer is not None and torch.all(finished):
                break

        return generated_ids

    def _decode_fast_action_ids(self, pred_ids, B, device, dtype):
        pred_ids_list = []
        for seq in pred_ids.cpu().tolist():
            action_tokens = []
            for token_id in seq:
                if token_id == self.fast_eos_id:
                    break
                if token_id < self.fast_eos_id:
                    action_tokens.append(token_id)
            pred_ids_list.append(action_tokens)
        try:
            actions_np = self.fast_tokenizer.decode(
                pred_ids_list, time_horizon=self.num_actions, action_dim=self.action_dim
            )
            return torch.from_numpy(actions_np).to(device=device, dtype=dtype)
        except Exception as e:
            print(f"FAST decode failed: {e}. Returning zeros.")
            return torch.zeros(B, self.num_actions, self.action_dim, device=device, dtype=dtype)

    def _decode_bin_action_ids(self, pred_ids, dtype):
        pred_ids = pred_ids.reshape(pred_ids.shape[0], self.num_actions, self.action_dim)
        pose_dim = self.action_dim - 1
        pose_idx = pred_ids[:, :, :pose_dim].clamp(0, self.num_bins - 1)
        gripper_idx = pred_ids[:, :, pose_dim:pose_dim + 1].clamp(0, 1)
        pose_pred = (pose_idx.float() / (self.num_bins - 1)) * 2 - 1
        gripper_pred = gripper_idx.float() * 2 - 1
        return torch.cat([pose_pred, gripper_pred], dim=-1).to(dtype=dtype)

    # Utils for language action mode to generate inputs for inference
    @torch.no_grad()
    def _generate_language_action_inputs(
        self,
        input_ids,
        attention_mask,
        proprioception=None,
        proprio_attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        token_type_ids=None,
        max_new_tokens=256,
    ):
        max_new_tokens = int(max_new_tokens or 0)
        if max_new_tokens <= 0:
            raise ValueError(f"language_action_max_new_tokens must be positive, got {max_new_tokens}.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.model_family == "paligemma":
            token_type_ids = torch.zeros_like(input_ids) if token_type_ids is None else torch.zeros_like(token_type_ids)

        eos_ids = self._get_eos_token_ids()
        eos_tensor = None
        eos_fill = None
        if eos_ids:
            eos_values = sorted(eos_ids)
            eos_tensor = torch.tensor(eos_values, device=input_ids.device, dtype=input_ids.dtype)
            eos_fill = torch.full((input_ids.shape[0], 1), eos_values[0], device=input_ids.device, dtype=input_ids.dtype)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            _, _, _, last_hidden_state = self.get_vlm_condition(
                input_ids,
                attention_mask,
                proprioception=proprioception,
                proprio_attention_mask=proprio_attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                token_type_ids=token_type_ids,
                append_meta_queries=False,
                return_last_hidden_state=True,
            )
            if attention_mask is None:
                last_indices = torch.full(
                    (input_ids.shape[0],),
                    last_hidden_state.shape[1] - 1,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            else:
                last_indices = attention_mask.to(torch.long).flip(dims=[1]).argmax(dim=1)
                last_indices = attention_mask.shape[1] - 1 - last_indices
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            logits = self._project_lm_logits(last_hidden_state[batch_indices, last_indices].unsqueeze(1))[:, 0, :]
            next_token = torch.argmax(logits.float(), dim=-1, keepdim=True)
            if eos_tensor is not None:
                next_token = torch.where(finished[:, None], eos_fill, next_token)
                just_finished = (next_token == eos_tensor.view(1, -1)).any(dim=1)
                finished = finished | just_finished

            input_ids, attention_mask, token_type_ids, mm_token_type_ids = self._append_text_token(
                input_ids,
                attention_mask,
                next_token,
                token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids,
            )
            if finished.all().item():
                break

        return input_ids, attention_mask, token_type_ids, mm_token_type_ids

    def _get_eos_token_ids(self):
        eos = getattr(self.lmm.config, "eos_token_id", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if eos is None and tokenizer is not None:
            eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(x) for x in eos if x is not None}
        return {int(eos)}

    def _append_text_token(self, input_ids, attention_mask, next_token, token_type_ids=None, mm_token_type_ids=None):
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if attention_mask is not None:
            token_mask = torch.ones(
                next_token.shape,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, token_mask], dim=1)
        if token_type_ids is not None:
            type_value = 1 if self.model_family == "paligemma" else 0
            next_type = torch.full(
                next_token.shape,
                type_value,
                dtype=token_type_ids.dtype,
                device=token_type_ids.device,
            )
            token_type_ids = torch.cat([token_type_ids, next_type], dim=1)
        if mm_token_type_ids is not None:
            next_mm_type = torch.zeros(
                next_token.shape,
                dtype=mm_token_type_ids.dtype,
                device=mm_token_type_ids.device,
            )
            mm_token_type_ids = torch.cat([mm_token_type_ids, next_mm_type], dim=1)
        return input_ids, attention_mask, token_type_ids, mm_token_type_ids

    # utils for future image loss mode to generate conditions to policy for inference
    def _generate_future_image_tokens_and_hidden_states_emu(self, batch_size, vlm_hidden_states, device, max_new_tokens=None):
        if not self.enable_future_image_loss:
            raise ValueError("Future image generation requires future_image_loss_weight > 0.")
        if self.future_image_prediction_type != "emu_token":
            raise ValueError("Token generation is only available for future_image_prediction_type='emu_token'.")
        if vlm_hidden_states is None:
            raise ValueError("VLM hidden states are required for future image generation.")

        if max_new_tokens is None:
            max_new_tokens = self.future_image_num_tokens
        num_img_tokens = min(int(max_new_tokens), self.generator.pos_embed.shape[1])
        if num_img_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")

        curr_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        for _ in range(num_img_tokens):
            logits, _ = self.generator(curr_ids, vlm_hidden_states)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        generated_tokens = curr_ids[:, 1:]
        _, gen_hidden_states = self.generator(curr_ids[:, :-1], vlm_hidden_states)
        return generated_tokens, gen_hidden_states

    def _generate_future_image_features_and_hidden_states_dino(self, batch_size, vlm_hidden_states, device):
        if not self.enable_future_image_loss:
            raise ValueError("Future image generation requires future_image_loss_weight > 0.")
        if self.future_image_prediction_type != "dinov3_flow":
            raise ValueError("Feature generation is only available for future_image_prediction_type='dinov3_flow'.")
        if vlm_hidden_states is None:
            raise ValueError("VLM hidden states are required for future image generation.")

        image_size = self.future_image_dino_image_size
        patch_size = int(getattr(self.dino_model.config, "patch_size", 16) or 16)
        num_tokens = (image_size // patch_size) * (image_size // patch_size)
        feature_dim = self.generator.head.out_features
        features = torch.randn(batch_size, num_tokens, feature_dim, device=device, dtype=self.lmm.dtype)

        self.future_image_flow_scheduler.set_timesteps(
            self.future_image_flow_num_inference_timesteps,
            device=device,
        )
        gen_hidden_states = None
        for t in self.future_image_flow_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            pred_velocity, gen_hidden_states = self.generator(features, timesteps, vlm_hidden_states)
            features = self.future_image_flow_scheduler.step(pred_velocity, t, features).prev_sample
            features = features.to(dtype=self.lmm.dtype)

        return features, gen_hidden_states
    
    # generate videos and actions in inference, only supported when using video generation backbone 
    @torch.no_grad()
    def predict_video_and_action(
        self,
        prompt_texts=None,
        current_images=None,
        proprioception=None,
        history_actions=None,
        return_video=False,
    ):
        if self.model_family != "wan":
            raise ValueError("predict_video_and_action is only available for WAN video-generation backbone.")
        if self.loss_type != "diffusion" or self.scheduler_type != "flow_match":
            raise ValueError("WAN video/action inference requires diffusion loss with flow_match scheduler.")
        if current_images is None:
            raise ValueError("current_images must be provided for WAN inference.")
        if current_images.ndim == 5 and current_images.shape[2] == 1:
            current_images = current_images[:, :, 0]
        if current_images.ndim != 4:
            raise ValueError(f"current_images must be [B, C, H, W], got {tuple(current_images.shape)}.")
        if prompt_texts is None:
            raise ValueError("prompt_texts must be provided for WAN inference.")

        device = self.lmm.device
        dtype = self.lmm.dtype
        current_images = current_images.to(device=device, dtype=dtype)
        batch_size, _, height, width = current_images.shape
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts] * batch_size
        if len(prompt_texts) != batch_size:
            raise ValueError(f"prompt_texts batch size {len(prompt_texts)} does not match images batch size {batch_size}.")

        policy_history = None
        if self.use_action_input_policy and history_actions is not None:
            policy_history = history_actions.to(device=device, dtype=dtype)

        action = torch.randn(batch_size, self.num_actions, self.action_dim, device=device, dtype=dtype)
        sigmas = self.lmm.shifted_sigma(
            torch.linspace(1.0, 0.0, self.num_inference_timesteps + 1, device=device, dtype=torch.float32),
            self.wan_flow_shift,
        ).to(dtype=dtype)
        timestep_values = sigmas[:-1].float() * float(self.num_train_timesteps)
        deltas = sigmas[1:] - sigmas[:-1]

        if self.wan_action_condition_mode == "fast":
            if return_video:
                raise ValueError("return_video=True requires wan_action_condition_mode='joint'.")
            _, hidden_states, _ = self._get_vlm_condition_wam(
                prompt_texts=prompt_texts,
                current_images=current_images,
                proprioception=proprioception,
            )
            for timestep, delta, sigma in zip(timestep_values, deltas, sigmas[:-1]):
                timesteps = torch.full((batch_size,), timestep, device=device, dtype=dtype)
                output = self.action_head(
                    action,
                    timesteps,
                    hidden_states,
                    history_actions=policy_history,
                )
                if self.diffusion_loss_domain == "x0":
                    output = (action - output) / torch.clamp(sigma, min=1e-6)
                action = (action + output * delta).to(dtype=dtype)
            return action

        if self.wan_action_condition_mode != "joint":
            raise ValueError(f"Unknown WAN action condition mode: {self.wan_action_condition_mode}")

        num_video_frames = self.num_actions // self.future_video_downsample + 1
        if self.num_actions % self.future_video_downsample != 0:
            raise ValueError("num_actions must be divisible by future_video_downsample for WAN inference.")
        if num_video_frames % 4 != 1:
            raise ValueError(f"WAN generated video length must satisfy T % 4 == 1, got {num_video_frames}.")

        first_latent = self.lmm.encode_videos(current_images.unsqueeze(2))
        latent_shape = self.lmm.latent_shape(batch_size, num_video_frames, height, width, device=device, dtype=dtype)
        video_latents = torch.randn(latent_shape, device=device, dtype=dtype)
        video_latents[:, :, :1] = first_latent[:, :, :1]
        context = self._wan_context(prompt_texts, proprioception=proprioception)

        for timestep, delta, sigma in zip(timestep_values, deltas, sigmas[:-1]):
            timesteps = torch.full((batch_size,), timestep, device=device, dtype=dtype)
            video_timesteps = self._wan_first_frame_token_timesteps(video_latents, timesteps)
            pred_video, hidden_states = self.lmm.forward_latents(
                video_latents,
                video_timesteps,
                context,
                return_hidden_states=True,
            )
            pred_action = self.action_head(
                action,
                timesteps,
                hidden_states,
                history_actions=policy_history,
            )
            if self.diffusion_loss_domain == "x0":
                pred_action = (action - pred_action) / torch.clamp(sigma, min=1e-6)
            video_latents = (video_latents + pred_video * delta).to(dtype=dtype)
            video_latents[:, :, :1] = first_latent[:, :, :1]
            action = (action + pred_action * delta).to(dtype=dtype)

        if return_video:
            video = self.lmm.decode_latents(video_latents)
            return {"action": action, "video": video}
        return action

    # generate images in inference, only supported when enable_future_image_loss 
    @torch.no_grad()
    def predict_image(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, token_type_ids=None, max_new_tokens=None, return_tokens=False):
        if not self.enable_future_image_loss:
            raise ValueError("predict_image requires future_image_loss_weight > 0.")
        if self.future_image_prediction_type != "emu_token":
            raise ValueError("predict_image is only supported for future_image_prediction_type='emu_token'.")

        _, hidden_states, _ = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            token_type_ids=token_type_ids
        )
        generated_tokens, _ = self._generate_future_image_tokens_and_hidden_states_emu(
            input_ids.shape[0],
            hidden_states,
            input_ids.device,
            max_new_tokens=max_new_tokens,
        )
        decoded_images = self._decode_future_image_tokens(generated_tokens)
        return (decoded_images, generated_tokens) if return_tokens else decoded_images

    # utils for generate future image tokens in infernece
    def _decode_future_image_tokens(self, generated_tokens):
        num_tokens = generated_tokens.shape[1]
        latent_h = int(num_tokens ** 0.5)
        if latent_h * latent_h != num_tokens:
            raise ValueError(
                f"Generated image token count must be a square number, got {num_tokens}."
            )
        self.vq_model.eval()
        return self.vq_model.decode_code(
            generated_tokens,
            shape=(generated_tokens.shape[0], latent_h, latent_h),
        )

if __name__ == "__main__":
    import argparse
    from src.datasets.language_action import format_language_action_prompt, format_language_action_text

    parser = argparse.ArgumentParser()
    parser.add_argument("--lmm-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dino-path", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--future-image-type", default="none", choices=["none", "emu_token", "dinov3_flow"])
    parser.add_argument("--language-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--language-action-format", default="vla-0", choices=["vla-0", "lap", "language-action"])
    parser.add_argument("--language-action-max-new-tokens", type=int, default=16)
    parser.add_argument("--wan-action-condition-mode", default="fast", choices=["fast", "joint"])
    parser.add_argument("--future-video-downsample", type=int, default=1)
    parser.add_argument("--wan-text-len", type=int, default=512)
    parser.add_argument("--wan-tokenizer-model-id", default="google/umt5-xxl")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    print("Testing VLANeXt Model...")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    enable_future_image = args.future_image_type != "none"
    is_wan_smoke = "wan" in args.lmm_path.lower()

    if is_wan_smoke:
        model = VLANeXt(
            lmm_path=args.lmm_path,
            action_dim=7,
            num_actions=4,
            num_history=2,
            loss_type="diffusion",
            scheduler_type="flow_match",
            diffusion_loss_domain="noise",
            condition_type="tight",
            future_image_loss_weight=0.0,
            language_action_loss_weight=0.0,
            video_generation_loss_weight=1.0,
            wan_action_condition_mode=args.wan_action_condition_mode,
            wan_text_len=args.wan_text_len,
            wan_tokenizer_model_id=args.wan_tokenizer_model_id,
            future_video_downsample=args.future_video_downsample,
            generator_hidden_size=128,
            generator_depth=2,
            generator_num_heads=4,
            generator_max_seq_len=256,
            policy_hidden_size=128,
            policy_depth=2,
            policy_num_heads=4,
            backbone_mode="frozen",
            gradient_checkpointing=False,
        ).to(device, dtype)
        model.eval()

        batch_size = 1
        num_frames = model.num_actions // model.future_video_downsample + 1
        height = width = 256
        future_videos = torch.rand(batch_size, 3, num_frames, height, width, device=device, dtype=dtype) * 2 - 1
        current_images = future_videos[:, :, 0]
        actions = torch.randn(batch_size, model.num_actions, model.action_dim, device=device, dtype=dtype)
        proprio = torch.randn(batch_size, model.num_history, model.action_dim, device=device, dtype=dtype)
        hist_act = torch.randn(batch_size, model.num_history, model.action_dim, device=device, dtype=dtype)
        prompt_texts = ["A video recorded from a robot's point of view executing the instruction: pick up the object"]

        autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            loss = model(
                actions=actions,
                proprioception=proprio,
                history_actions=hist_act,
                future_videos=future_videos,
                current_images=current_images,
                prompt_texts=prompt_texts,
            )
            pred = model.predict_action(
                prompt_texts=prompt_texts,
                current_images=current_images,
                proprioception=proprio,
                history_actions=hist_act,
            )
        print(f"WAN condition mode: {args.wan_action_condition_mode}")
        print(f"WAN loss: {loss.item():.4f}")
        print(f"WAN action pred shape: {pred.shape}")
        raise SystemExit(0)

    model = VLANeXt(
        lmm_path=args.lmm_path,
        action_dim=7,
        num_actions=4,
        num_history=2,
        condition_type="soft",
        future_image_loss_weight=1.0 if enable_future_image else 0.0,
        future_image_prediction_type="emu_token" if args.future_image_type == "none" else args.future_image_type,
        future_image_dino_model_path=args.dino_path,
        future_image_dino_image_size=256,
        language_action_loss_weight=args.language_action_loss_weight,
        generator_hidden_size=128,
        generator_depth=2,
        generator_num_heads=4,
        generator_max_seq_len=256,
        policy_hidden_size=128,
        policy_depth=2,
        policy_num_heads=4,
        backbone_mode="finetune",
        gradient_checkpointing=False,
    ).to(device, dtype)
    model.eval()
    processor = model.processor
    enable_language_action = args.language_action_loss_weight > 0

    def build_future_images(images):
        if not enable_future_image:
            return None
        if args.future_image_type == "dinov3_flow":
            return model.future_image_processor(images=images, return_tensors="pt")["pixel_values"].to(
                device=device,
                dtype=dtype,
            )

        tensors = []
        for image in images:
            image_np = np.asarray(image).astype(np.uint8)
            tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
            tensors.append(tensor)
        return torch.stack(tensors).to(device=device, dtype=dtype)

    def add_answer_labels(full_inputs, prefix_inputs):
        language_action_labels = full_inputs["input_ids"].clone()
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is not None:
            language_action_labels = language_action_labels.masked_fill(attention_mask == 0, -100)

        prefix_attention = prefix_inputs.get("attention_mask")
        if prefix_attention is None:
            prefix_lengths = [prefix_inputs["input_ids"].shape[1]] * language_action_labels.shape[0]
        else:
            prefix_lengths = prefix_attention.long().sum(dim=1).tolist()

        for i, prefix_len in enumerate(prefix_lengths):
            prefix_len = int(prefix_len)
            if attention_mask is None:
                language_action_labels[i, : min(prefix_len, language_action_labels.shape[1])] = -100
                continue
            nonpad = torch.nonzero(attention_mask[i].bool(), as_tuple=False).flatten()
            if nonpad.numel() > 0:
                language_action_labels[i, nonpad[: min(prefix_len, nonpad.numel())]] = -100

        full_inputs["language_action_labels"] = language_action_labels
        return full_inputs

    def build_processor_inputs(texts, media, modality):
        kwargs = {f"{modality}s": media}
        if modality == "video":
            kwargs["videos_kwargs"] = {
                "fps": 20.0,
                "return_metadata": True,
                "video_metadata": [
                    {"total_num_frames": len(v), "fps": 20.0, "frames_indices": list(range(len(v)))}
                    for v in media
                ],
            }
        inputs = processor(text=texts, padding=True, return_tensors="pt", **kwargs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in ["pixel_values", "pixel_values_videos"]:
            if k in inputs:
                inputs[k] = inputs[k].to(dtype)
        return inputs

    def build_language_action_inputs(media, modality, actions):
        if not enable_language_action:
            return None, None

        answers = [
            format_language_action_text(
                action.detach().float().cpu().numpy(),
                language_action_format=args.language_action_format,
            )
            for action in actions
        ]
        prompt = (
            format_language_action_prompt(
                "Task.",
                language_action_format=args.language_action_format,
                frame_description=(
                    "normalized action space"
                    if args.language_action_format == "vla-0"
                    else "robot base frame"
                ),
                future_len=actions.shape[1],
                action_dim=actions.shape[2],
            )
            + "\nAnswer: "
        )

        content_key = "image" if modality == "image" else "video"
        prefix_msgs = [
            [{"role": "user", "content": [{"type": content_key, content_key: m}, {"type": "text", "text": prompt}]}]
            for m in media
        ]
        prefix_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in prefix_msgs
        ]
        full_texts = [text + answer for text, answer in zip(prefix_texts, answers)]

        full_inputs = build_processor_inputs(full_texts, media, modality)
        prefix_inputs = build_processor_inputs(prefix_texts, media, modality)
        return add_answer_labels(full_inputs, prefix_inputs), prefix_inputs

    def run_test(modality="image"):
        print(f"\n=== Testing {modality.capitalize()} ===")
        B = 2
        img = Image.new('RGB', (64, 64), color='red')
        media = [img] * B if modality == "image" else [[img]*8] * B
        content_key = "image" if modality == "image" else "video"
        msgs = [[{"role": "user", "content": [{"type": content_key, content_key: m}, {"type": "text", "text": "Task."}]}] for m in media]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

        valid_keys = {
            "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw", "mm_token_type_ids", "token_type_ids",
            "language_action_labels",
        }

        act_gt = torch.randn(B, 4, 7, device=device, dtype=dtype)
        proprio = torch.randn(B, 2, 7, device=device, dtype=dtype)
        hist_act = torch.randn(B, 2, 7, device=device, dtype=dtype)
        future_images = build_future_images([img] * B)
        if enable_language_action:
            inputs, pred_inputs = build_language_action_inputs(media, modality, act_gt)
        else:
            inputs = build_processor_inputs(texts, media, modality)
            pred_inputs = inputs
        fwd_args = {k: v for k, v in inputs.items() if k in valid_keys}
        pred_args = {k: v for k, v in pred_inputs.items() if k in valid_keys and k != "language_action_labels"}

        with torch.no_grad():
            loss = model(
                actions=act_gt,
                proprioception=proprio,
                history_actions=hist_act,
                future_images=future_images,
                **fwd_args,
            )
            pred = model.predict_action(
                proprioception=proprio,
                history_actions=hist_act,
                generate_language_action=enable_language_action,
                language_action_max_new_tokens=args.language_action_max_new_tokens,
                **pred_args,
            )
        print(f"Future image type: {args.future_image_type}")
        print(f"Language-action enabled: {enable_language_action}")
        if enable_language_action:
            supervised_tokens = int((inputs["language_action_labels"] != -100).sum().item())
            print(f"Language-action supervised tokens: {supervised_tokens}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Action Pred Shape: {pred.shape}")

    run_test("image")
    if not args.skip_video:
        run_test("video")
    print("\nTest Passed!")
