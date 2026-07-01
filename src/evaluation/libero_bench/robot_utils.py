"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

from src.evaluation.libero_bench.VLANeXt_utils import (
    get_vla as get_vlanext,
    get_vla_action as get_vlanext_action,
    get_processor as get_vlanext_processor,
)

ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg):
    """Load model for evaluation."""
    model = get_vlanext(cfg)
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    """
    return cfg.eval.image_size


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    action = get_vlanext_action(
        cfg, model, processor, obs, task_label
    )
    return action