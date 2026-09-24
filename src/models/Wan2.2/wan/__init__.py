# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

from . import configs, distributed, modules

_LAZY_EXPORTS = {
    "WanI2V": ".image2video",
    "WanS2V": ".speech2video",
    "WanT2V": ".text2video",
    "WanTI2V": ".textimage2video",
    "WanAnimate": ".animate",
}

__all__ = ["configs", "distributed", "modules", *_LAZY_EXPORTS]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
