"""capsel.models.inca — INCA continual learning model components.

Pure-Python classes are importable without torch.
Torch-dependent classes are accessed via helper functions.
"""

# Pure-Python — always importable
from .config  import INCAConfig, BaseConfig
from .replay  import INCAReplayBuffer
from .plateau import INCAPlateauDetector, GradNormTracker, SaturationEvent

# Torch-dependent — lazy
def _get(module_name, *class_names):
    import importlib
    mod = importlib.import_module(f".{module_name}", package=__name__)
    return tuple(getattr(mod, c) for c in class_names)

def get_layer_manager():
    (cls,) = _get("layer_manager", "INCALayerManager"); return cls

def get_selectors():
    return _get("selectors", "EmbeddingQuerySelector",
                "CrossAttentionSelector", "WeightedSumSelector")

def get_uclbr():
    (cls,) = _get("uclbr", "UCLBRSelector"); return cls

def get_cka():
    (cls,) = _get("cka", "CKAMonitor"); return cls

def get_lateral():
    (cls,) = _get("lateral", "LateralAdapter"); return cls

__all__ = [
    "INCAConfig", "BaseConfig",
    "INCAReplayBuffer",
    "INCAPlateauDetector", "GradNormTracker", "SaturationEvent",
    "get_layer_manager", "get_selectors", "get_uclbr", "get_cka", "get_lateral",
]
