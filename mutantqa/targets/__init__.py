from ..config import TargetConfig
from .base import Target, TargetResponse
from .http_target import HttpTarget
from .python_target import PythonTarget


def make_target(cfg: TargetConfig) -> Target:
    if cfg.type == "http":
        return HttpTarget(cfg)
    return PythonTarget(cfg.function)


__all__ = ["Target", "TargetResponse", "HttpTarget", "PythonTarget", "make_target"]
