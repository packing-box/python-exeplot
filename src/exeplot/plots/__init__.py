# -*- coding: UTF-8 -*-
import importlib
import os


__all__ = []


for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    if not f.endswith(".py") or f.startswith("_"):
        continue
    name = f[:-3]
    module = importlib.import_module(f".{name}", package=__name__)
    if hasattr(module, "plot") and callable(getattr(module, "plot")):
        globals()[f"{name}"] = getattr(module, "plot")
        __all__.append(name)

