# -*- coding: UTF-8 -*-
import importlib
import os


__all__ = []


for f in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
    if not f.endswith(".py") or f.startswith("_"):
        continue
    name = f[:-3]
    module = importlib.import_module(f".{name}", package=__name__)
    if getattr(module, "_IMP", True) and hasattr(module, "plot") and callable(getattr(module, "plot")):
        globals()[f"{name}"] = f = getattr(module, "plot")
        f.__args__ = getattr(module, "arguments")
        f.__name__ = name
        __all__.append(name)

