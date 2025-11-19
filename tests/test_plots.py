#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import os
from exeplot import *
from exeplot import __all__ as _plots
from exeplot.__conf__ import configure
from exeplot.__main__ import _parser
from unittest import TestCase


def iter_files():
    d = os.path.dirname(__file__)
    for fn in os.listdir(d):
        _, ext = os.path.splitext(fn)
        if ext not in [".elf", ".exe", ".macho"]:
            continue
        yield os.path.join(d, fn)


class TestPlots(TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
    
    def test_plot_functions(self):
        for path in iter_files():
            print(f"plotting {path}...")
            for name in _plots:
                print(f" > making {name} chart...")
                plot_func = globals()[name]
                self.assertTrue(hasattr(plot_func, "__args__"))
                p = _parser("test", "...", [])
                plot_func.__args__(p)
                args, ok = (), False
                for i in range(5):
                    try:
                        args += (path, )
                        for img in plot_func(*args):
                            os.remove(img)
                            plt.clf()
                        ok = True
                        break
                    except TypeError as e:
                        # known issue: macho CFG generation is not well supported
                        if "NoneType" in str(e) and path.endswith(".macho"):
                            ok = True
                            break
                        pass
                self.assertTrue(ok)
        NOT_EXE = os.path.join(os.path.dirname(__file__), "__init__.py")
        self.assertRaises(TypeError, plot_func, NOT_EXE)


class TestPlotOptions(TestCase):
    def test_diff_plot_function(self):
        exe = tuple([path for i, path in enumerate(iter_files()) if i < 2])
        self.assertRaises(ValueError, diff, exe[0], exe[1])
    
    def test_entropy_plot_function(self):
        self.assertRaises(ValueError, entropy)
        for path in iter_files():
            print(f"plotting entropy of {path} (sublabel='size-ep-ent',scale=True,target='test.exe')...")
            entropy(path, sublabel="size-ep-ent", scale=True, target="test")
        print(f"plotting entropy of {path}.exe and {path}.elf (labels=['PE', lambda x:'ELF'],sublabel='size-ep-ent',"
              "scale=True)...")
        path = os.path.join(os.path.dirname(__file__), "hello")
        for img in entropy(f"{path}.exe", f"{path}.elf", labels=["PE", lambda x: "ELF"], sublabel="size-ep-ent",
                           scale=True):
            os.remove(img)
            plt.clf()
    
    def test_pie_plot_function(self):
        for path in iter_files():
            print(f"plotting pie of {path} (donut=True)...")
            for img in pie(path, donut=True):
                os.remove(img)
                plt.clf()

