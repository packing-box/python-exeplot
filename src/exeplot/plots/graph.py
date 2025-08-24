# -*- coding: UTF-8 -*-
try:
    import angr
    import pygraphviz as pgv
    import networkx as nx
    _IMP = True
except ImportError:
    _IMP = False
import matplotlib.pyplot as plt

from .__common__ import Binary, CACHE_DIR, COLORS, MIN_ZONE_WIDTH
from ..__conf__ import save_figure


_DEFAULT_ALGORITHM, _DEFAULT_ENGINE = "fast", "default"
_ENGINES = ["default", "pcode", "vex"]


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    parser.add_argument("-a", "--algorithm", default=_DEFAULT_ALGORITHM, choices=["emulated", "fast"],
                        help="engine for CFG extraction by Angr")
    parser.add_argument("-e", "--engine", default=_DEFAULT_ENGINE, choices=_ENGINES,
                        help="engine for CFG extraction by Angr")
    return parser


@save_figure
def plot(executable, algorithm=_DEFAULT_ALGORITHM, engine=_DEFAULT_ENGINE, **kwargs):
    """ plot the Control Flow Graph (CFG) of an executable """
    from math import ceil, log2
    engine = {k: getattr(angr.engines, "UberEngine" if k != "pcode" else f"UberEngine{k.capitalize()}") \
              for k in _ENGINES}[engine]
    project = angr.Project(executable, auto_load_libs=False, engine=engine)
    cfg = getattr(project.analyses, f"CFG{algorithm.capitalize()}")()
    labels, node_colors = {}, []
    for node in cfg.graph.nodes():
        labels[node] = f"{node.name}\n0x{node.addr:x}" if hasattr(node, "name") and node.name else f"0x{node.addr:x}"
        node_colors.append("red" if node.function_address == node.addr else "lightblue")
    n = max(10, min(30, ceil(log2(n_nodes := len(cfg.graph.nodes()) + 1) * 2)))
    plt.figure(figsize=(n, n))
    nx.draw(cfg.graph, nx.kamada_kawai_layout(cfg.graph), font_size=8, with_labels=True, labels=labels,
            node_size=max(300, 15000 // n_nodes), node_color=node_colors)

