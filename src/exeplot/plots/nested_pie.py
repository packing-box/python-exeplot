# -*- coding: UTF-8 -*-
from .__common__ import Binary, COLORS, SHADOW
from ..__conf__ import save_figure
from ..utils import human_readable_size


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    return parser


@save_figure
def plot(executable, **kwargs):
    """ draw a nested pie chart of segments (if relevant) and sections (including overlaps) of the input binary """
    import matplotlib.pyplot as plt
    from math import ceil
    binary = Binary(executable)
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    pie_kw = {'shadow': SHADOW} if kwargs['config']['shadow'] else {}
    seg_layers = sum(1 for _ in binary._data(segments=True, overlap=True))
    if binary.type != "PE":
        for i, x, w, labels, colors, legend in binary._data(segments=True, overlap=True):
            size = .4 / seg_layers
            ax.pie(w, radius=.45-i*size, colors=colors, startangle=90,
                   wedgeprops={'width': size, 'edgecolor': "w", 'linewidth': 1}, **pie_kw)
    sec_layers = sum(1 for _ in binary._data(overlap=True))
    for i, x, w, labels, colors, legend in binary._data(overlap=True):
        size = .42 / sec_layers
        ax.pie(w, radius=1-i*size, colors=colors, startangle=90, wedgeprops={'width': size}, **pie_kw)
    ncols = ceil(len(legend['colors']) / 12)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in legend['colors']], legend['texts'], loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), ncol=ncols, fontsize=ceil(kwargs['config']['font_size']*.7))
    if not kwargs.get('no_title', False):
        fsp = plt.gcf().subplotpars
        plt.title(f"Nested pie plot of {binary.type} file: {binary.basename}", **kwargs['title-font'])
        plt.suptitle(binary.hash, y=fsp.top, x=(fsp.right+fsp.left)/2, **kwargs['annotation-font'])

