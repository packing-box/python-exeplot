# -*- coding: UTF-8 -*-
from .__common__ import Binary, COLORS, SHADOW
from ..__conf__ import save_figure
from ..utils import human_readable_size


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    parser.add_argument("--donut", action="store_true", help="plot as a donut instead of a pie")
    return parser


@save_figure
def plot(executable, donut=False, **kwargs):
    """ draw a pie chart of the sections of the input binary """
    import matplotlib.pyplot as plt
    from math import ceil
    fs_ref = kwargs['config']['font_size']
    binary = Binary(executable)
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    for _, _, data, _, colors, legend in binary._data():
        pass
    ncols, n = ceil(len(legend['colors']) / 12), sum(data)
    labels = [f"{100 * d / n:.1f}%" if round(d / n, 2) >= .02 and c != "white" else "" for d, c in zip(data, colors)]
    txt_kw = {k: v for k, v in kwargs['xlabel-font'].items()}
    txt_kw['color'] = "w"
    pie_kw = {'shadow': SHADOW} if kwargs['config']['shadow'] else {}
    _, texts = ax.pie(data, colors=colors, labels=labels, textprops=txt_kw, labeldistance=.55, startangle=90,
                           wedgeprops={'width': [1., .55][donut]}, **pie_kw)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in legend['colors']], legend['texts'], loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), ncol=ncols, fontsize=ceil(fs_ref*.7))
    plt.setp(texts, size=fs_ref*.8, weight="bold")
    if not kwargs.get('no_title', False):
        fsp = plt.gcf().subplotpars
        plt.title(f"Pie plot of {binary.type}: {binary.basename}", y=1., **kwargs['title-font'])
        plt.suptitle(binary.hash, y=fsp.top, x=(fsp.right+fsp.left)/2, **kwargs['annotation-font'])

