# -*- coding: UTF-8 -*-
from .__common__ import mean, Binary, COLORS, MIN_ZONE_WIDTH, N_SAMPLES, SUBLABELS
from ..__conf__ import save_figure
from ..utils import shannon_entropy


def arguments(parser):
    parser.add_argument("executable", nargs="+", help="executable samples to be plotted")
    parser.add_argument("-l", "--labels", nargs="*", help="list of custom labels to be used for the binaries")
    parser.add_argument("--sublabel", default="size-ep-ent", choices=tuple(SUBLABELS.keys()),
                        help="sublabel for display under the label")
    parser.add_argument("-s", "--scale", action="store_true", help="use the same scale for multiple files")
    parser.add_argument("-t", "--target", help="target name for the title (instead of first filename from input list)")
    return parser


def data(executable, n_samples=N_SAMPLES, window_size=lambda s: 2*s, **kwargs):
    """ Compute executable's desired data, including:
        - 'x' samples of entropy using a sliding window of size 'window_size'
        - sections' bounds (reduced according to the 'x' samples)
        - executable type
        - entry point (set according to the 'x' samples)
    
    :param executable:  path to executable whose data are to be computed
    :param n_samples:   number of samples of entropy required
    :param window_size: window size for computing the entropy
    """
    binary = Binary(executable)
    data = {'hash': binary.hash, 'name': binary.basename, 'size': binary.size, 'type': binary.type,
            'entropy': [], 'sections': []}
    # compute window-based entropy
    data['entropy*'] = shannon_entropy(binary.rawbytes)
    step, cs = abs(binary.size // n_samples), binary.size / n_samples  # chunk size
    if isinstance(window_size, type(lambda: 0)):
        window_size = window_size(step)
    # ensure the window interval is at least 256 (that is 2^8 ; with a 'security' factor of 2) because otherwise if
    #  using a too small executable, it may get undersampled and have lower entropy values than actual
    window, winter = b"", max(step, abs(window_size // 2), 256)
    # rectify the size of the window with the fixed interval
    window_size = 2 * winter
    with open(binary.path, 'rb') as f:
        for i in range(n_samples+1):
            # slice the window
            new_pos, cur_pos = int((i+1)*cs), int(i*cs)
            window += f.read(new_pos - cur_pos if i > 0 else winter)
            window = window[max(0, len(window)-window_size) if cur_pos + winter < binary.size else step:]
            # compute entropy
            data['entropy'].append(shannon_entropy(window)/8.)
    # compute other characteristics using the Binary instance parsed with LIEF
    # convert to 3-tuple (EP offset on plot, EP file offset, section name containing EP)
    ep, ep_sec = binary.entrypoint, binary.entrypoint_section
    data['ep'] = None if ep is None or ep_sec is None else (int(ep // cs), ep, binary.section_names[ep_sec.name])
    # sections
    __d = lambda s, e, n: (int(s), int(e), n, mean(data['entropy'][int(s):int(e)+1]))
    data['sections'] = [__d(0, int(max(MIN_ZONE_WIDTH, binary.sections[0].offset // cs)), "Header")] \
                       if len(binary.sections) > 0 else []
    for i, section in enumerate(binary.sections):
        name = binary.section_names[section.name]
        start = max(data['sections'][-1][1] if len(data['sections']) > 0 else 0, int(section.offset // cs))
        rawsize = max(section.size, len(section.content)) # take section header's raw size but consider real size too
        max_end = min(max(start + MIN_ZONE_WIDTH, int((section.offset + rawsize) // cs)),
                      len(data['entropy']) - 1)
        data['sections'].append(__d(int(min(start, max_end - MIN_ZONE_WIDTH)), int(max_end), name))
    # adjust the entry point (be sure that its position on the plot is within the EP section)
    if data['ep']:
        ep_pos, _, ep_sec_name = data['ep']
        for s, e, name, m in data['sections']:
            if name == ep_sec_name:
                data['ep'] = (min(max(ep_pos, s), e), ep, ep_sec_name)
    # fill in undefined sections
    prev_end = None
    for i, t in enumerate(data['sections'][:]):
        start, end, name, _ = t
        if prev_end and prev_end < start:
            data['sections'].insert(i, __d(prev_end, start, "<undef>"))
        prev_end = end
    if len(binary.sections) > 0:
        last = data['sections'][-1][1]
        if data['type'] == "ELF":
            # add section header table
            sh_size = binary.header.section_header_size * binary.header.numberof_sections
            data['sections'].append(__d(int(last), int(last) + sh_size // cs, "Section Header"))
        if last + 1 < n_samples:
            data['sections'].append(__d(int(last), int(n_samples), "Overlay"))
    return data


@save_figure
def plot(*filenames, labels=None, sublabel=None, scale=False, target=None, **kwargs):
    """ plot the sections of the input executable(s) with their entropy and entry point """
    import matplotlib.pyplot as plt
    from math import ceil
    from matplotlib.patches import Patch
    from os import fstat
    if len(filenames) == 0:
        raise ValueError("No executable to plot")
    lloc, title = kwargs.get('legend_location', "lower center"), not kwargs.get('no_title', False)
    lloc_side = lloc.split()[1] in ["left", "right"]
    nf, N_TOP, N_TOP2, N_BOT, N_BOT2 = len(filenames), 1.15, 1.37, -.15, -.37
    fig, objs = plt.subplots(nf+[0, 1][title], sharex=True)
    fig.set_size_inches(10, nf+[0, 1][title])
    fig.tight_layout(pad=1)
    (objs[0] if nf+[0, 1][title] > 1 else objs).axis("off")
    ref_size, ref_n, fs_ref = None, kwargs.get('n_samples', N_SAMPLES), kwargs['config']['font_size']
    for i, filepath in enumerate(filenames):
        if scale and ref_size:
            with open(filepath, "rb") as f:
                size = fstat(f.fileno()).st_size
            kwargs['n_samples'] = int(ref_n * size / ref_size)
        obj = objs[i+[0, 1][title]] if nf+[0, 1][title] > 1 else objs
        d = data(filepath, **kwargs)
        n, label = len(d['entropy']), None
        if not ref_size:
            ref_size = d['size']
        obj.axis("off")
        # set the main title for the whole figure
        if i == 0 and title:
            x_t, y_t = [.6, .5][labels is None], 1. - .6 / (nf + [0, 1][title])
            fig.suptitle(f"Entropy per section of {d['type']} file: {target or d['name']}", x=x_t, y=y_t,
                         ha="center", va="bottom", **kwargs['title-font'])
        # set the label and sublabel and display them
        try:
            label = labels[i]
            if isinstance(label, type(lambda: 0)):
                label = label(d)
        except:
            pass
        ref_point = .55
        if sublabel and not (isinstance(sublabel, str) and "ep" in sublabel and d['ep'] is None):
            if isinstance(sublabel, str):
                sublabel = SUBLABELS.get(sublabel)
            sl = sublabel(d) if isinstance(sublabel, type(lambda: 0)) else None
            if sl:
                nl, y_pos, f_color = len(sl.split("\n")), ref_point, "black"
                if label:
                    f_size, f_color = fs_ref*.6 if nl <= 2 else fs_ref*.5, "gray"
                    y_pos = max(0., ref_point - nl * [.16, .12, .09, .08][min(4, nl)-1])
                else:
                    f_size = fs_ref * [.7, .6, .5][min(3, nl)-1]
                obj.text(s=sl, x=-420., y=y_pos, fontsize=f_size, color=f_color, ha="left", va="center")
        if label:
            y_pos = ref_point
            if sublabel:
                nl = len(sl.split("\n"))
                y_pos = min(1.3, ref_point + .3 + nl * [.16, .12, .09, .08][min(4, nl)-1])
            obj.text(s=label, x=-420., y=y_pos, fontsize=fs_ref, ha="left", va="center")
            h, h_midlen = d['hash'], round(len(d['hash'])/2+.5)
            h = f"{h[:h_midlen]}\n{h[h_midlen:]}"
            obj.text(s=h, x=-420., y=y_pos-.35, fontsize=fs_ref*.5, ha="left", va="center")
        # display the entry point
        if d['ep']:
            obj.vlines(x=d['ep'][0], ymin=0, ymax=1, color="r", zorder=11).set_label("Entry point")
            obj.text(d['ep'][0], -.15, "______", c="r", ha="center", rotation=90, size=.8,
                     bbox={'boxstyle': "rarrow", 'fc': "r", 'ec': "r", 'lw': 1})
        i, color_cursor, last = 0, 0, None
        for start, end, name, avg_ent in d['sections']:
            x = range(start, min(n, end + 1))
            # select the right color first
            try:
                c = COLORS[name.lower().lstrip("._").strip("\x00\n ")]
            except KeyError:
                co = COLORS[None]
                c = co[color_cursor % len(co)]
                color_cursor += 1
            # draw the section
            obj.fill_between(x, 0, 1, facecolor=c, alpha=.2)
            if name not in ["Headers", "Overlay"]:
                if last is None or (start + end) // 2 - (last[0] + last[1]) // 2 > n // 12:
                    pos_y = N_TOP
                else:
                    pos_y = N_BOT if pos_y in [N_TOP, N_TOP2] else N_TOP
                if last and last[2] and (start + end) // 2 - (last[2] + last[3]) // 2 < n // 15:
                    if pos_y == N_TOP:
                        pos_y = N_TOP2
                    elif pos_y == N_BOT:
                        pos_y = N_BOT2
                obj.text(s=name, x=start + (end - start) // 2, y=pos_y, zorder=12, color=c, fontsize=fs_ref*.8,
                         ha="center", va="center")
                last = (start, end, last[0] if last else None, last[1] if last else None)
            # draw entropy
            obj.plot(x, d['entropy'][start:end+1], c=c, zorder=10, lw=.1)
            obj.fill_between(x, [0] * len(x), d['entropy'][start:end+1], facecolor=c)
            l = obj.hlines(y=mean(d['entropy'][start:end+1]), xmin=x[0], xmax=x[-1], color="black", linewidth=.5, linestyle=(0, (5, 5)))
            i += 1
        if len(d['sections']) > 0:
            l.set_label("Average entropy of section")
        else:
            obj.text(.5, ref_point, "Could not parse sections", fontsize=fs_ref*1.6, color="red", ha="center",
                     va="center")
    plt.subplots_adjust(left=[.15, .02][labels is None and sublabel is None], right=[1.02, .82][lloc_side],
                        bottom=.5/max(1.75, nf))
    h, l = (objs[[0, 1][title]] if nf+[0, 1][title] > 1 else objs).get_legend_handles_labels()
    h.append(Patch(facecolor="black")), l.append("Headers")
    h.append(Patch(facecolor="lightgray")), l.append("Overlay")
    if len(h) > 0:
        plt.figlegend(h, l, loc=lloc, ncol=1 if lloc_side else len(l), prop={'size': fs_ref*.7})

