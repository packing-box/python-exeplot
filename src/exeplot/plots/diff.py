# -*- coding: UTF-8 -*-
from .__common__ import Binary, CACHE_DIR, COLORS, MIN_ZONE_WIDTH
from ..__conf__ import save_figure


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    parser.add_argument("executable2", help="modified executable to be compared with the reference")
    return parser


def data(binary):
    """ Compute executable's desired data, including:
        - 'x' samples of entropy using a sliding window of size 'window_size'
        - sections' bounds (reduced according to the 'x' samples)
        - executable type
        - entry point (set according to the 'x' samples)
    
    :param binary: Binary instance
    :return:       dictionary of characteristics from the target binary
    """
    # convert to 3-tuple (EP offset on plot, EP file offset, section name containing EP)
    entrypoint = None if (ep := binary.entrypoint) is None else \
                 (ep, ep, binary.entrypoint_section.name if binary.entrypoint_section else None)
    # sections
    sections = [(0, max(MIN_ZONE_WIDTH, binary.sections[0].offset), "Header")] if len(binary.sections) > 0 else []
    for section in sorted(binary.sections, key=lambda x: x.offset):
        start = max(sections[-1][1] if len(sections) > 0 else 0, section.offset)
        end = min(max(start + MIN_ZONE_WIDTH, int(section.offset + section.size)), binary.size)
        sections.append((min(start, end - MIN_ZONE_WIDTH), end, binary.section_names[section.name]))
    # adjust the entry point (be sure that its position on the plot is within the EP section)
    if ep:
        ep_pos, _, ep_sec_name = entrypoint
        for s, e, name in sections:
            if name == ep_sec_name:
                entrypoint = (min(max(ep_pos, s), e), ep, ep_sec_name)
    # fill in undefined sections
    prev_end, j = 0, 0
    for i, t in enumerate(sections[:]):
        start, end, name = t
        if start > prev_end:
            sections.insert(i+j, (prev_end, start, ""))
            j += 1
        prev_end = end
    if len(binary.sections) > 0:
        last = sections[-1][1]
        if binary.type == "ELF":
            # add section header table
            sh_size = binary.header.section_header_size * binary.header.numberof_sections
            sections.append((last, last + sh_size, "Header"))
        elif binary.type == "PE":
            # add overlay
            if last + 1 < binary.size:
                sections.append((last, binary.size, "Overlay"))
    return sections, entrypoint


@save_figure
def plot(executable, executable2, legend1="", legend2="", **kwargs):
    """ plot a reference executable and its modified version, highlighting changes """
    import matplotlib.pyplot as plt
    from difflib import SequenceMatcher
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap
    try:
        from joblib import Memory
        cache = Memory(CACHE_DIR, verbose=0).cache
    except ImportError:
        cache = lambda x: x  # do nothing if joblib not installed
    # inner function for caching sequence matches)
    @cache
    def byte_differences(bytes1, bytes2):
        return zip(*SequenceMatcher(a=bytes1, b=bytes2).get_opcodes())
    fs_ref = kwargs['config']['font_size']
    title = not kwargs.get('no_title', False)
    lloc = kwargs.get('legend_location', "lower right")
    lloc_side = lloc.split()[1] in ["left", "right"]
    nf, N_TOP, N_TOP2, N_BOT, N_BOT2 = 2, 1.2, 1.6, -.15, -.37
    fig, objs = plt.subplots(nf+int(title), sharex=True)
    fig.set_size_inches(15, nf+int(title))
    fig.tight_layout(pad=2)
    objs[-1].axis("off")
    values, colors = {'delete': 0, 'replace': 1, 'equal': 2, 'insert': 3}, ["red", "gold", "lightgray", "green"]
    bin1, bin2 = Binary(executable), Binary(executable2)
    if bin1.type != bin2.type:
        raise ValueError(f"Inputs executables have different types ({bin1.type} != {bin2.type})")
    if title:
        fig.suptitle(f"Byte-wise difference of {bin1.type} files: {bin1.basename} VS {bin2.basename}",
                     x=[.5, .55][legend1 is None], y=1, ha="center", va="bottom", **kwargs['title-font'])
    legend1, legend2 = legend1 or bin1.basename, legend2 or bin2.basename
    (lg := kwargs['logger']).info("Matching binaries' byte sequences, this may take a while...")
    tags, alo, ahi, blo, bhi = byte_differences(bin1.rawbytes, bin2.rawbytes)
    lg.debug("Plotting binaries...")
    text_x = -0.012*max(bin1.size*(len(legend1)+3), bin2.size*(len(legend2)+3))
    for i, d in enumerate([(bin1, zip(tags, alo, ahi), legend1), (bin2, zip(tags, blo, bhi), legend2)]):
        binary, opcodes, label = d
        sections, ep = data(binary)
        n, obj = binary.size, objs[i]
        obj.axis("off")
        y_pos = ref_point = .65
        obj.text(s=label, x=text_x, y=y_pos, fontsize=fs_ref, ha="left", va="center")
        obj.text(s="\n".join(binary.hash[i:i+32] for i in range(0, len(binary.hash), 32)), x=text_x, y=y_pos-.65,
                 fontsize=fs_ref*.45, color="gray", ha="left", va="center")
        # display the entry point
        if ep:
            obj.vlines(x=ep[0], ymin=0, ymax=1, color="r", zorder=11).set_label("Entry point")
            obj.text(ep[0], -.15, "______", color="r", ha="center", rotation=90, size=.8,
                     bbox={'boxstyle': "rarrow", 'fc': "r", 'ec': "r", 'lw': 1})
        color_cursor, last, j = 0, None, 0
        for start, end, name in sections:
            x = range(start, min(n, end+1))
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
                pos_y = [N_TOP2, N_TOP][j % 2]
                obj.text(s=name, x=start + (end - start) // 2, y=pos_y, zorder=12, color=c, ha="center", va="center")
                last = (start, end, last[0] if last else None, last[1] if last else None)
            j += 1
        # draw modifications
        for (tag, lo, hi) in opcodes:
            obj.fill_between((lo, hi), 0, 0.7, facecolor=colors[values[tag]], alpha=1)
        if len(sections) == 0:
            obj.text(.5, ref_point, "Could not parse sections", fontsize=fs_ref*1.6, color="red", ha="center",
                     va="center")
    cb = plt.colorbar(ScalarMappable(cmap=ListedColormap(colors, N=4)),
                      location='bottom', ax=objs[-1], fraction=0.3, aspect=50, ticks=[0.125, 0.375, 0.625, 0.875])
    cb.set_ticklabels(['removed', 'modified', 'untouched', 'added'])
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    plt.subplots_adjust(left=[.15, .02][legend1 == "" and legend2 == ""], bottom=.5/max(1.75, nf))
    h, l = (objs[int(title)] if nf+int(title) > 1 else objs).get_legend_handles_labels()
    if len(h) > 0:
        plt.figlegend(h, l, loc=[.8, .135], ncol=1, prop={'size': fs_ref*.7})

