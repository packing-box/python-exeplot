# -*- coding: UTF-8 -*-
from .__common__ import _human_readable_size, Binary, COLORS
from ..__conf__ import save_figure


def _section_generator(binary):
    if binary.type == "PE":
        h_len = binary.sizeof_headers
    elif binary.type == "ELF":
        h_len = binary.header.header_size + binary.header.program_header_size * binary.header.numberof_segments
    elif binary.type == "MachO":
        h_len = [28, 32][str(bin.header.magic)[-3:] == "_64"] + binary.header.sizeof_cmds
    yield 0, f"[0] Header ({_human_readable_size(h_len)})", 0, h_len, "black"
    color_cursor, i = 0, 1
    for section in binary.sections:
        if section.name == "" and section.size == 0 and len(section.content) == 0:
            continue
        try:
            c = COLORS[binary.section_names[section.name].lower().lstrip("._").strip("\x00\n ")]
        except KeyError:
            co = COLORS[None]
            c = co[color_cursor % len(co)]
            color_cursor += 1
        start, end = section.offset, section.offset + max(section.size, len(section.content))
        yield i, f"[{i}] {binary.section_names[section.name]} ({_human_readable_size(end - start)})", start, end, c
        i += 1
    if binary.type == "ELF":
        start, end = end, end + binary.header.section_header_size * binary.header.numberof_sections
        yield i, f"[{i}] Section Header ({_human_readable_size(end - start)})", start, end, "black"
        i += 1
    start, end = binary.size - binary.overlay.nbytes, binary.size
    yield i, f"[{i}] Overlay ({_human_readable_size(end - start)})", start, binary.size, "lightgray"
    i += 1
    yield i, f"TOTAL: {_human_readable_size(binary.size)}", None, None, "black"


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    return parser


@save_figure
def plot(executable, height=600, **kwargs):
    """ draw a byte plot of the input binary """
    import matplotlib.colors as mcol
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    from matplotlib import font_manager, rcParams
    from PIL import Image, ImageDraw, ImageFont
    # determine base variables and some helper functions
    images, binary = [], Binary(executable)
    n_pixels = ceil(binary.size / 3)
    s = int(ceil(sqrt(n_pixels)))
    sf = height / s
    _rgb = lambda c: tuple(map(lambda x: int(255 * x), mcol.to_rgba(c)))
    # draw a byte plot
    rawbytes = binary.rawbytes + int((s * s * 3) - len(binary.rawbytes)) * b'\xff'
    images.append(Image.frombuffer("RGB", (s, s), rawbytes, "raw", "RGB", 0, 1) \
                       .resize((int(s * sf), height), resample=Image.Resampling.BOX))
    if len(binary.sections) > 0:
        # map matplotlib font to PIL ImageFont path
        font_name = rcParams[f"font.{kwargs['config']['font_family']}"][0]
        font_path = font_manager.findfont(font_name)
        font_size = ceil(height / 30)
        font = ImageFont.truetype(font_path, size=font_size)
        # determine the maximum width for section name labels
        txt_spacing = (txt_h := font_size) // 2
        n_lab_per_col = int((height - txt_spacing) / (txt_h + txt_spacing))
        n_cols = ceil((len(binary.sections) + 2) / n_lab_per_col)
        max_txt_w = [0] * n_cols
        sections = ["Headers"] + [s for s in binary.sections] + ["Overlay"]
        draw = ImageDraw.Draw(images[0])
        for i, name, _, _, end in _section_generator(binary):
            if end is None:
                continue
            max_txt_w[i // n_lab_per_col] = max(max_txt_w[i // n_lab_per_col],
                                                ceil(draw.textlength(name, font=font)) + 2 * txt_spacing)
        max_w = sum(max_txt_w) + (n_cols - 1) * txt_spacing
        # draw a separator
        images.append(Image.new("RGB", (int(.05 * height), height), "white"))
        # draw a sections plot aside
        img = Image.new("RGB", (s, s), "white")
        # draw the legend with section names
        legend = Image.new("RGB", (max_w, height), "white")
        draw = ImageDraw.Draw(legend)
        _xy = lambda n, c: (txt_spacing + sum(max_txt_w[:c]) + len(max_txt_w[:c]) * txt_spacing, \
                            txt_spacing + (n % n_lab_per_col) * (txt_spacing + txt_h))
        #xN_prev, yN_prev = 0, 0
        for i, name, start, end, color in _section_generator(binary):
            print(i, name, start, end, color, end="")
            if start != end:
                x0, y0 = min(max(ceil(((start / 3) % s)) - 1, 0), s - 1), \
                         min(max(ceil(start / s / 3) - 1, 0), s - 1)
                xN, yN = min(max(ceil(((end / 3) % s)) - 1, 0), s - 1), \
                         min(max(ceil(end / s / 3) - 1, 0), s - 1)
                if y0 == yN:
                    xN = min(max(x0 + 1, xN), s - 1)
                print("", x0, y0, xN, yN, end="")
                for x in range(x0, s if y0 < yN else xN):
                    img.putpixel((x, y0), _rgb(color))
                for y in range(y0 + 1, yN):
                    for x in range(0, s):
                        img.putpixel((x, y), _rgb(color))
                if yN > y0:
                    for x in range(0, xN):
                        img.putpixel((x, yN), _rgb(color))
                #xN_prev, yN_prev = xN, yN
            # fill the legend with the current section name
            draw.text(_xy(i, ceil((i + 1) / n_lab_per_col) - 1), name, fill=_rgb(color), font=font)
            print("")
        images.append(img.resize((int(img.size[0] * sf * .2), height), resample=Image.Resampling.BOX))
        images.append(Image.new("RGB", (int(.03 * height), height), "white"))  # draw another separator
        images.append(legend)
    # combine images horizontally
    w, h = zip(*(i.size for i in images))
    x, img = 0, Image.new("RGB", (sum(w), max(h)))
    for i in images:
        img.paste(i, (x, 0))
        x += i.size[0]
    # plot combined PIL images
    plt.imshow(img)
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.title(f"Byte plot of {binary.basename}", **kwargs['title-font'])
plot.__args__ = arguments

