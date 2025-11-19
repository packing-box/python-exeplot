# -*- coding: UTF-8 -*-
from .__common__ import Binary, COLORS
from ..__conf__ import save_figure
from ..utils import human_readable_size


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
        font = ImageFont.truetype(font_manager.findfont(font_name), size=(txt_h := ceil(height / 30)))
        # determine the maximum width for section name labels
        txt_spacing = txt_h // 2
        n_lab_per_col = int((height - txt_spacing) / (txt_h + txt_spacing))
        n_cols = ceil((len(binary.sections) + 2) / n_lab_per_col)
        #n_cols = 1
        max_txt_w = [0] * n_cols
        sections = ["Headers"] + [s for s in binary.sections] + ["Overlay"]
        draw = ImageDraw.Draw(images[0])
        for i, name, _, end, _ in binary:
            if end is None:
                continue
            try:
                max_txt_w[i // n_lab_per_col] = max(max_txt_w[i // n_lab_per_col],
                                                ceil(draw.textlength(name, font=font)) + 2 * txt_spacing)
            except:
                pass
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
        for i, name, start, end, color in binary:
            if start != end:
                x0, y0 = min(max(ceil(((start / 3) % s)) - 1, 0), s - 1), \
                         min(max(ceil(start / s / 3) - 1, 0), s - 1)
                xN, yN = min(max(ceil(((end / 3) % s)) - 1, 0), s - 1), \
                         min(max(ceil(end / s / 3) - 1, 0), s - 1)
                if y0 == yN:
                    xN = min(max(x0 + 1, xN), s - 1)
                for x in range(x0, s if y0 < yN else xN):
                    img.putpixel((x, y0), _rgb(color))
                for y in range(y0 + 1, yN):
                    for x in range(0, s):
                        img.putpixel((x, y), _rgb(color))
                if yN > y0:
                    for x in range(0, xN):
                        img.putpixel((x, yN), _rgb(color))
            # fill the legend with the current section name
            if name.startswith("TOTAL"):
                color = "black"
            draw.text(_xy(i, ceil((i + 1) / n_lab_per_col) - 1), name, fill=_rgb(color), font=font)
        images.append(img.resize((int(img.size[0] * sf * .2), height), resample=Image.Resampling.BOX))
        images.append(Image.new("RGB", (int(.03 * height), height), "white"))  # draw another separator
        images.append(legend)
    # combine images horizontally
    x, img = 0, Image.new("RGB", (sum(i.size[0] for i in images), height))
    for i in images:
        img.paste(i, (x, 0))
        x += i.size[0]
    # plot combined PIL images
    #plt.tight_layout(pad=0)
    if not kwargs.get('no_title', False):
        # set plot's title before displaying the image
        fsp = plt.gcf().subplotpars
        # important note: the newline is added to reserve space for the subtitle because tight_layout only considers
        #                  the title and some other elements, NOT the subtitle
        plt.title(f"Byte plot of {binary.type}: {binary.basename}", y=1.1, **kwargs['title-font'])
        plt.suptitle(binary.hash, x=(fsp.right+fsp.left)/2, y=.88, **kwargs['annotation-font'])  # y=fsp.top*1.02
    plt.axis("off")
    plt.imshow(img)

