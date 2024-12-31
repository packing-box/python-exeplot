# -*- coding: UTF-8 -*-
import logging
import matplotlib.pyplot as plt
from functools import wraps


logger = logging.getLogger("exeplot")
config = {
    'bbox_inches':    "tight",
#    'colormap_main':  "RdYlGn_r",
#    'colormap_other': "jet",
    'dpi':            300,
    'font_family':    "serif",
    'font_size':      10,
    'img_format':     "png",
    'style':          "default",
#    'transparent':    False,
}


def configure():
    from configparser import ConfigParser
    from os.path import exists, expanduser
    path = expanduser("~/.exeplot.conf")
    if exists(path):
        conf = ConfigParser()
        try:
            conf.read(path)
        except:
            raise ValueError("invalid configuration file (~/.exeplot.conf)")
        # overwrite config's default options
        for option in conf['Plot style']._options():
            config[option] = conf['Plot style'][option]
    plt.rcParams['font.family'] = config['font_family']


def configure_fonts(**kw):
    import matplotlib
    matplotlib.rc('font', **{k.split("_")[1]: kw.pop(k, config[k]) for k in ['font_family', 'font_size']})
    kw['title-font'] = {'fontfamily': kw.pop('title_font_family', config['font_family']),
                        'fontsize': kw.pop('title_font_size', int(config['font_size'] * 1.6)),
                        'fontweight': kw.pop('title_font_weight', "bold")}
    kw['suptitle-font'] = {'fontfamily': kw.pop('suptitle_font_family', config['font_family']),
                           'fontsize': kw.pop('suptitle_font_size', int(config['font_size'] * 1.2)),
                           'fontweight': kw.pop('suptitle_font_weight', "normal")}
    for p in "xy":
        kw[f'{p}label-font'] = {'fontfamily': kw.pop(f'{p}label_font_family', config['font_family']),
                                'fontsize': kw.pop(f'{p}label_font_size', config['font_size']),
                                'fontweight': kw.pop(f'{p}label_font_weight', "normal")}
    kw['config'], kw['logger'] = config, logger
    return kw


def save_figure(f):
    """ Decorator for computing the path of a figure and plotting it, given the filename returned by the wrapped
         function ; put it in the "figures" subfolder of the current experiment's folder if relevant. """
    @wraps(f)
    def _wrapper(*a, **kw):
        from os import makedirs
        from os.path import basename, dirname, splitext
        logger.info("Preparing plot data...")
        configure()
        imgs = f(*a, **configure_fonts(**kw))
        ext = "." + kw.get('img_format', config['img_format'])
        kw_plot = {k: kw.get(k, config[k]) for k in ["bbox_inches", "dpi"]}
        for img in (imgs if isinstance(imgs, (list, tuple, type(x for x in []))) else [imgs]):
            if img is None:
                img = kw.get('img_name') or splitext(basename(a[0]))[0]
            if not img.endswith(ext):
                img += ext
            if d := dirname(img):
                makedirs(d, exist_ok=True)
            if kw.get('interactive_mode', False):
                from code import interact
                ns = {k: v for k, v in globals().items()}
                ns.update(locals())
                l.info(f"{img}: use 'plt.savefig(img, **kw_plot)' to save the figure")
                interact(local=ns)
            logger.info(f"Saving to {img}...")
            plt.savefig(img, **kw_plot)
            logger.debug(f"> saved to {img}...")
    return _wrapper

