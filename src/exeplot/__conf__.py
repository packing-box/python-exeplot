# -*- coding: UTF-8 -*-
import logging
import matplotlib.pyplot as plt
import numpy
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
    'shadow':         True,
    'style':          "default",
    'transparent':    False,
}

numpy.int = numpy.int_  # dirty fix to "AttributeError: module 'numpy' has no attribute 'int'."


def check_imports(*names):
    import warnings
    from inspect import currentframe
    glob = currentframe().f_back.f_globals
    for name in names:
        try:
            __import__(name)
            glob['_IMP'] = True & glob.get('_IMP', True)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"{name} import failed: {e} ({type(e).__name__})", ImportWarning)
            glob['_IMP'] = False


def configure():  # pragma: no cover
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
    kw['annotation-font'] = {'fontfamily': kw.pop('suptitle_font_family', config['font_family']),
                           'fontsize': kw.pop('suptitle_font_size', int(config['font_size'] * .5)),
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
        import matplotlib.pyplot as plt
        from os import makedirs
        from os.path import basename, dirname, splitext
        from .plots.__common__ import Binary
        plot_type = f.__globals__['__name__'].split(".")[-1]
        logger.info(f"Preparing {plot_type} plot data...")
        configure()
        kw = configure_fonts(**kw)
        imgs = f(*a, **kw)
        r = []
        kw_plot = {k: kw.get(k, config[k]) for k in ["bbox_inches", "dpi", "transparent"]}
        for img in (imgs if isinstance(imgs, (list, tuple, type(x for x in []))) else [imgs]):
            img = img or kw.get('img_name') or f"{splitext(basename(a[0]))[0]}_{plot_type}"
            if not img.endswith(ext := "." + kw.get('img_format', config['img_format'])):
                img += ext
            makedirs(dirname(img) or ".", exist_ok=True)
            if kw.get('interactive_mode', False):  # pragma: no cover
                from code import interact
                logger.info(f"{img}: use 'plt.savefig(img, **kw_plot)' to save the figure")
                ns = {k: v for k, v in globals().items()}
                ns.update(locals())
                interact(local=ns)
            logger.info(f"Saving to {img}...")
            plt.savefig(img, **kw_plot)
            logger.debug(f"> saved to {img}...")
            r.append(img)
            plt.clf()
            plt.close()
        return r
    return _wrapper

