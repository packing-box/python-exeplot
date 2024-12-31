# -*- coding: UTF-8 -*-
from .__info__ import __author__, __copyright__, __email__, __license__, __source__, __version__
from .__init__ import *
from .__init__ import __all__ as _plots


def _parser(name, description, examples):
    from argparse import ArgumentParser, RawTextHelpFormatter
    descr = f"{name} {__version__}\n\nAuthor   : {__author__} ({__email__})\nCopyright: {__copyright__}\nLicense  :" \
            f" {__license__}\nSource   : {__source__}\n\n{description}.\n\n"
    examples = [f"exeplot {e}" if not e.startswith("exeplot ") else e for e in examples]
    return ArgumentParser(description=descr, formatter_class=RawTextHelpFormatter, add_help=False,
                          epilog="usage examples:\n  " + "\n  ".join(examples) if len(examples) > 0 else None)


def _setup(parser):
    args = parser.parse_args()
    if hasattr(args, "verbose"):
        import logging
        logging.basicConfig(level=[logging.INFO, logging.DEBUG][args.verbose])
    return args


def main():
    from os import makedirs
    parser = _parser("Exeplot", "This tool allows to plot executable sample(s) in different ways",
                     ["byte binary.exe", "entropy binary1.exe binary2.exe --scale"])
    extra = parser.add_argument_group("extra arguments")
    extra.add_argument("-h", "--help", action="help", help="show this help message and exit")
    extra.add_argument("-v", "--verbose", action="store_true", help="display debug information (default: False)")
    plots = parser.add_subparsers(dest="type", help="plot type")
    for plot in _plots:
        plot_func = globals()[plot]
        plot_parser = plot_func.__args__(plots.add_parser(plot, help=plot_func.__doc__.strip()))
        """
        opt = plot_parser.add_argument_group("style arguments")
        for a in ["title-font", "suptitle-font", "xlabel-font", "ylable-font"]:
            for i in ["family", "size", "weight"]:
                kw = {'help': "", 'metavar': i.upper()}
                if i == "size":
                    kw['type'] = int
                elif i == "weight":
                    kw['choices'] = ("normal", "bold", "italic")
                opt.add_argument(f"--{a}-{i}", **kw)
        """
    args = _setup(parser)
    exe = args.executable if isinstance(args.executable, list) else [args.executable]
    delattr(args, "executable")
    globals()[args.type](*exe, **vars(args))

