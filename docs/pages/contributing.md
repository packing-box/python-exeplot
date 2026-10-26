# Contributing

Contributions are welcome, especially **new plot types**.

## Reporting issues

Open an issue on [GitHub](https://github.com/packing-box/python-exeplot/issues)
and include:

- the ExePlot version (`exeplot --help` prints it) and the Python version;
- the exact command and its output, ideally with `-v`;
- if possible, a link to (or the hash of) the sample.

## Security

Please do **not** report vulnerabilities in public issues. See the
[security policy](https://github.com/packing-box/python-exeplot/blob/main/SECURITY.md):
report them privately through the repository's *Security* tab
(*Report a vulnerability*).

Dependencies are monitored by Dependabot and audited with
[pip-audit](https://github.com/pypa/pip-audit) by the `security` workflow
(`.github/workflows/security.yml`), on every push and pull request to `main`
and weekly.

## Development setup

```console
$ sudo apt-get install graphviz libgraphviz-dev
$ git clone https://github.com/packing-box/python-exeplot
$ cd python-exeplot
$ pip install -e ".[graph]"
$ pip install pytest pytest-cov
$ pytest --cov=exeplot
```

The tests plot the sample executables of `tests/` (`hello.exe`, `hello.elf`
and `hello.macho`) with every plot type. CI
(`.github/workflows/python-package.yml`) runs them on Python 3.10 to 3.13.

## Adding a plot type

Add a module to `src/exeplot/plots/`, for example `myplot.py`. It is found
automatically (see [Plot discovery](internals.md#plot-discovery)):

```python
from .__common__ import Binary
from ..__conf__ import *


def arguments(parser):
    parser.add_argument("executable", help="executable sample to be plotted")
    return parser


@save_figure
def plot(executable, **kwargs):
    """ draw my plot of the input binary """
    import matplotlib.pyplot as plt
    binary = Binary(executable)
    ...
    if not kwargs.get('no_title', False):
        plt.title(f"My plot of {binary.type}: {binary.basename}", **kwargs['title-font'])
```

`save_figure` passes the fonts (`kwargs['title-font']`, ...) and the
configuration (`kwargs['config']`) to the function, and saves the figure
after it returns. Document the new plot type in
[`exeplot`](cli/exeplot.md) and in the [Python API](api.md).

## Releasing

The version is stored in `src/exeplot/VERSION.txt`. On every push to
`main`, CI checks whether that version is already on PyPI. If not, it
builds the package and publishes it.

## Building this documentation

```console
$ pip install -r docs/requirements.txt
$ mkdocs serve -f docs/mkdocs.yml          # live preview at http://127.0.0.1:8000
$ mkdocs build --strict -f docs/mkdocs.yml # what Read the Docs runs
```

Read the Docs builds the site from `.readthedocs.yaml` at the root of the
repository.
