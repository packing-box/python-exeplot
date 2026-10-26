# Installation

## Requirements

- Python **3.10 or later**. CI runs on 3.10 to 3.13.
- [`lief`](https://pypi.org/project/lief/) and
  [`matplotlib`](https://pypi.org/project/matplotlib/), which pip installs
  automatically.

## From PyPI

```console
$ pip install exeplot
```

This installs the `exeplot` Python package and the `exeplot` console script,
with every plot type except [`graph`](cli/exeplot.md#graph).

### The `graph` extra

The `graph` plot recovers the control flow graph with
[angr](https://angr.io), which is a large dependency, so it is optional:

```console
$ sudo apt-get install graphviz libgraphviz-dev   # needed to build pygraphviz
$ pip install "exeplot[graph]"
```

If one of these dependencies cannot be imported, ExePlot still works, but
the `graph` plot is not available: it is missing from `exeplot --help` and
from `exeplot.__all__`. Import ExePlot with warnings enabled to see why:

```console
$ python -W always::ImportWarning -c "import exeplot"
```

!!! note "Python 3.10 and 3.11"
    The last angr release for Python 3.10 and 3.11 (9.2.x) does not pin
    `bitstring` and `pycparser`, whose latest versions break it. The `graph`
    extra therefore requires `bitstring<5` and `pycparser<3.12` on these
    Python versions.

## From source

```console
$ git clone https://github.com/packing-box/python-exeplot
$ cd python-exeplot
$ pip install .
```

For development, use an editable install with the `graph` extra:

```console
$ pip install -e ".[graph]"
```

## Checking the installation

```console
$ exeplot --help
usage: exeplot [-h] [-v] {byte,diff,entropy,graph,nested_pie,pie} ...

Exeplot 0.5.5
...
```

```python
>>> import exeplot
>>> exeplot.__version__
'0.5.5'
>>> exeplot.__all__
['byte', 'diff', 'entropy', 'graph', 'nested_pie', 'pie']
```
