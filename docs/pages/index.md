# ExePlot

<p align="center"><img src="img/logo.png" alt="ExePlot logo" width="200"></p>

**ExePlot** is a Python library and command-line tool for plotting executable
samples. It supports the **PE**, **ELF** and **Mach-O** formats, and relies on
[LIEF](https://github.com/lief-project/LIEF) to parse them.

[![PyPi](https://img.shields.io/pypi/v/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)
[![Python Versions](https://img.shields.io/pypi/pyversions/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)
[![Build Status](https://github.com/packing-box/python-exeplot/actions/workflows/python-package.yml/badge.svg)](https://github.com/packing-box/python-exeplot/actions/workflows/python-package.yml)
[![Security](https://github.com/packing-box/python-exeplot/actions/workflows/security.yml/badge.svg)](https://github.com/packing-box/python-exeplot/actions/workflows/security.yml)
[![License](https://img.shields.io/pypi/l/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)

## Features

Six plot types, each available as an `exeplot` subcommand and as a Python
function of the same name:

| Plot | What it shows |
|---|---|
| [`byte`](cli/exeplot.md#byte) | The raw bytes as an image, with a side bar of the sections. |
| [`entropy`](cli/exeplot.md#entropy) | The entropy along the file, per section, with the entry point. Several files can be stacked. |
| [`pie`](cli/exeplot.md#pie) | The share of the file taken by each section. |
| [`nested_pie`](cli/exeplot.md#nested_pie) | The sections, including overlapping ones, and the segments (ELF and Mach-O) as rings. |
| [`diff`](cli/exeplot.md#diff) | The byte-wise difference between an executable and a modified version of it. |
| [`graph`](cli/exeplot.md#graph) | The control flow graph (CFG), recovered with [angr](https://angr.io). Needs the `graph` extra. |

![Entropy plot of calc_orig.exe and calc_packed.exe](img/calc_orig_entropy.png)

## Quick start

```console
$ pip install exeplot
$ exeplot entropy calc_orig.exe calc_packed.exe   # writes calc_orig_entropy.png
$ exeplot byte calc_packed.exe --no-title          # writes calc_packed_byte.png
```

From Python:

```python
from exeplot import entropy, pie

pie("calc_packed.exe", donut=True)           # ['calc_packed_pie.png']
entropy("calc_orig.exe", "calc_packed.exe",
        labels=["Original", "Packed"], sublabel="size-ep-ent", scale=True)
```

## Related projects

ExePlot is part of the [packing-box](https://github.com/packing-box) ecosystem:

- [Awesome Executable Packing](https://github.com/packing-box/awesome-executable-packing): a curated list of resources about executable packing.
- [Bintropy](https://github.com/packing-box/bintropy): estimates how likely a binary is to contain compressed or encrypted bytes.
- [Dataset of packed ELF files](https://github.com/packing-box/dataset-packed-elf) and [Dataset of packed PE files](https://github.com/packing-box/dataset-packed-pe).
- [Docker Packing Box](https://github.com/packing-box/docker-packing-box): a Docker image of packers and tools for building datasets of packed executables.
- [DSFF](https://github.com/packing-box/python-dsff): the DataSet File Format library.
- [PEiD](https://github.com/packing-box/peid): detects packers on PE files using signatures.
- [PyPackerDetect](https://github.com/packing-box/pypackerdetect): a heuristic packing detector for PE files.
- [REMINDer](https://github.com/packing-box/reminder): a packing detector based on a simple entropy heuristic.

## License

ExePlot is released under the
[GNU General Public License v3 or later](https://www.gnu.org/licenses/gpl-3.0.html).
