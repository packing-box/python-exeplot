<p align="center" id="top"><img src="https://github.com/packing-box/python-exeplot/raw/main/docs/pages/img/logo.png"></p>
<h1 align="center">ExePlot <a href="https://twitter.com/intent/tweet?text=ExePlot%20-%20Plot%20executable%20samples%20easy.%0D%0ALibrary%20for%20plotting%20executable%20samples%20supporting%20multiple%20formats.%0D%0Ahttps%3a%2f%2fgithub%2ecom%2fpacking-box%2fpython-exeplot%0D%0A&hashtags=python,programming,executable-samples,plot"><img src="https://img.shields.io/badge/Tweet--lightgrey?logo=twitter&style=social" alt="Tweet" height="20"/></a></h1>
<h3 align="center">Search for samples from various malware databases.</h3>

[![PyPi](https://img.shields.io/pypi/v/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)
[![Read The Docs](https://readthedocs.org/projects/python-exeplot/badge/?version=latest)](https://python-exeplot.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/packing-box/python-exeplot/actions/workflows/python-package.yml/badge.svg)](https://github.com/packing-box/python-exeplot/actions/workflows/python-package.yml)
[![Coverage Status](https://raw.githubusercontent.com/packing-box/python-exeplot/main/docs/coverage.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)
[![Known Vulnerabilities](https://snyk.io/test/github/packing-box/python-exeplot/badge.svg?targetFile=requirements.txt)](https://snyk.io/test/github/packing-box/python-exeplot?targetFile=requirements.txt)
[![License](https://img.shields.io/pypi/l/exeplot.svg)](https://pypi.python.org/pypi/exeplot/)

This library implements multiple plot types for illustrating executable samples. It currently supports the PE, ELF and Mach-O formats, relying on [`lief`](https://github.com/lief-project/LIEF) for abstracting them.

```sh
$ pip install exeplot
```

## :sunglasses: Usage Examples

Draw a byte plot of `calc_packed.exe`:

```sh
$ exeplot byte calc_packed.exe
```

![Byte plot of `calc_packed.exe`](https://github.com/packing-box/python-exeplot/blob/main/docs/pages/img/calc_packed_byte.png?raw=true)

Draw a pie plot of `calc_packed.exe`:

```sh
$ exeplot pie calc_packed.exe
```

![Pie plot of `calc_packed.exe`](https://github.com/packing-box/python-exeplot/blob/main/docs/pages/img/calc_packed_pie.png?raw=true)

Draw a nested pie plot of `calc_packed.exe`:

```sh
$ exeplot nested_pie calc_packed.exe
```

![Nested pie plot of `calc_packed.exe`](https://github.com/packing-box/python-exeplot/blob/main/docs/pages/img/calc_packed_nested_pie.png?raw=true)

Draw a stacked and scaled entropy plot of `calc_orig.exe` and `calc_packed.exe`:

```sh
$ exeplot entropy calc_orig.exe calc_packed.exe
```

![Entropy plot of `calc_orig.exe` and `calc_packed.exe`](https://github.com/packing-box/python-exeplot/blob/main/docs/pages/img/calc_orig_entropy.png?raw=true)


## :clap: Supporters

[![Stargazers repo roster for @packing-box/python-exeplot](https://reporoster.com/stars/dark/packing-box/python-exeplot)](https://github.com/packing-box/python-exeplot/stargazers)

[![Forkers repo roster for @packing-box/python-exeplot](https://reporoster.com/forks/dark/packing-box/python-exeplot)](https://github.com/packing-box/python-exeplot/network/members)

<p align="center"><a href="#top"><img src="https://img.shields.io/badge/Back%20to%20top--lightgrey?style=social" alt="Back to top" height="20"/></a></p>
