# How it works

## Overview

```text
 exeplot <plot> ...  ──▶  exeplot.<plot>(...)  ──▶  @save_figure
                                                      │
               ~/.exeplot.conf ──configure()──────────┤
                                                      ▼
            plots/<plot>.py: plot() ──Binary (LIEF)──▶ matplotlib figure ──savefig──▶ <name>_<plot>.png
```

1. **Configure.** The `save_figure` decorator reads `~/.exeplot.conf` and
   computes the fonts from the keyword arguments.
2. **Parse.** The plot function wraps the executable in a `Binary`, which
   parses it with LIEF and lists its sections, headers and overlay.
3. **Draw** the plot with matplotlib (and PIL for the byte plot).
4. **Save.** `save_figure` saves every figure the plot function made, and
   returns the paths.

## Plot discovery

`exeplot/plots/__init__.py` imports every module of the `plots` folder whose
name does not start with `_`. A module becomes a plot type if it has:

- a `plot` function, decorated with `@save_figure`: its docstring is the
  help text of the subcommand;
- an `arguments(parser)` function that adds the subcommand's own arguments
  to an `argparse` parser and returns it.

The module name is the name of the plot type, of the Python function and of
the subcommand. A module can call `check_imports("module", ...)` at import
time: if one of these modules cannot be imported, an `ImportWarning` is
issued and the plot type is left out. `graph` does this for angr, networkx
and pygraphviz.

## The `Binary` wrapper

`exeplot.plots.__common__.Binary` parses the file with `lief.parse`,
silencing LIEF's own messages, and raises `TypeError("Not an executable")`
if LIEF cannot parse it. Unknown attributes are looked up on the LIEF
object, so `binary.sections` or `binary.header` work as in LIEF.

Iterating over a `Binary` yields `(index, label, start, end, color)` for
each part of the file, in file order:

| Part | Size |
|---|---|
| Headers | PE: `SizeOfHeaders`. ELF: the ELF header and the program header table. Mach-O: the Mach header (28 or 32 bytes) and the load commands. |
| Sections | Sorted by file offset. |
| Section header table | ELF only, after the last section. |
| Overlay | The bytes after the last part, as reported by LIEF. |

Each section gets a color from `COLORS`, by its name without leading dots or
underscores (`.text` and `__text` are both `text`). Unknown names cycle
through a fixed list of colors. PE section names of the form `/<n>` are
resolved from the COFF string table.

## Plot types

### Byte plot

The file is cut into pixels: three bytes per RGB pixel, or one byte per
gray pixel with `grayscale`. The pixels fill a square image, padded with
`0xFF`, which is then resized to `height` pixels. The side bar is a smaller
square where each pixel takes the color of the section it belongs to.

### Entropy plot

The file is read once, with a sliding window:

- the plot has `n_samples` points (2048 by default), one every
  `size / n_samples` bytes;
- each point is the Shannon entropy of a window of twice that step, and at
  least 512 bytes, so that small files are not undersampled;
- section bounds are converted to sample positions, and each section is at
  least `MIN_ZONE_WIDTH` (3) samples wide, so that small sections stay
  visible;
- gaps between sections are filled with `<undef>` zones, and the entry point
  is moved inside its section if rounding put it outside.

With `scale`, the number of samples of each file is proportional to its
size compared with the first file.

### Pie and nested pie plots

The parts of the file become wedges sized in bytes. When sections overlap,
`pie` cuts each section at the start of the next one. `nested_pie` instead
puts a section that starts before the end of the previous one on a new ring,
and fills gaps with white wedges. For ELF and Mach-O files, segments are
drawn the same way, as inner rings.

### Diff plot

`difflib.SequenceMatcher` compares the raw bytes of both files. Its opcodes
(`delete`, `replace`, `equal`, `insert`) color the byte ranges of each bar.
If [joblib](https://joblib.readthedocs.io) is installed, the result is
cached in `~/.exeplot`, so plotting the same pair again is fast.

### Graph plot

angr loads the executable without its libraries and recovers its CFG with
`CFGFast` or `CFGEmulated`. The graph is converted to a networkx graph and
drawn with the Kamada-Kawai layout (which needs scipy). Nodes are labeled
with their function name, if any, and address.
