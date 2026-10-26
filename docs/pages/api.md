# Python API

The `exeplot` package exposes one function per plot type:

```python
from exeplot import byte, diff, entropy, graph, nested_pie, pie
```

`exeplot.__all__` lists the plot types that are available. `graph` is only
there when its [extra](installation.md#the-graph-extra) is installed.

Every function draws the plot, saves it and returns the **list of saved
image paths**:

```python
>>> from exeplot import pie
>>> pie("calc_packed.exe", donut=True, img_format="svg")
['calc_packed_pie.svg']
```

---

## Common keyword arguments

All plot functions accept these keyword arguments, on top of their own:

| Argument | Default | Description |
|---|---|---|
| `no_title` | `False` | Do not display the title. |
| `no_legend` | `False` | Do not display the legend. |
| `img_name` | `<name>_<plot>` | Path of the image, with or without its extension. Missing folders are created. |
| `img_format` | `png` | Image format and file extension. |
| `dpi` | `300` | Resolution of the image. |
| `transparent` | `False` | Save with a transparent background. |
| `bbox_inches` | `"tight"` | Bounding box of the saved figure. |
| `grayscale` | `False` | Use the `gray` colormap instead of `colormap_main`. |
| `interactive_mode` | `False` | Open a Python console before saving (see [Interactive mode](cli/exeplot.md#common-options)). |

The defaults come from the [configuration file](cli/exeplot.md#configuration-file)
when it sets them.

### Fonts

`font_family` and `font_size` set the base font. Each text element can then
be changed with `<element>_font_family`, `<element>_font_size` and
`<element>_font_weight`, where `<element>` is one of:

| Element | Default size | Default weight | Used for |
|---|---|---|---|
| `title` | 1.6 × `font_size` | `bold` | The title. |
| `suptitle` | 1.2 × `font_size` | `normal` | The subtitle. |
| `annotation` | 0.5 × `font_size` | `normal` | The hash under the title. |
| `xlabel`, `ylabel` | `font_size` | `normal` | Axis labels and pie percentages. |

```python
entropy("a.exe", title_font_size=20, title_font_weight="normal")
```

---

## `byte`

```python
byte(executable, height=600, grayscale=False, **kwargs)
```

| Parameter | Description |
|---|---|
| `executable` | Path to the executable. |
| `height` | Height of the image, in pixels. |
| `grayscale` | One byte per gray pixel, instead of three bytes per RGB pixel. |

---

## `entropy`

```python
entropy(*filenames, labels=None, sublabel=None, scale=False, target=None, **kwargs)
```

| Parameter | Description |
|---|---|
| `*filenames` | Paths to one or more executables. At least one is required (`ValueError` otherwise). |
| `labels` | One label per file. Each item is a string, or a function that takes the [data dictionary](#entropy-data) of the file and returns a string. |
| `sublabel` | Text under each label: a key of `exeplot.plots.__common__.SUBLABELS` (`"ep"`, `"size"`, `"size-ep"`, `"size-ent"`, `"size-ep-ent"`), or a function of the data dictionary. `None` (the default here, unlike the CLI) shows no sublabel. Sublabels that mention the entry point are skipped for files without one. |
| `scale` | Draw every file on the scale of the first one. |
| `target` | Name shown in the title, instead of the first file name. |
| `n_samples` | Number of entropy samples per file (default `2048`). |
| `legend_location` | Matplotlib location of the legend (default `"lower center"`). A location ending with `left` or `right` puts the legend on one column, beside the plot. |
| `no_entrypoint` | Do not display the entry point. |
| `no_label` | Do not display the labels, sublabels and section names. |

```python
entropy("calc_orig.exe", "calc_packed.exe", scale=True,
        labels=["Original", lambda d: f"Packed ({d['type']})"], sublabel="size-ent")
```

### Entropy data

`labels` and `sublabel` functions receive the dictionary returned by
`exeplot.plots.entropy.data(executable)`:

| Key | Description |
|---|---|
| `name`, `hash`, `size`, `type` | File name, SHA-256, size in bytes and format (`PE`, `ELF` or `MachO`). |
| `entropy` | The entropy samples, divided by 8 (so from 0 to 1). |
| `entropy*` | The entropy of the whole file, in bits per byte. |
| `ep` | `(position on the plot, file offset, section name)` of the entry point, or `None`. |
| `sections` | `(start, end, name, average entropy)` tuples, in samples. |

---

## `pie`

```python
pie(executable, donut=False, **kwargs)
```

| Parameter | Description |
|---|---|
| `executable` | Path to the executable. |
| `donut` | Draw a donut instead of a pie. |
| `no_label` | Do not display the percentages. |

---

## `nested_pie`

```python
nested_pie(executable, **kwargs)
```

| Parameter | Description |
|---|---|
| `executable` | Path to the executable. |

---

## `diff`

```python
diff(executable, executable2, legend1="", legend2="", **kwargs)
```

| Parameter | Description |
|---|---|
| `executable` | Path to the reference executable. |
| `executable2` | Path to the modified executable. Both must have the same format (`ValueError` otherwise). |
| `legend1`, `legend2` | Labels of the two bars. The file names are used when empty. |
| `no_colormap` | Do not display the color bar. `no_colorbar` is accepted too. |

---

## `graph`

```python
graph(executable, algorithm="fast", engine="default", **kwargs)
```

| Parameter | Description |
|---|---|
| `executable` | Path to the executable. |
| `algorithm` | `"fast"` (`CFGFast`) or `"emulated"` (`CFGEmulated`). |
| `engine` | `"default"`, `"pcode"` or `"vex"`. |

---

## Utilities: `exeplot.utils`

```python
from exeplot.utils import ensure_str, human_readable_size, ngrams_counts, ngrams_distribution, shannon_entropy
```

| Function | Description |
|---|---|
| `shannon_entropy(data)` | Shannon entropy of a byte string, in bits per byte (0 to 8). |
| `human_readable_size(size, precision=0)` | Formats a size in bytes, for example `human_readable_size(1536, 1)` gives `"1.5KB"`. |
| `ensure_str(s, encoding="utf-8", errors="strict")` | Decodes `bytes`, falling back to Latin-1. Raises `TypeError` for other types. |
| `ngrams_counts(byte_obj, n=1, step=1)` | Sorted list of `(n-gram, count)` pairs, for `n` from 1 to 3. `byte_obj` is a byte string, or an object with a `bytes` attribute, whose result is then cached on the object. |
| `ngrams_distribution(byte_obj, n=1, step=1, n_most_common=None, n_exclude_top=0, exclude=None)` | Like `ngrams_counts`, keeping the `n_most_common` n-grams after leaving out the `n_exclude_top` most common ones and the n-grams in `exclude`. |

---

## Constants

| Name | Description |
|---|---|
| `exeplot.__version__` | Package version, read from `VERSION.txt`. |
| `exeplot.config` | The plot style options (see the [configuration file](cli/exeplot.md#configuration-file)). It is updated from `~/.exeplot.conf` before each plot. |
| `exeplot.logger` | The `exeplot` logger. |
