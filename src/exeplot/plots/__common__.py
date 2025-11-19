# -*- coding: UTF-8 -*-
import os
from functools import cached_property
from statistics import mean

from ..utils import *


CACHE_DIR = os.path.expanduser("~/.exeplot")
# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = {
    None:       ["salmon", "gold", "plum", "darkkhaki", "orchid", "sandybrown", "purple", "khaki", "peru", "thistle"],
    'header':   "black",
    'headers':  "black",
    'overlay':  "lightgray",
    'section header':  "black",
    'section headers': "black",
    '<undef>':  "lightgray",
    # common
    'text':     "darkseagreen",   # code
    'data':     "skyblue",        # initialized data
    'bss':      "steelblue",      # block started by symbol (uninitialized data)
    # PE
    'rdata':    "cornflowerblue", # read-only data
    'rsrc':     "royalblue",      # resources
    'tls':      "slateblue",      # thread-local storage
    'edata':    "turquoise",      # export data
    'idata':    "darkturquoise",  # import data
    'reloc':    "crimson",        # base relocations table
    # ELF
    'init':     "lightgreen",     # runtime initialization instructions
    'fini':     "yellowgreen",    # process termination code
    'data1':    "skyblue",        # initialized data (2)
    'rodata':   "cornflowerblue", # read-only data
    'rodata1':  "cornflowerblue", # read-only data (2)
    'symtab':   "royalblue",      # symbol table
    'strtab':   "navy",           # string table
    'strtab1':  "navy",           # string table (2)
    'dynamic':  "crimson",        # dynamic linking information
    # Mach-O
    'cstring':  "navy",           # string table
    'const':    "cornflowerblue", # read-only data
    'literal4': "blue",           # 4-byte literal values
    'literal4': "mediumblue",     # 8-byte literal values
    'common':   "royalblue",      # uninitialized imported symbol definitions
}
MIN_ZONE_WIDTH = 3  # minimum number of samples on the entropy plot for a section (so that it can still be visible even
                    #  if it is far smaller than the other sections)
N_SAMPLES = 2048
SHADOW = {'shade': .3, 'ox': .005, 'oy': -.005, 'linewidth': 0.}
SUBLABELS = {
    'ep':          lambda d: "EP at 0x%.8x in %s" % d['ep'][1:],
    'size':        lambda d: "Size = %s" % human_readable_size(d['size'], 1),
    'size-ep':     lambda d: "Size = %s\nEP at 0x%.8x in %s" % \
                             (human_readable_size(d['size'], 1), d['ep'][1], d['ep'][2]),
    'size-ent':    lambda d: "Size = %s\nAverage entropy: %.2f\nOverall entropy: %.2f" % \
                             (human_readable_size(d['size'], 1), mean(d['entropy']) * 8, d['entropy*']),
    'size-ep-ent': lambda d: "Size = %s\nEP at 0x%.8x in %s\nAverage entropy: %.2f\nOverall entropy: %.2f" % \
                             (human_readable_size(d['size'], 1), d['ep'][1], d['ep'][2], mean(d['entropy']) * 8,
                              d['entropy*']),
}


class Binary:
    def __init__(self, path, **kwargs):
        from lief import logging, parse
        self.path = os.path.abspath(str(path))
        self.basename = os.path.basename(self.path)
        self.stem = os.path.splitext(os.path.basename(self.path))[0]
        l = kwargs.get('logger')
        logging.enable() if l and l.level <= 10 else logging.disable()
        # compute other characteristics using LIEF (catch warnings from stderr)
        tmp_fd, null_fd = os.dup(2), os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 2)
        self.__binary = parse(self.path)
        os.dup2(tmp_fd, 2)  # restore stderr
        os.close(null_fd)
        if self.__binary is None:
            raise TypeError("Not an executable")
        self.type = str(type(self.__binary)).split(".")[2]
        if self.type not in ["ELF", "MachO", "PE"]:  # pragma: no cover
            raise OSError("Unknown format")
    
    def __getattr__(self, name):
        try:
            return super(Binary, self).__getttr__(name)
        except AttributeError:
            return getattr(self.__binary, name)
    
    def __iter__(self):
        for _ in self.__sections_data():
            yield _
    
    def __str__(self):
        return self.path
    
    def __get_ep_and_section(self):
        b = self.__binary
        try:
            if self.type in ["ELF", "MachO"]:
                self.__ep = b.virtual_address_to_offset(b.entrypoint)
                self.__ep_section = b.section_from_offset(self.__ep)
            elif self.type == "PE":
                self.__ep = b.rva_to_offset(b.optional_header.addressof_entrypoint)
                self.__ep_section = b.section_from_rva(b.optional_header.addressof_entrypoint)
        except (AttributeError, TypeError):  # pragma: no cover
            self.__ep, self.__ep_section = None, None
    
    def __sections_data(self):
        b = self.__binary
        # create a first section for the headers
        if self.type == "PE":
            h_len = b.sizeof_headers
        elif self.type == "ELF":
            h_len = b.header.header_size + b.header.program_header_size * b.header.numberof_segments
        elif self.type == "MachO":
            h_len = [28, 32][str(b.header.magic)[-3:] == "_64"] + b.header.sizeof_cmds
        yield 0, f"[0] Header ({human_readable_size(h_len)})", 0, h_len, "black"
        # then handle binary's sections
        color_cursor, i = 0, 1
        for section in sorted(b.sections, key=lambda s: s.offset):
            if section.name == "" and section.size == 0 and len(section.content) == 0:
                continue
            try:
                c = COLORS[self.section_names[section.name].lower().lstrip("._").strip("\x00\n ")]
            except KeyError:
                co = COLORS[None]
                c = co[color_cursor % len(co)]
                color_cursor += 1
            start, end = section.offset, section.offset + section.size
            yield i, f"[{i}] {self.section_names[section.name]} ({human_readable_size(end - start)})", start, end, c
            i += 1
        # sections header at the end for ELF files
        if self.type == "ELF":
            start, end = end, end + b.header.section_header_size * b.header.numberof_sections
            yield i, f"[{i}] Section Header ({human_readable_size(end - start)})", start, end, "black"
            i += 1
        # finally, handle the overlay
        start, end = self.size - b.overlay.nbytes, self.size
        yield i, f"[{i}] Overlay ({human_readable_size(end - start)})", start, self.size, "lightgray"
        i += 1
        yield i, f"TOTAL: {human_readable_size(self.size)}", None, None, "white"
    
    def __segments_data(self):
        b = self.__binary
        if self.type == "PE":
            return  # segments only apply to ELF and MachO
        elif self.type == "ELF":
            for i, s in enumerate(sorted(b.segments, key=lambda x: (x.file_offset, x.physical_size))):
                yield i, f"[{i}] {str(s.type).split('.')[1]} ({human_readable_size(s.physical_size)})", \
                      s.file_offset, s.file_offset+s.physical_size, "lightgray"
        elif self.type == "MachO":
            for i, s in enumerate(sorted(b.segments, key=lambda x: (x.file_offset, x.file_size))):
                yield i, f"[{i}] {s.name} ({human_readable_size(s.file_size)})", \
                      s.file_offset, s.file_offset+s.file_size, "lightgray"
    
    def _data(self, segments=False, overlap=False):
        data = [self.__sections_data, self.__segments_data][segments]
        # generator for getting next items, taking None value into account for the start offset
        def _nexts(n):
            for j, t, s, e, c in data():
                if j <= n or s is None:
                    continue
                yield j, t, s, e, c
        # collect data, including x positions, [w]idths, [t]exts and [c]olors
        x, w, t, c, cursors, legend, layer = {0: []}, {0: []}, {0: []}, {0: []}, {0: 0}, {'colors': [], 'texts': []}, 0
        for i, text, start, end, color in data():
            legend['colors'].append(color), legend['texts'].append(text)
            if start is None or end is None:
                continue
            end = min(self.size, end)
            width = end - start
            if overlap:
                # set the layer first
                for n in range(layer + 1):
                    if start >= cursors[n]:
                        layer = n
                        break
                if start < cursors[layer]:
                    layer += 1
                # create layer data if layer does not exist yet
                if layer not in x:
                    x[layer], w[layer], t[layer], c[layer], cursors[layer] = [], [], [], [], 0
                # if not starting at layer's cursor, fill up to start index with a blank section
                if start > cursors[layer]:
                    x[layer].append(cursors[layer]), w[layer].append(start - cursors[layer])
                    t[layer].append("_"), c[layer].append("white")
                # then add the current section
                cursors[layer] = end
                x[layer].append(start), w[layer].append(width), t[layer].append(text), c[layer].append(color)
            else:
                # adjust "end" if section overlap
                for j, _, start2, _, _ in _nexts(i):
                    end = min(start2, end)
                    width = end - start
                    break
                x[0].append(start), w[0].append(width), t[0].append(text), c[0].append(color)
                # add a blank if the next section does not start from the end
                for j, _, start2, _, _ in _nexts(i):
                    if j <= i or start2 is None:
                        continue
                    if start2 > end:
                        x[0].append(end), w[0].append(start2 - end), t[0].append("_"), c[0].append("white")
                    break
        for i in range(len(x)):
            if len(x[i]) > 0:
                end = x[i][-1] + w[i][-1]
                if end < self.size:
                    x[i].append(end), w[i].append(self.size-end), t[i].append("_"), c[i].append("white")
                if sum(w[i]) != self.size:
                    for start, width, section, color in zip(x[i], w[i], t[i], c[i]):
                        print(f"LAYER {i}", section, color, start, width)
                    raise ValueError(f"Sizes do not match at layer {i} ({sum(w[i])} != {self.size})")
                yield i, x[i], w[i], t[i], c[i], legend
    
    @cached_property
    def entrypoint(self):
        self.__get_ep_and_section()
        return self.__ep
    
    @cached_property
    def entrypoint_section(self):
        self.__get_ep_and_section()
        return self.__ep_section
    
    @cached_property
    def hash(self):
        from hashlib import sha256
        m = sha256()
        m.update(self.rawbytes)
        return m.hexdigest()
    
    @property
    def rawbytes(self):
        with open(self.path, "rb") as f:
            self.__size = os.fstat(f.fileno()).st_size
            return f.read()
    
    @cached_property        
    def section_names(self):
        names = {s.name: ensure_str(s.name).strip("\x00") or "<empty>" for s in self.__binary.sections}
        # names from string table only applies to PE
        if self.type != "PE":
            return names
        # start parsing section names
        from re import match
        if all(match(r"/\d+$", n) is None for n in names.keys()):  # pragma: no cover
            return names
        real_names = {}
        str_table_offset = self.__binary.header.pointerto_symbol_table + self.__binary.header.numberof_symbols * 18
        with open(self.path, "rb") as f:
            for n in names:
                if match(r"/\d+$", n):
                    f.seek(str_table_offset + int(n[1:]))
                    n2 = b"".join(iter(lambda: f.read(1), b'\x00')).decode("utf-8", errors="ignore")
                else:
                    n2 = n
                real_names[n] = n2
        return real_names
    
    @property
    def size(self):
        s = self.__binary.original_size
        try:
            if s != self.__size:
                raise ValueError("LIEF parsed size does not match actual size")
        except AttributeError:
            pass
        return s

