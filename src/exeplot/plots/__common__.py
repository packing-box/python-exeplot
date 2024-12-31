# -*- coding: UTF-8 -*-
import os
from functools import cached_property
from statistics import mean


# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = {
    None:       ["salmon", "gold", "plum", "darkkhaki", "orchid", "sandybrown", "purple", "khaki", "peru", "thistle"],
    'headers':  "black",
    'overlay':  "lightgray",
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
SUBLABELS = {
    'ep':          lambda d: "EP at 0x%.8x in %s" % d['entrypoint'][1:],
    'size':        lambda d: "Size = %s" % _human_readable_size(d['size'], 1),
    'size-ep':     lambda d: "Size = %s\nEP at 0x%.8x in %s" % \
                             (_human_readable_size(d['size'], 1), d['entrypoint'][1], d['entrypoint'][2]),
    'size-ep-ent': lambda d: "Size = %s\nEP at 0x%.8x in %s\nAverage entropy: %.2f\nOverall entropy: %.2f" % \
                             (_human_readable_size(d['size'], 1), d['entrypoint'][1], d['entrypoint'][2],
                              mean(d['entropy']) * 8, d['entropy*']),
}


def _ensure_str(s, encoding='utf-8', errors='strict'):
    if isinstance(s, bytes):
        try:
            return s.decode(encoding, errors)
        except:
            return s.decode("latin-1")
    elif not isinstance(s, (str, bytes)):
        raise TypeError("not expecting type '%s'" % type(s))
    return s


def _human_readable_size(size, precision=0):
    i, units = 0, ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    while size >= 1024 and i < len(units):
        i += 1
        size /= 1024.0
    return "%.*f%s" % (precision, size, units[i])


class Binary:
    def __init__(self, path, **kwargs):
        from lief import logging, parse
        self.path = str(path)
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
        if self.type not in ["ELF", "MachO", "PE"]:
            raise OSError("Unknown format")
    
    def __getattr__(self, name):
        try:
            return super(Binary, self).__getttr__(name)
        except AttributeError:
            return getattr(self.__binary, name)
    
    def __str__(self):
        return self.path
    
    def __get_ep_and_section(self):
        try:
            if self.type in ["ELF", "MachO"]:
                self.__ep = self.__binary.virtual_address_to_offset(self.__binary.entrypoint)
                self.__ep_section = self.__binary.section_from_offset(self.__ep)
            elif self.type == "PE":
                self.__ep = self.__binary.rva_to_offset(self.__binary.optional_header.addressof_entrypoint)
                self.__ep_section = self.__binary.section_from_rva(self.__binary.optional_header.addressof_entrypoint)
        except (AttributeError, TypeError):
            self.__ep, self.__ep_section = None, None
    
    @cached_property
    def entrypoint(self):
        self.__get_ep_and_section()
        return self.__ep
    
    @cached_property
    def entrypoint_section(self):
        self.__get_ep_and_section()
        return self.__ep_section
    
    @property
    def rawbytes(self):
        with open(self.path, "rb") as f:
            self.__size = os.fstat(f.fileno()).st_size
            return f.read()
    
    @cached_property        
    def section_names(self):
        __sn = lambda s: _ensure_str(s).strip("\x00") or _ensure_str(s) or "<empty>"
        names = {s.name: __sn(s.name) for s in self.__binary.sections}
        # names from string table only applies to PE
        if self.type != "PE":
            return names
        # start parsing section names
        from re import match
        if all(match(r"/\d+$", n) is None for n in names.keys()):
            return names
        real_names = {}
        with open(self.path, "rb") as f:
            for n in names:
                if match(r"/\d+$", n):
                    f.seek(string_table_offset + int(name[1:]))
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

