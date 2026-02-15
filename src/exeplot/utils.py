# -*- coding: UTF-8 -*-
import numpy as np
from math import log2
from typing import Optional


__all__ = ["ensure_str", "human_readable_size", "ngrams_counts", "ngrams_distribution", "shannon_entropy"]

shannon_entropy = lambda b: -sum([p*log2(p) for p in [float(ctr)/len(b) for ctr in [b.count(c) for c in set(b)]]]) or 0.


def ensure_str(s: str | bytes, encoding: str = "utf-8", errors: str = "strict") -> str:
    """ Ensure that an input string is decoded. """
    if isinstance(s, bytes):
        try:
            return s.decode(encoding, errors)
        except:
            return s.decode("latin-1")
    elif not isinstance(s, (str, bytes)):
        raise TypeError("not expecting type '%s'" % type(s))
    return s


def human_readable_size(size: int, precision: int = 0) -> str:
    """ Display bytes' size in a human-readable format given a precision. """
    i, units = 0, ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    while size >= 1024 and i < len(units)-1:
        i += 1
        size /= 1024.0
    return "%.*f%s" % (precision, size, units[i])


def ngrams_counts(byte_obj: bytes | object, n: int = 1, step: int = 1) -> dict[bytes, int]:
    """ Output the Counter instance for an input byte sequence or byte object based on n-grams.
         If the input is a byte object, cache the result.
    
    :param byte_obj:      byte sequence ('bytes') or byte object with "bytes" and "size" attributes (i.e. pathlib2.Path)
    :param n: n determining the size of n-grams, defaults to 1
    :param step:          step for sliding the n-grams
    :param start:         number of bytes to start from
    """
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")
    if step <= 0:
        raise ValueError("step must be positive")
    if isinstance(byte_obj, bytes) or hasattr(byte_obj, "bytes"):
        a = np.frombuffer(data := byte_obj if isinstance(byte_obj, bytes) else byte_obj.bytes, dtype=np.uint8)
        l = a.size
        if l < n:
            return {}
        if n == 1:
            counts = {b.to_bytes(1, "big"): int(c) for b, c in \
                      enumerate(np.bincount(np.frombuffer(data, dtype=np.uint8)))}
        else:
            end = (m := (l - n) // step + 1) * step
            grams = np.stack((a[0:end:step], a[1:1+end:step]), axis=1) if n == 2 else \
                    np.stack((a[0:end:step], a[1:1+end:step], a[2:2+end:step]), axis=1)
            counts = {bytes(row): int(c) for row, c in zip(*np.unique(grams, axis=0, return_counts=True))}
        if isinstance(byte_obj, bytes):
            return counts
        elif hasattr(byte_obj, "bytes"):
            if not hasattr(byte_obj, "_ngram_counts_cache"):
                byte_obj._ngram_counts_cache = {}
            if n not in byte_obj._ngram_counts_cache.keys():
                byte_obj._ngram_counts_cache[n] = counts
            return byte_obj._ngram_counts_cache[n]
    raise TypeError("Bad input type ; should be a byte sequence or object")


def ngrams_distribution(byte_obj: bytes | object, n: int = 1, step: int = 1, n_most_common: Optional[int] = None,
                        n_exclude_top: int = 0, exclude: Optional[list] = None) -> list[tuple[bytes, int]]:
    """ Compute the n-grams distribution of an input byte sequence or byte object given exclusions.
    
    :param byte_obj:      byte sequence ('bytes') or byte object with "bytes" and "size" attributes (i.e. pathlib2.Path)
    :param n:             n determining the size of n-grams, defaults to 1
    :param step:          step for sliding the n-grams
    :param n_most_common: number of n-grams to be kept in the result, keep all by default
    :param n_exclude_top: number of n-grams to be excluded from the top of the histogram, no exclusion by default
    :param exclude:       list of specific n-grams to be excluded, no exclusion by default
    :return:              list of n_most_common (n-gram, count) pairs
    """
    c = ngrams_counts(byte_obj, n, step)
    n = len(c) if n_most_common is None else n_most_common + n_exclude_top + len(exclude or [])
    r = sorted(c.items(), key=lambda p: p[1], reverse=True)[:n]
    if exclude is not None:
        r = [(ngram, count) for ngram, count in r if ngram not in exclude]
    return r[n_exclude_top:n_exclude_top+(n_most_common or len(c))]

