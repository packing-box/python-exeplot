# -*- coding: UTF-8 -*-
from math import log2


__all__ = ["ensure_str", "human_readable_size", "ngrams_counts", "ngrams_distribution", "shannon_entropy"]

shannon_entropy = lambda b: -sum([p*log2(p) for p in [float(ctr)/len(b) for ctr in [b.count(c) for c in set(b)]]]) or 0.


def ensure_str(s, encoding='utf-8', errors='strict'):
    """ Ensure that an input string is decoded. """
    if isinstance(s, bytes):
        try:
            return s.decode(encoding, errors)
        except:
            return s.decode("latin-1")
    elif not isinstance(s, (str, bytes)):
        raise TypeError("not expecting type '%s'" % type(s))
    return s


def human_readable_size(size, precision=0):
    """ Display bytes' size in a human-readable format given a precision. """
    i, units = 0, ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    while size >= 1024 and i < len(units)-1:
        i += 1
        size /= 1024.0
    return "%.*f%s" % (precision, size, units[i])


def ngrams_counts(byte_obj, n=1, step=1):
    """ Output the Counter instance for an input byte sequence or byte object based on n-grams.
         If the input is a byte object, cache the result.
    
    :param byte_obj:      byte sequence ('bytes') or byte object with "bytes" and "size" attributes (i.e. pathlib2.Path)
    :param n: n determining the size of n-grams, defaults to 1
    :param step:          step for sliding the n-grams
    """
    from collections import Counter
    if isinstance(byte_obj, (str, bytes)):
        return Counter(byte_obj[i:i+n] for i in range(0, len(byte_obj)-n+1, step))
    elif hasattr(byte_obj, "bytes") and hasattr(byte_obj, "size"):
        if not hasattr(byte_obj, "_ngram_counts_cache"):
            byte_obj._ngram_counts_cache = {}
        if n not in byte_obj._ngram_counts_cache.keys():
            byte_obj._ngram_counts_cache[n] = Counter(byte_obj.bytes[i:i+n] for i in range(0, byte_obj.size-n+1, step))
        return byte_obj._ngram_counts_cache[n]
    raise TypeError("Bad input type ; should be a byte sequence or object")


def ngrams_distribution(byte_obj, n=1, step=1, n_most_common=None, n_exclude_top=0, exclude=None):
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
    r = c.most_common(len(c) if n_most_common is None else n_most_common + n_exclude_top + len(exclude or []))
    if exclude is not None:
        r = [(ngram, count) for ngram, count in r if ngram not in exclude]
    return r[n_exclude_top:n_exclude_top+(n_most_common or len(c))]

