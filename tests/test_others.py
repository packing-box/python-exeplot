#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import os
from collections import Counter
from exeplot.plots.__common__ import Binary
from exeplot.utils import *
from unittest import TestCase


class TestOthers(TestCase):
    def test_miscellaneous(self):
        self.assertRaises(TypeError, ensure_str, 1)
        for i in range(256):
            self.assertIsNotNone(ensure_str(bytes([i])))
        self.assertRaises(TypeError, Binary, "BAD")
        binary = Binary(os.path.join(os.path.dirname(__file__), "hello.exe"))
        self.assertIsNotNone(str(binary))


class TestUtils(TestCase):
    def test_ngrams_functions(self):
        self.assertRaises(TypeError, ngrams_counts, 123)
        self.assertTrue(isinstance(ngrams_counts(seq := b"\x00" * 4 + os.urandom(120) + b"\xff" * 4), Counter))
        class Test:
            bytes = seq
            size = len(seq)
        histogram = ngrams_distribution(t := Test(), exclude=(b"\x00", b"\xff"))
        self.assertTrue(isinstance(histogram, list))
        self.assertNotIn(b"\x00", [b for b, c in histogram])
        self.assertNotIn(b"\xff", [b for b, c in histogram])
        histogram2 = ngrams_distribution(t, n_most_common=300)
        self.assertIn(b"\x00", [b for b, c in histogram2])
        self.assertIn(b"\xff", [b for b, c in histogram2])

