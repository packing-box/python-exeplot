#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import os
from exeplot.plots.__common__ import _ensure_str, Binary
from unittest import TestCase


class TestOthers(TestCase):
    def test_miscellaneous(self):
        self.assertRaises(TypeError, _ensure_str, 1)
        for i in range(256):
            self.assertIsNotNone(_ensure_str(bytes([i])))
        self.assertRaises(TypeError, Binary, "BAD")
        binary = Binary(os.path.join(os.path.dirname(__file__), "hello.exe"))
        self.assertIsNotNone(str(binary))

