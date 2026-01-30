#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import numpy as np


def parse_transform(transform_str):
    """SVGのtransform属性を3x3行列に変換する"""
    m = re.search(
        r"matrix\(\s*([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+"
        r"([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)\s*\)",
        transform_str,
    )
    if m:
        a, b, c, d, e, f = map(float, m.groups())
        return np.array([[a, c, e], [b, d, f], [0, 0, 1]])
    return np.eye(3)


def transform_point(x, y, T):
    """点(x, y)に変換行列Tを適用する"""
    pt = np.array([x, y, 1])
    tpt = T @ pt
    return tpt[0], tpt[1]
