# -*- coding: utf-8 -*-
"""
Module generics.py
===========================================

Module for default generic dataclass type for use by all *PyDASA* and its Data Structs.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""

# python native modules
from typing import TypeVar

# Type for the element stored in the dataclass
# :data: T: TypeVar
T = TypeVar("T")
"""
Type for creating Generics dataclasses in data structure classes, methods, and attrs.

NOTE: used for type hinting only in generics dataclasses.
"""
