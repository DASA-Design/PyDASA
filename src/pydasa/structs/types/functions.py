# -*- coding: utf-8 -*-
"""
Module default.py
===========================================

Module for default global variables and comparison functions for use by all *PyDASA* and its Data Structures.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""


# python native modules
from dataclasses import is_dataclass
from typing import Any

# custom modules
from pydasa.structs.types.generics import VLD_DTYPE_LT


def dflt_cmp_function_lt(elm1: Any, elm2: Any, key: str) -> int:
    """*dflt_cmp_function_lt()* Compare two elements of the ADT List (ArrayList, SingleLinked, DoubleLinked). They can be of Python native or user-defined.

    Args:
        elm1 (Any): First element to compare.
        elm2 (Any): Second element to compare.
        key (str): Key for comparing dictionary elements.

    Raises:
        TypeError: If elements are of different types or not comparable.
        KeyError: If the key is not found in dictionary elements.
        TypeError: If elements are not of built-in comparable types.

    Returns:
        int: -1 if elm1 < elm2, 0 if elm1 == elm2, 1 if elm1 > elm2.
    """

    val1, val2 = None, None
    # if elements are of different types, raise error
    if type(elm1) is not type(elm2):
        _msg = "Invalid comparison between "
        _msg += f"{type(elm1)} and {type(elm2)} elements."
        raise TypeError(_msg)

    # if both elements are dictionaries and a key is provided
    if key and isinstance(elm1, dict) and isinstance(elm2, dict):
        val1, val2 = elm1.get(key), elm2.get(key)
        if val1 is None or val2 is None:
            _msg = f"Invalid key: {key}, Key not found in one or both elements."
            raise KeyError(_msg)

    # if both elements are built-in comparable types
    elif isinstance(elm1, VLD_DTYPE_LT) and isinstance(elm2, VLD_DTYPE_LT):
        val1, val2 = elm1, elm2

    # if both elements are dataclasses
    elif is_dataclass(elm1) and is_dataclass(elm2):
        # Check if dataclasses support ordering
        # Dataclasses with order=True have __lt__ and __gt__ methods
        if not (hasattr(elm1, '__lt__') and hasattr(elm1, '__gt__')):
            _msg = f"Dataclass {type(elm1).__name__} does not support sorting. "
            _msg += "Use @dataclass(order=True) to enable comparisons."
            raise TypeError(_msg)
        val1, val2 = elm1, elm2

    # otherwise, raise error
    else:
        _msg = f"Elements of type {type(elm1)} are not comparable "
        _msg += f"with elements of type {type(elm2)}."
        raise TypeError(_msg)

    # Simplified comparison: returns -1, 0, or 1
    # Type checker can't verify all branches lead to comparable types
    try:
        return (val1 > val2) - (val1 < val2)  # type: ignore[operator]
    except TypeError as e:
        _msg = f"Cannot compare values of type {type(val1).__name__} "
        _msg += f"and {type(val2).__name__}: {e}"
        raise TypeError(_msg) from e


def dflt_cmp_function_ht(ekey1: Any, entry2, key: str, ) -> int:
    """*dflt_cmp_function_ht()* Compare the entries of the ADT Map (Hash Table). can be of Python native or user-defined.

    Args:
        ekey1 (Any): Key of the first entry (key-value pair) to compare.
        entry2 (MapEntry): Second entry (key-value pair) to compare.
        key (str): Key for comparing dictionary elements.

    Raises:
        TypeError: If the keys are of different types or not comparable.
        KeyError: If the key is not found in dictionary elements.
        TypeError: If keys are not of built-in comparable types.

    Returns:
        int: -1 if ekey1 < ekey2, 0 if ekey1 == ekey2, 1 if ekey1 > ekey2.
    """
    # Extract keys from entries
    ekey2 = entry2.key

    # if keys are of different types, raise error
    if type(ekey1) is not type(ekey2):
        _msg = "Invalid comparison between "
        _msg += f"{type(ekey1)} and {type(ekey2)} elements."
        raise TypeError(_msg)

    # if both keys are dictionaries and a key is provided
    if key and isinstance(ekey1, dict) and isinstance(ekey2, dict):
        val1, val2 = ekey1.get(key), ekey2.get(key)
        if val1 is None or val2 is None:
            _msg = f"Invalid key: '{key}', Key not found in one or both dictionary elements"
            raise KeyError(_msg)

    # if both keys are built-in comparable types
    elif isinstance(ekey1, VLD_DTYPE_LT) and isinstance(ekey2, VLD_DTYPE_LT):
        val1, val2 = ekey1, ekey2

    # if both keys are dataclasses
    elif is_dataclass(ekey1) and is_dataclass(ekey2):
        # Check if dataclasses support ordering
        # Dataclasses with order=True have __lt__ and __gt__ methods
        if not (hasattr(ekey1, '__lt__') and hasattr(ekey1, '__gt__')):
            _msg = f"Dataclass {type(ekey1).__name__} does not support sorting. "
            _msg += "Use @dataclass(order=True) to enable comparisons."
            raise TypeError(_msg)
        val1, val2 = ekey1, ekey2

    # otherwise, raise error
    else:
        _msg = f"Elements of type {type(ekey1)} are not comparable "
        _msg += f"with elements of type {type(ekey2)}."
        raise TypeError(_msg)

    # Simplified comparison: returns -1, 0, or 1
    # Type checker can't verify all branches lead to comparable types
    try:
        return (val1 > val2) - (val1 < val2)  # type: ignore[operator]
    except TypeError as e:
        _msg = f"Cannot compare values of type {type(val1).__name__} "
        _msg += f"and {type(val2).__name__}: {e}"
        raise TypeError(_msg) from e
