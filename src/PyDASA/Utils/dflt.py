# -*- coding: utf-8 -*-
"""
Module for default global variables and comparison functions for use by all *PyDASA* and its Data Structs.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""


# python native modules
from dataclasses import dataclass
from typing import TypeVar

# custom modules
# import global variables

# valid data types for the node
# :data: VLD_DTYPE_LT
VLD_DTYPE_LT: tuple = (
    int,
    float,
    str,
    bool,
    dict,
    list,
    tuple,
    set,
    dataclass,
)
"""
Tuple with the native data types in Python that are comparable in the structures.
"""

# default key for comparing dictionaries
# :data: DFLT_DICT_KEY
DFLT_DICT_KEY: str = "_id"
"""
Default key field for comparing dictionaries in the structures.
"""

# allowed input/output types for the ADTs
# :data: VLD_IODATA_LT
VLD_IODATA_LT: tuple = (
    list,
    tuple,
    set,
)
"""
Tuple with the allowed input/output types for the ADTs. Useful for loading and saving data in the ADTs with the *load* and *save* file methods.
"""

# default big prime number for MAD compression in hash tables
# :data: DFLT_PRIME
DFLT_PRIME: int = 109345121
"""
Default big prime number for MAD compression in hash tables. This number is used to calculate the hash value of the keys in the hash table.
"""


# Type for the element stored in the list
# :data: T: TypeVar
T = TypeVar("T")
"""
Type for the element stored in the list. This is used to define the type of the elements in the list and its methods.

This is used to define the type of the elements in the list and its methods.
"""


def dflt_cmp_func_lt(key: str, elm1, elm2) -> int:
    """*lt_default_cmp_funcion()* default comparison function for the elements of the ADT List (ArrayList, SingleLinked, DoubleLinked). can be of native type or user-defined.

    Args:
        key (str): Key for comparing dictionary elements.
        elm1 (any): First element to compare.
        elm2 (any): Second element to compare.

    Raises:
        TypeError: If elements are of different types or not comparable.
        KeyError: If the key is not found in dictionary elements.

    Returns:
        int: -1 if elm1 < elm2, 0 if elm1 == elm2, 1 if elm1 > elm2.
    """
    # Ensure elements are of the same type
    if type(elm1) is not type(elm2):
        _msg = f"Invalid comparison between {type(elm1)} and "
        _msg += f"{type(elm2)} elements"
        raise TypeError(_msg)

    # Handle dictionary comparison using the provided key
    if key and isinstance(elm1, dict) and isinstance(elm2, dict):
        key1, key2 = elm1.get(DFLT_DICT_KEY), elm2.get(DFLT_DICT_KEY)
        if key1 is None or key2 is None:
            _msg = f"Invalid key: {DFLT_DICT_KEY}, "
            _msg += "Key not found in one or both elements"
            raise KeyError(_msg)
        if key1 < key2:
            return -1
        elif key1 == key2:
            return 0
        elif key1 > key2:
            return 1
        # TODO check this simplified logic
        # return (key1 > key2) - (key1 < key2)  # Simplified comparison logic

    # Handle native type comparison
    if isinstance(elm1, VLD_DTYPE_LT) and isinstance(elm2, VLD_DTYPE_LT):
        if elm1 < elm2:
            return -1
        elif elm1 == elm2:
            return 0
        elif elm1 > elm2:
            return 1
        # TODO check this simplified logic
        # return (elm1 > elm2) - (elm1 < elm2)  # Simplified comparison logic

    # Raise error if elements are not comparable
    _msg = f"Elements of type {type(elm1)} and {type(elm2)} are not comparable"
    raise TypeError(_msg)


def dflt_cmp_func_ht(key: str, ekey1: T, entry2) -> int:
    """*dflt_cmp_func_ht()* default comparison function for the elements of the ADT Map (Hash Table). can be of native type or user-defined.

    Args:
        key (str): Key for comparing dictionary elements.
        ekey1 (T): Key of the first entry (key-value pair) to compare.
        entry2 (MapEntry): Second entry (key-value pair) to compare.

    Raises:
        TypeError: If the keys are of different types or not comparable.
        KeyError: If the key is not found in dictionary elements.

    Returns:
        int: -1 if ekey1 < ekey2, 0 if ekey1 == ekey2, 1 if ekey1 > ekey2.
    """
    ekey2 = entry2.key

    # Ensure keys are of the same type
    if type(ekey1) is not type(ekey2):
        _msg = f"Invalid comparison between {type(ekey1)} and "
        _msg += f"{type(ekey2)} elements"
        raise TypeError(_msg)

    # Handle dictionary comparison using the provided key
    if key and isinstance(ekey1, dict) and isinstance(ekey2, dict):
        key1, key2 = ekey1.get(DFLT_DICT_KEY), ekey2.get(DFLT_DICT_KEY)
        if None in [key1, key2]:
            _msg = f"Invalid key: {DFLT_DICT_KEY}, "
            _msg += "Key not found in one or both elements"
            raise KeyError(_msg)
        return (key1 > key2) - (key1 < key2)  # Simplified comparison logic

    # Handle tuple comparison
    # TODO tuple comparision could be redundant with VLD_DTYPE_LT present
    if isinstance(ekey1, tuple) and isinstance(ekey2, tuple):
        if ekey1 == ekey2:
            return 0
        elif ekey1 < ekey2:
            return -1
        elif ekey1 > ekey2:
            return 1
        # TODO check this simplified logic
        # return (list(ekey1) > list(ekey2)) - (list(ekey1) < list(ekey2))

    # Handle native type comparison
    if isinstance(ekey1, VLD_DTYPE_LT) and isinstance(ekey2, VLD_DTYPE_LT):
        if ekey1 < ekey2:
            return -1
        elif ekey1 == ekey2:
            return 0
        elif ekey1 > ekey2:
            return 1
        # TODO check this simplified logic
        return (ekey1 > ekey2) - (ekey1 < ekey2)  # Simplified comparison logic

    # Raise error if keys are not comparable
    _msg = f"Elements of type {type(ekey1)}"
    _msg += f" and {type(ekey2)} are not comparable"
    raise TypeError(_msg)
