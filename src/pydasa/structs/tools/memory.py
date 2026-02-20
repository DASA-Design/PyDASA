# -*- coding: utf-8 -*-
"""
Module memory.py
===========================================

Module with utility functions for handling memory allocation in the Data Structures of *PyDASA*.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""
# python native modules
import sys
from typing import Type, Callable, cast, Any
import dataclasses

# import global variables
from pydasa.structs.types.generics import T


def alloc_slots() -> Callable[[Type[T]], Type[T]]:
    """Decorator that converts a class into a dataclass with slots for memory optimization.

    This decorator uses Python 3.10+ native slots support in dataclasses to reduce memory usage
    by preventing the creation of `__dict__` for each instance.

    Raises:
        TypeError: If the decorated object is not a class type.
        RuntimeError: If the Python version is less than 3.10.

    Returns:
        Callable: Decorated class as a dataclass with slots.

    Example::

        # Simple usage - all fields slotted, no dynamic attributes
        @alloc_slots()
        class Point:
            x: int
            y: int

        # Works with existing dataclasses
        @alloc_slots()
        @dataclass
        class Vector:
            x: float
            y: float
            z: float
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate input is a class type
        if not isinstance(cls, type):
            raise TypeError(f"Invalid class: {cls}, class must be a type")

        # Check Python version for native slots support
        if sys.version_info < (3, 10):
            _msg = "alloc_slots requires Python 3.10+ for native support. "
            _msg += f"Current version: {sys.version_info.major}."
            _msg += f"{sys.version_info.minor}"
            raise RuntimeError(_msg)

        # Convert to dataclass with slots (standard implementation)
        result_cls: Any = dataclasses.dataclass(cls, slots=True)
        return cast(Type[T], result_cls)

    return decorator
