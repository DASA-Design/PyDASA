# -*- coding: utf-8 -*-
# FIXME old code, remove when tests are finished

"""
Module with utility functions for handling memory allocation in the Data Structures of *PyDASA*.
"""
# python native modules
# dataclases module handles the creation of classes with slots and fields
import dataclasses
# import global variables


def slot_dataclass(cls) -> dataclasses.dataclass:
    """slot_dataclass is a decorator that converts a class into a dataclass with slots.

    a slot is a special kind of attribute that is used to store data in a class.
    It is used to optimize memory usage and improve performance by reducing the overhead of the class.

    Returns:
        dataclasses.dataclass: A dataclass with slots.
    """
    # TODO check vialidity of this decorator
    # TODO integrate with the dataclass decorator
    # check if the class is a valid class
    if not isinstance(cls, type):
        raise TypeError(f"Invalid class: {cls}, class must be a type")
    # check if the class is a dataclass
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"Invalid class: {cls}, class must be a dataclass")
    # allocate slots for the class
    cls.__slots__ = [f.name for f in dataclasses.fields(cls)]
    return dataclasses.dataclass(cls)
