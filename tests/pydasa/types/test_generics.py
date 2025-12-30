# -*- coding: utf-8 -*-
"""
Test Module for generics.py
===========================================

Tests for generic type variables in PyDASA.
"""

import unittest
from dataclasses import dataclass
from typing import Generic, TypeVar
from pydasa.types.generics import T


class TestGenerics(unittest.TestCase):
    """Test cases for TypeVar T and generic dataclass usage."""

    def test_typevar_exists(self) -> None:
        """Test that TypeVar T exists and is accessible."""
        assert T is not None

    def test_typevar_is_typevar(self) -> None:
        """Test that T is a TypeVar instance."""
        assert isinstance(T, TypeVar)

    def test_typevar_name(self) -> None:
        """Test that TypeVar has correct name 'T'."""
        assert T.__name__ == "T"

    def test_typevar_no_constraints(self) -> None:
        """Test that TypeVar T has no type constraints."""
        assert T.__constraints__ == ()

    def test_typevar_not_bound(self) -> None:
        """Test that TypeVar T has no bound type."""
        assert T.__bound__ is None

    def test_generic_dataclass_with_int(self) -> None:
        """Test generic dataclass with integer type."""
        @dataclass
        class Container(Generic[T]):
            value: T

        container = Container(value=42)
        assert container.value == 42

    def test_generic_dataclass_with_string(self) -> None:
        """Test generic dataclass with string type."""
        @dataclass
        class Box(Generic[T]):
            item: T

        box = Box(item="test")
        assert box.item == "test"

    def test_generic_with_multiple_instances(self) -> None:
        """Test that T works with different type instances."""
        @dataclass
        class Wrapper(Generic[T]):
            data: T

        int_wrapper = Wrapper(data=100)
        str_wrapper = Wrapper(data="hello")

        assert int_wrapper.data == 100
        assert str_wrapper.data == "hello"
