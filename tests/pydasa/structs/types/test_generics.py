# -*- coding: utf-8 -*-
"""
Test Module for generics.py
===========================================

Tests for generic type variables and constants in PyDASA.
"""

import unittest
from dataclasses import dataclass
from typing import Generic, TypeVar
from pydasa.structs.types.generics import (
    T,
    DFLT_DICT_KEY,
    VLD_IOTYPE_LT,
    DFLT_PRIME,
    VLD_DTYPE_LT,
)


class TestTypeVar(unittest.TestCase):
    """Test cases for TypeVar T and generic dataclass usage."""

    def test_typevar_exists(self) -> None:
        """Test that TypeVar T exists and is accessible."""
        assert T is not None

    def test_typevar_is_typevar(self) -> None:
        """Test that T is a TypeVar instance."""
        # checking that T is a TypeVar instance
        assert isinstance(T, TypeVar)
        # checking TypeVar has correct name 'T'
        assert T.__name__ == "T"
        # checking that T has no constraints and no bound type
        assert T.__constraints__ == ()
        # checking that T has no bound type
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


class TestConstants(unittest.TestCase):
    """Test cases for module constants."""

    def test_vld_dtype_lt_contains_expected_types(self) -> None:
        """Test that VLD_DTYPE_LT contains all expected types."""
        assert int in VLD_DTYPE_LT
        assert float in VLD_DTYPE_LT
        assert str in VLD_DTYPE_LT
        assert bool in VLD_DTYPE_LT
        assert dict in VLD_DTYPE_LT
        assert list in VLD_DTYPE_LT
        assert tuple in VLD_DTYPE_LT
        assert set in VLD_DTYPE_LT

    def test_vld_dtype_lt_is_tuple(self) -> None:
        """Test that VLD_DTYPE_LT is a tuple."""
        assert isinstance(VLD_DTYPE_LT, tuple)

    def test_dflt_dict_key_exists(self) -> None:
        """Test that DFLT_DICT_KEY exists and has correct value."""
        assert DFLT_DICT_KEY is not None
        assert isinstance(DFLT_DICT_KEY, str)
        assert DFLT_DICT_KEY == "_idx"

    def test_vld_iotype_lt_exists(self) -> None:
        """Test that VLD_IOTYPE_LT exists and is accessible."""
        assert VLD_IOTYPE_LT is not None
        # checking that VLD_IOTYPE_LT is a tuple and contains expected types
        assert isinstance(VLD_IOTYPE_LT, tuple)
        assert list in VLD_IOTYPE_LT
        assert tuple in VLD_IOTYPE_LT
        assert set in VLD_IOTYPE_LT
        assert len(VLD_IOTYPE_LT) == 3

    def test_dflt_prime_exists(self) -> None:
        """Test that DFLT_PRIME exists and has correct value."""
        assert DFLT_PRIME is not None
        # checking that DFLT_PRIME is an integer and has the expected value
        assert DFLT_PRIME > 0
        assert DFLT_PRIME == 109345121
        assert isinstance(DFLT_PRIME, int)
