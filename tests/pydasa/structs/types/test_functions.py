# -*- coding: utf-8 -*-
"""
Test Module for functions.py
===========================================

Tests for default comparison functions in PyDASA.
"""

import unittest
import pytest
from dataclasses import dataclass, is_dataclass
from unittest.mock import patch
from pydasa.structs.types.functions import (
    dflt_cmp_function_lt,
    dflt_cmp_function_ht,
)
from tests.pydasa.data.test_data import get_functions_test_data


# Mock MapEntry class for testing hash table comparison
class MockMapEntry:
    """Mock entry for hash table testing."""

    def __init__(self, key, value):
        self.key = key
        self.value = value


# Custom test object for testing non-comparable types
class CustomTestObject:
    """Custom object for testing non-comparable types."""

    def __init__(self, value):
        self.value = value


# Dataclass for testing dataclass comparison
@dataclass(order=True)
class TestDataClass:
    """Test dataclass for comparison tests."""

    value: int
    name: str


# Non-orderable dataclass for testing ordering validation
@dataclass
class NonOrderableDataClass:
    """Test dataclass without ordering support."""

    value: int
    name: str


# Dataclass with broken comparison methods for testing try-except block
@dataclass
class BrokenComparisonDataClass:
    """Test dataclass with comparison methods that raise TypeError."""

    value: int
    name: str

    def __lt__(self, other):
        """Broken less-than that raises TypeError."""
        raise TypeError("Comparison not supported")

    def __gt__(self, other):
        """Broken greater-than that raises TypeError."""
        raise TypeError("Comparison not supported")


class TestDefaultCmpFunctionLT(unittest.TestCase):
    """Test cases for dflt_cmp_function_lt."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_functions_test_data()

    def test_compare_integers(self) -> None:
        """Test comparison of integer pairs."""
        for val1, val2, expected in self.test_data["INT_PAIRS"]:
            result = dflt_cmp_function_lt(val1, val2, "")
            assert result == expected

    def test_compare_floats(self) -> None:
        """Test comparison of float pairs."""
        for val1, val2, expected in self.test_data["FLOAT_PAIRS"]:
            result = dflt_cmp_function_lt(val1, val2, "")
            assert result == expected

    def test_compare_strings(self) -> None:
        """Test comparison of string pairs."""
        for val1, val2, expected in self.test_data["STRING_PAIRS"]:
            result = dflt_cmp_function_lt(val1, val2, "")
            assert result == expected

    def test_compare_dicts_with_key(self) -> None:
        """Test comparison of dictionaries with key."""
        for dict1, dict2, key, expected in self.test_data["DICT_PAIRS"]:
            result = dflt_cmp_function_lt(dict1, dict2, key)
            assert result == expected

    def test_compare_different_types_raises_error(self) -> None:
        """Test that comparing different types raises TypeError."""
        for val1, val2 in self.test_data["INVALID_TYPE_PAIRS"]:
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_lt(val1, val2, "")
            assert "Invalid comparison" in str(excinfo.value)

    def test_compare_dicts_with_invalid_key_raises_error(self) -> None:
        """Test that using invalid key raises KeyError."""
        dict1 = {"name": "Alice"}
        dict2 = {"name": "Bob"}
        for invalid_key in self.test_data["INVALID_KEYS"]:
            with pytest.raises(KeyError) as excinfo:
                dflt_cmp_function_lt(dict1, dict2, invalid_key)
            assert "Key not found" in str(excinfo.value)

    def test_compare_custom_objects_raises_error(self) -> None:
        """Test that comparing custom objects raises TypeError."""
        test_pairs = [
            (CustomTestObject(1), CustomTestObject(2)),
            (CustomTestObject("a"), CustomTestObject("b")),
        ]
        for obj1, obj2 in test_pairs:
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_lt(obj1, obj2, "")
            assert "are not comparable" in str(excinfo.value)

    def test_compare_dataclasses(self) -> None:
        """Test comparison of dataclass instances."""
        dc1 = TestDataClass(1, "a")
        dc2 = TestDataClass(2, "b")
        dc3 = TestDataClass(1, "a")

        result_less = dflt_cmp_function_lt(dc1, dc2, "")
        result_equal = dflt_cmp_function_lt(dc1, dc3, "")
        result_greater = dflt_cmp_function_lt(dc2, dc1, "")

        assert result_less == -1
        assert result_equal == 0
        assert result_greater == 1

    def test_compare_non_orderable_dataclasses_raises_error(self) -> None:
        """Test that comparing dataclasses without order=True raises TypeError."""
        # Test 1: Regular non-orderable dataclass (hits try/except path)
        dc1 = NonOrderableDataClass(1, "a")
        dc2 = NonOrderableDataClass(2, "b")

        with pytest.raises(TypeError) as exc_info:
            dflt_cmp_function_lt(dc1, dc2, "")

        error_msg = str(exc_info.value)
        con_1 = "does not support sorting" in error_msg
        con_2 = "Cannot compare values" in error_msg
        con_3 = "not supported between instances" in error_msg
        assert (con_1 or con_2 or con_3)

    def test_comparison_runtime_error_in_try_block(self) -> None:
        """Test dataclass comparison error paths with mocked hasattr."""
        # Test the hasattr check path by mocking hasattr to return False
        dc1 = TestDataClass(1, "a")
        dc2 = TestDataClass(2, "")

        # Mock hasattr to return False for __lt__ and __gt__
        def mock_hasattr(obj, name):
            if name in ('__lt__', '__gt__') and is_dataclass(obj):
                return False
            return hasattr(obj, name)  # Use builtin for other cases

        with patch('pydasa.structs.types.functions.hasattr', side_effect=mock_hasattr):
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_lt(dc1, dc2, "")
            assert "does not support sorting" in str(excinfo.value)


class TestDefaultCmpFunctionHT(unittest.TestCase):
    """Test cases for dflt_cmp_function_ht."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_functions_test_data()

    def test_compare_integer_keys(self) -> None:
        """Test comparison of entries with integer keys."""
        for key1, key2, expected in self.test_data["INT_PAIRS"]:
            entry2 = MockMapEntry(key2, "value")
            result = dflt_cmp_function_ht(key1, entry2, "")
            assert result == expected

    def test_compare_string_keys(self) -> None:
        """Test comparison of entries with string keys."""
        for key1, key2, expected in self.test_data["STRING_PAIRS"]:
            entry2 = MockMapEntry(key2, "value")
            result = dflt_cmp_function_ht(key1, entry2, "")
            assert result == expected

    def test_compare_dict_keys_with_key(self) -> None:
        """Test comparison of entries with dictionary keys."""
        for dict1, dict2, key, expected in self.test_data["DICT_PAIRS"]:
            entry2 = MockMapEntry(dict2, "value")
            result = dflt_cmp_function_ht(dict1, entry2, key)
            assert result == expected

    def test_compare_different_types_raises_error(self) -> None:
        """Test that comparing different key types raises TypeError."""
        for key1, key2 in self.test_data["INVALID_TYPE_PAIRS"]:
            entry2 = MockMapEntry(key2, "value")
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_ht(key1, entry2, "")
            assert "Invalid comparison" in str(excinfo.value)

    def test_compare_dict_keys_with_invalid_key_raises_error(self) -> None:
        """Test that using invalid key with dict keys raises KeyError."""
        dict_key1 = {"name": "Alice"}
        dict_key2 = {"name": "Bob"}
        entry2 = MockMapEntry(dict_key2, "value")
        for invalid_key in self.test_data["INVALID_KEYS"]:
            with pytest.raises(KeyError) as excinfo:
                dflt_cmp_function_ht(dict_key1, entry2, invalid_key)
            assert "Key not found" in str(excinfo.value)

    def test_compare_custom_object_keys_raises_error(self) -> None:
        """Test that comparing custom object keys raises TypeError."""
        test_pairs = [
            (CustomTestObject(1), CustomTestObject(2)),
            (CustomTestObject("a"), CustomTestObject("b")),
        ]
        for obj1, obj2 in test_pairs:
            entry2 = MockMapEntry(obj2, "value")
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_ht(obj1, entry2, "")
            assert "are not comparable" in str(excinfo.value)

    def test_compare_dataclass_keys(self) -> None:
        """Test comparison of dataclass keys in hash table."""
        dc1 = TestDataClass(1, "a")
        dc2 = TestDataClass(2, "b")
        dc3 = TestDataClass(1, "a")

        entry2_greater = MockMapEntry(dc2, "value1")
        entry2_equal = MockMapEntry(dc3, "value2")
        entry2_less = MockMapEntry(dc1, "value3")

        result_less = dflt_cmp_function_ht(dc1, entry2_greater, "")
        result_equal = dflt_cmp_function_ht(dc1, entry2_equal, "")
        result_greater = dflt_cmp_function_ht(dc2, entry2_less, "")

        assert result_less == -1
        assert result_equal == 0
        assert result_greater == 1

    def test_compare_non_orderable_dataclass_keys_raises_error(self) -> None:
        """Test that comparing dataclass keys without order=True raises TypeError."""
        dc1 = NonOrderableDataClass(1, "a")
        dc2 = NonOrderableDataClass(2, "b")
        entry2 = MockMapEntry(dc2, "value")

        with pytest.raises(TypeError) as exc_info:
            dflt_cmp_function_ht(dc1, entry2, "")

        # Error comes from the try/except block when comparison fails
        error_msg = str(exc_info.value)
        con_1 = "does not support sorting" in error_msg
        con_2 = "Cannot compare values" in error_msg
        con_3 = "not supported between instances" in error_msg
        assert (con_1 or con_2 or con_3)

    def test_comparison_runtime_error_in_try_block(self) -> None:
        """Test dataclass comparison error paths with mocked hasattr."""
        # Test the hasattr check path by mocking hasattr to return False
        dc1 = TestDataClass(1, "a")
        entry2 = MockMapEntry(TestDataClass(2, "b"), "value")

        # Mock hasattr to return False for __lt__ and __gt__
        def mock_hasattr(obj, name):
            if name in ('__lt__', '__gt__') and is_dataclass(obj):
                return False
            return hasattr(obj, name)  # Use builtin for other cases

        with patch('pydasa.structs.types.functions.hasattr', side_effect=mock_hasattr):
            with pytest.raises(TypeError) as excinfo:
                dflt_cmp_function_ht(dc1, entry2, "")
            assert "does not support sorting" in str(excinfo.value)
