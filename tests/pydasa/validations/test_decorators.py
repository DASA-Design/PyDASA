# -*- coding: utf-8 -*-
"""
Test module for validations.decorators
==========================================

Tests for decorator-based validation system.
"""

import unittest
from enum import Enum
import numpy as np
# import re
from pydasa.validations.decorators import (
    validate_type,
    validate_emptiness,
    validate_choices,
    validate_range,
    validate_index,
    validate_pattern,
    validate_list_types,
    validate_dict_types,
    validate_custom,
)


class TestValidationDecorators(unittest.TestCase):
    """Test validation decorators"""

    def test_validate_type_valid(self):
        """Test validate_type with valid types"""
        class TestClass:
            def __init__(self):
                self._value = None

            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str, int)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = "test"
        self.assertEqual(obj.value, "test")

        obj.value = 42
        self.assertEqual(obj.value, 42)

        obj.value = None  # None allowed by default
        self.assertIsNone(obj.value)

        # Test np.nan handling
        class TestNanClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(float, allow_nan=True)
            def value(self, val):
                self._value = val

        obj_nan = TestNanClass()
        obj_nan.value = np.nan  # Should work with allow_nan=True
        self.assertTrue(np.isnan(obj_nan.value))

        # Test np.nan rejection
        class TestNanRejectClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(float, allow_nan=False)
            def value(self, val):
                self._value = val

        obj_no_nan = TestNanRejectClass()
        with self.assertRaises(ValueError) as ctx:
            obj_no_nan.value = np.nan
        self.assertIn("cannot be np.nan", str(ctx.exception))

    def test_validate_type_invalid(self):
        """Test validate_type with invalid type"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = 42
        self.assertIn("must be str", str(ctx.exception))

    def test_validate_type_no_none(self):
        """Test validate_type with allow_none=False"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str, allow_none=False)
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = None
        self.assertIn("cannot be None", str(ctx.exception))

    def test_validate_emptiness_valid(self):
        """Test validate_emptiness with valid string"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_emptiness()
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = "test"
        self.assertEqual(obj.value, "test")

        # Test strip=False behavior
        class TestNoStripClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_emptiness(strip=False)
            def value(self, val):
                self._value = val

        obj_no_strip = TestNoStripClass()
        obj_no_strip.value = "  text  "  # Should pass with strip=False
        self.assertEqual(obj_no_strip.value, "  text  ")

        with self.assertRaises(ValueError) as ctx:
            obj_no_strip.value = ""
        self.assertIn("non-empty", str(ctx.exception))

    def test_validate_emptiness_invalid(self):
        """Test validate_emptiness with empty string"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_emptiness()
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = "   "
        self.assertIn("non-empty", str(ctx.exception))

        # Test empty collections
        class TestListClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(list)
            @validate_emptiness()
            def value(self, val):
                self._value = val

        obj_list = TestListClass()
        with self.assertRaises(ValueError) as ctx:
            obj_list.value = []
        self.assertIn("non-empty", str(ctx.exception))

        class TestDictClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(dict)
            @validate_emptiness()
            def value(self, val):
                self._value = val

        obj_dict = TestDictClass()
        with self.assertRaises(ValueError) as ctx:
            obj_dict.value = {}
        self.assertIn("non-empty", str(ctx.exception))

        class TestTupleClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(tuple)
            @validate_emptiness()
            def value(self, val):
                self._value = val

        obj_tuple = TestTupleClass()
        with self.assertRaises(ValueError) as ctx:
            obj_tuple.value = ()
        self.assertIn("non-empty", str(ctx.exception))

    def test_validate_choices_dict(self):
        """Test validate_choices with dictionary"""
        CHOICES = {"A": 1, "B": 2, "C": 3}

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_choices(CHOICES)
            def value(self, val):
                self._value = val.upper()

        obj = TestClass()
        obj.value = "a"  # Case insensitive by default
        self.assertEqual(obj.value, "A")

        with self.assertRaises(ValueError) as ctx:
            obj.value = "D"
        self.assertIn("Must be one of", str(ctx.exception))

        # Test with Enum class
        class Status(Enum):
            ACTIVE = 1
            INACTIVE = 2
            PENDING = 3

        class TestEnumClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_choices(Status)
            def value(self, val):
                self._value = val

        obj_enum = TestEnumClass()
        obj_enum.value = Status.ACTIVE
        self.assertEqual(obj_enum.value, Status.ACTIVE)

        with self.assertRaises(ValueError):
            obj_enum.value = "INVALID"

        # Test allow_none=True
        class TestNoneClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_choices({"X", "Y", "Z"}, allow_none=True)
            def value(self, val):
                self._value = val

        obj_none = TestNoneClass()
        obj_none.value = None  # Should work
        self.assertIsNone(obj_none.value)

    def test_validate_range_static(self):
        """Test validate_range with static min/max"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(int, float)
            @validate_range(min_value=0, max_value=100)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = 50
        self.assertEqual(obj.value, 50)

        with self.assertRaises(ValueError):
            obj.value = -1

        with self.assertRaises(ValueError):
            obj.value = 101

        # Test exclusive bounds
        class TestExclusiveClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(int, float)
            @validate_range(min_value=10, max_value=100, min_inclusive=False, max_inclusive=False)
            def value(self, val):
                self._value = val

        obj_excl = TestExclusiveClass()
        obj_excl.value = 50  # Should work
        self.assertEqual(obj_excl.value, 50)

        with self.assertRaises(ValueError) as ctx:
            obj_excl.value = 10  # Exclusive minimum
        self.assertIn("must be >", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            obj_excl.value = 100  # Exclusive maximum
        self.assertIn("must be <", str(ctx.exception))

    def test_validate_range_dynamic(self):
        """Test validate_range with dynamic attributes"""
        class TestClass:
            def __init__(self):
                self._min = 0
                self._max = 100
                self._value = None

            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(int, float)
            @validate_range(min_attr="_min", max_attr="_max")
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = 50
        self.assertEqual(obj.value, 50)

        with self.assertRaises(ValueError) as ctx:
            obj.value = -1
        self.assertIn("cannot be less than minimum", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            obj.value = 101
        self.assertIn("cannot be greater than maximum", str(ctx.exception))

    def test_validate_index(self):
        """Test validate_index decorator"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(int)
            @validate_index(allow_negative=False)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = 0  # Should work
        self.assertEqual(obj.value, 0)

        obj.value = 5
        self.assertEqual(obj.value, 5)

        with self.assertRaises(ValueError) as ctx:
            obj.value = -1
        self.assertIn("non-negative", str(ctx.exception))

        # Test allow_zero=False
        class TestNoZeroClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_index(allow_zero=False, allow_negative=False)
            def value(self, val):
                self._value = val

        obj_no_zero = TestNoZeroClass()
        with self.assertRaises(ValueError) as ctx:
            obj_no_zero.value = 0
        self.assertIn("cannot be zero", str(ctx.exception))

        # Test non-integer type error
        class TestIntTypeClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_index(allow_negative=False)
            def value(self, val):
                self._value = val

        obj_int_type = TestIntTypeClass()
        with self.assertRaises(ValueError) as ctx:
            obj_int_type.value = "5"
        self.assertIn("must be an integer", str(ctx.exception))

    def test_validate_pattern(self):
        """Test validate_pattern decorator"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(pattern=r'^[A-Z]\d{3}$', error_msg="Must be letter + 3 digits")
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = "A123"
        self.assertEqual(obj.value, "A123")

        with self.assertRaises(ValueError) as ctx:
            obj.value = "ABC"
        self.assertIn("Must be letter + 3 digits", str(ctx.exception))

    def test_multiple_decorators(self):
        """Test stacking multiple decorators"""
        class TestClass:
            def __init__(self):
                self._value = None

            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_emptiness()
            @validate_choices({"ALPHA", "BETA", "GAMMA"})
            def value(self, val):
                self._value = val.upper()

        obj = TestClass()
        obj.value = "alpha"
        self.assertEqual(obj.value, "ALPHA")

        # Test type validation
        with self.assertRaises(ValueError):
            obj.value = 123

        # Test empty validation
        with self.assertRaises(ValueError):
            obj.value = "  "

        # Test choices validation
        with self.assertRaises(ValueError):
            obj.value = "DELTA"

    def test_validate_pattern_alphanumeric(self):
        """Test validate_pattern with alphanumeric values"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(allow_alnum=True)
            def value(self, val):
                self._value = val

        obj = TestClass()
        # Valid alphanumeric symbols
        for symbol in ["V", "d", "x1", "rho1", "ABC123"]:
            obj.value = symbol
            self.assertEqual(obj.value, symbol)

    def test_validate_pattern_latex(self):
        """Test validate_pattern with LaTeX patterns"""
        latex_pattern = r'^\\[a-zA-Z]+(_\{[0-9]+\})?$'

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(pattern=latex_pattern, allow_alnum=True)
            def value(self, val):
                self._value = val

        obj = TestClass()
        # Valid LaTeX symbols
        for symbol in [r"\Pi", r"\rho", r"\alpha", r"\Pi_{0}", r"\rho_{1}"]:
            obj.value = symbol
            self.assertEqual(obj.value, symbol)

        # Valid alphanumeric also works (default allow_alnum=True)
        obj.value = "V"
        self.assertEqual(obj.value, "V")

    def test_validate_pattern_invalid(self):
        """Test validate_pattern with invalid values"""
        latex_pattern = r'^\\[a-zA-Z]+(_\{[0-9]+\})?$'

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(pattern=latex_pattern, allow_alnum=True)
            def value(self, val):
                self._value = val

        obj = TestClass()
        # Invalid symbols (not alphanumeric, not matching LaTeX pattern)
        for invalid in ["@#$", "test-value", "x y", "\\", "\\123"]:
            with self.assertRaises(ValueError) as ctx:
                obj.value = invalid
            self.assertIn("must be alphanumeric or match pattern", str(ctx.exception))

    def test_validate_pattern_no_alnum(self):
        """Test validate_pattern with allow_alnum=False"""
        latex_pattern = r'^\\[a-zA-Z]+$'

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(pattern=latex_pattern, allow_alnum=False)
            def value(self, val):
                self._value = val

        obj = TestClass()
        # LaTeX works
        obj.value = r"\Pi"
        self.assertEqual(obj.value, r"\Pi")

        # Alphanumeric should fail
        with self.assertRaises(ValueError):
            obj.value = "V"

    def test_validate_pattern_custom_examples(self):
        """Test validate_pattern with custom examples"""
        latex_pattern = r'^\\[a-zA-Z]+$'

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(str)
            @validate_pattern(pattern=latex_pattern, allow_alnum=True, examples="'\\\\alpha', '\\\\beta'")
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = "!!!"
        error_msg = str(ctx.exception)
        self.assertIn("Examples:", error_msg)
        self.assertIn("\\\\alpha", error_msg)

    def test_validate_list_types_valid(self):
        """Test validate_list_types with valid element types"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(list, allow_none=False)
            @validate_emptiness()
            @validate_list_types(int, float)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = [1, 2, 3]
        self.assertEqual(obj.value, [1, 2, 3])

        obj.value = [1.5, 2.5, 3.5]
        self.assertEqual(obj.value, [1.5, 2.5, 3.5])

        obj.value = [1, 2.5, 3]
        self.assertEqual(obj.value, [1, 2.5, 3])

    def test_validate_list_types_invalid(self):
        """Test validate_list_types with invalid element types"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(list, allow_none=False)
            @validate_emptiness()
            @validate_list_types(int, float)
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = [1, 2, "three"]
        self.assertIn("must contain only", str(ctx.exception))
        self.assertIn("int or float", str(ctx.exception))

    def test_validate_list_types_single_type(self):
        """Test validate_list_types with single element type"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(list, allow_none=False)
            @validate_emptiness()
            @validate_list_types(int)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = [1, 2, 3]
        self.assertEqual(obj.value, [1, 2, 3])

        with self.assertRaises(ValueError) as ctx:
            obj.value = [1, 2.5, 3]
        self.assertIn("must contain only", str(ctx.exception))

    def test_validate_dict_types_valid(self):
        """Test validate_dict_types with valid key and value types"""
        # Create a simple Variable-like class for testing
        class Variable:
            def __init__(self, name):
                self.name = name

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(dict, allow_none=False)
            @validate_emptiness()
            @validate_dict_types(str, Variable)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = {"a": Variable("A"), "b": Variable("B")}
        self.assertEqual(len(obj.value), 2)
        self.assertIn("a", obj.value)

    def test_validate_dict_types_invalid_keys(self):
        """Test validate_dict_types with invalid key types"""
        class Variable:
            def __init__(self, name):
                self.name = name

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(dict, allow_none=False)
            @validate_emptiness()
            @validate_dict_types(str, Variable)
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = {1: Variable("A"), "b": Variable("B")}
        self.assertIn("keys must be str", str(ctx.exception))
        self.assertIn("invalid keys", str(ctx.exception))

    def test_validate_dict_types_invalid_values(self):
        """Test validate_dict_types with invalid value types"""
        class Variable:
            def __init__(self, name):
                self.name = name

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(dict, allow_none=False)
            @validate_emptiness()
            @validate_dict_types(str, Variable)
            def value(self, val):
                self._value = val

        obj = TestClass()
        with self.assertRaises(ValueError) as ctx:
            obj.value = {"a": Variable("A"), "b": "not a variable"}
        self.assertIn("values must be Variable", str(ctx.exception))
        self.assertIn("invalid value(s)", str(ctx.exception))

    def test_validate_dict_types_simple(self):
        """Test validate_dict_types with simple types"""
        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(dict, allow_none=False)
            @validate_emptiness()
            @validate_dict_types(str, int)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(obj.value, {"a": 1, "b": 2, "c": 3})

        with self.assertRaises(ValueError) as ctx:
            obj.value = {"a": 1, "b": "two"}
        self.assertIn("values must be int", str(ctx.exception))

    def test_validate_custom(self):
        """Test validate_custom decorator with custom validation logic"""
        def check_positive(self, value):
            """Ensure value is positive."""
            if value is not None and value < 0:
                raise ValueError(f"Value must be positive, got {value}")

        class TestClass:
            @property
            def value(self):
                return self._value

            @value.setter
            @validate_type(int, float)
            @validate_custom(check_positive)
            def value(self, val):
                self._value = val

        obj = TestClass()
        obj.value = 10  # Should work
        self.assertEqual(obj.value, 10)

        obj.value = 0  # Should work (zero is not negative)
        self.assertEqual(obj.value, 0)

        with self.assertRaises(ValueError) as ctx:
            obj.value = -5
        self.assertIn("must be positive", str(ctx.exception))

        # Test custom validator with range consistency
        def check_range_consistency(self, value):
            """Ensure min does not exceed max."""
            if value is not None and hasattr(self, '_max') and self._max is not None:
                if value > self._max:
                    raise ValueError(f"min {value} > max {self._max}")

        class TestRangeClass:
            def __init__(self):
                self._min = None
                self._max = 100

            @property
            def min(self):
                return self._min

            @min.setter
            @validate_type(int, float)
            @validate_custom(check_range_consistency)
            def min(self, val):
                self._min = val

        obj_range = TestRangeClass()
        obj_range.min = 50  # Should work
        self.assertEqual(obj_range.min, 50)

        with self.assertRaises(ValueError) as ctx:
            obj_range.min = 150  # Exceeds max
        self.assertIn("min 150 > max 100", str(ctx.exception))
