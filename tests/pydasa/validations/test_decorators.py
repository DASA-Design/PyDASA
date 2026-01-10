# -*- coding: utf-8 -*-
"""
Test module for validations.decorators
==========================================

Tests for decorator-based validation system.
"""

import unittest
# import re
from pydasa.validations.decorators import (
    validate_type,
    validate_emptiness,
    validate_choices,
    validate_range,
    validate_index,
    validate_pattern,
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
