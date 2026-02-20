# -*- coding: utf-8 -*-
"""
Test Module for math.py
===========================================

Tests for math utility functions in PyDASA.
"""

import unittest
import pytest
from pydasa.structs.tools.math import is_prime, next_prime, previous_prime, gfactorial
from tests.pydasa.data.test_data import get_math_test_data


class TestIsPrime(unittest.TestCase):
    """Test cases for is_prime function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_math_test_data()

    def test_is_prime_with_primes(self) -> None:
        """Test is_prime correctly identifies prime numbers."""
        for num in self.test_data["PRIME_NUMBERS"]:
            assert is_prime(num) is True, f"{num} should be prime"

    def test_is_prime_with_non_primes(self) -> None:
        """Test is_prime correctly identifies non-prime numbers."""
        for num in self.test_data["NON_PRIME_NUMBERS"]:
            assert is_prime(num) is False, f"{num} should not be prime"

    def test_is_prime_with_negative(self) -> None:
        """Test is_prime returns False for negative numbers."""
        for num in self.test_data["NEGATIVE_NUMBERS"]:
            assert is_prime(num) is False


class TestNextPrime(unittest.TestCase):
    """Test cases for next_prime function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_math_test_data()

    def test_next_prime_values(self) -> None:
        """Test next_prime returns correct next prime."""
        for n, expected in self.test_data["NEXT_PRIME_CASES"]:
            result = next_prime(n)
            assert result == expected, f"next_prime({n}) should be {expected}, got {result}"
            assert is_prime(result), f"next_prime({n}) = {result} should be prime"


class TestPreviousPrime(unittest.TestCase):
    """Test cases for previous_prime function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_math_test_data()

    def test_previous_prime_values(self) -> None:
        """Test previous_prime returns correct previous prime."""
        for n, expected in self.test_data["PREVIOUS_PRIME_CASES"]:
            result = previous_prime(n)
            assert result == expected, f"previous_prime({n}) should be {expected}, got {result}"
            assert is_prime(result), f"previous_prime({n}) = {result} should be prime"


class TestGFactorial(unittest.TestCase):
    """Test cases for gfactorial function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_math_test_data()

    def test_gfactorial_integers(self) -> None:
        """Test gfactorial with integer inputs."""
        for x, expected in self.test_data["FACTORIAL_INTEGERS"]:
            result = gfactorial(x)
            assert result == expected, f"gfactorial({x}) should be {expected}, got {result}"
            assert isinstance(result, int)

    def test_gfactorial_floats(self) -> None:
        """Test gfactorial with float inputs (gamma function)."""
        for x, expected in self.test_data["FACTORIAL_FLOATS"]:
            result = gfactorial(x)
            assert abs(result - expected) < 1e-10, f"gfactorial({x}) should be close to {expected}"
            assert isinstance(result, float)

    def test_gfactorial_with_precision(self) -> None:
        """Test gfactorial with precision parameter."""
        for x, prec, expected in self.test_data["FACTORIAL_PRECISION"]:
            result = gfactorial(x, prec=prec)
            assert result == expected, f"gfactorial({x}, prec={prec}) should be {expected}"

    def test_gfactorial_invalid(self) -> None:
        """Test gfactorial raises ValueError for negative integers."""
        for x in self.test_data["FACTORIAL_INVALID"]:
            with pytest.raises(ValueError) as excinfo:
                gfactorial(x)
            assert "negative integers" in str(excinfo.value).lower()
