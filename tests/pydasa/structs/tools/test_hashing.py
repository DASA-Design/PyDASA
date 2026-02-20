# -*- coding: utf-8 -*-
"""
Test Module for hashing.py
===========================================

Tests for hash utility functions in PyDASA.
"""

import unittest
import pytest
from pydasa.structs.tools.hashing import mad_hash
from tests.pydasa.data.test_data import get_hashing_test_data


class TestMADHash(unittest.TestCase):
    """Test cases for MAD compression hash function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_hashing_test_data()
        self.scale = self.test_data["MAD_PARAMS"]["scale"]
        self.shift = self.test_data["MAD_PARAMS"]["shift"]
        self.prime = self.test_data["MAD_PARAMS"]["prime"]
        self.mcap = self.test_data["MAD_PARAMS"]["mcap"]

    def test_mad_hash_with_integer(self) -> None:
        """Test MAD hash with integer keys."""
        for key in self.test_data["INTEGER_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_string(self) -> None:
        """Test MAD hash with string keys."""
        for key in self.test_data["STRING_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_float(self) -> None:
        """Test MAD hash with float keys."""
        for key in self.test_data["FLOAT_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_tuple(self) -> None:
        """Test MAD hash with tuple keys."""
        for key in self.test_data["TUPLE_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_list(self) -> None:
        """Test MAD hash with list keys (converted to string)."""
        for key in self.test_data["LIST_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_set(self) -> None:
        """Test MAD hash with set keys (converted to string)."""
        for key in self.test_data["SET_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_with_dict(self) -> None:
        """Test MAD hash with dict keys (converted to string)."""
        for key in self.test_data["DICT_KEYS"]:
            result = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert isinstance(result, int)
            assert 0 <= result < self.mcap

    def test_mad_hash_consistency(self) -> None:
        """Test that same input produces same hash."""
        for key in self.test_data["STRING_KEYS"]:
            result1 = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)
            result2 = mad_hash(key, self.scale, self.shift, self.prime, self.mcap)

            assert result1 == result2

    def test_mad_hash_distribution(self) -> None:
        """Test that different keys produce different hashes (when possible)."""
        keys = self.test_data["STRING_KEYS"]
        hashes = [mad_hash(k, self.scale, self.shift, self.prime, self.mcap) for k in keys]

        # At least some hashes should be different
        assert len(set(hashes)) > 1
