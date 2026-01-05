# -*- coding: utf-8 -*-
"""
Test Module for specs/conceptual.py
===========================================

Tests for the **ConceptualSpecs** class in *PyDASA*.
"""

import unittest
import pytest

from pydasa.elements.parameter import Variable
# from pydasa.elements.specs.conceptual import ConceptualSpecs
from tests.pydasa.data.test_data import get_variable_test_data


class TestConceptualSpecs(unittest.TestCase):
    """Test cases for **ConceptualSpecs** class via **Variable**."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_variable_test_data()

    # Category property tests
    def test_cat_getter(self) -> None:
        """Test cat property getter."""
        spec = Variable(_cat="OUT")
        assert spec.cat == "OUT"

    def test_cat_setter_valid(self) -> None:
        """Test cat property setter with valid values."""
        spec = Variable()
        for cat in self.test_data["VALID_CATEGORIES"]:
            spec.cat = cat
            assert spec.cat == cat.upper()

    def test_cat_setter_invalid(self) -> None:
        """Test cat property setter with invalid values."""
        spec = Variable()
        for invalid_cat in self.test_data["INVALID_CATEGORIES"]:
            with pytest.raises(ValueError) as excinfo:
                spec.cat = invalid_cat
            assert "Invalid cat" in str(excinfo.value)
