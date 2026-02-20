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

    def test_schema_getter_and_setter(self) -> None:
        """Test schema property getter and setter."""
        from pydasa.dimensional.vaschy import Schema

        # Test default schema is None after init with CUSTOM framework
        spec = Variable(_fwk="CUSTOM")
        assert spec.schema is None

        # Test setting a schema
        test_schema = Schema(_fwk="PHYSICAL")
        spec.schema = test_schema
        assert spec.schema == test_schema

        # Test setting to None
        spec.schema = None
        assert spec.schema is None

    def test_schema_initialization_in_post_init(self) -> None:
        """Test that schema is auto-created for non-CUSTOM frameworks."""
        # When framework is PHYSICAL and no schema provided, it should create one
        spec = Variable(_fwk="PHYSICAL")
        assert spec.schema is not None
        assert spec.schema._fwk == "PHYSICAL"

        # When framework is CUSTOM, no auto-schema
        spec_custom = Variable(_fwk="CUSTOM")
        assert spec_custom.schema is None

    def test_clear_method(self) -> None:
        """Test clear() method resets conceptual attributes."""
        from pydasa.dimensional.vaschy import Schema

        # Create a variable with non-default values
        spec = Variable(_cat="OUT", _fwk="PHYSICAL")
        spec.schema = Schema(_fwk="COMPUTATION")
        spec.relevant = True

        # Clear and verify reset to defaults
        spec.clear()

        assert spec.cat == "IN"  # Default category
        assert spec.schema is None
        assert spec.relevant is False
