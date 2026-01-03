# -*- coding: utf-8 -*-
"""
Test Module for basic.py
===========================================

Tests for base validation classes in PyDASA.
"""

import unittest
import pytest
from pydasa.core.basic import SymBasis, IdxBasis, Foundation
from tests.pydasa.data.test_data import get_basic_test_data


class TestSymBase(unittest.TestCase):
    """Test cases for SymBasis class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_initialization(self) -> None:
        """Test creating SymBasis with default values."""
        obj = SymBasis()

        assert obj is not None
        assert obj._sym == ""
        assert obj._alias == ""
        assert obj._fwk == "PHYSICAL"
        assert isinstance(obj, SymBasis)

    def test_custom_initialization(self) -> None:
        """Test creating SymBasis with custom values."""
        obj = SymBasis(_sym="L", _alias="L", _fwk="COMPUTATION")

        assert obj._sym == "L"
        assert obj._alias == "L"
        assert obj._fwk == "COMPUTATION"

    def test_property_setters(self) -> None:
        """Test property setters work correctly."""
        obj = SymBasis()

        obj.sym = "M"
        assert obj.sym == "M"

        obj.alias = "mass"
        assert obj.alias == "mass"

        obj.fwk = "SOFTWARE"
        assert obj.fwk == "SOFTWARE"

    def test_validation_invalid_values(self) -> None:
        """Test validation rejects invalid values."""
        obj = SymBasis()

        # Invalid sym type
        with pytest.raises((ValueError, TypeError)):
            obj.sym = 123   # type: ignore

        # Invalid framework
        with pytest.raises(ValueError):
            obj.fwk = "INVALID"

    def test_clear_method(self) -> None:
        """Test clear() resets to defaults."""
        obj = SymBasis(_sym="L", _alias="L", _fwk="COMPUTATION")
        obj.clear()

        assert obj._sym == ""
        assert obj._alias == ""
        assert obj._fwk == "PHYSICAL"


class TestIdxBase(unittest.TestCase):
    """Test cases for IdxBasis class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_initialization(self) -> None:
        """Test creating IdxBasis with default values."""
        obj = IdxBasis()

        assert obj is not None
        assert obj._idx == -1
        assert obj._sym == ""
        assert isinstance(obj, IdxBasis)
        assert isinstance(obj, SymBasis)

    def test_custom_initialization(self) -> None:
        """Test creating IdxBasis with custom values."""
        obj = IdxBasis(_idx=5, _sym="M", _alias="M", _fwk="SOFTWARE")

        assert obj._idx == 5
        assert obj._sym == "M"
        assert obj._alias == "M"
        assert obj._fwk == "SOFTWARE"

    def test_idx_property(self) -> None:
        """Test idx property getter and setter."""
        obj = IdxBasis()
        obj.idx = 10
        assert obj.idx == 10

    def test_validation_invalid_idx(self) -> None:
        """Test idx validation rejects invalid values."""
        obj = IdxBasis()

        with pytest.raises((ValueError, TypeError)):
            obj.idx = -5

        with pytest.raises((ValueError, TypeError)):
            obj.idx = "invalid"     # type: ignore

    def test_clear_method(self) -> None:
        """Test clear() resets all attributes including inherited."""
        obj = IdxBasis(_idx=5, _sym="M", _fwk="SOFTWARE")
        obj.clear()

        assert obj._idx == -1
        assert obj._sym == ""
        assert obj._fwk == "PHYSICAL"


class TestValidation(unittest.TestCase):
    """Test cases for Foundation class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_initialization(self) -> None:
        """Test creating Foundation with default values."""
        obj = Foundation()

        assert obj is not None
        assert obj._idx == -1
        assert obj.name == ""
        assert obj.description == ""
        assert isinstance(obj, Foundation)
        assert isinstance(obj, IdxBasis)
        assert isinstance(obj, SymBasis)

    def test_custom_initialization(self) -> None:
        """Test creating Foundation with custom values."""
        obj = Foundation(
            _idx=0,
            _sym="L",
            _alias="L",
            _fwk="PHYSICAL",
            _name="Length",
            description="Physical length dimension"
        )

        assert obj._idx == 0
        assert obj._sym == "L"
        assert obj.name == "Length"
        assert obj.description == "Physical length dimension"

    def test_name_property(self) -> None:
        """Test name property getter and setter."""
        obj = Foundation()
        obj.name = "Mass"
        assert obj.name == "Mass"

    def test_validation_invalid_name(self) -> None:
        """Test name validation rejects invalid values."""
        obj = Foundation()

        with pytest.raises((ValueError, TypeError)):
            obj.name = 123      # type: ignore

        with pytest.raises((ValueError, TypeError)):
            obj.name = None     # type: ignore

    def test_str_representation(self) -> None:
        """Test __str__ and __repr__ methods."""
        obj = Foundation(_idx=0, _sym="L", _name="Length")
        
        str_repr = str(obj)
        assert "Foundation(" in str_repr
        assert "idx=0" in str_repr
        assert "sym='L'" in str_repr
        assert str(obj) == repr(obj)

    def test_clear_method(self) -> None:
        """Test clear() resets all Foundation attributes."""
        obj = Foundation(_idx=3, _sym="T", _name="Time", description="Time dimension")
        obj.clear()

        assert obj._idx == -1
        assert obj._sym == ""
        assert obj.name == ""
        assert obj.description == ""
