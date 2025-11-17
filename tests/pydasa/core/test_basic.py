# -*- coding: utf-8 -*-
"""
Test Module for basic.py
===========================================

Tests for base validation classes in PyDASA.
"""

import unittest
import pytest
from pydasa.core.basic import SymValidation, IdxValidation, Validation
from tests.pydasa.data.test_data import get_basic_test_data


class TestSymValidation(unittest.TestCase):
    """Test cases for SymValidation class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_sym_validation(self) -> None:
        """Test creating SymValidation with default values."""
        obj = SymValidation()

        assert obj is not None
        assert obj._sym == ""
        assert obj._alias == ""
        assert obj._fwk == "PHYSICAL"
        assert isinstance(obj, SymValidation)

    def test_custom_sym_validation(self) -> None:
        """Test creating SymValidation with custom values."""
        for data in self.test_data["VALID_SYM_DATA"]:
            obj = SymValidation(
                _sym=data["_sym"],
                _alias=data["_alias"],
                _fwk=data["_fwk"]
            )

            assert obj._sym == data["_sym"]
            assert obj._alias == data["_alias"]
            assert obj._fwk == data["_fwk"]

    def test_sym_property_getter(self) -> None:
        """Test sym property getter."""
        obj = SymValidation(_sym="L")
        assert obj.sym == "L"

    def test_sym_property_setter(self) -> None:
        """Test sym property setter with valid values."""
        obj = SymValidation()
        for sym in self.test_data["VALID_SYMBOLS"]:
            obj.sym = sym
            assert obj.sym == sym

    def test_sym_validation_invalid(self) -> None:
        """Test sym validation with invalid values."""
        obj = SymValidation()
        for invalid_sym in self.test_data["INVALID_SYMBOLS"]:
            with pytest.raises(ValueError) as excinfo:
                obj.sym = invalid_sym
            assert "Symbol must be a non-empty string. Provided:" in str(excinfo.value)

    def test_fwk_property_getter(self) -> None:
        """Test fwk property getter."""
        obj = SymValidation(_fwk="COMPUTATION")
        assert obj.fwk == "COMPUTATION"

    def test_fwk_property_setter(self) -> None:
        """Test fwk property setter with valid values."""
        obj = SymValidation()
        for fwk in self.test_data["VALID_FRAMEWORKS"]:
            obj.fwk = fwk
            assert obj.fwk == fwk

    def test_fwk_validation_invalid(self) -> None:
        """Test fwk validation with invalid values."""
        obj = SymValidation()
        for invalid_fwk in self.test_data["INVALID_FRAMEWORKS"]:
            with pytest.raises(ValueError) as excinfo:
                obj.fwk = invalid_fwk
            assert "Invalid framework" in str(excinfo.value)

    def test_alias_property_getter(self) -> None:
        """Test alias property getter."""
        obj = SymValidation(_alias="rho_1")
        assert obj.alias == "rho_1"

    def test_alias_property_setter(self) -> None:
        """Test alias property setter with valid values."""
        obj = SymValidation()
        for alias in self.test_data["VALID_ALIASES"]:
            obj.alias = alias
            assert obj.alias == alias

    def test_alias_validation_invalid(self) -> None:
        """Test alias validation with invalid values."""
        obj = SymValidation()
        with pytest.raises(ValueError) as excinfo:
            obj.alias = "   "
        assert "Symbol must be a non-empty string" in str(excinfo.value)

    def test_latex_symbols(self) -> None:
        """Test SymValidation with LaTeX symbols."""
        for latex_sym in self.test_data["LATEX_SYMBOLS"]:
            obj = SymValidation(_sym=latex_sym)
            assert obj.sym == latex_sym


class TestIdxValidation(unittest.TestCase):
    """Test cases for IdxValidation class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_idx_validation(self) -> None:
        """Test creating IdxValidation with default values."""
        obj = IdxValidation()

        assert obj is not None
        assert obj._idx == -1
        assert obj._sym == ""
        assert obj._alias == ""
        assert obj._fwk == "PHYSICAL"
        assert isinstance(obj, IdxValidation)
        assert isinstance(obj, SymValidation)

    def test_custom_idx_validation(self) -> None:
        """Test creating IdxValidation with custom values."""
        for data in self.test_data["VALID_IDX_DATA"]:
            obj = IdxValidation(
                _idx=data["_idx"],
                _sym=data["_sym"],
                _alias=data["_alias"],
                _fwk=data["_fwk"]
            )

            assert obj._idx == data["_idx"]
            assert obj._sym == data["_sym"]
            assert obj._alias == data["_alias"]
            assert obj._fwk == data["_fwk"]

    def test_idx_property_getter(self) -> None:
        """Test idx property getter."""
        obj = IdxValidation(_idx=5)
        assert obj.idx == 5

    def test_idx_property_setter(self) -> None:
        """Test idx property setter with valid values."""
        obj = IdxValidation()
        for idx in self.test_data["VALID_INDICES"]:
            obj.idx = idx
            assert obj.idx == idx

    def test_idx_validation_invalid(self) -> None:
        """Test idx validation with invalid values."""
        obj = IdxValidation()
        for invalid_idx in self.test_data["INVALID_INDICES"]:
            with pytest.raises(ValueError) as excinfo:
                obj.idx = invalid_idx
            assert "Index must be a non-negative integer" in str(excinfo.value)

    def test_idx_inheritance(self) -> None:
        """Test that IdxValidation inherits SymValidation properties."""
        obj = IdxValidation(_idx=0, _sym="L", _fwk="PHYSICAL")

        assert obj.idx == 0
        assert obj.sym == "L"
        assert obj.fwk == "PHYSICAL"


class TestValidation(unittest.TestCase):
    """Test cases for Validation class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_basic_test_data()

    def test_default_validation(self) -> None:
        """Test creating Validation with default values."""
        obj = Validation()

        assert obj is not None
        assert obj._idx == -1
        assert obj._sym == ""
        assert obj._alias == ""
        assert obj._fwk == "PHYSICAL"
        assert obj.name == ""
        assert obj.description == ""
        assert isinstance(obj, Validation)
        assert isinstance(obj, IdxValidation)
        assert isinstance(obj, SymValidation)

    def test_custom_validation(self) -> None:
        """Test creating Validation with custom values."""
        for data in self.test_data["VALID_VALIDATION_DATA"]:
            obj = Validation(
                _idx=data["_idx"],
                _sym=data["_sym"],
                _alias=data["_alias"],
                _fwk=data["_fwk"],
                name=data["name"],
                description=data["description"]
            )

            assert obj._idx == data["_idx"]
            assert obj._sym == data["_sym"]
            assert obj._alias == data["_alias"]
            assert obj._fwk == data["_fwk"]
            assert obj.name == data["name"]
            # Description should be capitalized
            assert obj.description[0].isupper()

    def test_name_validation(self) -> None:
        """Test name validation."""
        obj = Validation()

        # Test valid names
        for name in self.test_data["VALID_NAMES"]:
            obj._validate_name(name)
            assert obj.name == name.strip()

        # Test invalid names
        for invalid_name in self.test_data["INVALID_NAMES"]:
            with pytest.raises(ValueError) as excinfo:
                obj._validate_name(invalid_name)
            assert "Name must be a non-empty string" in str(excinfo.value)

    def test_description_capitalization(self) -> None:
        """Test that description is automatically capitalized."""
        descriptions = [
            ("test description", "Test description"),
            ("another test", "Another test"),
            ("UPPERCASE", "Uppercase"),
        ]

        for desc_in, desc_out in descriptions:
            obj = Validation(description=desc_in)
            assert obj.description == desc_out

    def test_str_representation(self) -> None:
        """Test __str__ method."""
        obj = Validation(
            _idx=0,
            _sym="L",
            _alias="L",
            _fwk="PHYSICAL",
            name="Length",
            description="Physical length dimension"
        )

        str_repr = str(obj)
        assert "Validation(" in str_repr
        assert "idx=0" in str_repr
        assert "sym='L'" in str_repr
        assert "fwk='PHYSICAL'" in str_repr
        assert "name='Length'" in str_repr

    def test_repr_representation(self) -> None:
        """Test __repr__ method."""
        obj = Validation(
            _idx=0,
            _sym="L",
            name="Length"
        )

        repr_str = repr(obj)
        assert "Validation(" in repr_str
        assert str(obj) == repr_str

    def test_inheritance_chain(self) -> None:
        """Test full inheritance chain."""
        obj = Validation(
            _idx=1,
            _sym="M",
            _alias="M",
            _fwk="PHYSICAL",
            name="Mass",
            description="Physical mass dimension"
        )

        # Test IdxValidation properties
        assert obj.idx == 1

        # Test SymValidation properties
        assert obj.sym == "M"
        assert obj.alias == "M"
        assert obj.fwk == "PHYSICAL"

        # Test Validation properties
        assert obj.name == "Mass"
        assert obj.description == "Physical mass dimension"


# Run tests with: pytest tests/pydasa/core/test_basic.py -v
