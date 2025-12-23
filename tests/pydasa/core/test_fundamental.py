# -*- coding: utf-8 -*-
"""
Test Module for fundamental.py
===========================================

Tests for the **Dimension** class in *PyDASA*.
"""

import unittest
import pytest
from pydasa.core.fundamental import Dimension
from tests.pydasa.data.test_data import get_dimension_test_data


class TestDimension(unittest.TestCase):
    """Test cases for Dimension class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_dimension_test_data()

    def test_default_dimension(self) -> None:
        """Test creating Dimension with default values."""
        dim = Dimension()

        assert dim is not None
        assert dim._idx == -1
        assert dim._sym == ""
        assert dim._alias == ""
        assert dim._fwk == "PHYSICAL"
        assert dim._unit == ""
        assert dim.name == ""
        assert dim.description == ""
        assert isinstance(dim, Dimension)

    def test_specific_dimensions(self) -> None:
        """Test creating specific dimensions (physical, computation, software)."""
        dimension_types = [
            self.test_data["PHYSICAL_DATA"],
            self.test_data["COMPUTATION_DATA"],
            self.test_data["SOFTWARE_DATA"]
        ]

        for data in dimension_types:
            dim = Dimension(
                _idx=data["_idx"],
                _sym=data["_sym"],
                _alias=data["_alias"],
                _fwk=data["_fwk"],
                _unit=data["_unit"],
                name=data["name"],
                description=data["description"])

            assert dim._idx == data["_idx"]
            assert dim._sym == data["_sym"]
            assert dim._alias == data["_alias"]
            assert dim._fwk == data["_fwk"]
            assert dim._unit == data["_unit"]
            assert dim.name == data["name"]
            assert dim.description[0].isupper()

    def test_unit_property_getter(self) -> None:
        """Test unit property getter."""
        dim = Dimension(_unit="m")
        assert dim.unit == "m"

    def test_unit_property_setter(self) -> None:
        """Test unit property setter with valid values."""
        dim = Dimension()
        for unit in self.test_data["VALID_UNITS"]:
            dim.unit = unit
            assert dim.unit == unit

    def test_unit_validation_invalid(self) -> None:
        """Test unit validation with invalid values."""
        dim = Dimension()
        for invalid_unit in self.test_data["INVALID_UNIT_DATA"]:
            with pytest.raises(ValueError) as excinfo:
                dim.unit = invalid_unit
            assert "Unit must be a non-empty string" in str(excinfo.value)

    def test_to_dict(self) -> None:
        """Test converting Dimension to dictionary."""
        data = self.test_data["PHYSICAL_DATA"]
        dim = Dimension(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _unit=data["_unit"],
            name=data["name"],
            description=data["description"])

        result = dim.to_dict()

        assert isinstance(result, dict)
        assert result["idx"] == data["_idx"]
        assert result["sym"] == data["_sym"]
        assert result["alias"] == data["_alias"]
        assert result["fwk"] == data["_fwk"]
        assert result["unit"] == data["_unit"]
        assert result["name"] == data["name"]
        assert "description" in result

    def test_from_dict(self) -> None:
        """Test creating Dimension from dictionary."""
        data = self.test_data["PHYSICAL_DATA"]
        dict_data = {
            "idx": data["_idx"],
            "sym": data["_sym"],
            "alias": data["_alias"],
            "fwk": data["_fwk"],
            "unit": data["_unit"],
            "name": data["name"],
            "description": data["description"]
        }

        dim = Dimension.from_dict(dict_data)

        assert dim._idx == data["_idx"]
        assert dim._sym == data["_sym"]
        assert dim._alias == data["_alias"]
        assert dim._fwk == data["_fwk"]
        assert dim._unit == data["_unit"]
        assert dim.name == data["name"]

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test round-trip conversion between Dimension and dict."""
        data = self.test_data["COMPUTATION_DATA"]
        dim1 = Dimension(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _unit=data["_unit"],
            name=data["name"],
            description=data["description"])

        # Convert to dict and back
        dict_data = dim1.to_dict()
        dim2 = Dimension.from_dict(dict_data)

        # Test equality
        assert dim1 == dim2

    def test_equality_same_dimensions(self) -> None:
        """Test equality of identical Dimension objects."""
        data = self.test_data["PHYSICAL_DATA"]
        dim1 = Dimension(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _unit=data["_unit"],
            name=data["name"],
            description=data["description"])

        dim2 = Dimension(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _unit=data["_unit"],
            name=data["name"],
            description=data["description"])

        assert dim1 == dim2

    def test_equality_different_dimensions(self) -> None:
        """Test inequality of different Dimension objects."""
        data1 = self.test_data["PHYSICAL_DATA"]
        data2 = self.test_data["COMPUTATION_DATA"]

        dim1 = Dimension(
            _idx=data1["_idx"],
            _sym=data1["_sym"],
            _alias=data1["_alias"],
            _fwk=data1["_fwk"],
            _unit=data1["_unit"],
            name=data1["name"],
            description=data1["description"])
        dim2 = Dimension(
            _idx=data2["_idx"],
            _sym=data2["_sym"],
            _alias=data2["_alias"],
            _fwk=data2["_fwk"],
            _unit=data2["_unit"],
            name=data2["name"],
            description=data2["description"])

        assert dim1 != dim2

    def test_equality_different_sym(self) -> None:
        """Test inequality when symbol differs."""
        # data = self.test_data["PHYSICAL_DATA"]
        dim1 = Dimension(_sym="L", _fwk="PHYSICAL", _unit="m")
        dim2 = Dimension(_sym="M", _fwk="PHYSICAL", _unit="m")

        assert dim1 != dim2

    def test_equality_different_fwk(self) -> None:
        """Test inequality when framework differs."""
        dim1 = Dimension(_sym="T", _fwk="PHYSICAL", _unit="s")
        dim2 = Dimension(_sym="T", _fwk="COMPUTATION", _unit="s")

        assert dim1 != dim2

    def test_equality_different_unit(self) -> None:
        """Test inequality when unit differs."""
        dim1 = Dimension(_sym="L", _fwk="PHYSICAL", _unit="m")
        dim2 = Dimension(_sym="L", _fwk="PHYSICAL", _unit="km")

        assert dim1 != dim2

    def test_equality_with_non_dimension(self) -> None:
        """Test equality with non-Dimension object."""
        dim = Dimension(_sym="L", _fwk="PHYSICAL", _unit="m")

        assert dim != "not a dimension"
        assert dim != 123
        assert dim is not None
        assert dim != {"sym": "L"}

    def test_inheritance_from_validation(self) -> None:
        """Test that Dimension inherits from Validation."""
        from pydasa.core.basic import Validation, IdxValidation, SymValidation

        dim = Dimension(_sym="L", _fwk="PHYSICAL", _unit="m")

        assert isinstance(dim, Validation)
        assert isinstance(dim, IdxValidation)
        assert isinstance(dim, SymValidation)

    def test_dimension_with_latex_symbol(self) -> None:
        """Test Dimension with LaTeX symbol."""
        dim = Dimension(
            _sym="\\alpha",
            _alias="alpha",
            _fwk="PHYSICAL",
            _unit="rad")

        assert dim._sym == "\\alpha"
        assert dim._alias == "alpha"
        assert dim._unit == "rad"

    def test_dimension_str_representation(self) -> None:
        """Test __str__ method inherited from Validation."""
        data = self.test_data["PHYSICAL_DATA"]
        dim = Dimension(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _unit=data["_unit"],
            name=data["name"],
            description=data["description"])

        str_repr = str(dim)
        assert "Validation(" in str_repr or "Dimension" in str_repr
        assert data["_sym"] in str_repr
