# -*- coding: utf-8 -*-
"""
Test Module for parameter.py
===========================================

Integration tests for the **Variable** class composition in *PyDASA*.

This module tests the Variable class as a whole, verifying that all four compositional perspectives (Conceptual, Symbolic, Numerical, Statistical) work together correctly.

Individual perspective tests are in the specs/ subdirectory:
- specs/test_conceptual.py - ConceptualSpecs tests
- specs/test_symbolic.py - SymbolicSpecs tests
- specs/test_numerical.py - NumericalSpecs tests
- specs/test_statistical.py - StatisticalSpecs tests
"""

import unittest
import pytest

from pydasa.dimensional.framework import Schema
from pydasa.elements.parameter import Variable
from tests.pydasa.data.test_data import get_variable_test_data


class TestVariable(unittest.TestCase):
    """Integration test cases for Variable class composition."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_variable_test_data()

        # resseting config due to singletton patterm fix later
        self.test_scheme = Schema(_fwk="PHYSICAL")
        self.test_scheme.update_global_config()

    # Initialization tests - verify all 4 perspectives initialize correctly
    def test_default_initialization(self) -> None:
        """Test creating Variable with default values from all perspectives."""
        var = Variable()

        assert var is not None
        # Foundation (via ConceptualSpecs)
        assert var._idx == -1
        assert var._sym.startswith("V_")
        assert var._alias != ""
        assert var._fwk == "PHYSICAL"
        # ConceptualSpecs
        assert var._cat == "IN"
        assert var.relevant is False
        # SymbolicSpecs
        assert var._dims == ""
        assert var._units == ""
        # NumericalSpecs - all None by default
        # StatisticalSpecs
        assert var.dist_type == "uniform"
        assert isinstance(var, Variable)

    def test_specific_variable_types(self) -> None:
        """Test creating specific variable types across different frameworks."""
        variable_types = [
            self.test_data["PHYSICAL_VARIABLE"],
            self.test_data["COMPUTATION_VARIABLE"],
            self.test_data["SOFTWARE_VARIABLE"]
        ]

        for data in variable_types:
            # Update dimensional scheme for each framework type
            scheme = Schema(_fwk=data["_fwk"])
            scheme.update_global_config()

            var = Variable(
                _idx=data["_idx"],
                _sym=data["_sym"],
                _alias=data["_alias"],
                _fwk=data["_fwk"],
                _cat=data["_cat"],
                _dims=data["_dims"],
                _units=data["_units"],
                _name=data["_name"],
                description=data["description"])

            # Verify Foundation attributes
            assert var._idx == data["_idx"]
            assert var._sym == data["_sym"]
            assert var._alias == data["_alias"]
            assert var._fwk == data["_fwk"]
            assert var.name == data["_name"]

            # Verify ConceptualSpecs
            assert var._cat == data["_cat"]

            # Verify SymbolicSpecs
            assert var._dims == data["_dims"]
            assert var._units == data["_units"]

    # Utility methods tests - verify integration across all specs
    def test_clear_resets_all_attributes(self) -> None:
        """Test clear method resets attributes across all 4 perspectives."""
        data = self.test_data["PHYSICAL_VARIABLE"]
        var = Variable(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _cat=data["_cat"],
            _dims=data["_dims"],
            _units=data["_units"],
            _name=data["_name"],
            description=data["description"])

        var.clear()

        # Foundation attributes
        assert var._idx == -1
        assert var._sym == ""
        assert var._alias == ""
        assert var._fwk == "PHYSICAL"
        assert var.name == ""
        assert var.description == ""

        # ConceptualSpecs
        assert var._cat == "IN"
        assert var.relevant is False

        # SymbolicSpecs
        assert var._dims == ""
        assert var._units == ""

        # StatisticalSpecs
        assert var.dist_type == "uniform"

    def test_to_dict_structure(self) -> None:
        """Test to_dict method returns correct structure."""
        data = self.test_data["PHYSICAL_VARIABLE"]
        var = Variable(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _cat=data["_cat"],
            _dims=data["_dims"],
            _units=data["_units"],
            _name=data["_name"],
            description=data["description"])

        result = var.to_dict()

        assert isinstance(result, dict)
        assert "idx" in result
        assert "sym" in result
        assert "alias" in result
        assert "fwk" in result
        assert "cat" in result
        assert "dims" in result
        assert "units" in result
        assert "name" in result
        assert "description" in result

    def test_to_dict_values(self) -> None:
        """Test to_dict method returns correct values."""
        data = self.test_data["PHYSICAL_VARIABLE"]
        var = Variable(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _cat=data["_cat"],
            _dims=data["_dims"],
            _units=data["_units"],
            _name=data["_name"],
            description=data["description"])

        result = var.to_dict()

        assert result["idx"] == data["_idx"]
        assert result["sym"] == data["_sym"]
        assert result["alias"] == data["_alias"]
        assert result["fwk"] == data["_fwk"]
        assert result["cat"] == data["_cat"]
        assert result["dims"] == data["_dims"]
        assert result["units"] == data["_units"]
        assert result["name"] == data["_name"]

    def test_from_dict_creates_instance(self) -> None:
        """Test from_dict method creates Variable instance."""
        data = self.test_data["PHYSICAL_VARIABLE"]

        var = Variable.from_dict(data)

        assert isinstance(var, Variable)
        assert var._idx == data["_idx"]
        assert var._sym == data["_sym"]
        assert var._alias == data["_alias"]
        assert var._fwk == data["_fwk"]
        assert var._cat == data["_cat"]

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test round-trip conversion between Variable and dict."""
        data = self.test_data["PHYSICAL_VARIABLE"]
        var1 = Variable(
            _idx=data["_idx"],
            _sym=data["_sym"],
            _alias=data["_alias"],
            _fwk=data["_fwk"],
            _cat=data["_cat"],
            _dims=data["_dims"],
            _units=data["_units"],
            _name=data["_name"],
            description=data["description"])

        # Convert to dict and back
        dict_data = var1.to_dict()
        var2 = Variable.from_dict(dict_data)

        # Test key attributes match
        assert var1._idx == var2._idx
        assert var1._sym == var2._sym
        assert var1._alias == var2._alias
        assert var1._fwk == var2._fwk
        assert var1._cat == var2._cat
        assert var1._dims == var2._dims
        assert var1._units == var2._units
        assert var1.name == var2.name

    def test_inheritance_from_validation(self) -> None:
        """Test that Variable inherits from Foundation hierarchy and all 4 spec classes."""
        from pydasa.core.basic import Foundation, IdxBasis, SymBasis
        from pydasa.elements.specs.conceptual import ConceptualSpecs
        from pydasa.elements.specs.symbolic import SymbolicSpecs
        from pydasa.elements.specs.numerical import NumericalSpecs
        from pydasa.elements.specs.statistical import StatisticalSpecs

        var = Variable()

        # Foundation hierarchy
        assert isinstance(var, Variable)
        assert isinstance(var, Foundation)
        assert isinstance(var, IdxBasis)
        assert isinstance(var, SymBasis)

        # Four compositional perspectives
        assert isinstance(var, ConceptualSpecs)
        assert isinstance(var, SymbolicSpecs)
        assert isinstance(var, NumericalSpecs)
        assert isinstance(var, StatisticalSpecs)
