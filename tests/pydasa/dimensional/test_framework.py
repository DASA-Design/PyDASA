# -*- coding: utf-8 -*-
"""
Test Module for framework.py
===========================================

Simple tests for the **DimSchema** class in *PyDASA*.
"""

import unittest
import pytest

from pydasa.dimensional.framework import DimSchema
from pydasa.core.fundamental import Dimension
from tests.pydasa.data.test_data import get_framework_test_data
from pydasa.core import config as cfg


class TestDimSchema(unittest.TestCase):
    """Test cases for DimSchema class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.data = get_framework_test_data()

    def test_default_initialization(self) -> None:
        """Default PHYSICAL framework has FDUs and regex set."""
        scheme = DimSchema()
        assert scheme is not None
        assert isinstance(scheme, DimSchema)
        assert len(scheme.fdu_lt) > 0
        assert isinstance(scheme.fdu_regex, str) and scheme.fdu_regex != ""
        assert isinstance(scheme.fdu_pow_regex, str) and scheme.fdu_pow_regex != ""
        assert isinstance(scheme.fdu_no_pow_regex, str) and scheme.fdu_no_pow_regex != ""
        assert isinstance(scheme.fdu_sym_regex, str) and scheme.fdu_sym_regex != ""

    def test_framework_initialization_param(self) -> None:
        """Frameworks PHYSICAL/COMPUTATION/SOFTWARE initialize expected counts."""
        cases = [
            ("PHYSICAL", len(self.data["PHYSICAL_FDU_LIST"])),
            ("COMPUTATION", len(self.data["COMPUTATION_FDU_LIST"])),
            ("SOFTWARE", len(self.data["SOFTWARE_FDU_LIST"])),
        ]
        for fwk, expected in cases:
            scheme = DimSchema(_fwk=fwk)
            assert scheme.size == expected
            assert all(isinstance(d, Dimension) for d in scheme.fdu_lt)

    def test_custom_framework_initialization(self) -> None:
        """CUSTOM accepts list of dicts and converts to Dimension."""
        custom = self.data["CUSTOM_FDU_LIST"]
        scheme = DimSchema(_fwk="CUSTOM", _fdu_lt=custom)
        assert scheme.size == len(custom)
        assert all(isinstance(d, Dimension) for d in scheme.fdu_lt)
        # symbols preserved
        expected_syms = [d["_sym"] for d in custom]
        assert scheme.fdu_symbols == expected_syms

    def test_get_and_has_fdu(self) -> None:
        """get_fdu() and has_fdu() work for known symbols."""
        scheme = DimSchema(_fwk="PHYSICAL")
        for sym in self.data["PHYSICAL_SYMBOLS"]:
            assert scheme.has_fdu(sym) is True
            fdu = scheme.get_fdu(sym)
            assert isinstance(fdu, Dimension)
            assert fdu.sym == sym
        # invalid symbol
        assert scheme.has_fdu("ZZZ") is False
        assert scheme.get_fdu("ZZZ") is None

    def test_fdu_symbols_and_count(self) -> None:
        """fdu_symbols order matches framework precedence."""
        scheme = DimSchema(_fwk="PHYSICAL")
        assert scheme.fdu_symbols == self.data["PHYSICAL_SYMBOLS"]
        assert scheme.size == len(self.data["PHYSICAL_SYMBOLS"])

    def test_add_and_remove_fdu_custom(self) -> None:
        """add_fdu() and remove_fdu() on CUSTOM framework."""
        custom = self.data["CUSTOM_FDU_LIST"]
        scheme = DimSchema(_fwk="CUSTOM", _fdu_lt=custom)
        n0 = scheme.size

        new_dim = Dimension(
            _idx=n0,
            _sym="W",
            _alias="W",
            _fwk="CUSTOM",
            _unit="w",
            name="Warg",
            description="Warg dim"
        )
        scheme.add_fdu(new_dim)
        assert scheme.size == n0 + 1
        assert scheme.has_fdu("W")

        ans = scheme.remove_fdu("W")
        assert ans is new_dim
        assert scheme.size == n0
        assert scheme.has_fdu("W") is False

    def test_add_fdu_duplicate_and_mismatch(self) -> None:
        """add_fdu() raises on duplicate symbol and framework mismatch."""
        scheme = DimSchema(_fwk="PHYSICAL")
        dup = Dimension(_idx=99, _sym="L", _alias="L", _fwk="PHYSICAL", _unit="m", name="L", description="dup")
        with pytest.raises(ValueError):
            scheme.add_fdu(dup)

        mismatch = Dimension(_idx=0, _sym="Z", _alias="Z", _fwk="COMPUTATION", _unit="bit", name="Z", description="mm")
        with pytest.raises(ValueError):
            scheme.add_fdu(mismatch)

    def test_remove_fdu_invalid(self) -> None:
        """remove_fdu() invalid symbol raises."""
        scheme = DimSchema(_fwk="PHYSICAL")
        with pytest.raises(ValueError):
            scheme.remove_fdu("ZZZ")

    def test_regex_property_setters(self) -> None:
        """Regex property setters accept non-empty strings and reject empty."""
        scheme = DimSchema(_fwk="PHYSICAL")

        new_main = r"^[LMT](\^-?\d+)?(\*[LMT](?:\^-?\d+)?)*$"
        scheme.fdu_regex = new_main
        assert scheme.fdu_regex == new_main
        with pytest.raises(ValueError):
            scheme.fdu_regex = ""

        new_pow = r"\^(-?\d+)"
        scheme.fdu_pow_regex = new_pow
        assert scheme.fdu_pow_regex == new_pow
        with pytest.raises(ValueError):
            scheme.fdu_pow_regex = ""

        new_no = r"[LMT](?!\^)"
        scheme.fdu_no_pow_regex = new_no
        assert scheme.fdu_no_pow_regex == new_no
        with pytest.raises(ValueError):
            scheme.fdu_no_pow_regex = ""

        new_sym = r"[LMTKINC]"
        scheme.fdu_sym_regex = new_sym
        assert scheme.fdu_sym_regex == new_sym
        with pytest.raises(ValueError):
            scheme.fdu_sym_regex = ""

    def test_fdu_lt_setter_validation(self) -> None:
        """fdu_lt setter validates type and content."""
        custom = self.data["CUSTOM_FDU_LIST"]
        scheme = DimSchema(_fwk="CUSTOM", _fdu_lt=custom)

        # valid: set as list[Dimension]
        new_dims = [
            Dimension(_idx=0, _sym="A", _alias="A", _fwk="CUSTOM", _unit="x", name="A", description="A"),
            Dimension(_idx=1, _sym="B", _alias="B", _fwk="CUSTOM", _unit="y", name="B", description="B"),
        ]
        scheme.fdu_lt = new_dims
        assert scheme.fdu_lt == new_dims

        # invalid: not a list of Dimensions
        with pytest.raises(ValueError):
            scheme.fdu_lt = []  # empty
        with pytest.raises(ValueError):
            scheme.fdu_lt = ["not", "dimensions"]  # type: ignore[assignment]

    def test_update_global_config(self) -> None:
        """update_global_config populates global regex and symbol config."""
        scheme = DimSchema(_fwk="COMPUTATION")
        scheme.update_global_config()
        # Symbols
        assert cfg.WKNG_FDU_PREC_LT == self.data["COMPUTATION_SYMBOLS"]
        # Regex strings set
        assert isinstance(cfg.WKNG_FDU_RE, str) and cfg.WKNG_FDU_RE != ""
        assert isinstance(cfg.WKNG_POW_RE, str) and cfg.WKNG_POW_RE != ""
        assert isinstance(cfg.WKNG_NO_POW_RE, str) and cfg.WKNG_NO_POW_RE != ""
        assert isinstance(cfg.WKNG_FDU_SYM_RE, str) and cfg.WKNG_FDU_SYM_RE != ""

    def test_reset(self) -> None:
        """reset clears internal state."""
        scheme = DimSchema(_fwk="PHYSICAL")
        scheme.reset()
        assert scheme.size == 0
        assert scheme.fdu_symbols == []
        assert scheme.fdu_regex == ""
        assert scheme.fdu_no_pow_regex == ""
        assert scheme.fdu_sym_regex == ""

    def test_to_dict(self) -> None:
        """to_dict() returns expected dictionary representation."""
        scheme = DimSchema(_fwk="PHYSICAL")
        dct = scheme.to_dict()
        assert isinstance(dct, dict)
        assert dct["fwk"] == "PHYSICAL"
        assert isinstance(dct["fdu_lt"], list)
        assert len(dct["fdu_lt"]) == scheme.size
        for item in dct["fdu_lt"]:
            assert isinstance(item, dict)
            assert "sym" in item
            assert "unit" in item
            assert "name" in item
            assert "description" in item

    def test_from_dict(self) -> None:
        """from_dict() creates DimSchema from dictionary."""
        scheme = DimSchema(_fwk="SOFTWARE")
        dct = scheme.to_dict()
        new_scheme = DimSchema.from_dict(dct)
        assert isinstance(new_scheme, DimSchema)
        assert new_scheme.fwk == "SOFTWARE"
        assert new_scheme.size == scheme.size
        for sym in scheme.fdu_symbols:
            assert new_scheme.has_fdu(sym)
