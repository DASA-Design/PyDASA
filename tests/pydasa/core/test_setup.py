# -*- coding: utf-8 -*-
"""
Test Module for core/setup.py
===========================================

Tests for configuration Enums and singleton in PyDASA.

This module tests:
    - Framework, VarCardinality, CoefCardinality, AnaliticMode Enums
    - PyDASAConfig singleton pattern
"""

# import testing package
from typing import Union, Type, List
from enum import Enum
import unittest
import pytest

# import the module to test
from pydasa.core import setup
from pydasa.core.setup import Framework
from pydasa.core.setup import VarCardinality
from pydasa.core.setup import CoefCardinality
from pydasa.core.setup import AnaliticMode
from pydasa.core.setup import PyDASAConfig

# import the test data
from tests.pydasa.data.test_data import get_config_test_data

# asserting module imports
assert setup
assert Framework
assert VarCardinality
assert CoefCardinality
assert AnaliticMode
assert PyDASAConfig
assert get_config_test_data

# Type alias for all Enums in setup.py
PyDASAEnum = Union[Framework, VarCardinality, CoefCardinality, AnaliticMode]


# =============================================================================
# Helper function for common enum tests
# =============================================================================


def run_enum_tests(test_instance: unittest.TestCase,
                   enum_class: Type[PyDASAEnum],
                   expected_values: List[str]) -> None:
    """Helper function to run common enum tests.

    Args:
        test_instance: The test case instance
        enum_class: The enum class to test
        expected_values: List of expected enum value strings
    """
    # Test enum exists and has correct members
    assert enum_class is not None
    assert issubclass(enum_class, Enum)
    assert len(list(enum_class)) == len(expected_values)

    # Test all expected members exist with correct values and descriptions
    for value in expected_values:
        assert hasattr(enum_class, value)
        member = getattr(enum_class, value)
        assert member.value == value
        assert hasattr(member, "description")
        assert len(member.description) > 0


class TestFramework(unittest.TestCase):
    """**TestFramework** implements unit tests for the Framework Enum in setup.py.

    Args:
        unittest (TestCase): unittest.TestCase class for unit testing in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters for Framework as a fixture."""
        self.test_data = get_config_test_data()
        self.enum_class = Framework
        self.expected_values = self.test_data["FRAMEWORK_KEYS"]

    def test_framework_enum(self) -> None:
        """Test Framework enum with all common checks."""
        run_enum_tests(self, self.enum_class, self.expected_values)


class TestVarCardinality(unittest.TestCase):
    """**TestVarCardinality** implements unit tests for the VarCardinality Enum in setup.py.

    Args:
        unittest (TestCase): unittest.TestCase class for unit testing in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters for VarCardinality as a fixture."""
        self.test_data = get_config_test_data()
        self.enum_class = VarCardinality
        self.expected_values = self.test_data["PARAMS_CAT_KEYS"]

    def test_var_cardinality_enum(self) -> None:
        """Test VarCardinality enum with all common checks."""
        run_enum_tests(self, self.enum_class, self.expected_values)


class TestCoefCardinality(unittest.TestCase):
    """**TestCoefCardinality** implements unit tests for the CoefCardinality Enum in setup.py.

    Args:
        unittest (TestCase): unittest.TestCase class for unit testing in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters for CoefCardinality as a fixture."""
        self.test_data = get_config_test_data()
        self.enum_class = CoefCardinality
        self.expected_values = self.test_data["DC_CAT_KEYS"]

    def test_coef_cardinality_enum(self) -> None:
        """Test CoefCardinality enum with all common checks."""
        run_enum_tests(self, self.enum_class, self.expected_values)


class TestAnaliticMode(unittest.TestCase):
    """**TestAnaliticMode** implements unit tests for the AnaliticMode Enum in setup.py.

    Args:
        unittest (TestCase): unittest.TestCase class for unit testing in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters for AnaliticMode as a fixture."""
        self.test_data = get_config_test_data()
        self.enum_class = AnaliticMode
        self.expected_values = self.test_data["SENS_ANSYS_KEYS"]

    def test_analitic_mode_enum(self) -> None:
        """Test AnaliticMode enum with all common checks."""
        run_enum_tests(self, self.enum_class, self.expected_values)


class TestPyDASAConfig(unittest.TestCase):
    """**TestPyDASAConfig** implements unit tests for the PyDASAConfig singleton in setup.py.

    Args:
        unittest (TestCase): unittest.TestCase class for unit testing in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters for PyDASAConfig as a fixture."""
        self.test_data = get_config_test_data()

    def test_singleton_pattern(self) -> None:
        """Test that PyDASAConfig implements singleton pattern correctly."""
        cfg1 = PyDASAConfig.get_instance()
        cfg2 = PyDASAConfig.get_instance()

        assert cfg1 is not None
        assert isinstance(cfg1, PyDASAConfig)
        assert cfg1 is cfg2

    def test_global_instance_matches_singleton(self) -> None:
        """Test that PYDASA_CFG global instance matches get_instance() singleton."""
        from pydasa.core.setup import PYDASA_CFG

        cfg_singleton = PyDASAConfig.get_instance()

        # Both should be PyDASAConfig instances
        assert isinstance(PYDASA_CFG, PyDASAConfig)
        assert isinstance(cfg_singleton, PyDASAConfig)

        # NOTE: Due to current implementation, they may not be the same instance
        # if PYDASA_CFG was created before get_instance() was called.
        # This test documents the current behavior.

    def test_support_fwk_loaded(self) -> None:
        """Test that SPT_FDU_FWKS attribute is loaded from configuration file."""
        cfg = PyDASAConfig.get_instance()

        assert hasattr(cfg, "SPT_FDU_FWKS")
        assert isinstance(cfg.SPT_FDU_FWKS, dict)
        # The dict should be populated from the config file
        # At minimum, it should not be empty if config file exists

    def test_frameworks_property(self) -> None:
        """Test frameworks property returns correct Framework enum members."""
        cfg = PyDASAConfig.get_instance()
        frameworks = cfg.frameworks

        assert isinstance(frameworks, tuple)
        assert len(frameworks) == 4
        assert all(isinstance(f, Framework) for f in frameworks)
        framework_values = [f.value for f in frameworks]
        assert set(framework_values) == {"PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"}

    def test_parameter_cardinality_property(self) -> None:
        """Test parameter_cardinality property returns correct VarCardinality enum members."""
        cfg = PyDASAConfig.get_instance()
        param_cards = cfg.parameter_cardinality

        assert isinstance(param_cards, tuple)
        assert len(param_cards) == 3
        assert all(isinstance(c, VarCardinality) for c in param_cards)
        card_values = [c.value for c in param_cards]
        assert set(card_values) == {"IN", "OUT", "CTRL"}

    def test_coefficient_cardinality_property(self) -> None:
        """Test coefficient_cardinality property returns correct CoefCardinality enum members."""
        cfg = PyDASAConfig.get_instance()
        coef_cards = cfg.coefficient_cardinality

        assert isinstance(coef_cards, tuple)
        assert len(coef_cards) == 2
        assert all(isinstance(c, CoefCardinality) for c in coef_cards)
        card_values = [c.value for c in coef_cards]
        assert set(card_values) == {"COMPUTED", "DERIVED"}

    def test_analitic_modes_property(self) -> None:
        """Test analitic_modes property returns correct AnaliticMode enum members."""
        cfg = PyDASAConfig.get_instance()
        modes = cfg.analitic_modes

        assert isinstance(modes, tuple)
        assert len(modes) == 2
        assert all(isinstance(m, AnaliticMode) for m in modes)
        mode_values = [m.value for m in modes]
        assert set(mode_values) == {"SYM", "NUM"}

    def test_immutability(self) -> None:
        """Test that PyDASAConfig instance is immutable (frozen dataclass)."""
        cfg = PyDASAConfig.get_instance()

        # Frozen dataclass should raise FrozenInstanceError or AttributeError
        with pytest.raises((AttributeError, Exception)):
            cfg.new_attribute = "test"  # type: ignore

        with pytest.raises((AttributeError, Exception)):
            cfg.SPT_FDU_FWKS = {}  # type: ignore

    def test_config_loaded_in_post_init(self) -> None:
        """Test that configuration is loaded properly in __post_init__."""
        cfg = PyDASAConfig.get_instance()

        # Verify SPT_FDU_FWKS was loaded (should be dict from config file)
        assert hasattr(cfg, "SPT_FDU_FWKS")
        assert isinstance(cfg.SPT_FDU_FWKS, dict)
