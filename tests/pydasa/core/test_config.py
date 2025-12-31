# -*- coding: utf-8 -*-
"""
Test Module for core/config.py
===========================================

Tests for configuration Enums, singleton, and backward compatibility dicts in PyDASA.

Note:
    - Dimensional constants (PHY_FDU_PREC_DT, etc.) are tested in tests/dimensional/test_constants.py
    - Regex patterns (LATEX_RE, DFLT_FDU_RE, etc.) are tested in tests/utils/test_patterns.py
"""

import unittest
import pytest
from pydasa.core import config
from pydasa.dimensional.framework import DimSchema
from tests.pydasa.data.test_data import get_config_test_data


# Test data fixture
# @pytest.fixture(scope="module")
class TestConfig(unittest.TestCase):
    """Test cases for config module."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_config_test_data()
        # resetting config due to singleton pattern fix later
        self.test_scheme = DimSchema(_fwk="PHYSICAL")
        self.test_scheme.update_global_config()

    # =============================================================================
    # Enum Tests - NEW functionality
    # =============================================================================
    
    def test_framework_enum_exists(self) -> None:
        """Test Framework enum exists and is accessible."""
        assert hasattr(config, 'Framework')
        assert hasattr(config.Framework, 'PHYSICAL')
        assert hasattr(config.Framework, 'COMPUTATION')
        assert hasattr(config.Framework, 'SOFTWARE')
        assert hasattr(config.Framework, 'CUSTOM')
    
    def test_framework_enum_members(self) -> None:
        """Test Framework enum has all expected members with correct values."""
        assert config.Framework.PHYSICAL.value == "PHYSICAL"
        assert config.Framework.COMPUTATION.value == "COMPUTATION"
        assert config.Framework.SOFTWARE.value == "SOFTWARE"
        assert config.Framework.CUSTOM.value == "CUSTOM"

    def test_framework_descriptions(self) -> None:
        """Test Framework enum descriptions are accessible and valid."""
        assert "Traditional physical" in config.Framework.PHYSICAL.description
        assert "Computer science" in config.Framework.COMPUTATION.description
        assert "Software architecture" in config.Framework.SOFTWARE.description
        assert "User-defined" in config.Framework.CUSTOM.description

    def test_parameter_category_enum_exists(self) -> None:
        """Test VarCardinality enum exists and is accessible."""
        assert hasattr(config, 'VarCardinality')
        assert hasattr(config.VarCardinality, 'IN')
        assert hasattr(config.VarCardinality, 'OUT')
        assert hasattr(config.VarCardinality, 'CTRL')

    def test_parameter_category_enum(self) -> None:
        """Test VarCardinality enum members with correct values."""
        assert config.VarCardinality.IN.value == "IN"
        assert config.VarCardinality.OUT.value == "OUT"
        assert config.VarCardinality.CTRL.value == "CTRL"
        assert "influence the system" in config.VarCardinality.IN.description

    def test_coefficient_category_enum_exists(self) -> None:
        """Test CoefCardinality enum exists and is accessible."""
        assert hasattr(config, 'CoefCardinality')
        assert hasattr(config.CoefCardinality, 'COMPUTED')
        assert hasattr(config.CoefCardinality, 'DERIVED')

    def test_coefficient_category_enum(self) -> None:
        """Test CoefCardinality enum members with correct values."""
        assert config.CoefCardinality.COMPUTED.value == "COMPUTED"
        assert config.CoefCardinality.DERIVED.value == "DERIVED"
        assert "Dimensional Matrix" in config.CoefCardinality.COMPUTED.description

    def test_sensitivity_mode_enum_exists(self) -> None:
        """Test AnaliticMode enum exists and is accessible."""
        assert hasattr(config, 'AnaliticMode')
        assert hasattr(config.AnaliticMode, 'SYM')
        assert hasattr(config.AnaliticMode, 'NUM')

    def test_sensitivity_mode_enum(self) -> None:
        """Test AnaliticMode enum members with correct values."""
        assert config.AnaliticMode.SYM.value == "SYM"
        assert config.AnaliticMode.NUM.value == "NUM"
        assert "symbolic" in config.AnaliticMode.SYM.description

    # =============================================================================
    # Singleton Tests - NEW functionality
    # =============================================================================

    def test_config_singleton_exists(self) -> None:
        """Test PyDASAConfig singleton class exists."""
        assert hasattr(config, 'PyDASAConfig')
        assert hasattr(config.PyDASAConfig, 'get_instance')

    def test_config_singleton_pattern(self) -> None:
        """Test PyDASAConfig singleton returns same instance."""
        cfg1 = config.PyDASAConfig.get_instance()
        cfg2 = config.PyDASAConfig.get_instance()
        assert cfg1 is cfg2  # Same instance
        assert id(cfg1) == id(cfg2)  # Same memory address

    def test_config_singleton_properties(self) -> None:
        """Test PyDASAConfig singleton provides access to all enums."""
        cfg = config.PyDASAConfig.get_instance()
        assert hasattr(cfg, 'frameworks')
        assert len(cfg.frameworks) == 4

    def test_config_singleton_immutable(self) -> None:
        """Test PyDASAConfig singleton is immutable (frozen dataclass)."""
        cfg = config.PyDASAConfig.get_instance()
        # Should raise FrozenInstanceError or AttributeError
        with pytest.raises(Exception):
            cfg.new_attribute = "test"  # type: ignore

    # =============================================================================
    # Backward Compatibility Dict Tests
    # =============================================================================

    # Framework Dictionaries Tests
    def test_fdu_fwk_exists(self) -> None:
        """Test that FDU_FWK_DT exists and is a dictionary."""
        assert hasattr(config, 'FDU_FWK_DT')
        assert isinstance(config.FDU_FWK_DT, dict)

    def test_fdu_fwk_keys(self) -> None:
        """Test that FDU_FWK_DT has expected frameworks."""
        assert set(config.FDU_FWK_DT.keys()) == set(self.test_data["FRAMEWORK_KEYS"])

    def test_params_cat_exists(self) -> None:
        """Test that PARAMS_CAT_DT exists and has expected keys."""
        assert hasattr(config, 'PARAMS_CAT_DT')
        assert isinstance(config.PARAMS_CAT_DT, dict)
        assert set(config.PARAMS_CAT_DT.keys()) == set(self.test_data["PARAMS_CAT_KEYS"])

    def test_dc_cat_exists(self) -> None:
        """Test that DC_CAT_DT exists and has expected keys."""
        assert hasattr(config, 'DC_CAT_DT')
        assert isinstance(config.DC_CAT_DT, dict)
        assert set(config.DC_CAT_DT.keys()) == set(self.test_data["DC_CAT_KEYS"])

    def test_sens_ansys_exists(self) -> None:
        """Test that SENS_ANSYS_DT exists and has expected keys."""
        assert hasattr(config, 'SENS_ANSYS_DT')
        assert isinstance(config.SENS_ANSYS_DT, dict)
        assert set(config.SENS_ANSYS_DT.keys()) == set(self.test_data["SENS_ANSYS_KEYS"])

    def test_backward_compatibility_dicts(self) -> None:
        """Test that legacy dict exports still work and match Enum descriptions."""
        # All dicts should still exist and be accessible
        assert isinstance(config.FDU_FWK_DT, dict)
        assert isinstance(config.PARAMS_CAT_DT, dict)
        assert isinstance(config.DC_CAT_DT, dict)
        assert isinstance(config.SENS_ANSYS_DT, dict)

        # Dict values should match Enum descriptions
        assert config.FDU_FWK_DT["PHYSICAL"] == config.Framework.PHYSICAL.description
        assert config.PARAMS_CAT_DT["IN"] == config.VarCardinality.IN.description
        assert config.DC_CAT_DT["COMPUTED"] == config.CoefCardinality.COMPUTED.description
        assert config.SENS_ANSYS_DT["SYM"] == config.AnaliticMode.SYM.description

    def test_backward_compatibility_dict_keys(self) -> None:
        """Test that legacy dicts have all expected keys."""
        # Framework keys
        assert set(config.FDU_FWK_DT.keys()) == {"PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"}

        # Parameter category keys
        assert set(config.PARAMS_CAT_DT.keys()) == {"IN", "OUT", "CTRL"}

        # Coefficient category keys
        assert set(config.DC_CAT_DT.keys()) == {"COMPUTED", "DERIVED"}
        # Sensitivity mode keys
        assert set(config.SENS_ANSYS_DT.keys()) == {"SYM", "NUM"}
