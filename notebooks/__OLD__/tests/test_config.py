# # -*- coding: utf-8 -*-
# """
# Test Module for core/config.py
# ===========================================

# Tests for configuration Enums, singleton, and backward compatibility dicts in PyDASA.

# Note:
#     - Dimensional constants (PHY_FDU_PREC_DT, etc.) are tested in tests/dimensional/test_constants.py
#     - Regex patterns (LATEX_RE, DFLT_FDU_RE, etc.) are tested in tests/utils/test_patterns.py
# """

# import unittest
# import pytest
# from pydasa.core import setup
# from pydasa.dimensional.framework import Schema
# from tests.pydasa.data.test_data import get_config_test_data


# # Test data fixture
# # @pytest.fixture(scope="module")
# class TestConfig(unittest.TestCase):
#     """Test cases for config module."""

#     @pytest.fixture(autouse=True)
#     def inject_fixtures(self):
#         """Inject test data fixture."""
#         self.test_data = get_config_test_data()
#         # resetting config due to singleton pattern fix later
#         self.test_scheme = Schema(_fwk="PHYSICAL")
#         self.test_scheme.update_global_config()

#     # =============================================================================
#     # Enum Tests - NEW functionality
#     # =============================================================================

#     def test_framework_enum_exists(self) -> None:
#         """Test Framework enum exists and is accessible."""
#         assert hasattr(setup, 'Framework')
#         assert hasattr(setup.Framework, 'PHYSICAL')
#         assert hasattr(setup.Framework, 'COMPUTATION')
#         assert hasattr(setup.Framework, 'SOFTWARE')
#         assert hasattr(setup.Framework, 'CUSTOM')

#     def test_framework_enum_members(self) -> None:
#         """Test Framework enum has all expected members with correct values."""
#         assert setup.Framework.PHYSICAL.value == "PHYSICAL"
#         assert setup.Framework.COMPUTATION.value == "COMPUTATION"
#         assert setup.Framework.SOFTWARE.value == "SOFTWARE"
#         assert setup.Framework.CUSTOM.value == "CUSTOM"

#     def test_framework_descriptions(self) -> None:
#         """Test Framework enum descriptions are accessible and valid."""
#         assert "Traditional physical" in setup.Framework.PHYSICAL.description
#         assert "Computer science" in setup.Framework.COMPUTATION.description
#         assert "Software architecture" in setup.Framework.SOFTWARE.description
#         assert "User-defined" in setup.Framework.CUSTOM.description

#     def test_parameter_category_enum_exists(self) -> None:
#         """Test VarCardinality enum exists and is accessible."""
#         assert hasattr(setup, 'VarCardinality')
#         assert hasattr(setup.VarCardinality, 'IN')
#         assert hasattr(setup.VarCardinality, 'OUT')
#         assert hasattr(setup.VarCardinality, 'CTRL')

#     def test_parameter_category_enum(self) -> None:
#         """Test VarCardinality enum members with correct values."""
#         assert setup.VarCardinality.IN.value == "IN"
#         assert setup.VarCardinality.OUT.value == "OUT"
#         assert setup.VarCardinality.CTRL.value == "CTRL"
#         assert "influence the system" in setup.VarCardinality.IN.description

#     def test_coefficient_category_enum_exists(self) -> None:
#         """Test CoefCardinality enum exists and is accessible."""
#         assert hasattr(setup, 'CoefCardinality')
#         assert hasattr(setup.CoefCardinality, 'COMPUTED')
#         assert hasattr(setup.CoefCardinality, 'DERIVED')

#     def test_coefficient_category_enum(self) -> None:
#         """Test CoefCardinality enum members with correct values."""
#         assert setup.CoefCardinality.COMPUTED.value == "COMPUTED"
#         assert setup.CoefCardinality.DERIVED.value == "DERIVED"
#         assert "Dimensional Matrix" in setup.CoefCardinality.COMPUTED.description

#     def test_sensitivity_mode_enum_exists(self) -> None:
#         """Test AnaliticMode enum exists and is accessible."""
#         assert hasattr(setup, 'AnaliticMode')
#         assert hasattr(setup.AnaliticMode, 'SYM')
#         assert hasattr(setup.AnaliticMode, 'NUM')

#     def test_sensitivity_mode_enum(self) -> None:
#         """Test AnaliticMode enum members with correct values."""
#         assert setup.AnaliticMode.SYM.value == "SYM"
#         assert setup.AnaliticMode.NUM.value == "NUM"
#         assert "symbolic" in setup.AnaliticMode.SYM.description

#     # =============================================================================
#     # Singleton Tests - NEW functionality
#     # =============================================================================

#     def test_config_singleton_exists(self) -> None:
#         """Test PyDASAConfig singleton class exists."""
#         assert hasattr(setup, 'PyDASAConfig')
#         assert hasattr(setup.PyDASAConfig, 'get_instance')

#     def test_config_singleton_pattern(self) -> None:
#         """Test PyDASAConfig singleton returns same instance."""
#         cfg1 = setup.PyDASAConfig.get_instance()
#         cfg2 = setup.PyDASAConfig.get_instance()
#         assert cfg1 is cfg2  # Same instance
#         assert id(cfg1) == id(cfg2)  # Same memory address

#     def test_config_singleton_properties(self) -> None:
#         """Test PyDASAConfig singleton provides access to all enums."""
#         cfg = setup.PyDASAConfig.get_instance()
#         assert hasattr(cfg, 'frameworks')
#         assert len(cfg.frameworks) == 4

#     def test_config_singleton_immutable(self) -> None:
#         """Test PyDASAConfig singleton is immutable (frozen dataclass)."""
#         cfg = setup.PyDASAConfig.get_instance()
#         # Should raise FrozenInstanceError or AttributeError
#         with pytest.raises(Exception):
#             cfg.new_attribute = "test"  # type: ignore

#     # =============================================================================
#     # Backward Compatibility Dict Tests
#     # =============================================================================

#     # Framework Dictionaries Tests
#     def test_fdu_fwk_exists(self) -> None:
#         """Test that FDU_FWK_DT exists and is a dictionary."""
#         assert hasattr(setup, 'FDU_FWK_DT')
#         assert isinstance(setup.FDU_FWK_DT, dict)

#     def test_fdu_fwk_keys(self) -> None:
#         """Test that FDU_FWK_DT has expected frameworks."""
#         assert set(setup.FDU_FWK_DT.keys()) == set(self.test_data["FRAMEWORK_KEYS"])

#     def test_params_cat_exists(self) -> None:
#         """Test that PARAMS_CAT_DT exists and has expected keys."""
#         assert hasattr(setup, 'PARAMS_CAT_DT')
#         assert isinstance(setup.PARAMS_CAT_DT, dict)
#         assert set(setup.PARAMS_CAT_DT.keys()) == set(self.test_data["PARAMS_CAT_KEYS"])

#     def test_dc_cat_exists(self) -> None:
#         """Test that DC_CAT_DT exists and has expected keys."""
#         assert hasattr(setup, 'DC_CAT_DT')
#         assert isinstance(setup.DC_CAT_DT, dict)
#         assert set(setup.DC_CAT_DT.keys()) == set(self.test_data["DC_CAT_KEYS"])

#     def test_sens_ansys_exists(self) -> None:
#         """Test that SENS_ANSYS_DT exists and has expected keys."""
#         assert hasattr(setup, 'SENS_ANSYS_DT')
#         assert isinstance(setup.SENS_ANSYS_DT, dict)
#         assert set(setup.SENS_ANSYS_DT.keys()) == set(self.test_data["SENS_ANSYS_KEYS"])

#     def test_backward_compatibility_dicts(self) -> None:
#         """Test that legacy dict exports still work and match Enum descriptions."""
#         # All dicts should still exist and be accessible
#         assert isinstance(setup.FDU_FWK_DT, dict)
#         assert isinstance(setup.PARAMS_CAT_DT, dict)
#         assert isinstance(setup.DC_CAT_DT, dict)
#         assert isinstance(setup.SENS_ANSYS_DT, dict)

#         # Dict values should match Enum descriptions
#         assert setup.FDU_FWK_DT["PHYSICAL"] == setup.Framework.PHYSICAL.description
#         assert setup.PARAMS_CAT_DT["IN"] == setup.VarCardinality.IN.description
#         assert setup.DC_CAT_DT["COMPUTED"] == setup.CoefCardinality.COMPUTED.description
#         assert setup.SENS_ANSYS_DT["SYM"] == setup.AnaliticMode.SYM.description

#     def test_backward_compatibility_dict_keys(self) -> None:
#         """Test that legacy dicts have all expected keys."""
#         # Framework keys
#         assert set(setup.FDU_FWK_DT.keys()) == {"PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"}

#         # Parameter category keys
#         assert set(setup.PARAMS_CAT_DT.keys()) == {"IN", "OUT", "CTRL"}

#         # Coefficient category keys
#         assert set(setup.DC_CAT_DT.keys()) == {"COMPUTED", "DERIVED"}
#         # Sensitivity mode keys
#         assert set(setup.SENS_ANSYS_DT.keys()) == {"SYM", "NUM"}
