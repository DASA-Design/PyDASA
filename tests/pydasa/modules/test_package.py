"""
Test suite for PyDASA package initialization and configuration.

Tests verify that the package can be imported correctly, version information
is accessible, and required configuration files are properly included.
"""
import pytest
from pathlib import Path


def test_pydasa_import():
    """Test that pydasa can be imported without errors."""
    import pydasa
    assert pydasa is not None


def test_version_accessible():
    """Test that __version__ is accessible from pydasa module."""
    import pydasa
    assert hasattr(pydasa, '__version__')
    assert isinstance(pydasa.__version__, str)
    assert len(pydasa.__version__.split('.')) == 3  # MAJOR.MINOR.PATCH format


def test_version_format():
    """Test that version follows semantic versioning format."""
    import pydasa
    version_parts = pydasa.__version__.split('.')
    assert len(version_parts) == 3
    for part in version_parts:
        assert part.isdigit(), f"Version part '{part}' is not a number"


def test_config_file_exists():
    """Test that the default configuration file is included in the package."""
    import pydasa.core.setup
    from pydasa.core.constants import DFLT_CFG_FOLDER, DFLT_CFG_FILE

    # Get the module directory
    module_dir = Path(pydasa.core.setup.__file__).parent
    config_file = module_dir / DFLT_CFG_FOLDER / DFLT_CFG_FILE

    assert config_file.exists(), f"Configuration file not found: {config_file}"
    assert config_file.is_file(), f"Configuration path is not a file: {config_file}"


def test_config_loads_successfully():
    """Test that the configuration can be loaded without errors."""
    from pydasa.core.setup import PYDASA_CFG

    assert PYDASA_CFG is not None
    assert hasattr(PYDASA_CFG, 'SPT_FDU_FWKS')


def test_main_exports():
    """Test that main package exports are accessible."""
    import pydasa

    # Check analytics modules
    assert hasattr(pydasa, 'Sensitivity')
    assert hasattr(pydasa, 'MonteCarlo')

    # Check dimensional analysis modules
    assert hasattr(pydasa, 'Coefficient')
    assert hasattr(pydasa, 'Dimension')
    assert hasattr(pydasa, 'Schema')
    assert hasattr(pydasa, 'Matrix')

    # Check core modules
    assert hasattr(pydasa, 'Variable')
    assert hasattr(pydasa, 'load')
    assert hasattr(pydasa, 'save')

    # Check workflow modules
    assert hasattr(pydasa, 'SensitivityAnalysis')
    assert hasattr(pydasa, 'MonteCarloSimulation')


def test_all_exports():
    """Test that __all__ list is properly defined."""
    import pydasa

    assert hasattr(pydasa, '__all__')
    assert isinstance(pydasa.__all__, list)
    assert '__version__' in pydasa.__all__
    assert len(pydasa.__all__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
