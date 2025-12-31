# -*- coding: utf-8 -*-
"""
Core Module
===========

Core domain classes and configuration for PyDASA.

This module contains fundamental domain entities and system configuration.
"""

# TODO remove unnecesary code after refactor
from pydasa.core import config
from pydasa.core.config import (
    # Type-safe Enums (NEW - preferred)
    Framework,
    VarCardinality,
    CoefCardinality,
    AnaliticMode,
    PyDASAConfig,
    # Backward compatibility dict exports (DEPRECATED)
    FDU_FWK_DT,
    PARAMS_CAT_DT,
    DC_CAT_DT,
    SENS_ANSYS_DT,
)

__all__ = [
    "config",
    # Enums
    "Framework",
    "VarCardinality",
    "CoefCardinality",
    "AnaliticMode",
    "PyDASAConfig",
    # Legacy dicts
    "FDU_FWK_DT",
    "PARAMS_CAT_DT",
    "DC_CAT_DT",
    "SENS_ANSYS_DT",
]
