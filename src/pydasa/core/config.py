# -*- coding: utf-8 -*-
"""
Module config.py
===========================================

Configuration module for Fundamental Dimensional Units (FDUs), Parameters, and Variables in *PyDASA*.

This module provides type-safe configuration through Enums and frozen dataclasses,
replacing the previous mutable dictionary-based approach with immutable, type-checked alternatives.

Key Features:
    - Type-safe Enum definitions for frameworks, categories, and modes
    - Immutable configuration via frozen dataclass with singleton pattern
    - Backward compatibility with legacy dict-based access

    Supported Frameworks:
        - PHYSICAL: Traditional physical dimensional framework.
        - COMPUTATION: Computer science dimensional framework.
        - SOFTWARE: Software architecture dimensional framework.
        - CUSTOM: User-defined dimensional framework.

    Supported Variable Categories:
        - IN: Input variables influencing the system.
        - OUT: Output variables representing analysis results.
        - CTRL: Control variables constraining the system.

    Supported Analysis Modes:
        - SYM: Analysis for symbolic processable Parameters (e.g., 'x + y').
        - NUM: Analysis for numeric Variables (e.g., 1.0, 2.5).

*IMPORTANT* Based on:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# python native modules
from enum import Enum
from dataclasses import dataclass
from typing import ClassVar

# import global constants (kept for backward compatibility re-exports)
from pydasa.dimensional.constants import PHY_FDU_PREC_DT  # noqa: F401
from pydasa.dimensional.constants import COMPU_FDU_PREC_DT  # noqa: F401
from pydasa.dimensional.constants import SOFT_FDU_PREC_DT  # noqa: F401
from pydasa.dimensional.constants import DFLT_FDU_PREC_LT  # noqa: F401

# import regex patterns (kept for backward compatibility re-exports)
from pydasa.utils.patterns import (  # noqa: F401
    LATEX_RE,
    DFLT_FDU_RE,
    DFLT_POW_RE,
    DFLT_NO_POW_RE,
    DFLT_FDU_SYM_RE,
    WKNG_FDU_PREC_LT,
    WKNG_FDU_RE,
    WKNG_POW_RE,
    WKNG_NO_POW_RE,
    WKNG_FDU_SYM_RE,
)


# =============================================================================
# Type-Safe Configuration Enums
# =============================================================================


class Framework(str, Enum):
    """**Framework** Enum for Fundamental Dimensional Units (FDUs) frameworks.

    Args:
        str (class): Python native str class.
        Enum (class): Python native Enum class.

    Returns:
        Framework: Enum member representing the FDU framework.
    """

    PHYSICAL = "PHYSICAL"
    COMPUTATION = "COMPUTATION"
    SOFTWARE = "SOFTWARE"
    CUSTOM = "CUSTOM"

    @property
    def description(self) -> str:
        """*description* Get human-readable description of the framework.

        Returns:
            str: Human-readable framework's description.
        """

        descriptions = {
            Framework.PHYSICAL: "Traditional physical dimensional framework (e.g., Length, Mass, Time).",
            Framework.COMPUTATION: "Computer science dimensional framework (e.g., Time, Space, Complexity).",
            Framework.SOFTWARE: "Software architecture dimensional framework (e.g., Time, Data, Connectivity).",
            Framework.CUSTOM: "User-defined dimensional framework for specific use cases.",
        }
        return descriptions[self]


class VarCardinality(str, Enum):
    """**VarCardinality** Enum for Variable cardinality.

    Args:
        str (class): Python native str class.
        Enum (class): Python native Enum class.

    Returns:
        VarCardinality: Enum member representing the variable cardinality.
    """
    IN = "IN"
    OUT = "OUT"
    CTRL = "CTRL"

    @property
    def description(self) -> str:
        """*description* Get human-readable description of the variable cardinality.

        Returns:
            str: Human-readable variable cardinality description.
        """
        descriptions = {
            VarCardinality.IN: "Variables that influence the system (e.g., known inputs).",
            VarCardinality.OUT: "Variable that represent the results of the analysis.",
            VarCardinality.CTRL: "Variables used to control or constrain the system (e.g., constants).",
        }
        return descriptions[self]


class CoefCardinality(str, Enum):
    """**CoefCardinality** Enum for Dimensionless Coefficient cardinality.

    Args:
        str (class): Python native str class.
        Enum (class): Python native Enum class.

    Returns:
        CoefCardinality: Enum member representing the coefficient cardinality.
    """
    COMPUTED = "COMPUTED"
    DERIVED = "DERIVED"

    @property
    def description(self) -> str:
        """*description* Get human-readable description of the coefficient cardinality.

        Returns:
            str: Human-readable coefficient cardinality description.
        """
        descriptions = {
            CoefCardinality.COMPUTED: "Coefficients directly calculated using the Dimensional Matrix.",
            CoefCardinality.DERIVED: "Coefficients obtained by combining or manipulating Computed Coefficients.",
        }
        return descriptions[self]


class AnaliticMode(str, Enum):
    """**AnaliticMode** Enum for analysis modes (e.g. sensitivity analysis, Monte Carlo simulation).

    Args:
        str (class): Python native str class.
        Enum (class): python native Enum class.

    Returns:
        AnaliticMode: Enum member representing the analysis mode.
    """
    SYM = "SYM"
    NUM = "NUM"

    @property
    def description(self) -> str:
        descriptions = {
            AnaliticMode.SYM: "Analysis for symbolic processable parameters (e.g., 'z = x + y').",
            AnaliticMode.NUM: "analysis for numeric variable ranges (e.g., 1.0, 2.5).",
        }
        return descriptions[self]


# =============================================================================
# Immutable Configuration Singleton
# =============================================================================


@dataclass(frozen=True)
class PyDASAConfig:
    """ **PyDASAConfig** Singleton class for PyDASA configuration. It uses dataclass decorator to freeze the data.

    Returns:
        PyDASAConfig: frozen singleton configuration instance.
    """

    # :attr: _instance
    _instance: ClassVar['PyDASAConfig | None'] = None
    """Singleton instance of PyDASAConfig."""

    @classmethod
    def get_instance(cls) -> 'PyDASAConfig':
        """*get_instance()* Get the singleton instance of PyDASAConfig.

        Returns:
            PyDASAConfig: Singleton instance of PyDASAConfig.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def frameworks(self) -> tuple[Framework, ...]:
        """*frameworks* Get supported frameworks.

        Returns:
            tuple[Framework, ...]: Tuple of supported Frameworks.
        """
        return tuple(Framework)

    @property
    def parameter_cardinality(self) -> tuple[VarCardinality, ...]:
        """*parameter_cardinality* Get supported variable cardinalities.

        Returns:
            tuple[VarCardinality, ...]: Tuple of supported VarCardinality.
        """
        return tuple(VarCardinality)

    @property
    def coefficient_cardinality(self) -> tuple[CoefCardinality, ...]:
        """*coefficient_cardinality* Get supported coefficient cardinalities.

        Returns:
            tuple[CoefCardinality, ...]: Tuple of supported CoefCardinality.
        """
        return tuple(CoefCardinality)

    @property
    def analitic_modes(self) -> tuple[AnaliticMode, ...]:
        """*analitic_modes* Get supported analysis modes.

        Returns:
            tuple[AnaliticMode, ...]: Tuple of supported AnaliticMode.
        """
        return tuple(AnaliticMode)


# =============================================================================
# Backward Compatibility Layer
# =============================================================================
# These dict exports maintain compatibility with existing code that expects
# dictionary-based configuration access.

# TODO find a way to integrate this in otherp files
FDU_FWK_DT: dict[str, str] = {e.value: e.description for e in Framework}
"""
Supported Fundamental Dimensional Units (FDUs) Frameworks in *PyDASA*.

Purpose:
    - Defines the dimensional frameworks supported in *PyDASA*.

Note:
    DEPRECATED: Use Framework enum directly for type-safe access.
    Example: Framework.PHYSICAL instead of FDU_FWK_DT["PHYSICAL"]
"""


# TODO find a way to integrate this in otherp files
PARAMS_CAT_DT: dict[str, str] = {e.value: e.description for e in VarCardinality}
"""
Supported Variables categories in *PyDASA*.

Purpose:
    - Defines the variable categories supported in *PyDASA*.
    - Used to classify variables in the dimensional matrix.

Note:
    DEPRECATED: Use VarCardinality enum directly for type-safe access.
    Example: VarCardinality.IN instead of PARAMS_CAT_DT["IN"]
"""


# TODO find a way to integrate this in otherp files
DC_CAT_DT: dict[str, str] = {e.value: e.description for e in CoefCardinality}
"""
Supported categories of Dimensionless Coefficients (DN) in *PyDASA*.

Purpose:
    - Defines the categories of dimensionless coefficients supported in *PyDASA*.
    - Used to classify dimensionless coefficients in formulas and equations.
    - Helps in organizing and managing dimensionless coefficients in the analysis.

Note:
    DEPRECATED: Use CoefCardinality enum directly for type-safe access.
    Example: CoefCardinality.COMPUTED instead of DC_CAT_DT["COMPUTED"]
"""


# TODO find a way to integrate this in otherp files
SENS_ANSYS_DT: dict[str, str] = {e.value: e.description for e in AnaliticMode}
"""
Supported Sensitivity Analysis modes in *PyDASA*.

Note:
    DEPRECATED: Use AnaliticMode enum directly for type-safe access.
    Example: AnaliticMode.SYM instead of SENS_ANSYS_DT["SYM"]
"""
