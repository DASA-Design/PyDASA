# -*- coding: utf-8 -*-
"""
Module config.py
===========================================

Configuration module for **PyDASA** Dimensional Analysis parameters. Provides type-safe, immutable configuration using Enums and frozen dataclasses.

Its key features are:
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

NOTE: in the future the enum should be configurated via external files (e.g., `JSON`, `YAML`) to allow user customization.

**IMPORTANT**
    - Based on the theory from H. Görtler, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# native python modules
from enum import Enum
# dataclass imports
from dataclasses import dataclass, field
# data type imports
from typing import ClassVar

# custom modules
from pydasa.core.io import Path, load     # , save
from pydasa.core.constants import DFLT_CFG_FOLDER, DFLT_CFG_FILE

# checking custom modules
assert load
assert DFLT_CFG_FOLDER
assert DFLT_CFG_FILE

# =============================================================================
# *PyDASA* Enum Definitions
# =============================================================================


class Frameworks(str, Enum):
    """Enumerator for Fundamental Dimensional Units (FDUs) frameworks/domains supported in *PyDASA*.

    Inherits from:
        - str: To allow string comparison and representation.
        - Enum: To define enumeration members.
    """

    PHYSICAL = "PHYSICAL"
    COMPUTATION = "COMPUTATION"
    SOFTWARE = "SOFTWARE"
    CUSTOM = "CUSTOM"

    @property
    def description(self) -> str:
        """Get the human-readable description of the dimensional frameworks/domains.

        Returns:
            str: Human-readable framework's description.
        """

        descriptions = {
            Frameworks.PHYSICAL: "Traditional physical dimensional framework (e.g., Length, Mass, Time).",
            Frameworks.COMPUTATION: "Computer science dimensional framework (e.g., Time, Space, Complexity).",
            Frameworks.SOFTWARE: "Software architecture dimensional framework (e.g., Time, Data, Connectivity).",
            Frameworks.CUSTOM: "User-defined dimensional framework for specific use cases.",
        }
        return descriptions[self]


class VarCardinality(str, Enum):
    """Enumerator for Variable cardinality used to classify variables in the dimensional matrix.

    Inherits from:
        - str (class): Python native str class.
        - Enum (class): Python native Enum class.
    """
    IN = "IN"
    OUT = "OUT"
    CTRL = "CTRL"

    @property
    def description(self) -> str:
        """Get the human-readable description of the variable cardinality.

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
    """Enumerator for Dimensionless Coefficient/Numbers/Groups (DC/DN/DG) cardinality used to classify dimensionless coefficients in formulas and equations in **PyDASA**.

    Inherits from:
        - str (class): Python native str class.
        - Enum (class): Python native Enum class.
    """
    COMPUTED = "COMPUTED"
    DERIVED = "DERIVED"

    @property
    def description(self) -> str:
        """Get the human-readable description of the coefficient cardinality.

        Returns:
            str: Human-readable coefficient cardinality description.
        """
        descriptions = {
            CoefCardinality.COMPUTED: "Coefficients directly calculated using the Dimensional Matrix.",
            CoefCardinality.DERIVED: "Coefficients obtained by combining or manipulating Computed Coefficients.",
        }
        return descriptions[self]


class AnaliticMode(str, Enum):
    """Enumerator for the sensitivity analysis modes used to specify the type of analysis performed on variables, coefficients, or functions in **PyDASA**.

    Inherits from:
        - str (class): Python native str class.
        - Enum (class): Python native Enum class.
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


class SimulationMode(str, Enum):
    """Enumerator for simulation modes used to specify sample generation strategy (distribution-based or data-based) in **PyDASA**.

    Inherits from:
        - str (class): Python native str class.
        - Enum (class): Python native Enum class.
    """
    DIST = "DIST"
    DATA = "DATA"

    @property
    def description(self) -> str:
        descriptions = {
            SimulationMode.DIST: "Generate samples using distribution functions (Monte Carlo).",
            SimulationMode.DATA: "Use pre-existing data from Variable._data attributes.",
        }
        return descriptions[self]

# =============================================================================
# Immutable Configuration Singleton
# =============================================================================


@dataclass(frozen=True)
class PyDASAConfig:
    """Singleton class for **PyDASA** configuration. It uses dataclass decorator to freeze the data.
    """

    # :attr: _instance
    _instance: ClassVar["PyDASAConfig | None"] = None
    """Singleton instance of PyDASAConfig."""

    # :attr: SPT_FDU_FWKS
    SPT_FDU_FWKS: dict = field(default_factory=dict)
    """Supported Fundamental Dimensional Units (FDUs) frameworks and their configurations."""

    def __post_init__(self):
        """Post-initialization to load default configuration from file."""
        # Load configuration from default file (relative to this module's directory)
        module_dir = Path(__file__).parent
        fp = module_dir / DFLT_CFG_FOLDER / DFLT_CFG_FILE
        cfg_data = load(fp)

        # Since the dataclass is frozen, use object.__setattr__ to set attributes
        object.__setattr__(self,
                           "SPT_FDU_FWKS",
                           cfg_data.get("frameworks", {}))

    @classmethod
    def get_instance(cls) -> "PyDASAConfig":
        """Get the singleton instance of PyDASAConfig for accessing the global configuration.

        Returns:
            PyDASAConfig: Singleton instance of PyDASAConfig.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def frameworks(self) -> tuple[Frameworks, ...]:
        """Get **PyDASA** supported frameworks.

        Returns:
            tuple[Frameworks, ...]: Tuple of supported Frameworks.
        """
        return tuple(Frameworks)

    @property
    def parameter_cardinality(self) -> tuple[VarCardinality, ...]:
        """Get **PyDASA** supported variable cardinalities.

        Returns:
            tuple[VarCardinality, ...]: Tuple of supported VarCardinality.
        """
        return tuple(VarCardinality)

    @property
    def coefficient_cardinality(self) -> tuple[CoefCardinality, ...]:
        """Get **PyDASA** supported coefficient cardinalities.

        Returns:
            tuple[CoefCardinality, ...]: Tuple of supported CoefCardinality.
        """
        return tuple(CoefCardinality)

    @property
    def analitic_modes(self) -> tuple[AnaliticMode, ...]:
        """Get **PyDASA** supported analysis modes.

        Returns:
            tuple[AnaliticMode, ...]: Tuple of supported AnaliticMode.
        """
        return tuple(AnaliticMode)

    @property
    def simulation_modes(self) -> tuple[SimulationMode, ...]:
        """Get **PyDASA** supported simulation modes.

        Returns:
            tuple[SimulationMode, ...]: Tuple of supported SimulationMode.
        """
        return tuple(SimulationMode)


# Get singleton instance for configuration
# :attr: PYDASA_CFG
PYDASA_CFG: PyDASAConfig = PyDASAConfig()
"""
Singleton instance of PyDASAConfig for accessing global configuration.
"""
