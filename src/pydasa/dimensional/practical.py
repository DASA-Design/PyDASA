# -*- coding: utf-8 -*-
"""
Module practical.py
===========================================

Module for **MonteCarloHandler** to manage the Monte Carlo experiments in *PyDASA*.

This module provides classes for managing Monte Carlo simulations for sensitivity analysis of dimensionless coefficients.

Classes:
    **MonteCarloHandler**: Manages Monte Carlo simulations analysis, including configuration and execution of the experiments.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, Optional, Dict, List, Callable, Any, Tuple
import random
# import re


# Import validation base classes
from src.pydasa.core.basic import Validation

# Import related classes
from src.pydasa.core.parameter import Variable
from src.pydasa.buckingham.vashchy import Coefficient
from src.pydasa.analysis.simulation import MonteCarloSim

# Import utils
from src.pydasa.utils.default import T
from src.pydasa.utils.error import inspect_var
from src.pydasa.utils.latex import latex_to_python

# Import global configuration
# Import the 'cfg' module to allow global variable editing
from src.pydasa.utils import config as cfg


@dataclass
class MonteCarloHandler(Validation, Generic[T]):
    """**MonteCarloHandler** class for managing Monte Carlo simulations in *PyDASA*.

    Manages the creation, configuration, and execution of Monte Carlo simulations of dimensionless coefficients.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the sensitivity handler.
        description (str): Brief summary of the sensitivity handler.
        _idx (int): Index/precedence of the sensitivity handler.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Simulation Components
        _variables (Dict[str, Variable]): all available parameters/variables in the model (*Variable*).
        _coefficients (Dict[str, Coefficient]): all available coefficients in the model (*Coefficient*).
        _distributions (Dict[str, Callable]): all distribution functions used in the simulations.
        specs (Dict[str, Tuple[float]]): Distribution specifications (probability function x) for all the variables.
        _lead_distribution (Callable): main distribution function for simulations. Uniform distribution by default.
        _iterations (int): Number of simulation to run. Default is 1000.

        # Simulation Results
        _simulations (Dict[str, MonteCarloSim]): all Monte Carlo simulations performed.
        _results (Dict[str, Any]): all results from the simulations.
    """

    # Identification and Classification
    # :attr: _cat
    _cat: str = "NUM"
    """Category of sensitivity analysis (SYM, NUM)."""

    # Variable management
    # :attr: _variables
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of all parameters/variables in the model (*Variable*)."""

    # :attr: _coefficients
    _coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    """Dictionary of all coefficients in the model (*Coefficient*)."""

    # Simulation configuration
    # :attr: _distributions
    _distributions: Dict[str, Callable] = field(default_factory=dict)
    """Variable sampling distributions."""

    # :attr: _specs
    specs: Dict[str, Tuple[float]] = field(default_factory=dict)
    """Distribution specifications (probability function x) for each variable."""

    # :attr: _lead_distribution
    _lead_distribution: Callable = random.uniform
    """Main distribution function for simulations. By default, the native python uniform distribution is used."""

    # :attr: _iterations
    _iterations: int = 1000
    """Number of simulation to run."""

    # Simulation Management
    # :attr: _simulations
    _simulations: Dict[str, MonteCarloSim] = field(default_factory=dict)
    """Dictionary of Monte Carlo simulations."""

    # :attr: _results
    _results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Consolidated results of the Monte Carlo simulations."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the Monte Carlo handler."""
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"MCH_{{\\Pi_{{{self._idx}}}}}" if self._idx >= 0 else "MCH_\\Pi_{-1}"

        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"Monte Carlo Simulation Handler {self._idx}"

        if not self.description:
            self.description = f"Manages Monte Carlo simulations for [{self._coefficients.keys()}] coefficients."

    def _validate_dict(self, dt: dict, exp_type: List[type]) -> bool:
        """*_validate_dict()* Validates a dictionary with expected value types.

        Args:
            dt (dict): Dictionary to validate.
            exp_type (List[type]): Expected types for dictionary values.

        Raises:
            ValueError: If the object is not a dictionary.
            ValueError: If the dictionary is empty.
            ValueError: If the dictionary contains values of unexpected types.

        Returns:
            bool: True if the dictionary is valid.
        """
        if not isinstance(dt, dict):
            _msg = f"{inspect_var(dt)} must be a dictionary. "
            _msg += f"Provided: {type(dt)}"
            raise ValueError(_msg)
        if len(dt) == 0:
            _msg = f"{inspect_var(dt)} cannot be empty. "
            _msg += f"Provided: {dt}"
            raise ValueError(_msg)
        if not all(isinstance(v, exp_type) for v in dt.values()):
            _msg = f"{inspect_var(dt)} must contain {exp_type} values."
            _msg += f" Provided: {[type(v).__name__ for v in dt.values()]}"
            raise ValueError(_msg)
        return True

    def _create_simulations(self) -> None:
        """*_create_simulations()* Creates Monte Carlo simulations for each coefficient.

        Sets up Monte Carlo simulation objects for each coefficient to be analyzed.
        """
        self._simulations.clear()

        for i, (pi, coef) in enumerate(self._coefficients.items()):
            # Create Monte Carlo simulation
            simulation = MonteCarloSim(
                _idx=i,
                _sym=f"MCS_{{{coef.sym}}}",
                _fwk=self._fwk,
                _cat=self._cat,
                name=f"Monte Carlo Simulation for {coef.name}",
                description=f"Monte Carlo simulation for {coef.sym}"
            )

            # Configure with coefficient
            simulation.set_coefficient(coef)

            # Add to list
            self._simulations[pi] = simulation
