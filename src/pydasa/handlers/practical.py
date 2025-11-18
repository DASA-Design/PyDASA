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
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, Optional, Dict, List, Callable, Any
import random
# import re

# Third-party imports
import numpy as np

# Import validation base classes
from pydasa.core.basic import Validation

# Import related classes
from pydasa.core.parameter import Variable
from pydasa.buckingham.vashchy import Coefficient
from pydasa.analysis.simulation import MonteCarloSim

# Import utils
from pydasa.utils.default import T
from pydasa.utils.error import inspect_var
from pydasa.utils.latex import latex_to_python

# Import global configuration
# Import the 'cfg' module to allow global variable editing
from pydasa.utils import config as cfg


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
        _cat (str): Category of analysis (SYM, NUM).

        # Simulation Components
        _variables (Dict[str, Variable]): all available parameters/variables in the model (*Variable*).
        _coefficients (Dict[str, Coefficient]): all available coefficients in the model (*Coefficient*).
        _distributions (Dict[str, Callable]): all distribution functions used in the simulations.
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
    _distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Variable sampling distributions and specifications for simulations (specific name, parameters, and function)."""

    # :attr: _lead_distribution
    _lead_distribution: Optional[Callable] = random.uniform
    """Main distribution function for simulations. By default, the native python uniform distribution is used."""

    # :attr: _iterations
    _iterations: int = 1000
    """Number of simulation to run."""

    # Simulation Management
    # :arttr: _dependencies
    _mem_cache: Dict[str, List[str]] = field(default_factory=dict)

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

        # if len(self._distributions) == 0:
        #     self._config_distributions()
        # if len(self._simulations) == 0:
        #     self._config_simulations()

    def config_simulations(self) -> None:
        if len(self._distributions) == 0:
            self._config_distributions()
        if len(self._simulations) == 0:
            self._config_simulations()

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

    def _config_distributions(self) -> None:
        """*_config_distributions()* Creates the Monte Carlo distributions for each variable.

        Raises:
            ValueError: If the distribution specifications are invalid.
        """
        # Clear existing distributions
        self._distributions.clear()

        for var in self._variables.values():
            sym = var.sym
            if sym not in self._distributions:
                specs = [var.dist_type, var.dist_params, var.dist_func]
                if not any(specs):
                    _msg = f"Invalid distribution for variable '{sym}'. "
                    _msg += f"Incomplete specifications provided: {specs}"
                    raise ValueError(_msg)
                self._distributions[sym] = {
                    "depends": var.depends,
                    "dtype": var.dist_type,
                    "params": var.dist_params,
                    "func": var.dist_func
                }

    def _get_distributions(self, var_keys: List[str]) -> Dict[str, Any]:
        """*_get_distributions()* Retrieves the distribution specifications for a list of variable keys.

        Args:
            var_keys (List[str]): List of variable keys.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of distribution specifications.
        """
        dist = {k: v for k, v in self._distributions.items() if k in var_keys}
        return dist

    def _get_dependencies(self, var_keys: List[str]) -> Dict[str, Any]:
        deps = {k: v.depends for k, v in self._variables.items()
                if k in var_keys}
        return deps

    def _config_simulations(self) -> None:
        """*_config_simulations()* Sets up Monte Carlo simulation objects for each coefficient to be analyzed, by specifing:
        """
        # clear existing simulations
        self._simulations.clear()

        # Create simulations for each coefficient
        for i, (pi, coef) in enumerate(self._coefficients.items()):
            # Create Monte Carlo simulation
            # get the key subset relevant to the coefficient
            # create the Monte Carlo Simulation fo the coefficient
            sim = MonteCarloSim(
                _idx=i,
                _sym=f"MC_{{{coef.sym}}}",
                _fwk=self._fwk,
                _cat=self._cat,
                _pi_expr=coef.pi_expr,
                _variables=self._variables,
                # _distributions=self._get_distributions(keys),
                name=f"Monte Carlo Simulation for {coef.name}",
                description=f"Monte Carlo simulation for {coef.sym}"
            )

            # Configure with coefficient, this is critical!!!
            sim.set_coefficient(coef)

            # Extract variables from the coefficient's expression
            vars_in_coef = list(coef.var_dims.keys())

            # Set the distributions and dependencies
            sim._distributions = self._get_distributions(vars_in_coef)
            sim._dependencies = self._get_dependencies(vars_in_coef)
            # Add to list
            self._simulations[pi] = sim

    def simulate(self,
                 n_samples: int = 1000) -> Dict[str, Dict[str, Any]]:
        """*_simulate()* Runs the Monte Carlo simulations.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.

        Returns:
            Dict[str, Dict[str, Any]]: Simulation results.
        """
        results = {}

        for sym in self._coefficients:
            # Get the simulation object
            sim = self._simulations.get(sym)
            if not sim:
                _msg = f"Simulation for coefficient '{sym}' not found."
                raise ValueError(_msg)

            # Run the simulation
            sim.run(n_samples)
            res = {
                "inputs": sim.inputs,
                "results": sim.results,
                "statistics": sim.statistics,
            }

            # Store results
            results[sym] = res
        self._results = results

    def get_simulation(self, name: str) -> MonteCarloSim:
        """*get_simulation()* Get a simulation by name.

        Args:
            name (str): Name of the simulation.

        Returns:
            MonteCarloSim: The requested simulation.

        Raises:
            ValueError: If the simulation doesn't exist.
        """
        if name not in self._simulations:
            raise ValueError(f"Simulation '{name}' does not exist.")

        return self._simulations[name]

    def get_distribution(self, name: str) -> Dict[str, Any]:
        """*get_distribution()* Get the distribution by name.

        Args:
            name (str): Name of the distribution.

        Returns:
            Dict[str, Any]: The requested distribution.

        Raises:
            ValueError: If the distribution doesn't exist.
        """
        if name not in self._distributions:
            raise ValueError(f"Distribution '{name}' does not exist.")

        return self._distributions[name]

    def get_results(self, name: str) -> Dict[str, Any]:
        """*get_results()* Get the results of a simulation by name.

        Args:
            name (str): Name of the simulation.

        Returns:
            Dict[str, Any]: The results of the requested simulation.

        Raises:
            ValueError: If the results for the simulation don't exist.
        """
        if name not in self._results:
            raise ValueError(f"Results for simulation '{name}' do not exist.")

        return self._results[name]

    # Property getters and setters

    @property
    def cat(self) -> str:
        """*cat* Get the analysis category.

        Returns:
            str: Category (SYM, NUM, HYB).
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Set the analysis category.

        Args:
            val (str): Category value.

        Raises:
            ValueError: If category is invalid.
        """
        if val.upper() not in cfg.SENS_ANSYS_DT:
            raise ValueError(
                f"Invalid category: {val}. "
                f"Must be one of: {', '.join(cfg.SENS_ANSYS_DT.keys())}"
            )
        self._cat = val.upper()

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the list of variables.

        Returns:
            Dict[str, Variable]: Dictionary of variables.
        """
        return self._variables.copy()

    @variables.setter
    def variables(self, val: Dict[str, Variable]) -> None:
        """*variables* Set the list of variables.

        Args:
            val (Dict[str, Variable]): Dictionary of variables.

        Raises:
            ValueError: If dictionary is invalid.
        """
        if self._validate_dict(val, (Variable,)):
            self._variables = val

            # Clear existing analyses
            self._simulations.clear()

    @property
    def coefficients(self) -> Dict[str, Coefficient]:
        """*coefficients* Get the dictionary of coefficients.

        Returns:
            Dict[str, Coefficient]: Dictionary of coefficients.
        """
        return self._coefficients.copy()

    @coefficients.setter
    def coefficients(self, val: Dict[str, Coefficient]) -> None:
        """*coefficients* Set the dictionary of coefficients.

        Args:
            val (Dict[str, Coefficient]): Dictionary of coefficients.

        Raises:
            ValueError: If dictionary is invalid.
        """
        if self._validate_dict(val, (Coefficient,)):
            self._coefficients = val

            # Clear existing analyses
            self._simulations.clear()

    @property
    def simulations(self) -> Dict[str, MonteCarloSim]:
        """*simulations* Get the dictionary of Monte Carlo simulations.

        Returns:
            Dict[str, MonteCarloSim]: Dictionary of Monte Carlo simulations.
        """
        return self._simulations.copy()

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        """*results* Get the Monte Carlo results.

        Returns:
            Dict[str, Dict[str, Any]]: Monte Carlo results.
        """
        return self._results.copy()

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all handler properties to their initial state.
        """
        # Reset base class attributes
        super().clear()

        # Reset specific attributes
        self._lead_distribution = None
        self._simulations.clear()
        self._distributions.clear()
        self._results.clear()

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert the handler's state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the handler's state.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "fwk": self._fwk,
            "cat": self._cat,
            "variables": [var.to_dict() for var in self._variables],
            "coefficients": [coef.to_dict() for coef in self._coefficients],
            "simulations": [sim.to_dict() for sim in self._simulations],
            "results": self._results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MonteCarloHandler:
        """*from_dict()* Create a MonteCarloHandler instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing the handler's state.

        Returns:
            MonteCarloHandler: New instance of MonteCarloHandler.
        """
        # Create instance with basic attributes
        instance = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            _idx=data.get("idx", -1),
            _sym=data.get("sym", ""),
            _fwk=data.get("fwk", "CUSTOM"),
            _cat=data.get("cat", "NUM")
        )

        # Set variables
        vars_data = data.get("variables", {})
        vars_dict = {var_data["sym"]: Variable.from_dict(var_data) for var_data in vars_data}
        instance.variables = vars_dict

        # Set coefficients
        coefs_data = data.get("coefficients", {})
        coefs_dict = {coef_data["sym"]: Coefficient.from_dict(coef_data) for coef_data in coefs_data}
        instance.coefficients = coefs_dict

        # Configure simulations
        instance.config_simulations()

        # Set results if available
        results_data = data.get("results", {})
        if results_data:
            instance._results = results_data

        return instance
