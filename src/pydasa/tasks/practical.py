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
from typing import Dict, List, Any, Tuple, Union, Optional
# import random
# import re, Optional, Callable

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Import validation base classes
from pydasa.core.basic import Foundation

# Import related classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.analysis.simulation import MonteCarloSim

# Import utils
from pydasa.validations.error import inspect_var
from pydasa.utils.latex import latex_to_python

# Import global configuration
# Import the 'cfg' module to allow global variable editing
from pydasa.core import setup as cfg


@dataclass
class MonteCarloHandler(Foundation):
    """**MonteCarloHandler** class for managing Monte Carlo simulations in *PyDASA*.

    Manages the creation, configuration, and execution of Monte Carlo simulations of dimensionless coefficients.

    Args:
        Foundation: Foundation class for validation of symbols and frameworks.

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
        _distributions (Dict[str, Dict[str, Any]]): all distribution functions used in the simulations.
        _experiments (int): Number of simulation to run. Default is 1000.

        # Simulation Results
        _simulations (Dict[str, MonteCarloSim]): all Monte Carlo simulations performed.
        _results (Dict[str, Any]): all results from the simulations.
        _shared_cache (Dict[str, NDArray[np.float64]]): In-memory cache for simulation data between coefficients.
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

    # :attr: _experiments
    _experiments: int = 1000
    """Number of simulation to run."""

    # Simulation Management
    # :attr: _shared_cache
    _shared_cache: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    """In-memory cache for simulation data between coefficients."""

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
            coef_keys = ", ".join(self._coefficients.keys()) if self._coefficients else "no coefficients"
            self.description = f"Manages Monte Carlo simulations for [{coef_keys}] coefficients."

        # Ensure mem_cache is always initialized
        if self._shared_cache is None:
            self._shared_cache = {}

    def config_simulations(self) -> None:
        """*config_simulations()* Configures distributions and simulations if not already set."""
        if len(self._distributions) == 0:
            self._config_distributions()
        if len(self._simulations) == 0:
            self._config_simulations()

    # ========================================================================
    # Foundation Methods
    # ========================================================================

    def _validate_dict(self,
                       dt: Dict[str, Any],
                       exp_type: Union[type, Tuple[type, ...]]) -> bool:
        """*_validate_dict()* Validates a dictionary with expected value types.

        Args:
            dt (Dict[str, Any]): Dictionary to validate.
            exp_type (Union[type, Tuple[type, ...]]): Expected type(s) for dictionary values.

        Raises:
            ValueError: If the object is not a dictionary.
            ValueError: If the dictionary is empty.
            ValueError: If the dictionary contains values of unexpected types.

        Returns:
            bool: True if the dictionary is valid.
        """
        # variable inspection
        var_name = inspect_var(dt)

        # Validate is dictionary
        if not isinstance(dt, dict):
            _msg = f"{var_name} must be a dictionary. "
            _msg += f"Provided: {type(dt).__name__}"
            raise ValueError(_msg)

        # Validate not empty
        if len(dt) == 0:
            _msg = f"{var_name} cannot be empty. "
            _msg += f"Provided: {dt}"
            raise ValueError(_msg)

        # Convert list to tuple for isinstance()
        type_check = exp_type if isinstance(exp_type, tuple) else (exp_type,) if not isinstance(exp_type, tuple) else exp_type

        # Validate value types
        if not all(isinstance(v, type_check) for v in dt.values()):
            # Format expected types for error message
            if isinstance(exp_type, tuple):
                type_names = " or ".join(t.__name__ for t in exp_type)
            else:
                type_names = exp_type.__name__

            actual_types = [type(v).__name__ for v in dt.values()]
            _msg = f"{var_name} must contain {type_names} values. "
            _msg += f"Provided: {actual_types}"
            raise ValueError(_msg)

        return True

    def _validate_coefficient_vars(self,
                                   coef: Coefficient,
                                   pi_sym: str) -> Dict[str, Any]:
        """*_validate_coefficient_vars()* Validates and returns coefficient's var_dims.

        Args:
            coef (Coefficient): The coefficient to validate.
            pi_sym (str): The coefficient symbol for error messages.

        Returns:
            Dict[str, Any]: The validated var_dims dictionary.

        Raises:
            ValueError: If var_dims is None or missing.
        """
        if not hasattr(coef, 'var_dims'):
            _msg = f"Coefficient '{pi_sym}' missing var_dims attribute."
            raise ValueError(_msg)

        var_dims = coef.var_dims
        if var_dims is None:
            _msg = f"Coefficient '{pi_sym}' has None var_dims. "
            _msg += "Ensure the coefficient was properly initialized."
            raise ValueError(_msg)

        if not isinstance(var_dims, dict):
            _msg = f"Coefficient '{pi_sym}' var_dims must be a dictionary. "
            _msg += f"Got: {type(var_dims).__name__}"
            raise TypeError(_msg)

        return var_dims

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def _config_distributions(self) -> None:
        """*_config_distributions()* Creates the Monte Carlo distributions for each variable.

        Raises:
            ValueError: If the distribution specifications are invalid.
        """
        # Clear existing distributions
        self._distributions.clear()

        # Validate variables exist before processing
        if not self._variables:
            _msg = "Cannot configure distributions: no variables defined."
            raise ValueError(_msg)

        for var in self._variables.values():
            sym = var.sym

            # Skip if already configured
            if sym in self._distributions:
                continue

            # Collect specs for better error reporting
            specs = {
                "dist_type": var.dist_type,
                "dist_params": var.dist_params,
                "dist_func": var.dist_func
            }

            # Validate distribution specifications
            if not any(specs.values()):
                _msg = f"Invalid distribution for variable '{sym}'. "
                _msg += f"Incomplete specifications provided: {specs}"
                raise ValueError(_msg)

            # Store distribution configuration
            self._distributions[sym] = {
                "depends": var.depends,
                "dtype": var.dist_type,
                "params": var.dist_params,
                "func": var.dist_func
            }

    def _get_distributions(self,
                           var_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """*_get_distributions()* Retrieves the distribution specifications for a list of variable keys.

        Args:
            var_keys (List[str]): List of variable keys.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of distribution specifications.

        Raises:
            ValueError: If required distributions are missing.
        """
        # Filter distributions for requested variables
        dist = {k: v for k, v in self._distributions.items() if k in var_keys}

        # Warn about missing distributions
        missing = [k for k in var_keys if k not in dist]

        if missing:
            _msg = f"Missing distributions for variables: {missing}. "
            _msg += "Ensure _config_distributions() has been called."
            raise ValueError(_msg)

        return dist

    def _get_dependencies(self, var_keys: List[str]) -> Dict[str, List[str]]:
        """*_get_dependencies()* Retrieves variable dependencies for a list of variable keys.

        Args:
            var_keys (List[str]): List of variable keys.

        Returns:
            Dict[str, List[str]]: Dictionary mapping variable symbols to their dependencies.
        """
        deps = {
            k: v.depends for k, v in self._variables.items() if k in var_keys
        }
        return deps

    def _init_shared_cache(self) -> None:
        """*_init_shared_cache()* Initialize shared cache for all variables."""
        # Initialize cache for each variable once
        for var_sym in self._variables.keys():
            self._shared_cache[var_sym] = np.full((self._experiments, 1),
                                                  np.nan,
                                                  dtype=np.float64)

    def _config_simulations(self) -> None:
        """*_config_simulations()* Sets up Monte Carlo simulation objects for each coefficient to be analyzed.

        Creates a MonteCarloSim instance for each coefficient with appropriate distributions and dependencies.

        Raises:
            ValueError: If coefficients or variables are not properly configured.
        """
        # Validate prerequisites
        if not self._coefficients:
            _msg = "Cannot configure simulations: no coefficients defined."
            raise ValueError(_msg)

        if not self._variables:
            _msg = "Cannot configure simulations: no variables defined."
            raise ValueError(_msg)

        if not self._distributions:
            _msg = "Cannot configure simulations: distributions not defined. "
            raise ValueError(_msg)

        # Clear existing simulations
        self._simulations.clear()

        # Initialize shared cache once
        if not self._shared_cache:
            self._init_shared_cache()

        # Create simulations for each coefficient
        for i, (pi, coef) in enumerate(self._coefficients.items()):
            # Validate coefficient before processing
            var_dims = self._validate_coefficient_vars(coef, pi)

            # Extract variables from the coefficient's expression
            vars_in_coef = list(var_dims.keys())

            # Skip coefficients with no variables
            if not vars_in_coef:
                _msg = f"Coefficient '{pi}' has no variables in expression. Skipping simulation."
                print(f"Warning: {_msg}")
                continue

            try:
                # Create Monte Carlo simulation
                sim = MonteCarloSim(
                    _idx=i,
                    _sym=f"MC_{{{coef.sym}}}",
                    _fwk=self._fwk,
                    _cat=self._cat,
                    _pi_expr=coef.pi_expr,
                    _coefficient=coef,
                    _variables=self._variables,
                    # _simul_cache=self._shared_cache,
                    _experiments=self._experiments,
                    name=f"Monte Carlo Simulation for {coef.name}",
                    description=f"Monte Carlo simulation for {coef.sym}",
                )

                # Configure with coefficient
                sim.set_coefficient(coef)

                # Get distributions with validation
                sim._distributions = self._get_distributions(vars_in_coef)
                sim._dependencies = self._get_dependencies(vars_in_coef)

                # CRITICAL: Share the cache reference
                sim._simul_cache = self._shared_cache

                # Add to simulations dictionary
                self._simulations[pi] = sim

            except Exception as e:
                _msg = f"Failed to create simulation for '{pi}': {str(e)}"
                raise RuntimeError(_msg) from e

    # ========================================================================
    # Simulation Execution Methods
    # ========================================================================

    def simulate(self, n_samples: Optional[int] = None) -> None:
        """*simulate()* Runs the Monte Carlo simulations.

        Args:
            n_samples (Optional[int]): Number of samples to generate.
                If None, uses self._experiments value. Defaults to None.

        Raises:
            ValueError: If simulations are not configured.
            ValueError: If a required simulation is not found.
        """
        # Validate simulations exist
        if not self._simulations:
            _msg = "No simulations configured. Call config_simulations() first."
            raise ValueError(_msg)

        # Use default if not specified
        if n_samples is not None:
            self._experiments = n_samples

        #  Validate n_samples
        if self._experiments < 1:
            _msg = f"Experiments must be positive. Got: {n_samples}"
            raise ValueError(_msg)

        # ✅ Initialize shared cache BEFORE running simulations
        if not self._shared_cache:
            for var_sym in self._variables.keys():
                self._shared_cache[var_sym] = np.full((self._experiments, 1),
                                                      np.nan,
                                                      dtype=np.float64)

        # print("----------")
        # print(f"_shared_cache keys: {self._shared_cache.keys()}")
        # # ✅ Assign shared cache to ALL simulations
        # for sim in self._simulations.values():
        #     sim._simul_cache = self._shared_cache

        results = {}

        for sym in self._coefficients:
            # Get the simulation object
            sim = self._simulations.get(sym)
            if not sim:
                _msg = f"Simulation for coefficient '{sym}' not found. "
                _msg += "Ensure _config_simulations() completed successfully."
                raise ValueError(_msg)

            try:
                # print("-----------------------------------")
                # print(f"_shared_cache status:\n {self._shared_cache}")
                # print("-----------------------------------")

                # ✅ Use shared cache
                sim._simul_cache = self._shared_cache

                # Run the simulation
                sim.run(self._experiments)

                # Store comprehensive results
                res = {
                    "inputs": sim.inputs,
                    "results": sim.results,
                    "statistics": sim.statistics,
                }

                # Store results
                results[sym] = res

            except Exception as e:
                _msg = f"Simulation failed for coefficient '{sym}': {str(e)}"
                raise RuntimeError(_msg) from e

        self._results = results

    # ========================================================================
    # Getter Methods
    # ========================================================================

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
            available = ", ".join(self._simulations.keys())
            _msg = f"Simulation '{name}' does not exist. "
            _msg += f"Available: {available}"
            raise ValueError(_msg)

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
            available = ", ".join(self._distributions.keys())
            _msg = f"Distribution '{name}' does not exist. "
            _msg += f"Available: {available}"
            raise ValueError(_msg)

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
            available = ", ".join(self._results.keys())
            _msg = f"Results for simulation '{name}' do not exist. "
            _msg += f"Available: {available}"
            raise ValueError(_msg)

        return self._results[name]

    # ========================================================================
    # Property Getters and Setters
    # ========================================================================

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
        val_upper = val.upper()
        if val_upper not in cfg.SENS_ANSYS_DT:
            available = ", ".join(cfg.SENS_ANSYS_DT.keys())
            _msg = f"Invalid category: {val}. "
            _msg += f"Must be one of: {available}"
            raise ValueError(_msg)
        self._cat = val_upper

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
        if self._validate_dict(val, Variable):
            self._variables = val

            # Clear existing analyses since variables changed
            self._simulations.clear()
            self._distributions.clear()
            self._results.clear()

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
        if self._validate_dict(val, Coefficient):
            self._coefficients = val

            # Clear existing analyses since coefficients changed
            self._simulations.clear()
            self._results.clear()

    @property
    def experiments(self) -> int:
        """*experiments* Get the number of experiments.

        Returns:
            int: Number of experiments to run.
        """
        return self._experiments

    @experiments.setter
    def experiments(self, val: int) -> None:
        """*experiments* Set the number of experiments.

        Args:
            val (int): Number of experiments.

        Raises:
            ValueError: If value is not positive.
        """
        if val < 1:
            _msg = f"Number of experiments must be positive. Got: {val}"
            raise ValueError(_msg)
        self._experiments = val

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

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all handler properties to their initial state.
        """
        # Reset base class attributes
        # super().clear()

        # Reset specific attributes
        self._simulations.clear()
        self._distributions.clear()
        self._results.clear()
        self._shared_cache.clear()

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
            "alias": self._alias,
            "fwk": self._fwk,
            "cat": self._cat,
            "experiments": self._experiments,
            "variables": {
                k: v.to_dict() for k, v in self._variables.items()
            },
            "coefficients": {
                k: v.to_dict() for k, v in self._coefficients.items()
            },
            "simulations": {
                k: v.to_dict() for k, v in self._simulations.items()
            },
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
            _alias=data.get("alias", ""),
            _fwk=data.get("fwk", "CUSTOM"),
            _cat=data.get("cat", "NUM"),
            _experiments=data.get("experiments", 1000)
        )

        # Set variables
        vars_data = data.get("variables", {})
        if vars_data:
            vars_dict = {k: Variable.from_dict(v) for k, v in vars_data.items()}
            instance.variables = vars_dict

        # Set coefficients
        coefs_data = data.get("coefficients", {})
        if coefs_data:
            coefs_dict = {k: Coefficient.from_dict(v) for k, v in coefs_data.items()}
            instance.coefficients = coefs_dict

        # Configure simulations if we have variables and coefficients
        if vars_data and coefs_data:
            instance.config_simulations()

        # Set results if available
        results_data = data.get("results", {})
        if results_data:
            instance._results = results_data

        return instance
