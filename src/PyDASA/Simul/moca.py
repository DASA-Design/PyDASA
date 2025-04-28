# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation Module for *PyDASA*.

This module provides a class for performing Monte Carlo simulations based on a symbolic formula.
It allows users to define a formula, specify variable bounds, and generate random samples to evaluate the formula. The results can be summarized with basic statistics.
"""
# native python modules
from typing import Optional, List, Dict, Generic, Callable, Union
from dataclasses import dataclass, field
# import inspect
# import re

# Third-party modules
import numpy as np
import sympy as sp

# Custom modules
# Dimensional Analysis modules

# Utils modules
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.err import error_handler as _error
from Src.PyDASA.Utils.err import inspect_name as _insp_var

# import the 'cfg' module to allow global variable edition
from Src.PyDASA.Utils import cfg

# checking custom modules
assert _error
assert _insp_var
assert cfg
assert T


@dataclass
class MonteCarloSimulation(Generic[T]):
    """
    Class for performing Monte Carlo simulations based on a symbolic formula.

    Attributes:
        formula (Callable): A callable function generated from the symbolic formula.
        variables (list): List of variable names used in the formula.
        num_samples (int): Number of Monte Carlo samples to generate.
        results (np.ndarray): Results of the Monte Carlo simulation.
    """
    formula: str = ""
    exe_formula: Optional[Callable] = None
    variables: Optional[List[Union[Variable, PiNumber]]] = None
    variables: list
    num_samples: int = 1000
    results: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize the results array."""
        self.results = np.array([])

    def simulate(self, bounds: Dict[str, Union[float, tuple]]) -> np.ndarray:
        """
        Perform the Monte Carlo simulation.

        Args:
            bounds (dict): A dictionary where keys are variable names and values are either:
                - A single float (fixed value for the variable).
                - A tuple (min, max) to generate random values for the variable.

        Returns:
            np.ndarray: Array of simulation results.
        """
        # Generate random samples for each variable
        samples = {
            var: self._generate_samples(bounds[var]) for var in self.variables
        }

        # Stack samples into a matrix for evaluation
        sample_matrix = np.column_stack(
            [samples[var] for var in self.variables])

        # Evaluate the formula for all samples
        self.results = self.formula(*sample_matrix.T)
        return self.results

    def _generate_samples(self, bound: Union[float, tuple]) -> np.ndarray:
        """
        Generate random samples for a variable.

        Args:
            bound (Union[float, tuple]): Either a fixed value or a range (min, max).

        Returns:
            np.ndarray: Array of generated samples.
        """
        if isinstance(bound, tuple):
            return np.random.uniform(bound[0], bound[1], self.num_samples)
        return np.full(self.num_samples, bound)

    def summary(self) -> Dict[str, float]:
        """
        Generate a summary of the simulation results.

        Returns:
            dict: Summary statistics (mean, std, min, max).
        """
        return {
            "mean": np.mean(self.results),
            "std": np.std(self.results),
            "min": np.min(self.results),
            "max": np.max(self.results),
        }
