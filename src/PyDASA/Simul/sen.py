# -*- coding: utf-8 -*-
"""
Module for sensitivity analysis using Fourier Amplitude Sensitivity Testing (FAST).

This module provides the `SensitivityAnalyzer` class, which performs sensitivity analysis on a list of *PiNumbers* and *Variables*.

It computes the sensitivity of each *PiNumber* based on the provided *Variable* samples and ranks them accordingly.
"""

# native python modules
from typing import Optional, List, Generic
from dataclasses import dataclass, field

# Third-party modules
import numpy as np

# custom modules
# Utils modules
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.err import error_handler as error
# Dimensional Analysis modules
from Src.PyDASA.Pi.coef import PiNumber
from Src.PyDASA.Units.params import Variable
# Data Structures
from Src.PyDASA.DStructs.Tables.scht import SCHashTable

# import the 'cfg' module to allow global variable edition
from Src.PyDASA.Utils import cfg

# checking custom modules
assert error
assert cfg
assert T


@dataclass
class SensitivityAnalyzer(Generic[T]):

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *SensitivityAnalyzer*.
    """

    # :attr: _sym
    _sym: str = "DA_{x}"
    """
    Symbol of the *SensitivityAnalyzer*. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the instance.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *SensitivityAnalyzer*, can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. By default, it is set to `PHYSICAL`.
    """

    _relevance_lt: List[Variable] = field(default_factory=list)
    _pi_numbers: List[PiNumber] = field(default_factory=list)
    _vars_map: SCHashTable = field(default_factory=lambda: SCHashTable)
    _pi_num_map: SCHashTable = field(default_factory=lambda: SCHashTable)

    # Public attributes
    # :attr: name
    name: str = "Dimensional Model"
    """
    User-friendly name of the *SensitivityAnalyzer*.
    """

    # :attr: description
    description: str = ""
    """
    Small summary of the *SensitivityAnalyzer*.
    """

    sensitivity: Optional[dict] = None

    def perform_analysis(self, samples: int = 1000) -> None:
        """
        Performs sensitivity analysis using FAST and ranks the PiNumbers.

        Args:
            samples (int): Number of samples for the analysis. Default is 1000.
        """
        # Generate random samples for each variable
        variable_samples = {
            var.name: np.random.uniform(var.min, var.max, samples)
            for var in self.variables
        }

        # Analyze each PiNumber
        for pi in self.pi_numbers:
            pi_values = self._compute_pi_values(pi, variable_samples)
            pi.sensitivity = self._compute_sensitivity(pi_values)
            pi.max_value = np.max(pi_values)
            pi.min_value = np.min(pi_values)
            pi.avg_value = np.mean(pi_values)

        # Sort PiNumbers by sensitivity in descending order
        self.pi_numbers.sort(key=lambda x: x.sensitivity, reverse=True)

    def _compute_pi_values(self, pi: PiNumber, variable_samples: dict) -> np.ndarray:
        """
        Computes the values of a PiNumber based on variable samples.

        Args:
            pi (PiNumber): The PiNumber object.
            variable_samples (dict): Dictionary of variable samples.

        Returns:
            np.ndarray: Array of computed Pi values.
        """
        pi_values = np.ones(len(next(iter(variable_samples.values()))))
        for var, exp in zip(pi.param_lt, pi.dim_col):
            if var in variable_samples:
                pi_values *= variable_samples[var] ** exp
        return pi_values

    def _compute_sensitivity(self, pi_values: np.ndarray) -> float:
        """
        Computes the sensitivity of a PiNumber using FAST.

        Args:
            pi_values (np.ndarray): Array of computed Pi values.

        Returns:
            float: Sensitivity score.
        """
        # Perform Fourier Transform
        fft_values = np.fft.fft(pi_values)
        amplitudes = np.abs(fft_values)

        # Compute sensitivity as the ratio of the first harmonic to the total amplitude
        sensitivity = amplitudes[1] / np.sum(amplitudes)
        return sensitivity

    def get_results(self) -> List[dict]:
        """
        Returns the ranked PiNumbers with their sensitivity, max, min, and avg values.

        Returns:
            List[dict]: List of dictionaries containing PiNumber analysis results.
        """
        return [
            {
                "pi_number": pi.sym,
                "sensitivity": pi.sensitivity,
                "max": pi.max_value,
                "min": pi.min_value,
                "avg": pi.avg_value,
            }
            for pi in self.pi_numbers
        ]
