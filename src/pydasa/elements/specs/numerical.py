# -*- coding: utf-8 -*-
"""
Module numerical.py
===========================================

Numerical perspective for variable representation.

This module defines the NumericalSpecs class representing the computational
value ranges and discretization properties of a variable.

Classes:
    **NumericalSpecs**: Numerical variable specifications

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# native python modules
# dataclass imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# third-party numerical imports
import numpy as np

# custom modules
from pydasa.validations.decorators import validate_type


@dataclass
class NumericalSpecs:
    """Numerical perspective: computational value ranges. Answers the question: "What VALUES can this variable take?"

    This perspective focuses on:
        - Concrete bounds (minimum, maximum)
        - Central tendency (mean value)
        - Variation (standard deviation)
        - Discretization for simulations (step size, range arrays)
        - Unit conversions (original ↔ standardized)
        - Variable dependencies (calculated variables)

    Attributes:
        # From NumericalSpecs:
        # Value ranges (original units)
            _metric (Optional[float]): Specific value in original units.
            _min (Optional[float]): Minimum value in original units.
            _max (Optional[float]): Maximum value in original units.
            _mean (Optional[float]): Mean value in original units.
            _dev (Optional[float]): Standard deviation in original units.
        # Value ranges (standardized units)
            _std_metric (Optional[str]): Specific metric system for standardized units.
            _std_min (Optional[float]): Minimum value in standard units.
            _std_max (Optional[float]): Maximum value in standard units.
            _std_mean (Optional[float]): Mean value in standard units.
            _std_dev (Optional[float]): Standard deviation in standard units.
            _step (Optional[float]): Step size for simulations.
            _std_range (np.ndarray): Range for numerical analysis in standardized units and discretization.
    """

    # Value ranges (original units)
    # :attr: _metric
    _metric: Optional[float] = None
    """Specific value in original units."""

    # :attr: _min
    _min: Optional[float] = None
    """Minimum value in original units."""

    # :attr: _max
    _max: Optional[float] = None
    """Maximum value in original units."""

    # :attr: _mean
    _mean: Optional[float] = None
    """Mean value in original units."""

    # :attr: _dev
    _dev: Optional[float] = None
    """Standard deviation in original units."""

    # Value ranges (standardized units)
    # :attr: _std_metric
    _std_metric: Optional[float] = None
    """Specific metric system for standardized units."""

    # :attr: _std_min
    _std_min: Optional[float] = None
    """Minimum value in standard units."""

    # :attr: _std_max
    _std_max: Optional[float] = None
    """Maximum value in standard units."""

    # :attr: _std_mean
    _std_mean: Optional[float] = None
    """Mean value in standard units."""

    # :attr: _std_dev
    _std_dev: Optional[float] = None
    """Standard deviation in standard units."""

    # :attr: _step
    _step: Optional[float] = None
    """Step size for simulations."""

    # :attr: _std_range
    _std_range: np.ndarray = field(default_factory=lambda: np.array([]))
    """Range array for analysis."""

    # Value Ranges (Original Units)
    @property
    def metric(self) -> Optional[float]:
        """*metric* Get specific metric system for original units.

        Returns:
            Optional[str]: Specific metric system for original units.
        """
        return self._metric

    @metric.setter
    @validate_type(int, float)
    def metric(self, val: Optional[float]) -> None:
        """*metric* Sets specific metric system for original units.

        Args:
            val (Optional[float]): Specific metric system for original units.

        Raises:
            ValueError: If value not a valid number.
        """
        self._metric = val

    @property
    def min(self) -> Optional[float]:
        """*min* Get minimum range value.

        Returns:
            Optional[float]: Minimum range value.
        """
        return self._min

    @min.setter
    @validate_type(int, float)
    def min(self, val: Optional[float]) -> None:
        """*min* Sets minimum range value.

        Args:
            val (Optional[float]): Minimum range value.

        Raises:
            ValueError: If value not a valid number.
            ValueError: If value is greater than max.
        """
        if val is not None and self._max is not None and val > self._max:
            _msg = f"Minimum {val} cannot be greater than maximum {self._max}."
            raise ValueError(_msg)

        self._min = val

        # TODO reassert this code later, seems redundant with _prepare_dims()
        # Update range if all values are available
        if all([self._min is not None,
                self._max is not None,
                self._step is not None]):
            self._range = np.arange(self._min,
                                    self._max,
                                    self._step)

    @property
    def max(self) -> Optional[float]:
        """*max* Get the maximum range value.

        Returns:
            Optional[float]: Maximum range value.
        """
        return self._max

    @max.setter
    @validate_type(int, float)
    def max(self, val: Optional[float]) -> None:
        """*max* Sets the maximum range value.

        Args:
            val (Optional[float]): Maximum range value.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is less than min.
        """
        # Check if both values exist before comparing
        if val is not None and self._min is not None and val < self._min:
            _msg = f"Maximum {val} cannot be less than minimum {self._min}."
            raise ValueError(_msg)

        self._max = val

        # TODO reassert this code later, seems redundant with _prepare_dims()
        # Update range if all values are available
        if all([self._min is not None,
                self._max is not None,
                self._step is not None]):
            self._range = np.arange(self._min,
                                    self._max,
                                    self._step)

    @property
    def mean(self) -> Optional[float]:
        """*mean* Get the Variable average value.

        Returns:
            Optional[float]: Variable average value.
        """
        return self._mean

    @mean.setter
    @validate_type(int, float)
    def mean(self, val: Optional[float]) -> None:
        """*mean* Sets the Variable mean value.

        Args:
            val (Optional[float]): Variable mean value.

        Raises:
            ValueError: If value not a valid number.
            ValueError: If value is outside min-max range.
        """
        # Only validate range if val is not None
        if val is not None:
            low = (self._min is not None and val < self._min)
            high = (self._max is not None and val > self._max)
            if low or high:
                _msg = f"Mean {val} "
                _msg += f"must be between {self._min} and {self._max}."
                raise ValueError(_msg)

        self._mean = val

    @property
    def dev(self) -> Optional[float]:
        """*dev* Get the Variable standard deviation.

        Returns:
            Optional[float]: Variable standard deviation.
        """
        return self._dev

    @dev.setter
    def dev(self, val: Optional[float]) -> None:
        """*dev* Sets the Variable standard deviation.

        Args:
            val (Optional[float]): Variable standard deviation.
        Raises:
            ValueError: If value not a valid number.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standard deviation must be a number.")

        self._dev = val

    # Value Ranges (Standardized Units)

    @property
    def std_metric(self) -> Optional[float]:
        """*std_metric* Get specific metric system for standardized units.

        Returns:
            Optional[str]: Specific metric system for standardized units.
        """
        return self._std_metric

    @std_metric.setter
    @validate_type(int, float)
    def std_metric(self, val: Optional[float]) -> None:
        """*std_metric* Sets specific metric system for standardized units.

        Args:
            val (Optional[float]): Specific metric system for standardized units.
        Raises:
            ValueError: If value not a valid number.
        """
        self._std_metric = val

    @property
    def std_min(self) -> Optional[float]:
        """*std_min* Get the standardized minimum range value.

        Returns:
            Optional[float]: standardized minimum range value.
        """
        return self._std_min

    @std_min.setter
    def std_min(self, val: Optional[float]) -> None:
        """*std_min* Sets the standardized minimum range value.

        Args:
            val (Optional[float]): standardized minimum range value.

        Raises:
            ValueError: If value not a valid number.
            ValueError: If value is greater than std_max.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized minimum must be a number")

        # Check if both values exist before comparing
        if val is not None and self._std_max is not None and val > self._std_max:
            _msg = f"Standard minimum {val} cannot be greater"
            _msg += f" than standard maximum {self._std_max}."
            raise ValueError(_msg)

        self._std_min = val

        # TODO reassert this code later, seems redundant with _prepare_dims()
        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_max(self) -> Optional[float]:
        """*std_max* Get the standardized maximum range value.

        Returns:
            Optional[float]: standardized maximum range value.
        """
        return self._std_max

    @std_max.setter
    def std_max(self, val: Optional[float]) -> None:
        """*std_max* Sets the standardized maximum range value.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is less than std_min.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized maximum must be a number")

        # Check if both values exist before comparing
        if val is not None and self._std_min is not None and val < self._std_min:
            _msg = f"Standard maximum {val} cannot be less"
            _msg += f" than standard minimum {self._std_min}."
            raise ValueError(_msg)

        self._std_max = val

        # TODO reassert this code later, seems redundant with _prepare_dims()
        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_mean(self) -> Optional[float]:
        """*std_mean* Get standardized mean value.

        Returns:
            Optional[float]: standardized mean.
        """
        return self._std_mean

    @std_mean.setter
    def std_mean(self, val: Optional[float]) -> None:
        """*std_mean* Sets the standardized mean value.

        Args:
            val (Optional[float]): standardized mean value.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is outside std_min-std_max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized mean must be a number")

        # Only validate range if val is not None
        if val is not None:
            low = (self._std_min is not None and val < self._std_min)
            high = (self._std_max is not None and val > self._std_max)

            if low or high:
                _msg = f"Standard mean {val} "
                _msg += f"must be between {self._std_min} and {self._std_max}."
                raise ValueError(_msg)

        self._std_mean = val

    @property
    def std_dev(self) -> Optional[float]:
        """*std_dev* Get standardized standard deviation.

        Returns:
            Optional[float]: Standardized standard deviation.
        """
        return self._std_dev

    @std_dev.setter
    def std_dev(self, val: Optional[float]) -> None:
        """*std_dev* Sets the standardized standard deviation.

        Args:
            val (Optional[float]): Standardized standard deviation.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is outside std_min-std_max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(
                "Standardized standard deviation must be a number")

        # Standard deviation should be non-negative
        if val is not None and val < 0:
            raise ValueError(f"Standard deviation {val} cannot be negative.")

        self._std_dev = val

    @property
    def step(self) -> Optional[float]:
        """*step* Get standardized step size.

        Returns:
            Optional[float]: Step size (always standardized).
        """
        return self._step

    @step.setter
    def step(self, val: Optional[float]) -> None:
        """*step* Set standardized step size.

        Args:
            val (Optional[float]): Step size (always standardized).

        Raises:
            ValueError: If step is not a valid number.
            ValueError: If step is zero.
            ValueError: If step is greater than range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Step must be a number.")

        if val == 0:
            raise ValueError("Step cannot be zero.")

        # Validate step against range (only if min/max are set)
        if val is not None and self._std_min is not None and self._std_max is not None:
            range_size = self._std_max - self._std_min
            if val >= range_size:
                _msg = f"Step {val} must be less than range: {range_size}."
                raise ValueError(_msg)

        self._step = val

        # TODO reassert this code later, seems redundant with _prepare_dims()
        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_range(self) -> np.ndarray:
        """*std_range* Get standardized range array.

        Returns:
            np.ndarray: Range array for range (always standardized).
        """
        return self._std_range

    @std_range.setter
    def std_range(self, val: Optional[np.ndarray]) -> None:
        """*std_range* Set standardized range array.

        Args:
            val (Optional[np.ndarray]): Data array for range (always standardized).

        Raises:
            ValueError: If value is not a numpy array.
        """
        # TODO reassert this code later, seems redundant with _prepare_dims()
        if val is None:
            # Generate range from min, max, step
            if all([self._std_min is not None,
                    self._std_max is not None,
                    self._step is not None]):
                self._std_range = np.arange(self._std_min,
                                            self._std_max,
                                            self._step)

        # TODO check this latter, might be a hindrance
        elif not isinstance(val, np.ndarray):
            _msg = f"Range must be a numpy array, got {type(val)}"
            raise ValueError(_msg)

        else:
            self._std_range = val

    def clear(self) -> None:
        """*clear()* Reset numerical attributes to default values.

        Resets all value ranges and step size.
        """
        self._min = None
        self._max = None
        self._mean = None
        self._std_min = None
        self._std_max = None
        self._std_mean = None
        self._step = None
        self._std_range = np.array([])
