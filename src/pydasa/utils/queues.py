# -*- coding: utf-8 -*-
"""
Configuration module for...

# FIXME: adjust documentation to match the actual implementation.
# TODO: Add description of the module.
# TODO: check Q-Model formula and consistency with the theory.

*IMPORTANT:* Based on the theory from:
    # TODO add proper references!!!

"""
# native python modules
# forward references + postpone eval type hints
from __future__ import annotations
# dataclasses
from dataclasses import dataclass, field

# data types
from typing import Any, Dict, Optional

# indicate it is an abstract base class
from abc import ABC, abstractmethod
# TODO: check if numpy is needed
# import numpy as np
# import math

# import custom factorial (gamma) function
from pydasa.utils.helpers import gfactorial
# from pydasa.utils.helpers import mad_hash


def Queue(_lambda: float,
          miu: float,
          n_servers: int = 1,
          kapacity: Optional[int] = None) -> BasicQueue:
    """*Queue()* factory function to create different queue models.

    NOTE: some variable names start with underscore (_) to avoid conflict with Python keywords.

    Args:
        _lambda (float): Arrival rate (λ) of the queue.
        miu (float): Service rate (μ) of the queue.
        n_servers (int, optional): Number of servers (s). Defaults to 1.
        kapacity (Optional[int], optional): Maximum capacity (K) of the queue. Defaults to None.

    Raises:
        NotImplementedError: If the queue configuration is not supported.

    Returns:
        BasicQueue: An instance of a specific queue model (based on the abstract basic model).
    """
    _queue = None
    # Single server, infinite capacity
    if n_servers == 1 and kapacity is None:
        _queue = QueueMM1(_lambda,
                          miu)

    # Multi-server, infinite capacity
    elif n_servers > 1 and kapacity is None:
        _queue = QueueMMs(_lambda,
                          miu,
                          n_servers)

    # Single server, finite capacity
    elif n_servers == 1 and kapacity is not None:
        _queue = QueueMM1K(_lambda,
                           miu,
                           kapacity)

    # Multi-server, finite capacity
    elif n_servers > 1 and kapacity is not None:
        _queue = QueueMMsK(_lambda,
                           miu,
                           n_servers,
                           kapacity)

    # Add more conditions for other queue types. e.g., M/G/1, G/G/1, etc.
    # TODO: Implement additional queue models
    # otherwise, raise an error
    else:
        _msg = f"Unsupported queue configuration: {n_servers} "
        _msg += f"servers, {kapacity} max capacity"
        raise NotImplementedError(_msg)
    return _queue


@dataclass
class BasicQueue(ABC):
    """**BasicQueue** is an abstract base class for queueing theory models.

    Attributes:
        Input parameters:
        _lambda (float): Arrival rate (λ: lambda).
        miu (float): Service rate (μ: miu).
        n_servers (int): Number of servers (s: servers).
        kapacity (Optional[int]): Maximum capacity (K: capacity).

        # Output parameters:
        rho (float): Server utilization (ρ: rho).
        avg_len (float): L, or mean number of requests in the system.
        avg_len_q (float): Lq, or mean number of requests in queue.
        avg_wait (float): W, or mean time a request spends in the system.
        avg_wait_q (float): Wq, or mean waiting time in queue.
    """

    # :attr: _lambda
    _lambda: float
    """Arrival rate (λ: lambda)."""

    # :attr: miu
    miu: float
    """Service rate (μ: miu)."""

    # :attr: n_servers
    n_servers: int = 1
    """Number of servers (s: servers)."""

    # :attr: rho
    rho: float = field(default=0.0, init=False)
    """Server utilization (ρ: rho)."""

    # :attr: kapacity
    kapacity: Optional[int] = None
    """Maximum capacity (K: capacity)."""

    # :attr: avg_len
    avg_len: float = field(default=0.0, init=False)
    """Average length of elements in the system (L: mean number of requests in the system with the Little's Law)."""

    # :attr: avg_len_q
    avg_len_q: float = field(default=0.0, init=False)
    """Average length of elements in the queue (Lq: mean number of requests in queue with the Little's Law)."""

    # :attr: avg_wait
    avg_wait: float = field(default=0.0, init=False)
    """Average time a request spends in the system (W: mean time in system with the Little's Law)."""

    # :attr: avg_wait_q
    avg_wait_q: float = field(default=0.0, init=False)
    """Average time a request spends waiting in the queue (Wq: mean waiting time in queue with the Little's Law)."""

    def __post_init__(self):
        """*__post_init__()* Post-initialization processing to validate parameters and calculate metrics.
        """
        self._validate_basic_params()
        self._validate_params()
        self._calculate_metrics()

    def _validate_basic_params(self) -> None:
        """*_validate_basic_params()* Validates basic parameters common to all queueing models.

        Raises:
            ValueError: If arrival rate is non-positive.
            ValueError: If service rate is non-positive.
            ValueError: If number of servers is non-positive.
        """

        if self._lambda <= 0:
            raise ValueError("Arrival rate must be positive.")
        if self.miu <= 0:
            raise ValueError("Service rate must be positive.")
        if self.n_servers < 1:
            raise ValueError("Number of servers must be positive.")

    @abstractmethod
    def _validate_params(self) -> None:
        """*_validate_params()* Validates parameters specific to each queueing model.
        """
        pass

    @abstractmethod
    def _calculate_metrics(self) -> None:
        """*_calculate_metrics()* Calculates analytical metrics for the queueing model.
        """
        pass

    @abstractmethod
    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* Calculates P(n), or the probability of having n requests in the system.

        Args:
            n (int): Number of requests in the system.

        Returns:
            float: Probability of having n requests in the system.
        """
        pass

    @abstractmethod
    def is_stable(self) -> bool:
        """*is_stable()* Checks if the queueing system is stable.

        Returns:
            bool: True if the system is stable, False otherwise.
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """*get_metrics()* Returns a summary of the queueing system's metrics.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated metrics with the following keys:
                - 'L': Average number requests inside the system.
                - 'Lq': Average number requests in queue.
                - 'W': Average request time in the system.
                - 'Wq': Average request time in queue.
                - 'rho': Server utilization.
        """
        return {
            "L": self.avg_len,
            "Lq": self.avg_len_q,
            "W": self.avg_wait,
            "Wq": self.avg_wait_q,
            "rho": self.rho,
        }

    def __str__(self) -> str:
        """*__str__()* String representation of the queue model.
        Returns:
            str: Formatted string with queue model details and metrics.
        """
        # Create header with class name
        output = [f"{self.__class__.__name__}("]

        # Add basic parameters
        params = [
            f"\tλ={self._lambda}",
            f"\tμ={self.miu}",
            f"\tservers={self.n_servers}"
        ]
        output.extend(params)
        if self.kapacity is not None:
            output.append(f"\tcapacity={self.kapacity}")

        # Add stability status
        status = f"\tStatus: {'STABLE' if self.is_stable() else 'UNSTABLE'}"
        output.append(status)

        # Add metrics with formatting
        metrics = self.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                output.append(f"\t{key}={value:.6f}")
            else:
                output.append(f"\t{key}={value}")
        output.append(")")
        # Join all lines with newlines
        return ",\n".join(output)

    def __repr__(self) -> str:
        """*__repr__()* Detailed string representation.

        Returns:
            str: String representation.
        """
        return self.__str__()


@dataclass
class QueueMM1(BasicQueue):
    """**QueueMM1** represents an M/M/1 queue system (1 server, infinite capacity).

    Args:
        BasicQueue (ABC, dataclass): Abstract base class for queueing theory models.
    """

    def _validate_params(self) -> None:
        """*_validate_params()* Validates the parameters for the M/M/1 queue.

        Raises:
            ValueError: If the number of servers is not 1.
            ValueError: If the capacity is not infinite.
            ValueError: If the system is unstable (λ ≥ μ).
        """

        if self.n_servers != 1:
            raise ValueError("M/M/1 must have exactly 1 server.")
        if self.kapacity is not None:
            raise ValueError("M/M/1 assumes infinite capacity.")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ μ).")

    def is_stable(self) -> bool:
        """*is_stable()* Checks if the queueing system is stable.

        Returns:
            bool: True if the system is stable, False otherwise.
        """

        return self._lambda < self.miu

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* Calculates the performance metrics for the M/M/1 queue.

        The model metrics are:
            - ρ (rho): Server utilization.
            - L (avg_len): Average number of requests in the system.
            - Lq (avg_len_q): Average number of requests in the queue.
            - W (avg_wait): Average time a request spends in the system.
            - Wq (avg_wait_q): Average time a request spends in the queue.
        """
        # Calculate utilization
        self.rho = self._lambda / self.miu

        # Calculate average number of requests in the system
        self.avg_len = self.rho / (1 - self.rho)

        # Calculate average number of requests in the queue
        self.avg_len_q = self.rho ** 2 / (1 - self.rho)

        # Calculate average time a request spends in the system
        self.avg_wait = self.avg_len / self._lambda

        # Calculate average time a request spends in the queue
        self.avg_wait_q = self.avg_len_q / self._lambda

    def _get_prob_zero(self) -> float:
        """*_get_prob_zero()* Calculates P(0) or the probability of having 0 requests in the system for M/M/1 model.

        Returns:
            float: The probability of having 0 requests in the system.
        """
        p_zero = 1 - self.rho
        return p_zero

    def _get_prob_n(self, n: int) -> float:
        """*_get_prob_n()* calculates P(n), or the probability of having n requests in the system for M/M/1.

        Args:
            n (int): The number of requests in the system.

        Raises:
            ValueError: If the system is unstable.

        Returns:
            float: The probability of having n requests in the system.
        """

        if not self.is_stable():
            _msg = f"Unstable System!, {type(self).__name__}. "
            _msg += "Server utilization (ρ: rho) must be < 1 for M/M/1"
            raise ValueError(_msg)

        if n < 0:
            return 0.0
        p_n = (1 - self.rho) * (self.rho ** n)
        return p_n


@dataclass
class QueueMMs(BasicQueue):
    """**QueueMMs** represents an M/M/s queue system (Multi-server, infinite capacity).

    Args:
        BasicQueue (ABC, dataclass): Abstract base class for queueing theory models.
    """
    def validate_parameters(self) -> None:
        """*validate_parameters()* Validates the parameters for the M/M/s model.

        Raises:
            ValueError: If the number of servers is less than 1.
            ValueError: If the capacity is not infinite.
            ValueError: If the system is unstable (λ ≥ c x μ).
        """
        if self.n_servers < 1:
            raise ValueError("M/M/s requires at least one server.")
        if self.kapacity is not None:
            raise ValueError("M/M/s assumes infinite capacity.")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ s * μ).")

    def is_stable(self) -> bool:
        """*is_stable()* Checks if the queueing system is stable.

        Returns:
            bool: True if the system is stable, False otherwise.
        """

        return self._lambda < (self.n_servers * self.miu)

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* Calculates the performance metrics for the M/M/s queue.

        The model metrics are:
            - ρ (rho): Server utilization.
            - L (avg_len): Average number of requests in the system.
            - Lq (avg_len_q): Average number of requests in the queue.
            - W (avg_wait): Average time a request spends in the system.
            - Wq (avg_wait_q): Average time a request spends in the queue.
        """
        # Calculate utilization (rho)
        self.rho = self._lambda / (self.n_servers * self.miu)

        # Calculate traffic intensity (tau)
        tau = self._lambda / self.miu

        # Calculate the probability of having 0 requests in the system
        _p_zero = self._get_prob_zero()
        numerator = (_p_zero * (tau ** self.n_servers) * self.rho)
        denominator = (gfactorial(self.n_servers) * ((1 - self.rho) ** 2))
        self.avg_len_q = numerator / denominator

        # Calculate the average number of requests in the system
        self.avg_len = self.avg_len_q + tau

        # Calculate the average time spent in the queue
        self.avg_wait_q = self.avg_len_q / self._lambda

        # Calculate the average time spent in the system
        self.avg_wait = self.avg_wait_q + 1 / self.miu

    def _get_prob_zero(self) -> float:
        """*_get_prob_zero()* Calculates P(0) or the probability of having 0 requests in the system for M/M/s model.

        Returns:
            float: The probability of having 0 requests in the system.
        """
        # Calculate the traffic intensity (tau)
        tau = self._lambda / self.miu

        # calculate first coefficient
        coef1 = sum((tau ** i) / gfactorial(i) for i in range(self.n_servers))

        # calculate second coefficient
        numerator = (tau ** self.n_servers)
        denominator = gfactorial(self.n_servers) * (1 - self.rho)
        coef2 = numerator / denominator

        # calculate the probability of having 0 requests in the system
        p_zero = 1 / (coef1 + coef2)
        return p_zero

    def _get_prob_n(self, n: int) -> float:
        """*_get_prob_n()* calculates P(n), or the probability of having n requests in the system for M/M/s.

        Args:
            n (int): The number of requests in the system.

        Raises:
            ValueError: If the system is unstable.

        Returns:
            float: The probability of having n requests in the system.
        """
        # default value, for
        p_n = -1.0
        # if request is less than 0
        if n < 0:
            # return default value
            return p_n

        # calculate traffic intensity (tau)
        tau = self._lambda / self.miu
        # calculate the probability of having 0 requests in the system
        _p_zero = self._get_prob_zero()

        # calculate the probability of having n requests in the system
        numerator = (tau ** n)
        denominator = 1.0
        # if there are less requests than servers
        if n < self.n_servers:
            denominator = gfactorial(n)

        # otherwise, there are more requests than servers
        else:
            power = (self.n_servers ** (n - self.n_servers))
            denominator = (gfactorial(self.n_servers) * power)

        # finishing up calculations
        p_n = (numerator / denominator) * _p_zero
        return p_n


@dataclass
class QueueMM1K(BasicQueue):
    """**QueueMM1K** Represents an M/M/1/K queue system with finite capacity 'k' and one server.

    Args:
        BasicQueue (ABC, dataclass): Abstract base class for queueing theory models.
    """

    # TODO aqui voy!!! revisar las equaciones
    def validate_parameters(self) -> None:
        """*validate_parameters()* Validates the parameters for the M/M/1/k model.

        Raises:
            ValueError: If the number of servers is not 1.
            ValueError: If the capacity is not positive.
        """
        if self.n_servers != 1:
            raise ValueError("M/M/1/k requires exactly 1 server.")
        if self.kapacity is None or self.kapacity < 1:
            raise ValueError("M/M/1/k requires a positive finite capacity 'k'.")

    def is_stable(self) -> bool:
        """*is_stable()* Checks if the queueing system is stable.

        Returns:
            bool: True if the system is stable, False otherwise.
        """
        return self._lambda <= self.miu

    def calculate_metrics(self) -> None:

        # Calculate the utilization (rho) == traffic intensity (tau)
        _rho = self._lambda / self.miu
        # Calculate the probability of having max capacity
        _p_kapacity = self._get_prob_n(self.kapacity)
        # Calculate the effective arrival rate
        _lambda_eff = self._lambda * (1 - _p_kapacity)
        # Calculate the server utilization (rho)
        self.rho = _rho * (1 - _p_kapacity)

        # if utilization (rho) is less than 1
        if _rho < 1.0:
            # Calculate average number of requests in the system
            coef1 = (1 - _rho) / (1 - _rho ** (self.kapacity + 1))
            numerator = (self.kapacity + 1) * self.rho ** (self.kapacity + 1)
            denominator = (1 - self.rho ** (self.kapacity + 1))
            coef2 = numerator / denominator
            self.avg_len = coef1 - coef2

            # Calculate average number of requests in the queue
            self.avg_len_q = self.avg_len - (1 - self._get_prob_zero())

            # Calculate average time spent in the system
            self.avg_time = self.avg_len / _lambda_eff

            # Calculate average time spent in the queue
            self.avg_time_q = self.avg_len_q / _lambda_eff

        # if utilization (rho) is equal to 1, saturation occurs
        elif _rho == 1.0:
            self.avg_len = self.kapacity / 2

            # Calculate average number of requests in the queue
            self.avg_len_q = self.avg_len - 1

            # Calculate average time spent in the system
            self.avg_time = self.avg_len / _lambda_eff

            # Calculate average time spent in the queue
            self.avg_time_q = self.avg_len_q / _lambda_eff
        # TODO aki voy!!!
        '''
        _rho = self._lambda / self.miu
        _p_kapacity = self.calculate_prob_n(self.kapacity)
        _lambda_eff = self._lambda * (1 - _p_kapacity)

        self.rho = _rho * (1 - _p_kapacity)
        self.avg_len = sum(n * self.calculate_prob_n(n)
                           for n in range(self.kapacity + 1))
        self.avg_len_q = sum(max(0, n - 1) * self.calculate_prob_n(n)
                             for n in range(self.kapacity + 1))

        if _lambda_eff > 0:
            self.avg_wait = self.avg_len / _lambda_eff
            self.avg_wait_q = self.avg_len_q / _lambda_eff
        else:
            self.avg_wait = 0
            self.avg_wait_q = 0
        '''

    def _get_prob_zero(self) -> float:
        """*_get_prob_zero()* Calculates P(0) or the probability of having 0 requests in the system for M/M/1/k model.

        NOTE: Unnecessary function but was weird not to have it.

        Returns:
            float: The probability of having 0 requests in the system.
        """
        # Calculate the utilization (rho) == traffic intensity (tau)
        _rho = self._lambda / self.miu
        # default values for checking errors
        p_zero = -1.0

        # if utilization (rho) is less than 1
        if _rho < 1.0:
            p_zero = (1 - _rho) / (1 - _rho ** (self.kapacity + 1))

        # if utilization (rho) is equal to 1, saturation occurs
        elif _rho == 1.0:
            p_zero = 1 / (self.kapacity + 1)

        # return the calculated probability of having 0 requests in the system
        return p_zero

    def _get_prob_n(self, n: int) -> float:
        """*get_prob_n()* Calculates P(n) or the probability of having n requests in the system for M/M/1/k model.

        Args:
            n (int): The number of requests.

        Returns:
            float: The probability of having n requests in the system.
        """
        # Calculate the utilization (rho) == traffic intensity (tau)
        _rho = self._lambda / self.miu
        # default values for checking errors
        p_zero = -1.0

        # if utilization (rho) is less than 1
        if _rho < 1.0:
            numerator = (1 - _rho) * (_rho ** self.kapacity)
            denominator = (1 - _rho ** (self.kapacity + 1))
            p_zero = numerator / denominator

        # if utilization (rho) is equal to 1, saturation occurs
        elif _rho == 1.0:
            p_zero = 1 / (self.kapacity + 1)

        # return the calculated probability of having 0 requests in the system
        return p_zero


@dataclass
class QueueMMsK(BasicQueue):
    """M/M/c/K queueing system: c servers, finite capacity K"""

    def validate_parameters(self) -> None:
        """Validations specific to M/M/c/L"""
        if self.n_servers < 1:
            raise ValueError("M/M/c/L requires at least one server")
        if self.kapacity is None or self.kapacity < self.n_servers:
            raise ValueError(
                "M/M/c/L requires finite capacity L >= c (number of servers)")

    def is_stable(self) -> bool:
        """Checks if the system is stable - M/M/c/L is always stable due to finite capacity"""
        return self._lambda > 0 and self.miu > 0

    def calculate_p0(self) -> float:
        """Calculates P(0) for M/M/c/L"""
        _rho = self._lambda / self.miu

        _sum1 = sum((_rho**n) / gfactorial(n)
                    for n in range(self.n_servers))

        if self._lambda == self.n_servers * self.miu:
            _sum2 = ((_rho**self.n_servers) / gfactorial(self.n_servers)
                     ) * (self.kapacity - self.n_servers + 1)
        else:
            rho = _rho / self.n_servers
            _sum2 = ((_rho**self.n_servers) / gfactorial(self.n_servers)) * ((1 - rho**(self.kapacity - self.n_servers + 1)) / (1 - rho))

        return 1 / (_sum1 + _sum2)

    def calculate_prob_n(self, n: int) -> float:
        """Calculate P(n) for n=0,...,L in M/M/c/L"""
        if n < 0 or n > self.kapacity:
            return 0.0

        _rho = self._lambda / self.miu
        _p_zero = self._get_prob_zero()

        if n < self.n_servers:
            return ((_rho**n) / gfactorial(n)) * _p_zero
        else:
            return ((_rho**n) / (gfactorial(self.n_servers) * (self.n_servers**(n - self.n_servers)))) * _p_zero

    def calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/c/L"""
        _p_kapacity = self.calculate_prob_n(self.kapacity)
        _lambda_eff = self._lambda * (1 - _p_kapacity)
        self.avg_len = sum(n * self.calculate_prob_n(n) for n in range(self.kapacity + 1))
        self.avg_len_q = sum(max(0, n - self.n_servers) * self.calculate_prob_n(n) for n in range(self.kapacity + 1))
        server_busy_prob = sum(min(n, self.n_servers) * self.calculate_prob_n(n) for n in range(self.kapacity + 1))
        self.rho = server_busy_prob / self.n_servers
        if _lambda_eff > 0:
            self.avg_wait = self.avg_len / _lambda_eff
            self.avg_wait_q = self.avg_len_q / _lambda_eff
        else:
            self.avg_wait = 0
            self.avg_wait_q = 0
