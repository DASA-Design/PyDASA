from dataclasses import dataclass, field
from typing import Optional
from scipy.special import gamma


@dataclass
class QueueModel:
    """
    Simulates queue models for M/M/1, M/M/1/K, M/M/c, and M/M/c/K.

    Attributes:
        arrival_rate (float): Average arrival rate (λ).
        service_rate (float): Average service rate (μ).
        servers (int): Number of servers (c).
        capacity (Optional[int]): Maximum capacity of the system (K). None for infinite capacity.
        departure_rate (float): Average departure rate (min(arrival_rate, servers * service_rate)).
    """
    arrival_rate: float = 0.0
    service_rate: float = 0.0
    servers: int = 1
    capacity: Optional[int] = None
    departure_rate: float = field(init=False)

    def __post_init__(self):
        """Initialize the QueueModel and calculate the departure rate."""
        self.departure_rate = min(self.arrival_rate,
                                  self.servers * self.service_rate)

    def utilization(self) -> float:
        """Calculate the utilization factor (ρ)."""
        return self.arrival_rate / (self.servers * self.service_rate)

    def flow_rate(self) -> float:
        """Calculate the flow rate (effective throughput)."""
        if self.capacity is None:
            return self.arrival_rate
        return self.arrival_rate * (1 - self.utilization() ** self.capacity)

    def mm1(self) -> dict:
        """Simulate M/M/1 queue."""
        rho = self.utilization()
        if rho >= 1:
            raise ValueError("System is unstable (ρ >= 1).")
        return {
            "L": rho / (1 - rho),
            "W": 1 / (self.service_rate - self.arrival_rate),
            "rho": rho,
            "flow_rate": self.flow_rate(),
        }

    def mm1k(self) -> dict:
        """Simulate M/M/1/K queue."""
        rho = self.utilization()
        if self.capacity is None:
            raise ValueError("Capacity (K) must be specified for M/M/1/K.")
        P0 = (1 - rho) / (1 - rho ** (self.capacity + 1))
        L = rho * (1 - (self.capacity + 1) * rho ** self.capacity + self.capacity * rho ** (self.capacity + 1)) / (
            (1 - rho) * (1 - rho ** (self.capacity + 1))
        )
        return {
            "L": L,
            "W": L / (self.arrival_rate * (1 - rho ** self.capacity)),
            "rho": rho,
            "P0": P0,
            "flow_rate": self.flow_rate(),
        }

    def mmc(self) -> dict:
        """Simulate M/M/c queue."""
        rho = self.utilization()
        if rho >= 1:
            raise ValueError("System is unstable (ρ >= 1).")
        P0 = self._calc_p0_mmc(rho)
        Lq = (P0 * (self.servers * rho) ** self.servers * rho) / (
            self._factorial(self.servers) * (1 - rho) ** 2
        )
        L = Lq + self.servers * rho
        return {
            "L": L,
            "W": L / self.arrival_rate,
            "rho": rho,
            "P0": P0,
            "flow_rate": self.flow_rate(),
        }

    def mmck(self) -> dict:
        """Simulate M/M/c/K queue."""
        rho = self.utilization()
        if self.capacity is None:
            raise ValueError("Capacity (K) must be specified for M/M/c/K.")
        P0 = self._calc_p0_mmck(rho)
        L = self._calc_l_mmck(P0, rho)
        return {
            "L": L,
            "W": L / (self.arrival_rate * (1 - self._prob_k(P0, rho))),
            "rho": rho,
            "P0": P0,
            "flow_rate": self.flow_rate(),
        }

    def _calc_p0_mmc(self, rho: float) -> float:
        """Calculate P0 for M/M/c."""
        sum_terms = sum((self.servers * rho) ** n / self._factorial(n)
                        for n in range(self.servers))
        last_term = (self.servers * rho) ** self.servers / \
            (self._factorial(self.servers) * (1 - rho))
        return 1 / (sum_terms + last_term)

    def _calc_p0_mmck(self, rho: float) -> float:
        """Calculate P0 for M/M/c/K."""
        sum_terms = sum((self.servers * rho) ** n / self._factorial(n)
                        for n in range(self.servers))
        last_term = (self.servers * rho) ** self.servers * (1 - rho ** (self.capacity - self.servers + 1)) / (
            self._factorial(self.servers) * (1 - rho)
        )
        return 1 / (sum_terms + last_term)

    def _calc_l_mmck(self, P0: float, rho: float) -> float:
        """Calculate L for M/M/c/K."""
        Lq = (
            P0 * (self.servers * rho) ** self.servers * (1 - (self.capacity - self.servers + 1) * rho ** (self.capacity - self.servers) + self.capacity * rho ** (self.capacity - self.servers + 1)) / (self._factorial(self.servers) * (1 - rho) ** 2)
        )
        return Lq + self.servers * rho

    def _prob_k(self, P0: float, rho: float) -> float:
        """Calculate the probability of having K customers in the system."""
        return (P0 * (self.servers * rho) ** self.capacity) / (self._factorial(self.servers) * (1 - rho))

    @staticmethod
    def _factorial(n: float) -> float:
        """Calculate the factorial of a number, supporting non-integer values using the Gamma function."""
        return gamma(n + 1)


def create_queue(arrival_rate: float,
                 service_rate: float,
                 servers: int = 1,
                 capacity: Optional[int] = None) -> QueueModel:
    """
    Factory function to create a QueueModel instance.

    Args:
        arrival_rate (float): Average arrival rate (λ).
        service_rate (float): Average service rate (μ).
        servers (int): Number of servers (c). Default is 1.
        capacity (Optional[int]): Maximum capacity of the system (K). Default is None (infinite capacity).

    Returns:
        QueueModel: An instance of the QueueModel class.
    """
    return QueueModel(arrival_rate, service_rate, servers, capacity)
