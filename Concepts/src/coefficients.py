# python native modules
# import re
from dataclasses import dataclass, field
from typing import Generic, List, Optional
# , Callable,

# python third-party modules
# import sympy as sp
import numpy as np

# custom modules
# from src.utils import FUND_DIM_UNIT_RE
# from src.utils import DIGIT_POW_RE
# from src.utils import BASIC_DIGIT_POW_RE
from src.utils import T
# from src.utils import FDU
# from src.utils import EXP_PATTERN_STR
# from src.utils import FDU_PATTERN_STR


@dataclass
class PiCoefficient(Generic[T]):
    """PiCoefficient _summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # TODO add docstring
    # name of the parameter
    name: str = ""
    # latex symbol for the parameter
    symbol: str = ""
    # optional string description of the parameter
    description: str = ""
    # expression for the dimensionless coefficient
    pi_expr: str = ""
    # private index in the dimensional matrix
    _idx: int = -1
    # list of parameters used in the dimensionless coefficient
    _pi_param_lt: List[str] = field(default_factory=list)
    # list of exponents used in the dimensionless coefficient
    _pi_exp_lt: List[int] = field(default_factory=list)
    # diagonal solved dimensional matrix
    _rref_dim_matrix: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([]))
    _pivot_lt: Optional[List[int]] = field(default_factory=list)
