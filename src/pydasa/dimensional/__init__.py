# -*- coding: utf-8 -*-
"""
Package dimensional
===========================================

Dimensional analysis constants and utilities for *PyDASA*.
"""

# populate the __init__.py to simplify the imports
from pydasa.dimensional.constants import (
    PHY_FDU_PREC_DT,
    COMPU_FDU_PREC_DT,
    SOFT_FDU_PREC_DT,
    DFLT_FDU_PREC_LT,
)

# Now sers can do:
# from pydasa.dimensional import PHY_FDU_PREC_DT
__all__ = [
    "PHY_FDU_PREC_DT",
    "COMPU_FDU_PREC_DT",
    "SOFT_FDU_PREC_DT",
    "DFLT_FDU_PREC_LT",
]
