# -*- coding: utf-8 -*-
"""
Module patterns.py
===========================================

Regex patterns for validation and parsing in PyDASA.

Contains:
    - LaTeX validation patterns
    - FDU (Fundamental Dimensional Unit) matching patterns
    - Default and working pattern sets for dimensional analysis
"""

from pydasa.core.constants import DFLT_FDU_PREC_LT

# LaTeX Patterns
# TODO clean the regex when tests are finished
# Allow valid LaTeX strings starting with a backslash or alphanumeric strings
LATEX_RE: str = r"\\?[a-zA-Z]+(?:_\{\d+\})?"
"""
LaTeX regex pattern to match LaTeX symbols (e.g., '\\alpha', '\\beta_{1}') in *PyDASA*.
"""

DFLT_POW_RE: str = r"\-?\d+"
"""
Default regex to match FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# Default FDU Patterns
DFLT_FDU_RE: str = rf"^[{''.join(DFLT_FDU_PREC_LT)}](\^-?\d+)?(\*[{''.join(DFLT_FDU_PREC_LT)}](?:\^-?\d+)?)*$"
"""
Default regex pattern to match FDUs in *PyDASA* (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

DFLT_NO_POW_RE: str = rf"[{''.join(DFLT_FDU_PREC_LT)}](?!\^)"
"""
Default regex to match FDUs without exponents (e.g., 'M*L*T' instead of 'M*L^-1*T^-2').
"""

DFLT_FDU_SYM_RE: str = rf"[{''.join(DFLT_FDU_PREC_LT)}]"
"""
Default regex to match FDU symbols in *PyDASA* (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
"""

# Working FDU Patterns (mutable, can be customized)
WKNG_FDU_PREC_LT: list = DFLT_FDU_PREC_LT.copy()
"""
Working FDUs precedence list for the dimensional matrix, allowing custom dimensions (e.g., 'DFLT_FDU_PREC_LT = [D, T, C]').
"""

WKNG_FDU_RE: str = DFLT_FDU_RE
"""
Working regex to match FDUs in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

WKNG_POW_RE: str = DFLT_POW_RE
"""
Working regex to match FDUs with exponents in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

WKNG_NO_POW_RE: str = DFLT_NO_POW_RE
"""
Working regex to match FDUs without exponents (e.g., 'T*D' instead of 'T^2*D^-1').
"""

WKNG_FDU_SYM_RE: str = DFLT_FDU_SYM_RE
"""
Working regex to match FDU symbols in *PyDASA* (e.g., 'T^(1)*D^(-1)' to 'D**(-1)*T**(2)').
"""
