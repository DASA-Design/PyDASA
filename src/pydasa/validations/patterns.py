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

# LaTeX Patterns
# Allow valid LaTeX strings starting with a backslash or alphanumeric strings
LATEX_RE: str = r"\\?[a-zA-Z]+(?:_\{\d+\})?"
"""
LaTeX regex pattern to match LaTeX symbols (e.g., '\\alpha', '\\beta_{1}') in *PyDASA*.
"""

DFLT_POW_RE: str = r"\-?\d+"
"""
Default regex to match FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""
