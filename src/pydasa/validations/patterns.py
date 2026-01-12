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


# NOTE: OG REGEX!
# LATEX_RE: str = r"([a-zA-Z]+)(?:_\{\d+\})?"
# :attr: LATEX_RE
LATEX_RE: str = r"\\?[a-zA-Z]+(?:_\{\d+\})?"
"""
LaTeX regex pattern to match LaTeX symbols (e.g., '\\alpha', '\\beta_{1}') in *PyDASA*.
"""

# NOTE: OG REGEX!
# DFLT_POW_RE: str = r"\-?\d+"   # r'\^(-?\d+)'
# :attr: DFLT_POW_RE
DFLT_POW_RE: str = r"\-?\d+"
"""
Default regex to match FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# NOTE: OG REGEX!!
# coef_pattern = r"\\Pi_\{\d+\}"
# Parse expression for power operations first (e.g., \Pi_{0}**(-1))
# Pattern to match coefficient with optional power: \Pi_{n}**(exponent)
# :attr: COEF_RE
PI_COEF_RE: str = r"\\Pi_\{\d+\}"
"""
Regex pattern to match dimensionless coefficients in *PyDASA* (e.g, '\\Pi_{1}', '\\Pi_{2}').
"""

# NOTE: OG REGEXT!!!
# power_pattern = r"(\\Pi_\{\d+\})\s*\*\*\s*\(([^)]+)\)"
# Parse expression for power operations first (e.g., \Pi_{0}**(-1))
# Pattern to match coefficient with optional power: \Pi_{n}**(exponent)
# :attr: POW_RE
PI_POW_RE: str = r"(\\Pi_\{\d+\})\s*\*\*\s*\(([^)]+)\)"
"""
Regex pattern to match dimensionless coefficients with power operations in *PyDASA* (e.g., '\\Pi_{1}**(-2)').
"""

# Parse expression for basic operations
# NOTE: OG REGEX
# parts = re.split(r"(\*|/|\^)", expr)
BASIC_OPS_RE = r"(\*|/|\^|\+|\-)"
"""
Regex pattern to split expressions by basic operations (*, /, ^, +, -) in *PyDASA*.
"""
