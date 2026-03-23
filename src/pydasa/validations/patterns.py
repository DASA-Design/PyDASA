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
# LATEX_RE: str = r"\\?[a-zA-Z]+(?:_\{[a-zA-Z0-9_]+\})?"
# :attr: LATEX_RE
LATEX_RE: str = r"\\?[a-zA-Z]+(?:_\{[a-zA-Z0-9_]+\})?"
"""
LaTeX regex pattern to match LaTeX symbols (e.g., '\\alpha') with optional subscripts (e.g., '\\mu_{1}', '\\lambda_{s}') in *PyDASA*.
"""

# :attr: LATEX_VAR_TOKEN_RE
# Brace nesting helpers – each level wraps the previous one so that
# the overall pattern supports up to 5 levels of nested braces inside
# subscripts (e.g. M_{a*(c*t_{R_{P*(A*(C*S))}})}).
_BRACE_L0: str = r"[^{}]*"
_BRACE_L1: str = r"(?:[^{}]|\{" + _BRACE_L0 + r"\})*"
_BRACE_L2: str = r"(?:[^{}]|\{" + _BRACE_L1 + r"\})*"
_BRACE_L3: str = r"(?:[^{}]|\{" + _BRACE_L2 + r"\})*"
_BRACE_L4: str = r"(?:[^{}]|\{" + _BRACE_L3 + r"\})*"

LATEX_VAR_TOKEN_RE: str = (
    r"(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)"
    r"(?:_(?:[A-Za-z0-9]+|\{" + _BRACE_L4 + r"\}))?"
)
"""
Regex pattern to match LaTeX-like variable tokens with optional subscripts,
supporting up to 5 levels of nested braces inside subscripts.

Examples:
    '\\alpha', '\\mu_{1}', 'M_{buf_{AS}}', 'M_{a*(c*t_{R_{P*(A*(C*S))}})}'
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

# parse expression for numeric constants
# :attr: NUM_CONST_RE
NUM_CONST_RE: str = r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$"
"""
Regex pattern to match numeric constants (e.g., '3.14', '-2.5e3') in *PyDASA*.
"""

# Parse expression for basic operations
# NOTE: OG REGEX
# parts = re.split(r"(\*|/|\^)", expr)
# :attr: BASIC_OPS_RE
BASIC_OPS_RE = r"(\*|\/|\+|\-|\^)"
"""
Regex pattern to split expressions by basic operations (*, /, ^, +, -) in *PyDASA*.
"""

# Global vars for special Latex symbos and functions to ignore
# :attr: IGNORE_EXPR
IGNORE_EXPR = {
    "\\frac",
    "\\sqrt",
    "\\sin",
    "\\cos",
    "\\tan",
    "\\log",
    "\\exp"
}
"""
Set of LaTeX expressions to ignore during parsing in *PyDASA*.
"""
