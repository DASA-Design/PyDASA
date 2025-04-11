# -*- coding: utf-8 -*-
"""
Configuration module for Fundamental Dimensional Units (FDUs), Parameters, and Variables in *PyDASA*. Includes regex patterns for validating dimensions and the traditional FDU list.

It can handle both physical and digital dimensions, as well as working (custom or otherwise) dimensions through the use of the WKNG_* variables.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# python native modules
# from dataclasses import dataclass
# from typing import TypeVar

# global variables

# Fundamental Dimensional Units (FDU) precedence list
# :attr: DFLT_FDU_PREC_LT
DFLT_FDU_PREC_LT: list = ["L", "M", "T", "θ", "I", "N", "J"]
"""
List of FDUs in precedence order for the dimensional matrix. i.e: 'M*L^-1*T^-2'

Defines the row order in the dimensional matrix and validates parameter/variable dimensions in *PyDASA*. The traditional FDUs are:
    - length [L]: the distance between two points in space.
    - mass [M]: the amount of matter in an object.
    - time [T]: the duration of an event or the interval between two events.
    - temperature [θ]: the measure of the average kinetic energy of particles in a substance.
    - electric current [I]: the flow of electric charge in a circuit.
    - amount of substance [N]: the quantity of entities (atoms, molecules, etc.) in a sample.
    - luminous intensity [J]: the measure of the perceived power of light emitted by a source in a given direction.
"""

# regex pattern for matching FDUs
# :attr: DFLT_FDU_REGEX
DFLT_FDU_REGEX: str = rf"^[{''.join(DFLT_FDU_PREC_LT)}](\^-?\d+)?(\*[{''.join(DFLT_FDU_PREC_LT)}](?:\^-?\d+)?)*$"
# DFLT_FDU_REGEX: str = r"^[LMTθINJ](\^-?\d+)?(\*[LMTθINJ](?:\^-?\d+)?)*$"
"""
Default Regex pattern to match FDUs in *PyDASA*. i.e.: from 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)'.
"""

# regex pattern for matching dimensions with exponent
# :attr: DFLT_POW_REGEX
DFLT_POW_REGEX: str = r"\-?\d+"   # r'\^(-?\d+)'
"""
Default Regex pattern to match FDUs in *PyDASA* with exponents in the dimensions of parameters and variables. i.e.: 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)'.
"""

# regex pattern for matching dimensions WITHOUT exponent
# :attr: DFLT_NO_POW_REGEX
DFLT_NO_POW_REGEX: str = rf"[{''.join(DFLT_FDU_PREC_LT)}](?!\^)"
"""
Default Regex pattern to match FDUs without exponents in parameter or variable dimensions. i.e.: 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)'.
"""

# regex pattern for matching FDUs in Sympy symbolic processor
# :attr: DFLT_FDU_SYM_REGEX
DFLT_FDU_SYM_REGEX: str = rf"[{''.join(DFLT_FDU_PREC_LT)}]"
"""
Default Regex pattern to match FDU symbols in *PyDASA*. i.e.: from 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)* M**(1)* T**(-2)'
"""

#  Custom Fundamental Dimensional Units (FDU) precedence list
#  :attr: WKNG_FDU_PREC_LT
WKNG_FDU_PREC_LT: list = DFLT_FDU_PREC_LT.copy()
"""
Custom FDUs precedence list for the dimensional matrix. It can be used to define custom dimensions in *PyDASA*. i.e: 'DFLT_FDU_PREC_LT = [D, T, C]'.
"""

# Custom regex pattern for matching FDUs
# :attr: WKNG_FDU_REGEX
WKNG_FDU_REGEX: str = DFLT_FDU_REGEX
# WKNG_FDU_REGEX: str = rf"^[{''.join(WKNG_FDU_PREC_LT)}](\^-?\d+)?(\*[{''.join(WKNG_FDU_PREC_LT)}](?:\^-?\d+)?)*$"
"""
Custom Regex pattern to match FDUs in *PyDASA*. i.e.: from [T^2*D^-1] to [T^2*D^(-1)].
"""

# Custom regex pattern for matching dimensions with exponent
# :attr: WKNG_POW_REGEX
WKNG_POW_REGEX: str = DFLT_POW_REGEX
"""
Custom Regex pattern to match FDUs in *PyDASA* with exponents in the dimensions of parameters and variables. i.e.: [T^2*D^-1] to [T^(2)*D^(-1)].
"""

# Custom regex pattern for matching dimensions WITHOUT exponent
# :attr: WKNG_NO_POW_REGEX
WKNG_NO_POW_REGEX: str = DFLT_NO_POW_REGEX
# WKNG_NO_POW_REGEX: str = rf"[{''.join(WKNG_FDU_PREC_LT)}](?!\^)"
"""
Custom Regex pattern to match FDUs without exponents in parameter or variable dimensions. i.e.: [T^2*D^-1] to [T^(2)*D^(-1)].
"""

# Custom regex pattern for matching dimensions in Sympy symbolic processor
# :attr: WKNG_FDU_SYM_REGEX
WKNG_FDU_SYM_REGEX: str = DFLT_FDU_SYM_REGEX
# WKNG_FDU_SYM_REGEX: str = rf"[{''.join(WKNG_FDU_PREC_LT)}]"
"""
Custom Regex pattern to match FDU symbols in *PyDASA*. i.e.: [T*D] to [T^1*D^1].
"""

# Set of supported Fundamental Dimensional Units (FDU)
# :data: FDU_FWK_DT
FDU_FWK_DT = {
    "PHYSICAL": "Traditional physical units",
    "DIGITAL": "Software Architecture units",
    "CUSTOM": "Custom units",
}
"""
Dictionary with the supported Fundamental Dimensional Units (FDU) in *PyDASA*.
"""

# Set of supported Fundamental Dimensional Units (FDU)
# :data: FDU_FWK_TP
PARAMS_FWK_DT = {
    "INPUT": "Input parameters, what I know affects the system",
    "OUTPUT": "Output parameters, Usually the result of the analysis",
    "CONTROL": "Control parameters, including constants",
}
"""
Dictionary with the supported Fundamental Dimensional Units (FDU) in *PyDASA*.
"""

# prints to check the regex patterns
# print("DFLT_FDU_PREC_LT:", DFLT_FDU_PREC_LT)
# print("DFLT_FDU_REGEX:", DFLT_FDU_REGEX)
# print("POW_REGEX:", POW_REGEX)
# print("NO_POW_REGEX:", NO_POW_REGEX)
# print("FDU_SYM_REGEX:", FDU_SYM_REGEX)
# print("WKNG_FDU_PREC_LT:", WKNG_FDU_PREC_LT)
# print("WKNG_FDU_REGEX:", WKNG_FDU_REGEX)
# print("WKNG_NO_POW_REGEX:", WKNG_NO_POW_REGEX)
# print("WKNG_FDU_SYM_REGEX:", WKNG_FDU_SYM_REGEX)
