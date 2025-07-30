# -*- coding: utf-8 -*-
# FIXME old code, remove when tests are finished
"""
Configuration module for Fundamental Dimensional Units (FDUs), Parameters, and Variables in *PyDASA*.

Key Features:
    - Defines default and custom FDUs (e.g., Length, Mass, Time).
    - Provides regex patterns for validating dimensions.
    - Supports physical, digital, and custom dimensional frameworks.

Physical FDUs Precedence List:
    - Length [L], Mass [M], Time [T], Temperature [K], Electric Current [A],
    Amount of Substance [N/n/mol], Luminous Intensity [C/c/cd].

Digital FDUs Precedence List:
    - Time [T], Space [S], Complexity [N].

Software Architecture Custom FDUs precedence List:
    - Time [T], Data [D], Effort [E], Connectivity [K], Capacity [A].

Supported Frameworks:
    - PHYSICAL: Traditional physical dimensional framework.
    - COMPUTATION: Computer science dimensional framework.
    - SOFTWARE: Software architecture dimensional framework.
    - CUSTOM: User-defined dimensional framework.

Supported Sensitivity Analysis Parameters:
    - SYMBOLIC: Sensitivity analysis for symbolic processable Parameters (e.g., 'x + y').
    - NUMERIC: Sensitivity analysis for numeric Variables (e.g., 1.0, 2.5).
    - HYBRID: Sensitivty analysis that includes both symbolic and numeric sensitivity analysis.
    - CUSTOM: User-defined sensitivity analysis for specific use cases.

Default Patterns:
    - `DFLT_FDU_RE`: Matches FDUs with exponents (e.g., 'M*L^-1*T^-2').
    - `DFLT_POW_RE`: Matches exponents in FDUs (e.g., '^1', '^-2').
    - `DFLT_NO_POW_RE`: Matches FDUs without exponents (e.g., 'M', 'L', 'T').
    - `DFLT_FDU_SYM_RE`: Matches FDU symbols for symbolic processing (e.g., 'M^(1)*L^(-1)*T^(-2)').

Working Patterns:
    - `WKNG_FDU_RE`: Working regex for user-defined/configured FDUs.
    - `WKNG_POW_RE`: Matches exponents in custom FDUs.
    - `WKNG_NO_POW_RE`: Matches custom FDUs without exponents.
    - `WKNG_FDU_SYM_RE`: Matches custom FDU symbols for symbolic processing.

*IMPORTANT* Based on:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# python native modules
# global variables

# Physical FDUs precedence dict
# :attr: PHY_FDU_PREC_DT
PHY_FDU_PREC_DT: dict = {
    "L": "Length",
    "M": "Mass",
    "T": "Time",
    "K": "Temperature",
    "I": "Electric Current",
    "N": "Amount of Substance",
    "C": "Luminous Intensity",
}
"""
Physical FDUs Precedence Dictionary.
    - Length [L]: Distance between two points in space.
    - Mass [M]: Amount of matter in an object.
    - Time [T]: Duration of an event or interval.
    - Temperature [K]: Measure of average kinetic energy of particles.
    - Electric Current [I]: Flow of electric charge.
    - Amount of Substance [N/n/mol]: Quantity of entities (e.g., atoms, molecules).
    - Luminous Intensity [C/c/cd]: Perceived power of light in a given direction.
"""

# Computation FDUs precedence dict
# :attr: COMPU_FDU_PREC_DT
COMPU_FDU_PREC_DT: dict = {
    "T": "Time",
    "S": "Space",
    "N": "Complexity",
}
"""
Computation FDUs Precedence Dictionary.
    - T: Time, the duration of an event or interval.
    - S: Space, the memory or storage capacity of a system.
    - N: Complexity, the measure of interconnectedness or intricacy in a system.
"""

# Software Architecture FDUs precedence dict
# :attr: DIGI_FDU_PREC_DT
DIGI_FDU_PREC_DT: dict = {
    "T": "Time",
    "D": "Data",
    "E": "Effort",
    "C": "Connectivity",
    "A": "Capacity",
}
"""
Digital or Dimensional Analysis Software Architecture (DASA) FDUs Precedence Dictionary.
        - T: Time, the duration of an event or interval.
        - D: Data, the information processed by a system.
        - E: Effort, the measure of how much computational effort/complexity the task demands to complete.
        - C: Connectivity, the measure of interconnections between components in a system.
        - A: Capacity, the maximum amount of data or information that can be stored or processed in a system component.
"""

# Supported Fundamental Dimensional Unit (FDU) Frameworks
# :data: FDU_FWK_DT
FDU_FWK_DT: dict = {
    "PHYSICAL": "Traditional physical dimensional framework (e.g., Length, Mass, Time).",
    "COMPUTATION": "Computer science dimensional framework (e.g., Time, Space, Complexity).",
    "SOFTWARE": "Software architecture dimensional framework (e.g., Time, Data, Connectivity).",
    "CUSTOM": "User-defined dimensional framework for specific use cases.",
}
"""
Supported Fundamental Dimensional Units (FDUs) Frameworks in *PyDASA*.

Purpose:
    - Defines the dimensional frameworks supported in *PyDASA*.
"""

# Supported Parameter and Variable categories
# :data: PARAMS_CAT_DT
PARAMS_CAT_DT: dict = {
    "INPUT": "Parameters that influence the system (e.g., known inputs).",
    "OUTPUT": "Parameters that represent the results of the analysis.",
    "CONTROL": "Parameters used to control or constrain the system (e.g., constants).",
}
"""
Supported Parameter and Variable categories in *PyDASA*.

Purpose:
    - Defines the parameter categories supported in *PyDASA*.
    - Used to classify parameters in the dimensional matrix.
"""

# Supported Dimensionless Coefficients (DC) categories
# :data: DC_CAT_DT
DC_CAT_DT: dict = {
    "COMPUTED": "Coefficients directly calculated using the Buckingham Pi theorem.",
    "DERIVED": "Coefficients obtained by combining or manipulating computed coefficients.",
}
"""
Supported categories of Dimensionless Coefficients (DN) in *PyDASA*.

Purpose:
    - Defines the categories of dimensionless coefficients supported in *PyDASA*.
    - Used to classify dimensionless coefficients in formulas and equations.
    - Helps in organizing and managing dimensionless coefficients in the analysis.
"""

# Supported Sensitivity Analysis Parameters
# :data: SENS_ANSYS_DT
SENS_ANSYS_DT: dict = {
    "SYMBOLIC": "Sensitivity analysis for symbolic processable Parameters (e.g., 'x + y').",
    "NUMERIC": "Sensitivity analysis for numeric Variables (e.g., 1.0, 2.5).",
    "HYBRID": "Sensitivity analysis that includes both symbolic and numeric sensitivity analysis.",
    "CUSTOM": "User-defined sensitivity analysis for specific use cases.",
}

# Default Fundamental Dimensional Units (FDU) precedence list
# :attr: DFLT_FDU_PREC_LT
DFLT_FDU_PREC_LT: list = list(PHY_FDU_PREC_DT.keys())
"""
Fundamental Dimensional Units (FDUs) in precedence order for the dimensional matrix (e.g., 'M*L^-1*T^-2').

Purpose:
    - Defines the row order in the dimensional matrix.
    - Validates parameter and variable dimensions in *PyDASA*.
"""

# TODO clean the regex when test are finished
# Allow valid LaTeX strings starting with a backslash or alphanumeric strings
# :attr: LATEX_RE
LATEX_RE: str = r"(\\?[a-zA-Z]+)(?:_\{\d+\})?"
"""
LaTeX regex pattern to match LaTeX symbols (e.g., '\alpha', '\beta_{1}') in *PyDASA*.
"""

# Default regex pattern for matching FDUs
# :attr: DFLT_FDU_RE
DFLT_FDU_RE: str = rf"^[{''.join(DFLT_FDU_PREC_LT)}](\^-?\d+)?(\*[{''.join(DFLT_FDU_PREC_LT)}](?:\^-?\d+)?)*$"
# DFLT_FDU_RE: str = r"^[LMTθINJ](\^-?\d+)?(\*[LMTθINJ](?:\^-?\d+)?)*$"
"""
Default regex pattern to match FDUs in *PyDASA* (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# Default regex pattern for matching dimensions with exponent
# :attr: DFLT_POW_RE
DFLT_POW_RE: str = r"\-?\d+"   # r'\^(-?\d+)'
"""
Default regex to match FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# Default regex pattern for matching dimensions WITHOUT exponent
# :attr: DFLT_NO_POW_RE
DFLT_NO_POW_RE: str = rf"[{''.join(DFLT_FDU_PREC_LT)}](?!\^)"
"""
Default regex to match FDUs without exponents (e.g., 'M*L*T' instead of 'M*L^-1*T^-2').
# """

# Default regex pattern for matching FDUs in Sympy symbolic processor
# :attr: DFLT_FDU_SYM_RE
DFLT_FDU_SYM_RE: str = rf"[{''.join(DFLT_FDU_PREC_LT)}]"
"""
Default regex to match FDU symbols in *PyDASA* (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
"""

# Working Fundamental Dimensional Units (FDU) precedence list
#  :attr: WKNG_FDU_PREC_LT
WKNG_FDU_PREC_LT: list = DFLT_FDU_PREC_LT.copy()
"""
Working FDUs precedence list for the dimensional matrix, allowing custom dimensions (e.g., 'DFLT_FDU_PREC_LT = [D, T, C]').
"""

# Working regex pattern for matching FDUs
# :attr: WKNG_FDU_RE
WKNG_FDU_RE: str = DFLT_FDU_RE
"""
Working regex to match FDUs in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

# Working regex pattern for matching dimensions with exponent
# :attr: WKNG_POW_RE
WKNG_POW_RE: str = DFLT_POW_RE
"""
Working regex to match FDUs with exponents in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

# Working regex pattern for matching dimensions WITHOUT exponent
# :attr: WKNG_NO_POW_RE
WKNG_NO_POW_RE: str = DFLT_NO_POW_RE
"""
Working regex to match FDUs without exponents (e.g., 'T*D' instead of 'T^2*D^-1').
"""

# Working regex pattern for matching dimensions in Sympy symbolic processor
# :attr: WKNG_FDU_SYM_RE
WKNG_FDU_SYM_RE: str = DFLT_FDU_SYM_RE
"""
Working regex to match FDU symbols in *PyDASA* (e.g., 'T^(1)*D^(-1)' to 'D**(-1)*T**(2)').
"""

"""
Purpose:
    - Define FDUs and their precedence for dimensional analysis.
    - Provide regex patterns for validating and processing FDUs.
    - Support physical, digital, and custom dimensional frameworks.
"""
# End of file
