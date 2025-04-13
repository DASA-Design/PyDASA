# -*- coding: utf-8 -*-
"""
Configuration module for Fundamental Dimensional Units (FDUs), Parameters, and Variables in *PyDASA*.

Key Features:
    - Defines default and custom FDUs (e.g., Length, Mass, Time).
    - Provides regex patterns for validating dimensions.
    - Supports physical, digital, and custom dimensional frameworks.

Physical FDUs Precedence List:
    - Length [L], Mass [M], Time [T], Temperature [θ], Electric Current [I],
      Amount of Substance [N], Luminous Intensity [J].

Digital FDUs Precedence List:
    - Time [T], Space [S], Complexity [N].

Software Architecture Custom FDUs precedence List:
    - Time [T], Data [D], Complexity [C], Conectivity [K], Capacity [P].

Supported Frameworks:
    - PHYSICAL: Traditional physical dimensional framework.
    - COMPUTATION: Computer science dimensional framework.
    - DIGITAL: Software architecture dimensional framework.
    - DASA: Software architecture dimensional framework.
    - CUSTOM: User-defined dimensional framework.

Default Patterns:
    - `DFLT_FDU_REGEX`: Matches FDUs with exponents (e.g., 'M*L^-1*T^-2').
    - `DFLT_POW_REGEX`: Matches exponents in FDUs (e.g., '^1', '^-2').
    - `DFLT_NO_POW_REGEX`: Matches FDUs without exponents (e.g., 'M', 'L', 'T').
    - `DFLT_FDU_SYM_REGEX`: Matches FDU symbols for symbolic processing (e.g., 'M^(1)*L^(-1)*T^(-2)').

Working Patterns:
    - `WKNG_FDU_REGEX`: Working regex for user-defined/configured FDUs.
    - `WKNG_POW_REGEX`: Matches exponents in custom FDUs.
    - `WKNG_NO_POW_REGEX`: Matches custom FDUs without exponents.
    - `WKNG_FDU_SYM_REGEX`: Matches custom FDU symbols for symbolic processing.

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
    "θ": "Temperature",
    "I": "Electric Current",
    "N": "Amount of Substance",
    "J": "Luminous Intensity",
}
"""
Physical FDUs Precedence Dictionary.
    - Length [L]: Distance between two points in space.
    - Mass [M]: Amount of matter in an object.
    - Time [T]: Duration of an event or interval.
    - Temperature [θ]: Measure of average kinetic energy of particles.
    - Electric Current [I]: Flow of electric charge.
    - Amount of Substance [N]: Quantity of entities (e.g., atoms, molecules).
    - Luminous Intensity [J]: Perceived power of light in a given direction.
"""

# Digital FDUs precedence dict
# :attr: DIGI_FDU_PREC_DT
DIGI_FDU_PREC_DT: dict = {
    "T": "Time",
    "S": "Space",
    "N": "Complexity",
}
"""
Digital FDUs Precedence Dictionary.
    - T: Time, the duration of an event or interval.
    - S: Space, the memory or storage capacity of a system.
    - N: Complexity, the measure of interconnectedness or intricacy in a system.
"""

# Software Architecture FDUs precedence dict
# :attr: DASA_FDU_PREC_DT
DASA_FDU_PREC_DT: dict = {
    "T": "Time",
    "D": "Data",
    "C": "Complexity",
    "K": "Conectivity",
    "P": "Capacity",
}
"""
Dimensional Analysis Software Architecture (DASA) FDUs Precedence Dictionary.
        - T: Time, the duration of an event or interval.
        - D: Data, the information processed by a system.
        - C: Complexity, the measure of interconnectedness or intricacy in a component operation.
        - K: Conectivity, the measure of interconnections between components in a system.
        - P: Capacity, the maximum amount of data or information that can be stored or processed in a system component.
"""

# Supported Fundamental Dimensional Unit (FDU) Frameworks
# :data: FDU_FWK_DT
FDU_FWK_DT: dict = {
    "PHYSICAL": "Traditional physical dimensional framework (e.g., Length, Mass, Time).",
    "COMPUTATION": "Computer science dimensional framework (e.g., Time, Space, Complexity).",
    "DIGITAL": "Software architecture dimensional framework (e.g., Time, Data, Connectivity).",
    "CUSTOM": "User-defined dimensional framework for specific use cases.",
}
"""
Supported Fundamental Dimensional Units (FDUs) Frameworks in *PyDASA*.

Purpose:
    - Defines the dimensional frameworks supported in *PyDASA*.
"""

# Supported Parameter and Variable categories
# :data: PARAMS_FWK_DT
PARAMS_FWK_DT: dict = {
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

# Default Fundamental Dimensional Units (FDU) precedence list
# :attr: DFLT_FDU_PREC_LT
DFLT_FDU_PREC_LT: list = list(PHY_FDU_PREC_DT.keys())
"""
Fundamental Dimensional Units (FDUs) in precedence order for the dimensional matrix (e.g., 'M*L^-1*T^-2').

Purpose:
    - Defines the row order in the dimensional matrix.
    - Validates parameter and variable dimensions in *PyDASA*.
"""

# Default regex pattern for matching FDUs
# :attr: DFLT_FDU_REGEX
DFLT_FDU_REGEX: str = rf"^[{''.join(DFLT_FDU_PREC_LT)}](\^-?\d+)?(\*[{''.join(DFLT_FDU_PREC_LT)}](?:\^-?\d+)?)*$"
# DFLT_FDU_REGEX: str = r"^[LMTθINJ](\^-?\d+)?(\*[LMTθINJ](?:\^-?\d+)?)*$"
"""
Default regex pattern to match FDUs in *PyDASA* (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# Default regex pattern for matching dimensions with exponent
# :attr: DFLT_POW_REGEX
DFLT_POW_REGEX: str = r"\-?\d+"   # r'\^(-?\d+)'
"""
Default regex to match FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
"""

# Default regex pattern for matching dimensions WITHOUT exponent
# :attr: DFLT_NO_POW_REGEX
DFLT_NO_POW_REGEX: str = rf"[{''.join(DFLT_FDU_PREC_LT)}](?!\^)"
"""
Default regex to match FDUs without exponents (e.g., 'M*L*T' instead of 'M*L^-1*T^-2').
# """

# Default regex pattern for matching FDUs in Sympy symbolic processor
# :attr: DFLT_FDU_SYM_REGEX
DFLT_FDU_SYM_REGEX: str = rf"[{''.join(DFLT_FDU_PREC_LT)}]"
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
# :attr: WKNG_FDU_REGEX
WKNG_FDU_REGEX: str = DFLT_FDU_REGEX
"""
Working regex to match FDUs in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

# Working regex pattern for matching dimensions with exponent
# :attr: WKNG_POW_REGEX
WKNG_POW_REGEX: str = DFLT_POW_REGEX
"""
Working regex to match FDUs with exponents in *PyDASA* (e.g., 'T^2*D^-1' to 'T^(2)*D^(-1)').
"""

# Working regex pattern for matching dimensions WITHOUT exponent
# :attr: WKNG_NO_POW_REGEX
WKNG_NO_POW_REGEX: str = DFLT_NO_POW_REGEX
"""
Working regex to match FDUs without exponents (e.g., 'T*D' instead of 'T^2*D^-1').
"""

# Working regex pattern for matching dimensions in Sympy symbolic processor
# :attr: WKNG_FDU_SYM_REGEX
WKNG_FDU_SYM_REGEX: str = DFLT_FDU_SYM_REGEX
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
