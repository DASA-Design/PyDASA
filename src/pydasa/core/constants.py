# -*- coding: utf-8 -*-
"""
Module constants.py
===========================================

This module specifies the Fundamental Dimensional Units (FDUs) available by default in *PyDASA*. The default framework are the Physical FDUs.

The three main types of FDUs are:
    1. Physical FDUs: in `PHY_FDU_PREC_DT` representing the standard physical dimensions.
    2. Digital FDUs: in `COMPU_FDU_PREC_DT` representing dimensions relevant to computation.
    3. Software Architecture FDUs: in `SOFT_FDU_PREC_DT` representing dimensions specific to software architecture.

The fourth FDU framework is:
    4. Custom FDUs: user-defined FDUs that can be specified as needed.

*IMPORTANT* Based on:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# python native modules

# global variables

# Default config folder name + settings
# :attr: DFLT_CFG_FOLDER
DFLT_CFG_FOLDER: str = "cfg"
# :attr: DFLT_CFG_FILE
DFLT_CFG_FILE: str = "default.json"
"""
*PyDASA* default configuration folder and file names.
"""

# Default Fundamental Dimensional Units (FDU) framework
# :attr: DFLT_FDU_FWK_DT
DFLT_FDU_FWK_DT: dict = dict()
"""
Fundamental Dimensional Units (FDUs) in default framework for *PyDASA*.
procesess (e.g., Mass [M], Length [L], Time [T]).

Purpose:
    - Defines the default dimensional framework used in *PyDASA*.
    - Used to initialize entities without a specified framework.
    - Basis for dimensional analysis precedence list in *PyDASA*.
    - Validates parameter and variable dimensions in *PyDASA*.
    - Default is the Physical FDUs framework.
    - Can be customized for specific applications or domains.
"""


# Default Fundamental Dimensional Units (FDU) precedence list
# :attr: DFLT_FDU_PREC_LT
DFLT_FDU_PREC_LT: list = list(DFLT_FDU_FWK_DT.keys())
"""
Fundamental Dimensional Units (FDUs) in precedence order for the dimensional matrix (e.g., 'M*L^-1*T^-2').

Purpose:
    - Defines the row order in the dimensional matrix.
    - Validates parameter and variable dimensions in *PyDASA*.
"""
