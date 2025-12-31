# -*- coding: utf-8 -*-
"""
Module constants.py
===========================================

This module specifies the Fundamental Dimensional Units (FDUs) available by default in *PyDASA*.

The three main types of FDUs are:
    1. Physical FDUs: in `PHY_FDU_PREC_DT` representing the standard physical dimensions.
    2. Digital FDUs: in `COMPU_FDU_PREC_DT` representing dimensions relevant to computation.
    3. Software Architecture FDUs: in `SOFT_FDU_PREC_DT` representing dimensions specific to software architecture.

*IMPORTANT* Based on:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# python native modules
# global variables
# TODO update!, this should be loaded from JSON file, not a dict()!!!


# Physical FDUs precedence dict
# :attr: PHY_FDU_PREC_DT
PHY_FDU_PREC_DT: dict = {
    "L": {
        "_unit": "m",
        "name": "Length",
        "description": "Distance between two points in space."
    },
    "M": {
        "_unit": "kg",
        "name": "Mass",
        "description": "Amount of matter in an object."
    },
    "T": {
        "_unit": "s",
        "name": "Time",
        "description": "Duration of an event or interval."
    },
    "K": {
        "_unit": "K",
        "name": "Temperature",
        "description": "Measure of average kinetic energy of particles."
    },
    "I": {
        "_unit": "A",
        "name": "Electric Current",
        "description": "Flow of electric charge."
    },
    "N": {
        "_unit": "mol",
        "name": "Amount of Substance",
        "description": "Quantity of entities (e.g., atoms, molecules)."
    },
    "C": {
        "_unit": "cd",
        "name": "Luminous Intensity",
        "description": "Perceived power of light in a given direction."
    },
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
    "T": {
        "_unit": "s",
        "name": "Time",
        "description": "Duration of an event or interval."
    },
    "S": {
        "_unit": "bit",
        "name": "Space",
        "description": "Physical extent in three dimensions."
    },
    "N": {
        "_unit": "op",
        "name": "Complexity",
        "description": "Measure of interconnectedness or intricacy in a system."
    },
}
"""
Computation FDUs Precedence Dictionary.
    - T: Time, the duration of an event or interval.
    - S: Space, the memory or storage capacity of a system.
    - N: Complexity, the measure of interconnectedness or intricacy in a system.
"""

# Software Architecture FDUs precedence dict
# :attr: SOFT_FDU_PREC_DT
SOFT_FDU_PREC_DT: dict = {
    "T": {
        "_unit": "s",
        "name": "Time",
        "description": "Duration of an event or interval."
    },
    "D": {
        "_unit": "bit",
        "name": "Data",
        "description": "Information processed by a system."
    },
    "E": {
        "_unit": "req",
        "name": "Effort",
        "description": "Measure of computational effort/complexity."
    },
    "C": {
        "_unit": "node",
        "name": "Connectivity",
        "description": "Measure of interconnections between components."
    },
    "A": {
        "_unit": "process",
        "name": "Capacity",
        "description": "Maximum amount of data that can be stored/processed."
    },
}
"""
Digital or Dimensional Analysis Software Architecture (DASA) FDUs Precedence Dictionary.
        - T: Time, the duration of an event or interval.
        - D: Data, the information processed by a system.
        - E: Effort, the measure of how much computational effort/complexity the task demands to complete.
        - C: Connectivity, the measure of interconnections between components in a system.
        - A: Capacity, the maximum amount of data or information that can be stored or processed in a system component.
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
