# -*- coding: utf-8 -*-
"""
Module to support other tests in PyDASA.
===========================================

Tests data for all the classes in PyDASA.
"""
# import testing package
# import unittest
# import pytest
import numpy as np


# test data for from pydasa.utils import config
def get_config_test_data():
    """Get test data for config tests."""
    return {
        "PHYSICAL_KEYS": ["L", "M", "T", "K", "I", "N", "C"],
        "PHYSICAL_UNITS": {
            "L": "m", "M": "kg", "T": "s", "K": "K",
            "I": "A", "N": "mol", "C": "cd"
        },
        "COMPUTATION_KEYS": ["T", "S", "N"],
        "COMPUTATION_UNITS": {"T": "s", "S": "bit", "N": "op"},
        "SOFTWARE_KEYS": ["T", "D", "E", "C", "A"],
        "SOFTWARE_UNITS": {
            "T": "s", "D": "bit", "E": "req",
            "C": "node", "A": "process"
        },
        "FRAMEWORK_KEYS": ["PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"],
        "PARAMS_CAT_KEYS": ["IN", "OUT", "CTRL"],
        "DC_CAT_KEYS": ["COMPUTED", "DERIVED"],
        "SENS_ANSYS_KEYS": ["SYM", "NUM"],
        "VALID_LATEX": ["alpha", "\\alpha", "beta_1", "\\beta_{1}", "\\Pi_{0}"],
        "VALID_DIMENSIONS": ["M", "L*T", "M*L^-1*T^-2", "L^2*T^-1", "T^-1"],
        "INVALID_DIMENSIONS": ["X", "M*X", "M**2", "M^2.5", "M L", ""],
        "PHYSICAL_DIMS": [
            "M*L^-1*T^-2", "M*L^2*T^-2", "M*L^2*T^-3",
            "L*T^-1", "L*T^-2", "M*L^-3", "M*T^-1", "L^3*T^-1"
        ],
        "REQUIRED_FIELDS": ["_unit", "_name", "description"],
    }


# test data for from pydasa.core.fundamental import Dimension
def get_dimension_test_data() -> dict:
    """Get test data for Dimension tests."""
    return {
        "PHYSICAL_DATA": {
            "_idx": 0,
            "_sym": "L",
            "_alias": "L",
            "_fwk": "PHYSICAL",
            "_unit": "m",
            "_name": "Length",
            "description": "Physical length dimension"
        },
        "COMPUTATION_DATA": {
            "_idx": 1,
            "_sym": "S",
            "_alias": "S",
            "_fwk": "COMPUTATION",
            "_unit": "bit",
            "_name": "Storage",
            "description": "Digital storage dimension"
        },
        "SOFTWARE_DATA": {
            "_idx": 2,
            "_sym": "E",
            "_alias": "E",
            "_fwk": "SOFTWARE",
            "_unit": "req",
            "_name": "Events",
            "description": "Software event dimension"
        },
        "INVALID_UNIT_DATA": ["", 123, None, []],
        "VALID_UNITS": ["m", "kg", "s", "bit", "req"],
    }


# test data for from pydasa.utils.error import handle_error, inspect_var
def get_error_test_data():
    """Get test data for error tests."""
    return {
        "VALID_CONTEXTS": ["TestClass", "MyModule", "TestContext"],
        "VALID_FUNCTIONS": ["test_method", "my_function", "test_func"],
        "EXCEPTION_TYPES": [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            IndexError("index error"),
            AttributeError("attribute error"),
        ],
        "INVALID_CONTEXT_TYPES": [123, None, [], {}],
        "INVALID_FUNCTION_TYPES": [456, None, [], {}],
        "INVALID_EXCEPTION_TYPES": ["not an exception", 123, None, []],
        "SPECIAL_CHARS": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "UNICODE_CONTEXT": "TestContext_üñíçödé",
        "UNICODE_FUNCTION": "test_func_αβγ",
        "UNICODE_MESSAGE": "Error with émojis 🚀",
    }


# test data for from pydasa.utils.io import load_json, save_json, load, save
def get_io_test_data():
    """Get test data for I/O tests."""
    return {
        "SIMPLE_JSON": {"key": "value", "number": 42},
        "NESTED_JSON": {
            "level1": {
                "level2": {
                    "value": "nested"
                }
            }
        },
        "UNICODE_JSON": {"text": "αβγ", "emoji": "🚀"},
        "ARRAY_JSON": {"items": [1, 2, 3], "names": ["a", "b", "c"]},
        "INVALID_JSON_CONTENT": "{invalid json}",
        "SUPPORTED_FORMATS": [".json"],
        "UNSUPPORTED_FORMATS": [".xml", ".csv", ".txt", ".yaml"],
        "TEST_FILENAMES": {
            "simple": "test.json",
            "nested": "nested.json",
            "unicode": "unicode.json",
            "output": "output.json",
            "invalid": "invalid.json",
            "nested_dir": "subdir/nested/file.json",
        },
    }


# test data for from pydasa.utils.latex import (latex_to_python,
# extract_latex_vars,
# create_latex_mapping,
# IGNORE_EXPR)
def get_latex_test_data():
    """Get test data for LaTeX tests."""
    return {
        "SIMPLE_STRINGS": ["x", "abc", "X123"],
        "LATEX_SYMBOLS": {
            "\\alpha": "alpha",
            "\\beta": "beta",
            "\\gamma": "gamma",
            "\\Pi": "Pi"
        },
        "SUBSCRIPTS": {
            "\\alpha_{1}": "alpha_1",
            "\\beta_{12}": "beta_12",
            "x_{i}": "x_i",
            "\\mu_{0}": "mu_0"
        },
        "GREEK_LETTERS": [
            "\\alpha", "\\beta", "\\gamma", "\\delta",
            "\\epsilon", "\\theta", "\\lambda", "\\mu",
            "\\pi", "\\sigma", "\\tau", "\\omega"
        ],
        "EXPECTED_FUNCTIONS": {
            "\\frac", "\\sqrt", "\\sin", "\\cos",
            "\\tan", "\\log", "\\exp"
        },
        "COMPLEX_EXPR": "\\alpha + \\beta_{1} + \\gamma_{2}",
        "PHYSICS_EXPR": "\\frac{U * y_{2}}{d} + \\frac{P * d^{2}}{\\mu_{1} * U}",
        "DIMENSIONAL_CASES": [
            ("\\Pi_{0}", ["Pi_0"]),
            ("\\Pi_{0} * \\Pi_{1}", ["Pi_0", "Pi_1"]),
            ("\\frac{\\mu_{1}}{U}", ["mu_1", "U"]),
            ("\\frac{y_{2}}{d}", ["y_2", "d"]),
        ],
    }


# test data for from pydasa.core.basic import SymBasis et al
def get_basic_test_data() -> dict:
    """Get test data for basic.py tests."""
    return {
        "VALID_SYM_DATA": [
            {"_sym": "L", "_alias": "L", "_fwk": "PHYSICAL"},
            {"_sym": "M", "_alias": "M", "_fwk": "PHYSICAL"},
            {"_sym": "T", "_alias": "T", "_fwk": "COMPUTATION"},
            {"_sym": "\\alpha", "_alias": "alpha", "_fwk": "PHYSICAL"},
            {"_sym": "\\Pi_{0}", "_alias": "Pi_0", "_fwk": "PHYSICAL"},
        ],
        "VALID_SYMBOLS": ["L", "M", "T", "d", "V", "\\alpha", "\\beta", "\\Pi_{0}"],
        "INVALID_SYMBOLS": [123, 45.6, [], {}, True],  # Non-string types
        "VALID_FRAMEWORKS": ["PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"],
        "INVALID_FRAMEWORKS": ["INVALID", "physical", "", None],
        "VALID_ALIASES": ["L", "M", "rho_1", "alpha", "Pi_0"],
        "LATEX_SYMBOLS": ["\\alpha", "\\beta", "\\gamma", "\\Pi_{0}", "\\mu_{1}"],
        "VALID_IDX_DATA": [
            {"_idx": 0, "_sym": "L", "_alias": "L", "_fwk": "PHYSICAL"},
            {"_idx": 1, "_sym": "M", "_alias": "M", "_fwk": "PHYSICAL"},
            {"_idx": 2, "_sym": "T", "_alias": "T", "_fwk": "COMPUTATION"},
        ],
        "VALID_INDICES": [0, 1, 2, 5, 10, 100],
        "INVALID_INDICES": [-1, -5, "0", None, 3.14],
        "VALID_VALIDATION_DATA": [
            {
                "_idx": 0,
                "_sym": "L",
                "_alias": "L",
                "_fwk": "PHYSICAL",
                "_name": "Length",
                "description": "physical length dimension"
            },
            {
                "_idx": 1,
                "_sym": "M",
                "_alias": "M",
                "_fwk": "PHYSICAL",
                "_name": "Mass",
                "description": "physical mass dimension"
            },
        ],
        "VALID_NAMES": ["Length", "Mass", "Time", "Storage"],
        "INVALID_NAMES": [123, [], {}, None, True],  # Non-string types
    }


# test data for from pydasa.core.parameter import Variable
def get_variable_test_data() -> dict:
    """Get test data for Variable tests."""
    return {
        "PHYSICAL_VARIABLE": {
            "_idx": 0,
            "_sym": "v",
            "_alias": "v",
            "_fwk": "PHYSICAL",
            "_cat": "IN",
            "_dims": "L*T^-1",
            "_units": "m/s",
            "_name": "Velocity",
            "description": "velocity of fluid",
            "_setpoint": 2.5,
            "_std_setpoint": 2.5
        },
        "COMPUTATION_VARIABLE": {
            "_idx": 1,
            "_sym": "t",
            "_alias": "t",
            "_fwk": "COMPUTATION",
            "_cat": "OUT",
            "_dims": "T",
            "_units": "s",
            "_name": "Time",
            "description": "computation time",
            "_setpoint": 1.0,
            "_std_setpoint": 1.0
        },
        "SOFTWARE_VARIABLE": {
            "_idx": 2,
            "_sym": "e",
            "_alias": "e",
            "_fwk": "SOFTWARE",
            "_cat": "CTRL",
            "_dims": "E",
            "_units": "req",
            "_name": "Events",
            "description": "software events",
            "_setpoint": 100.0,
            "_std_setpoint": 100.0
        },
        "VALID_CATEGORIES": ["IN", "OUT", "CTRL", "in", "out", "ctrl"],
        "INVALID_CATEGORIES": ["INVALID", "input", "output", ""],
        "VALID_DIMENSIONS": ["L", "M*L^-1", "L*T^-1", "M*L^2*T^-2"],
        "VALID_UNITS": ["m", "m/s", "kg/m3", "bit", "req/s"],
        "VALID_DIST_TYPES": ["uniform", "normal", "triangular",
                             "exponential", "lognormal", "custom"],
        # Minimal distribution function test data
        "SAMPLE_TEST_FUNCTIONS": {
            "constant": lambda: 42.0,
            "uniform": lambda: float(np.random.uniform(0, 10)),
            "dependent": lambda x: 2 * x + 1.0,
        }
    }


# test data for from pydasa.dimensional.framework import Schema, Dimension
def get_framework_test_data() -> dict:
    """Get test data for Schema/framework tests."""
    return {
        "VALID_FRAMEWORKS": ["PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"],
        "INVALID_FRAMEWORKS": ["INVALID", "physical", "", None],
        "PHYSICAL_FDU_LIST": [
            {"_idx": 0, "_sym": "L", "_alias": "L", "_fwk": "PHYSICAL",
             "_unit": "m", "_name": "Length",
             "description": "Distance between two points in space."},
            {"_idx": 1, "_sym": "M", "_alias": "M", "_fwk": "PHYSICAL",
             "_unit": "kg", "_name": "Mass",
             "description": "Amount of matter in an object."},
            {"_idx": 2, "_sym": "T", "_alias": "T", "_fwk": "PHYSICAL",
             "_unit": "s", "_name": "Time",
             "description": "Duration of an event or interval."},
            {"_idx": 3, "_sym": "K", "_alias": "K", "_fwk": "PHYSICAL",
             "_unit": "K", "_name": "Temperature",
             "description": "Measure of average kinetic energy of particles."},
            {"_idx": 4, "_sym": "I", "_alias": "I", "_fwk": "PHYSICAL",
             "_unit": "A", "_name": "Electric Current",
             "description": "Flow of electric charge."},
            {"_idx": 5, "_sym": "N", "_alias": "N", "_fwk": "PHYSICAL",
             "_unit": "mol", "_name": "Amount of Substance",
             "description": "Quantity of entities (e.g., atoms, molecules)."},
            {"_idx": 6, "_sym": "C", "_alias": "C", "_fwk": "PHYSICAL",
             "_unit": "cd", "_name": "Luminous Intensity",
             "description": "Perceived power of light in a given direction."},
        ],
        "COMPUTATION_FDU_LIST": [
            {"_idx": 0, "_sym": "T", "_alias": "T", "_fwk": "COMPUTATION",
             "_unit": "s", "_name": "Time",
             "description": "Duration of an event or interval."},
            {"_idx": 1, "_sym": "S", "_alias": "S", "_fwk": "COMPUTATION",
             "_unit": "bit", "_name": "Space",
             "description": "Physical extent in three dimensions."},
            {"_idx": 2, "_sym": "N", "_alias": "N", "_fwk": "COMPUTATION",
             "_unit": "op", "_name": "Complexity",
             "description": "Measure of interconnectedness or intricacy in a system."},
        ],
        "SOFTWARE_FDU_LIST": [
            {"_idx": 0, "_sym": "T", "_alias": "T", "_fwk": "SOFTWARE",
             "_unit": "s", "_name": "Time",
             "description": "Duration of an event or interval."},
            {"_idx": 1, "_sym": "D", "_alias": "D", "_fwk": "SOFTWARE",
             "_unit": "bit", "_name": "Data",
             "description": "Information processed by a system."},
            {"_idx": 2, "_sym": "E", "_alias": "E", "_fwk": "SOFTWARE",
             "_unit": "req", "_name": "Effort",
             "description": "Measure of computational effort/complexity."},
            {"_idx": 3, "_sym": "C", "_alias": "C", "_fwk": "SOFTWARE",
             "_unit": "node", "_name": "Connectivity",
             "description": "Measure of interconnections between components."},
            {"_idx": 4, "_sym": "A", "_alias": "A", "_fwk": "SOFTWARE",
             "_unit": "process", "_name": "Capacity",
             "description": "Maximum amount of data that can be stored/processed."},
        ],
        "CUSTOM_FDU_LIST": [
            {"_idx": 0, "_sym": "X", "_alias": "X", "_fwk": "CUSTOM",
             "_unit": "unit1", "_name": "Custom1",
             "description": "Custom dimension 1"},
            {"_idx": 1, "_sym": "Y", "_alias": "Y", "_fwk": "CUSTOM",
             "_unit": "unit2", "_name": "Custom2",
             "description": "Custom dimension 2"},
        ],
        "PHYSICAL_SYMBOLS": ["L", "M", "T", "K", "I", "N", "C"],
        "PHYSICAL_ALIASES": ["L", "M", "T", "K", "I", "N", "C"],
        "COMPUTATION_SYMBOLS": ["T", "S", "N"],
        "COMPUTATION_ALIASES": ["T", "S", "N"],
        "SOFTWARE_SYMBOLS": ["T", "D", "E", "C", "A"],
        "SOFTWARE_ALIASES": ["T", "D", "E", "C", "A"],
        "PHYSICAL_SCHEME_DICT": {
            "fwk": "PHYSICAL",
            "fdu_list": [
                {"idx": 0, "sym": "L", "alias": "L", "fwk": "PHYSICAL",
                 "unit": "m", "_name": "Length",
                 "description": "Distance between two points in space."},
                {"idx": 1, "sym": "M", "alias": "M", "fwk": "PHYSICAL",
                 "unit": "kg", "_name": "Mass",
                 "description": "Amount of matter in an object."},
                {"idx": 2, "sym": "T", "alias": "T", "fwk": "PHYSICAL",
                 "unit": "s", "_name": "Time",
                 "description": "Duration of an event or interval."},
            ]
        },
    }


# test data for from pydasa.buckingham.vashchy import Coefficient
def get_coefficient_test_data() -> dict:
    """Get test data for Coefficient tests."""
    return {
        "TEST_VARIABLES": {
            "v": {
                "_sym": "v",
                "_alias": "v",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L*T^-1",
                "_units": "m/s",
                "_name": "Velocity",
                "description": "Fluid velocity",
                "relevant": True,
                "_idx": 0,
                "_min": 0.0,
                "_max": 10.0,
                "_mean": 5.0,
                "_dev": 0.5,
                "_std_units": "m/s",
                "_std_min": 0.0,
                "_std_max": 10.0,
                "_std_mean": 5.0,
                "_std_dev": 0.5,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 10.0},
                "_depends": []
            },
            "L": {
                "_sym": "L",
                "_alias": "L",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L",
                "_units": "m",
                "_name": "Length",
                "description": "Characteristic length",
                "relevant": True,
                "_idx": 1,
                "_min": 0.1,
                "_max": 5.0,
                "_mean": 2.5,
                "_dev": 0.25,
                "_std_units": "m",
                "_std_min": 0.1,
                "_std_max": 5.0,
                "_std_mean": 2.5,
                "_std_dev": 0.25,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.1, "max": 5.0},
                "_depends": []
            },
            "\\rho": {
                "_sym": "\\rho",
                "_alias": "rho",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "M*L^-3",
                "_units": "kg/m^3",
                "_name": "Density",
                "description": "Fluid density",
                "relevant": True,
                "_idx": 2,
                "_min": 800.0,
                "_max": 1200.0,
                "_mean": 1000.0,
                "_dev": 50.0,
                "_std_units": "kg/m^3",
                "_std_min": 800.0,
                "_std_max": 1200.0,
                "_std_mean": 1000.0,
                "_std_dev": 50.0,
                "_step": 10.0,
                "_dist_type": "uniform",
                "_dist_params": {"min": 800.0, "max": 1200.0},
                "_depends": []
            },
            "\\mu": {
                "_sym": "\\mu",
                "_alias": "mu",
                "_fwk": "PHYSICAL",
                "_cat": "CTRL",
                "_dims": "M*L^-1*T^-1",
                "_units": "Pa*s",
                "_name": "Dynamic Viscosity",
                "description": "Dynamic viscosity of fluid",
                "relevant": True,
                "_idx": 3,
                "_min": 0.001,
                "_max": 0.1,
                "_mean": 0.05,
                "_dev": 0.005,
                "_std_units": "Pa*s",
                "_std_min": 0.001,
                "_std_max": 0.1,
                "_std_mean": 0.05,
                "_std_dev": 0.005,
                "_step": 0.001,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.001, "max": 0.1},
                "_depends": []
            },
            "d": {
                "_sym": "d",
                "_alias": "d",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L",
                "_units": "m",
                "_name": "Diameter",
                "description": "Pipe diameter",
                "relevant": True,
                "_idx": 4,
                "_min": 0.01,
                "_max": 1.0,
                "_mean": 0.5,
                "_dev": 0.05,
                "_std_units": "m",
                "_std_min": 0.01,
                "_std_max": 1.0,
                "_std_mean": 0.5,
                "_std_dev": 0.05,
                "_step": 0.01,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.01, "max": 1.0},
                "_depends": []
            },
            "P": {
                "_sym": "P",
                "_alias": "P",
                "_fwk": "PHYSICAL",
                "_cat": "OUT",
                "_dims": "M*L^-1*T^-2",
                "_units": "Pa",
                "_name": "Pressure",
                "description": "Fluid pressure",
                "relevant": True,
                "_idx": 5,
                "_min": 0.0,
                "_max": 100000.0,
                "_mean": 50000.0,
                "_dev": 5000.0,
                "_std_units": "Pa",
                "_std_min": 0.0,
                "_std_max": 100000.0,
                "_std_mean": 50000.0,
                "_std_dev": 5000.0,
                "_step": 1000.0,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 100000.0},
                "_depends": []
            }
        },
        "REYNOLDS_COEFFICIENT": {
            "_idx": 0,
            "_sym": "\\Pi_{Re}",
            "_alias": "Pi_Re",
            "_fwk": "PHYSICAL",
            "_cat": "COMPUTED",
            "_dim_col": [1, 1, 1, -1],  # v*L*rho/mu
            "variables": {
                "v": {
                    "_sym": "v",
                    "_dims": "L*T^-1",
                    "_fwk": "PHYSICAL",
                    "_name": "Velocity"
                },
                "L": {
                    "_sym": "L",
                    "_dims": "L",
                    "_fwk": "PHYSICAL",
                    "_name": "Length"
                },
                "\\rho": {
                    "_sym": "\\rho",
                    "_dims": "M*L^-3",
                    "_fwk": "PHYSICAL",
                    "_name": "Density"
                },
                "\\mu": {
                    "_sym": "\\mu",
                    "_dims": "M*L^-1*T^-1",
                    "_fwk": "PHYSICAL",
                    "_name": "Viscosity"
                }
            },
            "_name": "Reynolds Number",
            "description": "Ratio of inertial to viscous forces"
        },
        "SIMPLE_COEFFICIENT": {
            "_idx": 1,
            "_sym": "\\Pi_{1}",
            "_alias": "Pi_1",
            "_fwk": "PHYSICAL",
            "_cat": "COMPUTED",
            "_dim_col": [1, -1],  # v/L
            "variables": {
                "v": {
                    "_sym": "v",
                    "_dims": "L*T^-1",
                    "_fwk": "PHYSICAL",
                    "_name": "Velocity"
                },
                "L": {
                    "_sym": "L",
                    "_dims": "L",
                    "_fwk": "PHYSICAL",
                    "_name": "Length"
                }
            },
            "_name": "Simple Ratio",
            "description": "Simple velocity-length ratio"
        },
        "VALID_CATEGORIES": ["COMPUTED", "DERIVED", "computed", "derived"],
        "INVALID_CATEGORIES": ["INVALID", "INPUT", "OUTPUT", ""],
        # Test data for calculate_setpoint() tests
        "SETPOINT_TEST_VARS": {
            "v_with_setpoint": {
                "_sym": "v",
                "_alias": "v",
                "_fwk": "PHYSICAL",
                "_dims": "L*T^-1",
                "_setpoint": 5.0,
                "_std_setpoint": 5.0,
                "_name": "Velocity",
                "description": "Velocity with setpoint"
            },
            "L_with_setpoint": {
                "_sym": "L",
                "_alias": "L",
                "_fwk": "PHYSICAL",
                "_dims": "L",
                "_setpoint": 2.0,
                "_std_setpoint": 2.0,
                "_name": "Length",
                "description": "Length with setpoint"
            },
            "rho_with_setpoint": {
                "_sym": "\\rho",
                "_alias": "rho",
                "_fwk": "PHYSICAL",
                "_dims": "M*L^-3",
                "_setpoint": 4.0,
                "_std_setpoint": 4.0,
                "_name": "Density",
                "description": "Density with setpoint"
            },
            "mu_with_setpoint": {
                "_sym": "\\mu",
                "_alias": "mu",
                "_fwk": "PHYSICAL",
                "_dims": "M*L^-1*T^-1",
                "_setpoint": 0.5,
                "_std_setpoint": 0.5,
                "_name": "Viscosity",
                "description": "Viscosity with setpoint"
            },
            "v_no_setpoint": {
                "_sym": "v",
                "_alias": "v",
                "_fwk": "PHYSICAL",
                "_dims": "L*T^-1",
                "_name": "Velocity",
                "description": "Velocity without setpoint"
            }
        },
        "SETPOINT_TEST_VALUES": {
            "simple_three_vars": {
                "v": 2.0,
                "L": 3.0,
                "\\rho": 4.0
            },
            "two_vars": {
                "v": 5.0,
                "L": 2.0
            },
            "reynolds_vars": {
                "v": 10.0,
                "L": 2.0,
                "\\rho": 4.0,
                "\\mu": 0.5
            },
            "wrong_var_names": {
                "v": 2.0,
                "L": 3.0,
                "\\mu": 4.0  # Should be \\rho
            }
        },
        "SETPOINT_EXPECTED_RESULTS": {
            "simple_three_vars": 1.5,  # (2.0^1) * (3.0^1) * (4.0^-1) = 2 * 3 / 4 = 1.5
            "two_vars": 2.5,  # (5.0^1) * (2.0^-1) = 5.0 / 2.0 = 2.5
            "reynolds": 160.0  # (10.0^1) * (2.0^1) * (4.0^1) * (0.5^-1) = 10 * 2 * 4 / 0.5 = 160
        }
    }


# test data for from pydasa.dimensional.model import Matrix
def get_model_test_data() -> dict:
    """Get test data for Matrix/model tests."""
    return {
        "TEST_VARIABLES": {
            "v": {
                "_sym": "v",
                "_alias": "v",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L*T^-1",
                "_units": "m/s",
                "_name": "Velocity",
                "description": "Fluid velocity",
                "relevant": True,
                "_idx": 0
            },
            "L": {
                "_sym": "L",
                "_alias": "L",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L",
                "_units": "m",
                "_name": "Length",
                "description": "Characteristic length",
                "relevant": True,
                "_idx": 1
            },
            "\\rho": {
                "_sym": "\\rho",
                "_alias": "rho",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "M*L^-3",
                "_units": "kg/m^3",
                "_name": "Density",
                "description": "Fluid density",
                "relevant": True,
                "_idx": 2
            },
            "\\mu": {
                "_sym": "\\mu",
                "_alias": "mu",
                "_fwk": "PHYSICAL",
                "_cat": "CTRL",
                "_dims": "M*L^-1*T^-1",
                "_units": "Pa*s",
                "_name": "Dynamic Viscosity",
                "description": "Dynamic viscosity",
                "relevant": True,
                "_idx": 3
            },
            "P": {
                "_sym": "P",
                "_alias": "P",
                "_fwk": "PHYSICAL",
                "_cat": "OUT",
                "_dims": "M*L^-1*T^-2",
                "_units": "Pa",
                "_name": "Pressure",
                "description": "Fluid pressure",
                "relevant": True,
                "_idx": 4
            },
            "d": {
                "_sym": "d",
                "_alias": "d",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L",
                "_units": "m",
                "_name": "Diameter",
                "description": "Pipe diameter",
                "relevant": False,  # Not relevant for basic analysis
                "_idx": 5
            },
            "g": {
                "_sym": "g",
                "_alias": "g",
                "_fwk": "PHYSICAL",
                "_cat": "IN",
                "_dims": "L*T^-2",
                "_units": "m/s^2",
                "_name": "Gravity",
                "description": "Gravitational acceleration",
                "relevant": True,
                "_idx": 6
            },
            "\\nu": {
                "_sym": "\\nu",
                "_alias": "nu",
                "_fwk": "PHYSICAL",
                "_cat": "CTRL",
                "_dims": "L^2*T^-1",
                "_units": "m^2/s",
                "_name": "Kinematic Viscosity",
                "description": "Kinematic viscosity",
                "relevant": True,
                "_idx": 7
            }
        },
        "MINIMAL_VARIABLES": {
            "v": {
                "_sym": "v",
                "_cat": "IN",
                "_dims": "L*T^-1",
                "relevant": True
            },
            "P": {
                "_sym": "P",
                "_cat": "OUT",
                "_dims": "M*L^-1*T^-2",
                "relevant": True
            }
        },
        "VARIABLES_NO_OUTPUT": {
            "v": {
                "_sym": "v",
                "_cat": "IN",
                "_dims": "L*T^-1",
                "relevant": True
            },
            "L": {
                "_sym": "L",
                "_cat": "IN",
                "_dims": "L",
                "relevant": True
            }
        },
        "VARIABLES_MULTI_OUTPUT": {
            "v": {
                "_sym": "v",
                "_cat": "IN",
                "_dims": "L*T^-1",
                "relevant": True
            },
            "P1": {
                "_sym": "P1",
                "_cat": "OUT",
                "_dims": "M*L^-1*T^-2",
                "relevant": True
            },
            "P2": {
                "_sym": "P2",
                "_cat": "OUT",
                "_dims": "M*L^-1*T^-2",
                "relevant": True
            }
        },
        "EXPECTED_FDU_SYMBOLS": ["M", "L", "T"],
        "EXPECTED_CATEGORIES": ["IN", "OUT", "CTRL"],
        "VALID_MODEL_NAMES": [
            "Fluid Dynamics Model",
            "Reynolds Number Analysis",
            "Dimensional Matrix Test"
        ],
        "EXPECTED_N_COEFFICIENTS": 2,  # Expected number of Pi coefficients
        "VALID_DERIVED_EXPRESSIONS": [
            "\\Pi_{0} * \\Pi_{1}",
            "\\Pi_{0} / \\Pi_{1}",
            "\\Pi_{0}^2 * \\Pi_{1}"
        ],
        "INVALID_DERIVED_EXPRESSIONS": [
            "invalid * expression",
            "\\Pi_{999} * \\Pi_{1000}",
            ""
        ],
    }


# test data for from pydasa.elements.specs.numerical import BoundsSpecs, StandardizedSpecs, NumericalSpecs
def get_numerical_test_data() -> dict:
    """Get test data for numerical specs tests."""
    return {
        "BOUNDS_VALUES": {
            "setpoint": 5.0,
            "min": 0.0,
            "max": 10.0,
            "mean": 5.0,
            "median": 5.0,
            "dev": 2.0,
        },
        "STANDARDIZED_VALUES": {
            "std_setpoint": 50.0,
            "std_min": 0.0,
            "std_max": 100.0,
            "std_mean": 50.0,
            "std_median": 50.0,
            "std_dev": 20.0,
        },
        "DISCRETIZATION_VALUES": {
            "step": 0.1,
        },
        "INVALID_VALUES": {
            "type_error": "invalid",
            "out_of_range_high": 999.0,
            "out_of_range_low": -999.0,
            "negative_dev": -5.0,
            "zero_step": 0.0,
            "large_step": 50.0,
        },
    }


# test data for from pydasa.analysis.simulation import MonteCarlo
def get_simulation_test_data() -> dict:
    """Get test data for simulation tests."""
    return {
        "FDU_LIST": [
            {
                "_idx": 0,
                "_sym": "M",
                "_fwk": "CUSTOM",
                "description": "Mass",
                "_unit": "kg",
                "_name": "Mass"
            },
            {
                "_idx": 1,
                "_sym": "L",
                "_fwk": "CUSTOM",
                "description": "Length",
                "_unit": "m",
                "_name": "Length"
            },
            {
                "_idx": 2,
                "_sym": "T",
                "_fwk": "CUSTOM",
                "description": "Time",
                "_unit": "s",
                "_name": "Time"
            },
        ],
        "CHANNEL_FLOW_VARIABLES": {
            "U": {
                "_sym": "U",
                "_alias": "U",
                "_fwk": "CUSTOM",
                "_name": "Wall Velocity",
                "relevant": True,
                "description": "Velocity of the fluid wall",
                "_idx": 3,
                "_cat": "IN",
                "_units": "m/s",
                "_dims": "L*T^-1",
                "_min": 0.0,
                "_max": 15.0,
                "_mean": 7.50,
                "_dev": 0.75,
                "_std_units": "m/s",
                "_std_min": 0.0,
                "_std_max": 15.0,
                "_std_mean": 7.50,
                "_std_dev": 0.75,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 15.0},
                "_depends": [],
            },
            "\\mu_{1}": {
                "_sym": "\\mu_{1}",
                "_alias": "mu_1",
                "_fwk": "CUSTOM",
                "_name": "Fluid Viscosity",
                "description": "Dynamic viscosity of fluid",
                "relevant": True,
                "_idx": 0,
                "_cat": "OUT",
                "_units": "m/s",
                "_dims": "L*T^-1",
                "_min": 0.0,
                "_max": 15.0,
                "_mean": 7.50,
                "_dev": 0.75,
                "_std_units": "m/s",
                "_std_min": 0.0,
                "_std_max": 15.0,
                "_std_mean": 7.50,
                "_std_dev": 0.75,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 15.0},
                "_depends": ["U"],
            },
            "y_{2}": {
                "_sym": "y_{2}",
                "_alias": "y_2",
                "_fwk": "CUSTOM",
                "_name": "Distance from Wall",
                "description": "Distance from wall to measurement point",
                "relevant": True,
                "_idx": 1,
                "_cat": "IN",
                "_units": "m",
                "_dims": "L",
                "_min": 0.0,
                "_max": 10.0,
                "_mean": 5.0,
                "_dev": 0.50,
                "_std_units": "m",
                "_std_min": 0.0,
                "_std_max": 10.0,
                "_std_mean": 5.0,
                "_std_dev": 0.50,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 10.0},
                "_depends": [],
            },
            "d": {
                "_sym": "d",
                "_alias": "d",
                "_fwk": "CUSTOM",
                "_name": "Channel Diameter",
                "relevant": True,
                "description": "Diameter of the channel",
                "_idx": 2,
                "_cat": "IN",
                "_units": "m",
                "_dims": "L",
                "_min": 0.0,
                "_max": 5.0,
                "_mean": 2.5,
                "_dev": 0.25,
                "_std_units": "m",
                "_std_min": 0.0,
                "_std_max": 5.0,
                "_std_mean": 2.5,
                "_std_dev": 0.25,
                "_step": 0.1,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 5.0},
                "_depends": [],
            },
            "P": {
                "_sym": "P",
                "_alias": "P",
                "_fwk": "CUSTOM",
                "_name": "Pressure Drop",
                "relevant": True,
                "description": "Pressure drop across channel",
                "_idx": 4,
                "_cat": "CTRL",
                "_units": "Pa",
                "_dims": "T^-2*L^1",
                "_min": 0.0,
                "_max": 100000.0,
                "_mean": 50000.0,
                "_dev": 5000.0,
                "_std_units": "Pa",
                "_std_min": 0.0,
                "_std_max": 100000.0,
                "_std_mean": 50000.0,
                "_std_dev": 5000.0,
                "_step": 100.0,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 100000.0},
                "_depends": [],
            },
            "v": {
                "_sym": "v",
                "_alias": "v",
                "_fwk": "CUSTOM",
                "_name": "Kinematic Viscosity",
                "relevant": True,
                "description": "Kinematic viscosity of fluid",
                "_idx": 5,
                "_cat": "CTRL",
                "_units": "m^2/s",
                "_dims": "L^2*T^-1",
                "_min": 0.0,
                "_max": 1.0,
                "_mean": 0.5,
                "_dev": 0.05,
                "_std_units": "m^2/s",
                "_std_min": 0.0,
                "_std_max": 1.0,
                "_std_mean": 0.5,
                "_std_dev": 0.05,
                "_step": 0.01,
                "_dist_type": "uniform",
                "_dist_params": {"min": 0.0, "max": 1.0},
                "_depends": [],
            },
        }
    }


# test data for from pydasa.structs.tools.hashing import mad_hash
def get_hashing_test_data() -> dict:
    """Get test data for hashing tests."""
    return {
        "MAD_PARAMS": {
            "scale": 3,
            "shift": 5,
            "prime": 109345121,
            "mcap": 11,
        },
        "ALTERNATIVE_PARAMS": {
            "scale": 7,
            "shift": 13,
            "prime": 109345121,
            "mcap": 23,
        },
        "INTEGER_KEYS": [0, 1, 42, 100, -42, -1, 999],
        "STRING_KEYS": ["test", "a", "key1", "key2", "consistent", "example"],
        "FLOAT_KEYS": [3.14, 2.718, 0.0, -1.5, 100.5],
        "TUPLE_KEYS": [(1, 2), (1, 2, 3), ("a", "b"), ()],
        "LIST_KEYS": [[1, 2, 3], ["a", "b"], [], [1]],
        "SET_KEYS": [{1, 2, 3}, {"a", "b"}, set()],
        "DICT_KEYS": [
            {"a": 1, "b": 2},
            {"x": 5},
            {},
        ],
        "MIXED_KEYS": [0, 1, 100, "a", "test", 3.14, (1, 2), [3, 4], {"x": 5}],
    }


# test data for from pydasa.structs.tools.math import is_prime, next_prime, previous_prime, gfactorial
def get_math_test_data() -> dict:
    """Get test data for math utility tests."""
    return {
        "PRIME_NUMBERS": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        "NON_PRIME_NUMBERS": [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        "NEGATIVE_NUMBERS": [-1, -2, -5, -10],
        "NEXT_PRIME_CASES": [
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 5),
            (10, 11),
            (20, 23),
            (50, 53),
        ],
        "PREVIOUS_PRIME_CASES": [
            (0, 2),
            (1, 2),
            (3, 2),
            (5, 3),
            (10, 7),
            (20, 19),
            (50, 47),
        ],
        "FACTORIAL_INTEGERS": [
            (0, 1),
            (1, 1),
            (5, 120),
            (10, 3628800),
        ],
        "FACTORIAL_FLOATS": [
            (0.5, 0.8862269254527580),
            (-0.5, 1.7724538509055159),
        ],
        "FACTORIAL_PRECISION": [
            (0.5, 2, 0.89),
            (0.5, 4, 0.8862),
        ],
        "FACTORIAL_INVALID": [-1, -2, -5],
    }


# test data for from pydasa.structs.tools.memory import mem_slot
def get_memory_test_data() -> dict:
    """Get test data for memory utility tests."""
    return {
        "INVALID_INPUTS": [
            123,
            "not a class",
            3.14,
            None,
            [],
            {},
        ],
    }


# test data for from pydasa.structs.types.functions import dflt_cmp_function_lt, dflt_cmp_function_ht
def get_functions_test_data() -> dict:
    """Get test data for functions.py tests."""
    return {
        "INT_PAIRS": [
            (1, 2, -1),
            (5, 5, 0),
            (10, 3, 1),
        ],
        "FLOAT_PAIRS": [
            (1.5, 2.5, -1),
            (3.14, 3.14, 0),
            (10.0, 5.0, 1),
        ],
        "STRING_PAIRS": [
            ("apple", "banana", -1),
            ("test", "test", 0),
            ("zebra", "apple", 1),
        ],
        "DICT_PAIRS": [
            ({"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}, "age", -1),
            ({"name": "Alice", "age": 25}, {"name": "Bob", "age": 25}, "age", 0),
            ({"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, "age", 1),
        ],
        "INVALID_TYPE_PAIRS": [
            (1, "string"),
            (3.14, [1, 2]),
            ("text", {"key": "value"}),
        ],
        "INVALID_KEYS": ["missing_key", "invalid", "nonexistent"],
        "VALID_KEYS": ["name", "age", "value"],
    }
