# -*- coding: utf-8 -*-
"""
Module to support other tests in PyDASA.
===========================================

Tests data for all the classes in PyDASA.
"""
# import testing package
# import unittest
# import pytest


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
        "REQUIRED_FIELDS": ["_unit", "name", "description"],
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
            "name": "Length",
            "description": "Physical length dimension"
        },
        "COMPUTATION_DATA": {
            "_idx": 1,
            "_sym": "S",
            "_alias": "S",
            "_fwk": "COMPUTATION",
            "_unit": "bit",
            "name": "Storage",
            "description": "Digital storage dimension"
        },
        "SOFTWARE_DATA": {
            "_idx": 2,
            "_sym": "E",
            "_alias": "E",
            "_fwk": "SOFTWARE",
            "_unit": "req",
            "name": "Events",
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


# test data for from pydasa.core.basic import SymValidation et al
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
                "name": "Length",
                "description": "physical length dimension"
            },
            {
                "_idx": 1,
                "_sym": "M",
                "_alias": "M",
                "_fwk": "PHYSICAL",
                "name": "Mass",
                "description": "physical mass dimension"
            },
        ],
        "VALID_NAMES": ["Length", "Mass", "Time", "Storage"],
        "INVALID_NAMES": [123, [], {}, None, True],  # Non-string types
    }
