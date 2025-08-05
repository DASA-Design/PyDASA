# -*- coding: utf-8 -*-
"""
Module latex.py
===========================================

Module for default global variables and comparison functions for use by all *PyDASA* and its Data Structs.
"""
# python native modules
# from dataclasses import dataclass
# from typing import TypeVar
import re

# custom modules
from sympy.parsing.latex import parse_latex
from sympy import symbols

# import global variables
from src.pydasa.utils.config import LATEX_RE

# Global vars for special characters
IGNORE_EXPR = {
    "\\frac",
    "\\sqrt",
    "\\sin",
    "\\cos",
    "\\tan",
    "\\log",
    "\\exp"
}


def latex_to_python(expr: str) -> str:
    """*latex_to_python()* Convert a LaTeX expression to a Python-compatible string.

    Args:
        expr (str): The LaTeX expression to convert.

    Returns:
        str: The Python-compatible string.
    """
    # Replace LaTeX subscript with Python style
    if expr.isalnum():
        return expr
    # TODO this regex doesnt work, check latter
    # ans = re.sub(r"\\([a-zA-Z]+)_{(\d+)}", r"\1_\2", expr)
    alias = expr.replace("\\", "")
    alias = alias.replace("_{", "_").replace("}", "")
    return alias


def extract_latex_vars(expr: str) -> tuple[dict]:
    """*extract_latex_vars()* Extract variable names in LaTeX format with their Python equivalents.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        tuple [dict]: A tuple containing two dictionaries:
                - The first dictionary maps LaTeX variable names to their Python equivalents.
                - The second dictionary maps Python variable names to their LaTeX equivalents.
    """
    # Extract latex variable names with regex
    matches = re.findall(LATEX_RE, str(expr))

    # Filter out ignored LaTeX commands
    matches = [m for m in matches if m not in IGNORE_EXPR]

    # Create mappings both ways
    latex_to_py = {}
    py_to_latex = {}

    for m in matches:
        # Keep original LaTeX notation for external reference
        latex_var = m
        # Convert to Python style for internal use
        py_var = m.lstrip("\\").replace("_{", "_").replace("}", "")

        latex_to_py[latex_var] = py_var
        py_to_latex[py_var] = latex_var

    return latex_to_py, py_to_latex


def create_latex_mapping(expr: str) -> tuple[dict]:
    """*create_latex_mapping()* Create a mapping between LaTeX symbols and Python symbols.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        tuple[dict]: A tuple containing:
            - A dictionary mapping LaTeX symbols to Python symbols for internal substitution.
            - A dictionary mapping Python variable names to their corresponding sympy symbols for lambdify.
            - A dictionary mapping LaTeX variable names to their Python equivalents.
            - A dictionary mapping Python variable names to their LaTeX equivalents.
    """
    # Get LaTeX<->Python variable mappings
    latex_to_py, py_to_latex = extract_latex_vars(expr)

    # Parse to get LaTeX symbols
    matches = parse_latex(expr)
    latex_symbols = matches.free_symbols

    # Create mapping for sympy substitution
    symbol_map = {}         # For internal substitution
    py_symbol_map = {}      # For lambdify
    latex_symbol_map = {}   # For result keys

    for latex_sym in latex_symbols:
        latex_name = str(latex_sym)

        # Find corresponding Python name
        for latex_var, py_var in latex_to_py.items():
            # Check for various forms of equivalence
            # con1: exact match with LaTeX
            # con2: exact match with Python variable
            # con3: match with LaTeX subscript style
            con1 = (latex_name == latex_var)
            con2 = (latex_name == py_var)
            con3 = (latex_name.replace("_{", "_").replace("}", "") == py_var)
            if con1 or con2 or con3:
                # Create symbol for this variable
                sym = symbols(py_var)
                # Store mappings
                symbol_map[latex_var] = sym  # For substitution
                py_symbol_map[py_var] = sym  # For lambdify args
                latex_symbol_map[latex_var] = sym  # For original notation
                break

    return symbol_map, py_symbol_map, latex_to_py, py_to_latex
