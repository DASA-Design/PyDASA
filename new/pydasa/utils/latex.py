# -*- coding: utf-8 -*-
"""
Module for default global variables and comparison functions for use by all *PyDASA* and its Data Structs.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""


# python native modules
# from dataclasses import dataclass
# from typing import TypeVar
import re

# custom modules
from sympy.parsing.latex import parse_latex
from sympy import symbols, lambdify

# import global variables
from new.pydasa.utils.config import LATEX_RE

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
        dict: A dictionary mapping LaTeX variable names to Python-compatible aliases. 'var name': 'python alias'.
    """
    # # Step 1: Extract latex variable names with regex
    # latex_syms = re.findall(LATEX_RE, str(expr))
    # # Step 2: Filter out ignored LaTeX commands
    # latex_syms = [m for m in latex_syms if m not in IGNORE_EXPR]
    # # Step 3: Use a set to ensure uniqueness
    # py_vars = list(set(latex_syms))
    # # Step 4: Convert LaTeX names to Python-compatible aliases
    # # This involves removing the backslash and converting subscript notation
    # py_vars = [m.lstrip("\\") for m in py_vars]
    # py_vars = [m.replace("_{", "_").replace("}", "") for m in py_vars]
    # # Step 5: Sort the variable names and aliases
    # # This ensures consistent ordering for reproducibility
    # py_vars.sort()
    # # Step 6: Return the mapping
    # return py_vars

    # Extract latex variable names with regex
    matches = re.findall(LATEX_RE, str(expr))

    # Filter out ignored LaTeX commands
    matches = [m for m in matches if m not in IGNORE_EXPR]

    # Create mappings both ways
    latex_to_py = {}
    py_to_latex = {}

    for m in matches:
        # Convert to Python style for internal use
        py_var = m.lstrip("\\").replace("_{", "_").replace("}", "")
        # Keep original LaTeX notation for external reference
        latex_var = m

        latex_to_py[latex_var] = py_var
        py_to_latex[py_var] = latex_var

    return latex_to_py, py_to_latex


def create_latex_mapping(expr: str) -> tuple[dict]:
    """*create_latex_mapping()* Create a mapping between LaTeX symbols and Python symbols.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
            - The first dictionary maps LaTeX symbols to Python symbols.
            - The second dictionary maps Python variable names to their corresponding sympy symbols.
    """
    # # Parse to get LaTeX symbols
    # matches = parse_latex(expr)
    # latex_symbols = matches.free_symbols
    # print(f"freee symbols! {latex_symbols}")

    # # Get Python variable names
    # py_vars = extract_latex_vars(expr)
    # print(f"Symbol mapping: {py_vars}")

    # # Create mapping
    # symbol_map = {}
    # py_symbol_map = {}

    # for latex_sym in latex_symbols:
    #     latex_name = str(latex_sym)
    #     # Find corresponding Python name
    #     for py_var in py_vars:
    #         # Check if this Python var corresponds to the LaTeX symbol
    #         # Direct match
    #         con1 = (latex_name == py_var)
    #         # Subscript conversion
    #         con2 = (latex_name.replace('_{', '_').replace('}', '') == py_var)
    #         # Remove backslash
    #         con3 = (latex_name.replace('\\', '') == py_var)
    #         # Check for correspondence between LaTeX and Python names
    #         if (con1 or con2 or con3):
    #             # Add to mapping if any condition matches
    #             symbol_map[latex_sym] = symbols(py_var)
    #             py_symbol_map[py_var] = symbols(py_var)
    #             break

    # return symbol_map, py_symbol_map
# Get LaTeX<->Python variable mappings
    latex_to_py, py_to_latex = extract_latex_vars(expr)
    print(f"Symbol mapping: {latex_to_py}")
    print(f"Python symbols: {py_to_latex}")

    # Parse to get LaTeX symbols
    matches = parse_latex(expr)
    latex_symbols = matches.free_symbols
    print(f"freee symbols! {latex_symbols}")

    # Create mapping for sympy substitution
    symbol_map = {}         # For internal substitution
    py_symbol_map = {}      # For lambdify
    latex_symbol_map = {}   # For result keys

    for latex_sym in latex_symbols:
        latex_name = str(latex_sym)

        # Find corresponding Python name
        for latex_var, py_var in latex_to_py.items():
            # Check for various forms of equivalence
            con1 = (latex_name == latex_var)
            con2 = (latex_name == py_var)
            con3 = (latex_name.replace('_{', '_').replace('}', '') == py_var)
            if con1 or con2 or con3:
                # Create symbol for this variable
                sym = symbols(py_var)
                # Store mappings
                symbol_map[latex_sym] = sym  # For substitution
                py_symbol_map[py_var] = sym  # For lambdify args
                latex_symbol_map[latex_var] = sym  # For original notation
                break
    
    return symbol_map, py_symbol_map, latex_to_py, py_to_latex


def parse_latex_symbols_mine(expr: str) -> tuple:
    """*parse_latex_symbols()* Parse a LaTeX expression and extract variable names.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        tuple: A tuple containing the parsed expression and a list of variable names in alphabetical order.
    """
    # Extract variable names
    py_vars = set(sorted(extract_latex_vars(expr)))

    # Step 4: Parse the LaTeX expression
    expr_sym = parse_latex(expr)

    # Step 5: Create symbol mapping between LaTeX symbols and Python symbols
    symbol_map = {}
    py_symbols = {var: symbols(var) for var in py_vars}

    for latex_sym in expr_sym.free_symbols:
        latex_name = str(latex_sym)
        # Find corresponding Python variable
        for py_var in py_vars:
            # direct match
            con1 = (latex_name == py_var)
            # subscript conversion
            con2 = (latex_name.replace('_{', '_').replace('}', '') == py_var)
            # remove backslash
            con3 = (latex_name.lstrip('\\') == py_var)
            # Check for correspondence between LaTeX and Python names
            if con1 or con2 or con3:
                # Add to mapping if any condition matches
                symbol_map[latex_sym] = py_symbols[py_var]
                break

    # Step 6: Substitute LaTeX symbols with Python symbols
    for latex_sym, py_sym in symbol_map.items():
        expr_sym = expr_sym.subs(latex_sym, py_sym)

    sym_list = [symbols(v) for v in py_vars]
    # Ensure we create a numerical function that returns floats, not symbolic expressions
    exec_fun = lambdify(sym_list, expr_sym, "numpy")
    return expr_sym, exec_fun, py_vars
