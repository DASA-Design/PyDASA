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
from sympy import symbols

# import global variables
from new.pydasa.utils.config import LATEX_RE


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
    # ans = re.sub(LATEX_RE, r"\1_\2", expr)
    ans = re.sub(r'\\([a-zA-Z]+)_{(\d+)}', r'\1_\2', expr)
    return ans



def parse_latex_symbols(expr: str) -> tuple:
    """*parse_latex_symbols()* Parse a LaTeX expression and extract variable names.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        tuple: A tuple containing the parsed expression and a list of variable names in alphabetical order.
    """
    # Extract variable names
    all_vars = set(extract_latex_vars(expr))
    # create sympy symbols dict for all variables
    sym_dt = {name: symbols(name) for name in all_vars}
    # parse the LaTeX expression
    expr = parse_latex(expr)
    # print(all_vars)
    # iterate over the LaTeX variable names and replace them with sympy symbols
    for var in all_vars:
        # convert Python var back to LaTeX for matching
        # if the variable name contains a subscript, replace it with the corresponding LaTeX symbol
        if "_" in var:
            base, sub = var.split("_")
            latex_sym = symbols(f"{base}_{{{sub}}}")
        # otherwise, use the variable name as is
        else:
            latex_sym = symbols(var)
        # replace the LaTeX symbol with the corresponding sympy symbol
        expr = expr.subs(latex_sym, symbols(var))
    # substitute all variables with our symbols (ensures correct mapping)
    expr = expr.subs(sym_dt)
    # sort the variable names in alphabetical order
    # print(f"Parsed: {expr}", all_vars)
    return expr, all_vars


def extract_latex_vars(expr: str) -> list:
    """*extract_latex_vars()* Extract variable names from a LaTeX expression.
    This function uses a regular expression to match LaTeX symbols (e.g., '\alpha', '\beta_{1}') in the expression.

    Args:
        expr (str): The LaTeX expression to parse.

    Returns:
        list: list of variable names in the expression.
    """
    # Matches names like l_{1}, W_{2}, L_{1}, N_{2}, u, l, x, and LaTeX commands
    # print(expr, LATEX_RE)
    matches = re.findall(LATEX_RE, expr)
    ignore = {"\\frac", "\\sqrt", "\\sin", "\\cos",
              "\\tan", "\\log", "\\exp"}  # add more as needed
    # Convert LaTeX subscript to Python style, e.g., l_{1} -> l_1, and filter ignored commands
    py_vars = [m.replace("_{", "_").replace("}", "")
               for m in matches if m not in ignore]
    return py_vars
