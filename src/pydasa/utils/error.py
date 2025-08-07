# -*- coding: utf-8 -*-
"""
Module error.py
===========================================

General error handling module/function for the PyDASA Data Structures and Algorithms in PyDASA package.
"""
# native python modules
import inspect
from typing import Any
# custom modules
# import global variables


def handle_error(ctx: str, func: str, exc: Exception) -> None:
    """*handle_error()* generic function to handle errors iacross the whole PyDASA library.

    Args:
        ctx (str): The context (e.g., package/class) where the error occurred.
        func (str): The name of the function or method where the error occurred.
        exc (Exception): The exception that was raised.

    Raises:
        TypeError: If the context is not a string.
        TypeError: If the function name is not a string.
        TypeError: If the exception is not an instance of Exception.
        type: If the error message is not a string.
    """
    # Validate the context
    if not isinstance(ctx, str):
        _msg = f"Invalid context: {ctx}. Context must be a string."
        raise TypeError(_msg)

    # Validate the function name
    if not isinstance(func, str):
        _msg = f"Invalid function name: {func}. "
        _msg += "Function name must be a string."
        raise TypeError(_msg)

    # Validate the exception
    if not isinstance(exc, Exception):
        _msg = f"Invalid exception: {exc}. "
        _msg += "Exception must be an instance of Exception."
        raise TypeError(_msg)

    # Format and raise the error with additional context
    _err_msg = f"Error in {ctx}.{func}: {exc}"
    raise type(exc)(_err_msg).with_traceback(exc.__traceback__)


def inspect_var(var: Any) -> str:
    """*inspect_var() inspect a variable an gets its name in the source code.

    Args:
        var (Any): The variable to inspect.

    Returns:
        str: The name of the variable.

    Raises:
        ValueError: If the variable name cannot be found in the current scope.
    """
    frame = inspect.currentframe().f_back
    for name, value in frame.f_locals.items():
        # Check if the variable matches by identity or value
        if value is var:
            return name
    # If the variable is an object, try to find its name in the local scope
    for name, value in frame.f_locals.items():
        if id(value) == id(var):
            return name
    raise ValueError("Variable name not found in the current scope.")
    # return "Variable name not found"

    # FIXME old code, keep until we know the new version works
    # frame = inspect.currentframe().f_back
    # for name, value in frame.f_locals.items():
    #     if value is var:
    #         return name
