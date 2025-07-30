# -*- coding: utf-8 -*-
# FIXME old code, remove when tests are finished

"""
General error handling module/function for the PyDASA Data Structures and Algorithms in PyDASA package.
"""
# native python modules
import inspect
from typing import Any
# custom modules
# import global variables


def error_handler(context: str,
                  function_name: str,
                  exception: Exception) -> None:
    """Handles errors by formatting a detailed error message and traceback.

    Args:
        context (str): The context (e.g., package/class) where the error occurred.
        function_name (str): The name of the function or method where the error occurred.
        exception (Exception): The exception that was raised.

    Raises:
        Exception: A new exception with a detailed error message and traceback.
    """
    # Validate the context
    if not isinstance(context, str):
        _msg = f"Invalid context: {context}. Context must be a string."
        raise TypeError(_msg)

    # Validate the function name
    if not isinstance(function_name, str):
        _msg = f"Invalid function name: {function_name}. "
        _msg += "Function name must be a string."
        raise TypeError(_msg)

    # Validate the exception
    if not isinstance(exception, Exception):
        _msg = f"Invalid exception: {exception}. "
        _msg += "Exception must be an instance of Exception."
        raise TypeError(_msg)

    # Format and raise the error with additional context
    _err_msg = f"Error in {context}.{function_name}: {exception}"
    raise type(exception)(_err_msg).with_traceback(exception.__traceback__)


def inspect_name(var: Any) -> str:
    """*inspect_name() inspect a variable an gets its name in the source code.

    Args:
        var (Any): The variable to inspect.

    Returns:
        str: The name of the variable.
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
    return "Variable name not found"

    # FIXME old code, keep until we know the new version works
    # frame = inspect.currentframe().f_back
    # for name, value in frame.f_locals.items():
    #     if value is var:
    #         return name


#  Example usage
# lt = [1, 2, 3]
# variable_name = inspect_name(lt)
# print(f"The name of the variable is: {variable_name}")
