# -*- coding: utf-8 -*-
"""
General error handling module/function for the PyDASA Data Structures and Algorithms in PyDASA package.
"""
# native python modules
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
