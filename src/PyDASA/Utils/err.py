# -*- coding: utf-8 -*-
"""
General error handling module/function for the PyDASA Data Structures and Algorithms in PyDASA package.
"""
# native python modules
# custom modules
# import global variables


def error_handler(ctx: str,
                  fname: str,
                  err: Exception) -> None:
    """*error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

    Args:
        ctx (str): Python _context_ where the error occurred (package/class).
        fname (str): Name of the function (method) where the error occurred.
        err (Exception): Python _exception_ that was raised.

    Raises:
        type: Exception with the detailed error message and traceback.
    """
    # check if the context is a valid string
    if not isinstance(ctx, str):
        _msg = f"Invalid context: {ctx}, "
        _msg += "context must be a string"
        raise TypeError(_msg)
    # check if the function name is a valid string
    if not isinstance(fname, str):
        _msg = f"Invalid function name: {fname}, "
        _msg += "function name must be a string"
        raise TypeError(_msg)
    # check if the error is a valid exception
    if not isinstance(err, Exception):
        _msg = f"Invalid error: {err}, "
        _msg += "error must be an exception"
        raise TypeError(_msg)
    # check if the context is not empty
    err_msg = f"Error in {ctx}.{fname}: {err}"
    raise type(err)(err_msg).with_traceback(err.__traceback__)
