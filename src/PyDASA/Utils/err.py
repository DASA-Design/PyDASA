# -*- coding: utf-8 -*-
"""
# EN
General error handling module for Data Structures and Algorithms in Python (PyDASA) package.

# SPA
Módulo comum para manejar errores en las estructuras de datos y algoritmos en Python (PyDASA).
"""
# native python modules
# import typing for defining the type of the elements

# custom modules
# import global variables


def error_handler(context: str,
                  func_name: str,
                  err: Exception) -> None:
    """*error_handler()* recibe el contexto, nombre de la función y la excepción para lanzar un mensaje de error detallado y el traceback.

    Args:
        context (str): nombre del contexto donde ocurrió el error.
        func_name (str): nombre de la función donde ocurrió el error.
        err (Exception): excepción lanzada.

    Raises:
        type: excepción con el mensaje de error detallado y el traceback.
    """
    err_msg = f"Error in {context}.{func_name}: {err}"
    raise type(err)(err_msg).with_traceback(err.__traceback__)
