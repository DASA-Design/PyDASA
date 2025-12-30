# -*- coding: utf-8 -*-
"""
Module io.py
===========================================

Module for input/output operations in *PyDASA*.

This module provides functions for reading and writing data to files, handling different data formats, and ensuring compatibility with the data structures used in *PyDASA*.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""
# python native modules
import json
import os
from typing import Any, Dict


def load(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file.

    Reads data from a JSON file and returns it as a dictionary.
    The file must exist and contain valid JSON data.
    Automatically handles UTF-8 BOM (Byte Order Mark) if present.

    Args:
        file_path (str): Path to the JSON file to load.

    Returns:
        Dict[str, Any]: Dictionary containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        IOError: If there is an error reading the file.

    Examples:
        >>> data = load('data.json')
        >>> print(data)
        {'key': 'value', 'number': 42}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as err:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {file_path}: {err.msg}",
            err.doc,
            err.pos
        )
    except IOError as err:
        raise IOError(f"Error reading file {file_path}: {err}")


def save(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file.

    Writes a dictionary to a JSON file with proper formatting.
    Creates parent directories if they don't exist.

    Args:
        data (Dict[str, Any]): Dictionary containing the data to save.
        file_path (str): Path to the JSON file where data will be saved.

    Raises:
        TypeError: If data is not JSON serializable.
        IOError: If there is an error writing to the file.

    Examples:
        >>> data = {'key': 'value', 'number': 42}
        >>> save(data, 'output.json')
    """
    # Create parent directories if they don't exist
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except TypeError as err:
        raise TypeError(f"Data is not JSON serializable: {err}")
    except IOError as err:
        raise IOError(f"Error writing to file {file_path}: {err}")
