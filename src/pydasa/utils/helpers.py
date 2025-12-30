# -*- coding: utf-8 -*-
"""
Module helpers.py
===========================================

Module with utility functions for handling memory allocation in the Data Structures of *PyDASA*.

Module with utility functions for handling data in the maps of *PyDASA*. Specifically for Separate Chaining and Linear Probing Hash Tables.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    #. Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    #. Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.

*NOTE:* code contributed by Sanjit_Prasad in https://www.geeksforgeeks.org/prime-numbers/
"""
# python native modules
import math
from typing import Union, Optional

# dataclases module handles the creation of classes with slots and fields
import dataclasses

# import global variables
from pydasa.utils.default import VLD_IODATA_LT
from pydasa.types.generics import T


# Memory Helpers

def slot_dataclass(cls) -> dataclasses.dataclass:
    """slot_dataclass is a decorator that converts a class into a dataclass with slots.

    a slot is a special kind of attribute that is used to store data in a class.
    It is used to optimize memory usage and improve performance by reducing the overhead of the class.

    Returns:
        dataclasses.dataclass: A dataclass with slots.
    """
    # TODO check vialidity of this decorator
    # TODO integrate with the dataclass decorator
    # check if the class is a valid class
    if not isinstance(cls, type):
        raise TypeError(f"Invalid class: {cls}, class must be a type")
    # check if the class is a dataclass
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"Invalid class: {cls}, class must be a dataclass")
    # allocate slots for the class
    cls.__slots__ = [f.name for f in dataclasses.fields(cls)]
    return dataclasses.dataclass(cls)


# Hash Table Helpers

def is_prime(n: int) -> bool:
    """*is_prime()* checks if a number is prime or not. Original code from Sanjit_Prasad.

    Args:
        n (int): number to check if it is prime.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    # we asume that the number is prime
    # Corner cases
    # check if n is 1 or 0
    prime = True
    if n < 2:
        return False

    # checking if n is 2 or 3
    if n < 4:
        return prime

    # checking if n is divisible by 2 or 3
    if n % 2 == 0 or n % 3 == 0:
        return False

    # checking if n is divisible by 5 to to square root of n
    for i in range(5, int(math.sqrt(n) + 1), 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    # return True if the number is prime
    return prime


def next_prime(n: int) -> int:
    """*next_prime()* returns the next prime number greater than n.

    Args:
        n (int): number to check if it is prime.

    Returns:
        int: the next prime number greater than n.
    """
    # base case
    if n < 2:
        return 2

    # working with the next odd number
    prime = n
    found = False

    # Loop continuously until isPrime returns
    while not found:
        prime += 1
        # True for a prime number greater than n
        if is_prime(prime) is True:
            found = True
    # return the next prime number to n
    return prime


def previous_prime(n: int) -> int:
    """*previous_prime()* returns the previous prime number less than n.

    Args:
        n (int): number to check if it is prime.

    Returns:
        int: the previous prime number less than n.
    """
    # base case
    if n < 2:
        return 2

    # working with the next odd number
    prime = n
    found = False

    # Loop continuously until isPrime returns
    while not found:
        prime -= 1
        # True for a prime number greater than n
        if is_prime(prime) is True:
            found = True


def mad_hash(key: T,
             scale: int,
             shift: int,
             prime: int,
             mcap: int) -> int:
    """*mad_hash()* function to compress the indices of the Hash tables using the MAD (Multiply-Add-and-Divide) method.

    MAD is defined as: mad_hash(y) = ((a*y + b) % p) % M, where:
        a (scale) and b (shift) are random integers in the range [0,p-1], with a > 0
        p (prime) is a prime number greater than M,
        M (capacity) is the size of the table, prime

    Args:
        key (T): key to calculate the index in the Hash table, Can be any native data type in Python or user-defined.
        scale (int): line slope of the compression function.
        shift (int): offset of the compression function.
        prime (int): prime number much greater than the capacity of the Hash table.
        mcap (int): size of the Hash table, it is a prime number to avoid collisions.

    Returns:
        int: the index of the element in the Hash table.
    """
    # TODO is easier if we cast the dynamic keys to strings?
    # if it is a dynamic data type, we cast it to string
    # data types are (dict, list, set, tuple)
    if isinstance(key, VLD_IODATA_LT) or isinstance(key, dict):
        key = str(key)
    # getting the hash from the key
    hkey = hash(key)
    # calculating the index with the MAD compression function
    idx = int((abs(scale * hkey + shift) % prime) % mcap)
    return idx


def gfactorial(x: Union[int, float],
               prec: Optional[int] = None) -> Union[int, float]:
    """*gfactorial()* calculates the factorial of a number, including support for floats less than 1.0.

        - For integers n ≥ 0: Returns n! (n factorial).
        - For floats x: Returns Γ(x+1) (gamma function).

    Args:
        x (Union[int, float]): The number to compute the factorial for.
        prec (Optional[int], optional): precision, or the number of decimal places to round the result to. Defaults to None.

    Raises:
        ValueError: If x is a negative integer.

    Returns:
        Union[int, float]: The factorial of x. Returns an integer for integer inputs ≥ 0, and a float for float inputs or integers < 0.

    Examples:
        >>> gfactorial(5)
        120
        >>> gfactorial(0)
        1
        >>> gfactorial(0.5)  # Equivalent to Γ(1.5) = 0.5 * Γ(0.5) = 0.5 * √Pi
        0.8862269254527579
        >>> gfactorial(-0.5)  # Equivalent to Γ(0.5) = √Pi
        1.7724538509055159
    """
    if isinstance(x, int) and x >= 0:
        # Standard factorial for non-negative integers
        result = math.factorial(x)
    elif isinstance(x, int) and x < 0:
        # Factorial is not defined for negative integers
        raise ValueError("Factorial is not defined for negative integers")
    else:
        # For floats, use the gamma function: Γ(x+1)
        result = math.gamma(x + 1)

    # Apply precision if specified
    if prec is not None:
        result = round(result, prec)

    return result
