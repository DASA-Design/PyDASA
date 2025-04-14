# -*- coding: utf-8 -*-
"""
Module to represent the **SCHashTable** data structure for the **Hash Table** in *PyDASA*.

*IMPORTANT:* based on the implementations proposed by the following authors/books:

    # . Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne.
    # . Data Structure and Algorithms in Python, M.T. Goodrich, R. Tamassia, M.H. Goldwasser.
"""

# native python modules
# import dataclass to define the hash table
from dataclasses import dataclass, field
# import modules for defining the entries type in the hash table
from typing import List, Optional, Callable, Generic
# import inspect for getting the name of the current function
import inspect
# random module for the MAD compression function
import random

# custom modules
# generic error handling and type checking
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
# util functions for the hash table
from Src.PyDASA.Utils.nos import next_prime, previous_prime
from Src.PyDASA.Utils.nos import mad_hash
from Src.PyDASA.Utils.err import error_handler as error
# default cmp function for the hash table
from Src.PyDASA.Utils.dflt import dflt_cmp_func_ht
# default data type for the hash table
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.dflt import VLD_DTYPE_LT
from Src.PyDASA.Utils.dflt import DFLT_DICT_KEY
from Src.PyDASA.Utils.dflt import VLD_IODATA_LT
from Src.PyDASA.Utils.dflt import DFLT_PRIME

# checking custom modules
assert MapEntry
assert ArrayList
assert SingleLinkedList
assert next_prime
assert previous_prime
assert mad_hash
assert error
assert dflt_cmp_func_ht
assert T
assert VLD_DTYPE_LT
assert DFLT_DICT_KEY
assert VLD_IODATA_LT
assert DFLT_PRIME

# default load factor for separating chaining
# :data: DFLT_SC_ALPHA
DFLT_SC_ALPHA: float = 4.0
"""
Default load factor (*alpha*) for the *SCHashTable*, by default is 4.0.
"""

# :data: MAX_SC_ALPHA
MAX_SC_ALPHA: float = 8.0
"""
Maximum load factor (*alpha*) for the *SCHashTable*, by default is 8.0.
"""

# :data: MIN_SC_ALPHA
MIN_SC_ALPHA: float = 2.0
"""
Minimum load factor (*alpha*) for the *SCHashTable*, by default is 2.0.
"""


@dataclass
class Bucket(SingleLinkedList[T]):
    """**Bucket** class to represent a bucket in the **Hash Table** with the *Separate Chaining* method. The structure is based (inherits) on a custom singly linked list (*SingleLinkedList*) for *PyDASA*.

    Args:
        SingleLinkedList (dataclass): *PyDASA* custom class for a single linked list.
        Generic (T): Generic type for a Python data structure.
    """

    def __str__(self) -> str:
        """*__str__()* function to return a string representation of the *Bucket*. It also extends the *Node* class.

        Returns:
            str: string representation of the *Bucket*.
        """
        _str = super().__str__()
        return _str

    def __repr__(self) -> str:
        """*__repr__()* function to return a string representation of the *Bucket*. It also extends the *Node* class.

        Returns:
            str: string representation of the *Bucket*.
        """
        _str = super().__repr__()
        return _str


@dataclass
class SCHashTable(Generic[T]):

    # boolean to indicate if the hash table can be rehashed
    # :attr: rehashable
    rehashable: bool = True
    """
    Boolean to indicate if the hash table can be rehashed. By default is True.
    """

    # reserved space for the hash table
    # :attr: nentries
    nentries: int = 1
    """
    Inicial number of entries (n) for the hash table. By default is 1, but should be set according to the number of entries expected to be stored.

    NOTE: the reserved space (n) is NOT the capacity (M) of the hash table.
    """

    # starting capacity (M|m) for the hash table
    # :attr: mcapacity
    mcapacity: int = 1
    """
    The capacity (M) of the hash table. By default is 1, but should be set according to the number of entries expected to be stored.
    """

    # starting load factor (alpha) for the hash table
    # :attr: alpha
    alpha: Optional[float] = DFLT_SC_ALPHA
    """
    Load factor (*alpha*) for the hash table. By default is 4.0.

    NOTE: alpha = n/M (n: number of expected entries, M: capacity of the hash table).
    """

    # the cmp_function is used to compare emtries, not defined by default
    # :attr: cmp_function
    cmp_function: Optional[Callable[[T, T], int]] = None
    """
    Customizable comparison function for *SCHashTable* and its *MapEntry* objects. Defaults to *dflt_cmp_func_ht()* from *PyDASA*, but can be overridden by the user.
    """

    # actual place to store the entries in the hash table
    # :attr: hash_table
    hash_table: ArrayList[Bucket[T]] = field(default_factory=ArrayList)

    """
    Index of the hash table where the *Buckets* are stored. By default is an empty *ArrayList* initialized with the configured capacity (M).
    """
    # the key is used to compare entries, not defined by default
    # :attr: key
    key: Optional[str] = DFLT_DICT_KEY
    """
    Customizable key name for identifying elements in the *SCHashTable*. Defaults to *DFLT_DICT_KEY = '_id'* from *PyDASA*, but can be overridden by the user.
    """

    # prime number (P) for the MAD compression function
    # :attr: prime
    prime: Optional[int] = DFLT_PRIME
    """
    Prime number (P) for the MAD compression function. By default is 109345121, but can be overridden by the user.

    NOTE: the MAD compression function is: *h(k) = ((a*k + b) mod P) mod M*, where *a* and *b* are two random integers, *P* is a prime number and *M* is the hash table capacity.
    """

    # private scale (a) factor for the mad compression function
    # :attr: _scale
    _scale: Optional[int] = 1
    """
    MAD compression function scale factor (a). By default is 1, but can be overridden by the user.
    """
    # private shift (b) factor for the mad compression function
    # :attr: _shift
    _shift: Optional[int] = 0
    """
    MAD compression function shift factor (b). By default is 0, but can be overridden by the user.
    """

    # current factor (alpha) for the working hash table
    # :attr: _cur_alpha
    _cur_alpha: Optional[float] = 0.0
    """
    Current load factor (*alpha*) for the hash table. By default is 0.0, and it updates with each operation that modifies the structure.
    """

    # minimum load factor (alpha) for the hash table
    # :attr: min_alpha
    min_alpha: Optional[float] = MIN_SC_ALPHA
    """
    Minimum load factor (*alpha*) for the hash table. By default is 2.0. But can be overridden by the user.
    """

    # maximum load factor (alpha) for the hash table
    # :attr: max_alpha
    max_alpha: Optional[float] = MAX_SC_ALPHA
    """
    Maximum load factor (*alpha*) for the hash table. By default is 8.0. But can be overridden by the user.
    """

    # actual number of used entries (n) in the hash table
    # FIXME inconsistent use of _size and size()
    # :attr: _size
    _size: int = 0
    """
    Number of entries (*n*) in the hash table. By default is 0, but it updates with each operation that modifies the structure.
    """

    # :attr: collisions
    _collisions: Optional[int] = 0
    """
    Number of collisions in the hash table. By default is 0, but it updates with each operation that modifies the structure.
    """

    # the type of the entry keys in the hash table
    # :attr: _key_type
    _key_type: Optional[type] = None
    """
    Data type for the keys of the *MapEntry* (key-value pair) that contains the hash table, by default is *None* and is configured when loading the first record.
    """

    # the type of the entry values in the hash table
    # :attr: _value_type
    _value_type: Optional[type] = None
    """
    Data type for the values of the *MapEntry* (key-value pair) that contains the hash table, by default is *None* and is configured when loading the first record.
    """

    # input elements from python list
    # :attr: iodata
    iodata: Optional[List[T]] = None
    """
    Optional Python list for loading external data intho the *SCHashTable*. Defaults to *None* but can be provided during creation.
    """

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the *SCHashTable* after creation by setting attributes like *rehashable*, *mcapacity*, *alpha*, *cmp_function*, *key*, *prime*, *scale*, *shift*, and *iodata*.
        It also sets the default values for the *min_alpha* and *max_alpha* attributes, which are used to control the load factor of the hash table.

        *NOTE:* Special method called automatically after object creation.
        """
        try:
            # setting capacity
            self.mcapacity = next_prime(self.nentries // self.alpha)
            # setting scale and shift for MAD compression function
            self._scale = random.randint(1, self.prime - 1)
            self._shift = random.randint(0, self.prime - 1)
            # setting the compare function
            self.cmp_function = self.cmp_function or self.default_compare

            # initializing new hash table
            self.hash_table = ArrayList(cmp_function=self.cmp_function,
                                        key=self.key)
            i = 0
            # bulding buckets in the hash table
            while i < self.mcapacity:
                # bucket is a SingleLinkedList list
                _bucket = Bucket(cmp_function=self.cmp_function,
                                 key=self.key)
                # add the bucket to the hash table
                self.hash_table.append(_bucket)
                i += 1

            # setting the current load factor
            self._cur_alpha = self._size / self.mcapacity

            # checking the external input data type
            if isinstance(self.iodata, VLD_IODATA_LT):
                for entry in self.iodata:
                    # if is a dict, use the key type
                    if isinstance(entry, dict):
                        self.insert(entry.get(self.key), entry)
                    # otherwise, manage as data list
                    else:
                        self.insert(entry, entry)
            # clean input data
            self.iodata = None
            # TODO rethink this part
            # # fix discrepancies between size and number of entries (n).
            # if self._size != self.nentries:
            #     self.nentries = self._size
        except Exception as err:
            self._error_handler(err)

    def default_compare(self, key1: T, entry2: MapEntry) -> int:
        """*default_compare()* Default comparison function for the *SCHashTable* and its *MapEntry* objects. Compares the key of the *MapEntry* with the provided key *key1* and reurns:
            - 0 if they are equal.
            - 1 if the *MapEntry* key is less than *key1*.
            - -1 if the *MapEntry* key is greater than *key1*.

        Args:
            key1 (T): Key from the first *MapEntry* to compare.
            entry2 (MapEntry): Second *MapEntry* to compare.

        Returns:
            int: Comparison result.
        """
        try:
            # using the default compare function for the key
            return dflt_cmp_func_ht(self.key, key1, entry2)
        except Exception as err:
            self._error_handler(err)

    @property
    def size(self) -> int:
        """*size* Property to retrieve the number if entries (n) in the *SCHashTable*.
        Returns:
            int: Number of entries (n) in the *SCHashTable*.
        """
        return self._size

    @property
    def empty(self) -> bool:
        """*empty* Property to check if the *SCHashTable* has entries or not.

        Returns:
            bool: True if the *SCHashTable* is empty, False otherwise.
        """
        return self._size == 0

    @property
    def collisions(self) -> int:
        """*collisions* Property to retrieve the number of collisions in the *SCHashTable*.

        Returns:
            int: Number of collisions in the *SCHashTable*.
        """
        return self._collisions

    def clear(self) -> None:
        """*clear()* function to reset the *SCHashTable* to its initial state. It clears all the entries in the hash table and resets the size, collisions and current load factor.
        """
        try:
            # reset the size, collisions and current load factor
            self._size = 0
            self._collisions = 0
            self._cur_alpha = 0
            # clear the bukets in the hash table
            for _bucket in self.hash_table:
                _bucket.clear()
            # clear the hash table itself
            self.hash_table.clear()
        except Exception as err:
            self._error_handler(err)

    def insert(self, key: T, value: T) -> None:
        """insert _summary_

        Args:
            key (T): _description_
            value (T): _description_
        """
        try:
            # create a new entry for the hash table
            _new_entry = MapEntry(key, value)
            _idx = -1
            # cheking the type of the entry
            if self._check_type(_new_entry):
                # get the hash key for the entry
                _hash = mad_hash(key,
                                 self._scale,
                                 self._shift,
                                 self.prime,
                                 self.mcapacity)

                # checking the bucket
                _bucket = self.hash_table.get(_hash)
                # check if the bucket is empty
                if not _bucket.empty:
                    _idx = _bucket.index_of(key)
                # the entry is not in the bucket, add it and a collision
                # the entry is already in the bucket, update it
                if _idx > -1:
                    _bucket.update(_idx, _new_entry)
                # otherwise, is a new entry
                else:
                    if _bucket.size >= 1:
                        self._collisions += 1
                    _bucket.append(_new_entry)
                    self._size += 1
                    self._cur_alpha = self._size / self.mcapacity
                # check if the structure needs to be rehashed
                if self._cur_alpha >= self.max_alpha:
                    self.resize()
        except Exception as err:
            self._error_handler(err)

    def get_entry(self, key: T) -> Optional[MapEntry]:
        """get_entry _summary_

        Args:
            key (T): _description_

        Raises:
            IndexError: _description_

        Returns:
            Optional[MapEntry]: _description_
        """
        try:
            if self.empty:
                raise IndexError("Empty data structure")
            # assume the entry is not in the structure
            entry = None
            idx = -1
            # get the hash key for the entry
            _hash = mad_hash(key,
                             self._scale,
                             self._shift,
                             self.prime,
                             self.mcapacity)

            # checking the bucket
            _bucket = self.hash_table.get(_hash)
            # check if the bucket is empty
            if not _bucket.empty:
                idx = _bucket.index_of(key)
            # if the entry is in the bucket, return it
            if idx > -1:
                entry = _bucket.get(idx)
            return entry
        except Exception as err:
            self._error_handler(err)

    def get_bucket(self, key: T) -> Optional[Bucket]:
        """get_bucket _summary_

        Args:
            key (T): _description_

        Raises:
            IndexError: _description_

        Returns:
            Optional[Bucket]: _description_
        """
        try:
            if self.empty:
                raise IndexError("Empty data structure")
            # assume the entry is not in the structure
            _bucket = None
            # get the hash key for the entry
            _hash = mad_hash(key,
                             self._scale,
                             self._shift,
                             self.prime,
                             self.mcapacity)

            # recover the bucket
            _bucket = self.hash_table.get(_hash)
            # ceck if the bucket is empty
            if _bucket.empty:
                _bucket = None
            # otherwise, return the bucket
            return _bucket
        except Exception as err:
            self._error_handler(err)

    def is_present(self, key: T) -> bool:
        """is_present _summary_

        Args:
            key (T): _description_

        Raises:
            IndexError: _description_

        Returns:
            bool: _description_
        """
        try:
            if self.empty:
                raise IndexError("Empty data structure")
            # assume the entry is not in the structure
            found = False
            # use the MAD compression function to get the hash key
            _hash = mad_hash(key,
                             self._scale,
                             self._shift,
                             self.prime,
                             self.mcapacity)
            # look into the bucket
            _bucket = self.hash_table.get(_hash)
            _idx = _bucket.index_of(key)
            # if the entry is in the bucket, return True
            if _idx > -1:
                found = True
            return found
        except Exception as err:
            self._error_handler(err)

    def delete(self, key: T) -> Optional[MapEntry]:
        """delete _summary_

        Args:
            key (T): _description_

        Raises:
            IndexError: _description_
            IndexError: _description_

        Returns:
            Optional[MapEntry]: _description_
        """
        try:
            if self.empty:
                raise IndexError("Empty data structure")
            # assume the entry is not in the structure
            _entry = None
            _idx = -1
            # get the hash key for the entry
            _hash = mad_hash(key,
                             self._scale,
                             self._shift,
                             self.prime,
                             self.mcapacity)
            # checking the bucket
            _bucket = self.hash_table.get(_hash)
            # check if the bucket is not empty
            if not _bucket.empty:
                _idx = _bucket.index_of(key)
                # if the entry is in the bucket, remove it
                if _idx > -1:
                    _entry = _bucket.remove(_idx)
                    self.hash_table.update(_bucket, _hash)
                    # updating collisions
                    if _bucket.size > 1:
                        self._collisions -= 1
                    # updating size and alpha
                    self._size -= 1
                    self._cur_alpha = self._size / self.mcapacity
                # Otherwise, the entry is not in the map
                # TODO maybe i don't need this
                else:
                    raise IndexError(f"Entry for Key: {key} not found")
            if self._cur_alpha < self.min_alpha:
                self.resize()
            return _entry
        except Exception as err:
            self._error_handler(err)

    def keys(self) -> SingleLinkedList[T]:
        """keys _summary_

        Returns:
            SingleLinkedList[T]: _description_
        """
        try:
            _keys_lt = SingleLinkedList(key=self.key)
            for _bucket in self.hash_table:
                if not _bucket.empty:
                    for _entry in _bucket:
                        _keys_lt.append(_entry.key)
            return _keys_lt
        except Exception as err:
            self._error_handler(err)

    def values(self) -> SingleLinkedList[T]:
        """values _summary_

        Returns:
            SingleLinkedList[T]: _description_
        """
        try:
            _values_lt = SingleLinkedList(key=self.key)
            for _bucket in self.hash_table:
                if not _bucket.empty:
                    for _entry in _bucket:
                        _values_lt.append(_entry.value)
            return _values_lt
        except Exception as err:
            self._error_handler(err)

    def entries(self) -> SingleLinkedList[T]:
        """entries _summary_

        Returns:
            SingleLinkedList[T]: _description_
        """
        try:
            _entries_lt = SingleLinkedList(key=self.key)
            for _bucket in self.hash_table:
                if not _bucket.empty:
                    for _entry in _bucket:
                        _data = (_entry.key, _entry.value)
                        _entries_lt.append(_data)
            return _entries_lt
        except Exception as err:
            self._error_handler(err)

    def resize(self) -> None:
        """resize _summary_
        """
        try:
            # check if the structure is rehashable
            if self.rehashable:
                # gettting the current capacity to avoid null errors
                new_capacity = self.mcapacity
                # find the new capacity according to limits
                # augmenting the capacity
                if self._cur_alpha >= self.max_alpha:
                    new_capacity = next_prime(self.mcapacity * 2)
                # reducing the capacity
                elif self._cur_alpha < self.min_alpha:
                    new_capacity = next_prime(self.mcapacity // 2)

                # asigning the new capacity
                self.mcapacity = new_capacity

                # reseting the size, collisions and current load factor
                self._size = 0
                self._collisions = 0
                self._cur_alpha = 0

                # creating the new hash table
                new_table = ArrayList(cmp_function=self.cmp_function,
                                      key=self.key)
                # keep in memory the old hash table
                old_table = self.hash_table

                # Create the empty buckets in thenew hash table
                i = 0
                while i < self.mcapacity:
                    # bucket is a SingleLinkedList list
                    bucket = Bucket(cmp_function=self.cmp_function,
                                    key=self.key)
                    new_table.append(bucket)
                    i += 1

                # replace the old table with the new one
                self.hash_table = new_table

                # iterate over the old table
                for bucket in old_table:
                    if not bucket.empty:
                        for entry in bucket:
                            key = entry.key
                            value = entry.value
                            self.insert(key, value)
        except Exception as err:
            self._error_handler(err)

    def _error_handler(self, err: Exception) -> None:
        """*_error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

        Args:
            err (Exception): Python raised exception.
        """
        _context = self.__class__.__name__
        _function_name = inspect.currentframe().f_code.co_name
        error(_context, _function_name, err)

    def _check_type(self, entry: MapEntry) -> bool:
        """*_check_type()* función propia de la estructura que revisa si el tipo de dato del registro (pareja llave-valor) que se desea agregar al *SCHashTable* es del mismo tipo contenido dentro de los *MapEntry* del *SCHashTable*.

        Args:
            element (T): elemento que se desea procesar en *SCHashTable*.

        Raises:
            TypeError: error si el tipo de dato del elemento que se desea agregar no es el mismo que el tipo de dato de los elementos que ya contiene el *SCHashTable*.

        Returns:
            bool: operador que indica si el ADT *SCHashTable* es del mismo tipo que el elemento que se desea procesar.
        """
        # TODO check usability of this function
        # if datastruct is empty, set the entry type
        key = entry.key
        value = entry.value
        if self.empty:
            self._key_type = type(key)
            self._value_type = type(value)
        # check if the new entry is the same type as the other entries
        elif self._key_type is not type(key):
            err_msg = f"Invalid key type: {type(key)} "
            err_msg += f"for struct configured with type: {self._key_type}"
            raise TypeError(err_msg)
        elif self._value_type is not type(value):
            err_msg = f"Invalid value type: {type(value)} "
            err_msg += f"for struct configured with type: {self._value_type}"
            raise TypeError(err_msg)
        # otherwise, the type is valid
        return True

    def _validate_key_type(self, entry: MapEntry) -> bool:
        """_validate_key_type _summary_

        Args:
            entry (MapEntry): _description_

        Raises:
            TypeError: _description_

        Returns:
            bool: _description_
        """
        # TODO check usability of this function
        key = entry.key
        # if the new entry is the same type as the other entries
        if self._key_type is not type(key):
            err_msg = f"Invalid key type: {type(key)} "
            err_msg += f"for struct configured with type: {self._key_type}"
            raise TypeError(err_msg)
        # otherwise, the type is valid
        return True

    def _validate_value_type(self, entry: MapEntry) -> bool:
        """_validate_value_type _summary_

        Args:
            entry (MapEntry): _description_

        Raises:
            TypeError: _description_

        Returns:
            bool: _description_
        """
        # TODO check usability of this function
        value = entry.value
        # if the new entry is the same type as the other entries
        if self._value_type is not type(value):
            err_msg = f"Invalid value type: {type(value)} "
            err_msg += f"for struct configured with type: {self._value_type}"
            raise TypeError(err_msg)
        # otherwise, the type is valid
        return True

    def __len__(self) -> int:
        """*__len__()* function to return the number of entries (n) in the *SCHashTable*.

        Returns:
            int: Number of entries (n) in the *SCHashTable*.
        """
        return self._size

    def __str__(self) -> str:
        """*__str__()* function to return a string representation of the *SCHashTable*.

        Returns:
            str: string representation of the *SCHashTable*.
        """
        _attr_lt = []
        for attr, value in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format callable attributes
            if callable(value):
                try:
                    value = f"{value.__name__}{inspect.signature(value)}"
                except ValueError:
                    value = repr(value)  # Fallback for non-standard callables
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(value)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* function to return a string representation of the *SCHashTable*.

        Returns:
            str: string representation of the *SCHashTable*.
        """
        return self.__str__()
