# -*- coding: utf-8 -*-
"""
Module to represent the **SeparateChainingTable** data structure for the **Hash Table** in *PyDASA*.

*IMPORTANT:* This code and its specifications for Python are based on the implementations proposed by the following authors/books:

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
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinked
# util functions for the hash table
from Src.PyDASA.Utils.nos import next_prime, previous_prime
from Src.PyDASA.Utils.nos import mad_hash
from Src.PyDASA.Utils.err import handle_error as error
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
assert SingleLinked
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
Default load factor (*alpha*) for the *SeparateChainingTable*, by default is 4.0.
"""

# :data: MAX_SC_ALPHA
MAX_SC_ALPHA: float = 8.0
"""
Maximum load factor (*alpha*) for the *SeparateChainingTable*, by default is 8.0.
"""

# :data: MIN_SC_ALPHA
MIN_SC_ALPHA: float = 2.0
"""
Minimum load factor (*alpha*) for the *SeparateChainingTable*, by default is 2.0.
"""


@dataclass
class Bucket(SingleLinked, Generic[T]):
    """**Bucket** class to represent a bucket in the **Hash Table** with the *Separate Chaining* method. The structure is based (inherits) on a custom singly linked list (*SingleLinked*) for *PyDASA*.

    Args:
        SingleLinked (T): *PyDASA* class for a single linked list.
        Generic (T): Generic type for a Python data structure.
    """
    # keep as is...
    pass


@dataclass
class SeparateChainingTable(Generic[T]):

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
    Customizable comparison function for *SeparateChainingTable* and its *MapEntry* objects. Defaults to *dflt_cmp_func_ht()* from *PyDASA*, but can be overridden by the user.
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
    Customizable key name for identifying elements in the *SeparateChainingTable*. Defaults to *DFLT_DICT_KEY = '_id'* from *PyDASA*, but can be overridden by the user.
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
    Optional Python list for loading external data intho the *SeparateChainingTable*. Defaults to *None* but can be provided during creation.
    """

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the *SeparateChainingTable* after creation by setting attributes like *rehashable*, *mcapacity*, *alpha*, *cmp_function*, *key*, *prime*, *scale*, *shift*, and *iodata*.
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
                # bucket is a SingleLinked list
                bucket = Bucket(cmp_function=self.cmp_function,
                                key=self.key)
                # add the bucket to the hash table
                self.hash_table.add_last(bucket)
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
            self._handle_error(err)

    def default_compare(self, key1: T, entry2: MapEntry) -> int:
        """*default_compare()* es la función de comparación por defecto para comparar la llave de un elemento vs. el registro (pareja llave-valor) o *MapEntry* que se desea agregar al *SeparateChainingTable*, es una función crucial para que la estructura funcione correctamente.

        Args:
            key1 (Any): llave (*key*) del primer registro a comparar.
            entry2 (MapEntry): segundo registro (pareja llave-valor) a comparar.

        Returns:
            int: respuesta de la comparación entre los elementos, 0 si las llaves (*key*) son iguales, 1 si key1 es mayor que la llave (*key*) de entry2, -1 si key1 es menor.
        """
        # FIXME: aki voyyyyy!!!!!!!!!!!!!!!!
        try:
            # using the default compare function for the key
            return dflt_cmp_func_ht(self.key, key1, entry2)
        except Exception as err:
            self._handle_error(err)

    def _handle_error(self, err: Exception) -> None:
        """*_handle_error()* función propia de la estructura que maneja los errores que se pueden presentar en el *SeparateChainingTable*.

        Si se presenta un error en *SeparateChainingTable*, se formatea el error según el contexto (paquete/módulo/clase), la función (método) que lo generó y lo reenvia al componente superior en la jerarquía *DISCLib* para manejarlo segun se considere conveniente el usuario.

        Args:
            err (Exception): Excepción que se generó en el *SeparateChainingTable*.
        """
        # TODO check usability of this function
        cur_context = self.__class__.__name__
        cur_function = inspect.currentframe().f_code.co_name
        error(cur_context, cur_function, err)

    def _check_type(self, entry: MapEntry) -> bool:
        """*_check_type()* función propia de la estructura que revisa si el tipo de dato del registro (pareja llave-valor) que se desea agregar al *SeparateChainingTable* es del mismo tipo contenido dentro de los *MapEntry* del *SeparateChainingTable*.

        Args:
            element (T): elemento que se desea procesar en *SeparateChainingTable*.

        Raises:
            TypeError: error si el tipo de dato del elemento que se desea agregar no es el mismo que el tipo de dato de los elementos que ya contiene el *SeparateChainingTable*.

        Returns:
            bool: operador que indica si el ADT *SeparateChainingTable* es del mismo tipo que el elemento que se desea procesar.
        """
        # TODO check usability of this function
        # if datastruct is empty, set the entry type
        key = entry.get_key()
        value = entry.get_value()
        if self.is_empty():
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

    def _check_key_type(self, entry: MapEntry) -> bool:
        """*_check_key_type()* función propia de la estructura que revisa si el tipo de dato de la llave del registro (pareja llave-valor) que se desea agregar al *SeparateChainingTable* es del mismo tipo contenido dentro de los *MapEntry* del *SeparateChainingTable*.

        Args:
            element (T): elemento que se desea procesar en *SeparateChainingTable*.

        Raises:
            TypeError: error si el tipo de dato de la llave que se desea agregar no es el mismo que el tipo de dato de las llaves que ya contiene el *SeparateChainingTable*.

        Returns:
            bool: operador que indica si la llave del ADT *SeparateChainingTable* es del mismo tipo que la llave que se desea procesar.
        """
        # TODO check usability of this function
        key = entry.get_key()
        # if the new entry is the same type as the other entries
        if self._key_type is not type(key):
            err_msg = f"Invalid key type: {type(key)} "
            err_msg += f"for struct configured with type: {self._key_type}"
            raise TypeError(err_msg)
        # otherwise, the type is valid
        return True

    # @property
    def is_empty(self) -> bool:
        """*is_empty()* revisa si el *SeparateChainingTable* está vacío.

        Returns:
            bool: operador que indica si la estructura *SeparateChainingTable* está vacía.
        """
        # TODO change the method name to "empty" or @property "empty"?
        try:
            return self._size == 0
        except Exception as err:
            self._handle_error(err)

    # @property
    def size(self) -> int:
        """*size()* devuelve el numero de entradas *MapEntry* que actualmente contiene el *SeparateChainingTable*.

        Returns:
            int: tamaño de la estructura *SeparateChainingTable*.
        """
        # TODO change the method to @property "size"?
        try:
            return self._size
        except Exception as err:
            self._handle_error(err)

    def contains(self, key: T) -> bool:
        """*contains()* responde si el *SeparateChainingTable* contiene un registro *MapEntry* con la llave *key*.

        Args:
            key (T): llave del registro (pareja llave-valor) que se desea buscar en el *SeparateChainingTable*.

        Raises:
            IndexError: error si la estructura está vacía.

        Returns:
            bool: operador que indica si el *SeparateChainingTable* contiene o no un registro con la llave *key*.
        """
        try:
            if self.is_empty():
                raise IndexError("Empty data structure")
            else:
                # assume the entry is not in the structure
                found = False
                # use the MAD compression function to get the hash key
                hkey = mad_hash(key,
                                     self._scale,
                                     self._shift,
                                     self.prime,
                                     self.mcapacity)
                # look into the bucket
                bucket = self.hash_table.get_element(hkey)
                idx = bucket.find(key)
                # if the entry is in the bucket, return True
                if idx > -1:
                    found = True
                return found
        except Exception as err:
            self._handle_error(err)

    def put(self, key: T, value: T) -> None:
        """*put()* agrega un nuevo registro *MapEntry* al *SeparateChainingTable*, si la llave *key* ya existe en el *SeparateChainingTable* se reemplaza su valor *value*.

        Args:
            key (T): llave asociada la nuevo *MapEntry*.
            value (T): el valor asociado al nuevo *MapEntry*.

        Raises:
            Exception: si la operación no se puede realizar, se invoca la función *_handle_error()* para manejar el error.
        """
        try:
            # create a new entry for the hash table
            new_entry = MapEntry(key, value)
            # cheking the type of the entry
            if self._check_type(new_entry):
                # get the hash key for the entry
                hkey = mad_hash(key,
                                     self._scale,
                                     self._shift,
                                     self.prime,
                                     self.mcapacity)

                # checking the bucket
                bucket = self.hash_table.get_element(hkey)
                idx = bucket.find(key)
                # the entry is not in the bucket, add it and a collision
                # the entry is already in the bucket, update it
                if idx > -1:
                    bucket.change_info(new_entry, idx)
                # otherwise, is a new entry
                else:
                    if not bucket.is_empty():
                        self._collisions += 1
                    bucket.add_last(new_entry)
                    self._size += 1
                    self._cur_alpha = self._size / self.mcapacity
                # check if the structure needs to be rehashed
                if self._cur_alpha >= self.max_alpha:
                    self.rehash()
        except Exception as err:
            self._handle_error(err)

    def get(self, key: T) -> Optional[MapEntry]:
        """*get()* recupera el registro *MapEntry* cuya llave *key* sea igual a la que se encuentra dentro del *SeparateChainingTable*, si no existe un registro con la llave, devuelve *None*.

        Args:
            key (T): llave asociada al *MapEntry* que se desea buscar.

        Raises:
            IndexError: error si la estructura está vacía.

        Returns:
            Optional[MapEntry]: *MapEntry* asociado a la llave *key* que se desea. *None* si no se encuentra.
        """
        try:
            if self.is_empty():
                raise IndexError("Empty data structure")
            else:
                # assume the entry is not in the structure
                entry = None
                # get the hash key for the entry
                hkey = mad_hash(key,
                                     self._scale,
                                     self._shift,
                                     self.prime,
                                     self.mcapacity)

                # checking the bucket
                bucket = self.hash_table.get_element(hkey)
                idx = bucket.find(key)
                # if the entry is in the bucket, return it
                if idx > -1:
                    entry = bucket.get_element(idx)
                return entry
        except Exception as err:
            self._handle_error(err)

    def check_bucket(self, key: T) -> Optional[Bucket]:
        """*check_bucket()* revisa el *Bucket* asociado a la llave *key* dentro del *SeparateChainingTable*. Recupera todo el *Bucket* asociado a la llave y si no existe, devuelve *None*.

        Args:
            key (T): llave asociada al *Bucket* que se desea revisar

        Raises:
            IndexError: error si la estructura está vacía.

        Returns:
            Optional[Bucket]: *Bucket* asociado a la llave *key* que se desea. *None* si no se encuentra.
        """
        try:
            if self.is_empty():
                raise IndexError("Empty data structure")
            else:
                # assume the entry is not in the structure
                bucket = None
                # get the hash key for the entry
                hkey = mad_hash(key,
                                     self._scale,
                                     self._shift,
                                     self.prime,
                                     self.mcapacity)

                # checking the bucket
                bucket = self.hash_table.get_element(hkey)
                return bucket
        except Exception as err:
            self._handle_error(err)

    def remove(self, key: T) -> Optional[MapEntry]:
        """*remove()* elimina el registro *MapEntry* cuya llave *key* sea igual a la que se encuentra dentro del *SeparateChainingTable*, si no existe un registro con la llave, genera un error.

        Args:
            key (T): llave asociada al *MapEntry* que se desea eliminar.

        Raises:
            IndexError: error si la estructura está vacía.
            IndexError: error si el registro que se desea eliminar no existe dentro del *SeparateChainingTable*.

        Returns:
            Optional[MapEntry]: registro *MapEntry* que se eliminó del *SeparateChainingTable*. *None* si no existe el registro asociada a la llave *key*.
        """
        try:
            if self.is_empty():
                raise IndexError("Empty data structure")
            else:
                entry = None
                # get the hash key for the entry
                hkey = mad_hash(key,
                                     self._scale,
                                     self._shift,
                                     self.prime,
                                     self.mcapacity)

                # checking the bucket
                bucket = self.hash_table.get_element(hkey)
                if not bucket.is_empty():
                    idx = bucket.find(key)
                    if idx >= 0:
                        entry = bucket.remove_element(idx)
                        self._size -= 1
                        self._cur_alpha = self._size / self.mcapacity
                    # TODO maybe i don't need this
                    else:
                        raise IndexError(f"Entry for Key: {key} not found")
            if self._cur_alpha < self.min_alpha:
                self.rehash()
            return entry
        except Exception as err:
            self._handle_error(err)

    def keys(self) -> SingleLinked[T]:
        """*keys()* devuelve una lista (*SingleLinked*) con todas las llaves (*key*) de los registros (*MapEntry*) del *SeparateChainingTable*.

        Returns:
            SingleLinked[T]: lista (*SingleLinked*) con todas las llaves (*key*) del *SeparateChainingTable*.
        """
        try:
            keys_lt = SingleLinked(key=self.key)
            # TODO improve with SingleLinked concat() method?
            for bucket in self.hash_table:
                if not bucket.is_empty():
                    for entry in bucket:
                        keys_lt.add_last(entry.get_key())
            return keys_lt
        except Exception as err:
            self._handle_error(err)

    def values(self) -> SingleLinked[T]:
        """*values()* devuelve una lista (*SingleLinked*) con todos los valores de los registros (*MapEntry*) del *SeparateChainingTable*.

        Returns:
            SingleLinked[T]: lista (*SingleLinked*) con todos los valores (*value*) del *SeparateChainingTable*.
        """
        try:
            values_lt = SingleLinked(key=self.key)
            # TODO improve with SingleLinked concat() method?
            for bucket in self.hash_table:
                if not bucket.is_empty():
                    for entry in bucket:
                        values_lt.add_last(entry.get_value())
            return values_lt
        except Exception as err:
            self._handle_error(err)

    def entries(self) -> SingleLinked[T]:
        """*entries()* devuelve una lista (*SingleLinked*) con tuplas de todas los registros (*MapEntry*) del *SeparateChainingTable*. Cada tupla contiene en la primera posición la llave (*key*) y en la segunda posición el valor (*value*) del registro.

        Returns:
            SingleLinked[T]: lista (*SingleLinked*) de tuplas con todas los registros del *SeparateChainingTable*.
        """
        try:
            entries_lt = SingleLinked(key=self.key)
            # TODO improve with SingleLinked concat() method?
            for bucket in self.hash_table:
                if not bucket.is_empty():
                    for entry in bucket:
                        data = (entry.get_key(), entry.get_value())
                        entries_lt.add_last(data)
            return entries_lt
        except Exception as err:
            self._handle_error(err)

    def rehash(self) -> None:
        """*rehash()* reconstruye la tabla de hash con una nueva capacidad (*M*) y un nuevo factor de carga (*alpha*) según los límites configurados por los parametros *max_alpha* y *min_alpha*.

        Si el factor de carga (*alpha*) es mayor que el límite superior (*max_alpha*), se duplica la capacidad (*M*) buscando el siguiente número primo (*P*) reconstruyendo la tabla.

        Si el factor de carga (*alpha) es menor que el límite inferior (*min_alpha*), se reduce a la mitad la capacidad (*M*) de la tabla buscando el siguiente número primo (*P*) reconstruyendo la tabla.
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
                    # bucket is a SingleLinked list
                    bucket = Bucket(cmp_function=self.cmp_function,
                                    key=self.key)
                    new_table.add_last(bucket)
                    i += 1

                # replace the old table with the new one
                self.hash_table = new_table

                # iterate over the old table
                for bucket in old_table:
                    if not bucket.is_empty():
                        for entry in bucket:
                            key = entry.get_key()
                            value = entry.get_value()
                            self.put(key, value)
        except Exception as err:
            self._handle_error(err)

    def __len__(self) -> int:
        """*__len__()* función nativa de Python personalizada para el *SeparateChainingTable*. Permite utilizar la función *len()* de Python para recuperar el tamaño del *SeparateChainingTable*.

        Returns:
            int: tamaño del *SeparateChainingTable*.
        """
        return self._size
