Installation
============

PyPI Installation
-----------------

To install **PyDASA**, use pip command:

.. code-block:: bash

    pip install pydasa

PyPI will automatically handle the installation of all required dependencies. Then, to check the installed version of **PyDASA**, run:

.. code-block:: python

    import pydasa
    print(pydasa.__version__)

PyPI Update
--------------

To update **PyDASA** to the latest version, use pip with the --upgrade flag:

.. code-block:: bash

   pip install --upgrade pydasa


Development Installation
------------------------

To install the development version from GitHub:

.. code-block:: bash

   git clone https://github.com/DASA-Design/PyDASA.git
   cd PyDASA
   pip install -e ".[dev]"

Requirements
------------

PyDASA requires Python 3.10 or higher and the following dependencies:

* numpy >= 1.26.4
* scipy >= 1.13.0
* sympy >= 1.12
* SALib >= 1.4.5
.. * antlr4-python3-runtime == 4.11
.. * matplotlib >= 3.8.0
.. * pandas >= 2.1.0
