Installation
============

PyPI Installation
-----------------

Install PyDASA from PyPI using pip:

.. code-block:: bash

   pip install pydasa

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
* matplotlib >= 3.8.0
* pandas >= 2.1.0
* SALib >= 1.4.5
* antlr4-python3-runtime == 4.11