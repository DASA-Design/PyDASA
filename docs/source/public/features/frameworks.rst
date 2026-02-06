Dimensional Framework
======================

Overview
--------

Dimensional Analysis relies on **Fundamental Dimensional Units** (*FDUs*) as the building blocks for describing physical or any domain-specific (e.g.: software architecture)  quantities. The **Framework** (or **Schema**) class defines which *FDUs* are used, their precedence in the dimensional matrix, and the validation patterns for dimensional expressions in instances of the ``Variable`` class. **PyDASA** implements this through two core classes: ``Dimension`` for individual *FDUs* and ``Schema`` for managing the domain-specific FDU set.

Fundamental Dimensional Units (FDUs)
-------------------------------------

*FDUs* are the fundamental measurable quantities from which all metrics derive. Physical dimensional analysis uses seven *FDUs*: **Length** [L], **Mass** [M], **Time** [T], **Electric Current** [A], **Temperature** [K], **Amount of Substance** [mol], and **Luminous Intensity** [cd], though most applications primarily use *L*, *M*, and *T*. These principles can theoretically extend to other dimensional domains (e.g., computational, software).

The ``Dimension`` class in ``fundamental.py`` represents a single *FDU* with:

    - **Symbol** (``_sym``): Represents LaTeX or alphanumeric notation (e.g., "M", "L", "T").
    - **Index** (``_idx``): Determines precedence in the dimensional matrix (defines row order).
    - **Unit** (``_unit``): Specifies basic measurement unit (e.g., "m", "s", "kg").
    - **Framework** (``_fwk``): Identifies context (``PHYSICAL``, ``COMPUTATION``, ``SOFTWARE``, ``CUSTOM``).
    - **Alias** (``_alias``): Provides Python-compatible name for code execution.

Schema Responsibilities
-----------------------

The ``Schema`` class in ``vaschy.py`` manages the dimensional framework by:

    1. **Maintaining FDU Collections** by storing *FDUs* in precedence order (``_fdu_lt``) and providing symbol-to-object mapping (``_fdu_map``).
    2. **Establishing Matrix Structure** by defining row order in the dimensional matrix through FDU precedence.
    3. **Validating Expressions** by generating regex patterns for matching and validating dimensional expressions.
    4. **Enforcing Consistency** by ensuring all *FDUs* within a framework follow Görtler's principles of Measurability, Consistency, and Clarity.

Supported Frameworks
---------------------

PyDASA provides three predefined frameworks:

    1. **PHYSICAL Framework:** Provides standard physical quantities using *L*, *M*, *T*, *A*, *K*, *mol*, *cd* as default for engineering and scientific applications.
    2. **COMPUTATION Framework:** Defines computational quantities for performance analysis, including Time (*T*), Data (*D*), and Operations (*O*).
    3. **SOFTWARE Framework:** Supports software-specific dimensions for architectural analysis (experimental).
    4. **CUSTOM Framework:** Allows user-defined *FDUs* for domain-specific applications requiring explicit *FDU* definitions.

Creating Custom Frameworks
---------------------------

To create a custom framework, define *FDUs* and pass them to the Schema:

.. code-block:: python

    from pydasa.dimensional.fundamental import Dimension
    from pydasa.dimensional.vaschy import Schema

    # Define custom FDUs
    custom_fdus = [
        {"_idx": 0, "_sym": "T", "_unit": "s", "_name": "Time"},
        {"_idx": 1, "_sym": "D", "_unit": "bit", "_name": "Data"},
        {"_idx": 2, "_sym": "O", "_unit": "op", "_name": "Operations"}
    ]

    # Create schema with custom FDUs
    schema = Schema(_fwk="CUSTOM", _fdu_lt=custom_fdus)

    # OR use Dimension class to create the FDUs
    custom_fdus = [
        Dimension(_idx=0, _sym="T", _unit="s", _name="Time"),
        Dimension(_idx=1, _sym="D", _unit="bit", _name="Data"),
        Dimension(_idx=2, _sym="O", _unit="op", _name="Operations")
    ]

    schema = Schema(_fwk="CUSTOM", _fdu_lt=custom_fdus)

The ``Schema`` automatically constructs the dimensional matrix (row order determined by *FDU* item index), validates expressions, and ensures consistency across all dimensional operations. Furthermore, Custom frameworks must maintain dimensional homogeneity and adhere to the defined *FDU* precedence.