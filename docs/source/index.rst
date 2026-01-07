.. PyDASA documentation master file, created by
    sphinx-quickstart on Fri Dec  5 19:35:35 2025.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

PyDASA
======

**Python Dimensional Analysis for Scientific Applications**

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Getting Started
        :link: public/context/installation
        :link-type: doc

        New to PyDASA? Check out the getting started guide for installation
        and quick start examples.

    .. grid-item-card:: 📖 User Guide
        :link: public/features/index
        :link-type: doc

        The user guide provides in-depth information on dimensional analysis
        concepts and PyDASA features.

    .. grid-item-card:: 📚 API Reference
        :link: autoapi/index
        :link-type: doc

        Complete API documentation with detailed descriptions of all
        modules, classes, and functions.

    .. grid-item-card:: 💡 Examples
        :link: public/examples/index
        :link-type: doc

        Practical examples and tutorials demonstrating PyDASA capabilities
        in real-world scenarios.

Installation
------------

Install PyDASA using pip:

.. code-block:: bash

    pip install pydasa

Quick Example
-------------

.. code-block:: python

    from pydasa.handler.phenomena import Solver
    from pydasa.core.parameter import Variable
    from pydasa.core.fundamental import Dimension

    # Define dimensions
    L = Dimension(length=1)
    T = Dimension(time=1)

    # Create variables
    velocity = Variable("v", L/T)
    length = Variable("L", L)
    
    # Solve for dimensionless numbers
    solver = Solver()
    result = solver.solve([velocity, length])

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    public/context/index

.. toctree::
    :maxdepth: 2
    :caption: User Guide
    :hidden:

    public/features/index

.. toctree::
    :maxdepth: 2
    :caption: Design & Architecture
    :hidden:
    
    public/design/index

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:
    
    autoapi/index

.. toctree::
    :maxdepth: 2
    :caption: Examples
    :hidden:

    public/examples/index

.. toctree::
    :maxdepth: 2
    :caption: Development
    :hidden:

    public/development/index

.. toctree::
    :maxdepth: 2
    :caption: Project
    :hidden:

    public/project/changelog
