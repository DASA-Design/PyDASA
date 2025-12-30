Development Guide
=================

Information for contributors and developers.

.. toctree::
    :maxdepth: 2

    contributing
    testing

Contributing
------------

We welcome contributions! This guide will help you get started.

* :doc:`contributing` - How to contribute to PyDASA
* :doc:`testing` - Testing guidelines and procedures

Development Setup
-----------------

Quick setup for development:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/DASA-Design/PyDASA.git
    cd PyDASA

    # Install in development mode
    pip install -e ".[dev]"

    # Run tests
    pytest tests/

Workflow
--------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

See :doc:`contributing` for detailed guidelines.

Testing
-------

All contributions must include tests. See :doc:`testing` for:

* Writing unit tests
* Running the test suite
* Coverage requirements
* Testing best practices