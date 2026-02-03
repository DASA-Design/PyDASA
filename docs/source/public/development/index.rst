Development Guide
=================

We welcome contributions! This guide provides essential information to help contributors and developers get started with **PyDASA**.

* :doc:`roadmap` - Completed and planned features and improvements.
* :doc:`contributing` - How to contribute to **PyDASA**
* :doc:`tests` - Testing guidelines and procedures

Development Workflow
--------------------

**1. Setup**

Quick setup for development:

.. code-block:: bash

    # Clone and fork the repository
    git clone https://github.com/DASA-Design/PyDASA.git
    cd PyDASA

    # Install in development mode
    pip install -e ".[dev]"

    # Run tests to verify setup
    pytest tests/

**2. Development Process**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for new features
5. Run the full test suite
6. Submit a pull request

See :doc:`contributing` for commit message format and detailed guidelines.

**3. Testing Requirements**

All contributions must include tests. See :doc:`tests` for:

* Writing unit tests
* Running the test suite
* Coverage requirements
* Testing best practices

.. toctree::
    :maxdepth: 1
    :hidden:

    roadmap
    contributing
    tests