.. PyDASA documentation master file, created by
    sphinx-quickstart on Fri Dec  5 19:35:35 2025.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

PyDASA
======

.. |pypi| image:: https://img.shields.io/pypi/v/pydasa?cache=none
   :target: https://pypi.org/project/pydasa/
   :alt: PyPI

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/pydasa
   :target: https://pypi.org/project/pydasa/
   :alt: Python Version

.. |license| image:: https://img.shields.io/github/license/DASA-Design/PyDASA
   :target: https://github.com/DASA-Design/PyDASA/blob/main/LICENSE
   :alt: License

.. |docs| image:: https://readthedocs.org/projects/pydasa/badge/?version=latest
   :target: https://pydasa.readthedocs.io
   :alt: Documentation Status

.. |coverage| image:: https://codecov.io/github/DASA-Design/PyDASA/graph/badge.svg?token=UZFT3CURK1
   :target: https://codecov.io/gh/DASA-Design/PyDASA
   :alt: Coverage

.. centered:: |pypi| |pyversion| |license| |docs| |coverage|

**PyDASA** (Dimensional Analysis for Scientific Applications and Software Architecture) is an open-source Python library for dimensional analysis of complex phenomena across physical, chemical, computational, and software domains using the Buckingham Pi-theorem.

The Primary Need
---------------------------

.. card::

    **Epic User Story**
    ^^^
    **As a** researcher, engineer, or software architect analyzing complex systems,
    
    **I want** a comprehensive dimensional analysis library implementing the Buckingham Pi theorem,
    
    **So that** I can systematically discover dimensionless relationships, validate models, and understand system behavior across physical, computational, and software architecture domains.

Quick Navigation
----------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Getting Started
        :link: public/context/installation
        :link-type: doc

        New to **PyDASA**? Check out the getting started guide for installation
        and quick start examples.

    .. grid-item-card:: 📖 User Guide
        :link: public/features/index
        :link-type: doc

        The user guide provides in-depth information on dimensional analysis
        concepts and **PyDASA** features.

    .. grid-item-card:: 💡 Examples
        :link: public/examples/index
        :link-type: doc

        Practical examples and tutorials demonstrating **PyDASA** capabilities
        in real-world scenarios.

    .. grid-item-card:: 📚 API Reference
        :link: autoapi/index
        :link-type: doc

        Complete API documentation with detailed descriptions of all
        modules, classes, and functions.

Acknowledgements
----------------

The theoretical foundation of dimensional analysis in **PyDASA** draws upon the classical work:

.. card::

    **Dimensionsanalyse: Theorie der physikalischen Dimensionen mit Anwendungen**
    ^^^
    :Author: H. Görtler
    :Series: Ingenieurwissenschaftliche Bibliothek (Engineering Science Library)
    :Publisher: Springer-Verlag
    :Year: 1975
    :ISBN: 978-3642808739
    :Language: German

This comprehensive treatise provides the rigorous mathematical foundation for the theory of physical dimensions and dimensional homogeneity that underlies modern dimensional analysis methods.

Also, **PyDASA** was inspired by the work of Mokbel Karam and Tony Saad in **BuckinghamPy** presented in:

.. card::

    **BuckinghamPy: A Python software for dimensional analysis**
    ^^^
    :Authors: Mokbel Karam, Tony Saad
    :Journal: SoftwareX
    :Volume: 16
    :Year: 2021
    :Article: 100851
    :DOI: https://doi.org/10.1016/j.softx.2021.100851
    :License: Creative Commons

We are grateful to the authors for their contribution into making dimensional analysis more accessible through computational tools, which motivated our development of **PyDASA** with expanded and costumizable capabilities for scientific and engineering applications.

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
    :caption: Examples
    :hidden:

    public/examples/index

.. toctree::
    :maxdepth: 2
    :caption: Architecture & Design
    :hidden:
    
    public/design/index

.. toctree::
    :maxdepth: 2
    :caption: Development Status
    :hidden:

    public/development/index

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:
    
    autoapi/index

.. toctree::
    :maxdepth: 2
    :caption: Project History
    :hidden:

    public/project/changelog
