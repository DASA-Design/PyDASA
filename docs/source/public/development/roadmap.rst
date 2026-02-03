Development Roadmap
===================

**Emoji Convention:**
    - 📋 TODO
    - 🔶👨‍💻 WORKING
    - ✅ DONE
    - ⚠️ ATTENTION REQUIRED

**Current Version:** 0.6.4

✅ Core Modules (Implemented & Tested)
---------------------------------------

1. **core/**: Foundation classes, configuration, I/O.
2. **dimensional/**: Buckingham Pi theorem, dimensional matrix solver.
3. **elements/**: Variable and parameter management with specs.
4. **workflows/**: AnalysisEngine, MonteCarloSimulation, SensitivityAnalysis.
5. **validations/**: Decorator-based validation system.
6. **serialization/**: LaTeX and formula parsing.

👨‍💻 Currently Working
----------------------

1. **Documentation**: Improving API reference, tutorials, and user guides.
2. **Code Reduction**: Refactoring to eliminate redundancy, improve maintainability, readability, and performance.
3. **Data Structures**: Designing implementation for unit of measure and dimensional management systems to enable consistent unit conversion across frameworks.
4. **Testing**: Improving and expanding test coverage, especially for context/ and structs/ modules.

📋 Pending Development
-----------------------

1. **context/**: Implement Unit conversion system (stub implementation).
2. **structs/**: Implement Data structures (partial test coverage).
3. **Documentation**: Complete API reference completion and additional tutorials.

For Developers
--------------

If you're interested in contributing to **PyDASA** or understanding its internal structure, start with the :doc:`../design/architecture` document. See also the :doc:`contributing` section for our contribution guidelines.
