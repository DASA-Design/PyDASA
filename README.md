# PyDASA

Library to solve software architecture and physical problems with dimensionless analysis and the Pi-Theorem

## Requirements

I need an object oriented design option to include the following requirements in its specifications

- to manage for fundamental dimensions (traditional and extendable to software architecture)
- to manage dimensional parameters and variables, recognizing parameters as ageneralizaction of input, aoutput and control variables, it has to have a name, a symbol, a range (min, max, step) and dimensions.
- to manage manage the data for meassurements and metrics in the real world and software architecture. it need to manage imperial and metrics units and be related with the dimensional parameters.
- to manage dimensionless coefficients or numbers (they ar synonym) with their name, symbol, formula, and relation to their dimensional parameters.
- to classify the dimensionless coefficients based on non repeatable and repetead dimensional parameters.
- to create algorithmically the dimensionless coefficients with a four-step method described as follows:
  * To create a complete  and mutually independent parameters (variables and constants) thought to be relevant for the process and that can influence the phenomena, this is called a relevance list.
  * To shape this relevance list into a matrixial form divided into two parts. The square core matrix; and the residual matrix. The former contains the fundamental dimensions in the rows (i.e.: L, M, and T, or A, D, and, T) and the most critical dimensional variables as columns (i.e.: ρ, L, and V) and the latter contains the rest of the independently significant variables as columns; in particular, the variable we want to predict as the first one.
  * To linearly transform the core matrix into a unity matrix (ones as diagonal values, and the remaining elements are zero).
  * To divide the variables of the residual matrix by the variables of the unity matrix with the exponents indicated by the unit values of the residual matrix to generate DC/DN.
- to check the principle of similitude for traditional problems and extendable into software architecture.
- to calculate the dimensionless coefficient range (min, max) and the influence of their dimensional parameters in their behaviour.
- to simulate the dimensionless formula with its coeffcients and have a detailed behavioural data.
- to plot or graph possible dimensionless charts using the behaviour of dimensionless coefficients and the dimensional parameters.

## Emoji

1. TODO 📋
2. WORKING 🔶👨‍💻
3. DONE ✅
4. WARNING ⚠️

## Src Path Structure

1. **pydasa**
   1. **analysis:** 

      1. ~~conversion.py unit conversion handler for the solver, OUT OF SCOPE for now!!!📋~~
      2. scenario.py:  contains the DimSensitivity class for understanding variance in the coefficients. ✅
      3. simulation.py: monte carlo simulator for one coefficient (MonteCarloSim class). ✅

      ---
   2. **buckingham**

      1. fundamental.py: contains Dimension class for Buckingham Pi-theorem specific operations. ✅
      2. vashchy.py: contains the Coefficient class to represent the dimensionless number resulting of the analysis. ✅

      ---
   3. **core:** shared and core capabilities

      1. basic_OLD.py: ~~deprecated version of validation class.~~ ⚠️
      2. basic.py: contains Foundation, SymBasis, IdxBasis classes, shared capabilities for entities with validation. ✅
      3. basis.py: contains base type definitions and protocols. ✅
      4. config.py: contains all global and shared variables for the analysis. ✅
      5. fundamental.py: contains Dimension class, the basis of dimensional analysis (replaces FDU). ✅
      6. measurement.py: contains the Unit class, fundamental for unit conversion when necessary, NOT FEASIBLE!!! 📋
      7. parameter.py: ~~deprecated, replaced by variable.py~~ ⚠️
      8. ranges.py: contains range management functionality for parameters. ✅
      9. standarization.py: contains standardization utilities for dimensional expressions. ✅
      10. variable.py: contains Variable class to execute the analysis (replaces parameter.py). ✅

      ---
   4. **data:** input/output operations

      1. io.py: contains all the input/output functions for saving/loading data of the analysis. ✅

      ---
   5. **datastruct:** data structures to manage the unit conversion process.

      1. **lists**

         1. arlt.py: arraylist. ✅
         2. sllt.py: single linked list. ✅
         3. dllt.py: double linked list. ✅
         4. ndlt.py: node list for double and single linked. ✅
      2. **tables**

         1. scht.py: separate chaining hashtable. ✅
         2. htme.py: entry used in the separate chaining hashtable. ✅

         ---
   6. **dimensional**

      1. basis.py: contains base dimensional analysis utilities and type definitions. ✅
      2. constants.py: contains predefined dimensional frameworks (PHYSICAL, COMPUTATION, SOFTWARE). ✅
      3. ~~domain.py unit conversion handler/manager for the matrix UnitsManager, OUT OF SCOPE for now!!!📋~~
      4. framework.py: contains the DimSchema class to manage and control FDUs in the solving process. ✅
      5. model.py: contains the DimMatrix class to solve the dimensional matrix. ✅

      ---
   7. **enum:** enumeration types for domain-specific categorization

      1. basis.py: base enumeration definitions. ✅
      2. conversion.py: unit conversion enumeration types. 📋
      3. domain.py: domain-specific enumeration types. 📋
      4. metrics.py: metrics and measurement enumeration types. 📋

      ---
   8. **environments:** alternative implementations (may be duplicates)

      1. influence.py: contains the SensitivityHandler class for understanding variance. ✅
      2. phenomena.py: alternative Solver implementation. 🔶👨‍💻
      3. practical.py: contains the MonteCarloHandler class to control Monte Carlo simulations. ✅

      ---
   9. **handlers**

      1. influence.py: contains the SensitivityHandler class for understanding variance in the coefficients. ✅
      2. phenomena.py: has the main Solver() class of the project. 🔶👨‍💻⚠️ WORKING HERE ⚠️
      3. practical.py: contains the MonteCarloHandler class to control all the Monte Carlo simulations. ✅

      ---
   10. **types:** type definitions and generic utilities

       1. generics.py: contains generic type definitions and type aliases. ✅

       ---
   11. **utils**

       1. config_OLD.py: ~~deprecated configuration file.~~ ⚠️
       2. config.py: ~~moved to core/config.py~~ ⚠️
       3. default.py: contains all the default stuff needed for custom datastructures + other functionalities. ✅
       4. hashing.py: contains hashing utilities for data structures. ✅
       5. latex.py: contains all the LaTeX parsing functions for better representation of formulas. ✅
       6. math.py: contains mathematical utilities for dimensional analysis. ✅
       7. memory.py: contains memory management utilities. ✅
       8. patterns.py: contains regex patterns for validation and parsing. ✅
       9. ~~error.py: moved to validations/error.py~~ ⚠️
       10. ~~helpers.py: contains any other function useful for the process, include MAD for hashtable, check if is prime, and other stuff. ✅~~
       11. ~~queues.py: library that implement the queue theory for simulations and stuff ✅ ->  ⚠️ REMOVED FROM REPO~~
       12. ~~io.py: moved to data/io.py~~ ⚠️

       ---
   12. **validations:** validation decorators and error handling

       1. decorators.py: contains all validation decorators (@validate_type, @validate_pattern, etc.). ✅
       2. error.py: contains the generic error_handler() function and inspect_var() for all components. ✅
       3. protocols.py: contains validation protocols and interfaces. ✅
       4. validators.py: contains validator functions and classes. ✅

       ---
   13. ~~**math** ⚠️⚠️⚠️ TODO ⚠️⚠️⚠️ do i need them????📋 outside of lib scope!!!~~

       1. ~~numbers.py📋~~
       2. ~~queues.py📋~~

       ---
   14. ~~**visualization:** dont NEED it, USE MATPLOTLIB OR OTHER STUFF!!!!, but need to create plots and charts from vars + coefficients 📋~~

## Tests Path Structure

1. **pydasa**

🔶👨‍💻⚠️ WORKING HERE ⚠️

1. **analysis**

   1. ~~test_conversion.py: tests for unit conversion handler for the solver. NOT YET! 📋~~
   2. test_scenario.py: tests for sensitivity analysis of the Coefficients (DimSensitivity class). 📋
   3. test_simulation.py: tests for the Monte Carlo simulator for one coefficient (MonteCarloSim class). ✅

   ---
2. **buckingham**

   1. test_vashchy.py: tests for the Coefficient class. ✅

   ---
3. **core:** shared and core capabilities

   1. test_basic.py: tests for the Foundation, SymBasis, IdxBasis classes. ✅
   2. test_config.py: tests for all global and shared variables for the analysis. ✅
   3. test_fundamental.py: tests for the Dimension class. ✅
   4. test_measurement.py: tests for the Unit class. NOT YET! 📋
   5. test_parameter.py: tests for the Variable class (replaces old parameter tests). ✅

   ---
4. **data:** input/output operations tests

   1. test_data.py: contains test data for all PyDASA tests (fixtures and samples). ✅
   2. test_io.py: tests for all the input/output functions for saving/loading data. 📋

   ---
5. **datastruct:** data structures tests

   1. **lists**

      1. test_arlt.py: tests for the arraylist. 📋
      2. test_sllt.py: tests for the single linked list. 📋
      3. test_dllt.py: tests for the double linked list. 📋
      4. test_ndlt.py: tests for the node list for double and single linked. 📋
   2. **tables**

      1. test_scht.py: tests for the separate chaining hashtable. 📋
      2. test_htme.py: tests for the entry useful for the separate chaining hashtable. 📋

      ---
6. **dimensional**

   1. test_constants.py: tests for predefined dimensional frameworks. ✅
   2. ~~test_domain.py: tests for the unit conversion handler/manager. NOT YET! 📋~~
   3. test_framework.py: tests for the DimSchema class to manage and control FDUs. ✅
   4. test_model.py: tests for the DimMatrix class to solve the dimensional matrix. ✅

   ---
7. **handlers**

   1. test_influence.py: tests for the SensitivityHandler class for understanding variance in the coefficients. 📋
   2. test_phenomena.py: tests for the main Solver() class of the project. 🔶👨‍💻⚠️ TODO ⚠️
   3. test_practical.py: tests for the MonteCarloHandler class to control all the Monte Carlo simulations. ✅

   ---
8. **types:** type definitions tests

   1. test_generics.py: tests for generic type definitions and type aliases. ✅

   ---
9. **utils**

   1. test_default.py: tests for all the default stuff needed for custom datastructures + other functionalities. 📋
   2. ~~test_helpers.py: tests for any other function useful for the process, include MAD for hashtable, check if is prime, and other stuff. NOT YET! 📋~~
   3. test_latex.py: tests for all the LaTeX parsing functions for better representation of formulas and stuff. ✅
   4. test_patterns.py: tests for regex patterns for validation and parsing. ✅

   ---
10. **validations:** validation decorators and error handling tests

    1. test_decorators.py: tests for all validation decorators (@validate_type, @validate_pattern, etc.). ✅
    2. test_error.py: tests for the generic error_handler() function and inspect_var() for all components. ✅

    ---
