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

1. DONT DO ❌
2. WORKING 🔶👨‍💻
3. DONE ✅
4. WARNING ⚠️

## Src Path Structure

1. **pydasa**
   1. **analysis**

      1. conversion.py unit conversion handler for the solver, OUT OF SCOPE for now!!!❌
      2. scenario.py:  contains the Sensitivity class for understanding variance in the coefficients. ✅
      3. simulation.py: monte carlo slmulator for one coefficient. ✅

      ---
   2. **buckingham**

      1. vaschy.py: contains the Pi/PiCoefficient/Coefficient class to represent the dimensionless number resulting of the analysis. ✅

      ---
   3. **core:** shared and core capabilities

      1. basics.py: contains Validation class, shared capabilities for those who need it. ✅
      2. fundamental.py: contains Dimension class, the basis of dimensional analysis (replaces FDU), for the future it need _unit attribute/property. ✅
      3. measurements.py: contains the Unit class, fundamental for unit conversion when necessary, NOT FEASIBLE!!! ❌
      4. parameters.py: contain Variable class to execute the analysis ✅

      ---
   4. **datastructs:** data structures to manage the unit conversion process.

      1. **lists**

         1. arlt.py: arraylist. ✅
         2. sllt.py: single linked list. ✅
         3. dllt.py: double linked list. ✅
         4. ndlt.py: node list for double and single linked. ✅
      2. **tables**

         1. scht.py: separate chaining hashtable. ✅
         2. htme.py: entry used in the separate chaining hashtable. ✅

         ---
   5. **dimensional**

      1. domain.py unit conversion handler/manager for the the matrix UnitsManager, OUT OF SCOPE for now!!!❌
      2. framework.py: contaons de DimFramework class to manage and control the DimMatrix in the solving process. ✅
      3. model.py: contains de DimMatrix class to solve de dimensional matrix. ✅

      ---
   6. **handler**

      1. influence.py: contains the SensitivityHandler class for understanding variance in the coefficients. ✅
      2. phenomena.py: has the main Solver() class of the project. TODO ⚠️
      3. practical.py contains the MonteCarloHandler class to control all the montecartlo simulations of all data ✅ 🔶👨‍💻⚠️ WORKING HERE ⚠️

      ---
   7. **utils**

      1. config.py: contains all global and shared variables for the analysis. ✅
      2. default.py contains all the default stuff needed for custom datastructures + other functionalities, usefull in the future!!! ✅
      3. error.py: contains the generic error_handler() function for all components. ✅
      4. helpers.py: contains any other funcion useful for the process, include MAD for hashtable, check if is prime, and other stuff. ✅
      5. ~~queues.py: library that implement the queue theory for simulations and stuff ✅ ->  ⚠️ REMOVED FROM REPO~~
      6. ~~io.py: contains all the input/ouput functions for saving data of the analyisis, also exports to be use in other platforms (MATPLOTLIB and files!!) NOT NOW❌~~
      7. latex.py: contains all the LaTeX parsing functions for better representation of formulas and stuff. ✅

      ---
   8. ~~math ⚠️⚠️⚠️ TODO ⚠️⚠️⚠️ do i need them????❌ outside of lib scope!!!~~

      1. ~~numbers.py❌~~
      2. ~~queues.py❌~~

      ---
   9. ~~**visualization:** dont NEED it, USE MATPLOTLIB OR OTHER STUFF!!!!, but y need to create plots and charts from vars + coefficients ❌~~

## Tests Path Structure

1. **pydasa**

🔶👨‍💻⚠️ WORKING HERE ⚠️

1. **analysis**

   1. test_conversion.py: tests for unit conversion handler for the solver. NOT NOW!!! ❌
   2. test_scenario.py: tests for sensitivity analysis of the Coefficients TODO ⚠️
   3. test_simulation.py: tests for the monte carlo simulator for one coefficient. TODO ⚠️

   ---
2. **buckingham**

   1. test_vaschy.py: tests for the the Pi/PiCoefficient/Coefficient class. ✅

   ---
3. **core:** shared and core capabilities

   1. test_basics.py: tests for the Validation class. ✅
   2. test_fundamental.py: tests for the Dimension class ✅
   3. test_measurements.py: tests for the Unit class. NOT NOW!!! ❌
   4. test_parameters.py: tests for the Variable class. ✅

   ---
4. **datastructs:** data structures to manage the unit conversion process.

   1. **lists**

      1. test_arlt.py: tests for the arraylist. NOT NOW!!! ❌
      2. test_sllt.py: tests for the single linked list. NOT NOW!!! ❌
      3. test_dllt.py: tests for the double linked list. NOT NOW!!! ❌
      4. test_ndlt.py: tests for the node list for double and single linked. NOT NOW!!! ❌
   2. **tables**

      1. test_scht.py: tests for the separate chaining hashtable. NOT NOW!!! ❌
      2. test_htme.py: tests for the entry useful for the separate chaining hashtable. NOT NOW!!! ❌

      ---
5. **dimensional**

   1. test_domain.py tests for the unit conversion handler/manager. NOT NOW!!! ❌
   2. test_framework.py: test for the DimScheme class to manage and control the DimMatrix in the solving process. ✅
   3. test_model.py: test for the DimMatrix class to solve the dimensional matrix. TODO ⚠️

   ---
6. **handler**

   1. test_influence.py: test for the SensitivityHandler class for understanding variance in the coefficients. ✅
   2. test_phenomena.py: test for the main Solver() class of the project 🔶👨‍💻⚠️ WORKING HERE ⚠️
   3. test_practical.py test for the SimulationManager class to control all the montecartlo simulations of all data ✅

   ---
7. **utils**

   1. test_config.py: test for all global and shared variables for the analysis. ✅
   2. test_default.py test for all the default stuff needed for custom datastructures + other functionalities. NOT NOW!!! ❌
   3. test_errors.py: test for the generic error_handler() function for all components. ✅
   4. test_helpers.py: test for any other funcion useful for the process, include MAD for hashtable, check if is prime, and other stuff. NOT NOW!!! ❌
   5. test_io.py: tests for all the input/ouput functions for saving data of the analyisis, also exports to be use in other platforms (MATPLOTLIB and files!!) NOT NOW!!! ❌
   6. test_latex.py: tests for all the LaTeX parsing functions for better representation of formulas and stuff. ✅

   ---
