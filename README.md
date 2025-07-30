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

## Path Structure

1. **pydasa**

   1. **analysis**

      1. influence.py: contains the Sensitivity class for understanding variance in the coefficients.
      2. problem.py: has the main Solver() class of the project
      3. scenarios.py: creates alternative escenarios iterating over Sensitivty reports and montecarlo???? pretty advance, next iteration!!! what iterate over the relevance list and find unique an repeated coeffcients??
      4. simulation.py: monte carlo slmulator of the coeffcients

      ---
   2. **buckingham**

      1. vaschy.py: contains the Pi/PiCoefficient/Coefficient class to represent the dimensionless number resulting of the analysis.

      ---
   3. **core:** shared and core capabilities

      1. basics.py: contains Validation class, shared capabilities for those who need it.
      2. fundamental.py: contains Dimension class, the basis of dimensional analysis (replaces FDU), for the future it need _unit attribute/property.
      3. measurements.py: contains the Metric class, fundamental for unit conversion when necessary, NOT FEASIBLE!!!
      4. parameters.py: contain Variable class to execute the analysis

      ---
   4. **datastructs:** data structures to manage the unit conversion process.

      1. **lists**

         1. dymarray.py: arraylist.
         2. singlelinked.py: single linked list.
         3. doublelinked.py: double linked list.
         4. nodes.py: node list for double and single linked.
      2. **tables**

         1. chaining.py: separate chaining hashtable.
         2. entry.py: entry useful for the separate chaining hashtable

         ---
   5. **dimensional**

      1. model.py: contains de DModel class to solve de dimensiona matrix.
      2. framework.py: contaons de DFramwork class to manage and control de DMatrix in the solving process.
      3. conversion.py unit conversion handler for the solver, OUT OF SCOPE for now!!!

      ---
   6. **utils**

      1. config.py: contains all global and shared variables for the analysis.
      2. default.py contains all the default stuff needed for custom datastructures + other functionalities, usefull in the future!!!
      3. errors.py: contains the generic error_handler() function for all components.
      4. helpers.py: contains any other funcion useful for the process, include MAD for hashtable, check if is prime, and other stuff
      5. io.py: contains all the input/ouput functions for saving data of the analyisis, also exports to be use in other platforms (MATPLOTLIB!!)

      ---
   7. ~~**visualization:** dont NEED it, USE MATPLOTLIB OR OTHER STUFF!!!!, but y need to create plots and charts from vars + coefficients~~
