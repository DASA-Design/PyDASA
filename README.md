# PySimArch
Library to solve software architecture and physical problems with dimensionless analysis and the Pi-Theorem

## Requirements

I need an object oriented design option to include the following requirements in its specifications

- a class for fundamental dimensions (traditional and extendable to software architecture)
- a class for dimensional parameters and variables, recognizing parameters as ageneralizaction of input, aoutput and control variables, it has to have a name, a symbol, a range (min, max, step) and dimensions.
- a class to manage the data for meassurements and metrics in the real world and software architecture. it need to manage imperial and metrics units and be related with the dimensional parameters.
- a class for dimensionless coefficients or numbers (they ar synonym) with their name, symbol, formula, and relation to their dimensional parameters.
- a method to classify the dimensionless coefficients based on non repeatable and repetead dimensional parameters.
- a class to create algorithmically the dimensionless coefficients with a four-step method described as follows:
	* To create a complete  and mutually independent parameters (variables and constants) thought to be relevant for the process and that can influence the phenomena, this is called a relevance list.
	* To shape this relevance list into a matrixial form divided into two parts. The square core matrix; and the residual matrix. The former contains the fundamental dimensions in the rows (i.e.: L, M, and T, or A, D, and, T) and the most critical dimensional variables as columns (i.e.: ρ, L, and V) and the latter contains the rest of the independently significant variables as columns; in particular, the variable we want to predict as the first one.
	* To linearly transform the core matrix into a unity matrix (ones as diagonal values, and the remaining elements are zero).
	* To divide the variables of the residual matrix by the variables of the unity matrix with the exponents indicated by the unit values of the residual matrix to generate DC/DN.
- a class to check the principle of similitude for traditional problems and extendable into software architecture.
- a class to calculate the dimensionless coefficient range (min, max) and the influence of their dimensional parameters in their behaviour.
- a class to plot or graph possible dimensionless charts using the behaviour of dimensionless coefficients and the dimensional parameters.
