Quick Start
===========

Basic Usage
-----------

Here's a simple example of using PyDASA:

.. code-block:: python

   from pydasa.handler.phenomena import Solver
   from pydasa.core.parameters import Variable
   from pydasa.core.fundamental import Dimension

   # Define dimensions
   L = Dimension(length=1)
   T = Dimension(time=1)
   M = Dimension(mass=1)

   # Define variables
   velocity = Variable("v", L/T)
   length = Variable("L", L)
   time = Variable("t", T)

   # Create solver
   solver = Solver()
   solver.add_variable(velocity)
   solver.add_variable(length)
   solver.add_variable(time)

   # Solve for dimensionless numbers
   result = solver.solve()
   print(result)

Next Steps
----------

* Read the :doc:`../user_guide/dimensional_analysis` guide
* Explore :doc:`../examples/basic_example`
* Check the :doc:`../api/index` for detailed reference