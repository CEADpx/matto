Tests
=====

.. code-block:: bash

   python -m pytest
   mpirun -n 2 python -m pytest
   mpirun -n 4 python -m pytest
   mpirun -n 1 python -m pytest -m examples

The default run is the serial suite: adjoint checks of every operator
by the dot-product identity, finite-difference checks of the full
adjoint on coarse 2D and 3D beams, repeated-evaluation and constraint
gradient checks, material consistency checks, and regression values for
the beam. The ``examples`` marker runs one iteration of every example
input and compares the first objective with the recorded value; run it
under ``mpirun`` as well, since some errors only appear with more than
one rank.
