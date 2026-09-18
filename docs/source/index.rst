MatTO
=====

Material and topology optimization of stimulus-responsive soft
materials: nonlinear finite-element analysis with FEniCSx and
gradient-based optimization of structural density, material
composition and material orientation, with the constitutive model kept
apart from the optimization machinery. The supported material models
are hard-magnetic soft materials, isotropic and anisotropic
magneto-active elastomers, and liquid crystal elastomers; a new model is
a class with a free-energy density.

Source and issues: `github.com/CEADpx/matto <https://github.com/CEADpx/matto>`_.

Installation
------------

FEniCSx (``dolfinx``, ``basix``, ``ufl``, ``petsc4py``, ``mpi4py``) is
installed from ``conda-forge``, not pip. The environment file pins the
tested versions.

.. code-block:: bash

   conda env create -f environment.yml
   conda activate confenx
   python -m pip install -e .

Running an example
------------------

Each example is one input script that builds a ``problem`` dictionary
and runs it:

.. code-block:: bash

   python examples/hMSM/input_beam.py
   mpirun -n 4 python examples/hMSM/input_beam.py

Results go to the ``output_dir`` named in the script: a ``.bp`` file per
load case for ParaView, the final design fields as ``.npy`` arrays, and
``final_results.txt``.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   inputs
   materials
   testing

.. toctree::
   :maxdepth: 1
   :caption: API

   api/driver
   api/state
   api/design
   api/sensitivity
   api/postprocess
   api/materials
   api/mma
   api/utility
