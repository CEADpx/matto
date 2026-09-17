Writing an input file
=====================

An input file is a Python script that assembles one ``problem``
dictionary and hands it to :class:`matto.driver.OptimizationDriver`.
The examples under ``examples/`` are the reference; the closest one to a
new problem is the right starting point.

.. code-block:: python

   from matto.driver import OptimizationDriver
   from matto.design import volume_constraint
   from matto.materials import HardMagneticSoftMaterial

   problem = {
       "mesh": mesh,
       "mesh_serial": mesh_serial,
       "design_variables": design_variables,
       "boundary_conditions": boundary_conditions,
       "traction_boundaries": traction_boundaries,
       "load_steps": load_steps,
       "load_cases": load_cases,
       "material": HardMagneticSoftMaterial(**material_parameters),
       "build_objective": build_objective,
       "build_constraints": build_constraints,
       "requested_output_fields": requested_output_fields,
       "fem_options": fem_options,
       "optimization_options": optimization_options,
       "output_options": output_options,
   }

   OptimizationDriver(problem).run()

What the dictionary holds
-------------------------

``mesh``, ``mesh_serial``
    A dolfinx mesh on the working communicator, and a copy of it on
    ``MPI.COMM_SELF`` on rank 0 (``None`` elsewhere), used to gather the
    final fields into serial ordering for the saved arrays. 2D and 3D
    meshes are supported.

``design_variables``
    One entry per design field. Each gives ``active`` (optimized or
    prescribed), ``initial``, ``bounds``, and the chain of
    ``operators`` that maps the raw field the optimizer controls to the
    physical field the material model reads:

    .. code-block:: python

       "rho": {
           "active": True,
           "initial": 0.5,
           "bounds": (0.05, 1.0),
           "operators": [
               {"type": "density_filter", "radius": 1.0},
               {"type": "heaviside", "beta_initial": 1.0,
                "beta_update_interval": 25, "beta_max": 4.0},
           ],
       }

    The raw field lives in ``("DG", 0)`` and the physical field in
    ``("CG", 1)`` unless ``raw_space`` and ``physical_space`` say
    otherwise. An inactive variable takes ``prescribed_value``
    (default: ``initial``) as its physical field. ``fixed_regions``
    holds regions where the raw field is held at a value.

``boundary_conditions``, ``traction_boundaries``
    Dirichlet conditions as ``{name, on_boundary, value}`` and named
    facet sets on which load cases can apply tractions.

``load_steps``, ``load_cases``
    Every load case is solved from the undeformed state, its body force,
    tractions and stimuli ramped together in ``load_steps`` equal
    increments. A load case names its ``tractions`` by boundary and its
    ``stimuli`` by name; the material declares which stimuli it needs
    and with what shape. Load cases carry a ``weight`` in the objective.

``material``
    A :class:`matto.materials.Material`: one of the shipped models, or a
    subclass defined in the input script itself. The material declares
    the design fields it reads, the stimuli it needs, and its
    parameters; a mismatch with the rest of the problem is reported at
    construction.

``build_objective(u_field, external_work, dx)``
    Returns the objective as a UFL form. ``external_work`` is the work
    of the applied loads, the compliance.

``build_constraints(design_variables, dx)``
    Returns ``{name: {"form", "normalize_by", "upper_bound"}}``.
    :func:`matto.design.volume_constraint` builds the usual volume
    constraint. Constraints may depend on the design fields but not on
    the displacement.

``requested_output_fields``, ``build_output_fields``
    Which fields go to the ``.bp`` files. ``u`` and every ``<name>_raw``
    and ``<name>_phys`` are available; ``build_output_fields`` can add
    derived UFL expressions.

``fem_options``
    ``quadrature_degree`` and ``solver_options`` with ``state``,
    ``adjoint`` and ``filter`` blocks. Each block takes PETSc options;
    the state block also takes the Newton tolerances.

``optimization_options``
    ``max_iter``, ``opt_tol`` and the MMA ``move`` limit.

``output_options``
    ``output_dir`` and ``sim_output_interval``.

Using the driver without optimizing
-----------------------------------

Construction does all the setup and validation. After that the object
can be used for a single analysis or a gradient without an MMA step:

.. code-block:: python

   driver = OptimizationDriver(problem)
   driver.forward()                       # raw -> physical fields
   result = driver.evaluate()             # every load case; objective,
                                          # gradients, constraints
   name, max_u = driver.solve_load_case(problem["load_cases"][0])

or the state problem alone:

.. code-block:: python

   from matto.state import StateProblem

   state = StateProblem(problem, driver.design_variables)
   max_u = state.solve(load_case, load_steps)
