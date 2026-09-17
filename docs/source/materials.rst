Materials
=========

A material is a free-energy density ``W(F; fields, stimuli)``. The
package builds ``F = variable(I + grad u)`` once, calls
:meth:`matto.materials.Material.energy`, and takes the first
Piola-Kirchhoff stress as ``diff(W, F)``.

.. code-block:: python

   from matto.materials import Material, simp, director

   class MyMaterial(Material):
       fields = ("rho", "theta")
       stimuli = {"B_app": (2,)}
       parameters = {"G0": None, "p_rho": 3.0, "eps_rho": 1.0e-6}

       def energy(self, F, fields, stimuli):
           mu = self.G0 * simp(fields["rho"], self.p_rho, self.eps_rho)
           n = director(fields["theta"])
           ...
           return W

``fields`` names the design fields the energy reads; which of them are
optimized is the input file's choice. ``stimuli`` maps each stimulus to
its shape, ``()`` for a scalar and ``(dim,)`` for a vector; the load
cases supply the values. ``parameters`` gives a default or ``None`` for
a required value; construction refuses unknown or missing names.

The class can live in the package, next to the input scripts, or in the
input script itself. Before using a new model in an optimization, run
:func:`matto.materials.check_material` on it: it checks a stress-free
reference, frame indifference with any vector stimulus rotated along,
and that a moderate stretch raises the stored energy.

Shipped models
--------------

.. list-table::
   :header-rows: 1

   * - class
     - fields
     - stimulus
   * - :class:`~matto.materials.HardMagneticSoftMaterial`
     - rho, phi, theta
     - ``B_app``, a vector
   * - :class:`~matto.materials.LiquidCrystalElastomer`
     - rho, phi, theta
     - ``activation``
   * - :class:`~matto.materials.MagnetoActiveElastomer`
     - rho, phi
     - ``h``
   * - :class:`~matto.materials.AnisotropicMagnetoActiveElastomer`
     - rho, phi, theta
     - ``h``
   * - :class:`~matto.materials.LinearElastic`
     - rho
     - none

The hard-magnetic model takes ``dim=3`` for a 3D problem. The linear
elastic model is small-strain and exists so a compliance problem can be
run with the settings of a linear-elastic code and compared.
