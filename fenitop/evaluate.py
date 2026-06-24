"""
Authors:
- Ian Galloway (ian.galloway@mines.sdsmt.edu)
- Prashant Jha (prashant.jha@sdsmt.edu)

Purpose:
- Evaluate a fixed magnetic topology across multiple constitutive models
- Compare displacement and compliance metrics without re-optimization
- Export BP and CSV results for post-processing and model comparison
"""

import os
import csv
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.io
import ufl
from dolfinx import fem

from fenitop.fem import form_fem

def _safe_unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    return np.zeros_like(v)

def _load_npy_or_raise(path: str, label: str) -> np.ndarray:
    if path is None:
        raise RuntimeError(f"{label} path is None")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Design field file not found ({label}): {path}")
    return np.load(path)

def _assign_cg1_field(func: fem.Function, arr: np.ndarray, name: str):
    """
    Assign to CG1 function dofs from numpy array.
    """
    local = func.x.array
    if arr.shape[0] != local.shape[0]:
        raise ValueError(
            f"{name} size mismatch:\n"
            f"  file array: {arr.shape[0]}\n"
            f"  function dofs: {local.shape[0]}\n"
            "Make sure the .npy was saved from the same mesh and same function space (CG1)."
        )
    local[:] = arr[:]
    func.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD
    )


def _theta_from_dir(theta_func: fem.Function, fallback_dir):
    d = np.array(fallback_dir, dtype=float)
    nrm = np.linalg.norm(d)
    if nrm <= 0:
        raise ValueError("theta fallback_dir must be a nonzero vector, e.g. (1.0, 0.0).")
    d /= nrm
    theta0 = float(np.arctan2(d[1], d[0]))

    with theta_func.x.petsc_vec.localForm() as loc:
        loc.set(theta0)
    theta_func.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD
    )


def _get_design_arrays(mesh, design_source: dict):
    """
    Returns (rho_arr, phi_arr, theta_arr) where each may be None.
    Arrays are expected to match CG1 dof layout of the corresponding *_phys_field.
    """
    src_type = design_source.get("type", None)
    if src_type not in ("files", "callable"):
        raise ValueError("design_source['type'] must be 'files' or 'callable'")

    if src_type == "files":
        rho_path = design_source.get("rho", None)
        phi_path = design_source.get("phi", None)
        theta_path = design_source.get("theta", None)

        rho_arr = _load_npy_or_raise(rho_path, "rho") if rho_path is not None else None
        phi_arr = _load_npy_or_raise(phi_path, "phi") if phi_path is not None else None
        theta_arr = _load_npy_or_raise(theta_path, "theta") if theta_path is not None else None

        return rho_arr, phi_arr, theta_arr

    # callable
    builder = design_source.get("builder", None)
    if builder is None or not callable(builder):
        raise ValueError("design_source['builder'] must be a callable when type='callable'")

    out = builder(mesh)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise ValueError("builder(mesh) must return a 3-tuple: (rho_arr|None, phi_arr|None, theta_arr|None)")

    rho_arr, phi_arr, theta_arr = out
    return rho_arr, phi_arr, theta_arr


def _build_minimal_opt(design_variables: dict) -> dict:
    """
    Minimal opt dict required by fem.py / form_fem.
    """
    return {
        # required for rho penalization and some constraint weights
        "penalty": 3.0,
        "epsilon": 1e-6,

        # toggles consumed by fem.py
        "design_variables": design_variables,

        # objective can be anything; we are not optimizing
        "objective_type": "compliance",
        "objective_bcs": [],
    }


def _solve_load_case_with_load_stepping(
    femProblem,
    u_field: fem.Function,
    traction_constants: list,
    fem_params_local: dict,
    opt: dict,
    load_case: dict,
):
    """
    Mirrors topopt load stepping behavior:
      - reset u to 0
      - reset traction constants to 0
      - ramp B_app from 0 → target
      - ramp tractions from 0 → target
    """
    comm = fem_params_local["mesh"].comm
    N_steps = int(fem_params_local.get("load_steps", 1))
    if N_steps < 1:
        N_steps = 1

    # Load-case targets
    B_app_mag_target = float(load_case.get("B_app_mag", fem_params_local.get("B_app_mag", 0.0)))
    B_app_dir_target = _safe_unit(load_case.get("B_app_dir", fem_params_local.get("B_app_dir", (0.0, 0.0))))

    # Start from zero applied field
    opt["B_app"].value[:] = 0.0

    # RESET displacement field
    with u_field.x.petsc_vec.localForm() as loc:
        loc.set(0.0)
    u_field.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD
    )

    # Reset tractions
    for t_const in traction_constants:
        t_const.value = np.zeros_like(t_const.value)

    # Step loads
    traction_bcs = fem_params_local.get("traction_bcs", [])
    for step in range(1, N_steps + 1):
        alpha = step / N_steps

        # Ramp applied magnetic field
        opt["B_app"].value[:] = alpha * B_app_mag_target * B_app_dir_target

        # Ramp tractions
        for t_const, bc_dict in zip(traction_constants, traction_bcs):
            t_max = np.array(bc_dict.get("traction_max", (0.0, 0.0)), dtype=float)
            t_const.value += (1.0 / N_steps) * t_max

        femProblem.solve_fem()

def _build_bp_fields(mesh,
                     rho_phys_field,
                     phi_phys_field,
                     phi_eff_field,
                     u_field,
                     theta_phys_field=None,
                     write_theta=False):
    """
    Builds the list of fields written to BP output.
    """
    fields = [rho_phys_field, phi_phys_field, phi_eff_field, u_field]

    if write_theta and theta_phys_field is not None:
        fields.append(theta_phys_field)

        # m_eff field (debug-friendly)
        dim = mesh.geometry.dim
        V_vec_cg = fem.functionspace(
            mesh,
            ("CG", 1, (dim,))
        )
        m_eff = fem.Function(V_vec_cg, name="m_eff")
        m_expr = ufl.as_vector((
            phi_eff_field * ufl.cos(theta_phys_field),
            phi_eff_field * ufl.sin(theta_phys_field)
        ))
        m_eff.interpolate(fem.Expression(m_expr, V_vec_cg.element.interpolation_points()))
        fields.append(m_eff)

    return fields

def _compute_marker_avg_displacement(u_field: fem.Function, marker_func):
    """
    Computes average displacement over a user-defined marker.

    marker_func should be a callable like:
        lambda x: np.isclose(x[0], L)

    Returns:
        ux_avg, uy_avg, u_mag_avg

    If marker_func is None, or if no DOFs are found on the marker,
    all values return as zero.
    """
    if marker_func is None:
        return 0.0, 0.0, 0.0

    V = u_field.function_space
    coords = V.tabulate_dof_coordinates()

    dim = u_field.function_space.mesh.geometry.dim
    u_vals = u_field.x.array.reshape((-1, dim))

    mask = marker_func(coords.T)

    if np.count_nonzero(mask) == 0:
        return 0.0, 0.0, 0.0

    ux_avg = float(np.mean(u_vals[mask, 0]))
    uy_avg = float(np.mean(u_vals[mask, 1]))

    u_mag_avg = float(np.sqrt(ux_avg**2 + uy_avg**2))

    return ux_avg, uy_avg, u_mag_avg

def evaluate(fem_params: dict,
             eval_config: dict,
             design_source: dict) -> None:
    """
    Sweep constitutive models and evaluate a fixed design (rho/phi/theta).

    Required:
      - fem_params: same structure used by topopt/form_fem
      - eval_config:
          "G_models": list[str]
          "hyperelastic_models": list[str]
          "output_dir": str
        Optional:
          "write_bp": bool (default True)
          "write_csv": bool (default True)
          "csv_name": str (default "model_comparison.csv")
          "measurement_marker": callable, optional
          "compute_compliance": bool (default False)
          "allow_parallel_npy": bool (default False)
      - design_source:
          type="files": {rho, phi, theta paths or None}
          type="callable": {builder(mesh)->(rho,phi,theta)}
    """
    # ---------------------------
    # Validate / defaults
    # ---------------------------
    if "G_models" not in eval_config or "hyperelastic_models" not in eval_config:
        raise KeyError("eval_config must include 'G_models' and 'hyperelastic_models'")

    output_dir = os.path.abspath(eval_config["output_dir"])
    write_bp = bool(eval_config.get("write_bp", True))
    write_csv = bool(eval_config.get("write_csv", True))
    csv_name = str(eval_config.get("csv_name", "model_comparison.csv"))
    measurement_marker = eval_config.get("measurement_marker", None)
    compute_compliance = bool(eval_config.get("compute_compliance", False))

    mesh = fem_params["mesh"]
    comm = mesh.comm

    _ensure_serial_or_raise(comm, eval_config)

    if comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print("[evaluate] output_dir:", output_dir, flush=True)

    # ---------------------------
    # Load / build design arrays
    # ---------------------------
    rho_arr, phi_arr, theta_arr = _get_design_arrays(mesh, design_source)

    # Decide active flags based on array presence
    # (Files-mode: presence of path controls activity. Callable-mode: builder returns None to deactivate.)
    rho_active = (rho_arr is not None)
    phi_active = (phi_arr is not None)
    theta_active = (theta_arr is not None)

    # Allow explicit override if user wants theta active but no file (fallback_dir)
    theta_cfg = design_source.get("theta_cfg", {})
    theta_fallback_dir = theta_cfg.get("fallback_dir", fem_params.get("B_rem_dir", (1.0, 0.0)))
    theta_force_active = bool(theta_cfg.get("force_active", False))
    if theta_force_active:
        theta_active = True

    design_variables = {
        "rho": {"active": bool(rho_active)},
        "phi": {"active": bool(phi_active)},
        "theta": {"active": bool(theta_active)},
    }

    # ---------------------------
    # Sweep models
    # ---------------------------
    rows = []

    G_models = list(eval_config["G_models"])
    H_models = list(eval_config["hyperelastic_models"])

    for hyperModel in H_models:
        for G_model in G_models:

            fem_params_local = dict(fem_params)
            fem_params_local["hyperModel"] = hyperModel
            fem_params_local["G_model"] = G_model

            # If theta is NOT active, we want a deterministic remanence direction.
            # This must be set before form_fem() so fem.py initializes theta_phys_field properly.
            if not theta_active:
                fem_params_local["B_rem_dir"] = tuple(_safe_unit(theta_fallback_dir))

            # Minimal opt dict (form_fem uses it)
            opt = _build_minimal_opt(design_variables)

            # Build FEM once per (hyperModel, G_model)
            (
                femProblem,
                u_field,
                lambda_field,
                rho_field,
                rho_phys_field,
                phi_field,
                phi_phys_field,
                phi_eff_field,
                theta_phys_field,
                traction_constants,
                ds,
            ) = form_fem(fem_params_local, opt)

            # Assign design fields (physical CG1)
            if rho_active:
                _assign_cg1_field(rho_phys_field, rho_arr, "rho_phys")
            # else: fem.py already froze rho_phys=1

            if phi_active:
                _assign_cg1_field(phi_phys_field, phi_arr, "phi_phys")
            # else: fem.py already froze phi_phys=0

            if theta_active:
                if theta_arr is not None:
                    _assign_cg1_field(theta_phys_field, theta_arr, "theta_phys")
                else:
                    # active but no array: use fallback_dir to set theta_phys_field angle
                    _theta_from_dir(theta_phys_field, theta_fallback_dir)

            # Update phi_eff_field = rho_phys * phi_phys (for output convenience)
            phi_eff_field.x.array[:] = (rho_phys_field.x.array * phi_phys_field.x.array)
            phi_eff_field.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD
            )

            # Load cases
            load_cases = fem_params_local.get("load_cases", None)
            if load_cases is None:
                load_cases = [{"name": "single"}]

            # Open one BP writer per (G_model, HyperModel), with load cases as time steps
            bp_writer = None
            if write_bp:
                bp_name = f"disp_{G_model}_{hyperModel}.bp"
                bp_path = os.path.join(output_dir, bp_name)

                bp_fields = _build_bp_fields(
                    mesh=mesh,
                    rho_phys_field=rho_phys_field,
                    phi_phys_field=phi_phys_field,
                    phi_eff_field=phi_eff_field,
                    u_field=u_field,
                    theta_phys_field=theta_phys_field,
                    write_theta=bool(theta_active),
                )

                bp_writer = dolfinx.io.VTXWriter(
                    mesh.comm,
                    bp_path,
                    bp_fields,
                    engine="BP4"
                )

            # Loop load cases
            for lc in load_cases:
                lc_name = lc.get("name", "unnamed")

                # Set traction_max targets for this load case (used by load stepping)
                traction_bcs = fem_params_local.get("traction_bcs", [])
                for bc_dict in traction_bcs:
                    name = bc_dict.get("name", None)
                    if name is not None and name in lc.get("tractions", {}):
                        bc_dict["traction_max"] = lc["tractions"][name]
                    else:
                        bc_dict["traction_max"] = (0.0, 0.0)

                # Solve with load stepping (ramp B_app and tractions)
                _solve_load_case_with_load_stepping(
                    femProblem=femProblem,
                    u_field=u_field,
                    traction_constants=traction_constants,
                    fem_params_local=fem_params_local,
                    opt=opt,
                    load_case=lc
                )

                # Metrics
                ux_avg, uy_avg, u_mag_avg = _compute_marker_avg_displacement(
                    u_field,
                    measurement_marker
                )

                comp_val = None
                if compute_compliance:
                    # Compliance (work): ∫ u·b dx + Σ ∫ u·t ds
                    # Here b is in fem_params_local["body_force"] (usually 0 in your eval setups).
                    b_vec = np.array(fem_params_local.get("body_force", (0.0, 0.0)), dtype=float)
                    b = fem.Constant(mesh, PETSc.ScalarType(b_vec))
                    metadata = {"quadrature_degree": fem_params_local.get("quadrature_degree", 2)}
                    dx = ufl.Measure("dx", metadata=metadata)
                    Jform = ufl.inner(u_field, b) * dx
                    for marker, t_const in enumerate(traction_constants):
                        Jform += ufl.inner(u_field, t_const) * ds(marker)
                    comp_local = fem.assemble_scalar(fem.form(Jform))
                    comp_val = comm.allreduce(comp_local, op=MPI.SUM)

                if comm.rank == 0:
                    msg = f"[{G_model:>7s} | {hyperModel:>11s} | {lc_name}]"
                    msg += (
                        f"  ux_avg = {ux_avg:.6e}"
                        f"  uy_avg = {uy_avg:.6e}"
                        f"  |u|_avg = {u_mag_avg:.6e}"
                    )
                    if compute_compliance:
                        msg += f"  comp = {float(comp_val):.6e}"
                    print(msg, flush=True)

                # Output BP
                if bp_writer is not None:
                    bp_time = float(lc.get("B_app_mag", 0.0))
                    bp_writer.write(bp_time)

                # Record row
                row = {
                    "G_model": G_model,
                    "HyperModel": hyperModel,
                    "LoadCase": lc_name,
                }
                row["ux_avg"] = ux_avg
                row["uy_avg"] = uy_avg
                row["u_mag_avg"] = u_mag_avg
                if compute_compliance:
                    row["Compliance"] = float(comp_val)

                rows.append(row)
                
            if bp_writer is not None:
                bp_writer.close()

    # ---------------------------
    # Write CSV
    # ---------------------------
    if comm.rank == 0 and write_csv:
        csv_path = os.path.join(output_dir, csv_name)

        fieldnames = [
            "G_model",
            "HyperModel",
            "LoadCase",
            "ux_avg",
            "uy_avg",
            "u_mag_avg",
        ]

        if compute_compliance:
            fieldnames += ["Compliance"]

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[evaluate] wrote CSV: {csv_path}", flush=True)