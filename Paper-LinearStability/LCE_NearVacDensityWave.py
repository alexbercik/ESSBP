#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCE density wave local growth analysis.
Reproduces figures from section 3.3 of the paper.

Runs baseflow and a perturbed run for 4 volume-dissipation settings:
  1) no volume dissipation  (nd)
  2) no volume dissipation + positivity enforcement
  2) dcp volume dissipation
  3) entdcp volume dissipation

Tracks:
  - H-norm of the perturbation v(t)
  - H-norm of the baseflow error e(t) = ||u(t) - u_exact(t)||_H
  - maximum eigenvalue of the (smooth baseflow) Jacobian J(u(t))
  - maximum eigenvalue of the symmetric part of the Jacobian, J_sym
  - local growth rate of v along J_sym (Rayleigh quotient)
  - minimum value of the baseflow u(x,t) over space as a function of time
"""

import os
from sys import path

import numpy as np
import matplotlib.pyplot as plt
from Floquet_Analysis import run_floquet


n_nested_folder = 1
folder_path, _ = os.path.split(__file__)
for _ in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)
path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp

# Shared parameters
para = 1.0
tm_method = "rk4"
dt = 0.001
tf = 25.0

xmin = 0.0
xmax = 1.0
bc = "periodic"

disc_type = "had"
disc_nodes = "circulant"
p = 8
nelem = 1
nen = 39
had_flux = "logarithmic"

surf_type = {"diss_type": "nd", "fluxvec": "lf", "coeff": 1.0}
q0_type = "density_wave"
cons_obj_name = ("error", "time")
plot_bound = True

# Volume dissipation choices
s = int(p / 2) + 1
coeff = 0.625 / 5 ** s

# If not None, save figures as PDF with suffixes:
#   {savefile}_vH.pdf, {savefile}_eH.pdf, ...
savefile = None #'LCE_densitywave_near0_circulant_n39'

settings = {
    "warp_factor": 0.0,
    "warp_type": "none",
    "jac_method": "exact",
}

# Prediction anchor times for Floquet modal growth
pred_T0 = 0.0
pred_T1 = 5.0

cases = {
    "No diss.": {
        "diss_type": "nd",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "scalarscalar",
        "s": s,
        "coeff": coeff,
        "enforce_positivity": False,
    },
    "No diss. + pos.": {
        "diss_type": "nd",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "scalarscalar",
        "s": s,
        "coeff": coeff,
        "enforce_positivity": True,
    },
    "Cons. diss.": {
        "diss_type": "dcp",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "scalarscalar",
        "s": s,
        "coeff": coeff,
        "enforce_positivity": False,
    },
    "Ent. diss.": {
        "diss_type": "entdcp",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "scalarscalar",
        "s": s,
        "coeff": coeff,
        "enforce_positivity": False,
    },
}

colors = {
    "No diss.": "tab:blue",
    "No diss. + pos.": "tab:green",
    "Cons. diss.": "tab:orange",
    "Ent. diss.": "tab:red",
}

# Collect results per case
results = {}

# indices for the conservation objective components
time_idx = [name.lower() for name in cons_obj_name].index("time")
error_idx = [name.lower() for name in cons_obj_name].index("error")

for case_label, vol_diss in cases.items():
    print(f"Running case: {case_label}")

    diffeq_case = LinearConv(para, q0_type, had_flux=had_flux)

    solver = PdeSolverSbp(
        diffeq_case,
        settings,
        tm_method,
        dt,
        tf,
        p,
        disc_type,
        surf_type,
        vol_diss,
        had_flux,
        nelem,
        nen,
        disc_nodes,
        bc,
        xmin,
        xmax,
        cons_obj_name,
        bool_plot_sol=False,
        print_sol_norm=False,
        print_progress=True,
    )
    solver.tm_atol = 1e-8
    solver.tm_rtol = 1e-8
    if vol_diss["enforce_positivity"]:
        solver.clip_positivity = True
        solver.clip_positivity_floor = 5e-4
        solver.clip_positivity_cut = 5e-2

    T = 1.0
    K_floquet = 5000
    ampli = 1e-4
    floq = run_floquet(T=T, solver=solver, K=K_floquet, use_H=True, track_max_real_eig=False)
    max_growth_rate0 = float(np.max(floq["growth_rates"]))
    eigvals_M, eigvecs_M = np.linalg.eig(floq["M"])
    idx_max = int(np.argmax(eigvals_M.real))
    pert0 = ampli * eigvecs_M[:, idx_max].real.reshape(solver.qshape, order="F")

    # Baseflow solve
    q0 = diffeq_case.set_q0()
    solver.solve(q0=q0)
    base_qsol = np.copy(solver.q_sol)
    base_time = np.copy(solver.cons_obj[time_idx, :])
    base_error = np.copy(solver.cons_obj[error_idx, :])

    # Floquet modal growth rate at later anchor time (for prediction)
    max_growth_rate1 = np.nan
    idx_T1 = None
    if pred_T1 < base_time[-1]:
        idx_T1 = int(np.argmin(np.abs(base_time - pred_T1)))
        diffeq_case.set_q0_discrete(base_qsol[:, :, idx_T1], k=7, s=0, periodic=True)
        floq1 = run_floquet(T=T, solver=solver, K=K_floquet, use_H=True, track_max_real_eig=False)
        max_growth_rate1 = float(np.max(floq1["growth_rates"]))

    # Perturbed solve (u + v, but we only *analyze* v against smooth-baseflow J)
    solver.solve(q0=q0 + pert0)
    k_min = min(base_qsol.shape[2], solver.q_sol.shape[2])

    base_time = base_time[:k_min]
    base_error = base_error[:k_min]
    base_qsol = base_qsol[:, :, :k_min]
    pert = solver.q_sol[:, :, :k_min] - base_qsol

    # Track v(t) in H-norm
    v_H = np.sqrt(solver.energy(pert))

    # Track baseflow error in H-norm
    e_H = base_error

    # Jacobian-based quantities over time:
    #   J(t)     = calc_RHS_jac(u(t))
    #   J_sym(t) = (J(t) + J(t)^T)/2
    #   lambda_max_J(t)    = max Re(eig(J(t)))          (full J)
    #   lambda_max_J_sym(t)= max eig(J_sym(t))          (symmetric part)
    #   g(t) = (v(t)^T J_sym(t) v(t)) / (v(t)^T v(t))   (Rayleigh quotient)
    jac_stride = 1
    idxs_to_jac = range(0, k_min, jac_stride)

    lambda_max_J = np.full(k_min, np.nan, dtype=float)
    lambda_max_J_sym = np.full(k_min, np.nan, dtype=float)
    g = np.full(k_min, np.nan, dtype=float)

    for i in idxs_to_jac:
        J = solver.calc_RHS_jac(q=base_qsol[:, :, i], exact_dfdq=False)
        J_sym = 0.5 * (J + J.T)

        eigvals_J = np.linalg.eigvals(J)
        lambda_max_J[i] = float(np.max(eigvals_J.real))

        # Symmetric eigenvalue solver for numerical stability
        eigvals_J_sym = np.linalg.eigvalsh(J_sym.real)
        lambda_max_J_sym[i] = float(np.max(eigvals_J_sym))

        v_vec = pert[:, :, i].flatten("F")
        v2 = float(np.vdot(v_vec, v_vec).real)
        if v2 >= 1e-30:
            g[i] = float(np.vdot(v_vec, J_sym @ v_vec).real / v2)

    # Track min(u(x,t)) over space (baseflow only)
    min_u = np.min(base_qsol[:, :, :k_min], axis=(0, 1))
    max_u = np.max(base_qsol[:, :, :k_min], axis=(0, 1))

    # Predictions for ||v||_H growth at anchor times T0 and T1
    idx_T0 = int(np.argmin(np.abs(base_time - pred_T0)))
    pred0 = np.full(k_min, np.nan, dtype=float)
    pred5 = np.full(k_min, np.nan, dtype=float)
    pred0[idx_T0:] = v_H[idx_T0] * np.exp(max_growth_rate0 * (base_time[idx_T0:] - base_time[idx_T0]))
    if idx_T1 is not None and idx_T1 < k_min and np.isfinite(max_growth_rate1):
        pred5[idx_T1:] = v_H[idx_T1] * np.exp(
            max_growth_rate1 * (base_time[idx_T1:] - base_time[idx_T1])
        )

    results[case_label] = {
        "t": base_time,
        "v_H": v_H,
        "e_H": e_H,
        "lambda_max_J": lambda_max_J,
        "lambda_max_J_sym": lambda_max_J_sym,
        "g": g,
        "min_u": min_u,
        "max_u": max_u,
        "pred0": pred0,
        "pred5": pred5,
    }

# ---- Plot 1: H-norm of perturbation v(t)
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
    plt.semilogy(
        data["t"],
        data["v_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
    )

# Capture y-limits based only on actual data
ax = plt.gca()
ymin, ymax = ax.get_ylim()

# Floquet modal growth predictions (dashed, same color) without affecting y autoscaling
for case_label, data in results.items():
    t = data["t"]

    y_pred0 = data["pred0"]
    mask0 = np.isfinite(y_pred0)
    plt.semilogy(
        t[mask0],
        y_pred0[mask0],
        color=colors[case_label],
        linestyle="--",
        linewidth=1.2,
    )

    y_pred5 = data["pred5"]
    mask5 = np.isfinite(y_pred5)
    plt.semilogy(
        t[mask5],
        y_pred5[mask5],
        color=colors[case_label],
        linestyle="--",
        linewidth=1.2,
    )

if plot_bound:
    for case_label, data in results.items():
        t = data["t"]
        min_u = data["min_u"]
        max_u = data["max_u"]
        bound = np.sqrt(max_u / min_u) * data["v_H"][0]
        plt.semilogy(t, bound, color=colors[case_label], linestyle=":", linewidth=1.2)


ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vH.pdf", format="pdf")

# ---- Plot 2: H-norm of baseflow error e(t)
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
    plt.semilogy(
        data["t"],
        data["e_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
    )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{e} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.ylim(ymin=data["e_H"][3])
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_eH.pdf", format="pdf")

# ---- Plot 3: lambda_max(J) over time
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
        y = data["lambda_max_J"]
        x = data["t"]
        mask = np.isfinite(y)
        plt.semilogy(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
        )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\max \Re(\lambda(\mathsf{J}))$", fontsize=16)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_lambdaJ.pdf", format="pdf")

# ---- Plot 4: lambda_max(J_sym) over time
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
        y = data["lambda_max_J_sym"]
        x = data["t"]
        mask = np.isfinite(y)
        plt.semilogy(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
        )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\lambda_{\max}\left(\tfrac{1}{2}(\mathsf{J}+\mathsf{J}^\mathsf{T})\right)$", fontsize=16)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_lambdaJsym.pdf", format="pdf")

# ---- Plot 5: local growth rate g(t) of v along J_sym
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
        y = data["g"]
        x = data["t"]
        mask = np.isfinite(y)
        plt.plot(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
        )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\frac{\boldsymbol{v}^{\mathsf{T}} \mathsf{J}_{\mathrm{sym}}\boldsymbol{v}}{\boldsymbol{v}^{\mathsf{T}}\boldsymbol{v}}$", fontsize=20)
plt.yscale("symlog", linthresh=1e-2)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_g.pdf", format="pdf")

# ---- Plot 6: min(u(x,t)) over space
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
    plt.plot(
        data["t"],
        data["min_u"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
    )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\min \boldsymbol{u}$", fontsize=16)
plt.yscale("symlog", linthresh=1e-5)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_minu.pdf", format="pdf")

plt.show()

# Minimal summary for quick comparison
print("\nSummary (computed from smooth baseflow over time):")
for case_label, data in results.items():
    lamJ_max = float(np.nanmax(data["lambda_max_J"]))
    lamJsym_max = float(np.nanmax(data["lambda_max_J_sym"]))
    print(
        f"  {case_label}: max Re(lambda(J))={lamJ_max:.6g}, "
        f"max eig(J_sym)={lamJsym_max:.6g}"
    )

