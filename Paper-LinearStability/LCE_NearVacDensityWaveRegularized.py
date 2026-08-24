#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCE density wave: regularized two-point fluxes with and without conservative
volume dissipation.

Compares four Hadamard fluxes / dissipation settings:
  1) arcsinh without dissipation
  2) arcsinh with conservative (dcp) dissipation
  3) central without dissipation
  4) central with conservative (dcp) dissipation

First plots instantaneous eigenvalues of L at t=0 and Floquet exponents
for the regularized flux, overlaying the same u_reg values used in the
perturbation experiments. Then runs the baseflow + perturbation study
and produces the same time-series figures as LCE_NearVacDensityWave.py.
"""

import os
from sys import path

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

from Floquet_Analysis import baseflow_u, operator_L_from_u, run_floquet


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
tf = 100.0

xmin = 0.0
xmax = 1.0
bc = "periodic"

disc_type = "had"
disc_nodes = "circulant"  # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd', 'upwind', 'circulant'
p = 8
nelem = 1
nen = 39

surf_type = {"diss_type": "nd", "fluxvec": "lf", "coeff": 1.0}
q0_type = "density_wave"
cons_obj_name = ("error", "time")
plot_bound = False

# Volume dissipation choices
s = int(p / 2) + 1
coeff = 0.625 / 5 ** s

# If not None, save figures as PDF with suffixes:
#   {savefile}_eigs.pdf, {savefile}_floquet.pdf, {savefile}_vH.pdf, ...
savefile = None  # 'LCE_densitywave_near0_regularized_circulant_n39'

settings = {
    "warp_factor": 0.0,
    "warp_type": "none",
    "jac_method": "exact",
}

# Prediction anchor time for Floquet modal growth of ||v||_H
pred_T0 = 0.0

# Initial perturbation for the growth experiment:
#   "floquet"  leading Floquet mode of that discretization
#   "noise"    random noise, identical across cases (same seed + same grid)
pert_init = "noise"  # "floquet" or "noise"
pert_seed = 0
plot_floquet_pred = False  # dashed Floquet growth curves on the ||v||_H plot

# Floquet / IC spectrum (same grid as the perturbation experiments)
T = 1.0
K_floquet = 10000
ampli = 1e-4
normalize_eigs = False

vol_diss_nd = {
    "diss_type": "nd",
    "use_H": False,
    "bdy_fix": False,
    "jac_type": "scalarscalar",
    "s": s,
    "coeff": coeff,
    "enforce_positivity": False,
}
vol_diss_dcp = {
    "diss_type": "dcp",
    "use_H": False,
    "bdy_fix": False,
    "jac_type": "scalarscalar",
    "s": s,
    "coeff": coeff,
    "enforce_positivity": False,
}

# Regularized two-point flux compared across ureg on the perturbation plots.
# Pick the flux once; the same ureg list is applied to it.
reg_had_flux = "arcsinh"  # "arcsinh" or "logreg"
uregs = (0.05, 0.1, 0.5)
ureg_colors = {
    0.05: "tab:green",
    0.1: "tab:red",
    0.5: "tab:orange",
}
# uregs order: plus, x, circle. Draw circle first (behind), then x, then plus.
_ureg_marker_cycle = ("+", "x", "o")
if len(uregs) > len(_ureg_marker_cycle):
    raise ValueError("Need a marker for each ureg (plus, x, circle).")
ureg_markers = {u: _ureg_marker_cycle[i] for i, u in enumerate(uregs)}
spec_plot_uregs = tuple(reversed(uregs))
spec_legend_order = tuple(range(len(uregs) - 1, -1, -1))
logreg_m = 4  # integer >= 2; ignored unless reg_had_flux == "logreg"

labels2 = {'log': 'logarithmic', 
        'geom': 'geometric', 
        'arcsinh': 'arcsinh', 
        'central': 'central',
        '$u_{\\mathrm{reg}}=0.5$': r'asinh, $u_{\mathrm{reg}}=0.5$',
        '$u_{\\mathrm{reg}}=0.1$': r'asinh, $u_{\mathrm{reg}}=0.1$',
        '$u_{\\mathrm{reg}}=0.05$': r'asinh, $u_{\mathrm{reg}}=0.05$'}


def ureg_label(ureg):
    return rf"$u_{{\mathrm{{reg}}}}={ureg:g}$"


def print_reg_params(had_flux, ureg, logreg_m):
    flux = had_flux.lower()
    if flux in ("arcsinh", "logreg"):
        print(f"  ureg = {ureg:g}")
    if flux == "logreg":
        print(f"  logreg_m = {logreg_m}")


cases = {
    "log": {
        "had_flux": "logarithmic",
        "vol_diss": dict(vol_diss_nd),
    },
    # "geom": {
    #     "had_flux": "geometric",
    #     "vol_diss": dict(vol_diss_nd),
    # },
    # "Central, no diss.": {
    #     "had_flux": "central",
    #     "vol_diss": dict(vol_diss_nd),
    # },
    # "Central, cons. diss.": {
    #     "had_flux": "central",
    #     "vol_diss": dict(vol_diss_dcp),
    # },
}
for _ureg in uregs:
    cases[ureg_label(_ureg)] = {
        "had_flux": reg_had_flux,
        "vol_diss": dict(vol_diss_nd),
        "ureg": _ureg,
        "logreg_m": logreg_m,
    }

colors = {
    "Arcsinh, no diss.": "tab:orange",
    "Arcsinh, cons. diss.": "tab:red",
    "Central, no diss.": "tab:blue",
    "Central, cons. diss.": "tab:green",
    "log": "tab:blue",
    "geom": "tab:green",
}
for _ureg in uregs:
    colors[ureg_label(_ureg)] = ureg_colors[_ureg]

markers = {
    "Arcsinh, no diss.": "+",
    "Arcsinh, cons. diss.": "x",
    "Central, no diss.": "o",
    "Central, cons. diss.": "s",
}

# Collect results per case
spectra = {}
results = {}

# indices for the conservation objective components
time_idx = [name.lower() for name in cons_obj_name].index("time")
error_idx = [name.lower() for name in cons_obj_name].index("error")


def make_solver(had_flux, vol_diss, nen=nen, p=p, nelem=nelem,
               disc_nodes=disc_nodes, print_progress=True,
               ureg=0.1, logreg_m=2):
    flux = had_flux.lower()
    diffeq_kw = {}
    if flux == "arcsinh":
        diffeq_kw["arcsinh_ureg"] = ureg
    elif flux == "logreg":
        diffeq_kw["logreg_ureg"] = ureg
        diffeq_kw["logreg_m"] = int(logreg_m)
    diffeq_case = LinearConv(para, q0_type, had_flux=had_flux, **diffeq_kw)
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
        print_progress=print_progress,
    )
    solver.tm_atol = 1e-8
    solver.tm_rtol = 1e-8
    if vol_diss["enforce_positivity"]:
        solver.clip_positivity = True
        solver.clip_positivity_floor = 5e-4
        solver.clip_positivity_cut = 5e-2
    return diffeq_case, solver


def compute_spectrum(case_label, case, print_progress=False):
    had_flux = case["had_flux"]
    vol_diss = case["vol_diss"]
    ureg = case.get("ureg", 0.1)
    this_logreg_m = case.get("logreg_m", logreg_m)
    print(f"\nFloquet / IC spectrum: {case_label},  K = {K_floquet}")
    print_reg_params(had_flux, ureg, this_logreg_m)
    _, solver = make_solver(
        had_flux, vol_diss,
        nen=nen, p=p, nelem=nelem,
        disc_nodes=disc_nodes,
        print_progress=print_progress,
        ureg=ureg, logreg_m=this_logreg_m,
    )
    dx_n = solver.mesh.x[1] - solver.mesh.x[0]
    L0 = operator_L_from_u(baseflow_u(0.0, solver), solver)
    eigs0 = eigvals(L0)
    floq = run_floquet(T=T, solver=solver, K=K_floquet, use_H=True,
                       track_max_real_eig=False)
    max_growth_rate0 = float(np.max(floq["growth_rates"]))
    print(f"  Max Re(lambda(L(t=0))) = {float(np.max(eigs0.real)):.6g}")
    print(f"  Max Floquet multiplier  = {float(np.max(np.abs(floq['rho']))):.6g}")
    print(f"  Max Floquet growth rate = {max_growth_rate0:.6g}")
    eigvals_M, eigvecs_M = np.linalg.eig(floq["M"])
    idx_max = int(np.argmax(eigvals_M.real))
    pert0 = ampli * eigvecs_M[:, idx_max].real.reshape(solver.qshape, order="F")
    spectra[case_label] = {
        "eigs": eigs0,
        "exponents": floq["exponents"],
        "dx": dx_n,
        "max_growth_rate0": max_growth_rate0,
        "pert0": pert0,
        "had_flux": had_flux,
        "ureg": ureg,
    }
    return spectra[case_label]


# =====================================================================
# 1) Instantaneous eigenvalues at t=0 and Floquet exponents
#    Overlay the experiment u_reg values (no logarithmic flux)
# =====================================================================
print("\n" + "=" * 64)
print("u_reg spectrum study")
print(f"  had_flux = {reg_had_flux}")
print(f"  uregs = {list(uregs)}")
if reg_had_flux.lower() == "logreg":
    print(f"  logreg_m = {logreg_m}")
print(f"  disc_nodes = {disc_nodes},  p = {p},  nelem = {nelem},  nen = {nen}")
print(f"  K = {K_floquet}")
print("=" * 64)

first_spec = True
for case_label, case in cases.items():
    if case["had_flux"].lower() == "logarithmic":
        continue
    compute_spectrum(case_label, case, print_progress=first_spec)
    first_spec = False

print("\n" + "=" * 64)
print("Eigenvalues of L at t=0")
print(f"  had_flux = {reg_had_flux}")
print(f"  uregs = {list(uregs)}  (logarithmic flux omitted)")
print("  Close the figure window to continue.")
print("=" * 64)

plt.figure(figsize=(4.5, 4))
ax = plt.gca()
if normalize_eigs:
    plt.xlabel(r"$\Re{(\lambda \Delta x)}$", fontsize=14)
    plt.ylabel(r"$\Im{(\lambda \Delta x)}$", fontsize=14)
else:
    plt.xlabel(r"$\Re{(\lambda)}$", fontsize=14)
    plt.ylabel(r"$\Im{(\lambda)}$", fontsize=14)
for ureg in spec_plot_uregs:
    case_label = ureg_label(ureg)
    spec_data = spectra[case_label]
    eigs = spec_data["eigs"]
    if normalize_eigs:
        eigs = eigs * spec_data["dx"]
    plt.scatter(
        eigs.real, eigs.imag,
        color=ureg_colors[ureg], marker=ureg_markers[ureg], s=20,
        label=labels2[case_label],
    )
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    [handles[i] for i in spec_legend_order],
    [labels[i] for i in spec_legend_order],
    fontsize=14, loc="lower left",
)
plt.tick_params(axis="both", labelsize=14)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_eigs.pdf", format="pdf")
plt.show()

print("\n" + "=" * 64)
print("Floquet exponents")
print(f"  had_flux = {reg_had_flux}")
print(f"  uregs = {list(uregs)}  (logarithmic flux omitted)")
print("  Close the figure window to continue.")
print("=" * 64)

plt.figure(figsize=(4.5, 4))
ax = plt.gca()
plt.ylabel(r"$\Im{(\mu)}$", fontsize=14)
if normalize_eigs:
    plt.xlabel(r"$\Re{(\mu \Delta x)}$", fontsize=14)
else:
    plt.xlabel(r"$\Re{(\mu)}$", fontsize=14)
for ureg in spec_plot_uregs:
    case_label = ureg_label(ureg)
    spec_data = spectra[case_label]
    exponents = spec_data["exponents"]
    x_mu = (exponents.real * spec_data["dx"]
            if normalize_eigs else exponents.real)
    plt.scatter(
        x_mu, exponents.imag,
        color=ureg_colors[ureg], marker=ureg_markers[ureg], s=20,
        label=labels2[case_label],
    )
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    [handles[i] for i in spec_legend_order],
    [labels[i] for i in spec_legend_order],
    fontsize=14, loc="lower left",
)
plt.tick_params(axis="both", labelsize=14)
plt.ylim(-3.3, 3.3)
plt.ticklabel_format(axis="x", style="sci", scilimits=(-4, 4))
ax.xaxis.get_offset_text().set_fontsize(14)
ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_yticklabels(
    [r"$-\pi$", r"$\displaystyle -\frac{\pi}{2}$", r"$0$",
     r"$\displaystyle \frac{\pi}{2}$", r"$\pi$"]
)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_floquet.pdf", format="pdf")
plt.show()

# =====================================================================
# 2) Baseflow + perturbation study
# =====================================================================
pert_noise_shared = None
for case_label, case in cases.items():
    print(f"Perturbation test: {case_label}")
    had_flux = case["had_flux"]
    vol_diss = case["vol_diss"]
    ureg = case.get("ureg", 0.1)
    this_logreg_m = case.get("logreg_m", logreg_m)
    if case_label not in spectra:
        compute_spectrum(case_label, case, print_progress=False)
    # Reuse Floquet from the spectrum study (same discretization).
    diffeq_case, solver = make_solver(
        had_flux, vol_diss,
        nen=nen, p=p, nelem=nelem,
        disc_nodes=disc_nodes,
        ureg=ureg, logreg_m=this_logreg_m,
    )
    max_growth_rate0 = spectra[case_label]["max_growth_rate0"]
    print_reg_params(had_flux, ureg, this_logreg_m)
    print(f"  Max Floquet growth rate = {max_growth_rate0:.6g}")

    if pert_init == "floquet":
        pert0 = spectra[case_label]["pert0"]
        print(f"  Initial perturbation: Floquet mode (nen = {nen})")
    elif pert_init == "noise":
        if pert_noise_shared is None:
            rng = np.random.default_rng(seed=pert_seed)
            noise = 2.0 * rng.random(solver.qshape) - 1.0
            scale = float(np.max(np.abs(noise)))
            pert_noise_shared = ampli * noise / scale if scale > 0.0 else noise
        if pert_noise_shared.shape != solver.qshape:
            raise ValueError(
                "Shared random perturbation shape does not match this "
                "discretization; use the same nen/p/nelem/disc_nodes "
                "for all perturbation cases."
            )
        pert0 = pert_noise_shared
        print(f"  Initial perturbation: random noise (seed = {pert_seed})")
    else:
        raise ValueError("pert_init must be 'floquet' or 'noise'")

    # Baseflow solve
    q0 = diffeq_case.set_q0()
    solver.solve(q0=q0)
    base_qsol = np.copy(solver.q_sol)
    base_time = np.copy(solver.cons_obj[time_idx, :])
    base_error = np.copy(solver.cons_obj[error_idx, :])

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

    # Prediction for ||v||_H growth from the t=0 Floquet rate
    idx_T0 = int(np.argmin(np.abs(base_time - pred_T0)))
    pred0 = np.full(k_min, np.nan, dtype=float)
    pred0[idx_T0:] = v_H[idx_T0] * np.exp(max_growth_rate0 * (base_time[idx_T0:] - base_time[idx_T0]))

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
    }

# ---- Plot 1: H-norm of perturbation v(t)
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
    plt.semilogy(
        data["t"],
        data["v_H"],
        color=colors[case_label],
        label=labels2[case_label],
        linewidth=1.8,
    )

# Capture y-limits based only on actual data
ax = plt.gca()
ymin, ymax = ax.get_ylim()

# Floquet modal growth predictions (dashed, same color) without affecting y autoscaling
if plot_floquet_pred:
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
        label=labels2[case_label],
        linewidth=1.8,
    )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{\mathcal{E}} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
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
            label=labels2[case_label],
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
            label=labels2[case_label],
            linewidth=1.8,
        )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\max \lambda \left(\tfrac{1}{2}(\mathsf{J}+\mathsf{J}^\mathsf{T})\right)$", fontsize=16)
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
            label=labels2[case_label],
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
        label=labels2[case_label],
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
