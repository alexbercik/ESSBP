#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Compressible Euler Equations Density Wave local growth analysis.
Reproduces figures from section 4.2 of the paper.

Set load_from_npz=True (and datafile) to skip all solves and only replot
from a previously saved .npz archive.

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
import gc
from sys import path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
from Floquet_Analysis import run_floquet


n_nested_folder = 1
folder_path, _ = os.path.split(__file__)
for _ in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)
path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp

tm_method = "rk4"
dt = 0.0001 # 0.0001 for paper, 0.0005 for testing
tf = 500.0

xmin = -1.0
xmax = 1.0
bc = "periodic"

disc_type = "had"
disc_nodes = "circulant"
p = 8
nelem = 1
nen = 39
had_flux = "ranocha"

# If not None, save figures as PDF with suffixes:
#   {savefile}_vH.pdf, {savefile}_eH.pdf, ...
savefile = None #'1dEuler_densitywave_near0_circulant_n39_ranocha'

# Compressed data archive for fast replotting. Set load_from_npz=True to skip
# all solves/Jacobians and only regenerate plots from this file.
datafile = savefile + ".npz" if savefile is not None else "1dEuler_densitywave_near0.npz"
load_from_npz = False

perturbation = 2 # if 0, use Floquet mode, If 1, use random noise on density. If 2, use random noise on all variables. 
                # if 3, use singular vector of Monodromy matrix. If 4, use eigenvector of Jacobian.
plot_floquet = False


surf_type = {"diss_type": "nd", "fluxvec": "lf", "coeff": 1.0}
test_case = "density_wave" 
cons_obj_name = ("error", "var_error", "entropy", "time")

# Volume dissipation choices
s = int(p / 2) + 1
coeff = 0.625 / 5 ** s

settings = {
    "warp_factor": 0.0,
    "warp_type": "none",
    "jac_method": "exact",
}

cases = {
    "No diss.": {
        "diss_type": "nd",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "none",
        "s": s,
        "coeff": coeff,
        "avg_half_nodes": True,
        "enforce_positivity": False,
    },
    "No diss. + pos.": {
        "diss_type": "nd",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "none",
        "s": s,
        "coeff": coeff,
        "avg_half_nodes": True,
        "enforce_positivity": True,
    },
    "Cons. diss.": {
        "diss_type": "dcp",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "matrix",
        "s": s,
        "coeff": coeff,
        "avg_half_nodes": True,
        "enforce_positivity": False,
    },
    "Ent. diss.": {
        "diss_type": "entdcp",
        "use_H": False,
        "bdy_fix": False,
        "jac_type": "matrixmatrix",
        "s": s,
        "coeff": coeff,
        "avg_half_nodes": True,
        "enforce_positivity": False,
    },
}

colors = {
    "No diss.": "tab:blue",
    "No diss. + pos.": "tab:green",
    "Cons. diss.": "tab:orange",
    "Ent. diss.": "tab:red",
}
# Legend / iteration order (Ent. diss. last in legend). zorder puts Ent. diss. behind.
legend_order = list(colors.keys())
zorders = {lab: 2 + i for i, lab in enumerate(legend_order)}
zorders["Ent. diss."] = 1


def iter_cases(results_dict):
    """Iterate cases in legend order (only labels present in results)."""
    for case_label in legend_order:
        if case_label in results_dict:
            yield case_label, results_dict[case_label]


def save_results_compressed(results: dict, data_filename: str) -> str:
    """
    Save the full results dictionary to a compressed .npz file.
    Returns the path of the written file.
    """
    if not data_filename.endswith(".npz"):
        data_filename = f"{data_filename}.npz"
    # Flat arrays (no pickle) for robust round-trips across numpy versions.
    payload = {}
    case_labels = list(results.keys())
    payload["case_labels"] = np.array(case_labels, dtype=object)
    for case_label, data in results.items():
        prefix = case_label + "/"
        for key, val in data.items():
            if val is None:
                continue
            payload[prefix + key] = np.asarray(val)
    np.savez_compressed(data_filename, **payload)
    print(f"Saved results to {data_filename}")
    return data_filename


def load_results_compressed(data_filename: str) -> dict:
    """
    Load results dictionary from a compressed .npz file created by
    save_results_compressed.
    """
    if not data_filename.endswith(".npz"):
        data_filename = f"{data_filename}.npz"
    print(f"Loading results from {data_filename}")
    with np.load(data_filename, allow_pickle=True) as data:
        # Backward compatible with older object-array saves
        if "results" in data.files:
            return dict(data["results"][0])

        case_labels = [str(lab) for lab in data["case_labels"]]
        results = {}
        for case_label in case_labels:
            prefix = case_label + "/"
            case_data = {}
            for key in data.files:
                if key.startswith(prefix):
                    case_data[key[len(prefix):]] = data[key]
            # Scalar tmax_idx was saved as a 0-d array
            if "tmax_idx" in case_data:
                case_data["tmax_idx"] = int(np.asarray(case_data["tmax_idx"]))
            if "pred0" not in case_data:
                case_data["pred0"] = None
            results[case_label] = case_data
        return results


# Collect results per case (either from disk or by running the simulations)
if load_from_npz:
    results = load_results_compressed(datafile)
else:
    results = {}

    # indices for the conservation objective components
    time_idx = [name.lower() for name in cons_obj_name].index("time")
    error_idx = [name.lower() for name in cons_obj_name].index("error")
    rho_error_idx = [name.lower() for name in cons_obj_name].index("var_error")
    entropy_idx = [name.lower() for name in cons_obj_name].index("entropy")

    print("Running with flux:", had_flux)
    for case_label, vol_diss in cases.items():
        print(f"Running case: {case_label}")

        diffeq = Quasi1dEuler([287,1.4], 'density_wave', test_case, 'constant', 'periodic', nondimensionalize=False)
        solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                        tm_method, dt, tf,                    # Time marching
                        p, 'had',             # Discretization
                        surf_type, vol_diss, had_flux,
                        nelem, nen, disc_nodes,
                        'periodic', -1., 1.,         # Domain
                        cons_obj_name,              # Other
                        print_progress=True)

        #solver.tm_atol = 1e-8
        #solver.tm_rtol = 1e-8
        if vol_diss["enforce_positivity"]:
            solver.clip_positivity = True
            solver.clip_positivity_floor = 5e-4
            solver.clip_positivity_cut = 5e-2

        if (plot_floquet and perturbation != 3) or (perturbation == 0):
            T = 20.0
            K_floquet = 5000
            solver_temp = PdeSolverSbp(diffeq, settings,                     # Diffeq
                tm_method, dt, tf,                    # Time marching
                p, 'had',             # Discretization
                {"diss_type":"nd"}, {"diss_type":"nd"}, 'ranocha',
                nelem, nen, disc_nodes,
                'periodic', -1., 1.,         # Domain
                cons_obj_name,              # Other
                print_progress=True)
            floq = run_floquet(T=T, solver=solver_temp, K=K_floquet, use_H=True)
            max_growth_rate0 = float(np.max(floq["growth_rates"]))
            eigvals_M, eigvecs_M = np.linalg.eig(floq["M"])
            idx_max = int(np.argmax(eigvals_M.real))
            del eigvals_M, floq, solver_temp
            gc.collect()
        ampli = 1e-5
        rng = np.random.default_rng(seed=0)
        if perturbation == 0:
            pert0 = ampli * eigvecs_M[:, idx_max].real.reshape(solver.qshape, order="F")
        elif perturbation == 1:
            pert0 = ampli * 2*(np.random.rand(*solver.qshape).reshape(solver.qshape, order="F")-1)
            pert0[1::3] = 0.0
            pert0[2::3] = 0.0
        elif perturbation == 2:
            pert0 = ampli * 2*(np.random.rand(*solver.qshape).reshape(solver.qshape, order="F")-1)
        elif perturbation == 3:
            T = 20.0
            K_floquet = int(1000/20*T)
            floq = run_floquet(T=T, solver=solver, K=K_floquet, use_H=True, use_exact_sol=False)
            _,S,Vh = np.linalg.svd(floq["M"])
            pert0 = ampli * Vh[0, :].reshape(solver.qshape, order="F")
            max_growth_rate0 = np.log(S[0]) / T
            del floq, S, Vh
            gc.collect()
        elif perturbation == 4:
            solver_temp = PdeSolverSbp(diffeq, settings,                     # Diffeq
                        tm_method, dt, tf,                    # Time marching
                        p, 'had',             # Discretization
                        {"diss_type":"nd"}, {"diss_type":"nd"}, 'ranocha',
                        nelem, nen, disc_nodes,
                        'periodic', -1., 1.,         # Domain
                        cons_obj_name,              # Other
                        print_progress=True)
            RHS_jac = solver_temp.calc_RHS_jac(q=diffeq.set_q0())
            eigvals_RHS_jac, eigvecs_RHS_jac = np.linalg.eig(RHS_jac)
            idx_max = int(np.argmax(eigvals_RHS_jac.real))
            pert0 = ampli * eigvecs_RHS_jac[:, idx_max].real.reshape(solver.qshape, order="F")
            del RHS_jac, eigvals_RHS_jac, eigvecs_RHS_jac, solver_temp
            gc.collect()
        else:
            raise ValueError(f"Invalid perturbation type: {perturbation}")

        # Baseflow solve
        q0 = diffeq.set_q0()
        solver.solve(q0=q0)
        base_qsol = np.copy(solver.q_sol)
        solver.q_sol = None
        base_time = np.copy(solver.cons_obj[time_idx, :])
        base_error = np.copy(solver.cons_obj[error_idx, :])
        base_rho_error = np.copy(solver.cons_obj[rho_error_idx, :])
        base_entropy = np.copy(solver.cons_obj[entropy_idx, :])
        solver.cons_obj = None
        gc.collect()

        # Perturbed solve (u + v, but we only *analyze* v against smooth-baseflow J)
        solver.solve(q0=q0 + pert0)
        k_min = min(base_qsol.shape[2], solver.q_sol.shape[2])

        #base_time = base_time[:k_min]
        #base_error = base_error[:k_min]
        #base_rho_error = base_rho_error[:k_min]
        #base_qsol = base_qsol[:, :, :k_min]
        pert_qsol = solver.q_sol[:, :, :k_min]
        solver.q_sol = None

        # Track v(t) and derived perturbation quantities in H-norm
        # Vectorized computation for speed; drop intermediates as soon as done.
        pert = pert_qsol - base_qsol[:, :, :k_min]
        v_H = np.sqrt(solver.energy(pert))
        vrho_H = np.sqrt(solver.energy(pert[0::3, :, :], neq=1))
        vrhou_H = np.sqrt(solver.energy(pert[1::3, :, :], neq=1))
        ve_H = np.sqrt(solver.energy(pert[2::3, :, :], neq=1))

        u_base = base_qsol[1::3, :, :k_min] / base_qsol[0::3, :, :k_min]
        u_pert = pert_qsol[1::3, :, :k_min] / pert_qsol[0::3, :, :k_min]
        p_base = diffeq.calc_p(base_qsol[:, :, :k_min])
        p_pert = diffeq.calc_p(pert_qsol)
        vu = u_pert - u_base
        vp = p_pert - p_base

        vu_H = np.sqrt(solver.energy(vu, neq=1))
        vp_H = np.sqrt(solver.energy(vp, neq=1))

        # TODO: should I use the exact baseflow density or the numerical baseflow density?
        # TODO: should I use the exact baseflow pressure or the numerical baseflow pressure?
        vE = 0.5 * (base_qsol[::3, :, :k_min] * vu**2 + vp**2 / diffeq.p0 / diffeq.g)
        vE_H = solver.conservation(vE, neq=1)

        rho = base_qsol[::3, :, :k_min]
        u = base_qsol[1::3, :, :k_min] / rho
        pre = (0.4) * (base_qsol[2::3, :, :k_min] - 0.5 * (rho * u**2))
        min_rho = np.min(rho, axis=(0, 1))
        min_p = np.min(pre, axis=(0, 1))

        del u_base, u_pert, p_base, p_pert, vu, vp, vE, rho, u, pre
        gc.collect()


        # Jacobian-based quantities over time:
        #   J(t)     = calc_RHS_jac(u(t))
        #   J_sym(t) = (J(t) + J(t)^T)/2
        #   lambda_max_J(t)    = max Re(eig(J(t)))          (full J)
        #   lambda_max_J_sym(t)= max eig(J_sym(t))          (symmetric part)
        #   g(t) = (v(t)^T J_sym(t) v(t)) / (v(t)^T v(t))   (Rayleigh quotient)
        jac_stride = 500
        idxs_to_jac = list(range(0, base_qsol.shape[2], jac_stride))
        last_idx = base_qsol.shape[2] - 1
        if last_idx not in idxs_to_jac:
            idxs_to_jac.append(last_idx)

        lambda_max_J = np.full(base_qsol.shape[2], np.nan, dtype=float)
        lambda_max_J_sym = np.full(base_qsol.shape[2], np.nan, dtype=float)
        g = np.full(k_min, np.nan, dtype=float)

        for i in idxs_to_jac:
            J = solver.calc_RHS_jac(q=base_qsol[:, :, i], exact_dfdq=False)
            J_sym = 0.5 * (J + J.T)

            eigvals_J = np.linalg.eigvals(J)
            lambda_max_J[i] = float(np.max(eigvals_J.real))

            # Symmetric eigenvalue solver for numerical stability
            eigvals_J_sym = np.linalg.eigvalsh(J_sym.real)
            lambda_max_J_sym[i] = float(np.max(eigvals_J_sym))

            if i < k_min:
                v_vec = pert[:, :, i].flatten("F")
                v2 = float(np.vdot(v_vec, v_vec).real)
                if v2 >= 1e-30:
                    g[i] = float(np.vdot(v_vec, J_sym @ v_vec).real / v2)
            del J, J_sym, eigvals_J, eigvals_J_sym
        gc.collect()
        del pert
        gc.collect()

        if plot_floquet:
            # Prediction for ||v||_H growth at anchor time t=0
            pred0 = v_H[0] * np.exp(max_growth_rate0 * base_time)
        else:
            pred0 = None

        results[case_label] = {
            "t": base_time,
            "tmax_idx": k_min,
            "v_H": v_H,
            "vrho_H": vrho_H,
            "vrhou_H": vrhou_H,
            "vu_H": vu_H,
            "ve_H": ve_H,
            "vp_H": vp_H,
            "vE_H": vE_H,
            "e_H": base_error,
            "erho_H": base_rho_error,
            "lambda_max_J": lambda_max_J,
            "lambda_max_J_sym": lambda_max_J_sym,
            "g": g,
            "min_rho": min_rho,
            "min_p": min_p,
            "pred0": pred0,
            "ent": base_entropy
        }
        del pert_qsol, base_qsol, base_time, base_error, base_rho_error, base_entropy, q0, pert0
        gc.collect()

    save_results_compressed(results, datafile)

# ---- Plot 1: H-norm of perturbation v(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["v_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )

# Capture y-limits based only on actual data
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)

# Floquet modal growth predictions (dashed, same color) without affecting y autoscaling
for case_label, data in iter_cases(results):
    t = data["t"]

    if plot_floquet:
        y_pred0 = data["pred0"]
        mask0 = np.isfinite(y_pred0)
        plt.semilogy(
            t[mask0],
            y_pred0[mask0],
            color=colors[case_label],
            linestyle="--",
            linewidth=1.2,
            zorder=zorders[case_label],
        )


ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vH.pdf", format="pdf")

# ---- Plot 1.1: H-norm of perturbation v_rho(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["vrho_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
# Capture y-limits based only on actual data
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v}_{\rho} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vrhoH.pdf", format="pdf")

# ---- Plot 1.2: H-norm of perturbation v_rhou(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["vrhou_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v}_{\rho u} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vrhouH.pdf", format="pdf")

# ---- Plot 1.3: H-norm of perturbation v_u(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["vu_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v}_{u} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vuH.pdf", format="pdf")

# ---- Plot 1.4: H-norm of perturbation v_e(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["ve_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v}_{e} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_veH.pdf", format="pdf")

# ---- Plot 1.5: H-norm of perturbation v_p(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"][:data["tmax_idx"]],
        data["vp_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymax = min(ymax, 20)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{v}_{p} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_vpH.pdf", format="pdf")

# ---- Plot 1.6: H-norm of perturbation reduced energy v_E(t)
# plt.figure(figsize=(5, 4))
# for case_label, data in iter_cases(results):
#     plt.semilogy(
#         data["t"][:data["tmax_idx"]],
#         data["vE_H"],
#         color=colors[case_label],
#         label=case_label,
#         linewidth=1.8,
#         zorder=zorders[case_label],
#     )
# ax = plt.gca()
# ymin, ymax = ax.get_ylim()
# ymax = min(ymax, 20)
# ax.set_ylim(ymin, ymax)
# plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
# plt.ylabel(r"$\mathcal{E}_h(\boldsymbol{v})$", fontsize=16, labelpad=22, rotation=0)
# plt.tick_params(axis="both", labelsize=13)
# plt.legend(fontsize=12, loc="best")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# if savefile is not None:
#     plt.savefig(savefile + "_vEH.pdf", format="pdf")

# ---- Plot 2: H-norm of baseflow error e(t)
plt.figure(figsize=(5, 4))
ymin = 1
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"],
        data["e_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
    ymin = min(ymin, np.min(data["e_H"][5:]))

ax = plt.gca()
_, ymax = ax.get_ylim()
ymax = min(ymax, 10)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{\mathcal{E}} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_eH.pdf", format="pdf")

# ---- Plot 2.1: H-norm of baseflow error e_\rho(t)
plt.figure(figsize=(5, 4))
ymin = 1
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"],
        data["erho_H"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
    ymin = min(ymin, np.min(data["erho_H"][5:]))

ax = plt.gca()
_, ymax = ax.get_ylim()
ymax = min(ymax, 0.5)
ax.set_ylim(ymin, ymax)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\| \boldsymbol{\mathcal{E}}_{\rho} \|_{\mathsf{H}}$", fontsize=16, labelpad=22, rotation=0)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_erhoH.pdf", format="pdf")

# ---- Plot 2.2: H-sum of baseflow entropy S(t)
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
    plt.semilogy(
        data["t"],
        -data["ent"],  # take magnitude (since values are negative)
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )

# Invert the y-axis so larger negative values are lower
plt.gca().invert_yaxis()

plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\mathcal{S}_h$", fontsize=16, labelpad=22, rotation=0)

# Optional: relabel ticks to show negative values instead of positive magnitudes
from matplotlib.ticker import FuncFormatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{-y:.2e}"))

plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

if savefile is not None:
    plt.savefig(savefile + "_ent.pdf", format="pdf")

# ---- Plot 3: lambda_max(J) over time
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
        y = data["lambda_max_J"]
        x = data["t"]
        mask = np.isfinite(y)
        plt.semilogy(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
            zorder=zorders[case_label],
        )
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\max \Re(\lambda(\mathsf{J}))$", fontsize=16)
#plt.ylim(ymin=1e-3)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_lambdaJ.pdf", format="pdf")

# ---- Plot 4: lambda_max(J_sym) over time
plt.figure(figsize=(5, 4))
for case_label, data in iter_cases(results):
        y = data["lambda_max_J_sym"]
        x = data["t"]
        mask = np.isfinite(y)
        plt.semilogy(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
            zorder=zorders[case_label],
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
for case_label, data in iter_cases(results):
        y = data["g"]
        x = data["t"][:data["tmax_idx"]]
        mask = np.isfinite(y)
        plt.plot(
            x[mask],
            y[mask],
            color=colors[case_label],
            label=case_label,
            linewidth=1.8,
            zorder=zorders[case_label],
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
for case_label, data in iter_cases(results):
    plt.plot(
        data["t"][:data["tmax_idx"]],
        data["min_rho"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
        zorder=zorders[case_label],
    )
plt.ylim(1e-3, 0.8)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\min \boldsymbol{\rho}$", fontsize=16)
plt.yscale("symlog", linthresh=1e-5)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_minu.pdf", format="pdf")

plt.show()

# ---- Plot 7: min(p(x,t)) over space
plt.figure(figsize=(5, 4))
for case_label, data in results.items():
    plt.plot(
        data["t"][:data["tmax_idx"]],
        data["min_p"],
        color=colors[case_label],
        label=case_label,
        linewidth=1.8,
    )
plt.ylim(1e-3, 30)
plt.xlabel(r"$t$", fontsize=16, labelpad=-5)
plt.ylabel(r"$\min \boldsymbol{p}$", fontsize=16)
plt.yscale("symlog", linthresh=1e-5)
plt.tick_params(axis="both", labelsize=13)
plt.legend(fontsize=12, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
if savefile is not None:
    plt.savefig(savefile + "_minp.pdf", format="pdf")

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

