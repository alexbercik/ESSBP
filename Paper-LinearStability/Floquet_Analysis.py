"""
Floquet analysis utilities: 
- solver hooks to get exact baseflow
- compute monodromy matrix via midpoint exponential products
- postprocess to get Floquet multipliers / growth rates
- contains a high-level `run_floquet` driver. 
Import this module from application scripts
"""

import os
import time
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from numpy.linalg import eigvals, svd
from scipy.linalg import expm
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
# =========================
# User-provided hooks
# =========================

def baseflow_u(t: float, solver: PdeSolverSbp | PdeSolverCSbp,
               use_exact_sol: bool = True) -> np.ndarray:
    """
    Return u(t) as at time t.
    """
    if use_exact_sol:
        u = solver.diffeq.exact_sol(time=t)
        return u

    if t == 0.0:
        return solver.diffeq.exact_sol(time=t)
    else:
        solver.print_progress = False
        solver.tm_print_nothing = True
        solver.keep_all_ts = False
        solver.t_final = t
        solver.solve(t_final=t)
        u = solver.q_sol
        return u

def operator_L_from_u(u: np.ndarray, solver: PdeSolverSbp | PdeSolverCSbp) -> np.ndarray:
    """
    Return L(u) as an (N,N) array.
    TODO: If needed I could probably relatively easily compute the linearizations directly.
    """
    L = solver.calc_RHS_jac(q=u, exact_dfdq=False, print_nothing=True)
    return L


def _precompute_baseflow_trajectory(T: float, K: int,
                                    solver: PdeSolverSbp | PdeSolverCSbp,
                                    t0: float, q0: np.ndarray | None = None):
    """Solve the nonlinear PDE once and return stored states and times."""
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")

    t_end = float(t0 + T)
    n_fine = int(2 * K)
    if getattr(solver, "dt", None) is not None:
        n_from_solver = int(np.ceil(float(T) / float(solver.dt)))
        n_fine = max(n_fine, n_from_solver)
    if n_fine % 2 == 1:
        n_fine += 1
    dt_fine = float(T) / float(n_fine)
    if dt_fine <= 0.0:
        raise ValueError("Computed RK4 dt is not positive.")

    solver_attrs = [
        "tm_method",
        "dt",
        "t_final",
        "n_ts",
        "dt_to_be_set",
        "keep_all_ts",
        "skip_ts",
        "tm_nframes",
        "print_progress",
        "tm_print_nothing",
        "cons_obj_name",
        "n_cons_obj",
        "bool_calc_cons_obj",
        "q_sol",
        "cons_obj",
    ]
    solver_old = {name: getattr(solver, name) for name in solver_attrs if hasattr(solver, name)}

    diffeq_attrs = ["cons_obj_name", "n_cons_obj"]
    diffeq_old = {name: getattr(solver.diffeq, name) for name in diffeq_attrs if hasattr(solver.diffeq, name)}

    try:
        solver.tm_method = "rk4"
        solver.dt = dt_fine
        solver.t_final = t_end
        solver.keep_all_ts = True
        solver.skip_ts = 0
        solver.tm_nframes = None
        solver.print_progress = False
        solver.tm_print_nothing = False

        solver.cons_obj_name = ("time",)
        solver.n_cons_obj = 1
        solver.bool_calc_cons_obj = True
        solver.diffeq.cons_obj_name = ("time",)
        solver.diffeq.n_cons_obj = 1

        print("Floquet: Precomputing baseflow with RK4 and dt={:.2e}...".format(dt_fine))
        solver.solve(t_final=t_end, q0=q0)
        q_base = np.copy(solver.q_sol)
        t_base = np.copy(solver.cons_obj[0, :])
    finally:
        for name, value in diffeq_old.items():
            setattr(solver.diffeq, name, value)
        for name, value in solver_old.items():
            setattr(solver, name, value)

    if t_base.ndim != 1:
        raise RuntimeError("Expected solver.cons_obj to contain only time in row 0.")
    if q_base.ndim != 3:
        raise RuntimeError("Expected solver.q_sol to contain all timesteps as a 3D array.")

    return t_base, q_base


# =========================
# Core Floquet computation
# =========================

def _floquet_progress_eta_suffix(rem_s: float | None = None, width: int = 22) -> str:
    """Fixed-width ETR string for printProgressBar suffix (avoids \\r line jitter)."""
    if rem_s is None:
        return "Complete. ETR --".ljust(width)
    rem_s = max(0.0, float(rem_s))
    sec = int(round(rem_s))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    if h > 0:
        inner = f"~{h}h{m:02d}m{s:02d}s"
    else:
        inner = f"~{m:02d}m{s:02d}s"
    return f"Complete. ETR {inner}".ljust(width)


def compute_monodromy_midpoint_expm(T: float, K: int,
                                   solver: PdeSolverSbp | PdeSolverCSbp,
                                   t0: float = 0.0,
                                   track_transient: bool = True,
                                   transient_stride: int = 1,
                                   track_max_real_eig: bool = False,
                                   max_real_eig_stride: int | None = None,
                                   print_progress: bool = True,
                                   use_exact_sol: bool = True):
    """
    Compute monodromy matrix M ≈ Π exp(-a Δt L(t_mid)) using midpoint sampling.

    Parameters
    ----------
    T : total length of interval
    t0 : start time of interval
    K : number of time steps on [t0, t0+T], uniform; dt = T/K
    track_transient : if True, store sigma_max(Phi(t)) during the cycle
    transient_stride : store every this many steps to reduce cost
    track_max_real_eig : if True, store max(real(eig(L_mid))) and
        max(real(eig(0.5*(L_mid + L_mid.T)))) at each sampled step
    max_real_eig_stride : sample every this many steps; None => use transient_stride

    Returns
    -------
    M : (N,N) monodromy matrix
    t_sigma_max_Phi : times where transient gains recorded (if track_transient)
    sigma_max_Phi : corresponding sigma_max(Phi(t)) values (if track_transient)
    t_max_real_eig : times (t_mid) where max real eig(L_mid) recorded
                     (if track_max_real_eig)
    max_real_eig : corresponding max(real(eig(L_mid))) values
                    (if track_max_real_eig)
    mu2 : corresponding max(real(eig(0.5*(L_mid + L_mid.T)))) values
          (same times as max_real_eig; if track_max_real_eig)
    """
    if K <= 0:
        raise ValueError("K must be positive.")

    t_base = None
    q_base = None
    floquet_k = int(K)
    substeps_per_step = None
    half_substep = None
    t_start_eff = float(t0)
    if not use_exact_sol:
        t_base, q_base = _precompute_baseflow_trajectory(T=T, K=K, solver=solver, t0=t0)
        n_substeps = t_base.size - 1
        if n_substeps < 2:
            raise RuntimeError("Insufficient precomputed baseflow samples.")

        k_candidates = []
        for k_try in range(1, n_substeps + 1):
            if n_substeps % k_try != 0:
                continue
            n_per_step = n_substeps // k_try
            if n_per_step % 2 == 0:
                k_candidates.append(k_try)
        if len(k_candidates) == 0:
            raise RuntimeError("Could not find a midpoint-aligned K from precomputed baseflow samples.")

        floquet_k = min(k_candidates, key=lambda kk: abs(kk - K))
        if floquet_k != K:
            print("Floquet: Using {", floquet_k, "substeps for monodromy matrix computation instead of {", K, "}.")
        substeps_per_step = n_substeps // floquet_k
        half_substep = substeps_per_step // 2
        t_start_eff = float(t_base[0])

    dt = T / floquet_k

    if max_real_eig_stride is None:
        max_real_eig_stride = transient_stride if track_transient else 1

    nn = solver.nn
    if isinstance(nn, tuple):
        N = int(np.prod(nn)) * solver.neq_node
    else:
        N = nn * solver.neq_node

    Phi = np.eye(N)  # fundamental matrix; Phi(0)=I

    t_sigma_max_Phi = []
    sigma_max_Phi = []
    t_max_real_eig = []
    max_real_eig = []
    mu2 = []

    if track_transient:
        # sigma_max(Phi(0)) = 1
        t_sigma_max_Phi.append(t_start_eff)
        sigma_max_Phi.append(1.0)

    if print_progress:
        print('...Computing monodromy matrix with K={} substeps...'.format(floquet_k))
        from Source.Methods.Analysis import printProgressBar
        printProgressBar(0, floquet_k, prefix='Progress:', suffix=_floquet_progress_eta_suffix(None))
        update_interval = max(1, floquet_k // 100)
        t_prog_start = time.perf_counter()
    for k in range(floquet_k):
        if use_exact_sol:
            t_left = t0 + k * dt
            t_right = t_left + dt
            dt_step = dt
            t_mid = t_left + 0.5 * dt
            u_mid = baseflow_u(t_mid, solver, use_exact_sol)
        else:
            i_left = k * substeps_per_step
            i_mid = i_left + half_substep
            i_right = i_left + substeps_per_step
            t_left = t_base[i_left]
            t_mid = t_base[i_mid]
            t_right = t_base[i_right]
            dt_step = t_right - t_left
            u_mid = q_base[:, :, i_mid]

        # --- User hooks ---
        L_mid = operator_L_from_u(u_mid, solver)
        # ---------------

        if track_max_real_eig and (k % max_real_eig_stride == 0):
            evals = eigvals(L_mid)
            L_sym = 0.5 * (L_mid + L_mid.T)
            evals_sym = eigvals(L_sym)
            t_max_real_eig.append(t_mid)
            max_real_eig.append(float(np.max(evals.real)))
            mu2.append(float(np.max(evals_sym.real)))

        # One-step map for Phi: Phi(t+dt) = exp(dt L_mid) Phi(t)
        E = expm(dt_step * L_mid)
        Phi = E @ Phi

        if track_transient and ((k + 1) % transient_stride == 0):
            # non-normal transient gain proxy: largest singular value of Phi(t)
            # (expensive but ok for N <= 300)
            svals = svd(Phi, compute_uv=False)
            t_sigma_max_Phi.append(t_right)
            sigma_max_Phi.append(float(svals[0]))
        
        if print_progress:
            if (k+1) % update_interval == 0 or k+1 == floquet_k:
                done = k + 1
                elapsed = time.perf_counter() - t_prog_start
                rem = max(0.0, (elapsed / done) * (floquet_k - done))
                printProgressBar(
                    k + 1, floquet_k, prefix='Progress:', suffix=_floquet_progress_eta_suffix(rem))

    M = Phi
    return (
        M,
        np.array(t_sigma_max_Phi),
        np.array(sigma_max_Phi),
        np.array(t_max_real_eig),
        np.array(max_real_eig),
        np.array(mu2),
    )

# =========================
# helpful post-processing
# =========================

def floquet_spectrum_and_gain(M: np.ndarray, T: float,
                              H: np.ndarray | None = None):
    """
    Compute Floquet multipliers (eigs of M) and one-period singular values.
    Also report growth rates per unit time via log|rho|/T.
    """
    rho = eigvals(M)  # complex multipliers
    # sort by magnitude descending
    idx = np.argsort(-np.abs(rho))
    rho = rho[idx]

    # Floquet exponents real parts: (1/T) log |rho|
    # guard against log(0)
    mag = np.maximum(np.abs(rho), 1e-300)
    growth_rates = (1.0 / T) * np.log(mag)
    lam = (1.0 / T) * np.log(rho)

    # one-period non-normal gain: singular values of M
    if H is not None:
        M = np.sqrt(H)[:, None] * M / np.sqrt(H)[None, :]
        svals = svd(M, compute_uv=False)  # descending order
    else:
        svals = np.array([0.0])
    return rho, growth_rates, lam, svals


# =========================
# Main driver
# =========================

def run_floquet(T: float, solver: PdeSolverSbp | PdeSolverCSbp, 
                K: int = 2000, use_H: bool = False,
                track_transient: bool = False,
                transient_stride: int | None = None,
                track_max_real_eig: bool = False,
                max_real_eig_stride: int | None = None,
                use_exact_sol: bool = True):
    """
    Main entry point.
    Increase K until multipliers/singular values converge to desired tolerance.

    Parameters
    ----------
    track_transient : if True, store sigma_max(Phi(t)) during the cycle
    transient_stride : store every this many steps; None => ~200 points over cycle
    track_max_real_eig : if True, store max(real(eig(L_mid))) and
        max(real(eig(0.5*(L_mid + L_mid.T)))) during the cycle
    max_real_eig_stride : sample every this many steps; None => same as transient_stride
    """
    if solver.nn != solver.nen * solver.nelem:
        use_H = False

    if transient_stride is None:
        transient_stride = max(1, K // 200)  # store ~200 points

    (
        M,
        t_sigma_max_Phi,
        sigma_max_Phi,
        t_max_real_eig,
        max_real_eig,
        mu2,
    ) = compute_monodromy_midpoint_expm(
        T=T, K=K,
        solver=solver,
        track_transient=track_transient,
        transient_stride=transient_stride,
        track_max_real_eig=track_max_real_eig,
        max_real_eig_stride=max_real_eig_stride,
        use_exact_sol=use_exact_sol
    )
    if t_sigma_max_Phi.size == 0:
        t_sigma_max_Phi = np.array([0.0])
        sigma_max_Phi = np.array([0.0])
    if t_max_real_eig.size == 0:
        t_max_real_eig = np.array([0.0])
        max_real_eig = np.array([0.0])
        mu2 = np.array([0.0])

    if use_H:
        H = solver.H_phys.flatten('F')
        if solver.neq_node > 1:
            H = np.repeat(H, solver.neq_node, 0)
    else:
        H = None

    rho, rates, lam, svals = floquet_spectrum_and_gain(M, T=T, H=H)

    results = {
        "M": M,
        "rho": rho,
        "growth_rates": rates,     # per unit time
        "exponents": lam,
        "svals_M": svals,          # one-period singular values
        "t_sigma_max_Phi": t_sigma_max_Phi,   # for inside-cycle transient gain
        "sigma_max_Phi": sigma_max_Phi,
        "t_max_real_eig": t_max_real_eig,     # times (t_mid) for max_real_eig_L and mu2_L
        "max_real_eig_L": max_real_eig,       # max(real(eig(L_mid))) over cycle
        "mu2_L": mu2,                         # max real eig of 0.5*(L+L.T) over cycle
    }
    return results
