#!/usr/bin/env python3
"""Compare entropy-budgeted dissipation for a periodic Burgers shock."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


# Allow this paper driver to be run directly from its own directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.DiffEq.Burgers import Burgers
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


# Problem and discretization parameters.
XMIN = 0.0
XMAX = 1.0
T_FINAL = 0.1
P = 4
NELEM = 50
INTERPOLATE = False
INTERPOLATION_POINTS = 50
OP_TYPE = 'lgl'
NEN = 0

# Select one plot at a time. The C-SBP plot contains the AD run, while the
# D-SBP plot compares AD plus upwinding against upwinding alone.
USE_CSBP = True

# Set to a string or Path to save the figure, for example "Burgers_CSBP.pdf".
# Leave as None to display the plot without writing a file.
SAVEFILE = None #"Burgers_CSBP.pdf"

# Entropy-budgeted volume-dissipation parameters. These are kept near the top
# of the driver so later parameter studies only require changing this block.
KAPPA = 1.0
SENSOR_S = int(np.ceil(P/2)+1)
DISTRIBUTION_S = 1
BETA = (P+3)/(2*np.ceil(P/2))

TM_METHOD = 'rk8'
DT = 0.1 * (XMAX - XMIN) / (P * NELEM * 3.0)
TM_RTOL = 1.0e-10
TM_ATOL = 1.0e-10


def make_solver(solver_class, use_volume_dissipation):
    """Build one of the three solvers with otherwise identical settings."""
    diffeq = Burgers(
        q0_type='SinWave_shift2',
        use_split_form=True,
        split_alpha=2.0 / 3.0,
    )
    surface_dissipation = {
        'diss_type': 'es',
        'jac_type': 'scasca',
        'maxeig': 'rusanov',
        'coeff': 1.0,
    }
    if use_volume_dissipation:
        volume_dissipation = {
            'diss_type': 'new',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': 'cons',
            'distribution_type': 'cons_sca',
            'budget_type': 'cheap',
        }
    else:
        volume_dissipation = {'diss_type': 'nd'}

    solver = solver_class(
        diffeq,
        settings=None,
        tm_method=TM_METHOD,
        dt=DT,
        t_final=T_FINAL,
        p=P,
        disc_type='div',
        surf_diss=surface_dissipation,
        vol_diss=volume_dissipation,
        had_flux='ec',
        nelem=NELEM,
        nen=NEN,
        disc_nodes=OP_TYPE,
        bc='periodic',
        xmin=XMIN,
        xmax=XMAX,
        cons_obj_name=None,
        bool_plot_sol=False,
        print_sol_norm=False,
        sparse=False,
        print_progress=False,
    )
    solver.tm_rtol = TM_RTOL
    solver.tm_atol = TM_ATOL
    solver.tm_print_nothing = True
    # Only the final state is needed, so do not allocate an RK8 time history.
    solver.keep_all_ts = False
    return solver


def final_local_solution(solver):
    """Return the final state in the element-local storage layout."""
    q_final = solver.q_sol
    if q_final.ndim == 3:
        q_final = q_final[:, :, -1]
    if isinstance(solver, PdeSolverCSbp):
        return solver.gather(q_final)
    return q_final


def interpolate_solution(solver, q_local):
    """Optionally interpolate the nodal polynomial inside each element.

    If INTERPOLATE is False, return the nodal values directly.
    """
    if not INTERPOLATE:
        return solver.mesh.x_elem.flatten(order='F'), q_local.flatten(order='F')
    xi = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
    x_plot = solver.sbp.basis.eval_nodal_vec(solver.mesh.x_elem, xi)
    q_plot = solver.sbp.basis.eval_nodal_vec(q_local, xi)
    return x_plot.flatten(order='F'), q_plot.flatten(order='F')


def exact_solution(x, time):
    """Return the entropy solution for 2 + sin(2*pi*x).

    Writing u(x,t) = 2 + v(x-2t,t) reduces the problem to the zero-mean
    sine wave. After breaking, symmetry fixes its shock at y=1/2 and the
    entropy solution uses the outer characteristic branch on each side.
    """
    y = np.mod(np.asarray(x) - 2.0 * time, XMAX - XMIN)
    v = np.empty_like(y)
    breaking_time = 1.0 / (2.0 * np.pi)

    for i, yi in enumerate(y):
        if np.isclose(yi, 0.0) or np.isclose(yi, 1.0):
            v[i] = 0.0
            continue
        if time > breaking_time and np.isclose(yi, 0.5):
            # The value at the discontinuity is immaterial; the symmetric
            # Rankine-Hugoniot value makes the plotted jump centered.
            v[i] = 0.0
            continue

        characteristic = lambda x0: x0 + time * np.sin(2.0 * np.pi * x0) - yi
        if time <= breaking_time:
            bracket = (0.0, 1.0)
        elif yi < 0.5:
            bracket = (0.0, 0.5 - 1.0e-12)
        else:
            bracket = (0.5 + 1.0e-12, 1.0)
        x0 = brentq(characteristic, *bracket, xtol=5.0e-15, rtol=1.0e-14)
        v[i] = np.sin(2.0 * np.pi * x0)

    return 2.0 + v


def run_case(label, solver_class, use_volume_dissipation):
    """Run one discretization and collect its plotting diagnostics."""
    print(f'Running {label} ...', flush=True)
    start = time.perf_counter()
    solver = make_solver(solver_class, use_volume_dissipation)
    solver.solve()
    q_local = final_local_solution(solver)

    theta = None
    coefficient = None
    if use_volume_dissipation:
        # Refresh the diagnostics at the returned final state. During RK8 they
        # otherwise correspond to the last internal residual evaluation.
        solver.adiss.dissipation(q_local)
        theta = np.real_if_close(solver.adiss.theta).real.copy()
        coefficient = np.real_if_close(solver.adiss.element_coefficient).real.copy()

    x_plot, q_plot = interpolate_solution(solver, q_local)
    elapsed = time.perf_counter() - start
    print(f'Finished {label} in {elapsed:.2f} s at t={solver.t_final:.12g}.')
    return solver, x_plot, q_plot, theta, coefficient



"""Run and plot either the C-SBP or D-SBP comparison."""
if USE_CSBP:
    ad_case = run_case('C-SBP, AD', PdeSolverCSbp, True)
    solution_curves = (
        (ad_case, 'tab:red', '-', 'C-SBP, AD'),
    )
    diagnostic_label = 'C-SBP, AD'
else:
    ad_case = run_case('D-SBP, AD, upwind', PdeSolverSbp, True)
    upwind_case = run_case('D-SBP, upwind', PdeSolverSbp, False)
    solution_curves = (
        (ad_case, 'tab:red', '-', 'D-SBP, AD, upwind'),
        (upwind_case, 'tab:blue', '-', 'D-SBP, upwind'),
    )
    diagnostic_label = 'D-SBP, AD, upwind'

x_exact = np.linspace(XMIN, XMAX, 2001)
q_exact = exact_solution(x_exact, T_FINAL)
element_edges = np.linspace(XMIN, XMAX, NELEM + 1)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8.0, 8.5),
    sharex=True,
    gridspec_kw={'height_ratios': (2.2, 1.0, 1.0)},
)

axes[0].plot(x_exact, q_exact, color='black', linewidth=2.0, label='Exact')
for case, color, linestyle, label in solution_curves:
    axes[0].plot(case[1], case[2], color=color, linestyle=linestyle,
                    linewidth=1.5, label=label)
axes[0].set_ylabel(r'$u$')
axes[0].set_xlim(XMIN, XMAX)
axes[0].legend(loc='best', frameon=False)
axes[0].grid(alpha=0.2)

axes[1].stairs(ad_case[3], element_edges, color='tab:red', linewidth=1.5,
                label=diagnostic_label)
axes[1].set_ylabel(r'$\theta_k$')
axes[1].legend(loc='best', frameon=False)
axes[1].grid(alpha=0.2)

axes[2].stairs(-ad_case[4], element_edges, color='tab:red', linewidth=1.5,
                label=diagnostic_label)
axes[2].set_xlabel(r'$x$')
axes[2].set_ylabel(r'$\varepsilon_k$')
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[2].grid(alpha=0.2)

fig.tight_layout()
if SAVEFILE is not None:
    fig.savefig(SAVEFILE, bbox_inches='tight')
    print(f'Saved {Path(SAVEFILE)}')
if 'agg' in plt.get_backend().lower():
    plt.close(fig)
else:
    plt.show()
