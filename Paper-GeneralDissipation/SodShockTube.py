#!/usr/bin/env python3
"""Compare entropy-budgeted dissipation for the Sod shock tube."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


# Allow this paper driver to be run directly from its own directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


# Quasi1dEuler's shock_tube defaults already use Sod densities (1, 0.125) and
# rest states u=0. The textbook tube is otherwise dimensional: Omega=[0,10],
# membrane at x=5, (pL, pR)=(1e5, 1e4), and t_final=0.0061. Override only the
# values that differ from the classic problem on Omega=[0,1] until t_f=0.2.
XMIN = 0.0
XMAX = 1.0
XMEMBRANE = 0.5
PL = 1.0
PR = 0.1
T_FINAL = 0.2
P = 4
NELEM = 20
INTERPOLATE = True
INTERPOLATION_POINTS = 50
OP_TYPE = 'lgl'
NEN = 0
DISC_TYPE = 'had'
HAD_FLUX = 'ranocha'
MATRIX_INT_DISSIPATION = True

# Select one plot at a time. The C-SBP plot contains the AD run, while the
# D-SBP plot compares AD plus upwinding against upwinding alone. C-SBP currently
# requires periodic boundaries, so the shock tube defaults to D-SBP.
USE_CSBP = True

# Set to a string or Path to save the figure, for example "SodShockTube_DSBP.pdf".
# Leave as None to display the plot without writing a file.
SAVEFILE = None #"SodShockTube_CSBP.pdf"

# Variable shown in the top panel. One of 'density', 'pressure', or 'mach'.
PLOT_VAR = 'density'

# Entropy-budgeted volume-dissipation parameters. These are kept near the top
# of the driver so later parameter studies only require changing this block.
KAPPA = 1.0
SENSOR_S = int(np.ceil(P/2)+1)
DISTRIBUTION_S = 1
BETA = (P+1)/(2*np.ceil(P/2))
DISTRIBUTION_TYPE = 'cons_sca'

TM_METHOD = 'rk8'
DT = 0.1 * (XMAX - XMIN) / (P * NELEM * 3.0)
TM_RTOL = 1.0e-10
TM_ATOL = 1.0e-10


def apply_sod_overrides(diffeq):
    """Replace the textbook shock-tube data that does not match classic Sod."""
    diffeq.xmin_fix = XMIN
    diffeq.xmax_fix = XMAX
    diffeq.xmembrane = XMEMBRANE
    diffeq.pL = PL
    diffeq.pR = PR
    diffeq.t_final = T_FINAL

    # Dirichlet data were built from the textbook left/right states during
    # construction; recompute them from the overridden Riemann data.
    xx_temp = np.linspace(diffeq.xmin_fix, diffeq.xmax_fix, num=20, endpoint=True)
    q_exa_bdy = diffeq.exact_sol(x=xx_temp, time=0.0, print_warning=False)
    diffeq.qL = q_exa_bdy[:3]
    diffeq.qR = q_exa_bdy[-3:]
    diffeq.s_at_qL = diffeq.fun_s(np.array([diffeq.xmin_fix]))[0]
    diffeq.s_at_qR = diffeq.fun_s(np.array([diffeq.xmax_fix]))[0]
    diffeq.rhoL, diffeq.uL, diffeq.eL, diffeq.PL, diffeq.aL = diffeq.cons2prim(
        diffeq.qL, diffeq.s_at_qL
    )
    diffeq.rhoR, diffeq.uR, diffeq.eR, diffeq.PR, diffeq.aR = diffeq.cons2prim(
        diffeq.qR, diffeq.s_at_qR
    )
    diffeq.EL = diffeq.calcEx(np.reshape(q_exa_bdy[:3], (3, 1))).flatten()
    diffeq.ER = diffeq.calcEx(np.reshape(q_exa_bdy[-3:], (3, 1))).flatten()


def make_solver(solver_class, use_volume_dissipation):
    """Build one of the three solvers with otherwise identical settings."""
    diffeq = Quasi1dEuler(
        [287.0, 1.4],
        q0_type='shock_tube',
        test_case='shock_tube',
        nozzle_shape='constant',
        bc='dirichlet',
        nondimensionalize=False,
    )
    apply_sod_overrides(diffeq)

    if MATRIX_INT_DISSIPATION:
        surface_dissipation = {
        'diss_type': 'ent',
            'jac_type': 'matmat',
            'coeff': 1.0,
            'average': 'none',
            'entropy_fix': False,
            'P_derigs': True,
            'A_derigs': True,
            'maxeig': 'none',
        }
    else:
        surface_dissipation = {
            'diss_type': 'ent',
            'jac_type': 'scamat',
            'coeff': 1.0,
            'average': 'none',
            'entropy_fix': False,
            'P_derigs': True,
            'A_derigs': False,
            'maxeig': 'rusanov',
        }
    if use_volume_dissipation:
        volume_dissipation = {
            'diss_type': 'new',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': 'cons',
            'distribution_type': DISTRIBUTION_TYPE,
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
        disc_type=DISC_TYPE,
        surf_diss=surface_dissipation,
        vol_diss=volume_dissipation,
        had_flux=HAD_FLUX,
        nelem=NELEM,
        nen=NEN,
        disc_nodes=OP_TYPE,
        bc='dirichlet',
        xmin=XMIN,
        xmax=XMAX,
        cons_obj_name=None,
        bool_plot_sol=False,
        print_sol_norm=False,
        sparse=False,
        print_progress=True,
    )
    solver.tm_rtol = TM_RTOL
    solver.tm_atol = TM_ATOL
    solver.tm_print_nothing = False
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


def plot_var_key_and_label():
    """Return the exact-solution key and y-axis label for PLOT_VAR."""
    if PLOT_VAR == 'density':
        return 'rho', r'$\rho$'
    if PLOT_VAR == 'pressure':
        return 'p', r'$p$'
    if PLOT_VAR == 'mach':
        return 'mach', r'$M$'
    raise ValueError(
        "PLOT_VAR must be 'density', 'pressure', or 'mach', "
        f'not {PLOT_VAR!r}'
    )


def interpolate_solution(solver, q_local):
    """Optionally interpolate the nodal polynomial inside each element.

    If INTERPOLATE is False, return the nodal values directly. The plotted
    unknown is recovered from the conservative state with unit area.
    """
    if not INTERPOLATE:
        x_plot = solver.mesh.x_elem
        q_plot = q_local
    else:
        xi = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
        x_plot = solver.sbp.basis.eval_nodal_vec(solver.mesh.x_elem, xi)
        q_plot = solver.sbp.basis.eval_nodal_vec(
            q_local, xi, neq_node=solver.neq_node
        )
    svec = np.ones_like(q_plot[0::solver.neq_node])
    rho, u, _, pressure, a = solver.diffeq.cons2prim(q_plot, svec)
    if PLOT_VAR == 'density':
        var_plot = rho
    elif PLOT_VAR == 'pressure':
        var_plot = pressure
    elif PLOT_VAR == 'mach':
        var_plot = u / a
    else:
        raise ValueError(
            "PLOT_VAR must be 'density', 'pressure', or 'mach', "
            f'not {PLOT_VAR!r}'
        )
    return x_plot.flatten(order='F'), var_plot.flatten(order='F')


def exact_plot_variable(diffeq, x, time):
    """Return the exact Sod field selected by PLOT_VAR on a plotting grid."""
    extra_key, _ = plot_var_key_and_label()
    _, extra = diffeq.exact_sol(
        time=time, x=x, extra_vars=True, print_warning=False
    )
    return extra[extra_key]


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
q_exact = exact_plot_variable(ad_case[0].diffeq, x_exact, T_FINAL)
_, plot_var_label = plot_var_key_and_label()
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
axes[0].set_ylabel(plot_var_label)
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
