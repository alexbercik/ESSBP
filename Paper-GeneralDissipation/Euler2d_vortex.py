#!/usr/bin/env python3
"""Compare artificial-dissipation choices for the 2D isentropic vortex.

Edit the parameter block and run this file directly or in an interactive
Jupyter/VS Code window. All case histories and figures remain at module scope.
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


# Allow this paper driver to be run directly from its own directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


# Problem parameters. The vortex crosses the x-period once every 20 time units.
GAMMA = 1.4                   # Ratio of specific heats.
XMIN = (0.0, -5.0)           # Lower-left domain corner.
XMAX = (20.0, 5.0)           # Upper-right domain corner.
T_FINAL = 40.0                # Final time; two x-period crossings.
NELEM = (32, 32)              # Elements in x and y.

# Spatial-discretization hooks.
P = 4                         # Polynomial degree.
NEN = 0                       # Nodes per edge; 0 selects the P default.
DISC_NODES = 'lgl'            # Nodes: 'lgl', 'lg', 'nc', or an SBP family.
DISC_TYPE = 'had'             # Volume form: 'had' or 'div'.
HAD_FLUX = 'ranocha'          # Two-point flux; e.g. 'ranocha' or 'central'.
SETTINGS = {
    'metric_method': 'exact',       # Metrics: 'exact' or a solver method.
    'use_optz_metrics': False,      # Disable optimized discrete metrics.
}

# Change this to 'derigs' to use the matrix-matrix Derigs interface
INTERFACE_DISSIPATION = 'llf'  # 'llf' or 'derigs'.

# New entropy-budgeted volume-dissipation hooks.
KAPPA = 1.0                              # Overall dissipation strength.
SENSOR_S = int(np.ceil(P / 2) + 1)       # Sensor derivative order.
DISTRIBUTION_S = 1                       # Distribution derivative order.
BETA = (P + 1) / (2 * np.ceil(P / 2))   # Sensor exponent.

# Legacy entropy-DCP volume-dissipation hooks. An LGL operator has P + 1
# nodes, so its highest admissible DCP derivative order is P.
OLD_AD_S = P                  # Legacy DCP derivative order.
OLD_AD_COEFF = 0.004          # Legacy dissipation coefficient.
OLD_AD_USE_H = False          # Include the norm in the legacy operator.
OLD_AD_BDY_FIX = False        # Apply the legacy boundary correction.
OLD_AD_AVG_HALF_NODES = False # Average odd-order coefficients at half nodes.

# RK8 uses DT as its first step and subsequently adapts it. Diagnostics are
# retained every CONS_OBJ_SKIP + 1 accepted steps and at the final time.
TM_METHOD = 'rk8'             # Time marcher; common options: 'rk4', 'rk8'.
CFL = 0.1                     # Sets the initial adaptive RK8 step.
MIN_ELEMENT_WIDTH = min(
    (XMAX[0] - XMIN[0])/NELEM[0],
    (XMAX[1] - XMIN[1])/NELEM[1],
)
DT = CFL * MIN_ELEMENT_WIDTH / (P * 3.0)  # Initial RK8 step.
TM_RTOL = 1.0e-8              # Adaptive relative tolerance.
TM_ATOL = 1.0e-8              # Adaptive absolute tolerance.
CONS_OBJ_SKIP = 9             # Accepted steps skipped between diagnostics.

# Set this to None to display the figures without saving them. Otherwise the
# driver writes <SAVE_PREFIX>_{conservation,entropy,error}.pdf.
SAVE_PREFIX = None             # None: show; otherwise output filename prefix.


SURFACE_DISSIPATIONS = {
    'llf': {
        'diss_type': 'ent',
        'jac_type': 'scamat',
        'coeff': 1.0,
        'average': 'none',
        'entropy_fix': False,
        'P_derigs': True,
        'A_derigs': False,
        'maxeig': 'rusanov',
    },
    'derigs': {
        'diss_type': 'ent',
        'jac_type': 'matmat',
        'coeff': 1.0,
        'average': 'none',
        'entropy_fix': False,
        'P_derigs': True,
        'A_derigs': True,
        'maxeig': 'none',
    },
}

if INTERFACE_DISSIPATION not in SURFACE_DISSIPATIONS:
    raise ValueError(
        "INTERFACE_DISSIPATION must be either 'llf' or 'derigs'."
    )
INTERFACE_LABEL = {
    'llf': 'LLF',
    'derigs': 'Derigs',
}[INTERFACE_DISSIPATION]


# The two EC references are included only on the conservation and error plots.
CASES = (
    {
        'label': f'D-SBP, {INTERFACE_LABEL}',
        'solver_class': PdeSolverSbp,
        'volume': 'none',
        'interface': True,
        'color': 'black',
        'linestyle': '-',
        'plot_entropy': True,
    },
    {
        'label': 'C-SBP, AD',
        'solver_class': PdeSolverCSbp,
        'volume': 'new',
        'interface': False,
        'color': 'red',
        'linestyle': '-',
        'plot_entropy': True,
    },
    {
        'label': f'D-SBP, AD, {INTERFACE_LABEL}',
        'solver_class': PdeSolverSbp,
        'volume': 'new',
        'interface': True,
        'color': 'blue',
        'linestyle': '-',
        'plot_entropy': True,
    },
    {
        'label': f'D-SBP, old AD, {INTERFACE_LABEL}',
        'solver_class': PdeSolverSbp,
        'volume': 'old',
        'interface': True,
        'color': 'green',
        'linestyle': '-',
        'plot_entropy': True,
    },
    {
        'label': 'C-SBP, EC',
        'solver_class': PdeSolverCSbp,
        'volume': 'none',
        'interface': False,
        'color': 'darkorange',
        'linestyle': '--',
        'plot_entropy': False,
    },
    {
        'label': 'D-SBP, EC',
        'solver_class': PdeSolverSbp,
        'volume': 'none',
        'interface': False,
        'color': '0.45',
        'linestyle': '--',
        'plot_entropy': False,
    },
)


def make_solver(case):
    """Build one comparison case with otherwise identical settings."""
    diffeq = Euler(
        [287.0, GAMMA],
        q0_type='vortex_zelalem',
        test_case='vortex_zelalem',
        bc='periodic',
        nondimensionalize=False,
    )

    if case['interface']:
        surface_dissipation = dict(
            SURFACE_DISSIPATIONS[INTERFACE_DISSIPATION]
        )
    else:
        # With the Hadamard volume form this selects its entropy-conservative
        # two-point flux at element interfaces without added dissipation.
        surface_dissipation = {'diss_type': 'ec'}

    if case['volume'] == 'new':
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
    elif case['volume'] == 'old':
        volume_dissipation = {
            'diss_type': 'entdcp',
            'jac_type': 'scamat',
            's': OLD_AD_S,
            'coeff': OLD_AD_COEFF,
            'bdy_fix': OLD_AD_BDY_FIX,
            'use_H': OLD_AD_USE_H,
            'entropy_fix': False,
            'avg_half_nodes': OLD_AD_AVG_HALF_NODES,
        }
    else:
        volume_dissipation = {'diss_type': 'nd'}

    solver = case['solver_class'](
        diffeq,
        settings=dict(SETTINGS),
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
        disc_nodes=DISC_NODES,
        bc='periodic',
        xmin=XMIN,
        xmax=XMAX,
        cons_obj_name=('time', 'conservation', 'entropy', 'error'),
        bool_plot_sol=False,
        print_sol_norm=False,
        sparse=True,
        print_progress=True,
    )
    solver.tm_rtol = TM_RTOL
    solver.tm_atol = TM_ATOL
    solver.tm_print_nothing = False
    solver.keep_all_ts = False
    solver.skip_ts = CONS_OBJ_SKIP
    return solver


def run_case(case):
    """Run a case and return its conservation-object time histories."""
    print(f"Running {case['label']} ...", flush=True)
    start = time.perf_counter()
    solver = make_solver(case)
    solver.solve()
    elapsed = time.perf_counter() - start
    print(
        f"Finished {case['label']} in {elapsed:.2f} s "
        f"at t={solver.t_final:.12g}.",
        flush=True,
    )

    histories = {}
    for index, name in enumerate(solver.cons_obj_name):
        histories[name.lower()] = np.real_if_close(
            solver.cons_obj[index]
        ).real
    return histories


def plot_history(results, objective, ylabel, plot_change, include_ec,
                 negative=False):
    """Put every requested run for one objective on a single plot."""
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for case in CASES:
        if not include_ec and not case['plot_entropy']:
            continue

        history = results[case['label']]
        values = history[objective]
        if plot_change:
            values = values - values[0]
        if negative:
            values = -values
        axis.semilogy(
            history['time'],
            np.abs(values),
            color=case['color'],
            linestyle=case['linestyle'],
            linewidth=1.5,
            label=case['label'],
        )

    axis.set_xlabel('time')
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, T_FINAL)
    if negative:
        axis.invert_yaxis()
        axis.yaxis.set_major_formatter(
            lambda x, _pos: rf'$-10^{{{int(np.round(np.log10(x)))}}}$'
            if x > 0 else ''
        )
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
    axis.grid(alpha=0.2, which='both')
    axis.legend(loc='best', frameon=False)
    fig.tight_layout()

    if SAVE_PREFIX is not None:
        savefile = Path(f'{SAVE_PREFIX}_{objective}.pdf')
        fig.savefig(savefile, bbox_inches='tight')
        print(f'Saved {savefile}')
    return fig



case_results = {}
for case in CASES:
    case_results[case['label']] = run_case(case)

figures = (
    plot_history(
        case_results,
        objective='conservation',
        ylabel='change in conservation',
        plot_change=True,
        include_ec=True,
    ),
    plot_history(
        case_results,
        objective='entropy',
        ylabel='entropy dissipation',
        plot_change=True,
        include_ec=False,
        negative=True,
    ),
    plot_history(
        case_results,
        objective='error',
        ylabel='error',
        plot_change=False,
        include_ec=True,
    ),
)

if SAVE_PREFIX is not None or 'agg' in plt.get_backend().lower():
    for figure in figures:
        plt.close(figure)
else:
    plt.show()
