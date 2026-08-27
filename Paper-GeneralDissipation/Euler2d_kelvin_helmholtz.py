#!/usr/bin/env python3
"""Run one Kelvin--Helmholtz case and plot selected solution diagnostics.

Edit the parameter block, then run this file directly or in an interactive
Jupyter/VS Code window. The module-level ``run`` dictionary retains the solver,
snapshots, archive data, and figure for later inspection.
"""

import json
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
from Source.Disc.MakeSbpOp import MakeSbpOp
import Source.Methods.Functions as fn
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


# Kelvin--Helmholtz problem parameters.
GAS_CONSTANT = 287.0          # Specific gas constant.
GAMMA = 1.4                   # Ratio of specific heats.
TEST_CASE = 'kelvin-helmholtz' # Euler initial-condition/test-case name.
BOUNDARY_CONDITION = 'periodic' # Boundary type; this case uses 'periodic'.
NONDIMENSIONALIZE = False     # Keep the repository's dimensional variables.
XMIN = (-1.0, -1.0)          # Lower-left domain corner.
XMAX = (1.0, 1.0)            # Upper-right domain corner.
T_FINAL = 15.0                # Final solution time.

# This driver runs exactly one spatial discretization. OPERATOR may be
# 'C-SBP' or 'D-SBP'. C-SBP ignores INTERFACE_DISSIPATION because its
# conforming global operator has no discontinuous interior interfaces.
NELEM = (64, 64)              # Elements in x and y.
P = 4                         # Polynomial degree.
OPERATOR = 'C-SBP'            # Spatial operator: 'C-SBP' or 'D-SBP'.
DISC_NODES = 'lgl'            # Nodes: 'lgl', 'lg', 'nc', or an SBP family.
NEN = 0                       # Nodes per edge; 0 selects the P default.
DISC_TYPE = 'had'             # Volume form: 'had' or 'div'.
HAD_FLUX = 'ranocha'          # Two-point flux; e.g. 'ranocha' or 'central'.
SETTINGS = {
    'metric_method': 'exact',       # Metrics: 'exact' or a solver method.
    'use_optz_metrics': False,      # Disable optimized discrete metrics.
}

# Available volume choices are 'new', 'old', and 'none'. Available D-SBP
# interface choices are 'llf', 'derigs', and 'none'.
VOLUME_DISSIPATION = 'new'     # 'new', 'new_directional', 'old', or 'none'.
INTERFACE_DISSIPATION = 'llf'  # D-SBP: 'llf', 'derigs', or 'none'.

# New entropy-budgeted volume-dissipation parameters.
KAPPA = 1.0                            # Overall dissipation strength.
SENSOR_S = int(np.ceil(P / 2) + 1)     # Sensor derivative order.
DISTRIBUTION_S = 3                     # Distribution derivative order.
BETA = (P + 1) / (2 * np.ceil(P / 2)) # Sensor exponent.
SENSOR_TYPE = 'cons'                   # 'cons' or 'none'.
DISTRIBUTION_TYPE = 'cons_sca'         # 'cons_sca' or 'cons_mat'.
BUDGET_TYPE = 'cheap'                  # 'cheap'; 'entdcp' pairs with entdcp.

# Legacy entropy-DCP volume-dissipation parameters.
OLD_AD_S = P                  # Legacy DCP derivative order.
OLD_AD_COEFF = 0.004          # Legacy dissipation coefficient.
OLD_AD_USE_H = False          # Include the norm in the legacy operator.
OLD_AD_BDY_FIX = False        # Apply the legacy boundary correction.
OLD_AD_AVG_HALF_NODES = False # Average odd-order coefficients at half nodes.

# RK8 uses DT only for its first step and then adapts its step size. The 150
# uniform time intervals store the initial state plus t=0.1, ..., 15.0.
TM_METHOD = 'rk8'             # Time marcher; common options: 'rk4', 'rk8'.
CFL = 0.1                     # Sets the initial adaptive RK8 step.
MIN_ELEMENT_WIDTH = min(
    (XMAX[0] - XMIN[0]) / NELEM[0],
    (XMAX[1] - XMIN[1]) / NELEM[1],
)
DT = CFL * MIN_ELEMENT_WIDTH / (P * 3.0)
TM_RTOL = 1.0e-7              # Adaptive relative tolerance.
TM_ATOL = 1.0e-7              # Adaptive absolute tolerance.
TM_NFRAMES = 150              # Uniform time intervals retained by RK8.
PLOT_TIMES = (3.7, 6.7)       # Requested diagnostic snapshot times.

# Plot controls. None selects the range from the data in that panel.
PLOT_LEVELS = 100              # Filled density-contour levels.
# When a reference-element polynomial basis exists (LGL, LG, NC), evaluate
# it on INTERPOLATION_POINTS nodes per element edge before contouring.
INTERPOLATE = True             # Interpolate each element before contouring.
INTERPOLATION_POINTS = 10      # Plot points per element edge.
DENSITY_CMAP = 'viridis'       # Matplotlib colormap for density.
DENSITY_VMIN = None            # Density lower color limit; None uses data.
DENSITY_VMAX = None            # Density upper color limit; None uses data.
THETA_CMAP = 'magma'           # Matplotlib colormap for the sensor.
THETA_VMIN = None              # Sensor lower color limit; None uses data.
THETA_VMAX = None              # Sensor upper color limit; None uses data.
ENTROPY_BUDGET_CMAP = 'magma'  # Colormap for the entropy budget.
ENTROPY_BUDGET_VMIN = None     # Budget lower color limit; None uses data.
ENTROPY_BUDGET_VMAX = None     # Budget upper color limit; None uses data.

# Set SAVEFILE to a base filename, such as 'kelvin_helmholtz', to save the
# combined figure as kelvin_helmholtz.pdf. Any supplied suffix is replaced by
# .pdf. With None, no figure is written. The requested solution archive is
# always written; it shares the figure stem when SAVEFILE is provided.
SAVEFILE = 'KH_csbp_lgl_p4_64_new_s3_p0' # Output stem; None skips the figure.
DEFAULT_DATAFILE = 'Euler2d_kelvin_helmholtz_data.npz' # Used if no stem.


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
    # With the Hadamard form this retains the entropy-conservative two-point
    # flux at element interfaces without adding dissipation.
    'none': {'diss_type': 'ec'},
}


def make_solver():
    """Build the single configured Kelvin--Helmholtz solver."""
    operator = OPERATOR.lower().replace('-', '')
    if operator == 'csbp':
        solver_class = PdeSolverCSbp
        surface_dissipation = dict(SURFACE_DISSIPATIONS['none'])
    elif operator == 'dsbp':
        solver_class = PdeSolverSbp
        interface = INTERFACE_DISSIPATION.lower()
        if interface not in SURFACE_DISSIPATIONS:
            raise ValueError(
                "INTERFACE_DISSIPATION must be 'llf', 'derigs', or 'none'."
            )
        surface_dissipation = dict(SURFACE_DISSIPATIONS[interface])
    else:
        raise ValueError("OPERATOR must be either 'C-SBP' or 'D-SBP'.")

    volume = VOLUME_DISSIPATION.lower()
    if volume == 'new':
        volume_dissipation = {
            'diss_type': 'new',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': SENSOR_TYPE,
            'distribution_type': DISTRIBUTION_TYPE,
            'budget_type': BUDGET_TYPE,
        }
    elif volume == 'new_directional':
        volume_dissipation = {
            'diss_type': 'new_directional',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': SENSOR_TYPE,
            'distribution_type': DISTRIBUTION_TYPE,
            'budget_type': BUDGET_TYPE,
        }
    elif volume == 'old':
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
    elif volume == 'none':
        volume_dissipation = {'diss_type': 'nd'}
    else:
        raise ValueError("VOLUME_DISSIPATION must be 'new', 'old', or 'none'.")

    diffeq = Euler(
        [GAS_CONSTANT, GAMMA],
        q0_type=TEST_CASE,
        test_case=TEST_CASE,
        bc=BOUNDARY_CONDITION,
        nondimensionalize=NONDIMENSIONALIZE,
    )
    solver = solver_class(
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
        bc=BOUNDARY_CONDITION,
        xmin=XMIN,
        xmax=XMAX,
        # RK8 needs time in this list when a time history is retained.
        cons_obj_name=('time',),
        bool_plot_sol=False,
        print_sol_norm=False,
        sparse=True,
        print_progress=True,
    )
    solver.tm_rtol = TM_RTOL
    solver.tm_atol = TM_ATOL
    solver.tm_print_nothing = False
    solver.keep_all_ts = True
    solver.tm_nframes = TM_NFRAMES
    return solver, surface_dissipation, volume_dissipation


def local_state(solver, state):
    """Return a state in the element-local layout used for plots and output."""
    if isinstance(solver, PdeSolverCSbp):
        if np.asarray(state).shape == solver.qshape_local:
            return state
        return solver.gather(state)
    return state


def polynomial_basis(p, disc_nodes, nen):
    """Return the reference-element basis, or None if one is not available."""
    return MakeSbpOp(int(p), disc_nodes, int(nen), print_progress=False).basis


def density_plot_data(state, xy_elem, nen, nelem, basis=None):
    """Reshape an element-local conservative state onto the 2D plotting grid.

    If INTERPOLATE is True and a polynomial basis is available, the nodal
    density and coordinates are evaluated on a uniform reference grid with
    INTERPOLATION_POINTS nodes per element edge. Otherwise the raw nodes are
    used.
    """
    density = np.real_if_close(state[0::4, :]).real
    if INTERPOLATE and basis is not None:
        xi = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
        xy_elem = basis.eval_nodal_vec(xy_elem, xi, dim=2)
        density = basis.eval_nodal_vec(density, xi, dim=2)
        nen = INTERPOLATION_POINTS
    x_grid = fn.reshape_to_meshgrid_2D(
        xy_elem[:, 0, :], nen, *nelem
    )
    y_grid = fn.reshape_to_meshgrid_2D(
        xy_elem[:, 1, :], nen, *nelem
    )
    density_grid = fn.reshape_to_meshgrid_2D(
        density, nen, *nelem
    )
    return x_grid, y_grid, density_grid


def parameter_record(surface_dissipation, volume_dissipation):
    """Collect the complete set of user-editable driver parameters."""
    if OPERATOR.lower().replace('-', '') == 'csbp':
        effective_interface = 'none'
    else:
        effective_interface = INTERFACE_DISSIPATION
    return {
        'gas_constant': GAS_CONSTANT,
        'gamma': GAMMA,
        'test_case': TEST_CASE,
        'q0_type': TEST_CASE,
        'boundary_condition': BOUNDARY_CONDITION,
        'nondimensionalize': NONDIMENSIONALIZE,
        'xmin': XMIN,
        'xmax': XMAX,
        't_final': T_FINAL,
        'nelem': NELEM,
        'p': P,
        'operator': OPERATOR,
        'disc_nodes': DISC_NODES,
        'nen': NEN,
        'disc_type': DISC_TYPE,
        'had_flux': HAD_FLUX,
        'settings': SETTINGS,
        'volume_dissipation_name': VOLUME_DISSIPATION,
        'interface_dissipation_name': INTERFACE_DISSIPATION,
        'effective_interface_dissipation_name': effective_interface,
        'volume_dissipation': volume_dissipation,
        'surface_dissipation': surface_dissipation,
        'kappa': KAPPA,
        'sensor_s': SENSOR_S,
        'distribution_s': DISTRIBUTION_S,
        'beta': BETA,
        'sensor_type': SENSOR_TYPE,
        'distribution_type': DISTRIBUTION_TYPE,
        'budget_type': BUDGET_TYPE,
        'old_ad_s': OLD_AD_S,
        'old_ad_coeff': OLD_AD_COEFF,
        'old_ad_use_h': OLD_AD_USE_H,
        'old_ad_bdy_fix': OLD_AD_BDY_FIX,
        'old_ad_avg_half_nodes': OLD_AD_AVG_HALF_NODES,
        'tm_method': TM_METHOD,
        'cfl': CFL,
        'dt': DT,
        'tm_rtol': TM_RTOL,
        'tm_atol': TM_ATOL,
        'tm_nframes': TM_NFRAMES,
        'plot_times': PLOT_TIMES,
        'plot_levels': PLOT_LEVELS,
        'interpolate': INTERPOLATE,
        'interpolation_points': INTERPOLATION_POINTS,
        'density_cmap': DENSITY_CMAP,
        'density_vmin': DENSITY_VMIN,
        'density_vmax': DENSITY_VMAX,
        'theta_cmap': THETA_CMAP,
        'theta_vmin': THETA_VMIN,
        'theta_vmax': THETA_VMAX,
        'entropy_budget_cmap': ENTROPY_BUDGET_CMAP,
        'entropy_budget_vmin': ENTROPY_BUDGET_VMIN,
        'entropy_budget_vmax': ENTROPY_BUDGET_VMAX,
        'savefile': None if SAVEFILE is None else str(SAVEFILE),
        'default_datafile': str(DEFAULT_DATAFILE),
    }


def plot_snapshots(snapshots, xy_elem, nelem, xmin, xmax,
                   plot_new_dissipation, basis=None):
    """Create the density and optional new-dissipation snapshot panels."""
    panel_count = 3 if plot_new_dissipation else 1
    fig, axes = plt.subplots(
        len(snapshots),
        panel_count,
        figsize=(5.0 * panel_count, 4.1 * len(snapshots)),
        squeeze=False,
    )
    nen = int(round(np.sqrt(xy_elem.shape[0])))
    x_edges = np.linspace(xmin[0], xmax[0], nelem[0] + 1)
    y_edges = np.linspace(xmin[1], xmax[1], nelem[1] + 1)

    for row, (requested_time, snapshot) in enumerate(snapshots.items()):
        actual_time = snapshot['time']
        x_grid, y_grid, density = density_plot_data(
            snapshot['state'], xy_elem, nen, nelem, basis=basis
        )
        density_plot = axes[row, 0].contourf(
            x_grid,
            y_grid,
            density,
            levels=PLOT_LEVELS,
            cmap=DENSITY_CMAP,
            vmin=DENSITY_VMIN,
            vmax=DENSITY_VMAX,
        )
        # PDF/SVG backends anti-alias each filled polygon separately, which
        # leaves thin white seams between contour levels. Colour the edges
        # with the face colour and overlap them slightly to close the gaps.
        density_plot.set_edgecolor('face')
        density_plot.set_linewidth(0.4)
        fig.colorbar(density_plot, ax=axes[row, 0], label=r'$\rho$')
        axes[row, 0].set_title(
            rf'Density: requested $t={requested_time:g}$, '
            rf'stored $t={actual_time:.6g}$'
        )

        if plot_new_dissipation:
            theta_plot = axes[row, 1].pcolormesh(
                x_edges,
                y_edges,
                snapshot['theta'].T,
                shading='flat',
                cmap=THETA_CMAP,
                vmin=THETA_VMIN,
                vmax=THETA_VMAX,
            )
            fig.colorbar(theta_plot, ax=axes[row, 1], label=r'$\theta_k$')
            axes[row, 1].set_title(r'Sensor $\theta_k$')

            budget_plot = axes[row, 2].pcolormesh(
                x_edges,
                y_edges,
                snapshot['entropy_budget'].T,
                shading='flat',
                cmap=ENTROPY_BUDGET_CMAP,
                vmin=ENTROPY_BUDGET_VMIN,
                vmax=ENTROPY_BUDGET_VMAX,
            )
            fig.colorbar(
                budget_plot,
                ax=axes[row, 2],
                label=r'$\varepsilon_k$',
            )
            axes[row, 2].set_title(r'Total entropy budget $\varepsilon_k$')

        for axis in axes[row]:
            axis.set_xlabel(r'$x$')
            axis.set_ylabel(r'$y$')
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlim(xmin[0], xmax[0])
            axis.set_ylim(xmin[1], xmax[1])

    fig.tight_layout()
    return fig


def display_figure(fig, block=None):
    """Show *fig* once without hanging an interactive session.

    Interactive backends auto-draw a figure when it is created *and* when
    ``plt.show()`` is called, which is why ``load_and_plot()`` appeared to
    plot twice. Figures should be built with interactive mode off, then
    shown here. ``block=False`` returns immediately to the REPL; scripts
    keep the default blocking show so the window stays open.
    """
    if plt.get_backend().lower() == 'agg':
        plt.close(fig)
        return
    if block is None:
        block = not plt.isinteractive()
    plt.show(block=block)


def build_snapshot_figure(*args, **kwargs):
    """Build the snapshot figure without an interactive auto-show.

    Leaves matplotlib interactive mode off so the caller can show the
    figure once. The second return value is whether interactive mode
    should be restored afterwards.
    """
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        return plot_snapshots(*args, **kwargs), was_interactive
    except Exception:
        if was_interactive:
            plt.ion()
        raise


def load_run(datafile, print_parameters=False):
    """Load a saved Kelvin--Helmholtz archive without plotting.

    Failed runs still contain ``q_final`` at ``last_solution_time``. Requested
    plot times that were not reached are omitted from ``snapshots``. The full
    RK8 frame history is not stored; only those snapshots plus the last finite
    state are in the file.

    Example:
        data = load_run('KH_csbp_lgl_p4_64_new_s3_p0.npz')
        q = data['q_final']
        t = data['last_solution_time']
    """
    datafile = Path(datafile)
    with np.load(datafile) as data:
        if 'parameters_json' not in data.files:
            raise ValueError(f'{datafile} does not contain parameters_json.')

        parameters = json.loads(data['parameters_json'].item())
        if print_parameters:
            print(f'Parameters used in {datafile}:')
            print(json.dumps(parameters, indent=2, sort_keys=True))

        requested_times = tuple(
            float(value) for value in np.asarray(data['requested_times']).ravel()
        )
        snapshots = {}
        for requested_time in requested_times:
            key = str(requested_time).replace('.', 'p')
            state_key = f'q_t{key}'
            if state_key not in data.files or data[state_key].size == 0:
                continue
            snapshots[requested_time] = {
                'time': float(data[f'time_t{key}']),
                'state': data[state_key].copy(),
            }
            theta_key = f'theta_t{key}'
            budget_key = f'entropy_budget_t{key}'
            if theta_key in data.files and budget_key in data.files:
                snapshots[requested_time]['theta'] = data[theta_key].copy()
                snapshots[requested_time]['entropy_budget'] = data[
                    budget_key
                ].copy()

        run = {
            'parameters': parameters,
            'xy_elem': data['xy_elem'].copy(),
            'stored_times': data['stored_times'].copy(),
            'requested_times': requested_times,
            'run_end_time': float(np.asarray(data['run_end_time']).item()),
            'last_solution_time': float(
                np.asarray(data['last_solution_time']).item()
            ),
            'completed': bool(np.asarray(data['completed']).item()),
            'wall_time_seconds': float(
                np.asarray(data['wall_time_seconds']).item()
            ),
            'q_final': data['q_final'].copy(),
            'snapshots': snapshots,
        }
    return run


def load_and_plot(datafile, savefile=None):
    """Print a saved run's parameters and recreate its snapshot figure.

    For example, from an interactive window:
        load_and_plot('Euler2d_kelvin_helmholtz_data.npz')

    If none of the requested plot times were stored, the last saved state
    ``q_final`` is plotted instead. Use ``load_run`` to inspect the archive
    without plotting.

    Set savefile to a base name to also write the figure as a PDF.
    """
    run = load_run(datafile, print_parameters=True)
    parameters = run['parameters']
    snapshots = dict(run['snapshots'])
    if not snapshots:
        print(
            'No requested snapshots were stored; plotting the last saved '
            f"state at t={run['last_solution_time']:.6g}."
        )
        snapshots[run['last_solution_time']] = {
            'time': run['last_solution_time'],
            'state': run['q_final'],
        }

    requested_new_dissipation = (
        parameters.get('volume_dissipation_name', '').lower() == 'new'
    )
    diagnostics_available = all(
        'theta' in snapshot and 'entropy_budget' in snapshot
        for snapshot in snapshots.values()
    )
    plot_new_dissipation = requested_new_dissipation and diagnostics_available
    if requested_new_dissipation and not diagnostics_available:
        print(
            'This archive has no theta/entropy-budget diagnostics for the '
            'states being plotted; showing density only.'
        )

    nelem = tuple(int(value) for value in parameters['nelem'])
    xmin = tuple(float(value) for value in parameters['xmin'])
    xmax = tuple(float(value) for value in parameters['xmax'])
    basis = None
    if INTERPOLATE:
        basis = polynomial_basis(
            parameters['p'],
            parameters['disc_nodes'],
            parameters.get('nen', 0),
        )
    fig, was_interactive = build_snapshot_figure(
        snapshots,
        run['xy_elem'],
        nelem,
        xmin,
        xmax,
        plot_new_dissipation,
        basis=basis,
    )

    try:
        if savefile is not None:
            figure_file = Path(savefile).with_suffix('.pdf')
            figure_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(figure_file, format='pdf', bbox_inches='tight')
            print(f'Saved figure to {figure_file}.')

        display_figure(fig, block=not was_interactive)
        return fig
    finally:
        if was_interactive:
            plt.ion()


def main():
    """Run the case and return all data useful for interactive inspection."""
    solver, surface_dissipation, volume_dissipation = make_solver()
    if OPERATOR.lower().replace('-', '') == 'csbp':
        interface_description = 'ignored by C-SBP'
    else:
        interface_description = INTERFACE_DISSIPATION
    print(
        f'Running Kelvin--Helmholtz with {OPERATOR}, p={P}, '
        f'nelem={NELEM}, volume={VOLUME_DISSIPATION}, '
        f'interface={interface_description} ...',
        flush=True,
    )
    wall_start = time.perf_counter()
    try:
        solver.solve()
    except Exception as error:
        wall_time = time.perf_counter() - wall_start
        print(
            'Kelvin--Helmholtz crashed before the time marcher returned '
            f'after {wall_time:.2f} s of wall time: {error}',
            flush=True,
        )
        raise
    wall_time = time.perf_counter() - wall_start

    # The first conservation objective is the actual RK8 time associated with
    # each retained state. Remove a nonfinite last state if it caused a crash.
    stored_times = np.real_if_close(solver.cons_obj[0]).real
    history = solver.q_sol
    finite_states = np.all(np.isfinite(history), axis=(0, 1))
    finite_indices = np.flatnonzero(np.isfinite(stored_times) & finite_states)
    # A failure before RK8's first accepted step can leave one zero-filled
    # frame at t=0 behind the real initial state. Retain only increasing times.
    if finite_indices.size > 0:
        increasing_time = np.concatenate((
            np.array([True]),
            np.diff(stored_times[finite_indices]) > 0.0,
        ))
        valid_indices = finite_indices[increasing_time]
    else:
        valid_indices = finite_indices
    if valid_indices.size == 0:
        raise RuntimeError('The time marcher returned no finite solution states.')

    last_index = valid_indices[-1]
    last_state_time = float(stored_times[last_index])
    run_end_time = float(solver.t_final)
    last_local_state = local_state(solver, history[:, :, last_index])
    last_state_is_physical = not solver.diffeq.check_positivity(last_local_state)
    completed = (
        abs(run_end_time - T_FINAL) <= 1.0e-10
        and abs(last_state_time - T_FINAL) <= 1.0e-10
        and last_state_is_physical
    )
    if completed:
        print(
            f'Kelvin--Helmholtz completed at t={run_end_time:.12g} '
            f'in {wall_time:.2f} s of wall time.',
            flush=True,
        )
    else:
        print(
            f'Kelvin--Helmholtz crashed/stopped at t={run_end_time:.12g}; '
            f'the last finite stored state is at t={last_state_time:.12g}. '
            f'Wall time was {wall_time:.2f} s.',
            flush=True,
        )

    # Select the retained state nearest each requested time, but only after the
    # simulation has actually advanced that far.
    snapshots = {}
    for requested_time in PLOT_TIMES:
        if run_end_time + 1.0e-10 < requested_time:
            print(f'Skipping t={requested_time:g}; the run ended too early.')
            continue
        index = valid_indices[
            np.argmin(np.abs(stored_times[valid_indices] - requested_time))
        ]
        snapshots[requested_time] = {
            'index': int(index),
            'time': float(stored_times[index]),
            'state': local_state(solver, history[:, :, index]).copy(),
        }

    using_new_dissipation = VOLUME_DISSIPATION.lower() == 'new'
    if using_new_dissipation:
        for snapshot in snapshots.values():
            # Refresh diagnostics at each saved state. epsilon_k is the
            # positive total entropy budget used in the paper drivers.
            solver.adiss.dissipation(snapshot['state'])
            snapshot['theta'] = np.real_if_close(
                solver.adiss.theta
            ).real.reshape(NELEM).copy()
            snapshot['entropy_budget'] = -np.real_if_close(
                solver.adiss.element_coefficient
            ).real.reshape(NELEM).copy()

    # Save the requested states and the final returned state in a portable,
    # element-local layout. JSON keeps the parameter metadata pickle-free.
    if SAVEFILE is None:
        figure_file = None
        datafile = Path(DEFAULT_DATAFILE)
    else:
        output_base = Path(SAVEFILE)
        figure_file = output_base.with_suffix('.pdf')
        datafile = output_base.with_suffix('.npz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        'parameters_json': np.array(json.dumps(
            parameter_record(surface_dissipation, volume_dissipation),
            indent=2,
        )),
        'xy_elem': solver.mesh.xy_elem,
        'stored_times': stored_times,
        'requested_times': np.asarray(PLOT_TIMES),
        'run_end_time': np.array(run_end_time),
        'last_solution_time': np.array(last_state_time),
        'completed': np.array(completed),
        'wall_time_seconds': np.array(wall_time),
        'q_final': last_local_state,
    }
    for requested_time in PLOT_TIMES:
        key = str(requested_time).replace('.', 'p')
        if requested_time in snapshots:
            archive[f'time_t{key}'] = np.array(snapshots[requested_time]['time'])
            archive[f'q_t{key}'] = snapshots[requested_time]['state']
            if using_new_dissipation:
                archive[f'theta_t{key}'] = snapshots[requested_time]['theta']
                archive[f'entropy_budget_t{key}'] = snapshots[
                    requested_time
                ]['entropy_budget']
        else:
            archive[f'time_t{key}'] = np.array(np.nan)
            archive[f'q_t{key}'] = np.empty((0,))
    np.savez_compressed(datafile, **archive)
    print(f'Saved solution data to {datafile}.')

    fig = None
    if snapshots:
        fig, was_interactive = build_snapshot_figure(
            snapshots,
            solver.mesh.xy_elem,
            NELEM,
            XMIN,
            XMAX,
            using_new_dissipation,
            basis=solver.sbp.basis if INTERPOLATE else None,
        )
        try:
            if figure_file is not None:
                figure_file.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(figure_file, format='pdf', bbox_inches='tight')
                print(f'Saved figure to {figure_file}.')

            display_figure(fig, block=not was_interactive)
        finally:
            if was_interactive:
                plt.ion()

    return {
        'solver': solver,
        'surface_dissipation': surface_dissipation,
        'volume_dissipation': volume_dissipation,
        'snapshots': snapshots,
        'archive': archive,
        'datafile': datafile,
        'figure': fig,
        'completed': completed,
    }


# Run at module scope and retain everything needed for notebook-style analysis.
run = main()
