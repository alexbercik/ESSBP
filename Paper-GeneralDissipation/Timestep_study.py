#!/usr/bin/env python3
"""Study RK4 timestep limits for 1D linear-convection dissipation choices.

Edit the parameter block, then run this file directly or in an interactive
Jupyter/VS Code window. The study results, snapshot solutions, and figure are
created at module scope for later inspection.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np


# Allow this paper driver to be run directly from its own directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


# Problem and discretization parameters.
WAVE_SPEED = 1.0              # Constant convection speed.
XMIN = 0.0                    # Left boundary of the periodic interval.
XMAX = 1.0                    # Right boundary of the periodic interval.
T_FINAL = 4.0                 # Final solution time.
P = 4                         # Polynomial degree.
NELEM = 40                    # Number of uniform elements.
NEN = 0                       # Nodes per element; 0 selects the P default.
OP_TYPE = 'lgl'               # Nodes: 'lgl', 'lg', 'nc', or an SBP family.
DISC_TYPE = 'div'             # Volume form: 'div' or 'had'.
HAD_FLUX = 'central'          # Two-point flux used when DISC_TYPE='had'.

# 'test', 'gaussian', 'square', or 'morlet'.
INITIAL_CONDITION = 'test'    # 'test', 'gaussian', 'square', or 'morlet'.

# Select C-SBP or D-SBP. Interface dissipation only acts on D-SBP element
# faces; periodic C-SBP has no interior SATs.
USE_CSBP = False              # True: C-SBP; False: D-SBP.
USE_INTERFACE_DISSIPATION = True # Add the D-SBP interface LLF term.

# Entropy-budgeted volume-dissipation parameters. The three comparison
# flavours share these values; only the sensor, distribution, and budget
# selectors change. The nonlinear flavour uses the cons/cheap choices below.
# The standard (linear) flavour recovers entropy-DCP dissipation by setting
# sensor_type='none' with distribution_type=budget_type='entdcp'.
KAPPA = 1.0                            # Overall dissipation strength.
SENSOR_S = int(np.ceil(P / 2) + 1)     # Sensor derivative order.
DISTRIBUTION_S = 1                     # Distribution derivative order.
BETA = (P + 1) / (2 * np.ceil(P / 2)) # Sensor exponent.
SENSOR_TYPE = 'cons'                   # 'cons' or 'none'.
DISTRIBUTION_TYPE = 'cons_sca'         # 'cons_sca', 'cons_mat', or 'entdcp'.
BUDGET_TYPE = 'cheap'                  # 'cheap' or paired 'entdcp'.

# CFL is a * dt * p / h, with h the element width. The study first evaluates
# N_INITIAL geometrically spaced CFLs from CFL_MIN to CFL_MAX (endpoints
# included). If that scan brackets an instability, it spends up to N_BISECT
# extra runs bisecting the last-stable / first-unstable step counts.
CFL_MIN = 0.1                 # Smallest CFL in the initial scan.
CFL_MAX = 2.0                 # Largest CFL in the initial scan.
N_INITIAL = 8  # geometric CFL samples in [CFL_MIN, CFL_MAX]
N_BISECT = 6   # extra RK4 runs used to refine the stability cutoff

# After the CFL study, plot u(x) at this fraction of the most restrictive
# last-stable CFL (the smallest cutoff among the flavours).
SOLUTION_CFL_FRACTION = 0.9   # Fraction of the tightest stable CFL to plot.
INTERPOLATE = True            # Plot interpolated polynomials instead of nodes.
INTERPOLATION_POINTS = 50     # Plot points per element when interpolating.

# Errors larger than this are treated as blow-up and omitted from the plot.
MAX_STABLE_ERROR = 10.0       # Errors above this mark a run unstable.

# Set to a string or Path to save the figure. Leave as None to display it.
SAVEFILE = None                # None: show; otherwise save to this path.

Q0_TYPES = {
    'test': 'dissipation_test',
    'gaussian': 'gausswave',
    'square': 'squarewave',
    'morlet': 'morlet_wavelet',
}

VOLUME_CASES = (
    {
        'key': 'none',
        'label': 'No dissipation',
        'color': 'tab:blue',
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'key': 'nonlinear',
        'label': 'Nonlinear dissipation',
        'color': 'tab:red',
        'linestyle': '-',
        'marker': 's',
    }
)


def q0_type():
    """Return the DiffEq initial-condition name for INITIAL_CONDITION."""
    if INITIAL_CONDITION not in Q0_TYPES:
        raise ValueError(
            "INITIAL_CONDITION must be 'gaussian', 'square', or 'morlet', "
            f'not {INITIAL_CONDITION!r}.'
        )
    return Q0_TYPES[INITIAL_CONDITION]


def volume_dissipation(flavour):
    """Return the volume-dissipation dictionary for one comparison flavour."""
    if flavour == 'none':
        return {'diss_type': 'nd'}
    if flavour == 'nonlinear':
        return {
            'diss_type': 'new',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': SENSOR_TYPE,
            'distribution_type': DISTRIBUTION_TYPE,
            'budget_type': BUDGET_TYPE,
        }
    if flavour == 'standard':
        return {
            'diss_type': 'new',
            'kappa': KAPPA,
            'beta': BETA,
            'sensor_s': SENSOR_S,
            'distribution_s': DISTRIBUTION_S,
            'sensor_type': 'none',
            'distribution_type': 'entdcp',
            'budget_type': 'entdcp',
        }
    raise ValueError(f'Unknown volume-dissipation flavour {flavour!r}.')


def surface_dissipation():
    """Return the interface SAT settings, or a non-dissipative SAT."""
    if USE_CSBP or not USE_INTERFACE_DISSIPATION:
        return {'diss_type': 'nd'}
    return {
        'diss_type': 'lf',
        'jac_type': 'sca',
        'coeff': 1.0,
    }


def element_width():
    """Return the uniform element width."""
    return (XMAX - XMIN) / NELEM


def n_ts_from_cfl(cfl):
    """Return the number of RK4 steps that realizes a given CFL."""
    dt_requested = cfl * element_width() / (P * WAVE_SPEED)
    return max(1, int(np.round(T_FINAL / dt_requested)))


def dt_from_n_ts(n_ts):
    """Return the timestep that lands exactly on T_FINAL."""
    return T_FINAL / int(n_ts)


def cfl_from_n_ts(n_ts):
    """Return the CFL number implied by n_ts steps to T_FINAL."""
    return WAVE_SPEED * dt_from_n_ts(n_ts) * P / element_width()


def unique_n_ts(values, n_lo=None, n_hi=None):
    """Round, sort, and optionally clip a list of step counts."""
    ns = sorted({max(1, int(round(n))) for n in values})
    if n_lo is not None:
        ns = [n for n in ns if n >= n_lo]
    if n_hi is not None:
        ns = [n for n in ns if n <= n_hi]
    return ns


def geometric_n_ts(n_lo, n_hi, n_points):
    """Return unique integer step counts spaced geometrically in dt."""
    n_lo = max(1, int(n_lo))
    n_hi = max(n_lo, int(n_hi))
    if n_hi == n_lo:
        return [n_lo]
    samples = np.geomspace(n_lo, n_hi, num=max(2, int(n_points)))
    ns = unique_n_ts(samples, n_lo=n_lo, n_hi=n_hi)
    if n_lo not in ns:
        ns.insert(0, n_lo)
    if n_hi not in ns:
        ns.append(n_hi)
    return ns


def make_solver(flavour):
    """Build a linear-convection solver for one dissipation flavour."""
    diffeq = LinearConv(WAVE_SPEED, q0_type(), had_flux=HAD_FLUX)
    solver_class = PdeSolverCSbp if USE_CSBP else PdeSolverSbp
    solver = solver_class(
        diffeq,
        settings=None,
        tm_method='rk4',
        dt=dt_from_n_ts(n_ts_from_cfl(CFL_MIN)),
        t_final=T_FINAL,
        p=P,
        disc_type=DISC_TYPE,
        surf_diss=surface_dissipation(),
        vol_diss=volume_dissipation(flavour),
        had_flux=HAD_FLUX,
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
    solver.tm_print_nothing = True
    solver.keep_all_ts = False
    return solver


def final_error(solver):
    """Return the H-norm solution error at the stored final time, or NaN."""
    q_final = solver.q_sol
    if q_final is None:
        return np.nan
    if q_final.ndim == 3:
        q_final = q_final[:, :, -1]
    if not np.all(np.isfinite(q_final)):
        return np.nan
    if abs(solver.t_final - T_FINAL) > 1.0e-12:
        return np.nan
    error = solver.calc_error(q=q_final, tf=T_FINAL)
    if error is None or not np.isfinite(error):
        return np.nan
    error = float(np.real_if_close(error).real)
    if error > MAX_STABLE_ERROR:
        return np.nan
    return error


def is_stable(error):
    """Return True if a completed run stayed within the error cap."""
    return np.isfinite(error)


def final_local_solution(solver):
    """Return the final state in the element-local storage layout."""
    q_final = solver.q_sol
    if q_final is None:
        return None
    if q_final.ndim == 3:
        q_final = q_final[:, :, -1]
    if isinstance(solver, PdeSolverCSbp):
        return solver.gather(q_final)
    return q_final


def interpolate_solution(solver, q_local):
    """Optionally interpolate the nodal polynomial inside each element."""
    if not INTERPOLATE:
        return solver.mesh.x_elem.flatten(order='F'), q_local.flatten(order='F')
    xi = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
    x_plot = solver.sbp.basis.eval_nodal_vec(solver.mesh.x_elem, xi)
    q_plot = solver.sbp.basis.eval_nodal_vec(q_local, xi)
    return x_plot.flatten(order='F'), q_plot.flatten(order='F')


def evaluate(solver, n_ts, label, cache):
    """Run one snapped timestep, using the cache when possible."""
    n_ts = max(1, int(n_ts))
    if n_ts in cache:
        return cache[n_ts]

    dt = dt_from_n_ts(n_ts)
    cfl = cfl_from_n_ts(n_ts)
    solver.t_final = T_FINAL
    with redirect_stdout(StringIO()):
        solver.set_timestep(dt)
    n_ts = solver.n_ts
    dt = solver.dt
    cfl = cfl_from_n_ts(n_ts)
    if n_ts in cache:
        return cache[n_ts]

    print(
        f'  {label}: CFL={cfl:g}, dt={dt:.6g}, n_ts={n_ts}',
        flush=True,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        try:
            solver.solve()
            error = final_error(solver)
        except Exception as exc:
            print(f'  {label}: failed at dt={dt:.6g}: {exc}')
            error = np.nan
    solver.t_final = T_FINAL
    elapsed = time.perf_counter() - start
    if is_stable(error):
        print(f'    error={error:.6e} in {elapsed:.2f} s')
    else:
        print(f'    unstable or failed in {elapsed:.2f} s')
    cache[n_ts] = (dt, error, cfl)
    return cache[n_ts]


def cached_stable(cache):
    """Return (n_ts, dt, error, cfl) for every stable cached run."""
    rows = []
    for n_ts, (dt, error, cfl) in cache.items():
        if is_stable(error):
            rows.append((n_ts, dt, error, cfl))
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows


def initial_step_counts():
    """Return unique n_ts values realizing N_INITIAL CFLs in [CFL_MIN, CFL_MAX]."""
    n_hi = n_ts_from_cfl(CFL_MIN)
    n_lo = n_ts_from_cfl(CFL_MAX)
    if n_lo > n_hi:
        n_lo, n_hi = n_hi, n_lo
    return geometric_n_ts(n_lo, n_hi, N_INITIAL)


def scan_initial_cfls(solver, label, cache):
    """Evaluate the initial CFL grid, smallest CFL first."""
    for n_ts in reversed(initial_step_counts()):
        evaluate(solver, n_ts, label, cache)


def stability_bracket(cache):
    """Return (last stable n_ts, first unstable n_ts) from the current cache."""
    n_last_stable = None
    n_first_unstable = None
    for n_ts in sorted(cache, reverse=True):
        _, error, _ = cache[n_ts]
        if is_stable(error):
            n_last_stable = n_ts
        else:
            n_first_unstable = n_ts
            break
    return n_last_stable, n_first_unstable


def refine_stability_limit(solver, label, cache, n_last_stable, n_first_unstable):
    """Bisect the instability bracket, using at most N_BISECT new runs."""
    if n_first_unstable is None or n_last_stable is None:
        return n_last_stable, n_first_unstable

    lo, hi = n_first_unstable, n_last_stable
    remaining = max(0, int(N_BISECT))
    while hi - lo > 1 and remaining > 0:
        mid = (lo + hi) // 2
        was_cached = mid in cache
        _, error, _ = evaluate(solver, mid, label, cache)
        if not was_cached:
            remaining -= 1
        if is_stable(error):
            hi = mid
        else:
            lo = mid
    return hi, lo


def lift_from_samples(cache):
    """Estimate temporal-error onset from already computed stable samples."""
    rows = cached_stable(cache)
    if not rows:
        return None, np.nan
    error_floor = rows[0][2]
    n_lift = None
    for n_ts, _, error, _ in rows:
        if error > 1.25 * error_floor:
            n_lift = n_ts
            break
    return n_lift, error_floor


def packed_arrays(cache):
    """Return CFL and error arrays sorted by increasing CFL."""
    rows = [(cfl, error) for _, error, cfl in cache.values()]
    rows.sort(key=lambda row: row[0])
    cfls = np.array([row[0] for row in rows])
    errors = np.array([row[1] for row in rows])
    return cfls, errors


def run_flavour(case):
    """Sample an initial CFL grid, then refine the stability cutoff."""
    label = case['label']
    print(f'Building {label} ...', flush=True)
    solver = make_solver(case['key'])
    cache = {}

    scan_initial_cfls(solver, label, cache)
    n_last_stable, n_first_unstable = stability_bracket(cache)
    if n_last_stable is None:
        raise RuntimeError(
            f'{label}: every initial CFL was unstable. Decrease CFL_MIN.'
        )
    n_last_stable, n_first_unstable = refine_stability_limit(
        solver, label, cache, n_last_stable, n_first_unstable
    )
    n_lift, error_floor = lift_from_samples(cache)

    dt_last_stable, _, cfl_last_stable = cache[n_last_stable]
    meta = {
        'error_floor': error_floor,
        'dt_floor': dt_from_n_ts(max(cache)),
        'cfl_floor': cfl_from_n_ts(max(cache)),
        'n_last_stable': n_last_stable,
        'dt_last_stable': dt_last_stable,
        'cfl_last_stable': cfl_last_stable,
        'n_first_unstable': n_first_unstable,
        'dt_first_unstable': (
            None if n_first_unstable is None else dt_from_n_ts(n_first_unstable)
        ),
        'cfl_first_unstable': (
            None if n_first_unstable is None else cfl_from_n_ts(n_first_unstable)
        ),
        'n_lift': n_lift,
        'dt_lift': None if n_lift is None else dt_from_n_ts(n_lift),
        'cfl_lift': None if n_lift is None else cfl_from_n_ts(n_lift),
    }

    print(f'  {label} summary:')
    print(
        f'    spatial floor ≈ {error_floor:.6e} at CFL={meta["cfl_floor"]:g}'
    )
    if n_lift is None:
        print('    no temporal-error rise before the stability limit')
    else:
        print(
            f'    temporal rise starts near CFL={meta["cfl_lift"]:g}, '
            f'dt={meta["dt_lift"]:.6g} (n_ts={n_lift})'
        )
    print(
        f'    last stable CFL={cfl_last_stable:g}, dt={dt_last_stable:.6g} '
        f'(n_ts={n_last_stable})'
    )
    if n_first_unstable is None:
        print(f'    still stable at CFL_MAX={CFL_MAX:g}')
    else:
        print(
            f'    first unstable CFL={meta["cfl_first_unstable"]:g}, '
            f'dt={meta["dt_first_unstable"]:.6g} (n_ts={n_first_unstable})'
        )
        if n_last_stable - n_first_unstable > 1:
            print(
                f'    cutoff bracket is {n_first_unstable} < n_ts <= '
                f'{n_last_stable}; increase N_BISECT to resolve further'
            )
    print(f'    {len(cache)} timestep evaluations')

    cfls, errors = packed_arrays(cache)
    return solver, cfls, errors, meta


def snapshot_n_ts(results):
    """Return the n_ts / CFL used for the solution plot.

    This is SOLUTION_CFL_FRACTION of the smallest last-stable CFL among all
    flavours, snapped to an integer step count strictly below that cutoff.
    """
    reference = min(results, key=lambda row: row[3]['cfl_last_stable'])
    meta = reference[3]
    cfl_cut = meta['cfl_last_stable']
    n_cut = meta['n_last_stable']
    n_snap = n_ts_from_cfl(SOLUTION_CFL_FRACTION * cfl_cut)
    if n_snap <= n_cut:
        n_snap = n_cut + 1
    return n_snap, cfl_from_n_ts(n_snap), cfl_cut


def collect_snapshot_solutions(results, n_snap):
    """Re-run each flavour at n_snap and return interpolated (x, u) curves."""
    curves = []
    for case, (solver, _, _, _) in zip(VOLUME_CASES, results):
        _, error, cfl = evaluate(solver, n_snap, case['label'], {})
        if not is_stable(error):
            print(
                f"  skipping {case['label']} solution plot: unstable at "
                f'CFL={cfl:g}'
            )
            continue
        q_local = final_local_solution(solver)
        if q_local is None or not np.all(np.isfinite(q_local)):
            print(f"  skipping {case['label']} solution plot: nonfinite state")
            continue
        x_plot, q_plot = interpolate_solution(solver, q_local)
        curves.append((x_plot, q_plot, case))
    return curves


def operator_label():
    """Short description of the spatial discretization used in the title."""
    name = 'C-SBP' if USE_CSBP else 'D-SBP'
    if (not USE_CSBP) and USE_INTERFACE_DISSIPATION:
        name += ', LF interfaces'
    return f'{name}, {OP_TYPE.upper()} $p={P}$, {NELEM} elements'


def plot_timestep_study(results, cfl_snapshot, solution_curves, x_exact, q_exact):
    """Plot CFL-error curves and the interpolated solution snapshot."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.5),
        gridspec_kw={'height_ratios': (1.15, 1.0)},
    )
    ax = axes[0]
    slope_drawn = False
    for case, (_, cfls, errors, meta) in zip(VOLUME_CASES, results):
        finite = np.isfinite(errors) & (errors > 0.0)
        if not np.any(finite):
            print(f"No finite errors to plot for '{case['label']}'.")
            continue
        ax.loglog(
            cfls[finite],
            errors[finite],
            color=case['color'],
            linestyle=case['linestyle'],
            marker=case['marker'],
            linewidth=1.5,
            markersize=5,
            label=case['label'],
        )
        last_cfl = meta['cfl_last_stable']
        last_match = np.where(cfls == last_cfl)[0]
        if last_match.size:
            ax.loglog(
                cfls[last_match[0]],
                errors[last_match[0]],
                color=case['color'],
                marker='o',
                markersize=9,
                markerfacecolor='none',
                markeredgewidth=1.6,
                linestyle='None',
            )
        if meta['cfl_lift'] is not None:
            ax.axvline(
                meta['cfl_lift'],
                color=case['color'],
                linestyle=':',
                linewidth=1.0,
                alpha=0.55,
            )
        ax.axvline(
            last_cfl,
            color=case['color'],
            linestyle='-.',
            linewidth=1.0,
            alpha=0.55,
        )

        floor = meta['error_floor']
        rising = finite & np.isfinite(floor) & (errors >= 1.25 * floor)
        if (not slope_drawn) and np.any(rising):
            idx = np.where(rising)[0]
            pick = idx[len(idx) // 2]
            reference_cfl = cfls[pick]
            reference_error = errors[pick]
            cfl_line = np.array([
                meta['cfl_lift'] if meta['cfl_lift'] is not None else cfls[finite].min(),
                last_cfl,
            ])
            cfl_line.sort()
            if cfl_line[0] < cfl_line[1]:
                slope = reference_error * (cfl_line / reference_cfl) ** 4
                ax.loglog(
                    cfl_line,
                    slope,
                    color='0.4',
                    linestyle='--',
                    linewidth=1.0,
                    label=r'$\mathrm{CFL}^{4}$',
                )
                slope_drawn = True

    if cfl_snapshot is not None:
        ax.axvline(
            cfl_snapshot,
            color='black',
            linestyle='-',
            linewidth=1.2,
            alpha=0.8,
            label='Solution CFL',
        )

    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color='0.3', linestyle=':', label='Temporal rise'),
        Line2D([0], [0], color='0.3', linestyle='-.', label='Last stable'),
    ])
    ax.legend(handles=handles, loc='best', frameon=False)
    ax.set_xlabel(r'$\mathrm{CFL}$')
    ax.set_ylabel(r'$\| u - u_{\mathrm{ex}} \|_{\mathsf{H}}$')
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15)
    )
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: rf'${x:g}$' if x > 0 else '')
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=15))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(
        rf'{operator_label()}'
        '\n'
        rf'IC: {INITIAL_CONDITION}'
    )
    ax.grid(True, which='both', alpha=0.2)

    ax_sol = axes[1]
    ax_sol.plot(x_exact, q_exact, color='black', linewidth=2.0, label='Exact')
    for x_plot, q_plot, case in solution_curves:
        ax_sol.plot(
            x_plot,
            q_plot,
            color=case['color'],
            linestyle=case['linestyle'],
            linewidth=1.5,
            label=case['label'],
        )
    ax_sol.set_xlabel(r'$x$')
    ax_sol.set_ylabel(r'$u$')
    ax_sol.set_xlim(XMIN, XMAX)
    ax_sol.legend(loc='best', frameon=False)
    if cfl_snapshot is not None:
        ax_sol.set_title(rf'Solution at $\mathrm{{CFL}}={cfl_snapshot:g}$')
    ax_sol.grid(alpha=0.2)

    fig.tight_layout()
    if SAVEFILE is not None:
        fig.savefig(SAVEFILE, bbox_inches='tight')
        print(f'Saved {Path(SAVEFILE)}')
    if 'agg' in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()
    return fig


# Run at module scope so every scan and plotting array remains interactive.
if USE_CSBP and USE_INTERFACE_DISSIPATION:
    print(
        'Note: periodic C-SBP has no interior interfaces, so '
        'USE_INTERFACE_DISSIPATION is ignored.'
    )

results = [run_flavour(case) for case in VOLUME_CASES]
n_snap, cfl_snapshot, cfl_cut = snapshot_n_ts(results)
print(
    f'Solution snapshot at CFL={cfl_snapshot:g} '
    f'({SOLUTION_CFL_FRACTION:g} of the most restrictive cutoff '
    f'{cfl_cut:g}, n_ts={n_snap})'
)
solution_curves = collect_snapshot_solutions(results, n_snap)
solver0 = results[0][0]
x_exact = np.linspace(XMIN, XMAX, 2001)
q_exact = np.asarray(
    solver0.diffeq.exact_sol(time=T_FINAL, x=x_exact)
).reshape(-1)
figure = plot_timestep_study(
    results, cfl_snapshot, solution_curves, x_exact, q_exact
)
