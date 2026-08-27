#!/usr/bin/env python3
"""Measure the smooth-vortex scaling of entropy-budgeted dissipation terms."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# Find the repository when run as a file or as cells in an interactive window.
try:
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPOSITORY_ROOT = Path.cwd().resolve()
    if not (REPOSITORY_ROOT / 'Source').is_dir():
        REPOSITORY_ROOT = REPOSITORY_ROOT.parent
if not (REPOSITORY_ROOT / 'Source').is_dir():
    raise RuntimeError('Run this driver from the ESSBP repository or its paper folder.')
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


GAMMA = 1.4
XMIN = (0.0, -5.0)
XMAX = (20.0, 5.0)
DIMENSION = 2

POLYNOMIAL_DEGREE = 4
SENSOR_ORDERS = (3,)
# Change this independently of the sensor orders to study D_{s_a} in a.
DISTRIBUTION_S = 1
BETAS = (1.0, 1.25, 1.5)
# Retain the requested four grids and add N=128 so the finest-three fit for
# the s=2 sensor is beyond its visible pre-asymptotic range.
GRID_SIZES = (8, 16, 32, 64, 128)
DISSIPATION_TYPES = ('new', 'new_directional')

RATE_TOLERANCE = 0.5
MINIMUM_FIT_POINTS = 3
PLOT_RESULTS = True

QUANTITY_LABELS = {
    'sensor': r'Sensor $\theta$',
    'budget': r'Cheap budget $B$',
    'entropy_budget': r'Full budget $\mathcal{E}$',
    'distribution': r'Weak vector $\mathbf{a}$',
    'normalized_distribution': r'Normalized vector $\alpha_\epsilon$',
    'dissipation': r'Strong dissipation $\mathcal{D}$',
}
DIRECTION_LABELS = ('xi', 'eta')


def make_solver(dissipation_type, sensor_order, beta, elements_y):
    """Build one square-cell vortex discretization for a residual evaluation."""
    volume_dissipation = {
        'diss_type': dissipation_type,
        'kappa': 1.0,
        'beta': beta,
        'sensor_s': sensor_order,
        'distribution_s': DISTRIBUTION_S,
        'sensor_type': 'cons',
        'distribution_type': 'cons_sca',
        'budget_type': 'cheap',
        'store_diagnostics': True,
    }

    # Suppress legacy setup notices so the rate table remains readable.
    with redirect_stdout(io.StringIO()):
        equation = Euler(
            [287.0, GAMMA],
            q0_type='vortex_zelalem',
            test_case='vortex_zelalem',
            bc='periodic',
            nondimensionalize=False,
        )
        solver = PdeSolverSbp(
            equation,
            settings={
                'metric_method': 'exact',
                'use_optz_metrics': False,
            },
            tm_method='rk4',
            dt=1.0e-3,
            t_final=1.0e-3,
            p=POLYNOMIAL_DEGREE,
            disc_type='div',
            surf_diss='nd',
            vol_diss=volume_dissipation,
            nelem=(2 * elements_y, elements_y),
            disc_nodes='lgl',
            bc='periodic',
            xmin=XMIN,
            xmax=XMAX,
            print_progress=False,
        )
    return solver


def maximum_by_direction(values, directional):
    """Return max-norm magnitudes for aggregate or directional arrays."""
    values = np.abs(np.real_if_close(values))
    if not directional:
        return {'total': float(np.max(values))}
    return {
        DIRECTION_LABELS[direction]: float(np.max(values[direction]))
        for direction in range(DIMENSION)
    }


def active_normalized_distribution(adiss, directional):
    """Measure alpha where the entropy contraction is safely nonzero.

    A global maximum of alpha is dominated by elements where b is nearly
    zero and the fixed epsilon regularizer is active. The formal
    O(h^-s_a) scaling applies on active elements with b=O(h^(2s_a)), so use
    the element with the largest absolute contraction in each direction.
    """
    normalized = np.abs(np.real_if_close(
        adiss.normalized_distribution_vector
    ))
    contraction = np.abs(np.real_if_close(adiss.entropy_contraction))
    regularization_threshold = 1000.0

    if not directional:
        element = int(np.argmax(contraction))
        ratio = contraction[element] ** 2 / adiss.epsilon
        magnitude = float(np.max(normalized[..., element]))
        return {'total': magnitude if ratio > regularization_threshold else np.nan}

    magnitudes = {}
    for direction, label in enumerate(DIRECTION_LABELS):
        element = int(np.argmax(contraction[direction]))
        ratio = contraction[direction, element] ** 2 / adiss.epsilon
        magnitude = float(np.max(normalized[direction, ..., element]))
        magnitudes[label] = (
            magnitude if ratio > regularization_threshold else np.nan
        )
    return magnitudes


def add_series_value(series, key, value):
    """Append one grid's magnitude to a named convergence series."""
    series.setdefault(key, []).append(value)


def evaluate_grid(dissipation_type, sensor_order, elements_y, series):
    """Evaluate every requested beta on one exact initial vortex state."""
    solver = make_solver(
        dissipation_type, sensor_order, BETAS[0], elements_y
    )
    q0 = solver.diffeq.set_q0()
    directional = dissipation_type == 'new_directional'

    # The sensor, cheap budget, and distribution vectors do not depend on
    # beta. Record them during the first residual evaluation only.
    for beta_index, beta in enumerate(BETAS):
        solver.adiss.beta = beta
        dissipation = solver.adiss.dissipation(q0)

        if beta_index == 0:
            base_values = {
                'sensor': solver.adiss.theta,
                'budget': solver.adiss.entropy_viscosity_budget,
                'distribution': solver.adiss.distribution_vector,
            }
            for quantity, values in base_values.items():
                for direction, magnitude in maximum_by_direction(
                    values, directional
                ).items():
                    add_series_value(
                        series,
                        (
                            dissipation_type,
                            sensor_order,
                            None,
                            quantity,
                            direction,
                        ),
                        magnitude,
                    )
            for direction, magnitude in active_normalized_distribution(
                solver.adiss, directional
            ).items():
                add_series_value(
                    series,
                    (
                        dissipation_type,
                        sensor_order,
                        None,
                        'normalized_distribution',
                        direction,
                    ),
                    magnitude,
                )

        for direction, magnitude in maximum_by_direction(
            solver.adiss.entropy_budget, directional
        ).items():
            add_series_value(
                series,
                (
                    dissipation_type,
                    sensor_order,
                    beta,
                    'entropy_budget',
                    direction,
                ),
                magnitude,
            )
        add_series_value(
            series,
            (
                dissipation_type,
                sensor_order,
                beta,
                'dissipation',
                'total',
            ),
            float(np.max(np.abs(np.real_if_close(dissipation)))),
        )


def expected_order(sensor_order, beta, quantity):
    """Return the local mesh exponent predicted by the analysis."""
    sensor_order_exponent = 2 * sensor_order - 2
    if quantity == 'sensor':
        return float(sensor_order_exponent)
    if quantity == 'budget':
        return float(DIMENSION + 1)
    if quantity == 'entropy_budget':
        return float(beta * sensor_order_exponent + DIMENSION + 1)
    if quantity == 'distribution':
        return float(DISTRIBUTION_S)
    if quantity == 'normalized_distribution':
        return float(-DISTRIBUTION_S)
    if quantity == 'dissipation':
        return float(
            beta * sensor_order_exponent + 1 - DISTRIBUTION_S
        )
    raise ValueError(f'Unknown convergence quantity {quantity!r}.')


def fit_rate(mesh_widths, magnitudes):
    """Fit the finest three non-roundoff values and return pairwise rates."""
    mesh_widths = np.asarray(mesh_widths, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    roundoff_floor = 1000.0 * np.finfo(float).eps
    valid = np.isfinite(magnitudes) & (magnitudes > roundoff_floor)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < MINIMUM_FIT_POINTS:
        return np.nan, [], valid_indices

    fit_indices = valid_indices[-MINIMUM_FIT_POINTS:]
    fitted_rate = np.polyfit(
        np.log(mesh_widths[fit_indices]),
        np.log(magnitudes[fit_indices]),
        1,
    )[0]
    pairwise_rates = []
    for coarse, fine in zip(valid_indices[:-1], valid_indices[1:]):
        pairwise_rates.append(
            np.log(magnitudes[coarse] / magnitudes[fine])
            / np.log(mesh_widths[coarse] / mesh_widths[fine])
        )
    return float(fitted_rate), pairwise_rates, fit_indices


def report_rates(series, mesh_widths):
    """Print observed rates and collect every failed expectation."""
    failures = []
    header = (
        f"{'type':16s} {'s':>2s} {'beta':>5s} {'quantity':24s} "
        f"{'dir':>5s} {'expect':>8s} {'fit':>8s}  pairwise"
    )
    print('\n' + header)
    print('-' * len(header))

    rates = {}
    for key in sorted(series, key=lambda item: tuple(str(value) for value in item)):
        dissipation_type, sensor_order, beta, quantity, direction = key
        expected = expected_order(sensor_order, beta, quantity)
        fitted, pairwise, fit_indices = fit_rate(mesh_widths, series[key])
        rates[key] = (fitted, expected, fit_indices)
        beta_label = '-' if beta is None else f'{beta:g}'
        pairwise_label = ', '.join(f'{rate:.3f}' for rate in pairwise)
        print(
            f'{dissipation_type:16s} {sensor_order:2d} {beta_label:>5s} '
            f'{quantity:24s} {direction:>5s} {expected:8.3f} '
            f'{fitted:8.3f}  {pairwise_label}'
        )
        if not np.isfinite(fitted):
            failures.append(
                f'{key}: fewer than {MINIMUM_FIT_POINTS} values remain above roundoff.'
            )
        elif abs(fitted - expected) > RATE_TOLERANCE:
            failures.append(
                f'{key}: fitted order {fitted:.3f}, expected {expected:.3f}.'
            )
    return rates, failures


def plot_rates(series, mesh_widths):
    """Make one six-panel convergence figure for every (s, beta) pair."""
    figures = []
    quantities = tuple(QUANTITY_LABELS)
    colors = {'new': 'tab:blue', 'new_directional': 'tab:orange'}
    direction_styles = {'total': '-', 'xi': '-', 'eta': '--'}

    for sensor_order in SENSOR_ORDERS:
        for selected_beta in BETAS:
            figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
            for axis, quantity in zip(axes.flat, quantities):
                plotted_expected_orders = set()
                reference_anchor = None
                for key, magnitudes in series.items():
                    (
                        dissipation_type,
                        order,
                        series_beta,
                        key_quantity,
                        direction,
                    ) = key
                    if order != sensor_order or key_quantity != quantity:
                        continue
                    # Sensor, B, a, and alpha do not depend on beta and have
                    # series_beta=None. Plot only the selected beta for the
                    # full entropy budget and final dissipation.
                    if series_beta is not None and series_beta != selected_beta:
                        continue

                    direction_label = (
                        '' if direction == 'total' else f', {direction}'
                    )
                    label = f'{dissipation_type}{direction_label}'
                    axis.loglog(
                        mesh_widths,
                        magnitudes,
                        marker='o',
                        color=colors[dissipation_type],
                        linestyle=direction_styles[direction],
                        linewidth=1.3,
                        markersize=4,
                        label=label,
                    )
                    if reference_anchor is None:
                        reference_anchor = float(magnitudes[0])
                    plotted_expected_orders.add(
                        expected_order(sensor_order, series_beta, quantity)
                    )

                if reference_anchor is not None:
                    for reference_index, order in enumerate(sorted(
                        plotted_expected_orders
                    )):
                        offset = 1.8 ** reference_index
                        reference = (
                            reference_anchor
                            * offset
                            * (np.asarray(mesh_widths) / mesh_widths[0]) ** order
                        )
                        axis.loglog(
                            mesh_widths,
                            reference,
                            ':',
                            color='0.35',
                            linewidth=1.0,
                            label=rf'$h^{{{order:g}}}$',
                        )
                axis.set_title(QUANTITY_LABELS[quantity])
                axis.set_xlabel('$h$')
                axis.grid(alpha=0.2, which='both')
                axis.legend(fontsize=12, frameon=False)

            figure.suptitle(
                rf'Isentropic-vortex scaling: $s_\theta={sensor_order}$, '
                rf'$s_a={DISTRIBUTION_S}$, '
                rf'$\beta={selected_beta:g}$',
                y=0.995,
            )
            # Show both budget definitions because each figure compares the
            # aggregate and direction-split formulations.
            figure.text(
                0.5,
                0.955,
                r'$Q_{\alpha,k}=(D_\alpha w_k)^T'
                r'(H_\xi\otimes\overline{A}_{0,k})D_\alpha w_k,'
                r'\quad B_k=c_k h_{\mathrm{eff},k}\mu_k'
                r'\sum_{\alpha=1}^d Q_{\alpha,k},'
                r'\quad B_{\alpha,k}=\frac{c_{\alpha,k}}{n_{\mathrm{en}}}'
                r'Q_{\alpha,k}$',
                ha='center',
                va='top',
                fontsize=9,
            )
            figure.text(
                0.5,
                0.925,
                r'$\mathcal{E}_k=\kappa\theta_k^\beta B_k,'
                r'\qquad \mathcal{E}_{\alpha,k}'
                r'=\kappa\theta_{\alpha,k}^\beta B_{\alpha,k}$',
                ha='center',
                va='top',
                fontsize=9,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
            figures.append(figure)
    return figures


def main(plot=False):
    """Run the refinement study, report rates, and validate the predictions."""
    if DISTRIBUTION_S < 1 or DISTRIBUTION_S > POLYNOMIAL_DEGREE:
        raise ValueError(
            'DISTRIBUTION_S must be between 1 and POLYNOMIAL_DEGREE.'
        )

    series = {}
    print(f'Distribution derivative order s_a={DISTRIBUTION_S}.')
    mesh_widths = np.array([
        (XMAX[1] - XMIN[1]) / elements_y for elements_y in GRID_SIZES
    ])
    for dissipation_type in DISSIPATION_TYPES:
        for sensor_order in SENSOR_ORDERS:
            for elements_y in GRID_SIZES:
                print(
                    f'Evaluating {dissipation_type}, s={sensor_order}, '
                    f'grid=({2 * elements_y}, {elements_y}) ...',
                    flush=True,
                )
                evaluate_grid(
                    dissipation_type, sensor_order, elements_y, series
                )

    _rates, failures = report_rates(series, mesh_widths)
    figures = plot_rates(series, mesh_widths) if plot else []
    if plot and 'agg' not in plt.get_backend().lower():
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)

    # Rate mismatches are diagnostic: keep the interactive study and its
    # figures available even when an asymptotic slope is not yet visible.
    if failures:
        print('\nConvergence-rate warnings:')
        for failure in failures:
            print(f'WARNING: {failure}')
        print('\nStudy completed with convergence-rate warnings.')
    else:
        print('\nAll dissipation-term convergence rates are within tolerance.')


# Run at module scope so VS Code's "Run File in Interactive Window" executes
# the complete study without command-line handling.
main(plot=PLOT_RESULTS)
