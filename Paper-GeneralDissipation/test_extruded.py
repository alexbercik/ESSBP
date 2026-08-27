#!/usr/bin/env python3
"""Compare a 2D extruded square wave with the corresponding 1D problem.

The square mesh uses the same element count and SBP operator in both physical
directions. With velocity (1, 0), every horizontal profile of the 2D solution
must therefore reproduce the 1D solution. The comparison is performed first
without volume dissipation and then with direction-split volume dissipation.

Edit the parameters below, then run this file directly or in an interactive
Jupyter/VS Code window. The comparison arrays and optional figure are returned
to module scope for later inspection.
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

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

from Source.DiffEq.LinearConv import LinearConv as LinearConv1D
from Source.DiffEq.LinearConv2D import LinearConv as LinearConv2D
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp


POLYNOMIAL_DEGREE = 4         # Polynomial degree in both directions.
ELEMENTS_PER_DIRECTION = 10   # Elements in 1D and along each 2D axis.
TIME_STEP = 2.5e-4            # Fixed RK4 step.
FINAL_TIME = 1.0              # Final comparison time.
INTERPOLATION_POINTS = 30     # Plot points per element edge.
PLOT_SLICE_COUNT = 10         # Number of horizontal 2D profiles to compare.

# Set this to False when only the numerical regression output is wanted.
PLOT_RESULTS = True            # Show the profile and difference plots.

# The directional xi sensor must reproduce the 1D sensor, while the eta
# sensor must vanish for an exactly extruded state.
DISSIPATION = {
    'diss_type': 'new_directional', # Direction-split entropy budgets.
    'kappa': 1.0,                   # Overall dissipation strength.
    'beta': (POLYNOMIAL_DEGREE + 1)
            / (2 * np.ceil(POLYNOMIAL_DEGREE / 2)), # Sensor exponent.
    'sensor_s': int(np.ceil(POLYNOMIAL_DEGREE / 2) + 1), # Sensor order.
    'distribution_s': 1,            # Distribution derivative order.
    'sensor_type': 'cons',          # 'cons' or 'none'.
    'distribution_type': 'cons_sca', # 'cons_sca' or 'cons_mat'.
    'budget_type': 'cheap',         # Cheap entropy-viscosity budget.
}


def square_wave(x):
    """Return the repository's periodic square wave on unique mesh nodes."""
    x_scaled = np.mod(x, 1.0)
    return ((x_scaled > 0.25) & (x_scaled < 0.75)).astype(float)[:, None]


def make_solvers(volume_dissipation):
    """Construct matching continuous-SBP solvers in one and two dimensions."""
    common = {
        'settings': {},
        'tm_method': 'rk4',
        'dt': TIME_STEP,
        't_final': FINAL_TIME,
        'p': POLYNOMIAL_DEGREE,
        'disc_type': 'div',
        'surf_diss': 'nd',
        'vol_diss': volume_dissipation,
        'disc_nodes': 'lgl',
        'bc': 'periodic',
        'print_progress': False,
    }
    # Equation and metric setup contains unconditional diagnostic prints in a
    # few legacy paths. Keep this regression driver's output to its comparisons.
    with redirect_stdout(io.StringIO()):
        solver_1d = PdeSolverCSbp(
            LinearConv1D([1.0], q0_type='squarewave'),
            nelem=ELEMENTS_PER_DIRECTION,
            xmin=0.0,
            xmax=1.0,
            **common,
        )
        solver_2d = PdeSolverCSbp(
            LinearConv2D([1.0, 0.0]),
            nelem=(ELEMENTS_PER_DIRECTION, ELEMENTS_PER_DIRECTION),
            xmin=(0.0, 0.0),
            xmax=(1.0, 1.0),
            **common,
        )
    solver_1d.tm_print_nothing = True
    solver_2d.tm_print_nothing = True
    return solver_1d, solver_2d


def final_state(solver):
    """Extract the final state regardless of time-history storage settings."""
    if solver.q_sol.ndim == 3:
        return solver.q_sol[:, :, -1]
    return solver.q_sol


def interpolate_1d_solution(solver, q_global):
    """Interpolate a global C-SBP solution inside every 1D element."""
    reference_points = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
    q_local = solver.gather(q_global)
    x_plot = solver.sbp.basis.eval_nodal_vec(
        solver.mesh.x_elem, reference_points
    )
    q_plot = solver.sbp.basis.eval_nodal_vec(q_local, reference_points)
    return x_plot.flatten(order='F'), q_plot.flatten(order='F')


def interpolate_2d_slices(solver, q_global):
    """Interpolate the 2D polynomial on ten physical y-slices."""
    reference_points = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
    q_local = solver.gather(q_global)
    q_interpolated = solver.sbp.basis.eval_nodal_vec(
        q_local, reference_points, dim=2
    )
    xy_interpolated = solver.sbp.basis.eval_nodal_vec(
        solver.mesh.xy_elem, reference_points, dim=2
    )

    # Exclude the upper reference endpoint from each y-element because it is
    # the lower endpoint of the next periodic element. Select ten positions
    # from the resulting global interpolated y-grid.
    points_per_periodic_element = INTERPOLATION_POINTS - 1
    total_y_points = solver.nelem[1] * points_per_periodic_element
    slice_positions = np.unique(
        np.linspace(
            0,
            total_y_points - 1,
            min(PLOT_SLICE_COUNT, total_y_points),
            dtype=int,
        )
    )

    x_plot = None
    profiles = []
    for position in slice_positions:
        element_y = position // points_per_periodic_element
        local_y = position % points_per_periodic_element
        x_segments = []
        q_segments = []
        for element_x in range(solver.nelem[0]):
            element = solver.nelem[1] * element_x + element_y
            x_element = xy_interpolated[:, 0, element].reshape(
                INTERPOLATION_POINTS, INTERPOLATION_POINTS
            )
            q_element = q_interpolated[:, element].reshape(
                INTERPOLATION_POINTS, INTERPOLATION_POINTS
            )
            x_segments.append(x_element[:, local_y])
            q_segments.append(q_element[:, local_y])

        slice_x = np.concatenate(x_segments)
        if x_plot is None:
            x_plot = slice_x
        else:
            np.testing.assert_allclose(
                slice_x, x_plot, rtol=0.0, atol=1.0e-13
            )
        profiles.append(np.concatenate(q_segments))

    return x_plot, np.column_stack(profiles)


def run_comparison(label, volume_dissipation, final_tolerance=None):
    """Solve one configuration and collect extrusion-consistency failures."""
    failures = []
    solver_1d, solver_2d = make_solvers(volume_dissipation)
    q_1d = square_wave(solver_1d.mesh.x)
    q_2d = square_wave(solver_2d.mesh.xy[:, 0])

    nx = solver_1d.Nn_global
    ny = solver_2d.Nn_global // nx
    if solver_2d.Nn_global != nx * ny:
        raise AssertionError('The 2D global grid is not a tensor product.')

    # The C-SBP global ordering stores all y nodes for one x node together.
    expected_x = np.repeat(solver_1d.mesh.x, ny)
    np.testing.assert_allclose(
        solver_2d.mesh.xy[:, 0], expected_x, rtol=0.0, atol=1.0e-14
    )
    expected_q_2d = np.repeat(q_1d[:, 0], ny)
    np.testing.assert_array_equal(q_2d[:, 0], expected_q_2d)

    # Check the semidiscrete operators before time-integration roundoff can
    # perturb otherwise identical replicated states.
    rhs_1d = solver_1d.dqdt(q_1d, 0.0)[:, 0]
    rhs_2d = solver_2d.dqdt(q_2d, 0.0)[:, 0]
    rhs_error = np.max(np.abs(rhs_2d - np.repeat(rhs_1d, ny)))
    rhs_scale = max(np.max(np.abs(rhs_1d)), np.max(np.abs(rhs_2d)))
    relative_rhs_error = rhs_error / max(rhs_scale, np.finfo(float).eps)
    if final_tolerance is not None and rhs_error > 5.0e-13:
        failures.append(
            f'{label}: extruded RHS differs from 1D by {rhs_error:.3e}.'
        )

    sensor_ratio = None
    if volume_dissipation is not None:
        theta_1d = np.real_if_close(solver_1d.adiss.theta).real
        theta_directional = np.real_if_close(solver_2d.adiss.theta).real
        theta_2d = theta_directional[0].reshape(
            ELEMENTS_PER_DIRECTION, ELEMENTS_PER_DIRECTION
        )
        transverse_sensor = np.max(np.abs(theta_directional[1]))
        if transverse_sensor > 5.0e-13:
            failures.append(
                f'{label}: transverse sensor is {transverse_sensor:.3e}.'
            )
        active_sensor = theta_1d > 100.0 * np.finfo(float).eps
        if np.any(active_sensor):
            sensor_ratio = np.median(
                theta_2d[active_sensor, :] / theta_1d[active_sensor, None]
            )
            if abs(sensor_ratio - 1.0) > 5.0e-13:
                failures.append(
                    f'{label}: active xi sensor ratio is {sensor_ratio:.16g}.'
                )

    solver_1d.solve(q_1d)
    solver_2d.solve(q_2d)
    result_1d_global = final_state(solver_1d)
    result_2d_global = final_state(solver_2d)
    result_1d = result_1d_global[:, 0]
    result_2d = result_2d_global[:, 0].reshape(nx, ny)

    profile_error = np.max(np.abs(result_2d - result_1d[:, None]))
    profile_scale = max(np.max(np.abs(result_1d)), np.finfo(float).eps)
    relative_profile_error = profile_error / profile_scale
    # A true extrusion must remain constant in y and match the 1D evolution.
    slice_spread = np.max(np.abs(result_2d - result_2d[:, :1]))
    if slice_spread > 5.0e-13:
        failures.append(
            f'{label}: 2D y-slices differ by {slice_spread:.3e}.'
        )
    if final_tolerance is not None and profile_error > final_tolerance:
        failures.append(
            f'{label}: extruded profiles differ from 1D by '
            f'{profile_error:.3e}.'
        )

    print(
        f'{label:21s} RHS error = {rhs_error:.3e} '
        f'({relative_rhs_error:.3%}), final 1D/2D error = '
        f'{profile_error:.3e} ({relative_profile_error:.3%}), '
        f'2D slice spread = {slice_spread:.3e}'
    )
    if sensor_ratio is not None:
        print(f'  2D/1D active sensor ratio = {sensor_ratio:.6g}')

    x_plot, result_1d_plot = interpolate_1d_solution(
        solver_1d, result_1d_global
    )
    x_plot_2d, result_2d_plot = interpolate_2d_slices(
        solver_2d, result_2d_global
    )
    np.testing.assert_allclose(
        x_plot_2d, x_plot, rtol=0.0, atol=1.0e-13
    )
    return x_plot, result_1d_plot, result_2d_plot, failures


def main(plot=False):
    """Run the comparisons without and with volume dissipation."""
    figure = None
    x, baseline_1d, baseline_2d, baseline_failures = run_comparison(
        'No volume dissipation',
        None,
        final_tolerance=5.0e-13,
    )
    _, dissipative_1d, dissipative_2d, dissipative_failures = run_comparison(
        'Directional dissipation',
        DISSIPATION,
        final_tolerance=2.0e-11,
    )
    failures = baseline_failures + dissipative_failures

    dissipative_effect = np.max(np.abs(dissipative_1d - baseline_1d))
    if dissipative_effect <= 1.0e-8:
        failures.append(
            'The dissipative solution is indistinguishable from the baseline.'
        )
    print(f'Maximum effect of dissipation = {dissipative_effect:.3e}')

    if plot:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        cases = (
            ('No volume dissipation', baseline_1d, baseline_2d),
            ('Directional dissipation', dissipative_1d, dissipative_2d),
        )
        for axis, (title, profile_1d, profile_2d) in zip(axes[0], cases):
            axis.plot(x, profile_1d, 'k-', linewidth=2, label='1D')
            slice_lines = axis.plot(
                x,
                profile_2d,
                '--',
                linewidth=1,
            )
            slice_lines[0].set_label('2D y-slices')
            axis.set_title(title)
            axis.set_xlabel('x')
            axis.grid(alpha=0.25)
            axis.legend()

        # A direct overlay and difference plot make the relatively localized
        # dissipative change visible even when the full profiles look alike.
        axes[1, 0].plot(
            x, baseline_1d, 'k--', linewidth=1.5, label='No dissipation, 1D'
        )
        axes[1, 0].plot(
            x, dissipative_1d, color='tab:red', linewidth=1.5,
            label='Dissipation, 1D'
        )
        axes[1, 0].set_title('Direct 1D comparison')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend()

        axes[1, 1].plot(
            x,
            dissipative_1d - baseline_1d,
            'k-',
            linewidth=2,
            label='1D',
        )
        difference_lines = axes[1, 1].plot(
            x,
            dissipative_2d - baseline_2d,
            '--',
            linewidth=1,
        )
        difference_lines[0].set_label('2D y-slices')
        axes[1, 1].set_title('Change caused by dissipation')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend()

        axes[0, 0].set_ylabel('q')
        axes[1, 0].set_ylabel('q')
        figure.tight_layout()
        plt.show()

    # Defer the assertion until after plt.show() so failed high-order runs can
    # still be inspected in an interactive window.
    if failures:
        raise AssertionError('\n'.join(failures))
    print('Extrusion diagnostics passed: the 2D profiles reproduce the 1D case.')
    return (
        x,
        baseline_1d,
        baseline_2d,
        dissipative_1d,
        dissipative_2d,
        dissipative_effect,
        figure,
    )


# Keep every comparison array available in an interactive namespace.
(
    x,
    baseline_1d,
    baseline_2d,
    dissipative_1d,
    dissipative_2d,
    dissipative_effect,
    figure,
) = main(plot=PLOT_RESULTS)
