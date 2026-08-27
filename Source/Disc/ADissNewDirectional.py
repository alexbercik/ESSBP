#!/usr/bin/env python3
"""Direction-split entropy-budgeted artificial volume dissipation.

This variant repeats the same sensor, budget, entropy normalization, and weak
distribution construction as :mod:`ADissNew` in each reference direction,
then sums the directional weak terms before converting to strong form.
"""

import numpy as np

from Source.Disc.ADissNew import ADissNew
import Source.Methods.Functions as fn


class ADissNewDirectional(ADissNew):
    """Apply an independent entropy budget in each reference direction."""

    required_diss_type = 'new_directional'

    def __init__(self, solver):
        super().__init__(solver)

        # In one dimension the directional construction is exactly ADissNew.
        # Bind that implementation directly so there is no second 1D formula
        # to maintain or compare.
        if self.dim == 1:
            self.dissipation = super().dissipation
            return

        # Both scalar and matrix directional budgets require the frozen
        # contravariant vectors. ADissNew already constructs them for cons_mat.
        if not hasattr(self, 'element_metrics'):
            self._set_element_average_metrics()

        self._set_directional_reference_operators()
        if self.dim == 2:
            self.dissipation = self._dissipation_2d
        else:
            self.dissipation = self._dissipation_3d

    def _set_directional_reference_operators(self):
        """Build one sensor quadratic form and line projector per direction."""
        h_1d = np.diag(np.asarray(self.solver.sbp.H))
        mean_1d = np.ones((self.nen, 1)) @ (
            h_1d[None, :] / np.sum(h_1d)
        )
        eye = np.eye(self.nen)

        if self.dim == 2:
            self.line_projectors = (
                np.kron(mean_1d, eye),
                np.kron(eye, mean_1d),
            )
        else:
            self.line_projectors = (
                np.kron(mean_1d, np.kron(eye, eye)),
                np.kron(eye, np.kron(mean_1d, eye)),
                np.kron(eye, np.kron(eye, mean_1d)),
            )

        H_inv_sqrt = 1.0 / np.sqrt(self.H_ref)
        directional_matrices = []
        directional_eigenvalues = []
        for derivative in self.sensor_Ds_ref:
            matrix = derivative.T @ (self.H_ref[:, None] * derivative)
            normalized = (
                H_inv_sqrt[:, None] * matrix * H_inv_sqrt[None, :]
            )
            normalized = 0.5 * (normalized + normalized.T)
            eigenvalue = np.linalg.eigvalsh(normalized)[-1]
            if not np.isfinite(eigenvalue):
                raise FloatingPointError(
                    'A directional reference sensor eigenvalue is nonfinite.'
                )
            directional_matrices.append(matrix)
            directional_eigenvalues.append(eigenvalue)

        self.directional_sensor_M = tuple(directional_matrices)
        self.directional_sensor_Lambda = np.asarray(directional_eigenvalues)

    def _prepare_multidimensional_state(self, q):
        """Validate and reshape the data shared by the 2D and 3D kernels."""
        q = np.asarray(q)
        if q.ndim != 2 or q.shape != self.qshape:
            raise ValueError(f'Expected q with shape {self.qshape}; got {q.shape}.')

        q_nodes = q.reshape(self.np, self.neq_node, self.n_elem)
        w = np.asarray(self.solver.diffeq.entropy_var(q))
        if w.shape != q.shape:
            raise ValueError(
                f'entropy_var(q) must return shape {q.shape}; got {w.shape}.'
            )
        w_nodes = w.reshape(self.np, self.neq_node, self.n_elem)
        q_bar = np.einsum('ie,ice->ce', self.H_phys, q_nodes)
        q_bar *= self.volume_inv[None, :]

        A0_bar = np.asarray(self.solver.diffeq.dqdw(q_bar))
        if A0_bar.shape != self.A0_shape:
            raise ValueError(
                f'dqdw(q_bar) must return shape {self.A0_shape}; '
                f'got {A0_bar.shape}.'
            )

        scaled_state = None
        if self.sensor_type != 'none':
            scaled_state, _ = self._scaled_state_and_speed(q_nodes, q_bar)
        return q, q_nodes, w_nodes, q_bar, A0_bar, scaled_state

    def _directional_sensor(self, scaled_state):
        """Measure line-wise smoothness independently in each direction."""
        scaled_flat = scaled_state.reshape(self.np, -1)
        theta = np.zeros(
            (self.dim, self.n_elem), dtype=scaled_state.dtype
        )
        for direction, (projector, matrix, eigenvalue) in enumerate(zip(
            self.line_projectors,
            self.directional_sensor_M,
            self.directional_sensor_Lambda,
        )):
            if eigenvalue == 0.0:
                continue

            line_mean = (projector @ scaled_flat).reshape(scaled_state.shape)
            fluctuation = scaled_state - line_mean
            fluctuation_flat = fluctuation.reshape(self.np, -1)
            matrix_fluctuation = (matrix @ fluctuation_flat).reshape(
                fluctuation.shape
            )
            numerator = np.einsum(
                'ice,ice->e', fluctuation, matrix_fluctuation
            )
            numerator[np.real(numerator) <= self.epsilon_theta] = 0.0
            norm = np.einsum(
                'i,ice,ice->e', self.H_ref, fluctuation, fluctuation
            )
            theta[direction] = numerator / (
                eigenvalue * norm + self.epsilon_theta
            )
        return theta

    def _directional_distributions(self, q_nodes, q_bar):
        """Return one weak conservative distribution per direction."""
        absolute_jacobians = None
        if self.distribution_type == 'cons_mat':
            # Stack all directions so the differential equation evaluates the
            # frozen absolute Jacobians in one vectorized call.
            directional_q_bar = np.tile(q_bar, (self.dim, 1))
            absolute_jacobians = np.asarray(
                self.solver.diffeq.dEndq_abs(
                    directional_q_bar, self.element_metrics
                )
            )
            expected_shape = (
                self.dim,
                self.neq_node,
                self.neq_node,
                self.n_elem,
            )
            if absolute_jacobians.shape != expected_shape:
                raise ValueError(
                    'The absolute flux Jacobian must return shape '
                    f'{expected_shape}; got {absolute_jacobians.shape}.'
                )

        q_flat = q_nodes.reshape(self.np, -1)
        distributions = np.empty(
            (self.dim, self.np, self.neq_node, self.n_elem),
            dtype=q_nodes.dtype,
        )
        for direction, derivative_operator in enumerate(
            self.distribution_Ds_ref
        ):
            derivative = (derivative_operator @ q_flat).reshape(q_nodes.shape)
            if absolute_jacobians is None:
                weighted_derivative = derivative * self.H_ref[:, None, None]
            else:
                weighted_derivative = np.einsum(
                    'abe,ibe->iae',
                    absolute_jacobians[direction],
                    derivative,
                )
                weighted_derivative *= self.H_ref[:, None, None]
            distributions[direction] = (
                derivative_operator.T
                @ weighted_derivative.reshape(self.np, -1)
            ).reshape(q_nodes.shape)
        return distributions

    def _directional_wave_speeds(self, q_nodes):
        """Return the nodal-maximum contravariant speed in each direction."""
        if self.diffeq_name == 'LinearConvection':
            if self.dim == 2:
                physical_speed = np.array(
                    [self.solver.diffeq.ax, self.solver.diffeq.ay]
                )
            else:
                physical_speed = np.array([
                    self.solver.diffeq.ax,
                    self.solver.diffeq.ay,
                    self.solver.diffeq.az,
                ])
            speed = np.einsum(
                'aje,j->ae', self.element_metrics, physical_speed
            )
            return np.asarray(fn.cabs(speed), dtype=np.result_type(
                q_nodes.dtype, speed.dtype
            ))

        # Euler2d is the only multidimensional system currently supported.
        rho = q_nodes[:, 0, :]
        momentum = q_nodes[:, 1:-1, :]
        energy = q_nodes[:, -1, :]
        velocity = momentum / rho[:, None, :]
        velocity_sq = np.sum(velocity * velocity, axis=1)
        pressure = (self.solver.diffeq.g - 1.0) * (
            energy - 0.5 * rho * velocity_sq
        )
        if np.any(np.real(rho) <= 0) or np.any(np.real(pressure) <= 0):
            raise ValueError(
                'Directional Euler dissipation requires positive density and pressure.'
            )
        sound_speed = np.sqrt(self.solver.diffeq.g * pressure / rho)
        normal_velocity = np.einsum(
            'aje,ije->aie', self.element_metrics, velocity
        )
        metric_norm = np.sqrt(np.sum(
            self.element_metrics * self.element_metrics, axis=1
        ))
        nodal_speeds = (
            fn.cabs(normal_velocity)
            + metric_norm[:, None, :] * sound_speed[None, :, :]
        )
        active_nodes = np.argmax(np.real(nodal_speeds), axis=1)
        directions = np.arange(self.dim)[:, None]
        elements = np.arange(self.n_elem)[None, :]
        return nodal_speeds[directions, active_nodes, elements]

    def _directional_budgets(self, w_nodes, A0_bar, q_nodes):
        """Compute the cheap budget independently in each direction."""
        A0 = A0_bar[0]
        derivatives = (self.D_ref_stacked @ w_nodes.reshape(self.np, -1)).reshape(
            self.dim, self.np, self.neq_node, self.n_elem
        )
        if self.neq_node == 1:
            A0_derivatives = (
                derivatives * A0[0, 0, :][None, None, None, :]
            )
        else:
            A0_derivatives = np.matmul(
                derivatives.transpose(3, 0, 1, 2),
                A0.transpose(2, 1, 0)[:, None, :, :],
            ).transpose(1, 2, 3, 0)
        budget_form = np.einsum(
            'i,dice,dice->de',
            self.H_ref,
            derivatives,
            A0_derivatives,
        )
        return self._directional_wave_speeds(q_nodes) * budget_form / self.nen

    def _finish_directional_dissipation(
        self, q, q_nodes, w_nodes, q_bar, A0_bar, scaled_state
    ):
        """Assemble, normalize, and sum all directional contributions."""
        # These are the same ingredients as ADissNew, with one scalar sensor,
        # entropy budget, and contraction for each reference direction.
        if self.sensor_type == 'none':
            self.theta = np.ones((self.dim, self.n_elem), dtype=q.dtype)
        else:
            self.theta = self._directional_sensor(scaled_state)
        distributions = self._directional_distributions(q_nodes, q_bar)
        contractions = np.einsum(
            'ice,dice->de', w_nodes, distributions
        )
        budgets = self._directional_budgets(w_nodes, A0_bar, q_nodes)
        entropy_budgets = self.kappa * self.theta ** self.beta * budgets
        normalizations = contractions / (
            contractions * contractions + self.epsilon
        )
        self.element_coefficient = -entropy_budgets * normalizations

        # Sum the weak directional contributions first, then apply the one
        # physical inverse norm shared by the element-local residual.
        weak_dissipation = np.sum(
            distributions * self.element_coefficient[:, None, None, :],
            axis=0,
        )
        strong_dissipation = weak_dissipation * self.H_inv_phys[:, None, :]

        if self.store_diagnostics:
            self.entropy_viscosity_budget = budgets.copy()
            self.entropy_budget = entropy_budgets.copy()
            self.entropy_contraction = contractions.copy()
            self.distribution_vector = distributions.copy()
            self.normalized_distribution_vector = (
                distributions * normalizations[:, None, None, :]
            )

        if not np.all(np.isfinite(strong_dissipation)):
            raise FloatingPointError(
                'Directional entropy-budgeted dissipation produced nonfinite values.'
            )
        return np.ascontiguousarray(strong_dissipation.reshape(q.shape))

    def _dissipation_2d(self, q):
        """Return the sum of the xi and eta entropy-budgeted terms."""
        if self.dim != 2:
            raise ValueError('The 2D directional kernel requires dim=2.')
        state = self._prepare_multidimensional_state(q)
        return self._finish_directional_dissipation(*state)

    def _dissipation_3d(self, q):
        """Return the sum of the xi, eta, and zeta entropy-budgeted terms."""
        if self.dim != 3:
            raise ValueError('The 3D directional kernel requires dim=3.')
        state = self._prepare_multidimensional_state(q)
        return self._finish_directional_dissipation(*state)
