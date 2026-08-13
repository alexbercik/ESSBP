#!/usr/bin/env python3
"""Entropy-budgeted, element-local artificial volume dissipation."""

import numbers

import numpy as np

from Source.Disc.ADiss import ADiss
import Source.Methods.Functions as fn


class ADissNew:
    """Construct the nonlinear entropy-budgeted dissipation operator.

    The smoothness sensor and distribution vector use independently selected
    derivative orders. The cheap entropy-viscosity budget remains based on the
    first reference derivative.

    Absolute values and element maxima use a fixed active branch based on the
    real part of the state. Consequently, complex-step differentiation returns
    the derivative of that branch. As with the corresponding real functions,
    no unique derivative exists at an absolute-value zero or a maximum tie.
    """

    # Reuse the legacy central finite-difference diagnostic. ADiss implements
    # it in terms of the public attributes shared by both classes.
    calc_RHS_jac = ADiss.calc_RHS_jac

    def __init__(self, solver):
        if solver.print_progress:
            print('... Setting up Entropy-Budgeted Artificial Dissipation')

        self.solver = solver
        self.type = solver.vol_diss['diss_type'].lower()
        self.dim = solver.dim
        self.neq_node = solver.neq_node
        self.nen = solver.nen
        self.nelem = solver.nelem
        self.np = self.nen ** self.dim

        if self.type != 'new':
            raise ValueError("ADissNew requires diss_type='new'.")

        config = solver.vol_diss
        self.kappa = config.get('kappa', 1.0)
        self.beta = config.get('beta')
        self.sensor_s = config.get('sensor_s')
        self.distribution_s = config.get('distribution_s')

        if (
            isinstance(self.kappa, bool)
            or not isinstance(self.kappa, numbers.Real)
            or not np.isfinite(self.kappa)
            or self.kappa < 0
        ):
            raise ValueError('Entropy-budgeted dissipation: kappa must be a nonnegative real number.')
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, numbers.Real)
            or not np.isfinite(self.beta)
            or self.beta < 0
        ):
            raise ValueError('Entropy-budgeted dissipation: beta must be provided as a nonnegative real number.')
        for name in ('sensor_s', 'distribution_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(
                    f'Entropy-budgeted dissipation: {name} must be provided as '
                    'a positive integer.'
                )

        self.sensor_type = config.get('sensor_type', 'cons')
        self.distribution_type = config.get('distribution_type', 'cons_sca')
        self.budget_type = config.get('budget_type', 'cheap')
        supported_selectors = {
            'sensor_type': {'cons', 'none'},
            'distribution_type': {'cons_mat', 'cons_sca', 'entdcp'},
            'budget_type': {'cheap', 'entdcp'},
        }
        selectors = (
            ('sensor_type', self.sensor_type),
            ('distribution_type', self.distribution_type),
            ('budget_type', self.budget_type),
        )
        for name, value in selectors:
            if not isinstance(value, str) or value.lower() not in supported_selectors[name]:
                raise ValueError(
                    f'Entropy-budgeted dissipation: {name} must be one of '
                    f'{sorted(supported_selectors[name])}.'
                )
            setattr(self, name, value.lower())

        # The entropy-DCP distribution and budget are a matched pair: the
        # budget is exactly the entropy contraction of that distribution.
        if (self.distribution_type == 'entdcp') != (self.budget_type == 'entdcp'):
            raise ValueError(
                "Entropy-budgeted dissipation: distribution_type='entdcp' and "
                "budget_type='entdcp' must be selected together."
            )
        if self.distribution_type == 'entdcp' and self.dim != 1:
            raise ValueError(
                "Entropy-budgeted dissipation: the 'entdcp' choices are "
                'currently implemented only in one dimension.'
            )

        supported_equations = {'LinearConvection', 'Burgers', 'Quasi1dEuler', 'Euler2d'}
        self.diffeq_name = solver.diffeq.diffeq_name
        if self.diffeq_name not in supported_equations:
            raise ValueError(
                'Entropy-budgeted dissipation is currently implemented for '
                'LinearConvection, Burgers, Quasi1dEuler, and Euler2d.'
            )
        if not hasattr(solver.diffeq, 'entropy_var') or not hasattr(solver.diffeq, 'dqdw'):
            raise ValueError('The differential equation must define entropy_var(q) and dqdw(q).')

        self.epsilon = np.finfo(float).eps
        self.epsilon_theta = np.finfo(float).eps
        self.theta = None
        self.element_coefficient = None

        # Match the simple element-type defaults in legacy entdcp: include the
        # reference H weights and do not apply a boundary correction matrix.
        self.entdcp_use_H = True
        self.entdcp_bdy_fix = False
        self.entdcp_avg_half_nodes = True

        self._set_reference_operators()
        self._set_element_geometry()

    def _set_reference_operators(self):
        """Precompute tensor-product reference operators and sensor scaling."""
        H = np.asarray(self.solver.sbp.H)
        if H.ndim != 2 or not np.allclose(H, np.diag(np.diag(H))):
            raise ValueError('Entropy-budgeted dissipation requires a diagonal reference norm.')

        h_1d = np.diag(H)
        D = np.asarray(self.solver.sbp.D)
        eye = np.eye(self.nen)
        sensor_Ds = np.linalg.matrix_power(D, int(self.sensor_s))
        distribution_Ds = np.linalg.matrix_power(D, int(self.distribution_s))

        # The budget always uses the first derivative. The sensor and
        # distribution orders are independent and use their own quadratic
        # forms.
        M_1d = D.T @ (h_1d[:, None] * D)
        sensor_M_1d = sensor_Ds.T @ (h_1d[:, None] * sensor_Ds)
        distribution_M_1d = distribution_Ds.T @ (
            h_1d[:, None] * distribution_Ds
        )

        if self.dim == 1:
            self.H_ref = h_1d
            self.D_ref = (D,)
            self.sensor_Ds_ref = (sensor_Ds,)
            self.distribution_Ds_ref = (distribution_Ds,)
            self.M = M_1d
            self.sensor_M = sensor_M_1d
            self.distribution_M = distribution_M_1d
        elif self.dim == 2:
            self.H_ref = np.kron(h_1d, h_1d)
            self.D_ref = (np.kron(D, eye), np.kron(eye, D))
            self.sensor_Ds_ref = (
                np.kron(sensor_Ds, eye),
                np.kron(eye, sensor_Ds),
            )
            self.distribution_Ds_ref = (
                np.kron(distribution_Ds, eye),
                np.kron(eye, distribution_Ds),
            )
            self.M = np.kron(M_1d, H) + np.kron(H, M_1d)
            self.sensor_M = (
                np.kron(sensor_M_1d, H) + np.kron(H, sensor_M_1d)
            )
            self.distribution_M = (
                np.kron(distribution_M_1d, H)
                + np.kron(H, distribution_M_1d)
            )
        elif self.dim == 3:
            self.H_ref = np.kron(h_1d, np.kron(h_1d, h_1d))
            self.D_ref = (
                np.kron(D, np.kron(eye, eye)),
                np.kron(eye, np.kron(D, eye)),
                np.kron(eye, np.kron(eye, D)),
            )
            self.sensor_Ds_ref = (
                np.kron(sensor_Ds, np.kron(eye, eye)),
                np.kron(eye, np.kron(sensor_Ds, eye)),
                np.kron(eye, np.kron(eye, sensor_Ds)),
            )
            self.distribution_Ds_ref = (
                np.kron(distribution_Ds, np.kron(eye, eye)),
                np.kron(eye, np.kron(distribution_Ds, eye)),
                np.kron(eye, np.kron(eye, distribution_Ds)),
            )
            self.M = (
                np.kron(M_1d, np.kron(H, H))
                + np.kron(H, np.kron(M_1d, H))
                + np.kron(H, np.kron(H, M_1d))
            )
            self.sensor_M = (
                np.kron(sensor_M_1d, np.kron(H, H))
                + np.kron(H, np.kron(sensor_M_1d, H))
                + np.kron(H, np.kron(H, sensor_M_1d))
            )
            self.distribution_M = (
                np.kron(distribution_M_1d, np.kron(H, H))
                + np.kron(H, np.kron(distribution_M_1d, H))
                + np.kron(H, np.kron(H, distribution_M_1d))
            )
        else:
            raise ValueError('Entropy-budgeted dissipation supports dimensions 1, 2, and 3.')

        self.ref_measure = np.sum(self.H_ref)
        self.ref_measure_inv = 1.0 / self.ref_measure

        if np.any(self.H_ref <= 0):
            raise ValueError('Reference norm weights must be positive.')

        # All directional derivatives can be applied in one matrix product in
        # the entropy-viscosity budget.
        self.D_ref_stacked = np.concatenate(self.D_ref, axis=0)

        # Convert the generalized sensor eigenproblem to a symmetric standard
        # problem. The eigenvalue is static and need not carry complex data.
        H_inv_sqrt = 1.0 / np.sqrt(self.H_ref)
        normalized_sensor_M = (
            H_inv_sqrt[:, None] * self.sensor_M * H_inv_sqrt[None, :]
        )
        normalized_sensor_M = 0.5 * (
            normalized_sensor_M + normalized_sensor_M.T
        )
        self.sensor_Lambda = np.linalg.eigvalsh(normalized_sensor_M)[-1]
        if not np.isfinite(self.sensor_Lambda):
            raise FloatingPointError('The reference smoothness eigenvalue is nonfinite.')

        # A derivative is identically zero when its order exceeds the
        # polynomial space. Remove roundoff-sized remnants in either operator.
        roundoff_threshold = 100 * np.finfo(float).eps
        if self.sensor_Lambda <= roundoff_threshold:
            self.sensor_Ds_ref = tuple(
                np.zeros_like(Dsi) for Dsi in self.sensor_Ds_ref
            )
            self.sensor_M.fill(0.0)
            self.sensor_Lambda = 0.0

        normalized_distribution_M = (
            H_inv_sqrt[:, None] * self.distribution_M * H_inv_sqrt[None, :]
        )
        normalized_distribution_M = 0.5 * (
            normalized_distribution_M + normalized_distribution_M.T
        )
        self.distribution_Lambda = np.linalg.eigvalsh(
            normalized_distribution_M
        )[-1]
        if not np.isfinite(self.distribution_Lambda):
            raise FloatingPointError('The reference distribution eigenvalue is nonfinite.')
        if self.distribution_Lambda <= roundoff_threshold:
            self.distribution_Ds_ref = tuple(
                np.zeros_like(Dsi) for Dsi in self.distribution_Ds_ref
            )
            self.distribution_M.fill(0.0)
            self.distribution_Lambda = 0.0

    def _set_element_geometry(self):
        """Store element volumes and the physical scaling in the cheap budget."""
        self.H_phys = np.asarray(self.solver.H_phys)
        self.H_inv_phys = np.asarray(self.solver.H_inv_phys)
        expected_elements = (
            int(np.prod(self.nelem))
            if isinstance(self.nelem, tuple)
            else int(self.nelem)
        )
        expected_shape = (self.np, expected_elements)
        if self.H_phys.shape != expected_shape or self.H_inv_phys.shape != expected_shape:
            raise ValueError(
                f'Physical norm arrays must have shape {expected_shape}; got '
                f'{self.H_phys.shape} and {self.H_inv_phys.shape}.'
            )

        self.n_elem = expected_elements
        self.element_indices = np.arange(self.n_elem)
        self.volume = np.sum(self.H_phys, axis=0)
        if np.any(self.volume <= 0):
            raise ValueError('Element volumes must be positive.')
        self.volume_inv = 1.0 / self.volume
        self.h_eff = (self.volume / self.np) ** (1.0 / self.dim)
        self.mu = self.volume ** ((self.dim - 2.0) / self.dim)
        self.budget_scale = self.h_eff * self.mu
        self.qshape = (self.np * self.neq_node, self.n_elem)
        self.A0_shape = (1, self.neq_node, self.neq_node, self.n_elem)

        if self.distribution_type == 'cons_mat':
            jacobian_name = (
                'dExdq_abs' if self.dim == 1 else 'dEndq_abs'
            )
            if not hasattr(self.solver.diffeq, jacobian_name):
                raise ValueError(
                    "distribution_type='cons_mat' requires the differential "
                    f'equation to define {jacobian_name}.'
                )

            # The 1D cofactor metric J d(xi)/dx is identically one, so only
            # multidimensional distributions need frozen metric directions.
            if self.dim > 1:
                if not hasattr(self.solver, 'mesh') or not hasattr(
                    self.solver.mesh, 'metrics'
                ):
                    raise ValueError(
                        "distribution_type='cons_mat' requires mesh metric terms."
                    )

                nodal_metrics = np.asarray(self.solver.mesh.metrics)
                expected_metric_shape = (
                    self.np,
                    self.dim * self.dim,
                    self.n_elem,
                )
                if nodal_metrics.shape != expected_metric_shape:
                    raise ValueError(
                        f'Mesh metrics must have shape {expected_metric_shape}; '
                        f'got {nodal_metrics.shape}.'
                    )

                # Mesh metrics are stored row-major as
                # J d(xi_alpha)/d(x_j). Freeze each reference direction at its
                # reference-H-weighted element average.
                nodal_metrics = nodal_metrics.reshape(
                    self.np, self.dim, self.dim, self.n_elem
                )
                self.element_metrics = np.einsum(
                    'i,iaje->aje', self.H_ref, nodal_metrics
                ) * self.ref_measure_inv
                if not np.all(np.isfinite(self.element_metrics)):
                    raise FloatingPointError(
                        'Element-average contravariant metric terms are nonfinite.'
                    )
                if np.any(np.all(self.element_metrics == 0.0, axis=1)):
                    raise ValueError(
                        'Each element-average contravariant direction must be nonzero.'
                    )

        # Linear-convection speeds are constant and need not be reconstructed
        # during every residual evaluation.
        if self.diffeq_name == 'LinearConvection':
            if self.dim == 1:
                speeds = (self.solver.diffeq.a,)
            elif self.dim == 2:
                speeds = (self.solver.diffeq.ax, self.solver.diffeq.ay)
            else:
                speeds = (
                    self.solver.diffeq.ax,
                    self.solver.diffeq.ay,
                    self.solver.diffeq.az,
                )
            self.linear_speed = np.sqrt(
                sum(component * component for component in speeds)
            )

    def _active_max(self, values):
        """Select the complex value whose real part is the active maximum."""
        indices = np.argmax(np.real(values), axis=0)
        return values[indices, self.element_indices]

    def _euler_data(self, q_nodes, q_bar):
        """Return scaled Euler state and an element characteristic speed."""
        rho = q_nodes[:, 0, :]
        momentum = q_nodes[:, 1:-1, :]
        energy = q_nodes[:, -1, :]
        velocity = momentum / rho[:, None, :]
        velocity_sq = velocity[:, 0, :] * velocity[:, 0, :]
        if self.dim == 2:
            velocity_sq += velocity[:, 1, :] * velocity[:, 1, :]
        pressure = (self.solver.diffeq.g - 1.0) * (energy - 0.5 * rho * velocity_sq)
        if np.any(np.real(rho) <= 0) or np.any(np.real(pressure) <= 0):
            raise ValueError('Entropy-budgeted Euler dissipation requires positive density and pressure.')

        sound_speed = np.sqrt(self.solver.diffeq.g * pressure / rho)
        velocity_mag = np.sqrt(velocity_sq)
        wave_speed = self._active_max(velocity_mag + sound_speed)

        rho_bar = q_bar[0, :]
        momentum_bar = q_bar[1:-1, :]
        energy_bar = q_bar[-1, :]
        velocity_bar = momentum_bar / rho_bar[None, :]
        velocity_bar_sq = velocity_bar[0, :] * velocity_bar[0, :]
        if self.dim == 2:
            velocity_bar_sq += velocity_bar[1, :] * velocity_bar[1, :]
        pressure_bar = (self.solver.diffeq.g - 1.0) * (
            energy_bar - 0.5 * rho_bar * velocity_bar_sq
        )
        if np.any(np.real(rho_bar) <= 0) or np.any(np.real(pressure_bar) <= 0):
            raise ValueError('Element-average Euler states must have positive density and pressure.')

        sound_speed_bar = np.sqrt(self.solver.diffeq.g * pressure_bar / rho_bar)
        z = np.sqrt(velocity_bar_sq) + sound_speed_bar
        scales = np.empty_like(q_bar)
        scales[0, :] = rho_bar
        scales[1:-1, :] = rho_bar[None, :] * z[None, :]
        scales[-1, :] = rho_bar * z * z
        return q_nodes / scales[None, :, :], wave_speed

    def _scaled_state_and_speed(self, q_nodes, q_bar):
        """Apply state scaling and return one wave speed per element.

        q_nodes has shape (np, neq_node, n_elem), and q_bar has shape
        (neq_node, n_elem).
        """
        if self.diffeq_name == 'LinearConvection':
            wave_speed = np.full(
                self.n_elem,
                self.linear_speed,
                dtype=np.result_type(q_nodes.dtype, self.linear_speed),
            )
            return q_nodes, wave_speed

        if self.diffeq_name == 'Burgers':
            wave_speed = self._active_max(fn.cabs(q_nodes[:, 0, :]))
            return q_nodes, wave_speed

        return self._euler_data(q_nodes, q_bar)

    def _sensor(self, scaled_state):
        """Compute the fluctuation-normalized reference smoothness sensor."""
        if self.sensor_Lambda == 0.0:
            return np.zeros(self.n_elem, dtype=scaled_state.dtype)

        # Flatten the component and element axes so the fixed reference
        # operators are applied to every state column through matrix products.
        scaled_state_flat = scaled_state.reshape(self.np, -1)
        mean = (self.H_ref @ scaled_state_flat).reshape(
            self.neq_node, self.n_elem
        ) * self.ref_measure_inv
        fluctuation = scaled_state - mean[None, :, :]
        fluctuation_flat = fluctuation.reshape(self.np, -1)

        sensor_M_fluctuation = (self.sensor_M @ fluctuation_flat).reshape(
            fluctuation.shape
        )
        numerator = np.einsum(
            'ice,ice->e', fluctuation, sensor_M_fluctuation
        )
        # The sensor matrix is positive semidefinite. Nonpositive or
        # machine-precision values are roundoff at its nullspace.
        numerator[np.real(numerator) <= self.epsilon_theta] = 0.0
        norm = np.einsum('i,ice,ice->e', self.H_ref, fluctuation, fluctuation)
        return numerator / (self.sensor_Lambda * norm + self.epsilon_theta)

    def _entropy_viscosity_budget(self, w_nodes, A0_bar, wave_speed):
        """Compute the cheap cell-average entropy-viscosity budget."""
        A0 = A0_bar[0, :, :, :]
        w_flat = w_nodes.reshape(self.np, -1)
        derivatives = (self.D_ref_stacked @ w_flat).reshape(
            self.dim, self.np, self.neq_node, self.n_elem
        )

        if self.neq_node == 1:
            A0_derivatives = derivatives * A0[0, 0, :][None, None, None, :]
        else:
            # For every element, apply its small A0 matrix to all nodes and
            # coordinate directions in one batched matrix multiplication.
            A0_derivatives = np.matmul(
                derivatives.transpose(3, 0, 1, 2),
                A0.transpose(2, 1, 0)[:, None, :, :],
            ).transpose(1, 2, 3, 0)

        budget_form = np.einsum(
            'i,dice,dice->e', self.H_ref, derivatives, A0_derivatives
        )
        return wave_speed * self.budget_scale * budget_form

    def _entdcp_distribution(self, q, q_nodes, w_nodes):
        """Apply the reference-derivative entropy-DCP weak operator.

        This uses the distribution-order reference derivative, the reference
        H matrix, no boundary correction, and the nodal scalar-wave-speed
        times dq/dw coefficient used by the legacy ``entdcp`` ``scamat``
        implementation.
        """
        if self.diffeq_name == 'LinearConvection':
            nodal_speed = np.full(
                (self.np, self.n_elem),
                self.linear_speed,
                dtype=np.result_type(q.dtype, self.linear_speed),
            )
        elif self.diffeq_name == 'Burgers':
            nodal_speed = fn.cabs(q_nodes[:, 0, :])
        elif self.diffeq_name == 'Quasi1dEuler':
            rho = q_nodes[:, 0, :]
            momentum = q_nodes[:, 1, :]
            energy = q_nodes[:, 2, :]
            velocity = momentum / rho
            pressure = (self.solver.diffeq.g - 1.0) * (
                energy - 0.5 * momentum * momentum / rho
            )
            if np.any(np.real(rho) <= 0) or np.any(np.real(pressure) <= 0):
                raise ValueError(
                    'Entropy-DCP dissipation requires positive density and pressure.'
                )
            sound_speed = np.sqrt(self.solver.diffeq.g * pressure / rho)
            # max(|u+a|, |u-a|) = |u|+a for admissible real Euler states.
            # cabs preserves the derivative of the active fixed-sign branch.
            nodal_speed = fn.cabs(velocity) + sound_speed
        else:
            raise ValueError(
                "The one-dimensional 'entdcp' choices support linear "
                'convection, Burgers, and Quasi1dEuler.'
            )

        A0 = np.asarray(self.solver.diffeq.dqdw(q))
        expected_shape = (self.np, self.neq_node, self.neq_node, self.n_elem)
        if A0.shape != expected_shape:
            raise ValueError(
                f'dqdw(q) must return shape {expected_shape}; got {A0.shape}.'
            )

        # C_i A_0 is the nodal wave speed times dq/dw. Legacy entdcp averages
        # this product at half nodes for odd distribution order.
        coefficient = nodal_speed[:, None, None, :] * A0
        if self.distribution_s % 2 == 1 and self.entdcp_avg_half_nodes:
            coefficient = coefficient.copy()
            coefficient[:-1, :, :, :] = 0.5 * (
                coefficient[:-1, :, :, :] + coefficient[1:, :, :, :]
            )

        Ds = self.distribution_Ds_ref[0]
        derivative = (Ds @ w_nodes.reshape(self.np, -1)).reshape(w_nodes.shape)
        coefficient_derivative = np.matmul(
            coefficient.transpose(3, 0, 1, 2),
            derivative.transpose(2, 0, 1)[..., None],
        )[..., 0].transpose(1, 2, 0)
        coefficient_derivative *= self.H_ref[:, None, None]
        return (Ds.T @ coefficient_derivative.reshape(self.np, -1)).reshape(
            w_nodes.shape
        )

    def _matrix_conservative_distribution(self, q_nodes, q_bar):
        """Apply frozen contravariant absolute Jacobians at the cell average."""
        if self.dim == 1:
            # J d(xi)/dx = 1 in 1D, so the physical x Jacobian is already the
            # contravariant Jacobian required by the reference derivative.
            absolute_jacobians = np.asarray(
                self.solver.diffeq.dExdq_abs(q_bar)
            )
        else:
            # Stack the element-average state once per reference direction so
            # dEndq_abs forms every directional matrix in one vectorized call.
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

        distribution = np.zeros_like(q_nodes)
        for direction, Ds in enumerate(self.distribution_Ds_ref):
            derivative = (Ds @ q_nodes.reshape(self.np, -1)).reshape(
                q_nodes.shape
            )
            scaled_derivative = np.einsum(
                'abe,ibe->iae', absolute_jacobians[direction], derivative
            )
            scaled_derivative *= self.H_ref[:, None, None]
            distribution += (
                Ds.T @ scaled_derivative.reshape(self.np, -1)
            ).reshape(q_nodes.shape)

        return distribution

    def dissipation(self, q):
        """Return the strong-form element-local artificial dissipation."""
        q = np.asarray(q)
        if q.ndim != 2 or q.shape != self.qshape:
            raise ValueError(f'Expected q with shape {self.qshape}; got {q.shape}.')

        q_nodes = q.reshape(self.np, self.neq_node, self.n_elem)
        w = np.asarray(self.solver.diffeq.entropy_var(q))
        if w.shape != q.shape:
            raise ValueError(f'entropy_var(q) must return shape {q.shape}; got {w.shape}.')
        w_nodes = w.reshape(self.np, self.neq_node, self.n_elem)

        q_bar = None
        wave_speed = None
        if self.sensor_type == 'none':
            self.theta = np.ones(self.n_elem, dtype=q.dtype)
        else:
            q_bar = np.einsum('ie,ice->ce', self.H_phys, q_nodes)
            q_bar *= self.volume_inv[None, :]
            scaled_state, wave_speed = self._scaled_state_and_speed(q_nodes, q_bar)
            self.theta = self._sensor(scaled_state)

        if self.distribution_type == 'entdcp':
            distribution = self._entdcp_distribution(q, q_nodes, w_nodes)
        elif self.distribution_type == 'cons_mat':
            if q_bar is None:
                q_bar = np.einsum('ie,ice->ce', self.H_phys, q_nodes)
                q_bar *= self.volume_inv[None, :]
            distribution = self._matrix_conservative_distribution(q_nodes, q_bar)
        else:
            # cons_sca applies the same reference operator independently to
            # every conservative component.
            distribution = (
                self.distribution_M @ q_nodes.reshape(self.np, -1)
            ).reshape(q_nodes.shape)
        b = np.sum(w_nodes * distribution, axis=(0, 1))

        if self.budget_type == 'entdcp':
            # The unregularized normalized form then collapses to -kappa*a.
            bnu = b
        else:
            if q_bar is None:
                q_bar = np.einsum('ie,ice->ce', self.H_phys, q_nodes)
                q_bar *= self.volume_inv[None, :]
            A0_bar = np.asarray(self.solver.diffeq.dqdw(q_bar))
            if A0_bar.shape != self.A0_shape:
                raise ValueError(
                    f'dqdw(q_bar) must return shape {self.A0_shape}; '
                    f'got {A0_bar.shape}.'
                )
            if wave_speed is None:
                _, wave_speed = self._scaled_state_and_speed(q_nodes, q_bar)
            bnu = self._entropy_viscosity_budget(w_nodes, A0_bar, wave_speed)
        # Store the complete scalar multiplying the distribution vector so
        # drivers can inspect the amount selected in every element.
        self.element_coefficient = (
            -self.kappa
            * self.theta ** self.beta
            * bnu
            * b
            / (b * b + self.epsilon)
        )
        strong_dissipation = distribution * self.element_coefficient[None, None, :]
        strong_dissipation *= self.H_inv_phys[:, None, :]

        if not np.all(np.isfinite(strong_dissipation)):
            raise FloatingPointError('Entropy-budgeted dissipation produced nonfinite values.')
        return np.ascontiguousarray(strong_dissipation.reshape(q.shape))
