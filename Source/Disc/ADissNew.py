#!/usr/bin/env python3
"""Element-local, entropy-budgeted artificial volume dissipation.

The implementation follows the construction in the dissipation paper: build
a conservative weak distribution, contract it with the entropy variables,
assign a sensor-weighted entropy budget, normalize the distribution by its
entropy contraction, and finally convert the weak term to strong form.
"""

import numbers

import numpy as np

from Source.Disc.ADiss import ADiss
import Source.Methods.Functions as fn


def _kron_all(factors):
    """Return a Kronecker product in the solver's tensor-axis order."""
    product = factors[0]
    for factor in factors[1:]:
        product = np.kron(product, factor)
    return product


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

    # Subclasses can select a separate public configuration name while
    # reusing the common entropy-budgeted setup.
    required_diss_type = 'new'

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

        if self.type != self.required_diss_type:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"diss_type='{self.required_diss_type}'."
            )

        config = solver.vol_diss
        self.kappa = config.get('kappa', 1.0)
        self.beta = config.get('beta')
        self.sensor_s = config.get('sensor_s')
        self.distribution_s = config.get('distribution_s')
        self.store_diagnostics = config.get('store_diagnostics', False)

        for name in ('kappa', 'beta'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f'Entropy-budgeted dissipation: {name} must be provided '
                    'as a nonnegative real number.'
                )
        for name in ('sensor_s', 'distribution_s'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(
                    f'Entropy-budgeted dissipation: {name} must be provided as '
                    'a positive integer.'
                )
        if not isinstance(self.store_diagnostics, (bool, np.bool_)):
            raise ValueError(
                'Entropy-budgeted dissipation: store_diagnostics must be boolean.'
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
            if (
                not isinstance(value, str)
                or value.lower() not in supported_selectors[name]
            ):
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

        supported_equations = {
            'LinearConvection', 'Burgers', 'Quasi1dEuler', 'Euler2d'
        }
        self.diffeq_name = solver.diffeq.diffeq_name
        if self.diffeq_name not in supported_equations:
            raise ValueError(
                'Entropy-budgeted dissipation is currently implemented for '
                'LinearConvection, Burgers, Quasi1dEuler, and Euler2d.'
            )
        if (
            not hasattr(solver.diffeq, 'entropy_var')
            or not hasattr(solver.diffeq, 'dqdw')
        ):
            raise ValueError(
                'The differential equation must define entropy_var(q) and '
                'dqdw(q).'
            )

        self.epsilon = np.finfo(float).eps
        self.epsilon_theta = np.finfo(float).eps
        self.theta = None
        self.element_coefficient = None
        self.entropy_viscosity_budget = None
        self.entropy_budget = None
        self.entropy_contraction = None
        self.distribution_vector = None
        self.normalized_distribution_vector = None

        # Match the simple element-type defaults in legacy entdcp: include the
        # reference H weights and do not apply a boundary correction matrix.
        self.entdcp_use_H = True
        self.entdcp_bdy_fix = False
        self.entdcp_avg_half_nodes = True

        self._set_reference_operators()
        self._set_element_geometry()

    def _set_reference_operators(self):
        """Precompute the paper's reference derivatives and quadratic forms."""
        H = np.asarray(self.solver.sbp.H)
        if H.ndim != 2 or not np.allclose(H, np.diag(np.diag(H))):
            raise ValueError(
                'Entropy-budgeted dissipation requires a diagonal reference '
                'norm.'
            )

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

        if self.dim not in (1, 2, 3):
            raise ValueError('Entropy-budgeted dissipation supports dimensions 1, 2, and 3.')

        # Insert each 1D operator along one tensor-product direction. This is
        # the same ordering used by the solver's element-local state arrays.
        self.H_ref = _kron_all([h_1d] * self.dim)
        self.D_ref = tuple(
            _kron_all([
                D if axis == direction else eye
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        )
        self.sensor_Ds_ref = tuple(
            _kron_all([
                sensor_Ds if axis == direction else eye
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        )
        self.distribution_Ds_ref = tuple(
            _kron_all([
                distribution_Ds if axis == direction else eye
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        )

        # Sum one derivative quadratic form per reference direction.
        budget_forms = [
            _kron_all([
                M_1d if axis == direction else H
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        ]
        sensor_forms = [
            _kron_all([
                sensor_M_1d if axis == direction else H
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        ]
        distribution_forms = [
            _kron_all([
                distribution_M_1d if axis == direction else H
                for axis in range(self.dim)
            ])
            for direction in range(self.dim)
        ]
        self.M = sum(budget_forms[1:], start=budget_forms[0])
        self.sensor_M = sum(sensor_forms[1:], start=sensor_forms[0])
        self.distribution_M = sum(
            distribution_forms[1:], start=distribution_forms[0]
        )

        self.ref_measure = np.sum(self.H_ref)
        self.ref_measure_inv = 1.0 / self.ref_measure

        if np.any(self.H_ref <= 0):
            raise ValueError('Reference norm weights must be positive.')

        # All directional derivatives can be applied in one matrix product in
        # the entropy-viscosity budget.
        self.D_ref_stacked = np.concatenate(self.D_ref, axis=0)

        # Divide each Rayleigh quotient by its largest generalized eigenvalue.
        # Both eigenvalues depend only on the reference element and are static.
        H_inv_sqrt = 1.0 / np.sqrt(self.H_ref)
        roundoff_threshold = 100 * np.finfo(float).eps
        for name in ('sensor', 'distribution'):
            matrix = getattr(self, f'{name}_M')
            normalized = (
                H_inv_sqrt[:, None] * matrix * H_inv_sqrt[None, :]
            )
            normalized = 0.5 * (normalized + normalized.T)
            eigenvalue = np.linalg.eigvalsh(normalized)[-1]
            if not np.isfinite(eigenvalue):
                raise FloatingPointError(
                    f'The reference {name} eigenvalue is nonfinite.'
                )

            # Powers above the polynomial degree should vanish exactly. Clear
            # their roundoff remnants so constants stay in the exact nullspace.
            if eigenvalue <= roundoff_threshold:
                derivatives_name = f'{name}_Ds_ref'
                setattr(
                    self,
                    derivatives_name,
                    tuple(
                        np.zeros_like(operator)
                        for operator in getattr(self, derivatives_name)
                    ),
                )
                matrix.fill(0.0)
                eigenvalue = 0.0
            setattr(self, f'{name}_Lambda', eigenvalue)

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
        if (
            self.H_phys.shape != expected_shape
            or self.H_inv_phys.shape != expected_shape
        ):
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
                self._set_element_average_metrics()

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

    def _set_element_average_metrics(self):
        """Freeze every contravariant metric direction at its element mean."""
        if not hasattr(self.solver, 'mesh') or not hasattr(
            self.solver.mesh, 'metrics'
        ):
            raise ValueError(
                'Entropy-budgeted dissipation requires mesh metric terms in '
                'multiple dimensions.'
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

        # Mesh metrics are row-major J d(xi_alpha)/d(x_j). The reference-H
        # mean preserves the element-frozen approximation used by dEndq_abs.
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
        pressure = (self.solver.diffeq.g - 1.0) * (
            energy - 0.5 * rho * velocity_sq
        )
        if np.any(np.real(rho) <= 0) or np.any(np.real(pressure) <= 0):
            raise ValueError(
                'Entropy-budgeted Euler dissipation requires positive density '
                'and pressure.'
            )

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
            raise ValueError(
                'Element-average Euler states must have positive density and '
                'pressure.'
            )

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
        """Return the paper's strong-form element-local dissipation term."""
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

        # The conservative element mean supplies the component scaling, wave
        # speed, entropy Jacobian, and frozen matrix distribution when needed.
        q_bar = None
        wave_speed = None
        if self.sensor_type == 'none':
            self.theta = np.ones(self.n_elem, dtype=q.dtype)
        else:
            q_bar = np.einsum('ie,ice->ce', self.H_phys, q_nodes)
            q_bar *= self.volume_inv[None, :]
            scaled_state, wave_speed = self._scaled_state_and_speed(q_nodes, q_bar)
            self.theta = self._sensor(scaled_state)

        # Build the conservative weak distribution a. Every choice annihilates
        # constants; cons_sca is the paper's simple M-hat times u choice.
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
        # Its entropy contraction b = w^T a is the scalar normalization moment.
        b = np.sum(w_nodes * distribution, axis=(0, 1))

        # The cheap budget approximates entropy viscosity using the first
        # entropy-variable derivatives and a cell-average entropy Jacobian.
        if self.budget_type == 'entdcp':
            # The unregularized normalized form then collapses to -kappa*a.
            viscosity_budget = b
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
            viscosity_budget = self._entropy_viscosity_budget(
                w_nodes, A0_bar, wave_speed
            )
        # Activate the budget with the bounded smoothness sensor, then apply
        # the regularized entropy normalization b/(b^2 + epsilon).
        entropy_budget = (
            self.kappa * self.theta ** self.beta * viscosity_budget
        )
        normalization = b / (b * b + self.epsilon)
        # Keep the original operation order for diss_type='new'; the two
        # factored quantities above are diagnostic views of the same formula.
        self.element_coefficient = (
            -self.kappa
            * self.theta ** self.beta
            * viscosity_budget
            * b
            / (b * b + self.epsilon)
        )
        strong_dissipation = (
            distribution * self.element_coefficient[None, None, :]
        )
        # The construction above is a weak residual; H_k^{-1} converts it to
        # the strong element-local term returned to the PDE solver.
        strong_dissipation *= self.H_inv_phys[:, None, :]

        if self.store_diagnostics:
            # These arrays can be large, so production runs retain them only
            # when explicitly requested by a diagnostic driver.
            self.entropy_viscosity_budget = viscosity_budget.copy()
            self.entropy_budget = entropy_budget.copy()
            self.entropy_contraction = b.copy()
            self.distribution_vector = distribution.copy()
            self.normalized_distribution_vector = (
                distribution * normalization[None, None, :]
            )

        if not np.all(np.isfinite(strong_dissipation)):
            raise FloatingPointError(
                'Entropy-budgeted dissipation produced nonfinite values.'
            )
        return np.ascontiguousarray(strong_dissipation.reshape(q.shape))
