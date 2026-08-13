#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:16:51 2020

@author: bercik
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn

class LinearConv(PdeBase):
    '''
    Purpose
    ----------
    Variable-coefficient linear convection (time-independent coefficient):

        u_t + (a(x) u)_x = 0

    Exact solution (analytic characteristics)
    ----------
    Let q = a u. Then q_t + a(x) q_x = 0, so q is constant along dx/dt = a(x).

    Choose an analytic change of variables Phi(x) such that:
      - Phi(inflow_boundary) = 0
      - dPhi/dx = 1/a(x)
    Then along characteristics:
      - d/dt Phi(x(t)) = 1
      - Phi(x(t)) = Phi(x0) + t

    For periodic BCs:
      x0 = Phi^{-1}( wrap( Phi(x) - t ) )
      u(x,t) = a(x0) u0(x0) / a(x)

    For Dirichlet BCs:
      backtrace s = Phi(x) - t
      - if s >= 0: use initial condition (x0 = Phi^{-1}(s))
      - if s < 0 : characteristic hits inflow boundary at time tb = t - Phi(x)
                  use inflow boundary data u_in(tb)
                  u(x,t) = a(x_in)*u_in(tb) / a(x)
    '''

    diffeq_name = 'VariableCoefficientLinearConvection'
    dim = 1
    neq_node = 1
    npar = 0
    pde_order1 = True
    pde_order2 = False

    para_names = ('alpha',)
    use_exact_der = True
    extrapolate_bdy_flux = True

    # instance flags set in __init__
    has_exa_sol = False

    # analytic-map flags
    _phi_is_implemented = False
    _phi_period = None
    
    # interpolation for numerical Phi
    _phi_spline = None          # spline for Phi(x)
    _phi_inv_spline = None     # spline for Phi^{-1}(s)
    _phi_interp_order = 3       # interpolation order (default cubic)

    # inflow info determined by sign(a)
    _a_sign = None             # +1 or -1
    _inflow_side = None        # 'left' or 'right'
    _xin = None                # x-coordinate of inflow boundary (xmin or xmax)

    def __init__(self, para, q0_type='SinWave', a_type='Gaussian', 
                        flux_type='product', periodic=True):
        super().__init__(para, q0_type)
        self.alpha = self.para[0]
        self.a_type = a_type.lower()
        self.periodic = periodic
        self.flux_type = flux_type.lower()

        # Lazy initialization for skewed_sin coefficients (computed on first call)
        if self.a_type == 'skewed_sin' or self.a_type == 'skewed_sin_bigshift':
            # number of Fourier modes in the truncated series
            self._skewed_sin_n_fourier = 5
            # Will be (n_fourier,) arrays computed lazily
            self._skewed_sin_coeff = None       # coefficients for sin(k * xmod)
            self._skewed_sin_dcoeff = None      # corresponding derivative factors

        if self.flux_type == 'central' or self.flux_type == 'product':
            self.entropy = self.entropy_central
            self.entropy_var = self.entropy_var_central
            self.dqdw = self.dqdw_central
            self.maxeig_dqdw = self.maxeig_dqdw_central
            if self.flux_type == 'central' and self.alpha != 1.0:
                print('WARNING: alpha != 1.0 but selected flux_type == central. Overriding to 1.0.')
                self.alpha = 1.0
        elif self.flux_type == 'geometric':
            self.entropy = self.entropy_geom
            self.entropy_var = self.entropy_var_geom
            self.dqdw = self.dqdw_geom
            self.maxeig_dqdw = self.maxeig_dqdw_geom
        else:
            raise ValueError(f'Invalid flux_type: {self.flux_type}. Must be "central", "product", or "geometric".')

    def modx(self, x):
        " loops x around the periodic domain "
        return np.mod(x - self.xmin, self.dom_len) + self.xmin

    # ----------------------------
    # coefficient a(x) and a_x(x)
    # ----------------------------
    def afun(self, x):
        if self.a_type == 'gaussian' or self.a_type == 'gaussian_shift':
            mid_point = 0.5*(self.xmax + self.xmin)
            q0_max_q = self.q0_max_q/2
            k = (8*np.log(self.q0_gauss_wave_val_bc/q0_max_q))
            stdev2 = abs(self.dom_len**2/k)
            exp = -0.5*(x-mid_point)**2/stdev2
            a = q0_max_q * np.exp(exp)
            if 'shift' in self.a_type:
                a = a + 0.5

        elif self.a_type == 'constant':
            a = np.ones(np.shape(x))

        elif 'linear' in self.a_type:
            shift = 1.0
            xmod = (x - self.xmin) / self.dom_len
            if 'neg' in self.a_type:
                xmod = -xmod + 1
            if 'shift' in self.a_type:
                shift = 1.5
            elif '0' in self.a_type:
                shift = 0
            elif 'eps' in self.a_type:
                shift = 1e-8
            a = xmod + shift

        elif 'sinwave' in self.a_type or 'coswave' in self.a_type:
            if '4pi' in self.a_type:
                w = 4*np.pi
            elif '8pi' in self.a_type:
                w = 8*np.pi
            else:
                w = 2*np.pi
            xmod = (x - self.xmin) / self.dom_len
            if 'sinwave' in self.a_type:
                a = np.sin(w * xmod) * self.q0_max_q
            elif 'coswave' in self.a_type:
                a = np.cos(w * xmod) * self.q0_max_q
            if 'shift' in self.a_type:
                a = a + 1.5   

        elif self.a_type == 'skewed_sin' or self.a_type == 'skewed_sin_bigshift':
            # Lazy computation of coefficients on first call
            if self._skewed_sin_coeff is None:
                from scipy.special import comb
                n_fourier = self._skewed_sin_n_fourier
                binom_norm = comb(2 * n_fourier, n_fourier, exact=True)
                ks = np.arange(1, n_fourier + 1)
                # Use exact=False for array inputs (exact=True only works with scalars)
                comb_vals = comb(2 * n_fourier, n_fourier - ks, exact=False)
                self._skewed_sin_coeff = comb_vals / (binom_norm * ks)
                self._skewed_sin_dcoeff = comb_vals / binom_norm
            
            # Check if x is a scalar (Python float/int or numpy scalar)
            if np.isscalar(x):
                # Scalar case: return scalar
                xmod = 2.*np.pi*(x - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                kx = ks * xmod  # shape: (n_fourier,)
                sin_kx = np.sin(kx)  # shape: (n_fourier,)
                fourier_sum = np.dot(self._skewed_sin_coeff, sin_kx)  # scalar result
                if 'bigshift' in self.a_type:
                    a = fourier_sum + 10
                else:
                    a = float(fourier_sum + 1.5)
            else:
                # Array case: preserve shape
                xmod = 2.*np.pi*(x - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                ks_shape = (len(ks),) + (1,) * xmod.ndim
                kx = ks.reshape(ks_shape) * xmod  # shape: (n_fourier, ...)
                sin_kx = np.sin(kx)  # shape: (n_fourier, ...)
                # Multiply by coefficients and sum over k dimension (axis 0)
                # Use einsum to sum over first dimension while preserving other dimensions
                fourier_sum = np.einsum('i,i...->...', self._skewed_sin_coeff, sin_kx)
                if 'bigshift' in self.a_type:
                    a = fourier_sum + 10
                else:
                    a = fourier_sum + 1.5


        else:
            raise Exception('Variable coefficient not understood.')
        return a

    def afunder(self, x):
        if self.a_type == 'gaussian' or self.a_type == 'gaussian_shift':
            mid_point = 0.5*(self.xmax + self.xmin)
            q0_max_q = self.q0_max_q/2
            k = (8*np.log(self.q0_gauss_wave_val_bc/q0_max_q))
            stdev2 = abs(self.dom_len**2/k)
            exp = -0.5*(x-mid_point)**2/stdev2
            ader = - q0_max_q * np.exp(exp) * (x-mid_point) / stdev2

        elif self.a_type == 'constant':
            ader = np.zeros(np.shape(x))

        elif 'linear' in self.a_type:
            ader = np.ones(np.shape(x))/(self.xmin-self.xmax)

        elif 'sinwave' in self.a_type or 'coswave' in self.a_type:
            if '4pi' in self.a_type:
                w = 4*np.pi
            elif '8pi' in self.a_type:
                w = 8*np.pi
            else:
                w = 2*np.pi
            xmod = (x - self.xmin) / self.dom_len
            if 'sinwave' in self.a_type:
                ader = w * np.cos(w * xmod) * self.q0_max_q / self.dom_len
            elif 'coswave' in self.a_type:
                ader = - w * np.sin(w * xmod) * self.q0_max_q / self.dom_len

        elif self.a_type == 'skewed_sin' or self.a_type == 'skewed_sin_bigshift':
            # Lazy computation of coefficients on first call (if not already computed by afun)
            if self._skewed_sin_dcoeff is None:
                from scipy.special import comb
                n_fourier = self._skewed_sin_n_fourier
                binom_norm = comb(2 * n_fourier, n_fourier, exact=True)
                ks = np.arange(1, n_fourier + 1)
                # Use exact=False for array inputs (exact=True only works with scalars)
                comb_vals = comb(2 * n_fourier, n_fourier - ks, exact=False)
                self._skewed_sin_coeff = comb_vals / (binom_norm * ks)
                self._skewed_sin_dcoeff = comb_vals / binom_norm
            
            # Check if x is a scalar (Python float/int or numpy scalar)
            if np.isscalar(x):
                # Scalar case: return scalar
                xmod = 2.*np.pi*(x - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                kx = ks * xmod  # shape: (n_fourier,)
                cos_kx = np.cos(kx)  # shape: (n_fourier,)
                dfourier_dx = 2.0*np.pi*np.dot(self._skewed_sin_dcoeff, cos_kx) / self.dom_len  # scalar result
                ader = float(dfourier_dx)
            else:
                # Array case: preserve shape
                xmod = 2.*np.pi*(x - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                ks_shape = (len(ks),) + (1,) * xmod.ndim
                kx = ks.reshape(ks_shape) * xmod  # shape: (n_fourier, ...)
                cos_kx = np.cos(kx)  # shape: (n_fourier, ...)
                # Multiply by derivative coefficients and sum over k dimension, then divide by dom_len
                # Use einsum to sum over first dimension while preserving other dimensions
                dfourier_dx = 2.0*np.pi*np.einsum('i,i...->...', self._skewed_sin_dcoeff, cos_kx) / self.dom_len
                ader = dfourier_dx

        else:
            raise Exception('Variable coefficient not understood.')
        return ader

    def afunder2(self, x):
        if self.a_type == 'gaussian' or self.a_type == 'gaussian_shift':
            mid_point = 0.5 * (self.xmax + self.xmin)
            q0_max_q = self.q0_max_q / 2
            k = (8 * np.log(self.q0_gauss_wave_val_bc / q0_max_q))
            stdev2 = abs(self.dom_len**2 / k)
            exp = -0.5 * (x - mid_point)**2 / stdev2
            a = q0_max_q * np.exp(exp)
            ader2 = a * (((x - mid_point)**2) / (stdev2**2) - 1.0 / stdev2)

        elif self.a_type == 'constant':
            ader2 = np.zeros(np.shape(x))

        elif 'linear' in self.a_type:
            ader2 = np.zeros(np.shape(x))

        elif 'sinwave' in self.a_type or 'coswave' in self.a_type:
            if '4pi' in self.a_type:
                w = 4 * np.pi
            elif '8pi' in self.a_type:
                w = 8 * np.pi
            else:
                w = 2 * np.pi
            xmod = (x - self.xmin) / self.dom_len
            if 'sinwave' in self.a_type:
                ader2 = -(w**2) * np.sin(w * xmod) * self.q0_max_q / (self.dom_len**2)

            elif 'coswave' in self.a_type:
                ader2 = -(w**2) * np.cos(w * xmod) * self.q0_max_q / (self.dom_len**2)

        elif self.a_type == 'skewed_sin' or self.a_type == 'skewed_sin_bigshift':
            # Lazy computation of coefficients if needed
            if self._skewed_sin_coeff is None or self._skewed_sin_dcoeff is None:
                from scipy.special import comb
                n_fourier = self._skewed_sin_n_fourier
                binom_norm = comb(2 * n_fourier, n_fourier, exact=True)
                ks = np.arange(1, n_fourier + 1)
                comb_vals = comb(2 * n_fourier, n_fourier - ks, exact=False)
                self._skewed_sin_coeff = comb_vals / (binom_norm * ks)
                self._skewed_sin_dcoeff = comb_vals / binom_norm  # = k*c_k

            if np.isscalar(x):
                xmod = 2.0 * np.pi * (x - self.xmin) / self.dom_len + 4.0
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                kx = ks * xmod
                sin_kx = np.sin(kx)
                k2c = ks * self._skewed_sin_dcoeff
                sum_term = np.dot(k2c, sin_kx)
                ader2 = - (2.0 * np.pi / self.dom_len)**2 * float(sum_term)
            else:
                xmod = 2.0 * np.pi * (x - self.xmin) / self.dom_len + 4.0
                ks = np.arange(1, self._skewed_sin_n_fourier + 1)
                ks_shape = (len(ks),) + (1,) * xmod.ndim
                kx = ks.reshape(ks_shape) * xmod
                sin_kx = np.sin(kx)
                k2c = ks * self._skewed_sin_dcoeff  # shape (n_fourier,)
                sum_term = np.einsum('i,i...->...', k2c, sin_kx)
                ader2 = - (2.0 * np.pi / self.dom_len)**2 * sum_term

        else:
            raise Exception('Variable coefficient not understood.')
        return ader2

    # ----------------------------
    # exact-solution admissibility
    # ----------------------------
    def _a_has_strict_sign(self):
        """
        Returns True iff a(x) is guaranteed to stay strictly positive or strictly negative
        on [xmin, xmax], based on a_type, without numerical sampling if possible.
        """
        if self.a_type == 'constant':
            return True

        elif self.a_type == 'gaussian_shift':
            return True

        elif self.a_type == 'gaussian':
            return False

        elif 'linear' in self.a_type:
            aL = float(self.afun(self.xmin))
            aR = float(self.afun(self.xmax))
            amin = min(aL, aR)
            amax = max(aL, aR)
            if amin > 0.0:
                return True
            if amax < 0.0:
                return True
            return False

        elif 'sinwave' in self.a_type or 'coswave' in self.a_type:
            if 'shift' in self.a_type:
                return True
            else:
                return False

        elif self.a_type == 'skewed_sin':
            return True

        else:
            # Numerical check: sample a(x) on the domain
            # This is a fallback for cases not handled analytically above
            tolerance = 1e-4
            num_samples = 1000  # Sample at many points to check for sign changes
            x_samples = np.linspace(self.xmin, self.xmax, num_samples)
            a_samples = self.afun(x_samples)
            
            # Check if any value is close to zero (within tolerance)
            near_zero = np.any(np.abs(a_samples) < tolerance)
            
            # Check for sign change
            has_sign_change = np.any(a_samples > 0) and np.any(a_samples < 0)
            
            # If no sign change and no near-zero values, assume strict sign
            if not has_sign_change and not near_zero:
                print("WARNING: Using numerical check for a(x) sign. This may not be completely accurate.")
                return True
            else:
                return False

    def _set_inflow_info(self):
        """
        Determine inflow boundary from the sign of a. Assumes strict sign.
        If a>0 -> inflow at xmin. If a<0 -> inflow at xmax.
        """
        # for strict-sign coefficients, checking one point is sufficient
        amid = float(self.afun(0.5*(self.xmin + self.xmax)))
        if amid > 0.0:
            self._a_sign = +1
            self._inflow_side = 'left'
            self._xin = self.xmin
        elif amid < 0.0:
            self._a_sign = -1
            self._inflow_side = 'right'
            self._xin = self.xmax
        else:
            raise Exception('a(x) appears to hit 0; exact solution should be disabled.')

    # ----------------------------
    # analytic Phi interface (placeholders for later)
    # ----------------------------
    def _Phi(self, x):
        """
        Phi(x) satisfying:
        - Phi(xinflow) = 0
        - dPhi/dx = 1/a(x)
        """
        x = np.array(x, copy=False)

        # ---------- Analytic cases ----------
        if self.a_type == 'constant':
            # a = 1
            return x - self._xin

        if 'linear' in self.a_type:
            # Your a(x): a = xmod + shift OR a = (1-xmod)+shift for 'neg'
            L = self.dom_len
            xmod = (x - self.xmin) / L

            # Determine shift exactly as in your afun
            shift = 1.0
            if 'shift' in self.a_type:
                shift = 1.5
            elif '0' in self.a_type:
                shift = 0.0
            elif 'eps' in self.a_type:
                shift = 1e-8

            # Determine whether neg flips xmod
            neg = ('neg' in self.a_type)

            # We anchor Phi at inflow boundary x_in = self._xin
            # For your strict-sign linear cases you're typically positive so inflow is xmin,
            # but this works generally as long as self._xin is set correctly.

            # Compute Phi in terms of xmod and xmod_in
            x_in = float(self._xin)
            xmod_in = (x_in - self.xmin) / L

            if not neg:
                # a = xmod + shift
                # Phi(x) = L * ln((xmod+shift)/(xmod_in+shift))
                return L * np.log((xmod + shift) / (xmod_in + shift))
            else:
                # a = (1 - xmod) + shift = 1 + shift - xmod
                # Integral gives Phi = L * ln( (1+shift-xmod_in)/(1+shift-xmod) )
                return L * np.log((1.0 + shift - xmod_in) / (1.0 + shift - xmod))

        if ('sinwave' in self.a_type) or ('coswave' in self.a_type):
            # Only strictly sign if 'shift' in type (per your _a_has_strict_sign)
            # a(x) = c + A*sin(theta) or c + A*cos(theta), theta = w*(x-xmin)/L
            if 'shift' not in self.a_type:
                # fall back numerical
                return self._Phi_numeric(x)

            # parameters exactly as your afun
            if '4pi' in self.a_type:
                w = 4*np.pi
            elif '8pi' in self.a_type:
                w = 8*np.pi
            else:
                w = 2*np.pi

            L = self.dom_len
            xmod = (x - self.xmin) / L
            theta = w * xmod

            c = 1.5
            A = float(self.q0_max_q)

            # strict positivity requires c > |A|
            if not (c > abs(A)):
                # This is ill-conditioned / can cross zero. Use numerical builder only if you insist;
                # but your _a_has_strict_sign says True only when shifted, not checking magnitude.
                raise Exception(f'sin/cos shifted coefficient is not strictly positive: c={c}, A={A}')

            gamma = np.sqrt(c*c - A*A)

            # helper: I(theta) = ∫ dtheta / (c + A sin theta)
            # I(theta) = (2/gamma) * arctan((c*t + A)/gamma), t = tan(theta/2)
            # for cos, use sin(theta+pi/2)
            if 'coswave' in self.a_type:
                theta = theta + 0.5*np.pi

            t = np.tan(0.5*theta)
            I = (2.0/gamma) * np.arctan((c*t + A)/gamma)

            # anchor at inflow boundary self._xin
            x_in = float(self._xin)
            theta_in = w * ((x_in - self.xmin)/L)
            if 'coswave' in self.a_type:
                theta_in = theta_in + 0.5*np.pi
            t_in = np.tan(0.5*theta_in)
            I_in = (2.0/gamma) * np.arctan((c*t_in + A)/gamma)

            # Phi = (L/w) * (I - I_in)
            return (L / w) * (I - I_in)

        # ---------- Numerical fallback ----------
        return self._Phi_numeric(x)


    def _Phi_inv(self, s):
        """
        Inverse x = Phi^{-1}(s).
        - For periodic, s will be wrapped into [0, phi_period)
        - For Dirichlet, s will be in [0, phi_out] for points traced to initial data
        """
        s = np.array(s, copy=False)

        # ---------- Analytic cases ----------
        if self.a_type == 'constant':
            return self._xin + s

        if 'linear' in self.a_type:
            L = self.dom_len

            # same shift logic as afun
            shift = 1.0
            if 'shift' in self.a_type:
                shift = 1.5
            elif '0' in self.a_type:
                shift = 0.0
            elif 'eps' in self.a_type:
                shift = 1e-8

            neg = ('neg' in self.a_type)

            x_in = float(self._xin)
            xmod_in = (x_in - self.xmin) / L

            if not neg:
                # Phi = L ln((xmod+shift)/(xmod_in+shift))
                # => xmod = (xmod_in+shift)*exp(Phi/L) - shift
                xmod = (xmod_in + shift) * np.exp(s / L) - shift
            else:
                # Phi = L ln((1+shift-xmod_in)/(1+shift-xmod))
                # => 1+shift-xmod = (1+shift-xmod_in)*exp(-Phi/L)
                xmod = (1.0 + shift) - (1.0 + shift - xmod_in) * np.exp(-s / L)

            return self.xmin + L * xmod

        if ('sinwave' in self.a_type) or ('coswave' in self.a_type):
            if 'shift' not in self.a_type:
                return self._Phi_inv_numeric(s)

            if '4pi' in self.a_type:
                w = 4*np.pi
            elif '8pi' in self.a_type:
                w = 8*np.pi
            else:
                w = 2*np.pi

            L = self.dom_len
            c = 1.5
            A = float(self.q0_max_q)

            if not (c > abs(A)):
                raise Exception(f'sin/cos shifted coefficient is not strictly positive: c={c}, A={A}')

            gamma = np.sqrt(c*c - A*A)

            # same anchoring constant I_in
            x_in = float(self._xin)
            theta_in = w * ((x_in - self.xmin)/L)
            if 'coswave' in self.a_type:
                theta_in = theta_in + 0.5*np.pi

            t_in = np.tan(0.5*theta_in)
            I_in = (2.0/gamma) * np.arctan((c*t_in + A)/gamma)

            # We have: Phi = (L/w)*(I(theta)-I_in) => I(theta) = I_in + (w/L)*Phi
            I = I_in + (w / L) * s

            # I(theta) = (2/gamma) arctan((c*t + A)/gamma), t=tan(theta/2)
            # Let alpha = (gamma/2)*I => tan(alpha) = (c*t + A)/gamma => t = (gamma*tan(alpha)-A)/c
            alpha = 0.5 * gamma * I
            tan_alpha = np.tan(alpha)
            t = (gamma * tan_alpha - A) / c

            theta = 2.0 * np.arctan(t)

            # undo phase shift for cos
            if 'coswave' in self.a_type:
                theta = theta - 0.5*np.pi

            # map theta -> x, with theta = w*(x-xmin)/L
            x = self.xmin + (L / w) * theta

            # for safety, wrap into base domain (Phi^{-1} should land in domain, but branch cuts can add 2π)
            if self.periodic:
                x = self.modx(x)
            else:
                # keep in bounds (Dirichlet)
                x = np.clip(x, self.xmin, self.xmax)

            return x

        # ---------- Numerical fallback ----------
        return self._Phi_inv_numeric(s)


    def _wrap_phi(self, s):
        if self._phi_period is None:
            # If we are in an analytic case, define period cheaply.
            # Otherwise build numeric maps (sets period).
            self._ensure_phi_ready()

        return np.mod(s, self._phi_period)

    # ------------------------------------------------------------
    # Numerical fallback: high-order Gauss–Legendre, refinement check
    # ------------------------------------------------------------

    def _ensure_phi_ready(self, tol=1e-12, max_refine=8):
        """
        Ensure that we have a numerical Phi table and a consistent phi_period.
        Used only for non-analytic cases.

        Builds Phi on a uniform mesh and refines until the self-consistency
        estimate is < tol.
        """
        # analytic cases set phi_period directly without building tables
        if self._phi_is_implemented and (self._phi_period is not None):
            return

        # If analytic case but period not set, set it here:
        if self.a_type == 'constant':
            self._phi_period = abs(self.xmax - self.xmin)
            self._phi_is_implemented = True
            return

        if 'linear' in self.a_type:
            L = self.dom_len
            # same shift logic
            shift = 1.0
            if 'shift' in self.a_type:
                shift = 1.5
            elif '0' in self.a_type:
                shift = 0.0
            elif 'eps' in self.a_type:
                shift = 1e-8
            neg = ('neg' in self.a_type)

            x_in = float(self._xin)
            xmod_in = (x_in - self.xmin) / L

            # "outflow" boundary is the other end
            x_out = self.xmax if abs(x_in - self.xmin) < 1e-15 else self.xmin
            xmod_out = (x_out - self.xmin) / L

            if not neg:
                self._phi_period = L * np.log((xmod_out + shift) / (xmod_in + shift))
            else:
                self._phi_period = L * np.log((1.0 + shift - xmod_in) / (1.0 + shift - xmod_out))
            self._phi_is_implemented = True
            return

        if ('sinwave' in self.a_type) or ('coswave' in self.a_type):
            if 'shift' in self.a_type:
                c = 1.5
                A = float(self.q0_max_q)
                if not (c > abs(A)):
                    raise Exception(f'sin/cos shifted coefficient is not strictly positive: c={c}, A={A}')
                gamma = np.sqrt(c*c - A*A)
                self._phi_period = self.dom_len / gamma
                self._phi_is_implemented = True
                return

        # Otherwise, numeric fallback:
        self._build_phi_numeric(tol=tol, max_refine=max_refine)

    def _phi_selfcheck(self, tol=1e-12, ntest=80):
        """
        Sanity check that Phi^{-1}(Phi(x)) ≈ x to within tol.
        This is only meant for the numerical fallback maps.
        Cost is tiny compared to building Phi with quadrature.
        """
        # pick test points that are NOT exactly the grid nodes (use midpoints)
        xg = self._phi_x
        if xg.size < 3:
            return

        # choose roughly ntest midpoints spread across the domain
        idx = np.linspace(0, xg.size - 2, min(ntest, xg.size - 1)).astype(int)
        x_test = 0.5 * (xg[idx] + xg[idx + 1])

        s_test = self._Phi_numeric(x_test)          # forward via table
        x_back = self._Phi_inv_numeric(s_test)      # inverse via table + safeguarded Newton

        err = np.max(np.abs(x_back - x_test))
        if not np.isfinite(err):
            raise Exception('Phi self-check produced non-finite error.')

        if err > tol:
            raise Exception(
                f'Phi self-check failed: max|Phi_inv(Phi(x))-x| = {err} > tol={tol}. '
                f'Increase quadrature/mesh refinement.'
            )

    def _build_phi_splines(self):
        """
        Build splines for Phi(x) and Phi^{-1}(s) using current table and interpolation order.
        """
        self._phi_spline = UnivariateSpline(self._phi_x, self._phi_s, k=self._phi_interp_order, s=0)
        self._phi_inv_spline = UnivariateSpline(self._phi_s, self._phi_x, k=self._phi_interp_order, s=0)
    
    def _check_interpolation_error(self, ntest=100):
        """
        Check interpolation error at random points (not grid nodes).
        Computes 'true' Phi values using direct quadrature and compares with spline interpolation.
        Returns maximum error.
        """
        if self._phi_spline is None:
            return 0.0  # No spline yet, no error
        
        # Generate random test points (not at grid nodes)
        xmin_test = self._phi_x[0]
        xmax_test = self._phi_x[-1]
        # Use random points with small offset to avoid grid nodes
        np.random.seed(42)  # For reproducibility
        x_test = np.random.uniform(xmin_test, xmax_test, ntest)
        # Add small random offset to avoid exact grid alignment
        dx = (xmax_test - xmin_test) / len(self._phi_x)
        x_test = x_test + np.random.uniform(-dx/4, dx/4, ntest)
        x_test = np.clip(x_test, xmin_test, xmax_test)
        
        # Evaluate using current spline
        Phi_interp = self._phi_spline(x_test)
        
        # Evaluate "ground truth" using direct quadrature from xin to each test point
        Phi_true = np.zeros_like(x_test)
        n_gl = 8
        xi, wi = np.polynomial.legendre.leggauss(n_gl)
        xin = float(self._xin)
        
        def cell_integral(a, b):
            """Compute integral of 1/a(x) from a to b"""
            xm = 0.5*(a+b)
            xr = 0.5*(b-a)
            xq = xm + xr*xi
            return xr * np.sum(wi / self.afun(xq))
        
        # For each test point, find nearest grid point and compute Phi using table + quadrature
        for i, xt in enumerate(x_test):
            # Find grid index
            idx = np.searchsorted(self._phi_x, xt, side='right')
            idx = max(0, min(idx - 1, len(self._phi_x) - 2))
            
            # Use table value at grid point, then add quadrature to test point
            x_grid = self._phi_x[idx]
            Phi_grid = self._phi_s[idx]
            
            # Add quadrature from grid point to test point
            if abs(xt - x_grid) > 1e-14:
                Phi_true[i] = Phi_grid + cell_integral(x_grid, xt)
            else:
                Phi_true[i] = Phi_grid
        
        # Compute error
        err = np.abs(Phi_interp - Phi_true)
        return np.max(err)
    
    def _build_phi_numeric(self, tol=1e-12, max_refine=8):
        """
        Build Phi(x) table on [xin, xout] with Gauss-Legendre per cell.
        Store:
        self._phi_x  (monotone grid in x)
        self._phi_s  (Phi values)
        self._phi_period
        """
        # choose interval orientation so Phi increases with index
        xin = float(self._xin)
        xout = self.xmax if abs(xin - self.xmin) < 1e-15 else self.xmin

        # ensure increasing x-grid for the table
        if xout < xin:
            xin, xout = xout, xin
            flipped = True
        else:
            flipped = False

        # start mesh
        Nx0 = 200  # baseline; refine if needed
        n_gl = 8   # Gauss-Legendre nodes per cell

        # precompute GL nodes/weights on [-1,1]
        xi, wi = np.polynomial.legendre.leggauss(n_gl)

        def cell_integral(a, b):
            # map [-1,1] -> [a,b]
            xm = 0.5*(a+b)
            xr = 0.5*(b-a)
            xq = xm + xr*xi
            return xr * np.sum(wi / self.afun(xq))

        def build(Nx):
            xg = np.linspace(xin, xout, Nx+1)
            Phi = np.zeros_like(xg)
            for k in range(Nx):
                Phi[k+1] = Phi[k] + cell_integral(xg[k], xg[k+1])
            return xg, Phi

        xg, Phi = build(Nx0)
        for r in range(max_refine):
            xg2, Phi2 = build(2*(Nx0*(2**r)))
            # compare Phi on coarse nodes: Phi2 at every other node should match Phi
            # Need Phi built on same xin->xout; it is.
            Phi2_ds = Phi2[::2]
            err = np.max(np.abs(Phi2_ds - Phi))
            if err < tol:
                # accept fine solution (use the finer one for better accuracy)
                xg, Phi = xg2, Phi2
                break
            # refine: set coarse to the fine-downsampled and continue
            xg, Phi = xg2, Phi2
        else:
            raise Exception(f'Numerical Phi builder did not reach tol={tol}. Final err={err}')

        # store
        self._phi_x = xg if not flipped else xg[::-1]
        self._phi_s = Phi if not flipped else (Phi[-1] - Phi[::-1])  # keep Phi increasing with x
        self._phi_period = float(self._phi_s[-1])
        
        # Build initial splines with default cubic order
        self._phi_interp_order = 3
        self._build_phi_splines()
        
        # Adaptive refinement: check interpolation error and refine if needed
        interp_tol = tol
        max_interp_order = 7
        max_interp_refine = 3  # max iterations of interpolation refinement
        
        for interp_iter in range(max_interp_refine):
            interp_err = self._check_interpolation_error(ntest=100)
            if interp_err < interp_tol:
                break
            
            # Try higher-order interpolation first (cheap)
            if self._phi_interp_order < max_interp_order:
                self._phi_interp_order += 2  # cubic -> quintic -> 7th
                self._build_phi_splines()
            else:
                # Higher-order didn't help, refine grid
                # Double the grid and rebuild
                Nx_current = len(xg) - 1
                Nx_new = 2 * Nx_current
                xg, Phi = build(Nx_new)
                # Update stored arrays
                self._phi_x = xg if not flipped else xg[::-1]
                self._phi_s = Phi if not flipped else (Phi[-1] - Phi[::-1])
                self._phi_period = float(self._phi_s[-1])
                # Reset to cubic and rebuild splines
                self._phi_interp_order = 3
                self._build_phi_splines()
        
        self._phi_is_implemented = True

        # --- cheap self-check: verify inverse consistency on a small test set ---
        self._phi_selfcheck(tol=tol)


    def _Phi_numeric(self, x):
        """
        Evaluate Phi(x) using the numeric table built by _build_phi_numeric.
        Uses spline interpolation (cubic by default, adaptively refined if needed).
        """
        self._ensure_phi_ready()

        x = np.array(x, copy=False)
        # clamp into domain ends used by table
        x0 = self._phi_x[0]
        x1 = self._phi_x[-1]
        xc = np.clip(x, min(x0, x1), max(x0, x1))

        # Use spline interpolation
        if self._phi_spline is None:
            # Fallback to linear if spline not built yet
            return np.interp(xc, self._phi_x, self._phi_s)
        return self._phi_spline(xc)


    def _Phi_inv_numeric(self, s):
        """
        Invert Phi(x)=s using the numeric table and a bracketed Newton/bisection.
        Uses:
        - initial guess from spline interpolation of inverse table
        - refine with safeguarded Newton using exact derivative Phi'(x)=1/a(x)
        """
        self._ensure_phi_ready()

        s = np.array(s, copy=False)
        # clamp s into [0, period]
        s = np.clip(s, 0.0, self._phi_period)

        # initial guess from spline interpolation of inverse table
        if self._phi_inv_spline is None:
            # Fallback to linear if spline not built yet
            x = np.interp(s, self._phi_s, self._phi_x)
        else:
            x = self._phi_inv_spline(s)

        # bracket from nearest table indices
        idx = np.searchsorted(self._phi_s, s, side='left')
        idx = np.clip(idx, 1, self._phi_s.size - 1)

        x_lo = self._phi_x[idx - 1]
        x_hi = self._phi_x[idx]

        # Newton with safeguard
        tol = 5e-13
        maxit = 40
        for _ in range(maxit):
            F = self._Phi_numeric(x) - s
            if np.max(np.abs(F)) < tol:
                break
            a = self.afun(x)
            # Phi'(x)=1/a => Newton step x <- x - F / Phi' = x - F * a
            x_new = x - F * a

            # safeguard: if step escapes bracket, use bisection
            out = (x_new <= x_lo) | (x_new >= x_hi) | ~np.isfinite(x_new)
            if np.any(out):
                x_new[out] = 0.5 * (x_lo[out] + x_hi[out])

            # update bracket using monotonicity of Phi
            Phi_new = self._Phi_numeric(x_new)
            left = Phi_new < s
            right = ~left
            x_lo[left] = x_new[left]
            x_hi[right] = x_new[right]
            x = x_new
        else:
            raise Exception('Phi^{-1} numeric inversion did not converge.')

        return x

    # ----------------------------
    # boundary hooks for Dirichlet (TODO)
    # ----------------------------
    def u_inflow(self, t):
        """
        Dirichlet inflow boundary data u(xin, t).
        User will implement later.

        Notes:
          - If a>0, xin = xmin.
          - If a<0, xin = xmax.
        """
        raise NotImplementedError('Implement inflow Dirichlet boundary condition u_inflow(t).')

    # ----------------------------
    # exact solution
    # ----------------------------
    def exact_sol(self, time=0, x=None, **kwargs):
        """
        Exact solution via analytic characteristics. Works for:
          - periodic BCs (wrap in x and Phi-space)
          - Dirichlet BCs (inflow boundary hook)

        Requires:
          - has_exa_sol == True (a has strict sign)
          - analytic Phi, Phi^{-1}, and (for periodic) Phi_period are implemented
        """
        if not self.has_exa_sol:
            raise Exception(
                f'Exact solution is disabled for a_type="{self.a_type}" because a(x) '
                f'is not guaranteed to have a strict sign on the domain (it may cross or touch 0).'
            )

        if not self._phi_is_implemented:
            raise NotImplementedError(
                f'Exact solution requires analytic Phi/Phi^{-1} (and Phi_period for periodic) '
                f'for a_type="{self.a_type}". These maps are not implemented yet (by design).'
            )

        if x is None:
            x = self.x_elem

        reshape = False
        if np.ndim(x) > 1:
            reshape = True
            orig_shape = x.shape
            x = np.array(x).flatten('F')
        else:
            x = np.array(x, copy=False)

        if time == 0:
            # time zero: just initial condition on the domain
            # (if periodic, modx is fine; if Dirichlet, x should already be inside)
            xm = self.modx(x) if self.periodic else x
            u = self.set_q0(xy=xm)
            if reshape:
                u = np.reshape(u, orig_shape, 'F')
            return u

        # Periodic: reduce x to base period. Dirichlet: keep x as-is.
        xm = self.modx(x) if self.periodic else x

        # Phi-coordinate of x
        phi_x = self._Phi(xm)

        if self.periodic:
            # periodic backtrace in Phi-space
            s = self._wrap_phi(phi_x - time)
            x0 = self._Phi_inv(s)
            u = self.afun(x0) * self.set_q0(xy=x0) / self.afun(xm)

        else:
            # Dirichlet: backtrace without wrap
            # s = Phi(x0) = Phi(x) - t
            s = phi_x - time

            # If s >= 0 -> footpoint lies on initial line inside domain (by construction Phi(inflow)=0)
            # If s < 0  -> characteristic hits inflow boundary first at time tb = t - Phi(x)
            u = np.empty_like(xm, dtype=float)

            inside = (s >= 0.0)
            if np.any(inside):
                x0 = self._Phi_inv(s[inside])
                u[inside] = self.afun(x0) * self.set_q0(xy=x0) / self.afun(xm[inside])

            if np.any(~inside):
                tb = time - phi_x[~inside]  # time when char hits inflow boundary (Phi=0)
                uin = self.u_inflow(tb)     # Dirichlet u at inflow boundary
                ain = float(self.afun(self._xin))
                u[~inside] = ain * uin / self.afun(xm[~inside])

        if reshape:
            u = np.reshape(u, orig_shape, 'F')

        return u

    def dfdq(self, q):
        # TODO
        
        return None

    def calcEx(self, q):

        E = self.a * q
        return E
    
    def dExdx(self, q, E=None):
        ''' Overwrites default divergence form with a potentially split form '''
        
        if E is None: E = self.calcEx(q)

        if self.flux_type == 'product':
            if self.use_exact_der:      
                dExdx = self.alpha * self.gm_gv(self.Dx, E) + \
                    (1 - self.alpha) * ( self.a * self.gm_gv(self.Dx, q) + q * self.ader )
            else:
                dExdx = self.alpha * self.gm_gv(self.Dx, E) + \
                    (1 - self.alpha) * ( self.a * self.gm_gv(self.Dx, q) + \
                                            q * self.gm_gv(self.Dx, self.a) )

        elif self.flux_type == 'central':
            dExdx = self.gm_gv(self.Dx, E)

        elif self.flux_type == 'geometric':
            sqrt_E = np.sqrt(E) # Could do a positivity check here, but time marching should catch it.
            dExdx = 2 * sqrt_E * self.gm_gv(self.Dx, sqrt_E)
        
        else:
            raise Exception(f'Invalid flux type: {self.flux_type}. Must be "product" or "geometric".')
        return dExdx

    def dExdq(self, q):
        dExdq = np.reshape(self.a, (self.nen,1,1,self.nelem))
        return dExdq
    
    def d2Exdq2(self, q):

        d2Exdq2 = np.zeros(q.shape)
        return d2Exdq2
    
    def dExdq_abs(self, q, entropy_fix=False):
        if q.shape == self.qshape:
            dExdq = np.reshape(np.abs(self.a), (self.nen,1,1,self.nelem))
        else:
            qshape = q.shape
            dExdq = np.ones((qshape[0],1,1,qshape[1]))*self.max_absa
        return dExdq
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        if q.shape == self.qshape:
            return np.abs(self.a)
        else:
            # Note: Because I don't have access to x (usually q is actually a q_facet)
            # I can not do a local LF. Instead I return an overly dissipative global LF
            return np.ones(q.shape)*self.max_absa

    def calc_LF_const(self,xy,use_local=False):
        ''' Constant for the Lax-Friedrichs flux'''
        if use_local:
            c = np.max(abs(self.afun(xy)),axis=0)
        else:
            c = np.max(abs(self.afun(xy)))
        return c
    
    def a_energy(self,q):
        ''' compute the global A-norm SBP energy of global solution vector q '''
        return np.tensordot(q, self.a * self.H * q)
    
    def a_energy_der(self,q,dqdt):
        ''' compute the global A-norm SBP energy derivatve of global solution vector q '''
        return 2 * np.tensordot(q, self.a * self.H * dqdt)
    
    def a_cons(self,q):
        ''' compute the global A-conservation SBP of global solution vector q '''
        return np.sum(self.a * self.H * q)

    def entropy_central(self, q):
        ''' nodal values of the entropy for the central flux '''
        return q*self.calcEx(q)
    
    def entropy_var_central(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return self.calcEx(q)

    def dqdw_central(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(1/self.a)
    
    def maxeig_dqdw_central(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return 1/np.abs(self.a)

    def entropy_geom(self, q):
        ''' nodal values of the entropy for the central flux '''
        return -2*np.sqrt(q/self.a)
    
    def entropy_var_geom(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return -1/np.sqrt(q*self.a)

    def dqdw_geom(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(2*q*np.sqrt(self.a*q))
    
    def maxeig_dqdw_geom(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return 2*np.abs(q*np.sqrt(self.a*q))
    
    def set_mesh(self, mesh, H):
        '''
        Purpose
        ----------
        Needed to calculate the initial solution and to calculate source terms,
        must overwrite base case so we can set the variable coefficient
        '''

        self.mesh = mesh
        PdeBase.set_mesh(self, mesh, H)

        self.a = self.afun(self.x_elem)
        self.max_absa = np.max(np.abs(self.a))
        self.ader = self.afunder(self.x_elem)
        try:
            self.ader2 = self.afunder2(self.x_elem)
        except:
            pass
        if np.min(self.a) <= 0.: print('WARNING: Variable coefficient a(x) should be >=0')

        # Determine if a has strict sign (no crossing/touching 0)
        self.has_exa_sol = self._a_has_strict_sign()

        # Determine inflow side from sign of a (only meaningful if has_exa_sol)
        if self.has_exa_sol:
            self._set_inflow_info()
            self._ensure_phi_ready()
        else:
            self._a_sign = None
            self._inflow_side = None
            self._xin = None
        











