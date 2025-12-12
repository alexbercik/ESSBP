#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Sep 2019

@author: bercik
"""

import numpy as np
from numpy.polynomial import legendre as Leg
from numpy.polynomial import polynomial as Poly


class BasisFun:

    def __init__(self, x, nb, p, basis_type="legendre"):
        '''
        Parameters
        ----------
        x : np array
            A default set of nodes that will serve for nodal evaluations.
        n : int
            Number of basis functions in the basis.
        p : int
            Polynomial degree of the basis.
        basis_type : str, optional
            Indicates the type of basis function used.
            The default is "legendre".
        '''

        self.x = x
        self.p = p
        assert len(x.shape) == 1, f"x must be a 1D array, but given {x.shape}"
        self.nn = len(x) # number of nodes
        self.nb = nb # number of basis functions (cardinality)
        if self.nn != self.nb:
            # I do not have full cardinality! 
            print(f"WARNING: Number of nodes does not match number of basis functions, {self.nn} != {self.nb}")
        self.basis_type = basis_type.lower()

        ''' Initialize the specific basis class from which we inherit through __getattr__ '''
        if self.basis_type == "monomial":
            self._basis = MonomialBasis(x, self.p)
        elif self.basis_type == 'legendre':
            self._basis = LegendreBasis(x, self.p)
        elif self.basis_type == "exponential":
            self._basis = ExponentialBasis(x, self.nb)
        else:
            raise NotImplementedError(f"Unknown basis_type {basis_type!r}")

        # Precompute Vandermonde at the default nodes and its inverse if full cardinality
        self._basis._V = self._build_vandermonde(self.x)  # modal -> nodal
        if self.nb == self.nn:
            self._basis._Vinv = np.linalg.inv(self._V)    # nodal -> modal
        #else:
        #    # could probably use a projection instead. But this really doesn't matter much.
        #    self._basis._Vinv = np.linalg.pinv(self._V)
        self._basis._Vx = self._build_vandermonde_derivative(self.x) 

    def __getattr__(self, name):
        """
        Delegate attribute/method lookup to the underlying _basis object
        when not found on BasisFun itself. This allows us to directly access
        the methods and attributes of the underlying _basis object.
        """
        return getattr(self._basis, name)

    # ---------- Public Vandermonde interfaces ----------

    def vandermonde(self, x):
        r"""
        Evaluate the orthonormal basis functions at x.
        A wrapper for _build_vandermonde to handle both scalar and array inputs.
        """
        x_arr = np.asarray(x, dtype=float)
        scalar_input = (x_arr.ndim == 0)

        V = self._build_vandermonde(x_arr)

        if scalar_input:
            return V[0, :]   # shape (nb,)
        return V             # shape (m, nb)

    def vandermonde_derivative(self, x):
        r"""
        Evaluate the derivatives of theorthonormal basis functions at x.
        A wrapper for _build_vandermonde_derivative to handle both scalar and array inputs.
        """
        x_arr = np.asarray(x, dtype=float)
        scalar_input = (x_arr.ndim == 0)

        Vx = self._build_vandermonde_derivative(x_arr)

        if scalar_input:
            return Vx[0, :]   # shape (nb,)
        return Vx             # shape (m, nb)

    # ---------- Nodal <-> Modal transforms ----------

    def nodal_to_modal(self, u, neq_node=1, dim=1, reshape=True):
        r"""
        Transform from nodal (values at self.x) to modal (orthonormal Legendre) coefficients.

        Parameters
        ----------
        u : array_like
            Nodal values. First dimension must have length nn.
            Examples:
                shape (nn,)       -> single field
                shape (nn, k)     -> k fields, each defined at the nn nodes

        Returns
        -------
        a : ndarray
            Modal coefficients with same leading shape as `u`,
            but first dimension still nb.
        """
        u = np.asarray(u, dtype=float)
        if u.shape[0] != (self.nn**dim)*neq_node:
            raise ValueError(f"nodal_to_modal: first axis must have length (nn**dim)*neq_node={(self.nn**dim)*neq_node}, got {u.shape[0]}.")

        # prepare the Vandermonde matrix
        if self.nb == self.nn: # will use stored inverse
            if dim == 1: Vinv = self._Vinv
            elif dim == 2: Vinv = np.kron(self._Vinv,self._Vinv)
            elif dim == 3: Vinv = np.kron(self._Vinv,np.kron(self._Vinv,self._Vinv))
        else: # will solve the linear system with least squares
            if dim == 1: V = self._V
            elif dim == 2: V = np.kron(self._V,self._V)
            elif dim == 3: V = np.kron(self._V,np.kron(self._V,self._V))
        
        if neq_node > 1:
            u = u.reshape(self.nn**dim,neq_node,*u.shape[1:])

        if self.nb == self.nn:
            # a = V^{-1} u
            a =  Vinv @ u
        else:
            # TODO: could probably do some projection, but who cares. This doesn't matter too much.
            a = np.linalg.lstsq(V, u, rcond=None)[0]
        
        if neq_node > 1 and reshape:
            a = a.reshape((self.nb**dim)*neq_node, *u.shape[2:])
        
        return a

    def modal_to_nodal(self, a, neq_node=1, dim=1):
        r"""
        Transform from modal coefficients to nodal values at self.x.

        Parameters
        ----------
        a : array_like
            Modal coefficients. First dimension must have length nb.

        Returns
        -------
        u : ndarray
            Nodal values with same leading shape as `a`,
            but first dimension still nn.
        """
        a = np.asarray(a, dtype=float)
        if a.shape[0] != (self.nb**dim)*neq_node:
            raise ValueError(f"modal_to_nodal: first axis must have length (nb**dim)*neq_node={(self.nb**dim)*neq_node}, got {a.shape[0]}.")

        # prepare the Vandermonde matrix
        if dim == 1: V = self._V
        elif dim == 2: V = np.kron(self._V,self._V)
        elif dim == 3: V = np.kron(self._V,np.kron(self._V,self._V))

        if neq_node > 1:
            a = a.reshape(self.nb**dim,neq_node,*a.shape[1:])

        # u = V a
        u =  V @ a
        
        if neq_node > 1:
            u = u.reshape((self.nn**dim)*neq_node, *a.shape[2:])
        
        return u

    def eval_nodal_vec(self, u, x_eval, neq_node=1, dim=1):
        r"""
        Evaluate a function given in nodal form (values at self.x) at new evaluation points.

        Parameters
        ----------
        u : array_like
            Nodal coefficients. Last dimension must match len(self.x).
        x_eval : float or array_like
            Points in [0,1] to evaluate the function at.

        Returns
        -------
        out : ndarray
            Values of the function at x_eval.
        """
        # Step 1: nodal -> modal
        a = self.nodal_to_modal(u, neq_node=neq_node, dim=dim, reshape=False)

        # Step 2: modal -> evaluate at x_eval
        x_eval_arr = np.asarray(x_eval, float)

        # prepare the Vandermonde matrix
        V_eval = self._build_vandermonde(x_eval_arr)
        if dim == 1: pass
        elif dim == 2: V_eval = np.kron(V_eval,V_eval)
        elif dim == 3: V_eval = np.kron(V_eval,np.kron(V_eval,V_eval))

        # u = V a
        u_eval =  V_eval @ a
        
        if neq_node > 1:
            u_eval = u_eval.reshape((len(x_eval_arr)**dim)*neq_node, *a.shape[2:])
            
        return u_eval

    
    @staticmethod
    def cardinality(dim, p):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        p : int or numpy array of int
            Polynomial degree(s) of an operator

        Returns
        -------
        n_p : int or numpy array of int
            The cardinality of the set of basis functions of polynomial degree p.
        '''

        if dim == 0:
            n_p = 1
        elif dim == 1:
            n_p = p + 1
        elif dim == 2:
            n_p = (p + 1) * (p + 2) // 2
        elif dim == 3:
            n_p = (p + 1) * (p + 2) * (p + 3) // 6
        else:
            raise Exception('Only 0 to 3 dimensions accepted')

        if type(n_p) is np.ndarray:
            n_p = n_p.astype(int)
        else:
            n_p = int(n_p)

        return n_p






class LegendreBasis:
    r"""
    Orthonormal polynomial basis on [0, 1] built from shifted Legendre polynomials.

    Modal basis: phi_n(x) = sqrt(2n+1) * P_n(2x - 1),   n = 0,...,nb-1
    where P_n is the standard Legendre polynomial on [-1, 1].

    Nodal representation: values at the user-supplied nodes self.x in [0,1].
    """

    def __init__(self, x, p):
        r"""
        Parameters
        ----------
        x_nodes : array_like
            An important set of nodes that will always serve as the default for
            nodal evaluations in [0, 1]; defines nb (the number of basis functions).
        nb : int
            Total number of basis functions. Must satisfy nb >= 2.
        """
        self.x = np.asarray(x, dtype=float)
        self.nb = int(p+1)
        assert self.nb >= 2, "Legendre Basis requires nb >= 2."

        # Normalization factors for orthonormal basis on [0,1]
        # phi_n(x) = sqrt(2n+1) * P_n(2x - 1)
        self._norms = np.sqrt(2 * np.arange(self.nb) + 1.0)

    # ---------- Core internal builders ----------

    def _build_vandermonde(self, x):
        r"""
        Internal helper: evaluate the orthonormal basis at arbitrary x.

        For x with shape (nn,), returns V with shape (nn, nb):
            V[i, n] = phi_n(x[i]).
        """
        x = np.atleast_1d(x)
        xi = 2.0 * x - 1.0                  # map [0,1] -> [-1,1]
        # V_leg[i, n] = P_n(xi[i]) 
        V_unscaled = Leg.legvander(xi, self.nb - 1)
        # Scale columns to get orthonormal basis on [0,1]
        V = V_unscaled * self._norms # broadcast over columns
        return V

    def _build_vandermonde_derivative(self, x):
        r"""
        Internal helper: evaluate the derivatives of the orthonormal basis at arbitrary x.

        For x with shape (m,), returns Vx with shape (m, nb):
            Vx[i, n] = d/dx phi_n(x[i]).
        """
        x = np.atleast_1d(x)
        xi = 2.0 * x - 1.0
        m = x.size
        Vx = np.empty((m, self.nb), dtype=float)

        # For each n, differentiate P_n(ξ), evaluate at ξ = 2x-1,
        # then apply chain rule d/dx = 2 d/dξ and normalization.
        for n in range(self.nb):
            # coefficients for P_n(ξ): c = [0,...,0,1]
            c = np.zeros(n + 1)
            c[n] = 1.0
            dc = Leg.legder(c)  # derivative w.r.t. ξ
            # d/dx P_n(2x-1) = 2 * P_n'(2x-1)
            Vx[:, n] = 2.0 * Leg.legval(xi, dc) * self._norms[n]

        return Vx




class MonomialBasis:
    r"""
    Polynomial (monomial) basis on [0, 1] in the power basis.

    Modal basis: phi_n(x) = x^n,  n = 0,...,nb-1

    Nodal representation: values at the user-supplied nodes self.x in [0,1].
    """

    def __init__(self, x, p):
        r"""
        Parameters
        ----------
        x : array_like
            An important set of nodes that will always serve as the default for
            nodal evaluations in [0, 1]; defines nb (the number of basis functions).
        nb : int
            Total number of basis functions. Must satisfy nb >= 2.
        """
        self.x = np.asarray(x, dtype=float)
        self.nb = int(p+1)
        assert self.nb >= 2, "Monomial Basis requires nb >= 2."

    # ---------- Core internal builders ----------

    def _build_vandermonde(self, x):
        r"""
        Internal helper: evaluate the monomial basis at arbitrary x.

        For x with shape (nn,), returns V with shape (nn, nb):
            V[i, n] = x[i]**n.
        """
        x = np.atleast_1d(x)
        # Poly.polyvander(x, deg) returns columns [1, x, x^2, ..., x^deg]
        V = Poly.polyvander(x, self.nb - 1)   # shape (nn, nb)
        return V

    def _build_vandermonde_derivative(self, x):
        r"""
        Internal helper: evaluate the derivatives of the monomial basis at arbitrary x.

        For x with shape (m,), returns Vx with shape (m, nb):
            Vx[i, n] = d/dx (x^n) evaluated at x[i] = n * x[i]^(n-1),
            with the convention that the derivative of x^0 = 1 is 0.
        """
        x = np.atleast_1d(x)
        m = x.size
        Vx = np.empty((m, self.nb), dtype=float)

        # Column n=0 corresponds to derivative of x^0 = 1 => 0
        Vx[:, 0] = 0.0

        if self.nb > 1:
            # We need x^(n-1) for n=1,...,nb-1 => powers up to nb-2
            Xpow = Poly.polyvander(x, self.nb - 2)     # shape (m, nb-1), columns x^0,...,x^(nb-2)
            for n in range(1, self.nb):
                # d/dx x^n = n * x^(n-1)
                Vx[:, n] = n * Xpow[:, n - 1]

        return Vx




class ExponentialBasis:
    """
    Mixed basis on [0, 1] consisting of:
      - the first nb-1 orthonormal Legendre polynomials, and
      - an exponential-type function as the last basis vector:
          * either plain exp(x), or
          * its orthogonal complement to the Legendre subspace.

    The Legendre part is the same orthonormal basis as in LegendreBasis:
        phi_n(x) = sqrt(2n+1) * P_n(2x - 1),  n = 0,...,nb-2
    """

    def __init__(self, x, nb, orthogonalize_exp=True, n_quad=None):
        """
        Parameters
        ----------
        x : array_like
            Default nodal locations in [0, 1]; length must be nb.
        nb : int
            Total number of basis functions. Must satisfy nb >= 2.
        orthogonalize_exp : bool, optional
            If False, the last basis function is plain exp(x).
            If True, the last basis function is the orthogonal complement
            of exp(x) with respect to the first nb-1 orthonormal Legendre
            basis functions (in L2([0,1])).
        n_quad : int or None, optional
            Number of Gauss-Legendre quadrature points used to compute the
            projection coefficients for exp(x) when orthogonalize_exp=True.
            If None, we use an exact representation of exp(x) using a
            Legendre-coefficient series (i.e. no quadrature).
        """
        self.x = np.asarray(x, dtype=float)
        self.nb = int(nb)
        assert self.nb >= 2, "ExponentialBasis requires nb >= 2."

        # Legendre part uses degrees 0,...,nb-2 -> nb_leg basis functions
        self.nb_leg = self.nb - 1
        self.orthogonalize_exp = bool(orthogonalize_exp)

        # Orthonormalization factors for the Legendre part
        # phi_n(x) = sqrt(2n+1) * P_n(2x - 1),  n = 0,...,nb_leg-1
        self._leg_norms = np.sqrt(2 * np.arange(self.nb_leg) + 1.0)

        # Optional: coefficients of the projection of exp(x) onto the Legendre part
        # a_n = <exp, phi_n>, n=0,...,nb_leg-1
        if self.orthogonalize_exp:
            if n_quad is None:
                # Projection coefficients of e^x onto phi_n:
                # from the Fourier–Legendre expansion of e^{itx}, one can derive:
                # e^x = sum_n b_n phi_n(x) + higher modes,
                # with  b_n = exp(1/2) * sqrt(2n+1) * i_n(1/2),
                # where i_n is the modified spherical Bessel of the first kind.
                from scipy.special import spherical_in
                beta = 0.5
                orders = np.arange(self.nb_leg)
                # spherical_in(n, z) works with scalar z, vector n
                i_vals = spherical_in(orders, beta)
                self._exp_coeffs = np.exp(0.5) * self._leg_norms * i_vals

            else:
                # use a quadrature rule to compute the orthogonal coefficients
                assert n_quad > 0, "n_quad must be positive when orthogonalize_exp=True"

                # Gauss-Legendre quadrature on [-1,1], then map to [0,1]
                xi, wi = Leg.leggauss(n_quad)
                xq = 0.5 * (xi + 1.0)
                wq = 0.5 * wi

                # Evaluate Legendre basis at quadrature points
                V_leg_q = self._build_leg_vandermonde(xq)  # shape (n_quad, nb_leg)
                f_q = np.exp(xq)

                # Inner products <exp, phi_n> ≈ sum_i w_i f(x_i) phi_n(x_i)
                self._exp_coeffs = V_leg_q.T @ (wq * f_q)   # shape (nb_leg,)
        else:
            self._exp_coeffs = None

    # ---------- Internal Legendre-only builders (same as LegendreBasis) ----------

    def _build_leg_vandermonde(self, x):
        """
        Evaluate the orthonormal Legendre basis (nb_leg functions) at x.

        Returns a matrix V_leg with shape (m, nb_leg), V_leg[i, n] = phi_n(x[i]).
        """
        x = np.atleast_1d(x)
        xi = 2.0 * x - 1.0
        V_unscaled = Leg.legvander(xi, self.nb_leg - 1)  # P_n(2x-1), n=0,...,nb_leg-1
        return V_unscaled * self._leg_norms              # orthonormal basis

    def _build_leg_vandermonde_derivative(self, x):
        """
        Derivatives of the orthonormal Legendre basis (nb_leg functions) at x.

        Returns Vx_leg with shape (m, nb_leg), Vx_leg[i, n] = d/dx phi_n(x[i]).
        """
        x = np.atleast_1d(x)
        xi = 2.0 * x - 1.0
        m = x.size
        Vx = np.empty((m, self.nb_leg), dtype=float)

        for n in range(self.nb_leg):
            # P_n(ξ) has coefficients [0,...,0,1] in Legendre basis
            c = np.zeros(n + 1)
            c[n] = 1.0
            dc = Leg.legder(c)  # derivative w.r.t. ξ
            # d/dx P_n(2x-1) = 2 * P_n'(2x-1)
            Vx[:, n] = 2.0 * Leg.legval(xi, dc) * self._leg_norms[n]

        return Vx

    # ---------- Full mixed-basis builders (Legendre + exp / orthogonalized exp) ----------

    def _build_vandermonde(self, x):
        """
        Internal helper: full Vandermonde of the mixed basis at arbitrary x.

        For x with shape (m,), returns V with shape (m, nb):
          - columns 0..nb-2: orthonormal Legendre basis
          - column nb-1    : exp(x) or its orthogonal complement
        """
        x = np.atleast_1d(x)
        V_leg = self._build_leg_vandermonde(x)  # (m, nb_leg)

        # Exponential (or orthogonalized exponential) column
        exp_col = np.exp(x)
        if self.orthogonalize_exp and self._exp_coeffs is not None:
            # psi(x) = exp(x) - sum_n a_n phi_n(x)
            exp_col = exp_col - V_leg @ self._exp_coeffs

        V = np.column_stack([V_leg, exp_col])  # (m, nb_leg+1=nb)
        return V

    def _build_vandermonde_derivative(self, x):
        """
        Internal helper: derivatives of the mixed basis at arbitrary x.

        For x with shape (m,), returns Vx with shape (m, nb).
        """
        x = np.atleast_1d(x)
        Vx_leg = self._build_leg_vandermonde_derivative(x)  # (m, nb_leg)

        # derivative of exp(x) or of the orthogonalized version
        exp_col = np.exp(x)
        if self.orthogonalize_exp and self._exp_coeffs is not None:
            # d/dx psi(x) = exp(x) - sum_n a_n d/dx phi_n(x)
            exp_col = exp_col - Vx_leg @ self._exp_coeffs

        Vx = np.column_stack([Vx_leg, exp_col])
        return Vx

