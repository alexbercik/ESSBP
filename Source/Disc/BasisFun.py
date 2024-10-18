#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Sep 2019

@author: bercik
"""

import numpy as np
from Source.Disc.RefSimplexElem import RefSimplexElem


class BasisFun:

    def __init__(self, xy, p, basis_type="monomial"):
        '''
        Parameters
        ----------
        xy : np array
            Each row has the coord. for one node.
        p : int
            Degree of the basis function.
        basis_type : str, optional
            Indicates the type of basis function used.
            The default is "monomial".

        Returns
        -------
        None.

        '''


        ''' Add inputs to the class '''

        self.xy = xy
        self.p = p
        self.basis_type = basis_type.lower()

        ''' Extract required data '''

        self.nn, self.dim = np.shape(self.xy)
        self.n_p = self.cardinality(self.dim, self.p)

        ''' Build the Vandermonde matrix and its derivative '''

        if self.basis_type == "monomial":
            self.van, self.van_der, self.indices = self.basis_monomial(self.xy, self.p)
        elif self.basis_type == 'legendre':
            self.van, self.van_der, self.indices = self.basis_legendre(self.xy, self.p)
        elif self.basis_type == "lagrange":
            raise Exception("Error, not yet available")
            # self.van, self.van_der, self.indices = self.basis_lagrange(self.xy, self.p)
        else:
            raise Exception("Error, not yet available")

        self.idx_basis_dn1 = self.basis_idx_dn1(self.dim, self.p)


    @staticmethod
    def cardinality(dim, p):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        p : int or numpy array of int
            Degree(s) of an operator

        Returns
        -------
        n_p : int or numpy array of int
            The cardinality of the set of basis functions of degree p.
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

    @staticmethod
    def inv_cardinality(dim, n_p):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        n_p : int or numpy array of int
            The cardinality of the set of basis functions of degree p.

        Returns
        -------
        p : int
            Degree of an operator.
        '''

        if dim == 1:
            p = n_p - 1
        elif dim == 2:
            p = int(np.round((-3 + np.sqrt(1 + 8*n_p))/2))
        else:
            raise Exception('Only available for 1 and 2D')

        return p

    @staticmethod
    def basis_function(xy, p, basis='monomial'):
        '''
        Parameters
        ----------
        xy : numpy array
            Cartesian coordinates for nodes with one row per node.
        p : int
            Max degree for the basis function.
        basis : string, optional
            Type of basis function to use. The default is 'monomial'.

        Returns
        -------
        van : numpy array
            Vandermonde matrix evaluated at xy and up to degree p.
        van_der : numpy array
            Equivalent to the Vandermonde matrix but it holds the derivatives
            of the basis functions.
        indices : numpy array of int
            The 2D array is of size [dim x n_p]. Each col is for one basis
            function and each row is for one dimension. For monomial and
            Legendre basis functions each index indicates the number of the
            basis function in a respective direction
            e.g. for a 2D p=2 Vandermonde matrix the array would be
                [[0,1,0,2,1,0], [0,0,1,0,1,2]] for the basis: 1,x,y,x**2,xy,y**2
        '''

        # Convert basis to all lowercase
        basis = basis.lower()

        if basis == 'monomial':
            van, van_der, indices = BasisFun.basis_monomial(xy, p)
        elif basis == 'legendre':
            van, van_der, indices = BasisFun.basis_legendre(xy, p)
        elif basis == 'lagrange':
            raise Exception('The requested basis type is not available')
            # van, van_der, indices = BasisFun.basis_lagrange(xy, p)
        else:
            raise Exception('The requested basis type is not available')

        return van, van_der, indices

    @staticmethod
    def basis_monomial(xy, p):
        ''' See the function basis_function '''

        # Common data
        [nn, dim] = np.shape(xy)
        n_p = BasisFun.cardinality(dim, p)

        # Initiate matrices
        van = np.zeros((nn, n_p))
        van_der = np.zeros((dim, nn, n_p))
        indices = np.zeros((dim, n_p), dtype=int)

        if dim == 1:
            d = 0
            van[:,0] = 1
            van_der[d,:,0] = 0

            for i in range(1, n_p):
                van[:,i] = xy[:,0]**i
                van_der[d,:,i] = i * xy[:,0]**(i - 1)
                indices[0,i] = i

        elif dim == 2:

            van[:, 0] = 1

            x = xy[:, 0]
            y = xy[:, 1]

            for i in range(0, p+1):
                for j in range(0, i+1):
                    idx = i * (i + 1) // 2 + j
                    a = i - j
                    b = j
                    indices[:, idx] = np.array([a, b])
                    van[:, idx] = x**a * y**b

                    if a == 0:
                        van_der[0,:,idx] = 0
                    else:
                        van_der[0,:,idx] = a * x**(a - 1) * y**b

                    if b == 0:
                        van_der[1,:,idx] = 0
                    else:
                        van_der[1,:,idx] = b * x**a * y**(b - 1)

        elif dim == 3:

            van[:, 0] = 1

            x = xy[:, 0]
            y = xy[:, 1]
            z = xy[:, 2]

            for i in range(0, p + 1):
                for j in range(0, i + 1):
                    for k in range(0, j + 1):

                        idx = k + j * (j + 1) // 2 + i * (i + 1) * (i + 2) // 6
                        a = i - j
                        b = j - k
                        c = k
                        indices[:, idx] = np.array([a, b, c])

                        van[:, idx] = x**a * y**b * z**c

                        if a == 0:
                            van_der[0,:,idx] = 0
                        else:
                            van_der[0,:,idx] = a * x**(a - 1) * y**b * z**c

                        if b == 0:
                            van_der[1,:,idx] = 0
                        else:
                            van_der[1,:,idx] = b * x**a * y**(b - 1) * z**c

                        if c == 0:
                            van_der[2,:,idx] = 0
                        else:
                            van_der[2,:,idx] = c * x**a * y**b * z**(c - 1)
        else:
            raise Exception('Only 1 to 3 dimensions accepted')

        return van, van_der, indices

    @staticmethod
    def basis_legendre(xy, p):
        ''' See the function basis_function '''

        # Common data
        dim = xy.shape[1]

        if dim == 1:
            van, van_der, indices = BasisFun.basis_legendre_1d(xy[:, 0], p)
        elif dim == 2:
            van, van_der, indices = BasisFun.basis_legendre_2d(xy, p)
        elif dim == 3:
            van, van_der, indices = BasisFun.basis_legendre_3d(xy, p)
        else:
            raise Exception('Only 1 to 3 dimensions accepted')

        return van, van_der, indices

    @staticmethod
    def basis_legendre_1d(xy, p):
        ''' See the function basis_function '''

        dim = 1
        nn = xy.shape[0]
        n_p = BasisFun.cardinality(dim, p)

        leg_1d = np.zeros((nn, n_p))
        leg_1d_der = np.zeros((dim, nn, n_p))

        indices = np.arange(0, n_p, dtype=int)
        indices = indices[None, :]

        leg_1d[:, 0] = 1
        leg_1d_der[0,:,0] = 0

        if p >= 1:
            leg_1d[:, 1] = 2*xy - 1
            leg_1d_der[0,:,1] = 2

        for i in range(2, p + 1):
            n = i - 1
            c1 = (2 * n + 1) / (n + 1)
            c2 = n / (n + 1)

            leg_1d[:,n+1] = c1 * (2*xy-1) * leg_1d[:,n] - c2 * leg_1d[:,n-1]
            leg_1d_der[0,:,n+1] = c1 * (2 * leg_1d[:,n] + (2*xy-1) * leg_1d_der[0,:,n]) - c2 * leg_1d_der[0,:,n-1]
        return leg_1d, leg_1d_der, indices


    @staticmethod
    def basis_legendre_2d(xy, p):
        ''' See the function basis_function '''

        dim = 2
        nn = xy.shape[0]
        n_p = BasisFun.cardinality(dim, p)

        van = np.zeros((nn, n_p))
        van_der = np.zeros((dim, nn, n_p))
        indices = np.zeros((dim, n_p), dtype=int)

        # Calculate 1D Legendre shape functions
        leg_1d_x, leg_der_1d_x, _ = BasisFun.basis_legendre_1d(xy[:,0], p)
        leg_1d_y, leg_der_1d_y, _ = BasisFun.basis_legendre_1d(xy[:,1], p)

        # Build two dimensional basis
        for i in range(0, p + 1):
            for j in range(0, i + 1):
                idx = i * (i + 1)// 2 + j
                a = i - j   # x ^ a
                b = j       # y ^ b
                indices[:, idx] = np.array([a, b])

                van[:, idx] = leg_1d_x[:,a] * leg_1d_y[:,b]
                van_der[0,:,idx] = leg_der_1d_x[0,:,a] * leg_1d_y[:,b]
                van_der[1,:,idx] = leg_1d_x[:,a] * leg_der_1d_y[0,:,b]

        return van, van_der, indices

    @staticmethod
    def basis_legendre_3d(xy, p):
        ''' See the function basis_function '''

        dim = 3
        nn = xy.shape[0]
        n_p = BasisFun.cardinality(dim, p)

        van = np.zeros((nn, n_p))
        van_der = np.zeros((dim, nn, n_p))
        indices = np.zeros((dim, n_p), dtype=int)

        # Calculate 1D Legendre shape functions
        leg_1d_x, leg_der_1d_x, _ = BasisFun.basis_legendre_1d(xy[:,0], p)
        leg_1d_y, leg_der_1d_y, _ = BasisFun.basis_legendre_1d(xy[:,1], p)
        leg_1d_z, leg_der_1d_z, _ = BasisFun.basis_legendre_1d(xy[:,2], p)

        # Build two dimensional basis
        for i in range(0, p + 1):
            for j in range(0, i + 1):
                for k in range(0, j+1):
                    idx = k + j * (j + 1)// 2 + i * (i + 1) * (i + 2)// 6
                    a = i - j
                    b = j - k
                    c = k
                    indices[:, idx] = np.array([a, b, c])

                    van[:, idx] = leg_1d_x[:, a] * leg_1d_y[:, b] * leg_1d_z[:, c]
                    van_der[0,:,idx] = leg_der_1d_x[0,:,a] * leg_1d_y[:, b] * leg_1d_z[:, c]
                    van_der[1,:,idx] = leg_1d_x[:, a] * leg_der_1d_y[0,:,b] * leg_1d_z[:, c]
                    van_der[2,:,idx] = leg_1d_x[:, a] * leg_1d_y[:, b] * leg_der_1d_z[0,:,c]

        return van, van_der, indices

    @staticmethod
    def basis_idx_dn1(dim, p):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        p : int
            Degree of the operator.

        Returns
        -------
        idx_vec : numpy array
            Indicates the indices for the basis that need to be considered if
            the nodes are on a d-1 hyperdimensional plane. This is helpful to
            calculate the SBP operator rr for the Rdn1 family
        '''

        # Get the indices for basis functions
        xy = np.zeros((1, dim))
        _, _, indices = BasisFun.basis_monomial(xy, p)

        # Get the indices for basis function
        idx_dim_d = indices[-1,:]

        # Identify basis functions with no component on the d-th dimension
        idx_vec = np.argwhere(idx_dim_d == 0)
        idx_vec = idx_vec[:,0]

        return idx_vec

    @staticmethod
    def shp_fun(p, xquad, xint=np.empty([0]), basis='monomial'):

        nn, dim = xquad.shape

        if nn == 1 and dim == 1 and xquad[0, 0] == 0:
            shp = np.array([[1]])
            shpx = np.array([[0]])
            return shp, shpx

        n_p = BasisFun.cardinality(dim, p)

        if xint.size == 0:
            xint = RefSimplexElem.ref_vertices(dim)

        inv_coeff, _, _ = BasisFun.basis_function(xint, p, basis)
        psi, psix, _ = BasisFun.basis_function(xquad, p, basis)

        assert inv_coeff.shape[0] == inv_coeff.shape[1], \
            "The array inv_coeff is not square"

        shp = np.linalg.solve(inv_coeff.T, psi.T).T
        # shp = psi / inv_coeff

        shpx = np.zeros((dim, nn, n_p))

        for d in range(0, dim):
            shpx[d,:,:] = np.linalg.solve(inv_coeff.T, psix[d,:,:].T).T
            # shpx[d,:,:] = psix[d,:, :] / inv_coeff

        return shp, shpx

    @staticmethod
    def facet_vandermonde(xqf, p, gen_all_facets=True, basis="monomial"):
        '''
        Parameters
        ----------
        xqf : numpy array
            Cartesian coordinates for the facet quad rule
        p : int
            Max degree of the basis function for the Vandermonde matrix.
        gen_all_facets : bool, optional
            Indicates if the Vandermonde matrix should be constructed for the
            facet quad rule on each facet. If set to False then only the quad
            rule for the zero-th facet is evaluated.
            The default is True.
        basis : string, optional
            Type of basis function to use. The default is "monomial".

        Returns
        -------
        xfe : numpy array
            Cartesian coord. of the facet quad rule on the 0-th facet.
        van_f : numpy array
            Vandermonde matrix evaluated at the facet quad rule on the 0-th
            facet.
        van_f_der : numpy array
            Equivalent to the Vandermonde matrix but instead it is the
            derivative of the basis functions.
        '''

        nfn, dim_f = np.shape(xqf)

        if nfn == 1 and dim_f == 1 and xqf[0, 0] == 0:
            dim_f = 0

        dim = dim_f + 1
        n_p = BasisFun.cardinality(dim, p)
        f2v = RefSimplexElem.facet_2_vertices(dim)
        vert = RefSimplexElem.ref_vertices(dim)

        shpf_p1, _ = BasisFun.shp_fun(1, xqf, basis=basis)

        if gen_all_facets:
            nfacet = dim + 1
        else:
            nfacet = 1

        xfe = np.zeros((nfacet, nfn, dim))
        van_f = np.zeros((nfacet, nfn, n_p))
        van_f_der = np.zeros((nfacet, dim, nfn, n_p))

        for i in range(nfacet):
            tril = f2v[:, i]
            xl = vert[tril, :]
            xfe[i,:,:] = shpf_p1 @ xl
            van_f[i,:,:], van_f_der[i,:,:,:], _ = BasisFun.basis_function(xfe[i,:,:], p, basis)

        # Eliminate the last dimension of the array, which is one
        if ~gen_all_facets:
            xfe = xfe[0,:,:]
            van_f = van_f[0,:, :]
            van_f_der = van_f_der[0,:,:,:]

        return xfe, van_f, van_f_der

    @staticmethod
    def map2new_elem(xy_old, vert_old, vert_new, w_old=np.empty([0])):
        '''
        Parameters
        ----------
        xy_old : numpy array
            Cartesian coordinates of the nodal locations in the current element.
        vert_old : numpy array
            Cartesian coordinates of the vertices of the current element.
        vert_new : numpy array
            Cartesian coordinates of the vertices for the new element
        w_old : numpy array, optional
            Weights for the quad rule for the current element. If the
            parameter is not provided w_new is not calculated.

        Returns
        -------
        xy_new : numpy array
            Cartesian coordinates of the nodal locations for the new element.
        w_new : numpy array
            Weights of the quad rule for the new element.
        '''

        shp = BasisFun.shp_fun(1, xy_old, vert_old)[0]

        xy_new = shp @ vert_new

        if w_old.size == 0:
            w_new = np.empty([0])
        else:
            elem_size = RefSimplexElem.ref_elem_size(vert_new)
            w_new = w_old * (elem_size / np.sum(w_old))

        return xy_new, w_new

    @staticmethod
    def map2new_elem_curved(xy_old, vert_old, vert_new, w_old=np.empty([0])):

        shp = BasisFun.shp_fun(1, xy_old, vert_old)[0]

        xy_new = shp @ vert_new

        if w_old.size == 0:
            w_new = np.empty([0])
        else:
            elem_size = RefSimplexElem.ref_elem_size(vert_new)
            w_new = w_old * (elem_size / np.sum(w_old))

        return xy_new, w_new
