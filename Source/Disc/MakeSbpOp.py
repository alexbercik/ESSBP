#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Sep 2019

@author: andremarchildon
"""


# Add the root folder of ECO to the search path
import os
import sys

test_folder_path, _ = os.path.split(__file__)
root_folder_path, _ = os.path.split(test_folder_path)
sys.path.append(root_folder_path)

# Import the required modules
import numpy as np
from scipy.linalg import null_space

from Source.Disc.BasisFun import BasisFun
from Source.Disc.SbpQuadRule import SbpQuadRule
from Source.Disc.RefSimplexElem import RefSimplexElem
from Source.Disc.SimplexSymmetry import SimplexSymmetry
from Source.Disc.CSbpOp import CSbpOp


class MakeSbpOp:

    tol = 1e-8
    dim = 1 # note this restriction is not necessary for this module, but
    # simplifies things because the rest of the code works exclusively in 1d

    def __init__(self, p, sbp_type='lgl', nn=0, basis_type='legendre'):

        '''
        Parameters
        ----------
        quad : class
            Holds the nodal locations, weights and degrees for both the
            element and facet quad rules.
        p : int
            Degree of the SBP operator.
        sbp_type : string, optional
            The type of SBP family for the operator.
            The families are 'lgl' (legendre-gauss-lobatto), 'lg' (lobatto-gauss),
            'nc' (closed-form newton cotes), and 'csbp' (classical sbp).
            The default is 'lgl'.
        nn : int
            The number of nodes to use
            The default is 0, in which case nn is set automatically
        basis_type : string, optional
            Indicates the type of basis to use to construct the SBP operator.
            This does not change the final SBP operators but it can impact the
            condition number of the matrices used to construct the operators.
            The default is 'legendre'.

        Returns
        -------
        None.

        '''

        ''' Add inputs to the class '''

        self.p = p
        self.sbp_type = sbp_type
        self.basis_type = basis_type
        self.nn = nn

        ''' Check all inputs, beginning with quadrature '''

        if sbp_type.lower()=='csbp':
            self.sbp_fam = 'csbp'
            self.quad = None # if CSBP, it is meaningless to talk about quadrature
            assert self.nn > 1 , "Please specify number of nodes nn > 1"

            # Do things manually (operator is built later)
            self.xy = np.reshape(np.linspace(0,1,self.nn), (self.nn, 1))
            self.xqf = np.array([[0.]])
            self.bb = np.array([[1.]])

        else:
            if sbp_type=='lgl':
                self.sbp_fam = 'R0'
                self.quad = SbpQuadRule(1, p, sbp_fam='R0', nn=self.nn, quad_rule='lgl')
            elif sbp_type=='lg':
                self.sbp_fam = 'Rd'
                self.quad = SbpQuadRule(1, p, sbp_fam='Rd', nn=self.nn, quad_rule='lg')
            elif sbp_type=='nc':
                self.sbp_fam = 'R0'
                self.quad = SbpQuadRule(1, p, sbp_fam='R0', nn=self.nn, quad_rule='nc')
            else:
                raise Exception('Misunderstood SBP type.')

            ''' Extract required parameters '''

            # Element quad
            self.xy = self.quad.xq
            self.hh = np.diag(self.quad.wq)

            # Facet quad
            self.xqf = self.quad.xqf
            self.bb = np.diag(self.quad.wqf)

        # Common
        if ((self.nn != np.shape(self.xy)[0]) and (self.nn>0)):
            print('WARNING: Overwriting given nn, {0}, with size of given quadrature, {1}.'.format(nn,np.shape(self.xy)[0]))
        self.nn = np.shape(self.xy)[0]
        self.nnf = self.xqf.shape[0]
        self.n_facet = self.dim + 1

        ''' Construct SBP operators '''

        # Reference element
        ref_elem = RefSimplexElem(self.dim)
        self.vert = ref_elem.vert
        self.normal = ref_elem.normal
        self.f2v = ref_elem.f2v

        # Permutation information for elem and facet nodes
        self.perm = SimplexSymmetry(self.xy)
        self.perm_xqf = SimplexSymmetry(self.xqf)

        # Get the Vandermonde matrix at the element nodes
        elem_basis = BasisFun(self.xy, self.p, self.basis_type)
        self.van = elem_basis.van
        self.van_der = elem_basis.van_der
        self.n_p = elem_basis.n_p  # Cardinality
        self.idx_basis_dn1 = elem_basis.idx_basis_dn1 # Used for the Rdn1 family

        # Evaluate the facet nodes on the element facets
        [self.xfe, self.van_f, self.van_f_der] = BasisFun.facet_vandermonde(self.xqf, self.p, False, self.basis_type)

        # Calculate the interpolation/extrapolation operator
        self.rr = self.construct_rr()   # rr for facet 0
        self.rr_all = self.permute_rr() # rr for all facets

        # Calculate the directional surface integral operator
        ee_x = self.construct_dir_surf_int()

        # Calculate the operators for the derivative
        if self.sbp_fam=='csbp':
            self.hh, dd_x, qq_x, ss_x = CSbpOp(p,nn)
        else:
            dd_x, qq_x, ss_x = self.construct_ss(ee_x)

        # Permute all of the directional operators
        self.ee = self.permute_dir_op(ee_x)
        self.ss = self.permute_dir_op(ss_x)
        self.qq = self.permute_dir_op(qq_x)
        self.dd = self.permute_dir_op(dd_x)

        # Test the operators
        self.sbp_test_cub_rule(self.xy, self.hh, 2 * self.p - 1)
        self.sbp_test_acc_rr(self.rr, self.xy, self.xqf, self.p)
        self.sbp_test_derivative(self.dd, self.xy, p)
        self.sbp_test_decomposition(self.ee, self.ss, self.qq, self.dd, self.hh)

    def construct_rr(self):
        '''
        Returns
        -------
        rr : numpy array
            This is the int/ext operator R.
        '''

        # Refer to Section 2.3 of André Marchildon's SBP thesis for more
        # information on the construction of the operator rr that ensures
        # symmetrical contributions across symmetry lines or planes. Also
        # refer to Appendix C for the required equations.

        rr = np.zeros((self.nnf, self.nn))

        # Rdn1 for 1D is the same as R0
        if self.sbp_fam == 'R0' or (self.sbp_fam == 'Rdn1' and self.dim == 1) or self.sbp_fam == 'csbp':

            xy = self.xy
            xfe = self.xfe

            for i_e in range(self.nn):
                for i_f in range(self.nnf):
                    if np.max(np.abs(xy[i_e, :] - xfe[i_f, :])) < self.tol:
                        rr[i_f, i_e] = 1

            return rr
        else:
            # For sbp_fam Rd and Rdn1 there are two steps to solving for rr:
            #   1) Solve for the contribution to one facet node per sym group
            #   2) Use perm matrices to get the solution at the other facet
            #      nodes. This ensures the contribution from rr is symmetric

            # Step 1: Solve for rr for one facet node per sym group
            van = self.van
            f_nodes = self.perm_xqf.sym_g_mat[:, 0] # First facet node in each sym group
            van_f = self.van_f[f_nodes, :]

            # Find a solution for the operator rr
            if self.sbp_fam == 'Rd':
                rr[f_nodes, :] = np.linalg.lstsq(van.T, van_f.T, rcond=None)[0].T

            elif self.sbp_fam == 'Rdn1':
                # Consider only elem nodes on the facet
                elem_nodes = self.perm.node_on_facet0
                van = van[elem_nodes, :]
                van = van[:, self.idx_basis_dn1]

                # Since all the elem nodes used for int/ext are on a facet
                # not all basis functions are independent
                van_f = van_f[:, self.idx_basis_dn1]

                rr_temp = np.zeros((len(f_nodes), self.nn))
                rr_temp[:, elem_nodes] = np.linalg.lstsq(van.T, van_f.T, rcond=None)[0].T
                rr[f_nodes, :] = rr_temp

            else:
                raise Exception('Unknown family of SBP operators')

            # Step 2: Make the contributions from the operator rr symmetric
            if self.dim == 2:
                # Facet node symmetry groups
                sym_g_id = self.perm_xqf.sym_g_id
                sym_g_mat = self.perm_xqf.sym_g_mat

                # Element node permutation matrices
                I = np.eye(self.nn)
                perm_x2y = self.perm.perm_mat_dir[1,:,:]

                n_groups = sym_g_id.shape[0]

                for i in range(n_groups):
                    if sym_g_id[i] == 0:
                        node_idx = sym_g_mat[i, 0]
                        rr[node_idx, :] = 0.5 * (rr[node_idx, :] @ (I + perm_x2y))
                    elif sym_g_id[i] == 1:
                        sym_g_mat_i = sym_g_mat[i,:]
                        rr[sym_g_mat_i[1], :] = rr[sym_g_mat_i[0], :] @ perm_x2y
                    else:
                        raise Exception('Unknown symmetry group for 1D facet')
            elif self.dim == 3:
                # Facet node symmetry groups
                sym_g_id = self.perm_xqf.sym_g_id
                sym_g_mat = self.perm_xqf.sym_g_mat

                # Element node permutation matrices
                I = np.eye(self.nn)
                perm_x2y = self.perm.perm_mat_dir[1,:,:]
                perm_x2z = self.perm.perm_mat_dir[2,:,:]
                perm_y2z = self.perm.perm_mat_dir[3,:,:]

                n_groups = sym_g_id.shape[0]

                for i in range(n_groups):
                    sym_g_mat_i = sym_g_mat[i,:]
                    if sym_g_id[i] == 0:
                        node_idx = sym_g_mat[i, 0]
                        perm = (I + perm_x2y + perm_x2z @ (I+perm_x2y) + perm_y2z @ (I+perm_x2y)) / 6
                        rr[node_idx, :] = rr[node_idx, :] @ perm
                    elif sym_g_id[i] == 1:
                        rr[sym_g_mat_i[1], :] = rr[sym_g_mat_i[0], :] @ perm_x2y
                        rr[sym_g_mat_i[2], :] = rr[sym_g_mat_i[0], :] @ perm_x2z
                    elif  sym_g_id[i] == 2:
                        rr[sym_g_mat_i[1], :] = rr[sym_g_mat_i[0], :] @ perm_y2z @ perm_x2y
                        rr[sym_g_mat_i[2], :] = rr[sym_g_mat_i[0], :] @ perm_x2z @ perm_x2y
                        rr[sym_g_mat_i[3], :] = rr[sym_g_mat_i[0], :] @ perm_x2y
                        rr[sym_g_mat_i[4], :] = rr[sym_g_mat_i[0], :] @ perm_x2y
                        rr[sym_g_mat_i[5], :] = rr[sym_g_mat_i[0], :] @ perm_x2y

            return rr

    def permute_rr(self):

        nfacet = self.dim + 1
        rr_all = np.zeros((nfacet, self.nnf, self.nn))

        for i in range(nfacet):
            rr_all[i,:,:] = self.rr @ self.perm.perm_mat_r[i,:,:]

        return rr_all

    def construct_dir_surf_int(self):
        '''
        Returns
        -------
        ee : numpy array
            This operator calculate the directional surface integral for the
            0-th facet.
        '''

        # See Sections 1.3.4 and 2.3 in André Marchildon's SBP thesis for
        # construction of the operator ee

        ee = np.zeros((self.nn, self.nn))
        aa = self.rr.T @ self.bb @ self.rr

        for i in range(0, self.n_facet):
            perm_r = self.perm.perm_mat_r[i,:,:]
            ee = ee + self.normal[i, 0] * (perm_r.T @ aa @ perm_r)

        return ee

    def construct_ss(self, ee_x):
        '''
        Returns
        -------
        dd : numpy array
            This is the derivative operator.
        qq : numpy array
            This is the weak derivative operator.
        ss : numpy array
            This is the skew-symmetric matrix.
        '''

        # The optz method is presented in André Marchildon's SBP thesis.
        # For more information on this method refer to Section 4.6 and
        # Appendix D in the thesis. The other method is the one presented in
        # the original multidimensional SBP paper.
        use_optz_method = True

        if use_optz_method:
            n_p = self.n_p
            nn = self.nn
            van = self.van
            van_der = self.van_der[0,:,:] # Derivative wrt x

            n_dof = int(np.round(nn*(nn-1)/2))
            n_ind_eq = int(np.round(n_p * (n_p-1)/2 + n_p*(nn-n_p)))

            rhs1 = 0.5*(van.T @ self.hh @ van_der - van_der.T @ self.hh @ van)

            A = np.zeros((n_ind_eq, n_dof))
            bvec = np.zeros(n_ind_eq)

            for a in range(1, n_p):
                for b in range(a):
                    m = int(np.round((a-1)*a/2 + b))

                    bvec[m] = rhs1[a,b]
                    for i in range(1, nn):
                        for j in range(i):
                            k = int(np.round((i-1)*i/2 + j))
                            A[m,k] = van[i,a]*van[j,b] - van[i,b]*van[j,a]

            if nn > n_p:
                ww = null_space(van.T)
                rhs2 = ww.T @ (self.hh @ van_der - 0.5*ee_x @ van)
                for a in range(nn-n_p):
                    for b in range(n_p):
                        m = int(np.round(a*n_p + b + n_p * (n_p-1)/2))
                        bvec[m] = rhs2[a,b]

                        for j in range(nn):
                            for i in range(j+1,nn):
                                k = int(np.round((i-1)*i/2 + j))
                                A[m,k] = ww[i,a]*van[j,b] - ww[j,a]*van[i,b]

            # This is not the global solution to the minimization of
            # Eq. (4.38) but it provides a good solution.
            svec = np.linalg.lstsq(A, bvec, rcond=None)[0]

            # Convert the 1D array svec into the 2D skew-symmetric aray ss
            ss = np.zeros((nn, nn))

            idx = -1
            for i in range(1,nn):
                for j in range(i):
                    idx += 1
                    ss[i,j] = svec[idx]
                    ss[j,i] = -svec[idx]

            qq = ss + 0.5*ee_x
            dd = np.linalg.solve(self.hh, qq)

        else:
            # Refer to the original multidimensional SBP paper for this method
            van_der = self.van_der[0,:,:] # Derivative wrt x
            # ee_x = self.ee[0,:,:]

            num_w_col = self.nn - self.n_p

            if num_w_col > 0:
                # Create square invertible matrix by appending self.van with its nullspace
                ww = null_space(self.van.T)
                van_tilda = np.concatenate((self.van, ww), axis=1)

                # Solve for wx
                mat_1 = van_tilda.T @ self.hh

                mat_2a = 0.5 * van_tilda.T @ ee_x @ ww
                mat_2b = (-van_der.T @ self.hh + 0.5 * self.van.T @ ee_x) @ ww
                mat_zero = np.zeros((num_w_col, num_w_col))
                mat_2b = np.concatenate((mat_2b, mat_zero), axis=0)
                mat_2 = mat_2a + mat_2b

                wx = np.linalg.solve(mat_1, mat_2)
                van_x_tilda = np.concatenate((van_der, wx), axis=1)

                # Solve for the derivative operator
                dd = np.linalg.solve(van_tilda.T, van_x_tilda.T).T
            else:
                dd = np.linalg.solve(self.van.T, van_der.T).T

            qq = self.hh @ dd
            ss = 0.5 * (qq - qq.T)

        return dd, qq, ss

    def permute_dir_op(self, op_dir0):
        '''
         Purpose
        -------
        Calculates the directional operators for each direction using the
        directional operators for the 0-th direction and the required
        permutation matrices.
        See section 2.2 of André Marchildon's thesis on SBP operators for more
        information.

        Parameters
        ----------
        op_dir0 : numpy array
            Directional operator for the zero-th (x) direction
        '''

        op_dir_all = np.zeros((self.dim, self.nn, self.nn))

        for d in range(self.dim):
            perm_mat_dir = self.perm.perm_mat_dir[d,:,:]
            op_dir_all[d,:,:] = perm_mat_dir @ op_dir0 @ perm_mat_dir

        return op_dir_all

    def ref_2_phys_op(self, det_jac, inv_jac):
        '''
        Parameters
        ----------
        det_jac : numpy array
            Determinant of the mesh Jacobian.
        inv_jac : numpy array
            Inverse of the mesh Jacobian matrix.

        Returns
        -------
        hh : numpy array
            diagonal weight matrix.
        qq : numpy array
            SBP operator for the weak derivative.
        dd : numpy array
            SBP derivative operator.
        '''

        hh_phys =  det_jac @ self.hh

        if self.dim == 1:

            lambda_xi_x = det_jac @ np.diag(inv_jac[:,0,0])
            qqx = self.qq[0,:,:]

            qq_phys = lambda_xi_x @ qqx
            dd_phys = np.linalg.solve(hh_phys, qq_phys)

        else:
            raise Exception('Curvilinear transformation only currently available in 1D')

        return hh_phys, qq_phys, dd_phys

    @staticmethod
    def sbp_test_cub_rule(xy, hh, pquad, tol=1e-6):

        dim = xy.shape[1]

        # Get the Vandermonde matrix for xy
        xy_basis = BasisFun(xy, pquad)
        xy_van = xy_basis.van

        # Get cubature rule of higher degree than pcub to compare the integration
        xquad, wquad, _ = SbpQuadRule.quad_lg(dim, pquad+2)
        wquad = np.diag(wquad)

        # Get the Vandermonde matrix for the higher degree cub rule
        cub_basis = BasisFun(xquad, pquad)
        xcub_van = cub_basis.van

        # Complete the integration for both
        xy_int = np.sum(hh @ xy_van, axis=0)       # Int. with xy and H
        cub_int = np.sum(wquad @ xcub_van, axis=0) # Int. with higher order cub rule

        # Compare the two integrations
        diff_int = np.max(np.abs(xy_int - cub_int))
        assert diff_int < tol, 'The cub rule is not exact to the required degree'

    @staticmethod
    def sbp_test_acc_rr(rr, xy, xqf, p, tol=1e-6):

        # Get the Vandermonde matrix for the element nodes
        elem_basis = BasisFun(xy, p)
        van = elem_basis.van

        # Get the Vandermonde matrix for the facet quadrature nodes on facet 1
        van_f = BasisFun.facet_vandermonde(xqf, p, False)[1]

        test = rr @ van - van_f
        max_err = np.max(np.abs(test))

        assert max_err < tol, 'The int/ext op R is not sufficiently accurate'

    @staticmethod
    def sbp_test_decomposition(ee, ss, qq, dd, hh, tol=1e-6):

        dim = ee.shape[0]

        for d in range(dim):
            # Extract the operators for direction d
            ee_d = ee[d,:,:]
            ss_d = ss[d,:,:]
            qq_d = qq[d,:,:]
            dd_d = dd[d,:,:]

            # Test that the matrix E is symmetric
            test_ee = np.max(np.abs(ee_d - ee_d.T))
            assert test_ee < tol, 'The matrix E is not symmetric'

            # Test that the matrix S is skew-symmetric
            test_ss = np.max(np.abs(ss_d + ss_d.T))
            assert test_ss < tol, 'The matrix S is not skew-symmetric'

            # Test that the matrix Q decomposes into E and S
            test_qq = np.max(np.abs(qq_d - (ss_d + 0.5*ee_d)))
            assert test_qq < tol, 'The matrix Q does not decompose properly into S and E'

            # Test that the matrix D decomposes into H and Q
            test_dd = np.max(np.abs(hh @ dd_d - qq_d))
            assert test_dd < tol, 'The matrix D does not decompose properly into Q and H'

    @staticmethod
    def sbp_test_derivative(dd, xy, p, tol=1e-6):

        # Get the Vandermonde matrix and its derivative for the element nodes
        elem_basis = BasisFun(xy, p)
        van = elem_basis.van
        van_der = elem_basis.van_der

        dim = xy.shape[1]

        for d in range(0, dim):
            test_dd = dd[d,:,:] @ van - van_der[d,:,:]
            max_err = np.max(np.abs(test_dd))
            assert max_err < tol, 'The derivative is not exact'
