#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Sep 2019

@author: bercik
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
from Source.Disc.CSbpOp import CSbpOp
import Source.Methods.Functions as fn


class MakeSbpOp:

    tol = 1e-8
    # Note this only produces one dimensional operators for use in tensor-product

    def __init__(self, p, sbp_type='lgl', nn=0, basis_type='legendre'):

        '''
        Parameters
        ----------
        p : int or (int,int)
            Degree of the SBP operator.
        sbp_type : string, optional
            The type of SBP family for the operator.
            The families are 'lgl' (legendre-gauss-lobatto), 'lg' (lobatto-gauss),
            'nc' (closed-form newton cotes), and 'csbp' (classical sbp).
            The default is 'lgl'.
        nn : int or (int,int)
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
        print('... Building reference operators')
        
        ''' Add inputs to the class '''
        self.sbp_type = sbp_type
        self.basis_type = basis_type
        assert isinstance(p, int), 'p must be an integer'
        assert isinstance(nn, int), 'nn must be an integer'
        self.p = p
        self.nn = nn
       
        if sbp_type.lower()=='csbp':
            ''' Build Classical SBP Operators '''
            
            self.sbp_fam = 'csbp'
            self.quad = None # if CSBP, it is meaningless to talk about quadrature
            assert self.nn > 1 , "Please specify number of nodes nn > 1"

            # Do things manually (operator is built later)
            self.x = np.linspace(0, 1,self.nn)
            self.H, self.D, self.Q, self.S = CSbpOp(self.p,self.nn)
            self.E = np.zeros((self.nn,self.nn))
            self.E[0,0], self.E[-1,-1] = -1 , 1
            self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
            self.tL[0] , self.tR[-1] = 1 , 1

        else:
            ''' Build Element-type SBP Operators '''
            
            if self.sbp_type == 'lgl':
                self.sbp_fam = 'R0'
                self.quad = SbpQuadRule(p, sbp_fam='R0', nn=self.nn, quad_rule='lgl')
            elif self.sbp_type == 'lg':
                self.sbp_fam = 'Rd'
                self.quad = SbpQuadRule(p, sbp_fam='Rd', nn=self.nn, quad_rule='lg')
            elif self.sbp_type == 'nc':
                self.sbp_fam = 'R0'
                self.quad = SbpQuadRule(p, sbp_fam='R0', nn=self.nn, quad_rule='nc')
            else:
                raise Exception('Misunderstood SBP type.')
                
            self.x = self.quad.xq[:,0]
            self.H = np.diag(self.quad.wq)
                
            if ((self.nn != len(self.x)) and (self.nn>0)):
                print('WARNING: Overwriting given nn, {0}, with size of given quadrature, {1}.'.format(nn,len(self.x)))
            self.nn = len(self.x)
        
            # Get the Vandermonde matrix at the element nodes
            elem_basis = BasisFun(self.quad.xq, self.p, self.basis_type)
            self.van = elem_basis.van
            self.van_der = elem_basis.van_der[0]
            self.tL = self.construct_tL(self.nn,self.van,self.p)   # interpolation vector for left node
            self.tR = np.flip(self.tL)                             # interpolation vector for right node
            self.E = np.outer(self.tR, self.tR) - np.outer(self.tL, self.tL) # surface integral
            self.D, self.Q, self.S = self.construct_op(self.p, self.nn, self.van, self.van_der, self.E, self.H)

        ''' Test the operators '''
        self.check_diagH(self.x, self.H)
        self.check_accuracy(self.D, self.x)
        self.check_compatibility(self.x,self.H,self.E)
        self.check_interpolation(self.tL, self.tR, self.x)
        self.check_decomposition(self.E, self.S, self.Q, self.D, self.H)


    def construct_tL(self,nn,van,p):
        '''
        Returns
        -------
        tL : numpy array
            This is the left int/ext operator tL.
        '''

        # Refer to Section 2.3 of André Marchildon's SBP thesis for more
        # information on the construction of the operator rr that ensures
        # symmetrical contributions across symmetry lines or planes. Also
        # refer to Appendix C for the required equations.

        if self.sbp_fam != 'Rd': 
            tL = np.zeros(nn)
            tL[0] = 1

        else:
            # This is simplified code from Andre assuming everything 1D, facet 0D
            van_f = BasisFun.facet_vandermonde(np.array([[0.]]), p, False, self.basis_type)[1].flatten()
            tL = np.linalg.lstsq(van.T, van_f.T, rcond=None)[0]

        return tL

    @staticmethod
    def construct_op(p, nn, van, van_der, E, H):
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
        use_optz_method = False
        # Slightly more accurate to set as False for 1D operators 
        # (roundoff error, the actual operators are the same really)
        
        n_p = p + 1 # for 1D

        if use_optz_method:
            n_dof = int(np.round(nn*(nn-1)/2))
            n_ind_eq = int(np.round(n_p * (n_p-1)/2 + n_p*(nn-n_p)))

            rhs1 = 0.5*(van.T @ H @ van_der - van_der.T @ H @ van)

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
                rhs2 = ww.T @ (H @ van_der - 0.5*E @ van)
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

            qq = ss + 0.5*E
            dd = np.linalg.solve(H, qq)

        else:
            num_w_col = nn - n_p

            if num_w_col > 0:
                # Create square invertible matrix by appending self.van with its nullspace
                ww = null_space(van.T)
                van_tilda = np.concatenate((van, ww), axis=1)

                # Solve for wx
                mat_1 = van_tilda.T @ H

                mat_2a = 0.5 * van_tilda.T @ E @ ww
                mat_2b = (-van_der.T @ H + 0.5 * van.T @ E) @ ww
                mat_zero = np.zeros((num_w_col, num_w_col))
                mat_2b = np.concatenate((mat_2b, mat_zero), axis=0)
                mat_2 = mat_2a + mat_2b

                wx = np.linalg.solve(mat_1, mat_2)
                van_x_tilda = np.concatenate((van_der, wx), axis=1)

                # Solve for the derivative operator
                dd = np.linalg.solve(van_tilda.T, van_x_tilda.T).T
            else:
                dd = np.linalg.solve(van.T, van_der.T).T

            qq = H @ dd
            ss = 0.5 * (qq - qq.T)

        return dd, qq, ss



    def ref_2_phys(self, mesh):
        '''
        Parameters
        ----------
        mesh : class instance of MakeMesh.py defining mesh

        Returns
        -------
        H_phys : numpy array 
            Quadrature matrix. Note although this should be a (nen,nen,nelem)
            or (nen^2,nen^2,nelem) matrix, it is actually stored as a (nen,nelem)
            or (nen^2,nelem) to save memory because it is diagonal.
        D_phys : numpy array
            SBP physical derivative operator, size (nen,nen,nelem) or (nen^2,nen^2,nelem).
        '''
        print('... Creating physical operators')
        
        form = 'skew-sym' # 'skew-sym' or 'strong'
        
        assert fn.isDiag(self.H), 'H matrix is not diagonal!'

        if mesh.dim == 1:
            
            H_phys = np.diag(self.H)[:,None] * mesh.det_jac
            D_phys = fn.gdiag_lm(mesh.det_jac_inv, self.D)
            
            return H_phys, D_phys
        
        elif mesh.dim == 2:
            
            eye = np.eye(mesh.nen)
            H_phys = np.diag(np.kron(self.H,self.H))[:,None] * mesh.det_jac
            
            Dx = np.kron(self.D, eye) # shape (nen^2,nen^2)
            Dy = np.kron(eye, self.D)

            if form == 'skew-sym':
                Dx_phys = 0.5*fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,0,:]) + fn.gdiag_lm(mesh.metrics[:,0,:],Dx) 
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,2,:]) + fn.gdiag_lm(mesh.metrics[:,2,:],Dy)))
                Dy_phys = 0.5*fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,1,:]) + fn.gdiag_lm(mesh.metrics[:,1,:],Dx) 
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,3,:]) + fn.gdiag_lm(mesh.metrics[:,3,:],Dy)))
            elif form == 'strong': # not provably stable
                Dx_phys = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,0,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,2,:])))
                Dy_phys = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,1,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,3,:])))
            else:
                raise Exception('Physical operator form not understood.')

            return H_phys, Dx_phys, Dy_phys
        
        elif mesh.dim == 3:
            
            eye = np.eye(mesh.nen)
            H_phys = np.diag(np.kron(np.kron(self.H,self.H),self.H))[:,None] * mesh.det_jac
            
            Dx = np.kron(np.kron(self.D, eye),eye) # shape (nen^3,nen^3)
            Dy = np.kron(np.kron(eye, self.D), eye)
            Dz = np.kron(np.kron(eye, eye), self.D)

            if form == 'skew-sym':
                Dx_phys = 0.5*fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,0,:]) + fn.gdiag_lm(mesh.metrics[:,0,:],Dx) 
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,3,:]) + fn.gdiag_lm(mesh.metrics[:,3,:],Dy)
                                                          + fn.lm_gdiag(Dz,mesh.metrics[:,6,:]) + fn.gdiag_lm(mesh.metrics[:,6,:],Dz)))
                Dy_phys = 0.5*fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,1,:]) + fn.gdiag_lm(mesh.metrics[:,1,:],Dx) 
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,4,:]) + fn.gdiag_lm(mesh.metrics[:,4,:],Dy)
                                                          + fn.lm_gdiag(Dz,mesh.metrics[:,7,:]) + fn.gdiag_lm(mesh.metrics[:,7,:],Dz)))
                Dz_phys = 0.5*fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,2,:]) + fn.gdiag_lm(mesh.metrics[:,2,:],Dx) 
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,5,:]) + fn.gdiag_lm(mesh.metrics[:,5,:],Dy)
                                                          + fn.lm_gdiag(Dz,mesh.metrics[:,8,:]) + fn.gdiag_lm(mesh.metrics[:,8,:],Dz)))
            elif form == 'strong': # not provably stable
                Dx_phys = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,0,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,3,:])
                                                         + fn.lm_gdiag(Dz,mesh.metrics[:,6,:])))
                Dy_phys = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,1,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,4,:])
                                                         + fn.lm_gdiag(Dz,mesh.metrics[:,7,:])))
                Dz_phys = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,2,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,5,:])
                                                         + fn.lm_gdiag(Dz,mesh.metrics[:,8,:])))
            else:
                raise Exception('Physical operator form not understood.')

            return H_phys, Dx_phys, Dy_phys, Dz_phys


    @staticmethod
    def check_diagH(x,H,tol=1e-10):
        ''' tests order of quadrature for H, (j+1)*H*x^j = b^(j+1)-a(j+1). 
        Based on reference element [0,1] '''
        p=0
        er=0
        while er<tol:
            p+=1
            er = abs((p+1)*(np.diag(H) @ x**p) - 1)
        print('Test: Quadrature H is order {0}.'.format(p-1))
        
    @staticmethod
    def check_compatibility(x,H,E,tol=1e-10):
        ''' tests order of compatibility equations x^i@E@x^j = j*x^i@H@x^(j-1) + i*x^j@H@x^(i-1) . 
        Based on reference element [0,1] '''
        p=0
        er=0
        while er<tol:
            p+=1
            mesh = np.array(np.meshgrid(np.arange(p+1),np.arange(p+1))).T.reshape(-1, 2)
            exs = [list(e) for e in set(frozenset(d) for d in mesh)]
            for i in range(len(exs)):
                if len(exs[i]) ==1: exs[i] = [exs[i][0],exs[i][0]]  
            exs.sort()
            er = 0
            for ex in exs:
                i , j = ex[0] , ex[1]
                if i+j == 0: ans = 0
                elif i ==0: ans = j*((x**i)@H@(x**(j-1)))
                elif j ==0: ans = i*((x**j)@H@(x**(i-1)))
                else: ans = j*((x**i)@H@(x**(j-1))) + i*((x**j)@H@(x**(i-1)))
                er += abs(x**i @ E @ x**j - ans)
        print('Test: Compatibility equations hold to order {0}'.format(p-1))        

    @staticmethod
    def check_interpolation(tL, tR, x, tol=1e-10):
        ''' tests the accuracy of the interpolation tL. Based on reference element [0,1] '''
        if abs(tL[0]-1)<tol and abs(np.sum(tL) - 1)<tol:
            print('Test: The interpolation tL/tR is exact, i.e. there are boundary nodes.')
        else:
            p=0
            er=0
            while er<tol:
                p+=1
                er = abs(tL@x**p) + abs(tR@(2*x)**p - 2**p)
            print('Test: The interpolation tL/tR is order {0}.'.format(p-1))

    @staticmethod
    def check_decomposition(E, S, Q, D, H, tol=1e-10):

        # Test that the matrix E is symmetric
        assert np.max(np.abs(E - E.T)) < tol, 'The matrix E is not symmetric'

        # Test that the matrix S is skew-symmetric
        assert np.max(np.abs(S + S.T)) < tol, 'The matrix S is not skew-symmetric'
        assert np.max(np.diag(S)) < tol, 'The matrix S is not skew-symmetric'

        # Test that the matrix Q decomposes into E and S
        test_Q = np.max(np.abs(Q - (S + 0.5*E)))
        assert test_Q < tol, 'The matrix Q does not decompose properly into S and E'

        # Test that the matrix D decomposes into H and Q
        test_D = np.max(np.abs(H @ D - Q))
        assert test_D < tol, 'The matrix D does not decompose properly into Q and H'
        
        # Test that the matrix E decomposes into Q and Q.T
        test_E = np.max(np.abs(E - Q - Q.T))
        assert test_E < tol, 'The matrix E does not decompose properly into Q and Q.T'
        
        print('Test: The operator succesfully passed all decomposition tests.')

    @staticmethod
    def check_accuracy(D, x, tol=1e-10):
        ''' tests order of derivative D, D@x^j = j*x^(j-1) 
        Based on reference element [0,1] '''
        p=1
        er=0
        while er<tol:
            p+=1
            er = np.sum(abs(D @ x**p - p*x**(p-1)))
        print('Test: Derivative D is order {0}.'.format(p-1))