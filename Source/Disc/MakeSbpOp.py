#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Sep 2019

@author: bercik
"""


# Add the root folder of ECO to the search path
import os
from sys import path

n_nested_folder = 2
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

# Import the required modules
import numpy as np
from scipy.linalg import null_space

from Source.Disc.BasisFun import BasisFun
from Source.Disc.SbpQuadRule import SbpQuadRule
from Source.Disc.CSbpOp import CSbpOp, HGTLOp, HGTOp, MattOp, HGTLOp_DDRF, HGTLOp_DDRF2
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp
from Source.Disc.MakeMesh import MakeMesh
from contextlib import redirect_stdout


class MakeSbpOp:

    tol = 1e-8
    # Note this only produces one dimensional operators for use in tensor-product

    def __init__(self, p, sbp_type='lgl', nn=0, basis_type='legendre',
                 print_progress=True):

        '''
        Parameters
        ----------
        p : int or (int,int)
            Degree of the SBP operator.
        sbp_type : string, optional
            The type of SBP family for the operator.
            The families are 'lgl' (legendre-gauss-lobatto), 'lg' (lobatto-gauss),
            'nc' (closed-form newton cotes), 'csbp' (classical sbp), 
            'optz_map'.
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
        if print_progress: print('... Building reference operators')
        
        ''' Add inputs to the class '''
        self.sbp_type = sbp_type
        self.basis_type = basis_type
        assert isinstance(p, int), 'p must be an integer'
        assert isinstance(nn, int), 'nn must be an integer'
        self.p = p
        self.nn = nn
        self.print_progress = print_progress
       
        if sbp_type.lower()=='csbp' or sbp_type.lower()=='optz':
            ''' Build Classical SBP Operators '''
            
            self.sbp_fam = 'csbp'
            self.quad = None # if CSBP, it is meaningless to talk about quadrature
            assert self.nn > 1 , "Please specify number of nodes nn > 1"
            if p==1 and nn<3:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 3.'.format(nn))
                self.nn = 3
            elif p==2 and nn<9:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 9.'.format(nn))
                self.nn=9
            elif p==3 and nn<13:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 13.'.format(nn))
                self.nn = 13
            elif p==4 and nn<17:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 17.'.format(nn))
                self.nn = 17

            # Do things manually (operator is built later)
            self.x = np.linspace(0, 1,self.nn)
            self.H, self.D, self.Q, self.S, self.dx = CSbpOp(self.p,self.nn)
            self.E = np.zeros((self.nn,self.nn))
            self.E[0,0], self.E[-1,-1] = -1 , 1
            self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
            self.tL[0] , self.tR[-1] = 1 , 1

            if sbp_type.lower()=='optz':
                if p==1 and self.nn==3:
                    warp_factors = 0.0
                    trans = 'sigmoid'
                elif p==1 and self.nn==20:
                    warp_factors = [2.593505526588188, 1.84727641471781, 0.2359412051939956]
                    trans = 'corners'
                elif p==2 and self.nn==9:
                    warp_factors = 0.18991350096392648
                    trans = 'sigmoid'
                elif p==2 and self.nn==20:
                    warp_factors = [2.770906371088635, 1.2849104355129346, 0.288447754977867]
                    trans = 'corners'
                elif p==3 and self.nn==13:
                    warp_factors = 0.1035576820946656
                    trans = 'sigmoid'
                elif p==3 and self.nn==20:
                    warp_factors = [5.024511643375132, 19.999041198274156, 0.1866586054617816]
                    trans = 'corners'
                elif p==4 and self.nn==17:
                    warp_factors = 0.03464339479945
                    trans = 'sigmoid'
                elif p==2 and self.nn==30:
                    warp_factors = [4.19668524183192, 1.0903630402992481, 0.1871758767436309]
                    trans = 'corners'
                elif p==2 and self.nn==51:
                    warp_factors = [3.88219157126648, 29.986780432571532, 0.1041278770541774]
                    trans = 'corners'
                elif p==4 and self.nn==51:
                    #warp_factors = [3.00145084, 2.99917623, 0.35528195]
                    warp_factors = [1.9515004691739295, 1.0668208330264204, 0.3740774535232256]
                    trans = 'corners'
                elif p==1 and self.nn==51:
                    # this is not optimal, but is what Julia / Kaxie are using
                    warp_factors = np.sqrt(1-(1/1.01)), 0., 0.
                    trans = 'tanh'
                else:
                    print('WARNING: Not set up yet, defaulting to CSBP.')
                    warp_factors = 0.
                    trans = 'default'

                with redirect_stdout(None):
                    mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=self.x,warp_type=trans,
                                    warp_factor=warp_factors)
                    mesh.get_jac_metrics(self, periodic=False,
                            metric_method = 'exact', 
                            bdy_metric_method = 'exact',
                            jac_method='exact',
                            use_optz_metrics = 'False',
                            calc_exact_metrics = False)
                    H, D, _ = self.ref_2_phys(mesh, 'skew_sym')
                    self.x = mesh.x
                    self.H, self.D = np.diag(H[:,0]), D[:,:,0]
                    self.Q = self.H @ self.D
                    self.S = self.Q - self.E/2

        elif sbp_type.lower()=='hgtl' or sbp_type.lower()=='hgt' or sbp_type.lower()=='mattsson' \
            or sbp_type.lower()=='hgtl_ddrf' or sbp_type.lower()=='hgtl_ddrf2':
            ''' Build Hybrid Gauss Trapezoidal Lobatto and Hybrid Gauss Trapezoidal Operators '''
            
            self.quad = None 
            assert self.nn > 1 , "Please specify number of nodes nn > 1"
            if p==2 and nn<9:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 9.'.format(nn))
                self.nn=9
            elif p==3 and nn<13:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 13.'.format(nn))
                self.nn = 13
            elif p==4 and nn<17:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 17.'.format(nn))
                self.nn = 17

            if sbp_type.lower()=='hgtl':
                self.sbp_fam = 'hgtl'
                self.H, self.D, self.Q, self.E, self.S, self.dx, self.x = HGTLOp(self.p,self.nn)
                self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
                self.tL[0] , self.tR[-1] = 1 , 1
            elif sbp_type.lower()=='hgtl_ddrf':
                self.sbp_fam = 'hgtl_ddrf'
                self.H, self.D, self.Q, self.E, self.S, self.dx, self.x = HGTLOp_DDRF(self.p,self.nn)
                self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
                self.tL[0] , self.tR[-1] = 1 , 1
            elif sbp_type.lower()=='hgtl_ddrf2':
                self.sbp_fam = 'hgtl_ddrf2'
                self.H, self.D, self.Q, self.E, self.S, self.dx, self.x = HGTLOp_DDRF2(self.p,self.nn)
                self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
                self.tL[0] , self.tR[-1] = 1 , 1
            elif sbp_type.lower()=='hgt':
                self.sbp_fam = 'hgt'
                self.H, self.D, self.Q, self.E, self.S, self.dx, self.x, self.tL, self.tR = HGTOp(self.p,self.nn)
            else:
                self.sbp_fam = 'mattsson'
                self.H, self.D, self.Q, self.E, self.S, self.dx, self.x = MattOp(self.p,self.nn)
                self.tL, self.tR = np.zeros(self.nn), np.zeros(self.nn)
                self.tL[0] , self.tR[-1] = 1 , 1


        elif sbp_type.lower()=='upwind':
            from Source.Disc.UpwindOp import UpwindOp
            assert self.nn > 1 , "Please specify number of nodes nn > 1"
            assert self.nn > 1 , "Please specify degree p > 1"
            if (p==2 or p==3)  and nn<5:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 5.'.format(nn))
                self.nn = 5
            elif (p==4 or p==5) and nn<9:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 9.'.format(nn))
                self.nn = 9
            elif (p==6 or p==7) and nn<13:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 13.'.format(nn))
                self.nn = 13
            elif (p==8 or p==9) and nn<17:
                print('WARNING: nn set too small ({0}). Automatically increasing to minimum 17.'.format(nn))
                self.nn = 17

            self.D, self.Dp, self.Dm, self.Q, self.H, self.E, self.S, self.tL, self.tR, self.x, self.Ddiss = UpwindOp(p,self.nn)
            self.dx = 1./(nn-1)

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
            self.dx = 1.
        
            # Get the Vandermonde matrix at the element nodes
            elem_basis = BasisFun(self.quad.xq, self.p, self.basis_type)
            self.van = elem_basis.van
            self.van_der = elem_basis.van_der[0]
            self.tL = self.construct_tL(self.nn,self.van,self.p)   # interpolation vector for left node
            self.tR = np.flip(self.tL)                             # interpolation vector for right node
            self.E = np.outer(self.tR, self.tR) - np.outer(self.tL, self.tL) # surface integral
            self.D, self.Q, self.S = self.construct_op(self.p, self.nn, self.van, self.van_der, self.E, self.H)

        ''' Test the operators '''
        if print_progress: 
            self.check_diagH(self.x, self.H)
            self.check_degree(self.D, self.x)
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



    def ref_2_phys(self, mesh, neq, form, disc_type, sparse, sat_sparse, calc_nd_ops):
        '''
        Set the physical operators, and set to sparse if necessary

        Parameters
        ----------
        mesh : class instance of MakeMesh.py defining mesh
        form : 'skew_sym' or 'div'

        Returns
        -------
        H_phys : numpy array 
            Quadrature matrix. Note although this should be a (nen,nen,nelem)
            or (nen^2,nen^2,nelem) matrix, it is actually stored as a (nen,nelem)
            or (nen^2,nelem) to save memory because it is diagonal.
        D_phys : numpy array
            SBP physical derivative operator, size (nen,nen,nelem) or (nen^2,nen^2,nelem).
        '''
        if self.print_progress: print('... Creating physical operators')
        
        assert fn.isDiag(self.H), 'H matrix is not diagonal!'

        if mesh.dim == 1:

            self.H_phys_unkronned = np.diag(self.H)[:,None] * mesh.det_jac
            self.H_inv_phys_unkronned = 1/self.H_phys_unkronned
            self.H_phys = fn.repeat_neq_gv(self.H_phys_unkronned,neq)
            self.H_inv_phys = 1/self.H_phys

            if sat_sparse:
                self.tb_unkronned = sp.lm_to_sp(self.tR.reshape(self.nn,1))
                self.ta_unkronned = sp.lm_to_sp(self.tL.reshape(self.nn,1))
                self.tbT_unkronned = sp.lm_to_sp(self.tR.reshape(1,self.nn))
                self.taT_unkronned = sp.lm_to_sp(self.tL.reshape(1,self.nn))
                self.tb = sp.kron_neq_lm(self.tb_unkronned, neq)
                self.ta = sp.kron_neq_lm(self.ta_unkronned, neq)
                self.tbT = sp.kron_neq_lm(self.tbT_unkronned, neq)
                self.taT = sp.kron_neq_lm(self.taT_unkronned, neq)
                E_unkron = sp.lm_to_sp(self.E)
                self.Esurf = sp.kron_neq_lm(E_unkron, neq)
            else:
                self.tb_unkronned = self.tR.reshape(self.nn,1)
                self.ta_unkronned = self.tL.reshape(self.nn,1)
                self.tbT_unkronned = self.tb_unkronned.T
                self.taT_unkronned = self.ta_unkronned.T
                self.tb = fn.kron_neq_lm(self.tb_unkronned, neq)
                self.ta = fn.kron_neq_lm(self.ta_unkronned, neq)
                self.tbT = self.tb.T
                self.taT = self.ta.T
                self.Esurf = fn.kron_neq_lm(self.E, neq)
            
            # remember in 1D metrics = 1
            if sparse:
                D = sp.lm_to_sp(self.D)
                Dx = sp.gdiag_lm(mesh.det_jac_inv, D)
                self.Dx = sp.kron_gm_eye(Dx, neq)
                if disc_type == 'div':
                    Volx = sp.prune_gm(sp.subtract_gm_gm(Dx, sp.gdiag_lm(0.5*self.H_inv_phys_unkronned, E_unkron)))
                    self.Volx = sp.kron_neq_gm(Volx, neq)
                elif disc_type == 'had':
                    self.Volx = sp.prune_gm(sp.subtract_gm_gm(sp.scalar_gm(2.,Dx), sp.gdiag_lm(self.H_inv_phys_unkronned, E_unkron))) # NOT Kronned
            else:
                Dx = fn.gdiag_lm(mesh.det_jac_inv, self.D)
                self.Dx = fn.kron_neq_gm(Dx, neq)
                if disc_type == 'div':
                    Volx = Dx - fn.gdiag_lm(0.5*self.H_inv_phys_unkronned, self.E)
                    self.Volx = fn.kron_neq_gm(Volx, neq)
                elif disc_type == 'had':
                    self.Volx = 2.*Dx - fn.gdiag_lm(self.H_inv_phys_unkronned, self.E) # NOT kronned
            
            if calc_nd_ops:
                # use unkronned for multi-dim Dx
                self.Dx_nd = Dx

        
        elif mesh.dim == 2:
            
            self.Hperp = np.diag(self.H)
            self.H_phys_unkronned = np.kron(self.Hperp,self.Hperp)[:,None] * mesh.det_jac
            self.H_inv_phys_unkronned = 1/self.H_phys_unkronned
            self.H_phys = fn.repeat_neq_gv(self.H_phys_unkronned,neq)
            self.H_inv_phys = 1/self.H_phys

            if sat_sparse:
                # using a sparse sat, so want to save certain SAT matrices
                tR = sp.lm_to_sp(self.tR.reshape(self.nn,1))
                tL = sp.lm_to_sp(self.tL.reshape(self.nn,1))
                tRT = sp.lm_to_sp(self.tR.reshape(1,self.nn))
                tLT = sp.lm_to_sp(self.tL.reshape(1,self.nn))
                self.txb_unkronned = sp.kron_lm_eye(tR, self.nn)
                self.txa_unkronned = sp.kron_lm_eye(tL, self.nn)
                self.tyb_unkronned = sp.kron_eye_lm(tR, self.nn, 1)
                self.tya_unkronned = sp.kron_eye_lm(tL, self.nn, 1)
                self.txbT_unkronned = sp.kron_lm_eye(tRT, self.nn)
                self.txaT_unkronned = sp.kron_lm_eye(tLT, self.nn)
                self.tybT_unkronned = sp.kron_eye_lm(tRT, self.nn, self.nn)
                self.tyaT_unkronned = sp.kron_eye_lm(tLT, self.nn, self.nn)
                self.txb = sp.kron_neq_lm(self.txb_unkronned, neq)
                self.txa = sp.kron_neq_lm(self.txa_unkronned, neq)
                self.tyb = sp.kron_neq_lm(self.tyb_unkronned, neq)
                self.tya = sp.kron_neq_lm(self.tya_unkronned, neq)
                self.txbT = sp.kron_neq_lm(self.txbT_unkronned, neq)
                self.txaT = sp.kron_neq_lm(self.txaT_unkronned, neq)
                self.tybT = sp.kron_neq_lm(self.tybT_unkronned, neq)
                self.tyaT = sp.kron_neq_lm(self.tyaT_unkronned, neq)
                Exsurf = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned,  self.Hperp), self.txbT_unkronned), 
                                       sp.lm_lm(sp.lm_ldiag(self.txa_unkronned, -self.Hperp), self.txaT_unkronned))
                Eysurf = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned,  self.Hperp), self.tybT_unkronned), 
                                       sp.lm_lm(sp.lm_ldiag(self.tya_unkronned, -self.Hperp), self.tyaT_unkronned))
                self.Exsurf = sp.kron_neq_lm(Exsurf, neq)
                self.Eysurf = sp.kron_neq_lm(Eysurf, neq)

            # calculate and save the important volume operators
            if sparse:
                D = sp.lm_to_sp(self.D)
                Dx = sp.kron_lm_eye(D, self.nn)
                Dy = sp.kron_eye_lm(D, self.nn, self.nn)

                if form == 'skew_sym':
                    xm = 0 # l=x, m=x
                    ym = 2 # l=y, m=x
                    tempx = sp.add_gm_gm( sp.lm_gdiag(Dx,mesh.metrics[:,xm,:]), sp.gdiag_lm(mesh.metrics[:,xm,:],Dx) )
                    tempy = sp.add_gm_gm( sp.lm_gdiag(Dy,mesh.metrics[:,ym,:]), sp.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                    Dx_phys = sp.gdiag_gm(0.5/mesh.det_jac, sp.add_gm_gm(tempx, tempy))
                    Ex_phys1 = sp.add_gm_gm(sp.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]), sp.lm_gdiag(Eysurf,mesh.metrics[:,ym,:]))
                    if disc_type == 'div':
                        Volx = sp.prune_gm(sp.subtract_gm_gm(Dx_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ex_phys1)))
                        self.Volx = sp.kron_neq_gm(Volx, neq)
                    elif disc_type == 'had':
                        self.Volx = sp.prune_gm(sp.subtract_gm_gm(sp.scalar_gm(2.,Dx_phys), sp.gdiag_gm(self.H_inv_phys_unkronned, Ex_phys1))) # NOT Kronned
                    self.Dx = sp.kron_neq_gm(Dx_phys, neq)

                    if calc_nd_ops:
                        tempx = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned, self.Hperp * mesh.bdy_metrics[:,1,xm,:]), self.txbT_unkronned), 
                                              sp.lm_lm(sp.lm_ldiag(self.txa_unkronned,-self.Hperp * mesh.bdy_metrics[:,0,xm,:]), self.txaT_unkronned))
                        tempy = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned, self.Hperp * mesh.bdy_metrics[:,3,ym,:]), self.tybT_unkronned), 
                                              sp.lm_lm(sp.lm_ldiag(self.tya_unkronned,-self.Hperp * mesh.bdy_metrics[:,2,ym,:]), self.tyaT_unkronned))
                        Ex_phys_diff = sp.subtract_gm_gm(Ex_phys1, sp.add_gm_gm(tempx, tempy))
                        # use unkronned for multi-dim Dx
                        self.Dx_nd = sp.prune_gm(sp.subtract_gm_gm(Dx_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ex_phys_diff)))

                    xm = 1 # l=x, m=y
                    ym = 3 # l=y, m=y
                    tempx = sp.add_gm_gm( sp.lm_gdiag(Dx,mesh.metrics[:,xm,:]), sp.gdiag_lm(mesh.metrics[:,xm,:],Dx) )
                    tempy = sp.add_gm_gm( sp.lm_gdiag(Dy,mesh.metrics[:,ym,:]), sp.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                    Dy_phys = sp.gdiag_gm(0.5/mesh.det_jac, sp.add_gm_gm(tempx, tempy))
                    Ey_phys1 = sp.add_gm_gm(sp.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]), sp.lm_gdiag(Eysurf,mesh.metrics[:,ym,:]))
                    if disc_type == 'div':
                        Voly = sp.prune_gm(sp.subtract_gm_gm(Dy_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ey_phys1)))
                        self.Voly = sp.kron_neq_gm(Voly, neq)
                    elif disc_type == 'had':
                        self.Voly = sp.prune_gm(sp.subtract_gm_gm(sp.scalar_gm(2.,Dy_phys), sp.gdiag_gm(self.H_inv_phys_unkronned, Ey_phys1))) # NOT Kronned
                    self.Dy = sp.kron_neq_gm(Dy_phys, neq)

                    if calc_nd_ops:
                        tempx = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned, self.Hperp * mesh.bdy_metrics[:,1,xm,:]), self.txbT_unkronned), 
                                              sp.lm_lm(sp.lm_ldiag(self.txa_unkronned,-self.Hperp * mesh.bdy_metrics[:,0,xm,:]), self.txaT_unkronned))
                        tempy = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned, self.Hperp * mesh.bdy_metrics[:,3,ym,:]), self.tybT_unkronned), 
                                              sp.lm_lm(sp.lm_ldiag(self.tya_unkronned,-self.Hperp * mesh.bdy_metrics[:,2,ym,:]), self.tyaT_unkronned))
                        Ey_phys_diff = sp.subtract_gm_gm(Ey_phys1, sp.add_gm_gm(tempx, tempy))
                        # use unkronned for multi-dim Dy
                        self.Dx_nd = sp.prune_gm(sp.subtract_gm_gm(Dy_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ey_phys_diff)))
                else:
                    #TODO
                    raise Exception('Not implemented in sparse yet')
            
            else:
                # need to compute / recompute some things first
                tR = self.tR.reshape(self.nn,1)
                tL = self.tL.reshape(self.nn,1)
                txb = fn.kron_neq_lm(tR, self.nn)
                txa = fn.kron_neq_lm(tL, self.nn)
                tyb = np.kron(np.eye(self.nn), tR)
                tya = np.kron(np.eye(self.nn), tL)
                Exsurf = fn.lm_ldiag(txb, self.Hperp) @ txb.T - fn.lm_ldiag(txa, self.Hperp) @ txa.T
                Eysurf = fn.lm_ldiag(tyb, self.Hperp) @ tyb.T - fn.lm_ldiag(tya, self.Hperp) @ tya.T
                if (not sat_sparse):
                    # save the dense SAT matrices
                    self.txb_unkronned = txb
                    self.txa_unkronned = txa
                    self.tyb_unkronned = tyb
                    self.tya_unkronned = tya
                    self.txbT_unkronned = txb.T
                    self.txaT_unkronned = txa.T
                    self.tybT_unkronned = tyb.T
                    self.tyaT_unkronned = tya.T
                    self.txb = fn.kron_neq_lm(txb, neq)
                    self.txa = fn.kron_neq_lm(txa, neq)
                    self.tyb = fn.kron_neq_lm(tyb, neq)
                    self.tya = fn.kron_neq_lm(tya, neq)
                    self.txbT = self.txb.T
                    self.txaT = self.txa.T
                    self.tybT = self.tyb.T
                    self.tyaT = self.tya.T
                    self.Exsurf = fn.kron_neq_lm(Exsurf, neq)
                    self.Eysurf = fn.kron_neq_lm(Eysurf, neq)

                Dx = fn.kron_neq_lm(self.D, self.nn)
                Dy = np.kron(np.eye(self.nn), self.D)

                if form == 'skew_sym':
                    xm = 0 # l=x, m=x
                    ym = 2 # l=y, m=x
                    Dx_phys = fn.gdiag_gm(0.5/mesh.det_jac, fn.lm_gdiag(Dx,mesh.metrics[:,xm,:]) + fn.gdiag_lm(mesh.metrics[:,xm,:],Dx) \
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,ym,:]) + fn.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                    Ex_phys1 = fn.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]) + fn.lm_gdiag(Eysurf,mesh.metrics[:,ym,:])
                    if disc_type == 'div':
                        Volx = Dx_phys - fn.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ex_phys1)
                        self.Volx = fn.kron_neq_gm(Volx, neq)
                    elif disc_type == 'had':
                        self.Volx = 2.*Dx_phys - fn.gdiag_gm(self.H_inv_phys_unkronned, Ex_phys1) # NOT kronned
                    self.Dx = fn.kron_neq_gm(Dx_phys, neq)
                    


                    if calc_nd_ops:
                        Ex_phys2 = fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txb, self.Hperp), mesh.bdy_metrics[:,1,xm,:]), txb.T) - fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txa, self.Hperp), mesh.bdy_metrics[:,0,xm,:]), txa.T) \
                                 + fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txb, self.Hperp), mesh.bdy_metrics[:,3,ym,:]), tyb.T) - fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(tya, self.Hperp), mesh.bdy_metrics[:,2,ym,:]), tya.T) 
                        self.Dx_nd = Dx_phys - fn.gdiag_gm(0.5*self.H_inv_phys_unkronned, (Ex_phys1 - Ex_phys2))
                    
                    xm = 1 # l=x, m=x
                    ym = 3 # l=y, m=x
                    Dy_phys = fn.gdiag_gm(0.5/mesh.det_jac, fn.lm_gdiag(Dx,mesh.metrics[:,xm,:]) + fn.gdiag_lm(mesh.metrics[:,xm,:],Dx) \
                                                          + fn.lm_gdiag(Dy,mesh.metrics[:,ym,:]) + fn.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                    Ey_phys1 = fn.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]) + fn.lm_gdiag(Eysurf,mesh.metrics[:,ym,:])
                    if disc_type == 'div':
                        Voly = Dy_phys - fn.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ey_phys1)
                        self.Voly = fn.kron_neq_gm(Voly, neq)
                    elif disc_type == 'had':
                        self.Voly = 2.*Dy_phys - fn.gdiag_gm(self.H_inv_phys_unkronned, Ey_phys1) # NOT kronned
                    self.Dy = fn.kron_neq_gm(Dy_phys, neq)
                    
                    if calc_nd_ops:
                        Ey_phys2 = fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txb, self.Hperp), mesh.bdy_metrics[:,1,xm,:]), txb.T) - fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txa, self.Hperp), mesh.bdy_metrics[:,0,xm,:]), txa.T) \
                                 + fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(txb, self.Hperp), mesh.bdy_metrics[:,3,ym,:]), tyb.T) - fn.gm_lm(fn.lm_gdiag(fn.lm_ldiag(tya, self.Hperp), mesh.bdy_metrics[:,2,ym,:]), tya.T) 
                        self.Dy_nd = Dy_phys - fn.gdiag_gm(0.5*self.H_inv_phys_unkronned, (Ey_phys1 - Ey_phys2))
                
                elif form == 'div': # not provably stable
                    self.Dx = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,0,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,2,:])))
                    self.Dy = fn.gdiag_gm(mesh.det_jac_inv, (fn.lm_gdiag(Dx,mesh.metrics[:,1,:]) + fn.lm_gdiag(Dy,mesh.metrics[:,3,:])))
                    self.Dx_nd = None
                    self.Dy_nd = None
                
                else:
                    raise Exception('Physical operator form not understood.')

        
        elif mesh.dim == 3:

            if not sparse and not sat_sparse:
                raise Exception('3D operators not implemented for dense matrices.')

            H = np.diag(self.H)
            self.Hperp = np.kron(H,H)
            self.H_phys_unkronned = np.kron(H,self.Hperp)[:,None] * mesh.det_jac
            self.H_inv_phys_unkronned = 1/self.H_phys_unkronned
            self.H_phys = fn.repeat_neq_gv(self.H_phys_unkronned,neq)
            self.H_inv_phys = 1/self.H_phys

            # using a sparse sat, so want to save certain SAT matrices
            tR = sp.lm_to_sp(self.tR.reshape(self.nn,1))
            tL = sp.lm_to_sp(self.tL.reshape(self.nn,1))
            tRT = sp.lm_to_sp(self.tR.reshape(1,self.nn))
            tLT = sp.lm_to_sp(self.tL.reshape(1,self.nn))
            self.txb_unkronned = sp.kron_lm_eye(sp.kron_lm_eye(tR, self.nn), self.nn)
            self.txa_unkronned = sp.kron_lm_eye(sp.kron_lm_eye(tL, self.nn), self.nn)
            self.tyb_unkronned = sp.kron_lm_eye(sp.kron_eye_lm(tR, self.nn, 1), self.nn)
            self.tya_unkronned = sp.kron_lm_eye(sp.kron_eye_lm(tL, self.nn, 1), self.nn)
            self.tzb_unkronned = sp.kron_eye_lm(sp.kron_eye_lm(tR, self.nn, 1), self.nn, self.nn)
            self.tza_unkronned = sp.kron_eye_lm(sp.kron_eye_lm(tL, self.nn, 1), self.nn, self.nn)
            self.txbT_unkronned = sp.kron_lm_eye(sp.kron_lm_eye(tRT, self.nn), self.nn)
            self.txaT_unkronned = sp.kron_lm_eye(sp.kron_lm_eye(tLT, self.nn), self.nn)
            self.tybT_unkronned = sp.kron_lm_eye(sp.kron_eye_lm(tRT, self.nn, self.nn), self.nn)
            self.tyaT_unkronned = sp.kron_lm_eye(sp.kron_eye_lm(tLT, self.nn, self.nn), self.nn)
            self.tzbT_unkronned = sp.kron_eye_lm(sp.kron_eye_lm(tRT, self.nn, self.nn), self.nn, self.nn**2)
            self.tzaT_unkronned = sp.kron_eye_lm(sp.kron_eye_lm(tLT, self.nn, self.nn), self.nn, self.nn**2)
            self.txb = sp.kron_neq_lm(self.txb_unkronned, neq)
            self.txa = sp.kron_neq_lm(self.txa_unkronned, neq)
            self.tyb = sp.kron_neq_lm(self.tyb_unkronned, neq)
            self.tya = sp.kron_neq_lm(self.tya_unkronned, neq)
            self.tzb = sp.kron_neq_lm(self.tzb_unkronned, neq)
            self.tza = sp.kron_neq_lm(self.tza_unkronned, neq)
            self.txbT = sp.kron_neq_lm(self.txbT_unkronned, neq)
            self.txaT = sp.kron_neq_lm(self.txaT_unkronned, neq)
            self.tybT = sp.kron_neq_lm(self.tybT_unkronned, neq)
            self.tyaT = sp.kron_neq_lm(self.tyaT_unkronned, neq)
            self.tzbT = sp.kron_neq_lm(self.tzbT_unkronned, neq)
            self.tzaT = sp.kron_neq_lm(self.tzaT_unkronned, neq)
            Exsurf = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned,  self.Hperp), self.txbT_unkronned), 
                                    sp.lm_lm(sp.lm_ldiag(self.txa_unkronned, -self.Hperp), self.txaT_unkronned))
            Eysurf = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned,  self.Hperp), self.tybT_unkronned), 
                                    sp.lm_lm(sp.lm_ldiag(self.tya_unkronned, -self.Hperp), self.tyaT_unkronned))
            Ezsurf = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tzb_unkronned,  self.Hperp), self.tzbT_unkronned),
                                    sp.lm_lm(sp.lm_ldiag(self.tza_unkronned, -self.Hperp), self.tzaT_unkronned))
            self.Exsurf = sp.kron_neq_lm(Exsurf, neq)
            self.Eysurf = sp.kron_neq_lm(Eysurf, neq)
            self.Ezsurf = sp.kron_neq_lm(Ezsurf, neq)

            # calculate and save the important volume operators
            D = sp.lm_to_sp(self.D)
            Dx = sp.kron_lm_eye(sp.kron_lm_eye(D, self.nn), self.nn)
            Dy = sp.kron_lm_eye(sp.kron_eye_lm(D, self.nn, self.nn), self.nn)
            Dz = sp.kron_eye_lm(sp.kron_eye_lm(D, self.nn, self.nn), self.nn, self.nn**2)

            if form == 'skew_sym':
                # TODO Here
                raise Exception('3D Not completed yet')
                xm = 0 # l=x, m=x
                ym = 2 # l=y, m=x
                tempx = sp.add_gm_gm( sp.lm_gdiag(Dx,mesh.metrics[:,xm,:]), sp.gdiag_lm(mesh.metrics[:,xm,:],Dx) )
                tempy = sp.add_gm_gm( sp.lm_gdiag(Dy,mesh.metrics[:,ym,:]), sp.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                Dx_phys = sp.gdiag_gm(0.5/mesh.det_jac, sp.add_gm_gm(tempx, tempy))
                Ex_phys1 = sp.add_gm_gm(sp.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]), sp.lm_gdiag(Eysurf,mesh.metrics[:,ym,:]))
                if disc_type == 'div':
                    Volx = sp.prune_gm(sp.subtract_gm_gm(Dx_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ex_phys1)))
                    self.Volx = sp.kron_neq_gm(Volx, neq)
                elif disc_type == 'had':
                    self.Volx = sp.prune_gm(sp.subtract_gm_gm(sp.scalar_gm(2.,Dx_phys), sp.gdiag_gm(self.H_inv_phys_unkronned, Ex_phys1))) # NOT Kronned
                self.Dx = sp.kron_neq_gm(Dx_phys, neq)

                if calc_nd_ops:
                    tempx = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned, self.Hperp * mesh.bdy_metrics[:,1,xm,:]), self.txbT_unkronned), 
                                            sp.lm_lm(sp.lm_ldiag(self.txa_unkronned,-self.Hperp * mesh.bdy_metrics[:,0,xm,:]), self.txaT_unkronned))
                    tempy = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned, self.Hperp * mesh.bdy_metrics[:,3,ym,:]), self.tybT_unkronned), 
                                            sp.lm_lm(sp.lm_ldiag(self.tya_unkronned,-self.Hperp * mesh.bdy_metrics[:,2,ym,:]), self.tyaT_unkronned))
                    Ex_phys_diff = sp.subtract_gm_gm(Ex_phys1, sp.add_gm_gm(tempx, tempy))
                    # use unkronned for multi-dim Dx
                    self.Dx_nd = sp.prune_gm(sp.subtract_gm_gm(Dx_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ex_phys_diff)))

                xm = 1 # l=x, m=y
                ym = 3 # l=y, m=y
                tempx = sp.add_gm_gm( sp.lm_gdiag(Dx,mesh.metrics[:,xm,:]), sp.gdiag_lm(mesh.metrics[:,xm,:],Dx) )
                tempy = sp.add_gm_gm( sp.lm_gdiag(Dy,mesh.metrics[:,ym,:]), sp.gdiag_lm(mesh.metrics[:,ym,:],Dy) )
                Dy_phys = sp.gdiag_gm(0.5/mesh.det_jac, sp.add_gm_gm(tempx, tempy))
                Ey_phys1 = sp.add_gm_gm(sp.lm_gdiag(Exsurf,mesh.metrics[:,xm,:]), sp.lm_gdiag(Eysurf,mesh.metrics[:,ym,:]))
                if disc_type == 'div':
                    Voly = sp.prune_gm(sp.subtract_gm_gm(Dy_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ey_phys1)))
                    self.Voly = sp.kron_neq_gm(Voly, neq)
                elif disc_type == 'had':
                    self.Voly = sp.prune_gm(sp.subtract_gm_gm(sp.scalar_gm(2.,Dy_phys), sp.gdiag_gm(self.H_inv_phys_unkronned, Ey_phys1))) # NOT Kronned
                self.Dy = sp.kron_neq_gm(Dy_phys, neq)

                if calc_nd_ops:
                    tempx = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.txb_unkronned, self.Hperp * mesh.bdy_metrics[:,1,xm,:]), self.txbT_unkronned), 
                                            sp.lm_lm(sp.lm_ldiag(self.txa_unkronned,-self.Hperp * mesh.bdy_metrics[:,0,xm,:]), self.txaT_unkronned))
                    tempy = sp.add_lm_lm( sp.lm_lm(sp.lm_ldiag(self.tyb_unkronned, self.Hperp * mesh.bdy_metrics[:,3,ym,:]), self.tybT_unkronned), 
                                            sp.lm_lm(sp.lm_ldiag(self.tya_unkronned,-self.Hperp * mesh.bdy_metrics[:,2,ym,:]), self.tyaT_unkronned))
                    Ey_phys_diff = sp.subtract_gm_gm(Ey_phys1, sp.add_gm_gm(tempx, tempy))
                    # use unkronned for multi-dim Dy
                    self.Dx_nd = sp.prune_gm(sp.subtract_gm_gm(Dy_phys, sp.gdiag_gm(0.5*self.H_inv_phys_unkronned, Ey_phys_diff)))
            else:
                #TODO
                raise Exception('Not implemented yet')

    @staticmethod
    def check_diagH(x,H,tol=1e-10,returndegree=False):
        ''' tests order of quadrature for H, (j+1)*H*x^j = b^(j+1)-a(j+1). 
        Based on reference element [0,1] '''
        p=0
        er=0
        while er<tol:
            p+=1
            er = abs((p+1)*(np.diag(H) @ x**p) - 1)
        if returndegree:
            return int(p-1)
        else:
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
    def check_degree(D, x, tol=1e-10):
        ''' tests degree of derivative D, D@x^j = j*x^(j-1) 
        Based on reference element [0,1] '''
        p=1
        er=0
        while er<tol:
            p+=1
            er = np.sum(abs(D @ x**p - p*x**(p-1)))
        print('Test: Derivative D is degree {0}.'.format(p-1))

    @staticmethod
    def check_accuracy(D, x):
        ''' tests degree of derivative D, D@u - dudx = O(h^p) 
        Based on reference element [0,1] '''
        er1 = np.mean(np.abs(D @ np.sin(0.5*x+0.1) - 0.5*np.cos(0.5*x+0.1)))
        er2 = np.mean(np.abs(D @ np.sin(0.25*x+0.1) - 0.25*np.cos(0.25*x+0.1)))
        er3 = np.mean(np.abs(D @ np.sin(0.125*x+0.1) - 0.125*np.cos(0.125*x+0.1)))
        o1 = (np.log(er2) - np.log(er1)) / (np.log(0.25) - np.log(0.5))
        o2 = (np.log(er3) - np.log(er2)) / (np.log(0.125) - np.log(0.25))
        print('Test: Derivative D is order {0:.2} in test 1, {1:.2} in test 2. (element-refinement, so should expect p+1)'.format(o1,o2))