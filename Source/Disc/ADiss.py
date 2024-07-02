#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2024

@author: bercik
"""
#import numpy as np

import numpy as np
import Source.Methods.Functions as fn
from Source.Disc.DissOp import BaselineDiss, make_dcp_diss_op


class ADiss():

    '''
    Creates artificial dissipation operators and functions
    '''

    def __init__(self, solver):
        '''
        Sets up the artificial dissipation function and reauired operators

        Parameters
        ----------
        solver : class instance
            The solver class which contains all the important functions.
            Must have attribute self.vol_diss that contains all the 
            necessary information about the desired dissipation

        type =  'ND': No Dissipation
                'W': Simplest implementation with repeating D^s operator
                'entW': Entropy-stable implementation of the W operator
                'B': Baseline operator constructed using the framework of DCP 2018
                'entB': Entropy-stable implementation of the B operator
                'BM': Leading truncation error term of stencil immediately adjacent to the boundary matches that of the interior stencil
                'entBM': Entropy-stable implementation of the BM operator
                'MSN': Constructed by modifying the approach in Mattsson 2004 to include a variable coefficient and consistent dimensions
                'SMSN': Constructed using the new framework to give the same coe cients as the MSN operators for the constant-coefficient case
        '''
        
        print('... Setting up Artificial Dissipation')
        
        self.solver = solver
        self.dim = self.solver.dim
        self.nelem = self.solver.nelem
        self.neq_node = self.solver.neq_node
        self.nen = self.solver.nen

        self.type = self.solver.vol_diss['diss_type']
        if self.type.lower() == 'nd':
            self.dissipation = self.no_diss
            return
        else:
            self.maxeig_dEndq = self.solver.diffeq.maxeig_dEndq

            if 'jac_type' in self.solver.vol_diss.keys():
                self.jac_type = self.solver.vol_diss['jac_type']
            else:
                self.jac_type = None
            
            if not (self.solver.disc_nodes.lower() == 'upwind' and self.type.lower() == 'upwind'):
                if 's' in self.solver.vol_diss.keys():
                    self.s = self.solver.vol_diss['s']
                    if self.s is None:
                        print('WARNING: No s provided to artificial dissipation. Defaulting to s=p')
                        self.s = self.solver.p
                    elif self.s == 'p':
                        self.s = self.solver.p
                    elif self.s == 'p+1':
                        self.s = self.solver.p + 1
                    elif self.s == '2p-1':
                        self.s = 2*self.solver.p - 1
                    elif self.s == '2p':
                        self.s = 2*self.solver.p
                    elif self.s == '2p+1':
                        self.s = 2*self.solver.p + 1
                    assert isinstance(self.s, int), 'Artificial Dissipation: s must be an integer, {0}'.format(self.s)
                else:
                    print('WARNING: No s provided to artificial dissipation. Defaulting to s=p')
                    self.s = self.solver.p

            if 'coeff' in  self.solver.vol_diss.keys():
                assert isinstance(self.solver.vol_diss['coeff'], float), 'Artificial Dissipation: coeff must be a float, {0}'.format(self.solver.vol_diss['coeff'])
                self.coeff = self.solver.vol_diss['coeff']
            else:
                self.coeff = 0.1

            if self.type.lower() == 'upwind':
                print("WARNING: upwind volume dissipation is experimental and only provably stable for linear, constant-coeff. equations.")
                if 'fluxvec' in self.solver.vol_diss.keys():
                    self.fluxvec = self.solver.vol_diss['fluxvec']
                else:
                    print('WARNING: No fluxvec provided to artificial dissipation. Defaulting to fluxvec=lf')
                    self.fluxvec = 'lf'
                if self.coeff != 1.:
                    print(f'WARNING: coeff = {self.coeff} != 1. Make sure this is intentional as it is not a typical flux-vector splitting.')


        if self.dim == 1:
            self.dxidx = self.solver.mesh.metrics[:,0,:]
            self.set_ops_1D()
        elif self.dim == 2:
            self.dxidx = self.solver.mesh.metrics[:,:2,:]
            self.detadx = self.solver.mesh.metrics[:,2:,:]
            self.set_ops_2D()
        elif self.dim == 3:
            self.dxidx = self.solver.mesh.metrics[:,:3,:]
            self.detadx = self.solver.mesh.metrics[:,3:6,:]
            self.dzetadx = self.solver.mesh.metrics[:,6:,:]

        if self.type.lower() == 'w':
            if self.jac_type.lower() == 'scalar':
                self.dissipation = lambda q: self.dissipation_W_scalar(q,self.coeff)
            else:
                print("WARNING: Only scalar dissipation set up for type='W' dissipation. Defaulting to jac_type = 'scalar'.")
                self.jac_type = 'scalar'
                self.dissipation = lambda q: self.dissipation_W_scalar(q,self.coeff)

        elif self.type.lower() == 'entw':
            self.entropy_var = self.solver.diffeq.entropy_var
            self.dqdw = self.solver.diffeq.dqdw
            if self.neq_node == 1:
                assert((self.jac_type.lower() == 'scalar') or (self.jac_type.lower() == 'scalar')),'scalar equations must have scalar dissipation'
                self.dissipation = lambda q: self.dissipation_entW_scalar(q,self.coeff)
            else:
                if self.jac_type.lower() == 'scalarscalar':
                    self.dissipation = lambda q: self.dissipation_entW_scalarscalar(q,self.coeff)
                elif self.jac_type.lower() == 'scalarmatrix':
                    self.dissipation = lambda q: self.dissipation_entW_scalarmatrix(q,self.coeff)
                elif self.jac_type.lower() == 'matrixmatrix':
                    self.dissipation = lambda q: self.dissipation_entW_matrixmatrix(q,self.coeff)
                    self.dEndw_abs = self.solver.diffeq.dEndw_abs
                else:
                    print("WARNING: jac method not understood for type='entW' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = lambda q: self.dissipation_entW_scalarscalar(q,self.coeff)


        elif self.type.lower() == 'b':
            if self.jac_type.lower() == 'scalar':
                self.dissipation = lambda q: self.dissipation_B_scalar(q,self.coeff)
            else:
                print("WARNING: Only scalar dissipation set up for type='B' dissipation. Defaulting to jac_type = 'scalar'.")
                self.jac_type = 'scalar'
                self.dissipation = lambda q: self.dissipation_B_scalar(q,self.coeff)
        
        elif self.type.lower() == 'entb':
            self.entropy_var = self.solver.diffeq.entropy_var
            self.dqdw = self.solver.diffeq.dqdw
            if self.neq_node == 1:
                assert((self.jac_type.lower() == 'scalar') or (self.jac_type.lower() == 'scalar')),'scalar equations must have scalar dissipation'
                self.dissipation = lambda q: self.dissipation_entB_scalar(q,self.coeff)
            else:
                if self.jac_type.lower() == 'scalarscalar':
                    self.dissipation = lambda q: self.dissipation_entB_scalarscalar(q,self.coeff)
                elif self.jac_type.lower() == 'scalarmatrix':
                    self.dissipation = lambda q: self.dissipation_entB_scalarmatrix(q,self.coeff)
                elif self.jac_type.lower() == 'matrixmatrix':
                    self.dissipation = lambda q: self.dissipation_entB_matrixmatrix(q,self.coeff)
                    self.dEndw_abs = self.solver.diffeq.dEndw_abs
                else:
                    print("WARNING: jac method not understood for type='entB' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = lambda q: self.dissipation_entB_scalarscalar(q,self.coeff)
        
        elif self.type.lower() == 'dcp':
            if self.jac_type.lower() == 'scalar':
                self.dissipation = lambda q: self.dissipation_dcp_scalar(q,self.coeff)
            else:
                print("WARNING: Only scalar dissipation set up for type='DCP' dissipation. Defaulting to jac_type = 'scalar'.")
                self.jac_type = 'scalar'
                self.dissipation = lambda q: self.dissipation_dcp_scalar(q,self.coeff)

        elif self.type.lower() == 'entdcp':
            self.entropy_var = self.solver.diffeq.entropy_var
            self.dqdw = self.solver.diffeq.dqdw
            if self.neq_node == 1:
                assert((self.jac_type.lower() == 'scalar') or (self.jac_type.lower() == 'scalar')),'scalar equations must have scalar dissipation'
                self.dissipation = lambda q: self.dissipation_entdcp_scalar(q,self.coeff)
            else:
                if self.jac_type.lower() == 'scalarscalar':
                    self.dissipation = lambda q: self.dissipation_entdcp_scalarscalar(q,self.coeff)
                elif self.jac_type.lower() == 'scalarmatrix':
                    self.dissipation = lambda q: self.dissipation_entdcp_scalarmatrix(q,self.coeff)
                elif self.jac_type.lower() == 'matrixmatrix':
                    self.dissipation = lambda q: self.dissipation_entdcp_matrixmatrix(q,self.coeff)
                    self.dEndw_abs = self.solver.diffeq.dEndw_abs
                else:
                    print("WARNING: jac method not understood for type='entDCP' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = lambda q: self.dissipation_entdcp_scalarscalar(q,self.coeff)

        elif self.type.lower() == 'upwind':
            if self.fluxvec.lower() == 'lf':
                self.dissipation = lambda q: self.dissipation_upwind_lf(q,self.coeff)
            elif self.fluxvec.lower() == 'sw' or self.fluxvec.lower()=='stegerwarming':
                if self.dim == 1: 
                    from Source.DiffEq.EulerFunctions import StegerWarming_diss_1D
                    self.stegerwarming = StegerWarming_diss_1D
                elif self.dim == 2: 
                    from Source.DiffEq.EulerFunctions import StegerWarming_diss_2D
                    self.stegerwarming = StegerWarming_diss_2D
                self.dissipation = lambda q: self.dissipation_upwind_stegerwarming(q,self.coeff)
            else:
                print("WARNING: fluxvec method not understood. Defaulting to fluxvec = 'lf'.")
                self.fluxvec = 'lf'
                self.dissipation = lambda q: self.dissipation_upwind_lf(q,self.coeff)
        
        else:
            raise Exception('Artificial dissipation: diss_type not understood, '+ str(self.type))


    def set_ops_1D(self):
        ''' prepare the various operators needed for the dissipation function '''
        # xavg = self.solver.mesh.dom_len/self.nelem/(self.nen-1) # this was used in the original work
        #xavg = (self.solver.mesh.bdy_x[1,:]-self.solver.mesh.bdy_x[0,:])/(self.nen-1)
        xavg = (1.-0.)/(self.nen-1) # this is the reference spacing. Physical spacing is taken care of implicitly by metrics.

        if self.type.lower() == 'w' or self.type.lower() == 'entw':
            #Ds = np.copy(self.solver.Dx_phys_nd)
            Ds = np.copy(self.solver.sbp.D)
            for i in range(1,self.s):
                #Ds = fn.gm_gm(self.solver.Dx_phys_nd,Ds)
                Ds = self.solver.sbp.D @ Ds
            #DsT = np.transpose(Ds,axes=(1,0,2))
            DsT = Ds.T
            
            self.rhs_D = fn.kron_neq_lm(Ds,self.neq_node) 
            #self.lhs_D = fn.gdiag_gm(-(self.solver.H_inv_phys * xavg**(2*self.s-1)), DsT * self.solver.H_phys)
            self.lhs_D = fn.gdiag_lm(-(self.solver.H_inv_phys * xavg**(2*self.s-1)),fn.kron_neq_lm(DsT @ self.solver.sbp.H,self.neq_node))
        elif self.type.lower() == 'dcp' or self.type.lower() == 'entdcp':
            Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen)
            self.rhs_D = fn.kron_neq_lm(Ds,self.neq_node) 
            self.lhs_D = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(Ds.T @ np.diag(B) @ self.solver.sbp.H, self.neq_node))
        elif self.type.lower() == 'upwind':
            if self.solver.disc_nodes.lower() == 'upwind':
                Ddiss = self.solver.sbp.Ddiss
            else:
                from Source.Disc.UpwindOp import UpwindOp
                _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindOp(self.s,self.nen)
                if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                    print('WARNING: x of sbp operator does not match x of dissipation operator!')
                    print(self.solver.sbp.x)
                    print(x)
                if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                    print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
            self.Ddiss = fn.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), fn.kron_neq_lm(Ddiss,self.neq_node))
        elif self.type.lower() == 'b' or self.type.lower() == 'entb':
            D = BaselineDiss(self.s, self.nen)
            if self.s == 1:
                D.updateD1()
                D.updateD2()
                D.updateB1()
                D.updateB2()
                self.rhs_D1 = fn.kron_neq_lm(D.D1,self.neq_node) 
                self.rhs_D2 = fn.kron_neq_lm(D.D2,self.neq_node)
                self.lhs_D1 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D1.T @ D.B1 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D2 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D2.T @ D.B2 @ self.solver.sbp.H, self.neq_node))
            elif self.s == 2:
                D.updateD1()
                D.updateD3()
                D.updateD4()
                D.updateB1()
                D.updateB3()
                D.updateB4()
                D2 = D.D1 @ D.D1
                self.rhs_D2 = fn.kron_neq_lm(D2,self.neq_node) 
                self.rhs_D3 = fn.kron_neq_lm(D.D3,self.neq_node)
                self.rhs_D4 = fn.kron_neq_lm(D.D4,self.neq_node)
                self.lhs_D2 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D2.T @ D.B1 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D3 = 0.5 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D3.T @ D.B3 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D4 = 0.0625 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D4.T @ D.B4 @ self.solver.sbp.H, self.neq_node))
            elif self.s == 3:
                D.updateD1()
                D.updateD4()
                D.updateD5()
                D.updateD6()
                D.updateB1()
                D.updateB4()
                D.updateB5()
                D.updateB6()
                D3 = D.D1 @ D.D1 @ D.D1
                self.rhs_D3 = fn.kron_neq_lm(D3,self.neq_node) 
                self.rhs_D4 = fn.kron_neq_lm(D.D4,self.neq_node)
                self.rhs_D5 = fn.kron_neq_lm(D.D5,self.neq_node)
                self.rhs_D6 = fn.kron_neq_lm(D.D6,self.neq_node)
                self.lhs_D3 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D3.T @ D.B1 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D4 = 0.75 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D4.T @ D.B4 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D5 = 0.3125 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D5.T @ D.B5 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D6 = (1./96.) * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D6.T @ D.B6 @ self.solver.sbp.H, self.neq_node))
            elif self.s == 4:
                D.updateD1()
                D.updateD5()
                D.updateD6()
                D.updateD7()
                D.updateD8()
                D.updateB1()
                D.updateB5()
                D.updateB6()
                D.updateB7()
                D.updateB8()
                D4 = D.D1 @ D.D1 @ D.D1 @ D.D1
                self.rhs_D4 = fn.kron_neq_lm(D4,self.neq_node) 
                self.rhs_D5 = fn.kron_neq_lm(D.D5,self.neq_node)
                self.rhs_D6 = fn.kron_neq_lm(D.D6,self.neq_node)
                self.rhs_D7 = fn.kron_neq_lm(D.D7,self.neq_node)
                self.rhs_D8 = fn.kron_neq_lm(D.D8,self.neq_node)
                self.lhs_D4 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D4.T @ D.B1 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D5 = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D5.T @ D.B5 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D6 = 0.875 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D6.T @ D.B6 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D7 = 0.0875 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D7.T @ D.B7 @ self.solver.sbp.H, self.neq_node))
                self.lhs_D8 = 0.00171875 * fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(D.D8.T @ D.B8 @ self.solver.sbp.H, self.neq_node))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')

        else:
            raise Exception(self.type + ' not set up yet')
        
    def set_ops_2D(self):
        ''' prepare the various operators needed for the dissipation function '''
        # xavg = self.solver.mesh.dom_len/self.nelem/(self.nen-1) # this was used in the original work
        #xavg = (self.solver.mesh.bdy_x[1,:]-self.solver.mesh.bdy_x[0,:])/(self.nen-1)
        xavg = (1.-0.)/(self.nen-1) # this is the reference spacing. Physical spacing is taken care of implicitly by metrics.
        # NOTE: assumes D is the same in each direction

        if self.type == 'W' or self.type == 'entW':
            Ds = np.copy(self.solver.sbp.D)
            for i in range(1,self.s):
                Ds = self.solver.sbp.D @ Ds
            DsT = Ds.T
            
            eye = np.eye(self.mesh.nen)
            self.rhs_Dxi = fn.kron_neq_lm(np.kron(Ds, eye),self.neq_node) 
            self.rhs_Deta = fn.kron_neq_lm(np.kron(eye, Ds),self.neq_node) 
            DsTxi = np.kron(DsT @ self.solver.sbp.H, eye)
            DsTeta = np.kron(eye, DsT @ self.solver.sbp.H)
            self.lhs_Dxi = fn.kron_neq_gm(fn.gdiag_lm(-(self.solver.H_inv_phys * xavg**(2*self.s-1)),DsTxi),self.neq_node) 
            self.lhs_Deta = fn.kron_neq_gm(fn.gdiag_lm(-(self.solver.H_inv_phys * xavg**(2*self.s-1)),DsTeta),self.neq_node) 
        elif self.type == 'dcp':
            Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen)
            eye = np.eye(self.mesh.nen)
            self.rhs_Dxi = fn.kron_neq_lm(np.kron(Ds, eye),self.neq_node) 
            self.rhs_Deta = fn.kron_neq_lm(np.kron(eye, Ds),self.neq_node) 
            DsTxi = np.kron(Ds.T @ np.diag(B) @ self.solver.sbp.H, eye)
            DsTeta = np.kron(eye, Ds.T @ np.diag(B) @ self.solver.sbp.H)
            self.lhs_Dxi = fn.kron_neq_gm(fn.gdiag_lm(-(self.solver.H_inv_phys / xavg),DsTxi),self.neq_node) 
            self.lhs_Deta = fn.kron_neq_gm(fn.gdiag_lm(-(self.solver.H_inv_phys / xavg),DsTeta),self.neq_node) 
        elif self.type.lower() == 'upwind':
            if self.solver.disc_nodes.lower() == 'upwind':
                Ddiss = self.solver.sbp.Ddiss
            else:
                from Source.Disc.UpwindOp import UpwindOp
                _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindOp(self.s,self.nen)
                if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                    print('WARNING: x of sbp operator does not match x of dissipation operator!')
                    print(self.solver.sbp.x)
                    print(x)
                if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                    print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
            eye = np.eye(self.mesh.nen)
            self.Dxidiss = fn.gdiag_lm( fn.repeat_neq_gv(-self.solver.det_jac_inv), fn.kron_neq_lm(np.kron(Ddiss, eye),self.neq_node)) 
            self.Detadiss = fn.gdiag_lm( fn.repeat_neq_gv(-self.solver.det_jac_inv), fn.kron_neq_lm(np.kron(eye, Ddiss),self.neq_node))
        else:
            raise Exception(self.type + ' not set up yet')

    def no_diss(self, q):
        return 0
    
    def dissipation_upwind_lf(self, q, coeff=1.):
        ''' dissipation function for upwind / LF flux-vector splitting'''
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.Ddiss, A * q)
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.Dxidiss, A * q) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.Detadiss, A * q) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.Dxidiss, A * q) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.Detadiss, A * q) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.Dzetadiss, A * q) # zeta part
        return coeff*diss
    
    def dissipation_upwind_stegerwarming(self, q, coeff=1.):
        ''' dissipation function for upwind steger warming flux-vector splitting'''
        if self.dim == 1:
            fdiss = self.stegerwarming(q,self.dxidx)
            diss = fn.gm_gv(self.Ddiss, fdiss)
        elif self.dim == 2:
            fdiss = self.stegerwarming(q,self.dxidx)
            diss = fn.gm_gv(self.Dxidiss, fdiss) # xi part
            fdiss = self.stegerwarming(q,self.detadx)
            diss += fn.gm_gv(self.Detadiss, fdiss) # eta part
        elif self.dim == 3:
            diss = 0.
        return coeff*diss

    def dissipation_W_scalar(self, q, coeff=0.1):
        ''' dissipation function for wide interior stencils, scalar functions or systems'''
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ q))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Dzeta, A * (self.rhs_Dzeta @ q)) # zeta part
        return coeff*diss
    
    def dissipation_B_scalar(self, q, coeff=0.1):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar functions or systems'''
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            if self.s == 1:
                diss = fn.gm_gv(self.lhs_D1, A * (self.rhs_D1 @ q)) + \
                       fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ q))
            elif self.s == 2:
                diss = fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ q)) + \
                       fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ q)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ q))
            elif self.s == 3:
                diss = fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ q)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ q)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ q)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ q))
            elif self.s == 4:
                diss = fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ q)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ q)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ q)) + \
                       fn.gm_gv(self.lhs_D7, A * (self.rhs_D7 @ q)) + \
                       fn.gm_gv(self.lhs_D8, A * (self.rhs_D8 @ q))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_dcp_scalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, scalar functions or systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        avg_half_nodes = False
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ q))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Dzeta, A * (self.rhs_Dzeta @ q)) # zeta part
        return coeff*diss
    
    def dissipation_entW_scalar(self, q, coeff=0.1):
        ''' dissipation function for entB, scalar functions'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = maxeig*dqdw
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) 
            A = maxeig*dqdw
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) 
            A = maxeig*dqdw
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ w)) # eta part
        elif self.dim == 3:
            raise Exception('TODO')

        return coeff*diss
    
    def dissipation_entB_scalar(self, q, coeff=0.1):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar functions'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = maxeig*dqdw
            if self.s == 1:
                diss = fn.gm_gv(self.lhs_D1, A * (self.rhs_D1 @ w)) + \
                       fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ w))
            elif self.s == 2:
                diss = fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ w)) + \
                       fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w))
            elif self.s == 3:
                diss = fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ w))
            elif self.s == 4:
                diss = fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ w)) + \
                       fn.gm_gv(self.lhs_D7, A * (self.rhs_D7 @ w)) + \
                       fn.gm_gv(self.lhs_D8, A * (self.rhs_D8 @ w))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entdcp_scalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable scalar functions'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        avg_half_nodes = False
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            diss = fn.gm_gv(self.lhs_D, maxeig * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            diss = fn.gm_gv(self.lhs_Dxi, maxeig * (self.rhs_Dxi @ w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            diss += fn.gm_gv(self.lhs_Deta, maxeig * (self.rhs_Deta @ w)) # eta part
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entW_scalarscalar(self, q, coeff=0.1):
        ''' dissipation function for scalar-scalar entB, systems'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        rho = fn.spec_rad(dqdw,self.neq_node)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig*rho,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig*rho,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ w))
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = fn.repeat_neq_gv(maxeig*rho,self.neq_node)
            diss = fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ w))
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entB_scalarscalar(self, q, coeff=0.1):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar-scalar systems'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        rho = fn.spec_rad(dqdw,self.neq_node)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = fn.repeat_neq_gv(maxeig*rho,self.neq_node)
            if self.s == 1:
                diss = fn.gm_gv(self.lhs_D1, A * (self.rhs_D1 @ w)) + \
                       fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ w))
            elif self.s == 2:
                diss = fn.gm_gv(self.lhs_D2, A * (self.rhs_D2 @ w)) + \
                       fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w))
            elif self.s == 3:
                diss = fn.gm_gv(self.lhs_D3, A * (self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ w))
            elif self.s == 4:
                diss = fn.gm_gv(self.lhs_D4, A * (self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, A * (self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, A * (self.rhs_D6 @ w)) + \
                       fn.gm_gv(self.lhs_D7, A * (self.rhs_D7 @ w)) + \
                       fn.gm_gv(self.lhs_D8, A * (self.rhs_D8 @ w))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entdcp_scalarscalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        avg_half_nodes = False
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        rho = fn.spec_rad(dqdw,self.neq_node)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * rho
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * rho
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) * rho
            if self.s % 2 == 1 and avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ w)) # eta part
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entW_scalarmatrix(self, q, coeff=0.1):
        ''' dissipation function for scalar-matrix entB, systems'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.detadx),self.neq_node) * dqdw
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entB_scalarmatrix(self, q, coeff=0.1):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar-matrix systems'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            if self.s == 1:
                diss = fn.gm_gv(self.lhs_D1, fn.gm_gv(A, self.rhs_D1 @ w)) + \
                       fn.gm_gv(self.lhs_D2, fn.gm_gv(A, self.rhs_D2 @ w))
            elif self.s == 2:
                diss = fn.gm_gv(self.lhs_D2, fn.gm_gv(A, self.rhs_D2 @ w)) + \
                       fn.gm_gv(self.lhs_D3, fn.gm_gv(A, self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w))
            elif self.s == 3:
                diss = fn.gm_gv(self.lhs_D3, fn.gm_gv(A, self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, fn.gm_gv(A, self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, fn.gm_gv(A, self.rhs_D6 @ w))
            elif self.s == 4:
                diss = fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, fn.gm_gv(A, self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, fn.gm_gv(A, self.rhs_D6 @ w)) + \
                       fn.gm_gv(self.lhs_D7, fn.gm_gv(A, self.rhs_D7 @ w)) + \
                       fn.gm_gv(self.lhs_D8, fn.gm_gv(A, self.rhs_D8 @ w))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entdcp_scalarmatrix(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        avg_half_nodes = False
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.detadx),self.neq_node) * dqdw
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entW_matrixmatrix(self, q, coeff=0.1):
        ''' dissipation function for matrix-matrix entB, systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            A = self.dEndw_abs(q,self.dxidx)
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = self.dEndw_abs(q,self.dxidx)
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
        elif self.dim == 3:
            A = self.dEndw_abs(q,self.dxidx)
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
            A = self.dEndw_abs(q,self.dzetadx)
            diss += fn.gm_gv(self.lhs_Dzeta, fn.gm_gv(A, self.rhs_Dzeta @ w))
        return coeff*diss
    
    def dissipation_entB_matrixmatrix(self, q, coeff=0.1):
        ''' DCP 2018 dissipation function for narrow interior stencils, matrix-matrix systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s == 1:
                diss = fn.gm_gv(self.lhs_D1, fn.gm_gv(A, self.rhs_D1 @ w)) + \
                       fn.gm_gv(self.lhs_D2, fn.gm_gv(A, self.rhs_D2 @ w))
            elif self.s == 2:
                diss = fn.gm_gv(self.lhs_D2, fn.gm_gv(A, self.rhs_D2 @ w)) + \
                       fn.gm_gv(self.lhs_D3, fn.gm_gv(A, self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w))
            elif self.s == 3:
                diss = fn.gm_gv(self.lhs_D3, fn.gm_gv(A, self.rhs_D3 @ w)) + \
                       fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, fn.gm_gv(A, self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, fn.gm_gv(A, self.rhs_D6 @ w))
            elif self.s == 4:
                diss = fn.gm_gv(self.lhs_D4, fn.gm_gv(A, self.rhs_D4 @ w)) + \
                       fn.gm_gv(self.lhs_D5, fn.gm_gv(A, self.rhs_D5 @ w)) + \
                       fn.gm_gv(self.lhs_D6, fn.gm_gv(A, self.rhs_D6 @ w)) + \
                       fn.gm_gv(self.lhs_D7, fn.gm_gv(A, self.rhs_D7 @ w)) + \
                       fn.gm_gv(self.lhs_D8, fn.gm_gv(A, self.rhs_D8 @ w))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return coeff*diss
    
    def dissipation_entdcp_matrixmatrix(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        avg_half_nodes = False
        w = self.entropy_var(q)
        if self.dim == 1:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
        elif self.dim == 3:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
            A = self.dEndw_abs(q,self.dzetadx)
            if self.s % 2 == 1 and avg_half_nodes:
                A[:,:,:-1] = 0.5*(A[:,:,:-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Dzeta, fn.gm_gv(A, self.rhs_Dzeta @ w))
        return coeff*diss
    
    
    
    def get_LHS(self, q=None, step=1.0e-4):
        ''' could form explicitly... but for simplicity just do finite difference. '''
        if q is None:
            q = self.solver.diffeq.set_q0()
        nen = self.nen*self.neq_node  
        nelem = self.nelem
        assert((nen,nelem)==q.shape),"ERROR: sizes don't match"     
        nelem = 1 # only make it for the first element
        A = np.zeros((nen*nelem,nen*nelem))              
        for i in range(nen):
            for j in range(nelem):
                ei = np.zeros((nen,nelem))
                ei[i,j] = 1.*step
                q_r = self.dissipation(q+ei)[:,0] #.flatten('F')
                q_l = self.dissipation(q-ei)[:,0] #.flatten('F')
                idx = np.where(ei.flatten('F')>step/10)[0][0]
                A[:,idx] = (q_r - q_l)/(2*step)
        return A





        
