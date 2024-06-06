#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2024

@author: bercik
"""
#import numpy as np

import numpy as np
import Source.Methods.Functions as fn


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
            raise Exception('TODO')
        
        elif self.type.lower() == 'entb':
            raise Exception('TODO')
        
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
            Ds, B = self.make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen)
            self.rhs_D = fn.kron_neq_lm(Ds,self.neq_node) 
            self.lhs_D = fn.gdiag_lm(-(self.solver.H_inv_phys/xavg),fn.kron_neq_lm(Ds.T @ np.diag(B) @ self.solver.sbp.H,self.neq_node))
        elif self.type.lower() == 'upwind':
            if self.solver.disc_nodes.lower() == 'upwind':
                Ddiss = fn.kron_neq_lm(self.solver.sbp.Ddiss,self.neq_node) 
            else:
                from Source.Disc.UpwindOp import UpwindOp
                _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindOp(self.s,self.nen)
                if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                    print('WARNING: x of sbp operator does not match x of dissipation operator!')
                    print(self.solver.sbp.x)
                    print(x)
                if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                    print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
            self.Ddiss = fn.gdiag_lm( - self.solver.mesh.det_jac_inv, fn.kron_neq_lm(Ddiss,self.neq_node))
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
            Ds, B = self.make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen)
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
            self.Dxidiss = fn.gdiag_lm( - self.solver.mesh.det_jac_inv, fn.kron_neq_lm(np.kron(Ddiss, eye),self.neq_node)) 
            self.Detadiss = fn.gdiag_lm( - self.solver.mesh.det_jac_inv, fn.kron_neq_lm(np.kron(eye, Ddiss),self.neq_node))
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
    
    def dissipation_dcp_scalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, scalar functions or systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ q))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss += fn.gm_gv(self.lhs_Deta, A * (self.rhs_Deta @ q)) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            if self.s % 2 == 1:
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
    
    def dissipation_entdcp_scalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable scalar functions'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * dqdw
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            diss = fn.gm_gv(self.lhs_D, maxeig * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * dqdw
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            diss = fn.gm_gv(self.lhs_Dxi, maxeig * (self.rhs_Dxi @ w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) * dqdw
            if self.s % 2 == 1:
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
    
    def dissipation_entdcp_scalarscalar(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        rho = fn.spec_rad(dqdw,self.neq_node)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * rho
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_D, A * (self.rhs_D @ w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) * rho
            if self.s % 2 == 1:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = fn.repeat_neq_gv(maxeig,self.neq_node)
            diss = fn.gm_gv(self.lhs_Dxi, A * (self.rhs_Dxi @ w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) * rho
            if self.s % 2 == 1:
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
    
    def dissipation_entdcp_scalarmatrix(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx),self.neq_node) * dqdw
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = fn.repeat_neq_gv(self.maxeig_dEndq(q,self.detadx),self.neq_node) * dqdw
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
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
    
    def dissipation_entdcp_matrixmatrix(self, q, coeff=0.1):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        if self.dim == 1:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_D, fn.gm_gv(A, self.rhs_D @ w))
        elif self.dim == 2:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
        elif self.dim == 3:
            A = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss = fn.gm_gv(self.lhs_Dxi, fn.gm_gv(A, self.rhs_Dxi @ w))
            A = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Deta, fn.gm_gv(A, self.rhs_Deta @ w))
            A = self.dEndw_abs(q,self.dzetadx)
            if self.s % 2 == 1:
                A[:,:,-1] = 0.5*(A[:,:,-1] + A[:,:,1:])
            diss += fn.gm_gv(self.lhs_Dzeta, fn.gm_gv(A, self.rhs_Dzeta @ w))
        return coeff*diss
    
    def make_dcp_diss_op(self, sbp_type, s, nen):
        ''' make the relevant operators according to DCP implementation in diablo '''
        if sbp_type.lower() == 'csbp':
            # Initialize the matrix as a dense NumPy array
            Ds = np.zeros((nen, nen))
            B = np.ones(nen)

            if s==2:
                if nen < 3:
                    raise ValueError(f"Invalid number of nodes. nen = {nen}")

                # Row 1
                Ds[0, 0] = 1.0
                Ds[0, 1] = -2.0
                Ds[0, 2] = 1.0
                # Interior rows
                for i in range(1, nen-1):
                    Ds[i, i-1] = 1.0
                    Ds[i, i] = -2.0
                    Ds[i, i+1] = 1.0
                # Row nen
                Ds[nen-1, nen-3] = 1.0
                Ds[nen-1, nen-2] = -2.0
                Ds[nen-1, nen-1] = 1.0
                
                # correct boundary values
                B[0] = 0.
                B[-1] = 0.
            
            elif s==3:
                if nen < 9:
                    raise ValueError(f"Invalid number of nodes. nen = {nen}")
                
                # First half node
                Ds[0, 0] = -1.0
                Ds[0, 1] = 3.0
                Ds[0, 2] = -3.0
                Ds[0, 3] = 1.0

                # Interior half-nodes
                for i in range(1, nen-2):
                    Ds[i, i-1] = -1.0
                    Ds[i, i] = 3.0
                    Ds[i, i+1] = -3.0
                    Ds[i, i+2] = 1.0

                # Last half node
                Ds[nen-2, nen-4] = -1.0
                Ds[nen-2, nen-3] = 3.0
                Ds[nen-2, nen-2] = -3.0
                Ds[nen-2, nen-1] = 1.0

                # Last node; nothing is added to this node
                # The last row of Ds remains zero

                # correct boundary values
                B[0] = 0.
                #B[1] = 1.
                B[-1] = 0.
                B[-2] = 0.

            elif s==4:
                if nen < 13:
                    raise ValueError(f"Invalid number of nodes. nen = {nen}")

                # First node
                Ds[0, 0] = 1.0
                Ds[0, 1] = -4.0
                Ds[0, 2] = 6.0
                Ds[0, 3] = -4.0
                Ds[0, 4] = 1.0

                # Second node
                Ds[1, 0] = 1.0
                Ds[1, 1] = -4.0
                Ds[1, 2] = 6.0
                Ds[1, 3] = -4.0
                Ds[1, 4] = 1.0

                # Interior nodes
                for i in range(2, nen-2):
                    Ds[i, i-2] = 1.0
                    Ds[i, i-1] = -4.0
                    Ds[i, i] = 6.0
                    Ds[i, i+1] = -4.0
                    Ds[i, i+2] = 1.0

                # Second last node
                Ds[nen-2, nen-5] = 1.0
                Ds[nen-2, nen-4] = -4.0
                Ds[nen-2, nen-3] = 6.0
                Ds[nen-2, nen-2] = -4.0
                Ds[nen-2, nen-1] = 1.0

                # Last node
                Ds[nen-1, nen-5] = 1.0
                Ds[nen-1, nen-4] = -4.0
                Ds[nen-1, nen-3] = 6.0
                Ds[nen-1, nen-2] = -4.0
                Ds[nen-1, nen-1] = 1.0

                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
            
            elif s==5:
                if nen < 17:
                    raise ValueError(f"Invalid number of nodes. nen = {nen}")

                # First half-node
                Ds[0, 0] = -1.0
                Ds[0, 1] = 5.0
                Ds[0, 2] = -10.0
                Ds[0, 3] = 10.0
                Ds[0, 4] = -5.0
                Ds[0, 5] = 1.0

                # Second half-node
                Ds[1, 0] = -1.0
                Ds[1, 1] = 5.0
                Ds[1, 2] = -10.0
                Ds[1, 3] = 10.0
                Ds[1, 4] = -5.0
                Ds[1, 5] = 1.0

                # Interior half-nodes
                for i in range(2, nen-3):
                    Ds[i, i-2] = -1.0
                    Ds[i, i-1] = 5.0
                    Ds[i, i] = -10.0
                    Ds[i, i+1] = 10.0
                    Ds[i, i+2] = -5.0
                    Ds[i, i+3] = 1.0

                # Second last half-node
                Ds[nen-3, nen-6] = -1.0
                Ds[nen-3, nen-5] = 5.0
                Ds[nen-3, nen-4] = -10.0
                Ds[nen-3, nen-3] = 10.0
                Ds[nen-3, nen-2] = -5.0
                Ds[nen-3, nen-1] = 1.0

                # Last half-node
                Ds[nen-2, nen-6] = -1.0
                Ds[nen-2, nen-5] = 5.0
                Ds[nen-2, nen-4] = -10.0
                Ds[nen-2, nen-3] = 10.0
                Ds[nen-2, nen-2] = -5.0
                Ds[nen-2, nen-1] = 1.0

                # Last node; nothing is added to this node
                # The last row of Ds remains zero

                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
                B[-3] = 0.

            else:
                raise Exception('Invalid choice of s. Only coded up s=2,3,4,5.')
            
        elif sbp_type.lower() == 'lgl':
                    # Initialize the matrix as a dense NumPy array
                    Ds = np.zeros((nen, nen))
                    B = np.ones(nen)

                    if s==1:
                        if nen != 2:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = -1.0
                        Ds[0, 1] = 1.0

                        # Row 2
                        Ds[1, 0] = -1.0
                        Ds[1, 1] = 1.0

                    elif s==2:
                        if nen != 3:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = 1.0
                        Ds[0, 1] = -2.0
                        Ds[0, 2] = 1.0

                        # Row 2
                        Ds[1, 0] = 1.0
                        Ds[1, 1] = -2.0
                        Ds[1, 2] = 1.0

                        # Row 3
                        Ds[2, 0] = 1.0
                        Ds[2, 1] = -2.0
                        Ds[2, 2] = 1.0

                    elif s==3:
                        if nen != 4:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")
                        
                        # Row 1
                        Ds[0, 0] = -1.1111111111111111
                        Ds[0, 1] = 2.484519974999766
                        Ds[0, 2] = -2.484519974999766
                        Ds[0, 3] = 1.1111111111111111

                        # Row 2
                        Ds[1, 0] = -1.1111111111111111
                        Ds[1, 1] = 2.484519974999766
                        Ds[1, 2] = -2.484519974999766
                        Ds[1, 3] = 1.1111111111111111

                        # Row 3
                        Ds[2, 0] = -1.1111111111111111
                        Ds[2, 1] = 2.484519974999766
                        Ds[2, 2] = -2.484519974999766
                        Ds[2, 3] = 1.1111111111111111

                        # Row 4
                        Ds[3, 0] = -1.1111111111111111
                        Ds[3, 1] = 2.484519974999766
                        Ds[3, 2] = -2.484519974999766
                        Ds[3, 3] = 1.1111111111111111

                    elif s==4:
                        if nen != 5:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = 1.3125
                        Ds[0, 1] = -3.0625
                        Ds[0, 2] = 3.5
                        Ds[0, 3] = -3.0625
                        Ds[0, 4] = 1.3125

                        # Row 2
                        Ds[1, 0] = 1.3125
                        Ds[1, 1] = -3.0625
                        Ds[1, 2] = 3.5
                        Ds[1, 3] = -3.0625
                        Ds[1, 4] = 1.3125

                        # Row 3
                        Ds[2, 0] = 1.3125
                        Ds[2, 1] = -3.0625
                        Ds[2, 2] = 3.5
                        Ds[2, 3] = -3.0625
                        Ds[2, 4] = 1.3125

                        # Row 4
                        Ds[3, 0] = 1.3125
                        Ds[3, 1] = -3.0625
                        Ds[3, 2] = 3.5
                        Ds[3, 3] = -3.0625
                        Ds[3, 4] = 1.3125

                        # Row 5
                        Ds[4, 0] = 1.3125
                        Ds[4, 1] = -3.0625
                        Ds[4, 2] = 3.5
                        Ds[4, 3] = -3.0625
                        Ds[4, 4] = 1.3125

                    else:
                        raise Exception('Invalid choice of s. Only coded up s=1,2,3,4.')
                    
        elif sbp_type.lower() == 'lg':
                    # Initialize the matrix as a dense NumPy array
                    Ds = np.zeros((nen, nen))
                    B = np.ones(nen)

                    if s==1:
                        if nen != 2:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = -1.732050807568877
                        Ds[0, 1] = 1.732050807568877

                        # Row 2
                        Ds[1, 0] = -1.732050807568877
                        Ds[1, 1] = 1.732050807568877

                    elif s==2:
                        if nen != 3:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = 6.666666666666667
                        Ds[0, 1] = -13.333333333333334
                        Ds[0, 2] = 6.666666666666667

                        # Row 2
                        Ds[1, 0] = 6.666666666666667
                        Ds[1, 1] = -13.333333333333334
                        Ds[1, 2] = 6.666666666666667

                        # Row 3
                        Ds[2, 0] = 6.666666666666667
                        Ds[2, 1] = -13.333333333333334
                        Ds[2, 2] = 6.666666666666667

                    elif s==3:
                        if nen != 4:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = -5.56540505102921
                        Ds[0, 1] = 14.096587055666296
                        Ds[0, 2] = -14.096587055666296
                        Ds[0, 3] = 5.56540505102921

                        # Row 2
                        Ds[1, 0] = -5.56540505102921
                        Ds[1, 1] = 14.096587055666296
                        Ds[1, 2] = -14.096587055666296
                        Ds[1, 3] = 5.56540505102921

                        # Row 3
                        Ds[2, 0] = -5.56540505102921
                        Ds[2, 1] = 14.096587055666296
                        Ds[2, 2] = -14.096587055666296
                        Ds[2, 3] = 5.56540505102921

                        # Row 4
                        Ds[3, 0] = -5.56540505102921
                        Ds[3, 1] = 14.096587055666296
                        Ds[3, 2] = -14.096587055666296
                        Ds[3, 3] = 5.56540505102921

                    elif s==4:
                        if nen != 5:
                            raise ValueError(f"Invalid number of nodes. nen = {nen}")

                        # Row 1
                        Ds[0, 0] = 27.50958167164676
                        Ds[0, 1] = -77.90958167164676
                        Ds[0, 2] = 100.8
                        Ds[0, 3] = -77.90958167164676
                        Ds[0, 4] = 27.50958167164676

                        # Row 2
                        Ds[1, 0] = 27.50958167164676
                        Ds[1, 1] = -77.90958167164676
                        Ds[1, 2] = 100.8
                        Ds[1, 3] = -77.90958167164676
                        Ds[1, 4] = 27.50958167164676

                        # Row 3
                        Ds[2, 0] = 27.50958167164676
                        Ds[2, 1] = -77.90958167164676
                        Ds[2, 2] = 100.8
                        Ds[2, 3] = -77.90958167164676
                        Ds[2, 4] = 27.50958167164676

                        # Row 4
                        Ds[3, 0] = 27.50958167164676
                        Ds[3, 1] = -77.90958167164676
                        Ds[3, 2] = 100.8
                        Ds[3, 3] = -77.90958167164676
                        Ds[3, 4] = 27.50958167164676

                        # Row 5
                        Ds[4, 0] = 27.50958167164676
                        Ds[4, 1] = -77.90958167164676
                        Ds[4, 2] = 100.8
                        Ds[4, 3] = -77.90958167164676
                        Ds[4, 4] = 27.50958167164676

                    else:
                        raise Exception('Invalid choice of s. Only coded up s=1,2,3,4.')
        return Ds,B
    
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





        
