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
        '''
        
        print('... Setting up Artificial Dissipation')
        
        self.solver = solver
        self.dim = self.solver.dim
        self.nelem = self.solver.nelem
        self.neq_node = self.solver.neq_node
        self.nen = self.solver.nen

        self.type = self.solver.vol_diss['diss_type']
        self.jac_type = self.solver.vol_diss['jac_type']
        if self.type == 'ND':
            self.dissipation = self.no_diss
            return
        else:
            if 's' in self.solver.vol_diss.keys():
                self.s = self.solver.vol_diss['s']
                assert isinstance(self.s, int), 'Artificial Dissipation: s must be an integer, {0}'.format(self.s)
            else:
                print('WARNING: No s provided to artipifial dissipation. Defaulting to s=p')
                self.s = self.solver.p

            if 'coeff' in  self.solver.vol_diss.keys():
                assert isinstance(self.solver.vol_diss['coeff'], float), 'Artificial Dissipation: coeff must be a float, {0}'.format(self.solver.vol_diss['coeff'])
                self.coeff = self.solver.vol_diss['coeff']
            else:
                self.coeff = 1.


        if self.dim == 1:
            self.set_ops_1D()

            if self.type == 'B':
                if self.jac_type == 'scalar':
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                    self.dissipation = lambda q: self.dissipation_B_scalar(q,self.coeff)
                else:
                    print("WARNING: Only scalar dissipation set up for type='B' dissipation. Defaulting to jac_type = 'scalar'.")
                    self.jac_type = 'scalar'
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                    self.dissipation = lambda q: self.dissipation_B_scalar(q,self.coeff)
            
            elif self.type == 'entB':
                if self.jac_type == 'scalarscalar':
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                    self.entropy_var = self.solver.diffeq.entropy_var
                    self.dqdw = self.solver.diffeq.dqdw
                    self.dissipation = lambda q: self.dissipation_entB_scalarscalar(q,self.coeff)
                elif self.jac_type == 'scalarmatrix':
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                    self.entropy_var = self.solver.diffeq.entropy_var
                    self.dqdw = self.solver.diffeq.dqdw
                    self.dissipation = lambda q: self.dissipation_entB_scalarscalar(q,self.coeff)
                elif self.jac_type == 'matrixmatrix':
                    self.dExdw_abs = self.solver.diffeq.dExdw_abs
                    self.entropy_var = self.solver.diffeq.entropy_var
                    self.dissipation = lambda q: self.dissipation_entB_matrixmatrix(q,self.coeff)
                else:
                    print("WARNING: Only scalar dissipation set up for type='entB' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                    self.entropy_var = self.solver.diffeq.entropy_var
                    self.dqdw = self.solver.diffeq.dqdw
                    self.dissipation = lambda q: self.dissipation_entB_scalarscalar(q,self.coeff)
            
            else:
                raise Exception('Artificial dissipation: diss_type not understood, '+ str(self.diss_type))


    def set_ops_1D(self):
        ''' prepare the various operators needed for the dissipation function '''
        # xavg = self.solver.mesh.dom_len/self.nelem/(self.nen-1) # this was used in the original work
        xavg = (self.solver.mesh.bdy_x[1,:]-self.solver.mesh.bdy_x[0,:])/(self.nen-1)
        Ds = np.copy(self.solver.Dx_phys_nd)
        for i in range(1,self.s):
            Ds = fn.gm_gm(self.solver.Dx_phys_nd,Ds)
        DsT = np.transpose(Ds,axes=(1,0,2))
        self.rhs_D = Ds
        self.lhs_D = fn.gdiag_gm(-(self.solver.H_inv_phys * xavg**(2*self.s-1)), DsT * self.solver.H_phys)

        if self.type == 'B':
            pass

    def no_diss(self, q):
        return 0

    def dissipation_B_scalar(self, q, coeff=1.):
        ''' dissipation function for baseline, i.e. wide interior stencil'''
        maxeig = self.maxeig_dExdq(q)
        A = fn.repeat_neq_gv(maxeig,self.neq_node)
        diss = fn.gm_gv(self.lhs_D, fn.gdiag_gv(A, fn.gm_gv(self.rhs_D, q)))
        return coeff*diss
    
    def dissipation_entB_scalarscalar(self, q, coeff=1.):
        ''' dissipation function for baseline, i.e. wide interior stencil'''
        w = self.entropy_var(q)
        maxeig = self.maxeig_dExdq(q)
        dqdw = self.dqdw(q)
        rho = fn.spec_rad(dqdw,self.neq_node)
        A = fn.repeat_neq_gv(maxeig*rho,self.neq_node)
        diss = fn.gm_gv(self.lhs_D, fn.gdiag_gv(A, fn.gm_gv(self.rhs_D, w)))
        return coeff*diss
    
    def dissipation_entB_scalarmatrix(self, q, coeff=1.):
        ''' dissipation function for baseline, i.e. wide interior stencil'''
        w = self.entropy_var(q)
        maxeig = self.maxeig_dExdq(q)
        dqdw = self.dqdw(q)
        A = maxeig * dqdw
        diss = fn.gm_gv(self.lhs_D, fn.gdiag_gv(A, fn.gm_gv(self.rhs_D, w)))
        return coeff*diss
    
    def dissipation_entB_matrixmatrix(self, q, coeff=1.):
        ''' dissipation function for baseline, i.e. wide interior stencil'''
        w = self.entropy_var(q)
        A = self.dExdw_abs(q)
        diss = fn.gm_gv(self.lhs_D, fn.gdiag_gv(A, fn.gm_gv(self.rhs_D, w)))
        return coeff*diss

        
