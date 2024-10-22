#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2024

@author: bercik
"""
#import numpy as np

import numpy as np
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp
from Source.Disc.DissOp import BaselineDiss, make_dcp_diss_op
import Source.Methods.Sparse as sp


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
        self.sparse = self.solver.sparse

        self.type = self.solver.vol_diss['diss_type']
        if isinstance(self.type, int): self.type = self.type.lower()
        if self.type == 'nd':
            self.dissipation = self.no_diss
            return
        else:

            if 'jac_type' in self.solver.vol_diss.keys():
                self.jac_type = self.solver.vol_diss['jac_type']
                if isinstance(self.jac_type, int): self.jac_type = self.jac_type.lower()
            else:
                self.jac_type = ''
            
            if not (self.solver.disc_nodes.lower() == 'upwind' and self.type == 'upwind'):
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

            if 'coeff' in self.solver.vol_diss.keys():
                assert isinstance(self.solver.vol_diss['coeff'], float) or \
                       isinstance(self.solver.vol_diss['coeff'], int), 'Artificial Dissipation: coeff must be a float, {0}'.format(self.solver.vol_diss['coeff'])
                self.coeff = self.solver.vol_diss['coeff']
            else:
                self.coeff = 0.1

            if 'bdy_fix' in self.solver.vol_diss.keys():
                assert isinstance(self.solver.vol_diss['bdy_fix'], bool), 'Artificial Dissipation: bdy_fix must be a boolean, {0}'.format(self.solver.vol_diss['bdy_fix'])
                self.bdy_fix = self.solver.vol_diss['bdy_fix']
            else:
                self.bdy_fix = True

            if 'use_H' in self.solver.vol_diss.keys():
                if isinstance(self.solver.vol_diss['use_H'], bool):
                    self.use_H = self.solver.vol_diss['use_H']
                    self.use_noH = False
                else:
                    assert (isinstance(self.solver.vol_diss['use_H'], str) and (self.solver.vol_diss['use_H'] in ['True','False','noH'])), \
                            "Artificial Dissipation: use_H must be a boolean or one of 'True','False','Diablo', {0}".format(self.solver.vol_diss['use_H'])
                    self.use_H = self.solver.vol_diss['use_H']
                    if self.use_H == 'True':
                        self.use_H = True
                    elif self.use_H == 'False':
                        self.use_H = False
                        self.use_noH = False
                    else:
                        self.use_H = False
                        self.use_noH = True
            else:
                self.use_H = True

            if self.type == 'upwind':
                print("WARNING: upwind volume dissipation is experimental and only provably stable for linear, constant-coeff. equations.")
                if 'fluxvec' in self.solver.vol_diss.keys():
                    self.fluxvec = self.solver.vol_diss['fluxvec']
                else:
                    print('WARNING: No fluxvec provided to artificial dissipation. Defaulting to fluxvec=lf')
                    self.fluxvec = 'lf'
                if self.coeff != 1.:
                    print(f'WARNING: coeff = {self.coeff} != 1. Make sure this is intentional as it is not a typical flux-vector splitting.')

            if 'avg_half_nodes' in self.solver.vol_diss.keys():
                if isinstance(self.solver.vol_diss['avg_half_nodes'], bool):
                    self.avg_half_nodes = self.solver.vol_diss['avg_half_nodes']
                else:
                    assert (isinstance(self.solver.vol_diss['avg_half_nodes'], str) and (self.solver.vol_diss['avg_half_nodes'] in ['True','False'])), \
                            "Artificial Dissipation: avg_half_nodes must be a boolean, not {0}".format(self.solver.vol_diss['use_H'])
                    avg_half_nodes = self.solver.vol_diss['avg_half_nodes']
                    if avg_half_nodes == 'True':
                        self.avg_half_nodes = True
                    elif avg_half_nodes == 'False':
                        self.avg_half_nodes = False
            else:
                self.avg_half_nodes = True
            if self.avg_half_nodes and self.dim == 3:
                print("WARNING: avg_half_nodes not implemented for 3D. Defaulting to avg_half_nodes=False.")
                self.avg_half_nodes = False

            if 'entropy_fix' in self.solver.vol_diss.keys():
                assert isinstance(self.solver.vol_diss['entropy_fix'], bool), 'Artificial Dissipation: entropy_fix must be a boolean, {0}'.format(self.solver.vol_diss['entropy_fix'])
                self.entropy_fix = self.solver.vol_diss['entropy_fix']
            else:
                self.entropy_fix = True

            # Set operators and methods

            if self.dim == 1:
                self.set_ops_1D()
            elif self.dim == 2:
                self.dxidx = self.solver.mesh.metrics[:,:2,:]
                self.detadx = self.solver.mesh.metrics[:,2:,:]
                self.set_ops_2D()
            elif self.dim == 3:
                self.dxidx = self.solver.mesh.metrics[:,:3,:]
                self.detadx = self.solver.mesh.metrics[:,3:6,:]
                self.dzetadx = self.solver.mesh.metrics[:,6:,:]

            if self.type == 'b':
                if self.jac_type == 'scalar' or self.jac_type == 'sca':
                    self.dissipation = self.dissipation_B_scalar
                else:
                    print("WARNING: Only scalar dissipation set up for type='B' dissipation. Defaulting to jac_type = 'scalar'.")
                    self.jac_type = 'scalar'
                    self.dissipation = self.dissipation_B_scalar
            
            elif self.type == 'entb':
                if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca':
                    self.dissipation = self.dissipation_entB_scalarscalar
                elif self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                    self.dissipation = self.dissipation_entB_scalarmatrix
                elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                    self.dissipation = self.dissipation_entB_matrixmatrix
                else:
                    print("WARNING: jac method not understood for type='entB' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = self.dissipation_entB_scalarscalar
            
            elif self.type == 'dcp' or self.type == 'w':
                if self.jac_type == 'scalar' or self.jac_type == 'sca':
                    self.dissipation = self.dissipation_dcp_scalar
                else:
                    print("WARNING: Only scalar dissipation set up for type='DCP' and type='W' dissipation. Defaulting to jac_type = 'scalar'.")
                    self.jac_type = 'scalar'
                    self.dissipation = self.dissipation_dcp_scalar

            elif self.type == 'entdcp' or self.type == 'entw':
                if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca':
                    self.dissipation = self.dissipation_entdcp_scalarscalar
                elif self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                    self.dissipation = self.dissipation_entdcp_scalarmatrix
                elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                    self.dissipation = self.dissipation_entdcp_matrixmatrix
                    self.dEndw_abs = self.solver.diffeq.dEndw_abs
                else:
                    print("WARNING: jac method not understood for type='entDCP' or type='entW' dissipation. Defaulting to jac_type = 'scalarscalar'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = self.dissipation_entdcp_scalarscalar

            elif self.type == 'upwind':
                if self.fluxvec.lower() == 'lf':
                    self.dissipation = self.dissipation_upwind_lf
                elif self.fluxvec.lower() == 'sw' or self.fluxvec.lower()=='stegerwarming':
                    if self.dim == 1: 
                        from Source.DiffEq.EulerFunctions import StegerWarming_diss_1D
                        self.stegerwarming = StegerWarming_diss_1D
                    elif self.dim == 2: 
                        from Source.DiffEq.EulerFunctions import StegerWarming_diss_2D
                        self.stegerwarming = StegerWarming_diss_2D
                    self.dissipation = self.dissipation_upwind_stegerwarming
                else:
                    print("WARNING: fluxvec method not understood. Defaulting to fluxvec = 'lf'.")
                    self.fluxvec = 'lf'
                    self.dissipation = self.dissipation_upwind_lf
            
            else:
                raise Exception('Artificial dissipation: diss_type not understood, '+ str(self.type))
            
            if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca' \
            or self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat' \
            or self.jac_type == 'scalar' or self.jac_type == 'sca' \
            or (self.type == 'upwind' and self.fluxvec.lower() == 'lf'):
                if self.dim == 1:
                    self.maxeig_dExdq = self.solver.diffeq.maxeig_dExdq
                else:
                    self.maxeig_dEndq = self.solver.diffeq.maxeig_dEndq
            
            if self.jac_type == 'matrix' or self.jac_type == 'mat' \
            or self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                if self.dim == 1:
                    self.dExdq_abs = lambda q: self.solver.diffeq.dExdq_abs(q, self.entropy_fix)
                else:
                    self.dEndq_abs = lambda q, dxidx: self.solver.diffeq.dEndq_abs(q, dxidx, self.entropy_fix)
            
            if 'ent' in self.type:
                self.entropy_var = self.solver.diffeq.entropy_var
                self.dqdw = self.solver.diffeq.dqdw

                if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca':
                    if hasattr(self.solver.diffeq, 'maxeig_dqdw'):
                        self.rho_dqdw = self.solver.diffeq.maxeig_dqdw
                    else:
                        print('WARNING: diffeq.maxeig_dqdw not found. Defaulting to spectral radius of dqdw, which is slower.')
                        if self.neq_node == 1:
                            self.rho_dqdw = lambda q: self.dqdw(q)[:,0,0,:]
                        else:
                            self.rho_dqdw = lambda q: fn.spec_rad(self.dqdw(q),self.neq_node)
                elif self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                    self.dissipation = lambda q: self.dissipation_entdcp_scalarmatrix(q,self.coeff)
                elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                    self.dissipation = lambda q: self.dissipation_entdcp_matrixmatrix(q,self.coeff)
                    if self.neq_node == 1:
                        if hasattr(self.solver.diffeq, 'dExdw_abs'):
                            self.dExdw_abs = lambda q: self.solver.diffeq.dExdw_abs(q, self.entropy_fix)
                        else:
                            print('WARNING: diffeq.dExdw_abs not found. Defaulting to dExdq_abs @ dqdw, which is slower.')
                            self.dExdw_abs = self.calc_absAP_base_1d
                    else:
                        if hasattr(self.solver.diffeq, 'dEndw_abs'):
                            self.dEndw_abs = lambda q, metrics: self.solver.diffeq.dEndw_abs(q, metrics, self.entropy_fix)
                        else:
                            print('WARNING: diffeq.dEndw_abs not found. Defaulting to dEndq_abs @ dqdw, which is slower.')
                            self.dEndw_abs = self.calc_absAP_base_nd
                else:
                    raise Exception('Artificial dissipation: jac_type not understood, '+ str(self.jac_type)) # should not happen

            if self.neq_node > 1:
                self.repeat_neq_gv = lambda gv: fn.repeat_neq_gv(gv,self.neq_node)
            else:
                self.repeat_neq_gv = lambda gv: gv
            
            if self.sparse:
                self.gm_gv = sp.gm_gv
                self.lm_gv = sp.lm_gv
            else:
                self.gm_gv = fn.gm_gv
                self.lm_gv = fn.lm_gv
                self.gdiag_gm = fn.gdiag_gm

    def calc_absAP_base_1d(self, q):
        # calls the base methods for both absA and P
        absA = self.dExdq_abs(q)
        P = self.dqdw(q)
        absAP = fn.gbdiag_gbdiag(absA, P)
        return absAP
    
    def calc_absAP_base_nd(self, q, metrics):
        # calls the base method for both absA and P
        absA = self.dEndq_abs(q, metrics)
        P = self.dqdw(q)
        absAP = fn.gbdiag_gbdiag(absA, P)
        return absAP

    def set_ops_1D(self):
        ''' prepare the various operators needed for the dissipation function '''
        # xavg = self.solver.mesh.dom_len/self.nelem/(self.nen-1) # this was used in the original work
        #xavg = (self.solver.mesh.bdy_x[1,:]-self.solver.mesh.bdy_x[0,:])/(self.nen-1)
        xavg = (1.-0.)/(self.nen-1) # this is the reference spacing. Physical spacing is taken care of implicitly by metrics.

        if self.type == 'w' or self.type == 'entw':
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
        elif self.type == 'dcp' or self.type == 'entdcp':
            Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            self.rhs_D = fn.kron_neq_lm(Ds,self.neq_node) 
            if self.use_H:
                Hundvd = self.solver.sbp.H / self.solver.sbp.dx
                self.lhs_D = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(Ds.T @ np.diag(B) @ Hundvd, self.neq_node))
            else:
                self.lhs_D = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(Ds.T @ np.diag(B), self.neq_node))
        elif self.type == 'upwind':
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
        elif self.type == 'b' or self.type == 'entb':
            D = BaselineDiss(self.s, self.nen)
            if self.use_H:
                Hundvd = self.solver.sbp.H / self.solver.sbp.dx
            else:
                Hundvd = np.eye(self.nen)
            if self.s == 1:
                D.updateD1()
                D.updateD2()
                D.updateB1()
                D.updateB2()
                self.rhs_D1 = fn.kron_neq_lm(D.D1,self.neq_node) 
                self.rhs_D2 = fn.kron_neq_lm(D.D2,self.neq_node)
                self.lhs_D1 = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D1.T @ D.B1 @ Hundvd, self.neq_node))
                self.lhs_D2 = 0.25*fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D2.T @ D.B2 @ Hundvd, self.neq_node))
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
                self.lhs_D2 = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D2.T @ D.B1 @ Hundvd, self.neq_node))
                self.lhs_D3 = 0.5 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D3.T @ D.B3 @ Hundvd, self.neq_node))
                self.lhs_D4 = 0.0625 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D4.T @ D.B4 @ Hundvd, self.neq_node))
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
                self.lhs_D3 = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D3.T @ D.B1 @ Hundvd, self.neq_node))
                self.lhs_D4 = 0.75 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D4.T @ D.B4 @ Hundvd, self.neq_node))
                self.lhs_D5 = 0.3125 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D5.T @ D.B5 @ Hundvd, self.neq_node))
                self.lhs_D6 = (1./96.) * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D6.T @ D.B6 @ Hundvd, self.neq_node))
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
                self.lhs_D4 = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D4.T @ D.B1 @ Hundvd, self.neq_node))
                self.lhs_D5 = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D5.T @ D.B5 @ Hundvd, self.neq_node))
                self.lhs_D6 = 0.875 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D6.T @ D.B6 @ Hundvd, self.neq_node))
                self.lhs_D7 = 0.0875 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D7.T @ D.B7 @ Hundvd, self.neq_node))
                self.lhs_D8 = 0.00171875 * fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(D.D8.T @ D.B8 @ Hundvd, self.neq_node))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')

        else:
            raise Exception(self.type + ' not set up yet')
        
        if self.sparse:
            if self.type in ['w', 'entw', 'dcp', 'entdcp']:
                self.rhs_D = sp.lm_to_sp(self.rhs_D)
                self.lhs_D = sp.gm_to_sp(self.lhs_D)
            elif self.type == 'upwind':
                self.Ddiss = sp.gm_to_sp(self.Ddiss)
            elif self.type in ['b', 'entb']:
                if self.s == 1:
                    self.rhs_D1 = sp.lm_to_sp(self.rhs_D1)
                    self.rhs_D2 = sp.lm_to_sp(self.rhs_D2)
                    self.lhs_D1 = sp.gm_to_sp(self.lhs_D1)
                    self.lhs_D2 = sp.gm_to_sp(self.lhs_D2)
                elif self.s == 2:
                    self.rhs_D2 = sp.lm_to_sp(self.rhs_D2)
                    self.rhs_D3 = sp.lm_to_sp(self.rhs_D3)
                    self.rhs_D4 = sp.lm_to_sp(self.rhs_D4)
                    self.lhs_D2 = sp.gm_to_sp(self.lhs_D2)
                    self.lhs_D3 = sp.gm_to_sp(self.lhs_D3)
                    self.lhs_D4 = sp.gm_to_sp(self.lhs_D4)
                elif self.s == 3:
                    self.rhs_D3 = sp.lm_to_sp(self.rhs_D3)
                    self.rhs_D4 = sp.lm_to_sp(self.rhs_D4)
                    self.rhs_D5 = sp.lm_to_sp(self.rhs_D5)
                    self.rhs_D6 = sp.lm_to_sp(self.rhs_D6)
                    self.lhs_D3 = sp.gm_to_sp(self.lhs_D3)
                    self.lhs_D4 = sp.gm_to_sp(self.lhs_D4)
                    self.lhs_D5 = sp.gm_to_sp(self.lhs_D5)
                    self.lhs_D6 = sp.gm_to_sp(self.lhs_D6)
                elif self.s == 4:
                    self.rhs_D4 = sp.lm_to_sp(self.rhs_D4)
                    self.rhs_D5 = sp.lm_to_sp(self.rhs_D5)
                    self.rhs_D6 = sp.lm_to_sp(self.rhs_D6)
                    self.rhs_D7 = sp.lm_to_sp(self.rhs_D7)
                    self.rhs_D8 = sp.lm_to_sp(self.rhs_D8)
                    self.lhs_D4 = sp.gm_to_sp(self.lhs_D4)
                    self.lhs_D5 = sp.gm_to_sp(self.lhs_D5)
                    self.lhs_D6 = sp.gm_to_sp(self.lhs_D6)
                    self.lhs_D7 = sp.gm_to_sp(self.lhs_D7)
                    self.lhs_D8 = sp.gm_to_sp(self.lhs_D8)

        
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
        elif self.type == 'dcp' or self.type == 'entdcp':                
            Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            eye = np.eye(self.nen)
            self.rhs_Dxi = fn.kron_neq_lm(np.kron(Ds, eye),self.neq_node) 
            self.rhs_Deta = fn.kron_neq_lm(np.kron(eye, Ds),self.neq_node) 
            if self.use_H:
                Hundvd = self.solver.sbp.H / self.solver.sbp.dx
                DsTxi = np.kron(Ds.T @ np.diag(B) @ Hundvd, Hundvd)
                DsTeta = np.kron(Hundvd, Ds.T @ np.diag(B) @ Hundvd)
            else:
                if self.use_noH:
                    # uses no H at all
                    DsTxi = np.kron(Ds.T @ np.diag(B), eye)
                    DsTeta = np.kron(eye, Ds.T @ np.diag(B))
                else:
                    # uses H in the perpendicular direction only
                    Hundvd = self.solver.sbp.H
                    DsTxi = np.kron(Ds.T @ np.diag(B), eye) @ np.kron(eye, Hundvd)
                    DsTeta = np.kron(eye, Ds.T @ np.diag(B)) @ np.kron(Hundvd, eye)
            self.lhs_Dxi = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(DsTxi,self.neq_node))
            self.lhs_Deta = fn.gdiag_lm(-self.solver.H_inv_phys,fn.kron_neq_lm(DsTeta,self.neq_node))
        elif self.type == 'upwind':
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
        
        if self.sparse:
            if self.type in ['w', 'entw', 'dcp', 'entdcp']:
                self.rhs_Dxi = sp.lm_to_sp(self.rhs_Dxi)
                self.rhs_Deta = sp.lm_to_sp(self.rhs_Deta)
                self.lhs_Dxi = sp.gm_to_sp(self.lhs_Dxi)
                self.lhs_Deta = sp.gm_to_sp(self.lhs_Deta)
            elif self.type == 'upwind':
                self.Dxidiss = sp.gm_to_sp(self.Dxidiss)
                self.Dxidiss = sp.gm_to_sp(self.Dxidiss)


    def no_diss(self, *args, **kwargs):
        ''' no dissipation function '''
        return 0
    
    def dissipation_upwind_lf(self, q):
        ''' dissipation function for upwind / LF flux-vector splitting'''
        if self.dim == 1:
            maxeig = self.maxeig_dExdq(q)
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.Ddiss, A * q)
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.Dxidiss, A * q) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.Detadiss, A * q) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.Dxidiss, A * q) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.Detadiss, A * q) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.Dzetadiss, A * q) # zeta part
        return self.coeff*diss
    
    def dissipation_upwind_stegerwarming(self, q):
        ''' dissipation function for upwind steger warming flux-vector splitting'''
        if self.dim == 1:
            fdiss = self.stegerwarming(q)
            diss = self.gm_gv(self.Ddiss, fdiss)
        elif self.dim == 2:
            fdiss = self.stegerwarming(q,self.dxidx)
            diss = self.gm_gv(self.Dxidiss, fdiss) # xi part
            fdiss = self.stegerwarming(q,self.detadx)
            diss += self.gm_gv(self.Detadiss, fdiss) # eta part
        elif self.dim == 3:
            diss = 0.
        return self.coeff*diss

    def dissipation_dcp_scalar(self, q):
        ''' dissipation function for w and dcp, scalar functions or systems'''
        if self.dim == 1:
            maxeig = self.maxeig_dExdq(q)
            if self.s % 2 == 1 and self.avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.lhs_D, A * self.lm_gv(self.rhs_D, q))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                maxeig[:-self.nen:self.nen] = 0.5*(maxeig[:-self.nen:self.nen] + maxeig[self.nen::self.nen])
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.lhs_Dxi, A * self.lm_gv(self.rhs_Dxi, q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    maxeig[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(maxeig[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + maxeig[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.lhs_Deta, A * self.lm_gv(self.rhs_Deta, q)) # eta part
        elif self.dim == 3:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            A = self.repeat_neq_gv(maxeig)
            diss = self.gm_gv(self.lhs_Dxi, A * self.lm_gv(self.rhs_Dxi, q)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx)
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.lhs_Deta, A * self.lm_gv(self.rhs_Deta, q)) # eta part
            maxeig = self.maxeig_dEndq(q,self.dzetadx)
            A = self.repeat_neq_gv(maxeig)
            diss += self.gm_gv(self.lhs_Dzeta, A * self.lm_gv(self.rhs_Dzeta, q)) # zeta part
        return self.coeff*diss
    
    def dissipation_B_scalar(self, q):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar functions or systems'''
        if self.dim == 1:
            maxeig = self.maxeig_dExdq(q)
            A = self.repeat_neq_gv(maxeig)
            if self.s == 1:
                diss = self.gm_gv(self.lhs_D1, A * self.lm_gv(self.rhs_D1, q)) + \
                       self.gm_gv(self.lhs_D2, A * self.lm_gv(self.rhs_D2, q))
            elif self.s == 2:
                diss = self.gm_gv(self.lhs_D2, A * self.lm_gv(self.rhs_D2, q)) + \
                       self.gm_gv(self.lhs_D3, A * self.lm_gv(self.rhs_D3, q)) + \
                       self.gm_gv(self.lhs_D4, A * self.lm_gv(self.rhs_D4, q))
            elif self.s == 3:
                diss = self.gm_gv(self.lhs_D3, A * self.lm_gv(self.rhs_D3, q)) + \
                       self.gm_gv(self.lhs_D4, A * self.lm_gv(self.rhs_D4, q)) + \
                       self.gm_gv(self.lhs_D5, A * self.lm_gv(self.rhs_D5, q)) + \
                       self.gm_gv(self.lhs_D6, A * self.lm_gv(self.rhs_D6, q))
            elif self.s == 4:
                diss = self.gm_gv(self.lhs_D4, A * self.lm_gv(self.rhs_D4, q)) + \
                       self.gm_gv(self.lhs_D5, A * self.lm_gv(self.rhs_D5, q)) + \
                       self.gm_gv(self.lhs_D6, A * self.lm_gv(self.rhs_D6, q)) + \
                       self.gm_gv(self.lhs_D7, A * self.lm_gv(self.rhs_D7, q)) + \
                       self.gm_gv(self.lhs_D8, A * self.lm_gv(self.rhs_D8, q))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return self.coeff*diss
    
    def dissipation_entdcp_scalarscalar(self, q):
        ''' dissipation function for entW and entdcp, scalar functions'''
        w = self.entropy_var(q)
        rho_dqdw = self.rho_dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dExdq(q)
            AP = maxeig*rho_dqdw
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-1] = 0.5*(AP[:-1] + AP[1:])
            AP = fn.repeat_neq_gv(AP)
            diss = self.gm_gv(self.lhs_D, AP * self.lm_gv(self.rhs_D, w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) 
            AP = maxeig*rho_dqdw
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-self.nen:self.nen] = 0.5*(AP[:-self.nen:self.nen] + AP[self.nen::self.nen])
            AP = fn.repeat_neq_gv(AP)
            diss = self.gm_gv(self.lhs_Dxi, AP * self.lm_gv(self.rhs_Dxi, w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) 
            AP = fn.repeat_neq_gv(maxeig*rho_dqdw)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + AP[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            AP = fn.repeat_neq_gv(AP)        
            diss += self.gm_gv(self.lhs_Deta, AP * self.lm_gv(self.rhs_Deta, w)) # eta part
        elif self.dim == 3:
            raise Exception('TODO')

        return self.coeff*diss
    
    def dissipation_entB_scalarscalar(self, q):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar-scalar systems'''
        w = self.entropy_var(q)
        rho_dqdw = self.rho_dqdw(q)
        if self.dim == 1:
            maxeig = self.maxeig_dEndq(q,self.dxidx)
            AP = fn.repeat_neq_gv(maxeig*rho_dqdw,self.neq_node)
            if self.s == 1:
                diss = self.gm_gv(self.lhs_D1, AP * self.lm_gv(self.rhs_D1, w)) + \
                       self.gm_gv(self.lhs_D2, AP * self.lm_gv(self.rhs_D2, w))
            elif self.s == 2:
                diss = self.gm_gv(self.lhs_D2, AP * self.lm_gv(self.rhs_D2, w)) + \
                       self.gm_gv(self.lhs_D3, AP * self.lm_gv(self.rhs_D3, w)) + \
                       self.gm_gv(self.lhs_D4, AP * self.lm_gv(self.rhs_D4, w))
            elif self.s == 3:
                diss = self.gm_gv(self.lhs_D3, AP * self.lm_gv(self.rhs_D3, w)) + \
                       self.gm_gv(self.lhs_D4, AP * self.lm_gv(self.rhs_D4, w)) + \
                       self.gm_gv(self.lhs_D5, AP * self.lm_gv(self.rhs_D5, w)) + \
                       self.gm_gv(self.lhs_D6, AP * self.lm_gv(self.rhs_D6, w))
            elif self.s == 4:
                diss = self.gm_gv(self.lhs_D4, AP * self.lm_gv(self.rhs_D4, w)) + \
                       self.gm_gv(self.lhs_D5, AP * self.lm_gv(self.rhs_D5, w)) + \
                       self.gm_gv(self.lhs_D6, AP * self.lm_gv(self.rhs_D6, w)) + \
                       self.gm_gv(self.lhs_D7, AP * self.lm_gv(self.rhs_D7, w)) + \
                       self.gm_gv(self.lhs_D8, AP * self.lm_gv(self.rhs_D8, w))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return self.coeff*diss
    
    def dissipation_entdcp_scalarmatrix(self, q):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            maxeig = self.repeat_neq_gv(self.maxeig_dExdq(q))
            AP = fn.gdiag_gbdiag(maxeig, dqdw)
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-1] = 0.5*(AP[:-1] + AP[1:])
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
        elif self.dim == 2:
            maxeig = self.repeat_neq_gv(self.maxeig_dEndq(q,self.dxidx))
            AP = fn.gdiag_gbdiag(maxeig, dqdw)
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-self.nen:self.nen] = 0.5*(AP[:-self.nen:self.nen] + AP[self.nen::self.nen])
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
            maxeig = self.repeat_neq_gv(self.maxeig_dEndq(q,self.detadx))
            AP = fn.gdiag_gbdiag(maxeig, dqdw)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + AP[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            diss += self.gm_gv(self.lhs_Deta, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Deta, w)))
        elif self.dim == 3:
            raise Exception('TODO')
        return self.coeff*diss
    
    def dissipation_entB_scalarmatrix(self, q):
        ''' DCP 2018 dissipation function for narrow interior stencils, scalar-matrix systems'''
        w = self.entropy_var(q)
        dqdw = self.dqdw(q)
        if self.dim == 1:
            AP = fn.gdiag_gbdiag(self.repeat_neq_gv(self.maxeig_dExdq(q)), dqdw)
            if self.s == 1:
                diss = self.gm_gv(self.lhs_D1, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D1, w))) + \
                       self.gm_gv(self.lhs_D2, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D2, w)))
            elif self.s == 2:
                diss = self.gm_gv(self.lhs_D2, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D2, w))) + \
                       self.gm_gv(self.lhs_D3, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D3, w))) + \
                       self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w)))
            elif self.s == 3:
                diss = self.gm_gv(self.lhs_D3, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D3, w))) + \
                       self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w))) + \
                       self.gm_gv(self.lhs_D5, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D5, w))) + \
                       self.gm_gv(self.lhs_D6, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D6, w)))
            elif self.s == 4:
                diss = self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w))) + \
                       self.gm_gv(self.lhs_D5, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D5, w))) + \
                       self.gm_gv(self.lhs_D6, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D6, w))) + \
                       self.gm_gv(self.lhs_D7, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D7, w))) + \
                       self.gm_gv(self.lhs_D8, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D8, w)))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return self.coeff*diss
    
    def dissipation_entdcp_matrixmatrix(self, q):
        ''' dissipation function for DCP's narrow interior stencils, entropy-stable systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        w = self.entropy_var(q)
        if self.dim == 1:
            AP = self.dExdw_abs(q)
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-1] = 0.5*(AP[:-1] + AP[1:])
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
        elif self.dim == 2:
            AP = self.dEndw_abs(q,self.dxidx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-self.nen:self.nen] = 0.5*(AP[:-self.nen:self.nen] + AP[self.nen::self.nen])
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
            AP = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + AP[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            diss += self.gm_gv(self.lhs_Deta, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Deta, w)))
        elif self.dim == 3:
            if self.avg_half_nodes:
                raise Exception('TODO')
            AP = self.dEndw_abs(q,self.dxidx)
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
            AP = self.dEndw_abs(q,self.detadx)
            diss += self.gm_gv(self.lhs_Deta, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Deta, w)))
            AP = self.dEndw_abs(q,self.dzetadx)
            diss += self.gm_gv(self.lhs_Dzeta, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Dzeta, w)))
        return self.coeff*diss
    
    def dissipation_entB_matrixmatrix(self, q):
        ''' DCP 2018 dissipation function for narrow interior stencils, matrix-matrix systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            AP = self.dExdw_abs(q)
            if self.s == 1:
                diss = self.gm_gv(self.lhs_D1, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D1, w))) + \
                       self.gm_gv(self.lhs_D2, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D2, w)))
            elif self.s == 2:
                diss = self.gm_gv(self.lhs_D2, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D2, w))) + \
                       self.gm_gv(self.lhs_D3, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D3, w))) + \
                       self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w)))
            elif self.s == 3:
                diss = self.gm_gv(self.lhs_D3, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D3, w))) + \
                       self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w))) + \
                       self.gm_gv(self.lhs_D5, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D5, w))) + \
                       self.gm_gv(self.lhs_D6, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D6, w)))
            elif self.s == 4:
                diss = self.gm_gv(self.lhs_D4, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D4, w))) + \
                       self.gm_gv(self.lhs_D5, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D5, w))) + \
                       self.gm_gv(self.lhs_D6, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D6, w))) + \
                       self.gm_gv(self.lhs_D7, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D7, w))) + \
                       self.gm_gv(self.lhs_D8, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D8, w)))
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')
        elif self.dim == 2:
            raise Exception('TODO')
        elif self.dim == 3:
            raise Exception('TODO')
        return self.coeff*diss
    
    
    
    def calc_LHS(self, q=None, step=1.0e-4):
        ''' could form explicitly... but for simplicity just do finite difference. 
        Note: this does not include the coefficient '''
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





        
