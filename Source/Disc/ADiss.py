#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2024

@author: bercik
"""
import numpy as np
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp
from Source.Disc.DissOp import BaselineDiss, make_dcp_diss_op, make_dcp_diss_op2
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
        
        if solver.print_progress: print('... Setting up Artificial Dissipation')
        
        self.solver = solver
        self.dim = self.solver.dim
        self.nelem = self.solver.nelem
        self.neq_node = self.solver.neq_node
        self.nen = self.solver.nen
        self.sparse = self.solver.sparse

        self.type = self.solver.vol_diss['diss_type']
        if isinstance(self.type, str): self.type = self.type.lower()
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
                        if self.solver.disc_nodes.lower() not in ['lgl','lg']:
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
                    if self.solver.disc_nodes.lower() not in ['lgl','lg']:
                        print('WARNING: No s provided to artificial dissipation. Defaulting to s=p, but you probably want s=p+1.')
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
                    assert (isinstance(self.solver.vol_diss['use_H'], str) and (self.solver.vol_diss['use_H'] in ['True','False','None'])), \
                            "Artificial Dissipation: use_H must be a boolean or one of 'True','False','None', {0}".format(self.solver.vol_diss['use_H'])
                    self.use_H = self.solver.vol_diss['use_H']
                    if self.use_H == 'True':
                        self.use_H = True
                        self.use_noH = False
                    elif self.use_H == 'False':
                        self.use_H = False
                        self.use_noH = False
                    else: #== None
                        self.use_H = False
                        self.use_noH = True
            else:
                self.use_H = True
                self.use_noH = False

            if 'use_A' in self.solver.vol_diss.keys():
                if isinstance(self.solver.vol_diss['use_A'], bool):
                    self.use_A = self.solver.vol_diss['use_A']
                else:
                    assert (isinstance(self.solver.vol_diss['use_A'], str) and (self.solver.vol_diss['use_A'] in ['True','False'])), \
                            "Artificial Dissipation: use_A must be a boolean or one of 'True','False', {0}".format(self.solver.vol_diss['use_A'])
                    if self.solver.vol_diss['use_A']:
                        self.use_A = True
                    else:
                        self.use_A = False
            else:
                self.use_A = True

            if self.type == 'upwind' or self.type == 'upwindlgl':
                if 'fluxvec' in self.solver.vol_diss.keys():
                    self.fluxvec = self.solver.vol_diss['fluxvec']
                else:
                    print('WARNING: No fluxvec provided to artificial dissipation. Defaulting to fluxvec=lf')
                    self.fluxvec = 'lf'
                if self.coeff != 1. and self.type == 'upwind':
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
                    print("WARNING: jac method not understood for type='entB' dissipation. Defaulting to jac_type = 'scalarmatrix'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = self.dissipation_entB_scalarmatrix
            
            elif self.type == 'dcp' or self.type == 'w' or self.type == 'dcp2':
                if self.jac_type == 'scalar' or self.jac_type == 'sca':
                    self.dissipation = self.dissipation_dcp_scalar
                elif self.jac_type == 'matrix' or self.jac_type == 'mat':
                    self.dissipation = self.dissipation_dcp_matrix
                else:
                    print("WARNING: Only scalar and matrix dissipation set up for type='DCP' and type='W' dissipation. Defaulting to jac_type = 'scalar'.")
                    self.jac_type = 'scalar'
                    self.dissipation = self.dissipation_dcp_scalar

            elif self.type == 'entdcp' or self.type == 'entw':
                if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca':
                    self.dissipation = self.dissipation_entdcp_scalarscalar
                elif self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                    self.dissipation = self.dissipation_entdcp_scalarmatrix
                elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                    self.dissipation = self.dissipation_entdcp_matrixmatrix
                else:
                    print("WARNING: jac method not understood for type='entDCP' or type='entW' dissipation. Defaulting to jac_type = 'scalarmatrix'.")
                    self.jac_type = 'scalarscalar'
                    self.dissipation = self.dissipation_entdcp_scalarmatrix

            elif self.type == 'upwind' or self.type == 'upwindlgl':
                if self.fluxvec.lower() == 'lf':
                    self.dissipation = self.dissipation_upwind_lf
                elif self.fluxvec.lower() == 'burgers':
                    self.dissipation = self.dissipation_upwind_burgers
                elif self.fluxvec.lower() == 'sw' or self.fluxvec.lower()=='stegerwarming':
                    if self.dim == 1: 
                        from Source.DiffEq.EulerFunctions import StegerWarming_diss_1D
                        self.stegerwarming = StegerWarming_diss_1D
                    elif self.dim == 2:
                        print("WARNING: StegerWarming flux-vector splitting not yet debugged for 2D. Try DrikakisTsangaris instead.") 
                        from Source.DiffEq.EulerFunctions import StegerWarming_diss_2D
                        self.stegerwarming = StegerWarming_diss_2D
                    self.dissipation = self.dissipation_upwind_stegerwarming
                elif self.fluxvec.lower() == 'drikakistsangaris' or self.fluxvec.lower()=='dt':
                    if self.dim == 2: 
                        from Source.DiffEq.EulerFunctions import DrikakisTsangaris_diss_2D
                        self.DrikakisTsangaris = DrikakisTsangaris_diss_2D
                    self.dissipation = self.dissipation_upwind_DrikakisTsangaris
                else:
                    print("WARNING: fluxvec method not understood. Defaulting to fluxvec = 'lf'.")
                    self.fluxvec = 'lf'
                    self.dissipation = self.dissipation_upwind_lf

            elif self.type == 'zelalem' or self.type == 'entzelalem':

                if self.type == 'zelalem':
                    if self.jac_type == 'scalar' or self.jac_type == 'sca':
                        self.dissipation = self.dissipation_zelalem_scalar
                    elif self.jac_type == 'matrix' or self.jac_type == 'mat':
                        self.dissipation = self.dissipation_zelalem_matrix
                    else:
                        print("WARNING: Only scalar and matrix dissipation set up for type='zelalem' dissipation. Defaulting to jac_type = 'scalar'.")
                        self.jac_type = 'scalar'
                        self.dissipation = self.dissipation_zelalem_scalar
                else:
                    if self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                        self.dissipation = self.dissipation_entzelalem_scalarmatrix
                    elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                        self.dissipation = self.dissipation_entzelalem_matrixmatrix
                    else:
                        print("WARNING: jac method not understood for type='entzelalem' dissipation. Defaulting to jac_type = 'scalarmatrix'.")
                        self.jac_type = 'scalarmatrix'
                        self.dissipation = self.dissipation_entzelalem_scalarmatrix
                
                if 'eps_type' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['eps_type'], int), 'Artificial Dissipation: eps_type must be an int, {0}'.format(self.solver.vol_diss['eps_type'])
                    self.eps_type = self.solver.vol_diss['eps_type']
                else:
                    self.eps_type = 4

                if 'D_type' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['D_type'], str), 'Artificial Dissipation: D_type must be an str, {0}'.format(self.solver.vol_diss['D_type'])
                    assert self.solver.vol_diss['D_type'] in ['sbp','1st'], "D_type must be one of ['sbp','1st']"
                    self.D_type = self.solver.vol_diss['D_type']
                else:
                    self.D_type = 'sbp'
                    

            elif self.type == 'ranocha':
                if self.jac_type == 'scalar' or self.jac_type == 'sca':
                    self.dissipation = self.dissipation_ranocha_scalar
                else:
                    print("WARNING: Only scalar dissipation set up for type='ranocha' dissipation. Defaulting to jac_type = 'scalar'.")
                    self.jac_type = 'scalar'
                    self.dissipation = self.dissipation_ranocha_scalar
                
                if 'B_type' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['B_type'], int), 'Artificial Dissipation: B_type must be an int, {0}'.format(self.solver.vol_diss['B_type'])
                    self.B_type = self.solver.vol_diss['B_type']
                else:
                    self.B_type = 0

                if 'B_type' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['B_type'], int), 'Artificial Dissipation: B_type must be an int, {0}'.format(self.solver.vol_diss['B_type'])
                    self.B_type = self.solver.vol_diss['B_type']
                else:
                    self.B_type = 0
            
            elif self.type == 'filter' or self.type == 'entfilter':

                if self.type == 'filter':
                    if self.jac_type == 'scalar' or self.jac_type == 'sca':
                        self.dissipation = self.dissipation_filter_scalar
                    elif self.jac_type == 'matrix' or self.jac_type == 'mat':
                        raise Exception('Not yet implemented: filter matrix dissipation')
                        self.dissipation = self.dissipation_filter_matrix
                    else:
                        print("WARNING: Only scalar and matrix dissipation set up for type='filter' dissipation. Defaulting to jac_type = 'scalar'.")
                        self.jac_type = 'scalar'
                        self.dissipation = self.dissipation_filter_scalar
                elif self.type == 'entfilter':
                    if self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat':
                        self.dissipation = self.dissipation_entfilter_scalarmatrix
                    elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                        self.dissipation = self.dissipation_entfilter_matrixmatrix
                    else:
                        print("WARNING: jac method not understood for type='entfilter' dissipation. Defaulting to jac_type = 'scalarmatrix'.")
                        self.jac_type = 'scalarmatrix'
                        self.dissipation = self.dissipation_entfilter_scalarmatrix

                if 'alpha' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['alpha'], int) or isinstance(self.solver.vol_diss['alpha'], float), \
                    'Artificial Dissipation: filter alpha must be an int or float, {0}'.format(self.solver.vol_diss['alpha'])
                    self.filter_alpha = self.solver.vol_diss['alpha']
                else:
                    self.filter_alpha = 1

                if 'beta' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['beta'], int) or isinstance(self.solver.vol_diss['beta'], float), \
                    'Artificial Dissipation: filter beta must be an int or float, {0}'.format(self.solver.vol_diss['beta'])
                    self.filter_beta = self.solver.vol_diss['beta']
                else:
                    self.filter_beta = 4

                if 'eps_type' in self.solver.vol_diss.keys():
                    assert isinstance(self.solver.vol_diss['eps_type'], int), 'Artificial Dissipation: eps_type must be an int, {0}'.format(self.solver.vol_diss['eps_type'])
                    self.eps_type = self.solver.vol_diss['eps_type']
                else:
                    self.eps_type = 2


                


            
            else:
                raise Exception('Artificial dissipation: diss_type not understood, '+ str(self.type))
            
            if self.jac_type == 'scalarscalar' or self.jac_type == 'scasca' \
            or self.jac_type == 'scalarmatrix' or self.jac_type == 'scamat' \
            or self.jac_type == 'scalar' or self.jac_type == 'sca' \
            or (self.type == 'upwind' and self.fluxvec.lower() == 'lf') \
            or (self.type == 'upwindlgl' and self.fluxvec.lower() == 'lf'):
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
                    pass
                elif self.jac_type == 'matrixmatrix' or self.jac_type == 'matmat':
                    if self.neq_node == 1:
                        if hasattr(self.solver.diffeq, 'dExdw_abs'):
                            self.dExdw_abs = lambda q: self.solver.diffeq.dExdw_abs(q, self.entropy_fix)
                        else:
                            print('WARNING: diffeq.dExdw_abs not found. Defaulting to dExdq_abs @ dqdw, which is slower.')
                            self.dExdw_abs = self.calc_absAP_base_1d
                    else:
                        if self.dim == 1:
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
                self.gdiag_lm = sp.gdiag_lm
                self.lm_to_lmT = sp.lm_to_lmT
                self.lm_lm = sp.lm_lm
                self.lm_ldiag = sp.lm_ldiag
                self.ldiag_lm = sp.ldiag_lm
                self.kron_neq_lm = sp.kron_neq_lm
                self.kron_neq_gm = sp.kron_neq_gm
                self.kron_lm_eye = sp.kron_lm_eye
                self.kron_eye_lm = sp.kron_eye_lm
                self.kron_lm_ldiag = sp.kron_lm_ldiag
                self.kron_ldiag_lm = sp.kron_ldiag_lm
            else:
                self.gm_gv = fn.gm_gv
                self.lm_gv = fn.lm_gv
                self.gdiag_gm = fn.gdiag_gm
                self.gdiag_lm = fn.gdiag_lm
                self.lm_to_lmT = lambda lm, *args: lm.T
                self.lm_lm = lambda lm1, lm2: lm1 @ lm2
                self.lm_ldiag = fn.lm_ldiag
                self.ldiag_lm = fn.ldiag_lm
                self.kron_neq_lm = fn.kron_neq_lm
                self.kron_neq_gm = fn.kron_neq_gm
                self.kron_lm_eye = lambda lm, nen: np.kron(lm, np.eye(nen))
                self.kron_eye_lm = lambda lm, nen, nen2: np.kron(np.eye(nen), lm)
                self.kron_lm_ldiag = fn.kron_lm_ldiag
                self.kron_ldiag_lm = lambda diag, Dx, nen: fn.kron_ldiag_lm(diag, Dx)

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
        self.dxi = 1./(self.nen-1) # this is the reference spacing. Physical spacing is taken care of implicitly by metrics.
        self.dx = (self.solver.mesh.bdy_x[1,:]-self.solver.mesh.bdy_x[0,:]) # physical spacing - element size

        if self.type == 'w' or self.type == 'entw':
            Ds1 = self.solver.sbp.D
            if self.sparse: Ds1 = sp.lm_to_sp(Ds1)
            Ds = Ds1.copy()
            for i in range(1,self.s):
                Ds = self.lm_lm(Ds1, Ds)
            if self.sparse: Ds.prune()
            DsT = self.lm_to_lmT(Ds,self.nen,self.nen)
            
            self.rhs_D = self.kron_neq_lm(Ds,self.neq_node) 
            if self.use_H:
                H = np.diag(self.solver.sbp.H)
            else:
                H = np.ones(self.nen)
            self.lhs_D = self.gdiag_lm(-(self.solver.H_inv_phys * self.dxi**(2*self.s-1)),self.kron_neq_lm(self.lm_ldiag(DsT, H),self.neq_node))
        elif self.type == 'dcp' or self.type == 'entdcp' or self.type == 'dcp2':
            if self.type == 'dcp2':
                Ds, B = make_dcp_diss_op2(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            else:
                Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            if self.sparse: Ds = sp.lm_to_sp(Ds)
            self.rhs_D = self.kron_neq_lm(Ds,self.neq_node) 
            DsT = self.lm_to_lmT(Ds,self.nen,self.nen)
            if self.use_H:
                Hundvd = np.diag(self.solver.sbp.H) / self.solver.sbp.dx
                self.lhs_D = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.lm_ldiag(DsT, B * Hundvd), self.neq_node))
            else:
                self.lhs_D = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.lm_ldiag(DsT, B), self.neq_node))
        elif self.type == 'upwind':
            if self.solver.disc_nodes.lower() == 'upwind':
                ### OPTION 1: USE UPWIND DISSIPATION OPERATOR FROM CENTRAL SBP OPERATOR
                Ddiss = self.solver.sbp.Ddiss
                if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
                self.Ddiss = self.gdiag_lm( self.repeat_neq_gv(-self.solver.mesh.det_jac_inv), self.kron_neq_lm(Ddiss,self.neq_node))
            else:
                if self.solver.disc_nodes.lower() == 'lgl' or self.solver.disc_nodes.lower() == 'lg':
                    ### OPTION 2: USE UPWIND DISSIPATION DEFINED BY ELEMENT-TYPE
                    if self.bdy_fix: 
                        print('WARNING: ignoring bdy_fix and setting bdy_fix=False for element volume-dissipation.')
                        self.bdy_fix = False
                    if self.use_H: 
                        print('WARNING: ignoring use_H and setting use_H=False for element volume-dissipation.')
                        self.use_H = False
                    Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, False)
                    DsTDs = Ds.T @ Ds
                    if self.sparse: Ds = sp.lm_to_sp(DsTDs)
                    self.Ddiss = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(DsTDs,self.neq_node))
                else:
                    ### OPTION 3: USE UPWIND DISSIPATION OPERATOR FOR CLASSICAL SBP (MATTSSON)
                    from Source.Disc.UpwindOp import UpwindOp
                    _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindOp(self.s,self.nen)
                    if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                        print('WARNING: x of sbp operator does not match x of dissipation operator!')
                        print(self.solver.sbp.x)
                        print(x)
                    if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                        print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
                    if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
                    self.Ddiss = self.gdiag_lm( self.repeat_neq_gv(-self.solver.mesh.det_jac_inv), self.kron_neq_lm(Ddiss,self.neq_node))
        elif self.type == 'upwindlgl':
            assert (self.solver.disc_nodes.lower() == 'lgl'), 'upwindLGL dissipation only implemented for LGL.'
            assert (self.s == self.solver.p), 'upwindLGL dissipation only implemented for s=p.'
            from Source.Disc.UpwindOp import UpwindLGL
            _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindLGL(self.s,self.coeff)
            # reset coeff since it is contained in Ddiss
            self.coeff = 1.
            if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                print('WARNING: x of sbp operator does not match x of dissipation operator!')
                print(self.solver.sbp.x)
                print(x)
            if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
            if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
            self.Ddiss = self.gdiag_lm( self.repeat_neq_gv(-self.solver.mesh.det_jac_inv), self.kron_neq_lm(Ddiss,self.neq_node))
        elif self.type == 'zelalem' or self.type == 'entzelalem':
            if self.solver.disc_nodes.lower() in ['lgl', 'lg']:
                if self.D_type == 'sbp':
                    Ds = self.solver.sbp.D
                elif self.D_type == '1st':
                    nn = self.solver.sbp.nn
                    x = self.solver.sbp.x
                    Ds = np.zeros((nn,nn))
                    h = 1/(x[1] - x[0])
                    Ds[0,0], Ds[0,1] = -h, h
                    for i in range(1,nn):
                        h = 1/(x[i] - x[i-1])
                        Ds[i,i-1], Ds[i,i] = -h, h 
                else:
                    raise Exception('Something went wrong')
                Ds = Ds / (self.solver.sbp.nn-1) # make undivided
                for i in range(1,self.s):
                    Ds = fn.lm_lm(self.solver.sbp.D, Ds)
                B = np.ones(self.nen) # can play with later
            else:
                Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            if self.sparse: Ds = sp.lm_to_sp(Ds)
            self.rhs_D = self.kron_neq_lm(Ds,self.neq_node) 
            DsT = self.lm_to_lmT(Ds,self.nen,self.nen)
            if self.use_H:
                Hundvd = np.diag(self.solver.sbp.H) / self.solver.sbp.dx
                self.lhs_D = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.lm_ldiag(DsT, B * Hundvd), self.neq_node))
            else:
                self.lhs_D = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.lm_ldiag(DsT, B), self.neq_node))
        elif self.type == 'ranocha':
            # TODO: I have a feeling this is wrong, becuase it doesn't work as well as it should
            # NOTE: For now using the same central operator for FD, but might want to use a different approximation...
            D = self.solver.sbp.D / (self.solver.sbp.nn-1) # make undivided
            if self.B_type == 0:
                B = np.ones(self.nen)
            elif self.B_type == 1:
                B = np.ones(self.nen)
                B[0:self.solver.p] = 0
                B[-self.solver.p:] = 0
            elif self.B_type == 2:
                x = self.solver.sbp.x
                B = 1. - (2.*x - 1.)**2
            if self.sparse: D = sp.lm_to_sp(D)
            self.rhs_D = self.kron_neq_lm(D,self.neq_node) 
            DT = self.lm_to_lmT(D,self.nen,self.nen)
            if self.use_H:
                Hinv_undvd = self.solver.sbp.dx / np.diag(self.solver.sbp.H)
                self.lhs_D = self.kron_neq_lm(self.ldiag_lm(Hinv_undvd, self.lm_ldiag(DT, B)), self.neq_node)
            else:
                self.lhs_D = self.kron_neq_lm(self.lm_ldiag(DT, B), self.neq_node)
        elif self.type == 'filter' or self.type == 'entfilter':
            # Currently not dimensionally consistent. Where would I put the variable coefficient?
            assert self.solver.disc_nodes.lower() in ['lgl', 'lg'], 'Filter dissipation only implemented for LGL and LG.'
            from Source.Disc.MakeDgOp import MakeDgOp
            V = MakeDgOp.VandermondeLegendre1D(self.solver.sbp.x,self.solver.p)
            #Vinv = np.linalg.inv(V) # since square, equal to projection P
            H = np.diag(self.solver.sbp.H)
            M = np.diag((V.T * H) @ V)  # should be diagonal TODO: Generalize for non LGL or LG?
            Minv = 1/M
            P = Minv[:,np.newaxis] * V.T * H # # since square, equal to Vinv

            #F_diag = 1-MakeDgOp.Filter1D(self.solver.p,int(self.filter_Nc),self.filter_s)
            x = np.linspace(0,1,self.nen)
            #F_diag = self.filter_alpha*x**self.filter_beta
            F_diag = x**self.filter_beta
            C = M * F_diag # should be diagonal
            Csqrt = np.sqrt(C)
            FilterRHS = (V * Csqrt) @ P
            if self.use_H:
                FilterLHS = FilterRHS.T * H
            else:
                FilterLHS = FilterRHS.T

            # this Vandermonde maps from Legendre modal coefficients to Lagrange nodal values
            if self.sparse: 
                FilterRHS = sp.lm_to_sp(FilterRHS)
                FilterLHS = sp.lm_to_sp(FilterLHS) 

            self.FilterRHS = self.kron_neq_lm(FilterRHS, self.neq_node)
            self.FilterLHS = self.gdiag_lm(-self.solver.H_inv_phys, self.kron_neq_lm(FilterLHS, self.neq_node))
            


            
        elif self.type == 'b' or self.type == 'entb':
            assert(self.solver.disc_nodes.lower() == 'csbp'), 'Baseline dissipation only implemented for csbp.'
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
                if self.sparse:
                    self.rhs_D1 = sp.lm_to_sp(self.rhs_D1)
                    self.rhs_D2 = sp.lm_to_sp(self.rhs_D2)
                    self.lhs_D1 = sp.gm_to_sp(self.lhs_D1)
                    self.lhs_D2 = sp.gm_to_sp(self.lhs_D2)
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
                if self.sparse:
                    self.rhs_D2 = sp.lm_to_sp(self.rhs_D2)
                    self.rhs_D3 = sp.lm_to_sp(self.rhs_D3)
                    self.rhs_D4 = sp.lm_to_sp(self.rhs_D4)
                    self.lhs_D2 = sp.gm_to_sp(self.lhs_D2)
                    self.lhs_D3 = sp.gm_to_sp(self.lhs_D3)
                    self.lhs_D4 = sp.gm_to_sp(self.lhs_D4)
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
                if self.sparse:
                    self.rhs_D3 = sp.lm_to_sp(self.rhs_D3)
                    self.rhs_D4 = sp.lm_to_sp(self.rhs_D4)
                    self.rhs_D5 = sp.lm_to_sp(self.rhs_D5)
                    self.rhs_D6 = sp.lm_to_sp(self.rhs_D6)
                    self.lhs_D3 = sp.gm_to_sp(self.lhs_D3)
                    self.lhs_D4 = sp.gm_to_sp(self.lhs_D4)
                    self.lhs_D5 = sp.gm_to_sp(self.lhs_D5)
                    self.lhs_D6 = sp.gm_to_sp(self.lhs_D6)
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
                if self.sparse:
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
            else:
                raise ValueError('only s=1,2,3,4 coded up for baseline dissipation.')

        else:
            raise Exception(self.type + ' not set up yet')

        
    def set_ops_2D(self):
        ''' prepare the various operators needed for the dissipation function '''
        self.dxi = 1./((self.nen-1)*self.solver.nelem[0]) # this is the reference spacing. Physical spacing is taken care of implicitly by metrics.
        self.deta = 1./((self.nen-1)*self.solver.nelem[0])
        # NOTE: assumes D is the same in each direction

        if self.type == 'W' or self.type == 'entW':
            Ds1 = self.solver.sbp.D
            if self.sparse: Ds1 = sp.lm_to_sp(Ds1)
            Ds = Ds1.copy()
            for i in range(1,self.s):
                Ds = self.lm_lm(Ds1, Ds)
            if self.sparse: Ds.prune()
            DsT = self.lm_to_lmT(Ds,self.nen,self.nen)
            H = np.diag(self.solver.sbp.H)

            self.rhs_Dxi = self.kron_neq_lm(self.kron_lm_eye(Ds, self.nen),self.neq_node) 
            self.rhs_Deta = self.kron_neq_lm(self.kron_eye_lm(Ds, self.nen, self.nen),self.neq_node) 
            DsTxi = self.kron_lm_eye(self.lm_ldiag(DsT, H), self.nen)
            DsTeta = self.kron_eye_lm(self.lm_ldiag(DsT, self.solver.sbp.H), self.nen, self.nen)
            self.lhs_Dxi = self.kron_neq_gm(self.gdiag_lm(-(self.solver.H_inv_phys * self.dxi**(2*self.s-1)),DsTxi),self.neq_node) 
            self.lhs_Deta = self.kron_neq_gm(self.gdiag_lm(-(self.solver.H_inv_phys * self.deta**(2*self.s-1)),DsTeta),self.neq_node) 
        elif self.type == 'dcp' or self.type == 'entdcp':                
            Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            if self.sparse: Ds = sp.lm_to_sp(Ds)

            DsT = self.lm_to_lmT(Ds,self.nen,self.nen)
            self.rhs_Dxi = self.kron_neq_lm(self.kron_lm_eye(Ds, self.nen),self.neq_node) 
            self.rhs_Deta = self.kron_neq_lm(self.kron_eye_lm(Ds,self.nen,self.nen),self.neq_node) 
            if self.use_H:
                H = np.diag(self.solver.sbp.H)
                Hundvd = H / self.solver.sbp.dx
                DsTBH = self.lm_ldiag(DsT, B * Hundvd)
                DsTxi = self.kron_lm_ldiag(DsTBH, H)
                DsTeta = self.kron_ldiag_lm(H, DsTBH, self.nen)
            else:
                if self.use_noH:
                    # uses no H at all
                    DsTB = self.lm_ldiag(DsT, B * self.solver.sbp.dx)
                    DsTxi = self.kron_lm_eye(DsTB, self.nen)
                    DsTeta = self.kron_eye_lm(DsTB, self.nen, self.nen)
                else:
                    # uses divided H in the perpendicular direction
                    H = np.diag(self.solver.sbp.H)
                    DsTB = self.lm_ldiag(DsT, B)
                    DsTxi = self.kron_lm_ldiag(DsTB, H) 
                    DsTeta = self.kron_ldiag_lm(H, DsTB, self.nen)
            self.lhs_Dxi = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(DsTxi,self.neq_node))
            self.lhs_Deta = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(DsTeta,self.neq_node))
        elif self.type == 'upwind':
            if self.solver.disc_nodes.lower() == 'upwind':
                ### OPTION 1: USE UPWIND DISSIPATION DEFINED BY CENTRAL OPERATOR
                Ddiss = self.solver.sbp.Ddiss
                if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
                self.Dxidiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_lm_eye(Ddiss, self.nen),self.neq_node)) 
                self.Detadiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_eye_lm(Ddiss, self.nen, self.nen),self.neq_node))
            else:
                if self.solver.disc_nodes.lower() == 'lgl' or self.solver.disc_nodes.lower() == 'lg':
                    ### OPTION 2: USE UPWIND DISSIPATION DEFINED BY ELEMENT-TYPE
                    if self.bdy_fix: 
                        print('WARNING: ignoring bdy_fix and setting bdy_fix=False for element volume-dissipation.')
                        self.bdy_fix = False
                    if self.use_H: 
                        print('WARNING: ignoring use_H and setting use_H=False for element volume-dissipation.')
                        self.use_H = False
                    if self.use_noH: 
                        print('WARNING: ignoring use_noH and setting use_noH=False for element volume-dissipation.')
                        self.use_noH = False
                    Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, False)
                    DsTDs = Ds.T @ Ds
                    if self.sparse: DsTDs = sp.lm_to_sp(DsTDs)
                    H = np.diag(self.solver.sbp.H)
                    self.Dxidiss = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.kron_lm_ldiag(DsTDs, H),self.neq_node))
                    self.Detadiss = self.gdiag_lm(-self.solver.H_inv_phys,self.kron_neq_lm(self.kron_ldiag_lm(H, DsTDs, self.nen),self.neq_node))
                else:
                    ### OPTION 3: USE UPWIND DISSIPATION DEFINED BY CLASSICAL NODES (MATTSSON)
                    from Source.Disc.UpwindOp import UpwindOp
                    _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindOp(self.s,self.nen)
                    if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                        print('WARNING: x of sbp operator does not match x of dissipation operator!')
                        print(self.solver.sbp.x)
                        print(x)
                    if np.any(abs(np.diag(H) - np.diag(self.solver.sbp.H)) > 1e-14):
                        print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
                    if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
                    self.Dxidiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_lm_eye(Ddiss, self.nen),self.neq_node)) 
                    self.Detadiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_eye_lm(Ddiss, self.nen, self.nen),self.neq_node))
        elif self.type == 'upwindlgl':
            # takes from the upwind sbp paper (same as above, just different coefficient)
            assert (self.solver.disc_nodes.lower() == 'lgl'), 'upwindLGL dissipation only implemented for LGL.'
            assert (self.s == self.solver.p), 'upwindLGL dissipation only implemented for s=p.'
            from Source.Disc.UpwindOp import UpwindLGL
            _,_,_,_,H,_,_,_,_,x,Ddiss = UpwindLGL(self.s,self.coeff)
            # reset coeff since it is contained in Ddiss
            self.coeff = 1.
            if np.any(abs(x - self.solver.sbp.x) > 1e-14):
                print('WARNING: x of sbp operator does not match x of dissipation operator!')
                print(self.solver.sbp.x)
                print(x)
            if np.any(abs(H - self.solver.sbp.H) > 1e-14):
                print('WARNING: H of sbp operator does not match H of dissipation operator! Not provably stable.')
            if self.sparse: Ddiss = sp.lm_to_sp(Ddiss)
            self.Dxidiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_lm_eye(Ddiss, self.nen),self.neq_node)) 
            self.Detadiss = self.gdiag_lm( fn.repeat_neq_gv(-self.solver.mesh.det_jac_inv,self.neq_node), self.kron_neq_lm(self.kron_eye_lm(Ddiss, self.nen, self.nen),self.neq_node))

        else:
            raise Exception('ADiss: diss_type ' + self.type + ' not set up yet')


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
    
    def dissipation_upwind_burgers(self, q):
        ''' dissipation function for fully upwind burgers flux-vector splitting '''
        if self.dim == 1:
            diss = self.gm_gv(self.Ddiss, 0.5 * abs(q) * q)
        else:
            raise Exception('Burgers dissipation only implemented for 1D.')
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
    
    def dissipation_upwind_DrikakisTsangaris(self, q):
        ''' dissipation function for upwind steger warming flux-vector splitting'''
        if self.dim == 2:
            fdiss = self.DrikakisTsangaris(q,self.dxidx)
            diss = self.gm_gv(self.Dxidiss, fdiss) # xi part
            fdiss = self.DrikakisTsangaris(q,self.detadx)
            diss += self.gm_gv(self.Detadiss, fdiss) # eta part
        else:
            raise Exception('TODO: DrikakisTsangaris dissipation only implemented for 2D.')
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
    
    def dissipation_dcp_matrix(self, q):
        ''' dissipation function for w and dcp, systems'''
        # TODO: for now we always take a simple sum at half-nodes, but should generalize.
        if self.dim == 1:
            A = self.dExdq_abs(q)
            if self.s % 2 == 1 and self.avg_half_nodes:
                A[:-1] = 0.5*(A[:-1] + A[1:])
            diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(A, self.lm_gv(self.rhs_D, q)))
        elif self.dim == 2:
            A = self.dEndq_abs(q,self.dxidx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                A[:-self.nen:self.nen] = 0.5*(A[:-self.nen:self.nen] + A[self.nen::self.nen])
            diss = self.gm_gv(self.lhs_Dxi, fn.gbdiag_gv(A, self.lm_gv(self.rhs_Dxi, q)))
            A = self.dEndq_abs(q,self.detadx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    A[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(A[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + A[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            diss += self.gm_gv(self.lhs_Deta, fn.gbdiag_gv(A, self.lm_gv(self.rhs_Deta, q)))
        elif self.dim == 3:
            raise Exception('TODO')
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
            AP = self.repeat_neq_gv(AP)
            diss = self.gm_gv(self.lhs_D, AP * self.lm_gv(self.rhs_D, w))
        elif self.dim == 2:
            maxeig = self.maxeig_dEndq(q,self.dxidx) 
            AP = maxeig*rho_dqdw
            if self.s % 2 == 1 and self.avg_half_nodes:
                AP[:-self.nen:self.nen] = 0.5*(AP[:-self.nen:self.nen] + AP[self.nen::self.nen])
            AP = self.repeat_neq_gv(AP)
            diss = self.gm_gv(self.lhs_Dxi, AP * self.lm_gv(self.rhs_Dxi, w)) # xi part
            maxeig = self.maxeig_dEndq(q,self.detadx) 
            AP = maxeig*rho_dqdw
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + AP[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            AP = self.repeat_neq_gv(AP)  
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
            AP = self.repeat_neq_gv(maxeig*rho_dqdw,self.neq_node)
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
            diss = self.gm_gv(self.lhs_Dxi, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Dxi, w)))
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
            diss = self.gm_gv(self.lhs_Dxi, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Dxi, w)))
            AP = self.dEndw_abs(q,self.detadx)
            if self.s % 2 == 1 and self.avg_half_nodes:
                for xi_idx in range(self.nen):
                    AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] = 0.5*(AP[xi_idx*self.nen:(xi_idx+1)*self.nen-1] + AP[xi_idx*self.nen+1:(xi_idx+1)*self.nen])
            diss += self.gm_gv(self.lhs_Deta, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Deta, w)))
        elif self.dim == 3:
            if self.avg_half_nodes:
                raise Exception('TODO')
            AP = self.dEndw_abs(q,self.dxidx)
            diss = self.gm_gv(self.lhs_Dxi, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_Dxi, w)))
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
    
    def dissipation_zelalem_scalar(self, q):
        ''' dissipation function for zelalem's idea, scalar functions or systems'''
        if self.dim == 1:
            if self.use_A:
                maxeig = self.maxeig_dExdq(q)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
                A = self.repeat_neq_gv(maxeig)
            else:
                A = 1
            diss = self.gm_gv(self.lhs_D, A * self.lm_gv(self.rhs_D, q))
            if self.eps_type == 0:
                # plain idea - low order dissipation, should scale as h**(s-1)
                # TODO: scaling does not come from h**s, but from Jacobian...
                coeff = self.coeff
                #self.dxi**(self.solver.p-self.s+1)
            elif self.eps_type == 1:
                # simple idea to increase effective order, multiply by h**(p-s+1)
                # this makes the boundary scale as h**p and interior as h**(s+p)
                # e.g. for mattsson 2004 dissipation, using s=p we get h**p boundary and h**2p interior 
                coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
                #coeff = self.dxi**self.solver.p * self.coeff
            else:
                coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
                # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
                # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
                # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
                # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
            return coeff*diss

        else:
            raise Exception('TODO')
        
    def dissipation_zelalem_matrix(self, q):
        ''' dissipation function for zelalem's idea, systems'''
        if self.dim == 1:
            if self.use_A:
                A = self.dExdq_abs(q)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    A[:-1] = 0.5*(A[:-1] + A[1:])
                diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(A, self.lm_gv(self.rhs_D, q)))
            else:
                diss = self.gm_gv(self.lhs_D, self.lm_gv(self.rhs_D, q))

            if self.eps_type == 0:
                # plain idea - low order dissipation, should scale as h**(s-1)
                # TODO: scaling does not come from h**s, but from Jacobian...
                coeff = self.coeff
                #self.dxi**(self.solver.p-self.s+1)
            elif self.eps_type == 1:
                # simple idea to increase effective order, multiply by h**(p-s+1)
                # this makes the boundary scale as h**p and interior as h**(s+p)
                # e.g. for mattsson 2004 dissipation, using s=p we get h**p boundary and h**2p interior 
                coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
                #coeff = self.dxi**self.solver.p * self.coeff
            else:
                coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
                # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
                # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
                # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
                # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node

            return coeff*diss
            #diss_reshaped = diss.reshape((self.nen, self.neq_node, self.nelem), order='C')
            #res = np.multiply(diss_reshaped, coeff[np.newaxis, :, :], out=diss_reshaped).reshape((self.nen * self.neq_node, self.nelem)) 
            #return res

        else:
            raise Exception('TODO')
        
    def dissipation_entzelalem_scalarmatrix(self, q):
        ''' dissipation function for zelalem's idea, scalar functions or systems'''

        w = self.entropy_var(q)
        if self.dim == 1:
            if self.use_A:
                dqdw = self.dqdw(q)
                maxeig = self.repeat_neq_gv(self.maxeig_dExdq(q))
                AP = fn.gdiag_gbdiag(maxeig, dqdw)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    AP[:-1] = 0.5*(AP[:-1] + AP[1:])
                diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
            else:
                diss = self.gm_gv(self.lhs_D, self.lm_gv(self.rhs_D, w))

            if self.eps_type == 0:
                # plain idea - low order dissipation, should scale as h**(s-1)
                # TODO: scaling does not come from h**s, but from Jacobian...
                coeff = self.coeff
                #self.dxi**(self.solver.p-self.s+1)
            elif self.eps_type == 1:
                # simple idea to increase effective order, multiply by h**(p-s+1)
                # this makes the boundary scale as h**p and interior as h**(s+p)
                # e.g. for mattsson 2004 dissipation, using s=p we get h**p boundary and h**2p interior 
                coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
                #coeff = self.dxi**self.solver.p * self.coeff
            else:
                coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
                # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
                # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
                # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
                # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
            return coeff*diss

        else:
            raise Exception('TODO')
        
    def dissipation_entzelalem_matrixmatrix(self, q):
        ''' dissipation function for zelalem's idea, scalar functions or systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            if self.use_A:
                AP = self.dExdw_abs(q)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    AP[:-1] = 0.5*(AP[:-1] + AP[1:])
                diss = self.gm_gv(self.lhs_D, fn.gbdiag_gv(AP, self.lm_gv(self.rhs_D, w)))
            else:
                diss = self.gm_gv(self.lhs_D, self.lm_gv(self.rhs_D, w))

            if self.eps_type == 0:
                # plain idea - low order dissipation, should scale as h**(s-1)
                # TODO: scaling does not come from h**s, but from Jacobian...
                coeff = self.coeff
                #self.dxi**(self.solver.p-self.s+1)
            elif self.eps_type == 1:
                # simple idea to increase effective order, multiply by h**(p-s+1)
                # this makes the boundary scale as h**p and interior as h**(s+p)
                # e.g. for mattsson 2004 dissipation, using s=p we get h**p boundary and h**2p interior 
                coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
                #coeff = self.dxi**self.solver.p * self.coeff
            else:
                coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
                # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
                # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
                # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
                # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
            return coeff*diss

        else:
            raise Exception('TODO')
        
    def dissipation_ranocha_scalar(self, q):
        ''' dissipation function for Ranocha's dissipation, scalar functions or systems'''
        # TODO: Do something about the variable coefficient... take a sth root?
        if self.dim == 1:
            maxeig = self.maxeig_dExdq(q)
            if self.avg_half_nodes:
                maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
            A = self.repeat_neq_gv(maxeig)
            diss = self.lm_gv(self.lhs_D, A * self.lm_gv(self.rhs_D, q))
            for i in range(self.s-1):
                diss = self.lm_gv(self.lhs_D, A * self.lm_gv(self.rhs_D, diss))
            if self.use_H:
                # already included undivided Hinv, but not jacobian
                diss = (- self.coeff / self.solver.sbp.dx) * (self.solver.mesh.det_jac_inv * diss)
            else:
                diss = (- self.coeff) * (self.solver.H_inv_phys * diss)
            return diss

        else:
            raise Exception('TODO')
        
    def dissipation_filter_scalar(self, q):
        ''' dissipation function for filter-based dissipation, scalar functions or systems'''
        if self.dim == 1:
            if self.use_A:
                maxeig = self.maxeig_dExdq(q)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    maxeig[:-1] = 0.5*(maxeig[:-1] + maxeig[1:])
                A = self.repeat_neq_gv(maxeig)
            else:
                A = 1
            diss = self.gm_gv(self.FilterLHS, A * self.lm_gv(self.FilterRHS, q))

        else:
            raise Exception('TODO')
        
        if self.eps_type == 0:
            # plain idea - low order dissipation, should scale as h**(s-1)
            coeff = self.coeff
        elif self.eps_type == 1:
            coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
        else:
            coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
            # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
            # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
            # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
            # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
        return coeff*diss
        
    def dissipation_entfilter_scalarmatrix(self, q):
        ''' dissipation function for filter-based dissipation, scalar functions or systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            if self.use_A:
                dqdw = self.dqdw(q)
                maxeig = self.repeat_neq_gv(self.maxeig_dExdq(q))
                AP = fn.gdiag_gbdiag(maxeig, dqdw)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    AP[:-1] = 0.5*(AP[:-1] + AP[1:])
                diss = self.gm_gv(self.FilterLHS, fn.gbdiag_gv(AP, self.lm_gv(self.FilterRHS, w)))
            else:
                raise Exception("TODO")
                diss = self.gm_gv(self.FilterLHS, fn.gbdiag_gv(AP, self.lm_gv(self.FilterRHS, w)))

        else:
            raise Exception('TODO')
        
        if self.eps_type == 0:
            # plain idea - low order dissipation, should scale as h**(s-1)
            coeff = self.coeff
        elif self.eps_type == 1:
            coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
        else:
            coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
            # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
            # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
            # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
            # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
        return coeff*diss
        
    def dissipation_entfilter_matrixmatrix(self, q):
        ''' dissipation function for filter-based dissipation, scalar functions or systems'''
        w = self.entropy_var(q)
        if self.dim == 1:
            if self.use_A:
                AP = self.dExdw_abs(q)
                if self.s % 2 == 1 and self.avg_half_nodes:
                    AP[:-1] = 0.5*(AP[:-1] + AP[1:])
                diss = self.gm_gv(self.FilterLHS, fn.gbdiag_gv(AP, self.lm_gv(self.FilterRHS, w)))
            else:
                raise Exception('TODO')
        else:
            raise Exception('TODO')
        
        if self.eps_type == 0:
            # plain idea - low order dissipation, should scale as h**(s-1)
            coeff = self.coeff
        elif self.eps_type == 1:
            coeff = self.dx**(self.solver.p-self.s+1) * self.coeff
        else:
            coeff = self.coeff * self.solver.zelalem_diss_coeff(q)
            # self.eps_type = 2: boundary solution jumps, with normalization of neq_node
            # self.eps_type = 3: diff between cons. and non-cons. flux derivative, with normalization of neq_node
            # self.eps_type = 21: boundary solution jumps, without normalization of neq_node
            # self.eps_type = 31: diff between cons. and non-cons. flux derivative, without normalization of neq_node
                
        return coeff*diss
    
    def dissipation_filter_matrix(self, q):
        raise Exception('TODO')
        ''' dissipation function for Ranocha's dissipation, systems'''
        # TODO: Do something about the variable coefficient...
        if self.dim == 1:
            # TODO: something with coefficient
            # TODO: Have different filter for each equation?
            diss = self.gm_gv(self.Filter, q)
        else:
            raise Exception('TODO')
        
        if self.eps_type == 0:
            # plain idea - low order dissipation, should scale as h**(s-1)
            coeff = np.ones((self.neq_node, self.nelem)) * self.coeff
        elif self.eps_type == 2:
            # use difference between conservative and non-conservative flux derivative
            dEdx1 = self.solver.gm_gv(self.solver.Dx, self.solver.diffeq.calcEx(q))
            dEdx2 = self.solver.diffeq.nonconservative_coeff(q) * self.solver.gm_gv(self.solver.Dx, q)  
            #coeff = self.coeff * np.linalg.norm((dEdx1 - dEdx2).reshape((self.nen, self.neq_node, self.nelem),order='C'), axis=0) 
            coeff =  fn.repeat_nen_gv(np.abs(dEdx1 - dEdx2) * self.coeff, self.nen)  
        elif self.eps_type == 3:
            # use boundaries
            v = np.copy(q)
            v_a = self.solver.sat.lm_gv(self.solver.sat.tLT, v)
            v_b = self.solver.sat.lm_gv(self.solver.sat.tRT, v)
            # assume periodic boundaries
            vf_L = fn.pad_1dL(v_b, v_b[:,-1]) # this is the solution to the left of the interface
            vf_R = fn.pad_1dR(v_a, v_a[:,0]) # this is the solution to the right of the interface
            vf_jump = np.abs(vf_R - vf_L) # the absolute jump at each interface
            #coeff = 0.5*(vf_jump[:,1:] + vf_jump[:,:-1]) * self.coeff
            coeff = fn.repeat_nen_gv(0.5*(vf_jump[:,1:] + vf_jump[:,:-1]) * self.coeff, self.nen)
        return coeff*diss
        
        #diss_reshaped = diss.reshape((self.nen, self.neq_node, self.nelem), order='C')
        #res = np.multiply(diss_reshaped, coeff[np.newaxis, :, :], out=diss_reshaped).reshape((self.nen * self.neq_node, self.nelem)) 
        #return res
    
    
    
    def calc_LHS(self, q=None, step=1.0e-4):
        ''' could form explicitly... but for simplicity just do finite difference. 
        Note: this does not include the coefficient '''
        if self.type == 'nd':
            return 0.0
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
    
    def dispersion_analysis(self):
        ''' return the contribution to the dispersion analysis 
        aka just the dissipation operator on the reference element, without Hinv'''
        if self.type == 'dcp' or self.type == 'dcp2':
            if self.type == 'dcp2':
                Ds, B = make_dcp_diss_op2(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            else:
                Ds, B = make_dcp_diss_op(self.solver.disc_nodes, self.s, self.nen, self.bdy_fix)
            if self.use_H:
                Hundvd = np.diag(self.solver.sbp.H) / self.solver.sbp.dx
                A = self.coeff * (Ds.T * B * Hundvd) @ Ds 
            else:
                A = self.coeff * (Ds.T * B) @ Ds 
            return A
            # note lack of negative, which is intentional since we use the negative in the main function
            # well, actually negative is missing there because we then take the negative of the whole thing

        elif self.type == 'upwind':
            Ddiss = self.solver.sbp.Ddiss
            A = self.coeff * Ddiss
            return A

        elif self.type == 'nd':
            return np.zeros((self.nen,self.nen))

        else:
            raise Exception('dispersion_analysis not implemented for diss_type ' + self.type)





        
