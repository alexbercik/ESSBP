#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:42:28 2021

@author: bercik
"""
#import numpy as np

from Source.Disc.SatDer1 import SatDer1
from Source.Disc.SatDer2 import SatDer2
import numpy as np
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp


class Sat(SatDer1, SatDer2):

    # TODO: Finish the comments here
    '''
    The methods calc_sat_unstruc,
    '''

    def __init__(self, solver, direction, met_form):
        '''
        Sets up the SAT for a particular direction used in solver 

        Parameters
        ----------
        solver : class instance
            The solver class which contains all the important functions.
        direction : str
            the direction in computational space for SAT (xi).
        met_form: str
            metrics formulation. Either 'skew_sym' or 'div'
        '''
        
        self.print_progress = solver.print_progress
        if self.print_progress: print('... Setting up SATs')
        
        self.diss_type = solver.surf_diss['diss_type']
        self.disc_type = solver.disc_type
        self.diffeq_name = solver.diffeq.diffeq_name
        self.dim = solver.dim
        self.shape = solver.qshape
        self.nen = solver.nen
        self.neq_node = solver.neq_node
        self.nelem = solver.nelem
        self.direction = direction
        self.met_form = met_form
        assert met_form=='skew_sym','SATs not currently set up for divergence form metrics'
        self.bc = solver.bc
        self.sparse = solver.sat_sparse
        self.solver_sparse = solver.sparse
        self.sparsity, self.sparsity_unkronned = None, None
        self.xsparsity, self.xsparsity_unkronned = None, None
        self.ysparsity, self.ysparsity_unkronned = None, None
        self.zsparsity, self.zsparsity_unkronned = None, None

        if self.diss_type not in ['nd','symmetric','upwind','lf','llf','lax_friedrichs','ec']:
            if 'jac_type' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['jac_type'], str), 'SAT: jac_type must be a str, {0}'.format(solver.surf_diss['jac_type'])
                self.jac_type = solver.surf_diss['jac_type'].lower()
            else:
                self.jac_type = 'sca'
            if self.jac_type == 'scalar': self.jac_type = 'sca'
            if self.jac_type == 'matrix': self.jac_type = 'mat'
            if self.jac_type == 'scalar_matrix' or self.jac_type == 'scalarmatrix': self.jac_type = 'scamat'
            if self.jac_type == 'scalar_scalar' or self.jac_type == 'scalarscalar': self.jac_type = 'scasca'
            if self.jac_type == 'matrix_matrix' or self.jac_type == 'matrixmatrix': self.jac_type = 'matmat'
            options = ['sca','mat','scasca','scamat','matmat']
            assert self.jac_type in options, "SAT: jac_type must be one of" + str(options)

            if 'coeff' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['coeff'], float), 'SAT: diss coeff must be a float, {0}'.format(solver.surf_diss['coeff'])
                self.coeff = solver.surf_diss['coeff']
            else:
                self.coeff = 1.0

            if 'entropy_fix' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['entropy_fix'], bool), 'SAT: entropy_fix must be a bool, {0}'.format(solver.surf_diss['entropy_fix'])
                self.entropy_fix = solver.surf_diss['entropy_fix']
            else:
                self.entropy_fix = True

            if 'average' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['average'], str) or (solver.surf_diss['average'] is None), 'SAT: average must be a str, not {0}'.format(solver.surf_diss['average'])
                if solver.surf_diss['average'] is None:
                    self.average = 'none'
                else:
                    self.average = solver.surf_diss['average'].lower()
            else:
                self.average = 'simple' # roe, simple, derigs
            options = ['simple','arithmetic','roe','derigs','ismailroe','none']
            assert self.average in options, "SAT: average must be one of" + str(options)

            if 'maxeig' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['maxeig'], str), 'SAT: maxeig must be a str, {0}'.format(solver.surf_diss['maxeig'])
                self.maxeig_type = solver.surf_diss['maxeig'].lower()
            else:
                self.maxeig_type = 'lf'
            options = ['lf','rusanov','lf2','rusanov2','none']
            assert self.maxeig_type in options, "SAT: maxeig must be one of" + str(options)

            if 'A_derigs' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['A_derigs'], bool), 'SAT: A_derigs must be a bool, {0}'.format(solver.surf_diss['A_derigs'])
                self.A_derigs = solver.surf_diss['A_derigs']
            else:
                self.A_derigs = False

            if 'P_derigs' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['P_derigs'], bool), 'SAT: P_derigs must be a bool, {0}'.format(solver.surf_diss['P_derigs'])
                self.P_derigs = solver.surf_diss['P_derigs']
            else:
                self.P_derigs = False
        else:
            # Default fluxes, no need for choices - set defaults.
            self.coeff = 1.
            self.average = 'simple'
            self.maxeig_type = 'lf'
            self.jac_type = 'sca'
            self.entropy_fix = False

        if self.sparse:
            self.lm_lv = sp.lm_lv
            self.lm_gv = sp.lm_gv
            self.lm_lm = sp.lm_lm
            self.gm_gv = sp.gm_gv
            self.lm_gm = sp.lm_gm
            self.lm_gdiag = sp.lm_gdiag
            self.gdiag_lm = sp.gdiag_lm
            self.lm_ldiag = sp.lm_ldiag
            self.gdiag_gm = sp.gdiag_gm
            self.prune_gm = sp.prune_gm
            if self.dim != 2: # not needed for 2D, removed this
                self.lm_gm_had_diff = sp.lm_gm_had_diff
                self.lmT_gm_had_diff = sp.lmT_gm_had_diff
                self.gm_gm_had_diff = sp.gm_gm_had_diff
                self.gmT_gm_had_diff = sp.gmT_gm_had_diff
                if self.solver_sparse:
                    self.lm_Fvol_had_diff = sp.lm_gm_had_diff
                    self.gm_Fvol_had_diff = sp.gm_gm_had_diff
                else:  
                    self.lm_Fvol_had_diff = sp.lm_dgm_had_diff
                    self.gm_Fvol_had_diff = sp.gm_dgm_had_diff
            #self.lm_dgm = sp.lm_dgm
        else:
            self.lm_lv = fn.lm_lv
            self.lm_gv = fn.lm_gv
            self.lm_lm = fn.lm_lm
            self.gm_gv = fn.gm_gv
            self.lm_gm = fn.lm_gm
            self.lm_gdiag = fn.lm_gdiag
            self.gdiag_lm = fn.gdiag_lm
            self.lm_ldiag = fn.lm_ldiag
            self.gdiag_gm = fn.gdiag_gm
            self.prune_gm = lambda gm: gm
            if self.dim != 2: # not needed for 2D, removed this
                self.lm_gm_had_diff = fn.lm_gm_had_diff
                self.lm_gmT_had_diff = staticmethod(lambda lm,gm: fn.lm_gm_had_diff(lm,np.transpose(gm,(1,0,2))))
                self.gm_gm_had_diff = fn.gm_gm_had_diff
                self.gm_gmT_had_diff = staticmethod(lambda gm,gm2: fn.gm_gm_had_diff(gm,np.transpose(gm2,(1,0,2))))
                self.lm_Fvol_had_diff = fn.lm_gm_had_diff
                self.gm_Fvol_had_diff = fn.gm_gm_had_diff
            #self.lm_dgm = fn.lm_gm
        
        if self.dim == 1:
            self.tL = solver.sbp.ta
            self.tR = solver.sbp.tb
            self.tLT = solver.sbp.taT
            self.tRT = solver.sbp.tbT
            self.calcEx = solver.diffeq.calcEx
            self.dExdq = solver.diffeq.dExdq
            #self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.dExdq_abs = solver.diffeq.dExdq_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            # self.metrics = fn.repeat_neq_gv(solver.mesh.metrics[:,0,:],self.neq_node)
            #self.bdy_metrics = np.repeat(np.reshape(solver.mesh.bdy_metrics, (1,2,self.nelem)),self.neq_node,0)
            #self.bdy_metrics_neq = np.repeat(self.bdy_metrics,self.neq_node,0)
            # NOTE: metrics and bdy_metrics = 1 for 1D
            self.calc_had_flux = solver.calc_had_flux

            if self.disc_type == 'had':
                if self.neq_node == 1:
                    if self.sparse:
                        self.build_F = staticmethod(lambda q1, q2: sp.build_F_sca(q1, q2, self.calc_had_flux, self.sparsity))
                    else:
                        self.build_F = staticmethod(lambda q1, q2: fn.build_F_sca(q1, q2, self.calc_had_flux))
                else:
                    if self.sparse:
                        self.build_F = staticmethod(lambda q1, q2: sp.build_F_sys(self.neq_node, q1, q2, self.calc_had_flux, 
                                                                                    self.sparsity_unkronned, self.sparsity))
                    else:
                        self.build_F = staticmethod(lambda q1, q2: fn.build_F_sys(self.neq_node, q1, q2, self.calc_had_flux))
                    
            ''' save useful matrices so as not to calculate on each loop '''
            self.Esurf = solver.sbp.Esurf
            self.ta = self.lm_lm(self.tL, self.tRT)
            self.tb = self.lm_lm(self.tR, self.tLT)
            # NOTE: metrics and bdy_metrics = 1 for 1D

            if self.sparse and (self.disc_type == 'had'):
                nrows = self.nen*self.neq_node
                self.taT = sp.lm_to_lmT(self.ta,nrows,nrows)
                taphysT_pad = [self.taT]*self.nelem
                taphysT_pad.append(sp.lm_to_lmT(self.tb,nrows,nrows))
                tbphys_pad = [self.tb]*self.nelem
                tbphys_pad.insert(0,self.ta)
                self.sparsity = sp.set_spgm_union_sparsity([taphysT_pad,tbphys_pad])
                if self.neq_node > 1:
                    self.sparsity_unkronned = sp.unkron_neq_sparsity(self.sparsity, self.neq_node)
            
        elif self.dim == 2:
            
            self.Hperp = solver.sbp.Hperp # should be flat and unkronned
            if self.direction == 'x': # computational direction, not physical direction
                self.tL = solver.sbp.txa
                self.tR = solver.sbp.txb
                self.tLT = solver.sbp.txaT
                self.tRT = solver.sbp.txbT
                self.tL_unkronned = solver.sbp.txa_unkronned
                self.tR_unkronned = solver.sbp.txb_unkronned
                self.tLT_unkronned = solver.sbp.txaT_unkronned
                self.tRT_unkronned = solver.sbp.txbT_unkronned
                self.Esurf = solver.sbp.Exsurf
                self.set_metrics_2d_x(solver.mesh.bdy_metrics)
            elif self.direction == 'y':
                self.tL = solver.sbp.tya
                self.tR = solver.sbp.tyb
                self.tLT = solver.sbp.tyaT
                self.tRT = solver.sbp.tybT
                self.tL_unkronned = solver.sbp.tya_unkronned
                self.tR_unkronned = solver.sbp.tyb_unkronned
                self.tLT_unkronned = solver.sbp.tyaT_unkronned
                self.tRT_unkronned = solver.sbp.tybT_unkronned
                self.Esurf = solver.sbp.Eysurf
                self.set_metrics_2d_y(solver.mesh.bdy_metrics)
            self.dExdq = solver.diffeq.dExdq
            self.dEydq = solver.diffeq.dEydq
            self.dEndq = solver.diffeq.dEndq
            #self.d2Exdq2 = solver.diffeq.d2Exdq2
            #self.d2Eydq2 = solver.diffeq.d2Eydq2
            self.dExdq_abs = solver.diffeq.dExdq_abs
            self.dEydq_abs = solver.diffeq.dEydq_abs
            self.dEndq_abs = solver.diffeq.dEndq_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
            self.maxeig_dEndq = solver.diffeq.maxeig_dEndq
            self.calc_had_flux = solver.calc_had_flux

            #if self.disc_type == 'had':
            #    if self.neq_node == 1:
            #        if self.sparse:
            #            self.build_F = staticmethod(lambda q1, q2: sp.build_F_sca_2d(q1, q2, self.calc_had_flux,
            #                                                                         self.xsparsity,self.ysparsity))
            #        else:
            #            self.build_F = staticmethod(lambda q1, q2: fn.build_F_sca_2d(q1, q2, self.calc_had_flux))
            #    else:
            #        if self.sparse:
            #            self.build_F = staticmethod(lambda q1, q2: sp.build_F_sys_2d(self.neq_node, q1, q2, self.calc_had_flux, 
            #                                                                        self.xsparsity_unkronned, self.xsparsity,
            #                                                                        self.ysparsity_unkronned, self.ysparsity))
            #        else:
            #            self.build_F = staticmethod(lambda q1, q2: fn.build_F_sys_2d(self.neq_node, q1, q2, self.calc_had_flux))

            ''' save useful matrices so as not to calculate on each loop '''

            # for volume terms, matrices to contract with x_phys and y_phys flux matrices
            #self.vol_x_mat = [self.lm_gdiag(self.Esurf,metrics[:,0,:]) for metrics in self.metrics]
            #self.vol_y_mat = [self.lm_gdiag(self.Esurf,metrics[:,1,:]) for metrics in self.metrics]
            # for surface terms, matrices to contract with x_phys and y_phys flux matrices on a and b facets
            self.tLHperp = self.lm_ldiag(self.tL, fn.repeat_neq_lv(self.Hperp, self.neq_node))
            self.tRHperp = self.lm_ldiag(self.tR, fn.repeat_neq_lv(self.Hperp, self.neq_node))
            if self.disc_type == 'div':
                self.taphysx = [self.prune_gm(self.lm_gm(self.tL, self.gdiag_lm(fn.repeat_neq_gv(self.Hperp[:,None] * bdy_metrics[:,0,0,:], self.neq_node), self.tRT))) for bdy_metrics in self.bdy_metrics]
                self.taphysy = [self.prune_gm(self.lm_gm(self.tL, self.gdiag_lm(fn.repeat_neq_gv(self.Hperp[:,None] * bdy_metrics[:,0,1,:], self.neq_node), self.tRT))) for bdy_metrics in self.bdy_metrics]
                self.tbphysx = [self.prune_gm(self.lm_gm(self.tR, self.gdiag_lm(fn.repeat_neq_gv(self.Hperp[:,None] * bdy_metrics[:,1,0,:], self.neq_node), self.tLT))) for bdy_metrics in self.bdy_metrics]
                self.tbphysy = [self.prune_gm(self.lm_gm(self.tR, self.gdiag_lm(fn.repeat_neq_gv(self.Hperp[:,None] * bdy_metrics[:,1,1,:], self.neq_node), self.tLT))) for bdy_metrics in self.bdy_metrics]
            elif self.disc_type == 'had':
                taphysx = [self.prune_gm(self.lm_gm(self.tL_unkronned, self.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,0,0,:]), self.tRT_unkronned))) for bdy_metrics in self.bdy_metrics]
                taphysy = [self.prune_gm(self.lm_gm(self.tL_unkronned, self.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,0,1,:]), self.tRT_unkronned))) for bdy_metrics in self.bdy_metrics]
                self.tbphysx = [self.prune_gm(self.lm_gm(self.tR_unkronned, self.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,1,0,:]), self.tLT_unkronned))) for bdy_metrics in self.bdy_metrics]
                self.tbphysy = [self.prune_gm(self.lm_gm(self.tR_unkronned, self.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,1,1,:]), self.tLT_unkronned))) for bdy_metrics in self.bdy_metrics]

                if self.sparse:
                    nrows = self.nen*self.nen
                    self.taphysxT = [sp.gm_to_gmT(gm_mat,nrows,nrows) for gm_mat in taphysx]
                    self.taphysyT = [sp.gm_to_gmT(gm_mat,nrows,nrows) for gm_mat in taphysy]
                    self.Fsat_diff_periodic = lambda q, idx: sp.Sat2d_had_Fsat_diff_periodic(self.taphysxT[idx],self.taphysyT[idx],
                                                                                            self.tbphysx[idx],self.tbphysy[idx],
                                                                                            q, self.calc_had_flux, self.neq_node)
                    
                else:
                    self.taphysx = taphysx
                    self.taphysy = taphysy
                    self.Fsat_diff_periodic = lambda q, idx: fn.Sat2d_had_Fsat_diff_periodic(self.taphysx[idx],self.taphysy[idx],
                                                                                            self.tbphysx[idx],self.tbphysy[idx],
                                                                                            q, self.calc_had_flux, self.neq_node)
                    
                    #len_taphysxT = len(self.taphysxT)
                    #taphysxT_pad = [list(self.taphysxT[i]) + [self.tbphysx[i][-1]] for i in range(len_taphysxT)]
                    #taphysyT_pad = [list(self.taphysyT[i]) + [self.tbphysy[i][-1]] for i in range(len_taphysxT)]
                    #tbphysx_pad = [[self.taphysxT[i][0]] + list(self.tbphysx[i]) for i in range(len_taphysxT)]
                    #tbphysy_pad = [[self.taphysyT[i][0]] + list(self.tbphysy[i]) for i in range(len_taphysxT)]
                    #self.xsparsity = sp.set_spgm_union_sparsity([*taphysxT_pad,*tbphysx_pad])
                    #self.ysparsity = sp.set_spgm_union_sparsity([*taphysyT_pad,*tbphysy_pad])
                    #if self.neq_node > 1:
                    #    self.xsparsity_unkronned = sp.unkron_neq_sparsity(self.xsparsity, self.neq_node)
                    #    self.ysparsity_unkronned = sp.unkron_neq_sparsity(self.ysparsity, self.neq_node)
        
        elif self.dim == 3:

            assert not self.sparse, '3D SATs assume dense matrices.'
            eye = np.eye(self.nen*self.neq_node)
            self.Hperp = solver.H_perp
            if self.direction == 'x': 
                self.tL = np.kron(np.kron(solver.tL, eye), eye)
                self.tR = np.kron(np.kron(solver.tR, eye), eye)
                self.set_metrics_3d_x(solver.mesh.metrics, solver.mesh.bdy_metrics)
            elif self.direction == 'y':
                self.tL = np.kron(np.kron(eye, solver.tL), eye)
                self.tR = np.kron(np.kron(eye, solver.tR), eye)
                self.set_metrics_3d_y(solver.mesh.metrics, solver.mesh.bdy_metrics)
            elif self.direction == 'z':
                self.tL = np.kron(eye, np.kron(eye, solver.tL))
                self.tR = np.kron(eye, np.kron(eye, solver.tR))
                self.set_metrics_3d_z(solver.mesh.metrics, solver.mesh.bdy_metrics)
            self.calc_had_flux = solver.calc_had_flux
            self.dExdq = solver.diffeq.dExdq
            self.dEydq = solver.diffeq.dEydq
            self.dEzdq = solver.diffeq.dEzdq
            self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.d2Eydq2 = solver.diffeq.d2Eydq2
            self.d2Ezdq2 = solver.diffeq.d2Ezdq2
            self.dExdq_abs = solver.diffeq.dExdq_abs
            self.dEydq_abs = solver.diffeq.dEydq_abs
            self.dEzdq_abs = solver.diffeq.dEzdq_abs
            self.dEndq_abs = solver.diffeq.dEndq_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
            self.maxeig_dEzdq = solver.diffeq.maxeig_dEzdq
            self.maxeig_dEndq = solver.diffeq.maxeig_dEndq

            if self.disc_type == 'had':
                if self.neq_node == 1:
                    if self.sparse:
                        self.build_F = staticmethod(lambda q1, q2: sp.build_F_sca_3d(q1, q2, self.calc_had_flux))
                    else:
                        self.build_F = staticmethod(lambda q1, q2: fn.build_F_sca_3d(q1, q2, self.calc_had_flux))
                else:
                    if self.sparse:
                        self.build_F = staticmethod(lambda q1, q2: sp.build_F_sys_3d(self.neq_node, q1, q2, self.calc_had_flux, 
                                                                                    self.xsparsity_unkronned, self.xsparsity,
                                                                                    self.ysparsity_unkronned, self.ysparsity,
                                                                                    self.zsparsity_unkronned, self.zsparsity))
                    else:
                        self.build_F = staticmethod(lambda q1, q2: fn.build_F_sys_3d(self.neq_node, q1, q2, self.calc_had_flux, 
                                                                                    self.xsparsity_unkronned, self.xsparsity,
                                                                                    self.ysparsity_unkronned, self.ysparsity,
                                                                                    self.zsparsity_unkronned, self.zsparsity))
            #TODO: pick out the necessary sparse matrices

            self.Esurf = self.tR @ np.diag(self.Hperp) @ self.tRT - self.tL @ np.diag(self.Hperp) @ self.tLT
            # for volume terms, matrices to contract with x_phys, y_phys, and z_phys flux matrices
            self.vol_x_mat = [fn.lm_gdiag(self.Esurf,metrics[:,0,:]) for metrics in self.metrics]
            self.vol_y_mat = [fn.lm_gdiag(self.Esurf,metrics[:,1,:]) for metrics in self.metrics]
            self.vol_z_mat = [fn.lm_gdiag(self.Esurf,metrics[:,2,:]) for metrics in self.metrics]
            # for surface terms, matrices to contract with x_phys, y_phys, and z_phys flux matrices on a and b facets
            self.taphysx = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,0,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.taphysy = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,1,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.taphysz = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,2,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysx = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,0,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysy = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,1,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysz = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,2,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.tLHperp = self.tL * self.Hperp
            self.tRHperp = self.tR * self.Hperp
            


        ''' Set the methods that will be used to calculate the SATs '''

        if solver.pde_order == 1:

            if self.neq_node == 1:
                self.calc_spec_rad = lambda gm: np.abs(fn.gm_to_gdiag(gm))
                self.repeat_neq_gv = lambda q: q
            else:
                self.calc_spec_rad = lambda gm: fn.spec_rad(gm, self.neq_node)
                self.repeat_neq_gv = lambda q: fn.repeat_neq_gv(q, self.neq_node)

            # set averaging method
            if self.average=='simple' or self.average=='arithmetic':
                self.calc_avgq = fn.arith_mean
            elif self.average=='roe':
                self.calc_avgq = solver.diffeq.roe_avg
            elif self.average=='ismailroe':
                self.calc_avgq = solver.diffeq.ismail_roe_avg
            elif self.average=='derigs':
                self.calc_avgq = solver.diffeq.derigs_avg
            elif self.average=='none':
                self.calc_avgq = None
            else:
                print(f"WARNING: desired average method '{self.average}' not recognized. Defaulting to 'simple'." )
                self.calc_avgq = fn.arith_mean 

            # set scalar abs(A) method
            if (self.jac_type == 'sca') or (self.jac_type == 'scasca') or (self.jac_type == 'scamat'):
                if self.maxeig_type == 'lf':
                    if self.dim == 1: self.calc_absA = self.calc_absA_lf_1d
                    else: self.calc_absA = self.calc_absA_lfn_nd
                elif self.maxeig_type == 'rusanov':
                    if self.dim == 1: self.calc_absA = self.calc_absA_rusanov_1d
                    else: self.calc_absA = self.calc_absA_rusanovn_nd
                elif self.maxeig_type == 'lf2':
                    if self.dim == 1: self.calc_absA = self.calc_absA_lf_1d
                    elif self.dim == 2: self.calc_absA = self.calc_absA_lf_2d
                    else: self.calc_absA = self.calc_absA_lf_3d
                elif self.maxeig_type == 'rusanov2':
                    if self.dim == 1: self.calc_absA = self.calc_absA_rusanov_1d
                    elif self.dim == 2: self.calc_absA = self.calc_absA_rusanov_2d
                    else: self.calc_absA = self.calc_absA_rusanov_3d
                else:
                    raise Exception('maxeig type not understood.', self.maxeig_type)
            else:
                self.maxeig_type = 'none'

            
            # set base dissipation method
            if self.diss_type == 'nd' or self.diss_type == 'symmetric' or (self.diss_type == 'ec' and self.disc_type == 'had'):
                if self.disc_type == 'div':
                    if self.print_progress: print('... Using the central SAT with no dissipation.')
                    # just absorb this into the base function? (no, b/c this is slightly faster)
                    if self.dim == 1:
                        self.calc = self.central_div_1d
                        self.calc_dfdq = self.central_div_1d_dfdq
                    elif self.dim == 2:
                        self.calc = self.central_div_2d
                    elif self.dim == 3:
                        self.calc = self.central_div_3d
                elif self.disc_type == 'had':
                    if self.print_progress: print(f'... Using the base Had SAT with {solver.had_flux} flux and no diss.')
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = lambda *x: 0
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = lambda *x: 0
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = lambda *x: 0
                    
            elif self.diss_type == 'upwind':
                print('WARNING: upwind SATs are not provably stable because of metric terms.')
                if self.disc_type == 'div':
                    if self.print_progress: print('... Using the base upwind SAT.')
                    if self.dim == 1:
                        self.calc = self.upwind_div_1d
                    elif self.dim == 2:
                        self.calc = self.upwind_div_2d
                    elif self.dim == 3:
                        self.calc = self.upwind_div_3d
                elif self.disc_type == 'had':
                    raise Exception("upwind dissipation not coded up for Hadamard.")
                    
            elif self.diss_type == 'lf' or self.diss_type == 'llf' or self.diss_type == 'lax_friedrichs':
                # sets a default LF dissipation. For more options, should select cons or ent

                self.maxeig_type = 'lf'
                if self.dim == 1:
                    self.calc_absA = self.calc_absA_lf_1d
                    self.calc_absA_dq = self.calc_absA_dq_sca_1D
                else:
                    self.calc_absA = self.calc_absA_lfn_nd
                    self.calc_absA_dq = self.calc_absA_dq_sca_nD

                if self.disc_type == 'div':
                    if self.print_progress: print('... Using the base cons SAT with sca lf diss on cons vars.')
                    if self.print_progress: print(f'... average={self.average}, maxeig=lf, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                    if self.dim == 1:
                        self.calc = self.base_div_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_div_2d
                        self.diss = self.diss_cons_nd
                    elif self.dim == 3:
                        self.calc = self.base_div_3d
                        self.diss = self.diss_cons_nd
                elif self.disc_type == 'had':
                    if self.print_progress: print(f'... Using the base Had SAT with {solver.had_flux} flux and sca lf diss on cons vars.')
                    if self.print_progress: print(f'... average={self.average}, maxeig=lf, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.diss_cons_nd
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.diss_cons_nd

            elif self.diss_type == 'cons' or self.diss_type == 'conservative':

                if self.disc_type == 'div':
                    str_base = '... Using the base cons SAT'
                    if self.dim == 1:
                        self.calc = self.base_div_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_div_2d
                        self.diss = self.diss_cons_nd
                    #elif self.dim == 3:
                    #    self.calc = self.base_div_3d
                elif self.disc_type == 'had':
                    str_base = f'... Using the base Had SAT with {solver.had_flux} flux'
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.diss_cons_nd
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.diss_cons_nd

                if self.jac_type == 'sca':
                    if self.print_progress: print(str_base + ' and sca diss on cons vars')
                    # self.calc_absA is already set
                    if self.dim == 1: self.calc_absA_dq = self.calc_absA_dq_sca_1D
                    else: self.calc_absA_dq = self.calc_absA_dq_sca_nD
                    if self.print_progress: print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}')
                elif self.jac_type == 'mat':
                    if self.print_progress: print(str_base + ' and mat diss on cons vars')
                    if self.dim == 1: 
                        self.calc_absA = self.calc_absA_matdiffeq_1d
                        self.calc_absA_dq = self.calc_absA_dq_mat_1D
                    else: 
                        self.calc_absA = self.calc_absA_matdiffeq_nd
                        self.calc_absA_dq = self.calc_absA_dq_mat_nD
                    if self.print_progress: print(f'... average={self.average}, entropy_fix={self.entropy_fix}, coeff={self.coeff}')
                else:
                    raise Exception("SAT: jac_type must be one of 'sca' or 'mat' when diss_type == 'cons'. Given:", self.jac_type)
                
                if self.diffeq_name=='Quasi1dEuler' or self.diffeq_name=='Euler2d':
                    if self.print_progress: print("SAT Reminder: you can bypass the default cons options using ")
                    if self.print_progress: print("              diss_type == 'diablo1', 'diablo2', 'diablo3', or 'diablo4'.")
                
            elif self.diss_type=='ent' or self.diss_type=='entropy':

                self.entropy_var = solver.diffeq.entropy_var
                
                if self.disc_type == 'div':
                    str_base = '... Using the base cons SAT'
                    if self.dim == 1:
                        self.calc = self.base_div_1d
                        self.diss = self.diss_ent_1d
                    elif self.dim == 2:
                        self.calc = self.base_div_2d
                        self.diss = self.diss_ent_nd
                    #elif self.dim == 3:
                    #    self.calc = self.base_div_3d
                elif self.disc_type == 'had':
                    str_base = f'... Using the base Had SAT with {solver.had_flux} flux'
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.diss_ent_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.diss_ent_nd
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.diss_ent_nd

                if self.jac_type == 'scasca':
                    if self.print_progress: print(str_base + ' and sca-sca diss on ent vars')
                    # self.calc_absA is already set
                    if self.dim == 1: self.calc_absAP_dw = self.calc_absAP_dw_scasca_1D
                    else: self.calc_absAP_dw = self.calc_absAP_dw_scasca_nD
                    if self.P_derigs:
                        self.calc_P = solver.diffeq.dqdw_derigs
                    else:
                        self.calc_P = self.calc_P_avg
                        self.diffeq_dqdw = solver.diffeq.dqdw
                    if self.print_progress: print(f'... average={self.average}, maxeig={self.maxeig_type}, P_derigs={self.P_derigs}, coeff={self.coeff}')
                elif self.jac_type == 'scamat':
                    if self.print_progress: print(str_base + ' and sca-mat diss on ent vars')
                    # self.calc_absA is already set
                    if self.dim == 1: self.calc_absAP_dw = self.calc_absAP_dw_scamat_1D
                    else: self.calc_absAP_dw = self.calc_absAP_dw_scamat_nD
                    if self.P_derigs:
                        self.calc_P = solver.diffeq.dqdw_derigs
                    else:
                        self.calc_P = self.calc_P_avg
                        self.diffeq_dqdw = solver.diffeq.dqdw
                    if self.print_progress: print(f'... average={self.average}, maxeig={self.maxeig_type}, P_derigs={self.P_derigs}, coeff={self.coeff}')
                elif self.jac_type == 'matmat':
                    if self.print_progress: print(str_base + ' and mat-mat diss on ent vars')
                    if self.dim == 1: 
                        self.calc_absAP_dw = self.calc_absAP_dw_matmat_1D
                        if self.P_derigs and self.A_derigs:
                            self.calc_absAP = lambda qL,qR: solver.diffeq.dExdw_abs_derigs(qL,qR,self.entropy_fix)
                            self.average = 'none'
                        elif self.P_derigs:
                            self.calc_P = solver.diffeq.dqdw_derigs
                            self.calc_absA = self.calc_absA_matdiffeq_1d
                            self.calc_absAP = self.calc_absAP_base_1d
                        elif self.A_derigs:
                            self.calc_P = self.calc_P_avg
                            self.diffeq_dqdw = solver.diffeq.dqdw
                            self.calc_absA = lambda qL,qR: solver.diffeq.dExdq_abs_derigs(qL,qR,self.entropy_fix)
                            self.calc_absAP = self.calc_absAP_base_1d
                        else:
                            try:
                                self.dExdw_abs = solver.diffeq.dExdw_abs
                                self.calc_absAP = self.calc_absAP_diffeq_1d
                            except:
                                print('SAT: dExdw_abs not found in diffeq. Using base dExdw_abs.')
                                self.calc_absAP = self.calc_absAP_base_1d
                                self.calc_absA = self.calc_absA_matdiffeq_1d
                                self.calc_P = self.calc_P_avg
                                self.diffeq_dqdw = solver.diffeq.dqdw
                    else: 
                        self.calc_absAP_dw = self.calc_absAP_dw_matmat_nD
                        if self.P_derigs and self.A_derigs:
                            self.calc_absAP = lambda qL,qR,mets: solver.diffeq.dEndw_abs_derigs(qL,qR,mets,self.entropy_fix)
                            self.average = 'none'
                        elif self.P_derigs:
                            self.calc_P = solver.diffeq.dqdw_derigs
                            self.calc_absA = self.calc_absA_matdiffeq_nd
                            self.calc_absAP = self.calc_absAP_base_nd
                        elif self.A_derigs:
                            self.calc_P = self.calc_P_avg
                            self.diffeq_dqdw = solver.diffeq.dqdw
                            self.calc_absA = lambda qL,qR,mets: solver.diffeq.dExdq_abs_derigs(qL,qR,mets,self.entropy_fix)
                            self.calc_absAP = self.calc_absAP_base_nd
                        else:
                            try:
                                self.dEndw_abs = solver.diffeq.dEndw_abs
                                self.calc_absAP = self.calc_absAP_diffeq_nd
                            except:
                                print('SAT: dEndw_abs not found in diffeq. Using base dEndw_abs.')
                                self.calc_absAP = self.calc_absAP_base_nd
                                self.calc_absA = self.calc_absA_matdiffeq_nd
                                self.calc_P = self.calc_P_avg
                                self.diffeq_dqdw = solver.diffeq.dqdw
                    if self.print_progress: print(f'... average={self.average}, entropy_fix={self.entropy_fix}, A_derigs={self.P_derigs}, P_derigs={self.P_derigs}, coeff={self.coeff}')

                else:
                    raise Exception("SAT: jac_type must be one of 'scasca', 'scamat' or 'matmat' when diss_type == 'ent'. Given:", self.jac_type)
            
            elif 'diablo' in self.diss_type:

                if self.disc_type == 'div':
                    str_base = '... Using the base cons SAT'
                    if self.dim == 1:
                        self.calc = self.base_div_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_div_2d
                        self.diss = self.diss_cons_nd
                    #elif self.dim == 3:
                    #    self.calc = self.base_div_3d
                elif self.disc_type == 'had':
                    str_base = f'... Using the base Had SAT with {solver.had_flux} flux'
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.diss_cons_nd
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.diss_cons_nd
                
                if self.diss_type=='diablo1':
                    if self.print_progress: print(str_base + ' and mat diss on cons vars (diablo 1)')
                    if self.print_progress: print(f'... average=roe, entropy_fix=hicken, coeff={self.coeff}')
                    if self.dim == 1: 
                        self.calc_absA_dq = lambda qL,qR : solver.diffeq.dExdq_abs_dq(qR,qL,1)
                    else: 
                        self.calc_absA_dq = lambda qL,qR,met : solver.diffeq.dEndq_abs_dq(met,qR,qL,1)
                elif self.diss_type=='diablo2':
                    if self.print_progress: print(str_base + ' and mat diss on cons vars (diablo 2)')
                    if self.print_progress: print(f'... average=roe, entropy_fix=diablo, coeff={self.coeff}')
                    if self.dim == 1: 
                        self.calc_absA_dq = lambda qL,qR : solver.diffeq.dExdq_abs_dq(qR,qL,2)
                    else: 
                        self.calc_absA_dq = lambda qL,qR,met : solver.diffeq.dEndq_abs_dq(met,qR,qL,2)
                elif self.diss_type=='diablo3':
                    if self.print_progress: print(str_base + ' and sca diss on cons vars (diablo 3)')
                    if self.print_progress: print(f'... average=roe, maxeig=lf, entropy_fix=hicken, coeff={self.coeff}')
                    if self.dim == 1: 
                        self.calc_absA_dq = lambda qL,qR : solver.diffeq.dExdq_abs_dq(qR,qL,3)
                    else: 
                        self.calc_absA_dq = lambda qL,qR,met : solver.diffeq.dEndq_abs_dq(met,qR,qL,3)
                elif self.diss_type=='diablo4':
                    if self.print_progress: print(str_base + ' and sca diss on cons vars (diablo 4)')
                    if self.print_progress: print(f'... average=roe, maxeig=lf, entropy_fix=False, coeff={self.coeff}')
                    if self.dim == 1: 
                        self.calc_absA_dq = lambda qL,qR : solver.diffeq.dExdq_abs_dq(qR,qL,4)
                    else: 
                        self.calc_absA_dq = lambda qL,qR,met : solver.diffeq.dEndq_abs_dq(met,qR,qL,4)
                else:
                    raise Exception("SAT: diablo type not understood. Must be one of 'diablo1', 'diablo2', 'diablo3', or 'diablo4'. Given:", self.diss_type)
        
                
            elif self.diffeq_name=='Burgers':
                if self.dim >= 2:
                    raise Exception('Burgers equation SATs only set up for 1D!')
                else:
                    if self.diss_type=='split':
                        self.alpha = solver.diffeq.split_alpha
                        if self.print_progress: print('... Using a split form SAT mimicking the variable coefficient advection formulation.')
                        if self.print_progress: print(f'... average={self.average}, maxeig=lf, coeff={self.coeff}, alpha={self.alpha}')
                        if self.print_progress: print('WARNING: The split form follows the Variable Coefficient formulation and is not entropy-stable.')
                        self.calc = lambda q,E,q_bdyL=None,q_bdyR=None: self.div_1d_burgers_split(q, E, q_bdyL=q_bdyL, q_bdyR=q_bdyR,
                                                                 extrapolate_flux=True) # TODO: Add some solver setting for this
                    elif self.diss_type=='ec':
                        if self.print_progress: print('... Using an entropy-conservative SAT found in the SBP book.')
                        if self.print_progress: print("    (not the one recovered from the Hadamard form. For this use diss_type='ec_had').")
                        self.coeff = 0.
                        self.calc = self.div_1d_burgers_es
                    elif self.diss_type=='es' or self.diss_type=='ent':
                        assert(self.jac_type=='sca' or self.jac_type=='scasca'), 'Burgers es SATs require sca / scasca jac_type.'
                        if self.print_progress: print('... Using an entropy-dissipative SAT found in the SBP book.')
                        if self.print_progress: print("    (not the one recovered from the Hadamard form. For this use diss_type='es_had').")
                        if self.print_progress: print(f'... average=simple, maxeig={self.maxeig_type}, coeff={self.coeff}')
                        self.calc = self.div_1d_burgers_es
                    elif self.diss_type=='ec_had':
                        if self.print_progress: print('... Using the entropy-conservative SAT recovered from the Hadamard form.')
                        self.coeff = 0.
                        self.calc = self.div_1d_burgers_had
                    elif self.diss_type=='es_had':
                        assert(self.jac_type=='sca' or self.jac_type=='scasca'), 'Burgers es_had SATs require sca / scasca jac_type.'
                        if self.print_progress: print('... Using the entropy-dissipative SAT recovered from the Hadamard form.')
                        if self.print_progress: print(f'... average=simple, maxeig={self.maxeig_type}, coeff={self.coeff}')
                        self.calc = self.div_1d_burgers_had
                    else:
                        raise Exception("SAT type not understood. Try 'ec', 'es', 'ec_had', 'es_had', 'split', or 'split_diss'.")


            else:
                raise Exception(f'Choice of SAT not understood: {self.diss_type}')

            # TODO: Use hasattribute to check for additional parameters?
            # Set the method for the sat and dfdq_sat for the first derivative
            #self.calc_sat_der1 = getattr(self, self.diffeq.sat_type_der1)
            #self.calc_dfdq_sat_der1 = getattr(self, self.diffeq.dfdq_sat_type_der1)

        else:
            raise Exception('SAT methods for reqested order of PDE is not available')
            
        
        ''' Check if using a generalized Hadamard form, then adjust as needed '''
        if (solver.settings['had_alpha'] != 1 or solver.settings['had_beta'] != 1 or solver.settings['had_gamma'] != 1):
            assert self.dim == 2, 'Generalized Hadamard form only set up for 2 dimensions'
            if self.direction == 'x': # computational direction, not physical direction
                self.set_exact_metrics_2d_x(solver.mesh.metrics_exa)
            elif self.direction == 'y':
                self.set_exact_metrics_2d_y(solver.mesh.metrics_exa)
            assert self.disc_type == 'had', 'Use had_alpha and had_beta only for Hadamard form'
            
            # overwrite base hadamard function for a general case
            self.had_alpha = solver.settings['had_alpha']
            self.had_beta = solver.settings['had_beta']
            self.had_gamma = solver.settings['had_gamma']
            self.calc = self.base_generalized_had_2d
            
            # create additional useful matrices
            self.vol_x_mat2 = [fn.lm_gdiag(self.Esurf,metrics[:,0,:]) for metrics in self.metrics_exa]
            self.vol_y_mat2 = [fn.lm_gdiag(self.Esurf,metrics[:,1,:]) for metrics in self.metrics_exa]
            self.vol_x_mat3 = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,0,:]), self.tRT)) - fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,0,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.vol_y_mat3 = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,1,:]), self.tRT)) - fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,1,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            metricsR = self.metrics_exa.copy()
            metricsL = self.metrics_exa.copy()
            metricsR.append(metricsR.pop(0))
            metricsL.insert(0, metricsL.pop(-1))
            self.taphysx2 = [fn.lm_gdiag((self.tL @ (self.Hperp * self.tRT)), metrics[:,0,:]) for metrics in metricsL]
            self.taphysy2 = [fn.lm_gdiag((self.tL @ (self.Hperp * self.tRT)), metrics[:,1,:]) for metrics in metricsL]
            self.tbphysx2 = [fn.lm_gdiag((self.tR @ (self.Hperp * self.tLT)), metrics[:,0,:]) for metrics in metricsR]
            self.tbphysy2 = [fn.lm_gdiag((self.tR @ (self.Hperp * self.tLT)), metrics[:,1,:]) for metrics in metricsR]
    
        ''' special cases '''
        
        if self.diffeq_name == 'VariableCoefficientLinearConvection':
            self.alpha = solver.diffeq.alpha # splitting parameter
            self.a = solver.diffeq.a # variable coefficient
            self.afun = solver.diffeq.afun
            self.bdy_x = solver.mesh.bdy_x
            assert(self.dim == 1),'Only set up for 1D so far'
            assert(self.disc_type == 'div'),'Not set up for Hadamard form yet'
            if self.diss_type == 'central' or self.diss_type == 'nondissipative' or self.diss_type == 'symmetric':
                self.calc = lambda q,E,q_bdyL=None,q_bdyR=None: self.llf_div_1d_varcoeff(q, E, sigma=0, q_bdyL=q_bdyL, q_bdyR=q_bdyR,
                                                                 extrapolate_flux=solver.diffeq.extrapolate_bdy_flux)
            elif self.diss_type == 'lf' or self.diss_type == 'llf' or self.diss_type == 'lax_friedrichs':
                self.calc = lambda q,E,q_bdyL=None,q_bdyR=None: self.llf_div_1d_varcoeff(q, E, sigma=1, q_bdyL=q_bdyL, q_bdyR=q_bdyR,
                                                                 extrapolate_flux=solver.diffeq.extrapolate_bdy_flux)
        
        
    ''' functions to set metrics in 2D and 3D '''

    def set_metrics_2d_x(self, bdy_metrics):
        ''' create a list of metrics for each row '''  
        #self.metrics = []
        self.bdy_metrics = []
        for row in range(self.nelem[1]):
            #self.metrics.append(np.repeat(metrics[:,:2,row::self.nelem[1]],self.neq_node,0)) # only want dx_ref/dx_phys and dx_ref/dy_phys
            self.bdy_metrics.append(bdy_metrics[:,:2,:2,row::self.nelem[1]]) # facets 1 and 2, same matrix entries

    def set_metrics_2d_y(self, bdy_metrics):
        ''' create a list of metrics for each col '''  
        #self.metrics = []
        self.bdy_metrics = []
        for col in range(self.nelem[0]):
            start = col*self.nelem[0]
            end = start + self.nelem[1]
            #self.metrics.append(np.repeat(metrics[:,2:,start:end],self.neq_node,0)) # only want dy_ref/dx_phys and dy_ref/dy_phys
            self.bdy_metrics.append(bdy_metrics[:,2:,2:,start:end]) # facets 3 and 4, same matrix entries
           
    def set_exact_metrics_2d_x(self, metrics):
        ''' create a list of metrics for each row '''  
        self.metrics_exa = []
        for row in range(self.nelem[1]):
            self.metrics_exa.append(np.repeat(metrics[:,:2,row::self.nelem[1]],self.neq_node,0)) # only want dx_ref/dx_phys and dx_ref/dy_phys

    def set_exact_metrics_2d_y(self, metrics):
        ''' create a list of metrics for each col '''  
        self.metrics_exa = []
        for col in range(self.nelem[0]):
            start = col*self.nelem[0]
            end = start + self.nelem[1]
            self.metrics_exa.append(np.repeat(metrics[:,2:,start:end],self.neq_node,0)) # only want dy_ref/dx_phys and dy_ref/dy_phys
     
    def set_metrics_3d_x(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        #self.metrics = []
        self.bdy_metrics = []
        skipx = self.nelem[1]*self.nelem[2]
        for row in range(skipx):
            #self.metrics.append(np.repeat(metrics[:,:3,row::skipx],self.neq_node,0))
            self.bdy_metrics.append(bdy_metrics[:,:2,:3,row::skipx])
    
    def set_metrics_3d_y(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        #self.metrics = []
        self.bdy_metrics = []
        for coly in range(self.nelem[0]*self.nelem[2]):
            start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
            end = start + self.nelem[1]*self.nelem[2]
            #self.metrics.append(np.repeat(metrics[:,3:6,start:end:self.nelem[2]],self.neq_node,0))
            self.bdy_metrics.append(bdy_metrics[:,2:4,3:6,start:end:self.nelem[2]])
    
    def set_metrics_3d_z(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        #self.metrics = []
        self.bdy_metrics = []
        for colz in range(self.nelem[0]*self.nelem[2]):
            start = colz*self.nelem[2]
            end = start + self.nelem[2]
            #self.metrics.append(np.repeat(metrics[:,6:,start:end],self.neq_node,0))
            self.bdy_metrics.append(bdy_metrics[:,4:,6:,start:end])



    ''' functions to calculate abs(A) and abs(A)*P '''

    def calc_absA_lf_1d(self, qL, qR):
        qavg = self.calc_avgq(qL, qR)
        Lambda = self.repeat_neq_gv(self.maxeig_dExdq(qavg))
        return Lambda
    
    def calc_absA_lf_2d(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        qavg = self.calc_avgq(qL, qR)
        maxeigsx = self.maxeig_dExdq(qavg)
        maxeigsy = self.maxeig_dEydq(qavg)
        Lambda = self.repeat_neq_gv(abs( maxeigsx * metrics[:,0,:] \
                                       + maxeigsy * metrics[:,1,:]) )
        return Lambda
    
    def calc_absA_lf_3d(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        qavg = self.calc_avgq(qL, qR)
        maxeigsx = self.maxeig_dExdq(qavg)
        maxeigsy = self.maxeig_dEydq(qavg)
        maxeigsz = self.maxeig_dEzdq(qavg)
        Lambda = self.repeat_neq_gv(abs( maxeigsx * metrics[:,0,:] \
                                       + maxeigsy * metrics[:,1,:] \
                                       + maxeigsz * metrics[:,2,:]) )
        return Lambda
    
    def calc_absA_lfn_nd(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        qavg = self.calc_avgq(qL, qR)
        Lambda = self.repeat_neq_gv(self.maxeig_dEndq(qavg, metrics))
        return Lambda
    
    def calc_absA_rusanov_1d(self, qL, qR):
        maxeigs = self.repeat_neq_gv(np.maximum(self.maxeig_dExdq(qL), self.maxeig_dExdq(qR)))
        return maxeigs
    
    def calc_absA_rusanov_2d(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        maxeigsx = np.maximum(self.maxeig_dExdq(qL), self.maxeig_dExdq(qR))
        maxeigsy = np.maximum(self.maxeig_dEydq(qL), self.maxeig_dEydq(qR))
        Lambda = self.repeat_neq_gv(abs( maxeigsx * metrics[:,0,:] \
                                       + maxeigsy * metrics[:,1,:]) )
        return Lambda
    
    def calc_absA_rusanov_3d(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        maxeigsx = np.maximum(self.maxeig_dExdq(qL), self.maxeig_dExdq(qR))
        maxeigsy = np.maximum(self.maxeig_dEydq(qL), self.maxeig_dEydq(qR))
        maxeigsz = np.maximum(self.maxeig_dEzdq(qL), self.maxeig_dEzdq(qR))
        Lambda = self.repeat_neq_gv(abs( maxeigsx * metrics[:,0,:] \
                                       + maxeigsy * metrics[:,1,:] \
                                       + maxeigsz * metrics[:,2,:]) )
        return Lambda
    
    def calc_absA_rusanovn_nd(self, qL, qR, metrics):
        # accepts metrics that are not repeated by neq, return repeated by neq
        maxeig = np.maximum(self.maxeig_dEndq(qL, metrics), self.maxeig_dEndq(qR, metrics))
        Lambda = self.repeat_neq_gv(maxeig)
        return Lambda
    
    def calc_absA_matdiffeq_1d(self, qL, qR):
        # calls the base method for the specific diff eq
        qavg = self.calc_avgq(qL, qR)
        absA = self.dExdq_abs(qavg, self.entropy_fix)
        return absA
    
    def calc_absA_matdiffeq_nd(self, qL, qR, metrics):
        # calls the base method for the specific diff eq
        qavg = self.calc_avgq(qL, qR)
        absA = self.dEndq_abs(qavg, metrics, self.entropy_fix)
        return absA
    
    def calc_P_avg(self, qL, qR):
        # calls the base method for the specific diff eq using q_avg
        qavg = self.calc_avgq(qL, qR)
        P = self.diffeq_dqdw(qavg)
        return P
    
    def calc_absAP_diffeq_1d(self, qL, qR):
        # calls the base method for the specific diff eq
        qavg = self.calc_avgq(qL, qR)
        absAP = self.dExdw_abs(qavg)
        return absAP
    
    def calc_absAP_diffeq_nd(self, qL, qR, metrics):
        # calls the base method for the specific diff eq
        qavg = self.calc_avgq(qL, qR)
        absAP = self.dEndw_abs(qavg, metrics)
        return absAP   
    
    def calc_absAP_base_1d(self, qL, qR):
        # calls the base methods for both absA and P
        absA = self.calc_absA(qL, qR)
        P = self.calc_P(qL, qR)
        absAP = fn.gbdiag_gbdiag(absA, P)
        return absAP
    
    def calc_absAP_base_nd(self, qL, qR, metrics):
        # calls the base method for both absA and P
        absA = self.calc_absA(qL, qR, metrics)
        P = self.calc_P(qL, qR)
        absAP = fn.gbdiag_gbdiag(absA, P)
        return absAP

    
    
    ''' A collection of base functions for calc_absA_dq and calc_absAP_dw '''

    def calc_absA_dq_sca_1D(self,qL,qR):
        ''' base method for self.jac_type == 'sca' in 1D '''
        Lambda = self.calc_absA(qL,qR)
        q_jump = qR - qL
        absA_dq = Lambda * q_jump # assumes Lambda is gdiag
        return absA_dq
    
    def calc_absA_dq_sca_nD(self,qL,qR,metrics):
        ''' base method for self.jac_type == 'sca' in 1D '''
        Lambda = self.calc_absA(qL,qR,metrics)
        q_jump = qR - qL
        absA_dq = Lambda * q_jump # assumes Lambda is gdiag
        return absA_dq
    
    def calc_absA_dq_mat_1D(self,qL,qR):
        ''' base method for self.jac_type == 'mat' in 1D '''
        absA = self.calc_absA(qL,qR)
        q_jump = qR - qL
        absA_dq = fn.gbdiag_gv(absA, q_jump)
        return absA_dq
    
    def calc_absA_dq_mat_nD(self,qL,qR,metrics):
        ''' base method for self.jac_type == 'mat' in nD '''
        absA = self.calc_absA(qL,qR,metrics)
        q_jump = qR - qL
        absA_dq = fn.gbdiag_gv(absA, q_jump)
        return absA_dq

    def calc_absAP_dw_scasca_1D(self,qL,qR):
        ''' base method for self.jac_type == 'scasca' in 1D '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        rhoP = self.repeat_neq_gv(self.calc_spec_rad(self.calc_P(qL,qR)))
        Lambda = self.calc_absA(qL,qR)
        absAP_dw = Lambda * rhoP * w_jump # assumes Lambda and rhoP are gdiag
        return absAP_dw
    
    def calc_absAP_dw_scasca_nD(self,qL,qR,metrics):
        ''' base method for self.jac_type == 'scasca' in 1D '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        rhoP = self.repeat_neq_gv(self.calc_spec_rad(self.calc_P(qL,qR)))
        Lambda = self.calc_absA(qL,qR,metrics)
        absAP_dw = Lambda * rhoP * w_jump # assumes Lambda and rhoP are gdiag
        return absAP_dw
    
    def calc_absAP_dw_scamat_1D(self,qL,qR):
        ''' base method for self.jac_type == 'scamat' in 1D '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        P = self.calc_P(qL,qR)
        Lambda = self.calc_absA(qL,qR)
        absAP_dw = Lambda * fn.gbdiag_gv(P, w_jump) # assumes Lambda is gdiag
        return absAP_dw
    
    def calc_absAP_dw_scamat_nD(self,qL,qR,metrics):
        ''' base method for self.jac_type == 'scamat' in nD '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        P = self.calc_P(qL,qR)
        Lambda = self.calc_absA(qL,qR,metrics)
        absAP_dw = Lambda * fn.gbdiag_gv(P, w_jump) # assumes Lambda is gdiag
        return absAP_dw
    
    def calc_absAP_dw_matmat_1D(self,qL,qR):
        ''' base method for self.jac_type == 'scamat' in 1D '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        absAP = self.calc_absAP(qL,qR)
        absAP_dw = fn.gbdiag_gv(absAP,w_jump)
        return absAP_dw
    
    def calc_absAP_dw_matmat_nD(self,qL,qR,metrics):
        ''' base method for self.jac_type == 'scamat' in nD '''
        w_jump = self.entropy_var(qR) - self.entropy_var(qL)
        absAP = self.calc_absAP(qL,qR,metrics)
        absAP_dw = fn.gbdiag_gv(absAP,w_jump)
        return absAP_dw

    ''' Define the 2-point flux dissipation functions. For a conservative base SAT + dissipation '''
    
    def diss_cons_1d(self,qL,qR):
        ''' dissipation in conservative variables '''
        absA_dq = self.calc_absA_dq(qL,qR)
        dissL = self.lm_gv(self.tL, absA_dq[:,:-1])
        dissR = self.lm_gv(self.tR, absA_dq[:,1:])
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_cons_nd(self,qL,qR,idx):
        ''' dissipation in conservative variables '''
        metrics = fn.pad_ndR(self.bdy_metrics[idx][:,0,:,:], self.bdy_metrics[idx][:,1,:,-1])
        absA_dq = self.calc_absA_dq(qL,qR,metrics)
        dissL = self.lm_gv(self.tLHperp, absA_dq[:,:-1])
        dissR = self.lm_gv(self.tRHperp, absA_dq[:,1:])
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_ent_1d(self,qL,qR):
        ''' dissipation in entropy variables '''
        absAP_dw = self.calc_absAP_dw(qL,qR)
        dissL = self.lm_gv(self.tL, absAP_dw[:,:-1])
        dissR = self.lm_gv(self.tR, absAP_dw[:,1:])
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_ent_nd(self,qL,qR,idx):
        ''' dissipation in entropy variables '''
        metrics = fn.pad_ndR(self.bdy_metrics[idx][:,0,:,:], self.bdy_metrics[idx][:,1,:,-1])
        absAP_dw = self.calc_absAP_dw(qL,qR,metrics)
        dissL = self.lm_gv(self.tLHperp, absAP_dw[:,:-1])
        dissR = self.lm_gv(self.tRHperp, absAP_dw[:,1:])
        
        diss = (dissL - dissR)/2
        return diss
