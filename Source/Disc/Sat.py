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
        
        print('... Setting up SATs')
        
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
        self.sparsity = None
        self.sparsity_unkronned = None

        if self.diss_type not in ['nd','conservative','symmetric']:
            if 'jac_type' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['jac_type'], str), 'SAT: jac_type must be a str, {0}'.format(solver.surf_diss['jac_type'])
                self.jac_type = solver.surf_diss['jac_type'].lower()
            else:
                self.jac_type = ''
            if self.jac_type == 'scalar': self.jac_type = 'sca'
            if self.jac_type == 'matrix': self.jac_type = 'mat'
            if self.jac_type == 'scalar_matrix' or self.jac_type == 'scalarmatrix': self.jac_type = 'scamat'
            if self.jac_type == 'scalar_scalar' or self.jac_type == 'scalarscalar': self.jac_type = 'scasca'
            if self.jac_type == 'matrix_matrix' or self.jac_type == 'matrixmatrix': self.jac_type = 'matmat'
            assert self.jac_type in ['sca','mat','scasca','scamat','matmat',''], "SAT: jac_type must be either 'sca','mat','scasca','scamat','matmat' or ''"

            if 'coeff' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['coeff'], float), 'SAT: diss coeff must be a float, {0}'.format(solver.surf_diss['coeff'])
                self.coeff = solver.surf_diss['coeff']
            else:
                self.coeff = 1.0

            if 'entropy_fix' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['entropy_fix'], bool) or \
                        isinstance(solver.surf_diss['entropy_fix'], str), 'SAT: entropy_fix must be a bool or str, {0}'.format(solver.surf_diss['entropy_fix'])
                self.entropy_fix = solver.surf_diss['entropy_fix']
                if isinstance(solver.surf_diss['entropy_fix'], str): 
                    self.entropy_fix = self.entropy_fix.lower()
                if self.entropy_fix == 'none' or self.entropy_fix == 'false':
                    self.entropy_fix = False
            else:
                self.entropy_fix = True

            if 'average' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['average'], str), 'SAT: average must be a str, {0}'.format(solver.surf_diss['average'])
                self.average = solver.surf_diss['average'].lower()
            else:
                self.average = 'simple' # roe, simple, derigs
            assert self.average in ['simple','roe','derigs'], 'SAT: average must be either "simple" or "roe" or "derigs"'

            if 'maxeig' in solver.surf_diss.keys():
                assert isinstance(solver.surf_diss['maxeig'], str), 'SAT: maxeig must be a str, {0}'.format(solver.surf_diss['maxeig'])
                self.maxeig_type = solver.surf_diss['maxeig'].lower()
            else:
                self.maxeig_type = 'qavg' # lf, rusanov
            assert self.maxeig_type in ['lf','rusanov','qavg'], 'SAT: maxeig must be either "lf" or "rusanov"'
        else:
            self.coeff = 0.
        
        if self.dim == 1:
            self.tL = solver.tL
            self.tR = solver.tR
            self.calcEx = solver.diffeq.calcEx
            self.dExdq = solver.diffeq.dExdq
            self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEndq = solver.diffeq.maxeig_dEndq
            self.metrics = fn.repeat_neq_gv(solver.mesh.metrics[:,0,:],self.neq_node)
            self.bdy_metrics = np.repeat(np.reshape(solver.mesh.bdy_metrics, (1,2,self.nelem)),self.neq_node,0)
            self.calc_had_flux = solver.calc_had_flux

            if self.neq_node == 1:
                self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sca(q1, q2, flux))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sca(q1, q2, flux))
            else:
                self.build_F = staticmethod(lambda q1, q2, flux: sp.build_F_sys(self.neq_node, q1, q2, flux, 
                                                                                self.sparsity_unkronned, self.sparsity))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sys(q1, q2, flux))
            
        elif self.dim == 2:
            tL = solver.tL[::self.neq_node,::self.neq_node]
            tR = solver.tR[::self.neq_node,::self.neq_node]
            self.Hperp = solver.H_perp # should be flat
            if self.direction == 'x': # computational direction, not physical direction
                self.tL = np.kron(np.kron(tL, np.eye(self.nen)), np.eye(self.neq_node))
                self.tR = np.kron(np.kron(tR, np.eye(self.nen)), np.eye(self.neq_node))
                self.set_metrics_2d_x(solver.mesh.metrics, solver.mesh.bdy_metrics)
            elif self.direction == 'y':
                self.tL = np.kron(np.kron(np.eye(self.nen), tL), np.eye(self.neq_node))
                self.tR = np.kron(np.kron(np.eye(self.nen), tR), np.eye(self.neq_node))
                self.set_metrics_2d_y(solver.mesh.metrics, solver.mesh.bdy_metrics)
            self.calc_had_flux = solver.calc_had_flux
            self.dExdq = solver.diffeq.dExdq
            self.dEydq = solver.diffeq.dEydq
            #self.d2Exdq2 = solver.diffeq.d2Exdq2
            #self.d2Eydq2 = solver.diffeq.d2Eydq2
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.dEydq_eig_abs = solver.diffeq.dEydq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
            self.maxeig_dEndq = solver.diffeq.maxeig_dEndq

            if self.neq_node == 1:
                self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sca_2d(q1, q2, flux))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sca_2d(q1, q2, flux))
            else:
                self.build_F = staticmethod(lambda q1, q2, flux: sp.build_F_sys_2d(self.neq_node, q1, q2, flux, 
                                                                                self.xsparsity_unkronned, self.xsparsity,
                                                                                self.ysparsity_unkronned, self.ysparsity))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sys_2d(q1, q2, flux))
        
        elif self.dim == 3:
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
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.dEydq_eig_abs = solver.diffeq.dEydq_eig_abs
            self.dEzdq_eig_abs = solver.diffeq.dEzdq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
            self.maxeig_dEzdq = solver.diffeq.maxeig_dEzdq
            self.maxeig_dEndq = solver.diffeq.maxeig_dEndq

            if self.neq_node == 1:
                self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sca(q1, q2, flux))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sca(q1, q2, flux))
            else:
                self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sys(self.neq_node, q1, q2, flux))
                #self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sys(q1, q2, flux))
            
            
        ''' save useful matrices so as not to calculate on each loop '''

        self.tLT = self.tL.T
        self.tRT = self.tR.T
        if self.dim == 1:
            self.Esurf = self.tR @ self.tRT - self.tL @ self.tLT
            self.vol_mat = fn.lm_gdiag(self.Esurf,self.metrics)
            self.taphys = fn.lm_gm(self.tL, fn.gdiag_lm(self.bdy_metrics[:,0,:],self.tRT))
            self.tbphys = fn.lm_gm(self.tR, fn.gdiag_lm(self.bdy_metrics[:,1,:],self.tLT))

            if (self.disc_type == 'had') and (self.neq_node > 1):
                self.vol_mat_sp = sp.gm_to_sp(self.vol_mat)
                taphysT = np.ascontiguousarray(np.transpose(self.taphys,(1,0,2)))
                self.taphysT_sp = sp.gm_to_sp(taphysT)
                self.tbphys_sp = sp.gm_to_sp(self.tbphys)

                taphysT_pad = fn.pad_gm_1dR(taphysT,self.tbphys[:,:,-1])
                tbphys_pad = fn.pad_gm_1dL(self.tbphys,taphysT[:,:,0])
                self.sparsity = sp.set_gm_sparsity([taphysT_pad,tbphys_pad])
                self.sparsity_unkronned = sp.set_gm_sparsity([fn.unkron_neq_gm(taphysT_pad,self.neq_node),
                                                            fn.unkron_neq_gm(tbphys_pad,self.neq_node)])
    
        elif self.dim == 2:
            self.Esurf = self.tR @ np.diag(self.Hperp) @ self.tRT - self.tL @ np.diag(self.Hperp) @ self.tLT
            # for volume terms, matrices to contract with x_phys and y_phys flux matrices
            self.vol_x_mat = [fn.lm_gdiag(self.Esurf,metrics[:,0,:]) for metrics in self.metrics]
            self.vol_y_mat = [fn.lm_gdiag(self.Esurf,metrics[:,1,:]) for metrics in self.metrics]
            # for surface terms, matrices to contract with x_phys and y_phys flux matrices on a and b facets
            self.taphysx = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,0,0,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.taphysy = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,0,1,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysx = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,1,0,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysy = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp[:,None] * bdy_metrics[:,1,1,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]

            if self.neq_node > 1:
                self.vol_x_mat_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.vol_x_mat]   
                self.vol_y_mat_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.vol_y_mat]
                if (self.disc_type == 'had'):
                    taphysxT = [np.ascontiguousarray(np.transpose(gm_mat,(1,0,2))) for gm_mat in self.taphysx]
                    taphysyT = [np.ascontiguousarray(np.transpose(gm_mat,(1,0,2))) for gm_mat in self.taphysy]
                    self.taphysxT_sp = [sp.gm_to_sp(gm_mat) for gm_mat in taphysxT]
                    self.taphysyT_sp = [sp.gm_to_sp(gm_mat) for gm_mat in taphysyT]
                else:
                    self.taphysx_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.taphysx]
                    self.taphysy_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.taphysy]
                self.tbphysx_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.tbphysx]
                self.tbphysy_sp = [sp.gm_to_sp(gm_mat) for gm_mat in self.tbphysy]  

                if (self.disc_type == 'had'):
                    taphysxT_pad = [fn.pad_gm_1dR(taphysxT[i],self.tbphysx[i][:,:,-1]) for i in range(len(taphysxT))]
                    taphysyT_pad = [fn.pad_gm_1dR(taphysyT[i],self.tbphysy[i][:,:,-1]) for i in range(len(taphysxT))]
                    tbphysx_pad = [fn.pad_gm_1dL(self.tbphysx[i],taphysxT[i][:,:,0]) for i in range(len(taphysxT))]
                    tbphysy_pad = [fn.pad_gm_1dL(self.tbphysy[i],taphysyT[i][:,:,0]) for i in range(len(taphysyT))]
                    self.xsparsity = sp.set_gm_sparsity([*taphysxT_pad,*tbphysx_pad])
                    self.xsparsity_unkronned = sp.set_gm_sparsity([*[fn.unkron_neq_gm(mat,self.neq_node) for mat in taphysxT_pad],
                                                                *[fn.unkron_neq_gm(mat,self.neq_node) for mat in tbphysx_pad]]) 
                    self.ysparsity = sp.set_gm_sparsity([*taphysyT_pad,*tbphysy_pad])
                    self.ysparsity_unkronned = sp.set_gm_sparsity([*[fn.unkron_neq_gm(mat,self.neq_node) for mat in taphysyT_pad],
                                                                *[fn.unkron_neq_gm(mat,self.neq_node) for mat in tbphysy_pad]])          
        
        elif self.dim == 3:
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
            


        ''' Set the methods that will be used to calculate the SATs '''

        if solver.pde_order == 1:          
                
            if self.diss_type == 'nd' or self.diss_type == 'conservative' or self.diss_type == 'symmetric':
                if self.disc_type == 'div':
                    print('Using the central SAT with no dissipation.')
                    #TODO: just absorb this into the base function? (no, this is slightly faster)
                    if self.dim == 1:
                        self.calc = self.central_div_1d
                        self.calc_dfdq = self.central_div_1d_dfdq
                    elif self.dim == 2:
                        self.calc = self.central_div_2d
                    elif self.dim == 3:
                        self.calc = self.central_div_3d
                elif self.disc_type == 'had':
                    print(f'Using the base Hadamard SAT with {solver.had_flux} flux and no dissipation.')
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
                    print('Using the base upwind SAT.')
                    if self.dim == 1:
                        self.calc = self.upwind_div_1d
                    elif self.dim == 2:
                        self.calc = self.upwind_div_2d
                    elif self.dim == 3:
                        self.calc = self.upwind_div_3d
                elif self.disc_type == 'had':
                    raise Exception("upwind dissipation not coded up for Hadamard.")
                    
            elif self.diss_type == 'lf' or self.diss_type == 'llf' or self.diss_type == 'lax_friedrichs':
                if self.disc_type == 'div':
                    print('Using the default scalar SAT on conservative variables.')
                    if self.maxeig_type == 'rusanov': self.average = 'None'
                    print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix=False')
                    if self.dim == 1:
                        self.calc = self.llf_div_1d
                        #self.calc_dfdq = self.llf_div_1d_dfdq
                    elif self.dim == 2:
                        self.calc = self.llf_div_2d
                    elif self.dim == 3:
                        self.calc = self.llf_div_3d
                elif self.disc_type == 'had':
                    print(f'Using the base Hadamard SAT with {solver.had_flux} flux and scalar dissipation on conservative variables.')
                    if self.maxeig_type == 'rusanov': self.average = 'None'
                    print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix=False')
                    if self.jac_type != 'sca':
                            print(f'WARNING: overwriting jac_type={self.jac_type} to jac_type=sca.')
                            self.jac_type = 'sca'
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.diss_cons_2d
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.lf_diss_cons_3d
            
            elif self.diffeq_name=='Burgers':
                if self.dim >= 2:
                    raise Exception('Burgers equation SATs only set up for 1D!')
                else:
                    #TODO: This needs to be cleaned up now that I have changed the base functions
                    if self.disc_type == 'had':
                        assert (self.diss_type=='ec'),"Only entropy-conservative SATs set up for Hadamard formulation. Try surf_type='ec'."
                        print(f'Using the base Hadamard SAT with {solver.had_flux} flux and no dissipation.')
                        self.calc = self.base_had_1d
                        self.diss = lambda *x: 0
                    elif self.disc_type == 'div':
                        print('WARNING: This is not set up yet for curvilinear transformations.')
                        if self.diss_type=='split':
                            print('Using a split form SAT mimicking the variable coefficient advection formulation.')
                            print('WARNING: The split form follows the Variable Coefficient formulation and is not entropy-stable.')
                            self.alpha = solver.diffeq.split_alpha
                            self.calc = lambda q,E: self.div_1d_burgers_split(q, E, q_bdyL=None, q_bdyR=None, sigma=0., extrapolate_flux=True)
                        elif self.diss_type=='split_diss':
                            print('Using a split form SAT mimicking the variable coefficient advection formulation.')
                            print('WARNING: The split form follows the Variable Coefficient formulation and is not entropy-stable.')
                            self.alpha = solver.diffeq.split_alpha
                            self.calc = lambda q,E: self.div_1d_burgers_split(q, E, q_bdyL=None, q_bdyR=None, sigma=1., extrapolate_flux=True)
                        elif self.diss_type=='ec':
                            print('Using an entropy-conservative SAT found in the SBP book (not the one recovered from the Hadamard form).')
                            self.calc = lambda q,E: self.div_1d_burgers_es(q, E, q_bdyL=None, q_bdyR=None, sigma=0.)
                        elif self.diss_type=='es' or self.diss_type=='diss':
                            print('Using an entropy-dissipative SAT found in the SBP book (not the one recovered from the Hadamard form).')
                            self.calc = lambda q,E: self.div_1d_burgers_es(q, E, q_bdyL=None, q_bdyR=None, sigma=1.)
                        elif self.diss_type=='ec_had':
                            print('Using the entropy-conservative SAT recovered from the Hadamard form.')
                            self.calc = lambda q,E: self.div_1d_burgers_had(q, E, q_bdyL=None, q_bdyR=None, sigma=0.)
                        elif self.diss_type=='es_had':
                            print('Using the entropy-dissipative SAT recovered from the Hadamard form.')
                            self.calc = lambda q,E: self.div_1d_burgers_had(q, E, q_bdyL=None, q_bdyR=None, sigma=1.)
                        else:
                            raise Exception("SAT type not understood. Try 'ec', 'es', 'ec_had', 'es_had', 'split', or 'split_diss'.")

            elif self.diffeq_name=='Quasi1dEuler' or self.diffeq_name=='Euler2d':

                if self.disc_type == 'had':
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                    str_base = f'Using the base Hadamard SAT with {solver.had_flux} flux'
                elif self.disc_type == 'div':
                    if self.dim == 1:
                        self.calc = self.base_div_1d
                    elif self.dim == 2:
                        self.calc = self.base_div_2d
                    str_base = 'Using the base conservative SAT'

                if self.dim >= 3:
                    raise Exception('SATs only set up for 1D and 2D!')

                if self.entropy_fix == True:
                    self.entropy_fix=='hicken'
                
                if self.diss_type=='nd' or self.diss_type=='ec' or self.diss_type=='symmetric':
                    print(str_base + ' and no dissipation.')
                    self.diss = lambda *x: 0

                elif self.diss_type=='cons' or self.diss_type=='conservative':
                    if self.jac_type == 'sca':
                        if self.average == 'roe' and self.entropy_fix=='hicken' and self.maxeig_type=='qavg':
                            # equivalent to choice 'diablo3'
                            print(str_base + ' and scalar dissipation on conservative variables (diablo LF3)')
                            print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                            print("TODO: although I'm not convinced this is actually scalar...")
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            if self.dim == 1:
                                self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,3)
                            else:
                                self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,3)
                        elif self.average == 'roe' and self.entropy_fix==False and self.maxeig_type=='qavg':
                            # equivalent to choice 'diablo4'
                            print(str_base + ' and scalar dissipation on conservative variables (diablo LF4)')
                            print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            if self.dim == 1:
                                self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,4)
                            else:
                                self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,4)
                        elif self.entropy_fix==False and \
                            (self.average == 'simple' and self.maxeig_type=='qavg') or \
                            self.maxeig_type=='lf' or self.maxeig_type=='rusanov':
                            # use default scalar dissipation
                            print(str_base + ' and scalar dissipation on conservative variables')
                            if self.maxeig_type == 'rusanov': self.average = 'None'
                            print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix=False')
                            if self.dim == 1:
                                self.diss = self.diss_cons_1d
                            else:
                                self.diss = self.diss_cons_2d
                        else:
                            raise Exception('The following SAT combination is not currently possible:')
                            print(f'diss_type=cons, jac_type=sca, average={self.average}, maxeig={self.maxeig_type}, entropy_fix={self.entropy_fix}')

                    elif self.jac_type == 'mat':
                        if self.average == 'roe' and self.entropy_fix=='hicken':
                            # equivalent to choice 'diablo1'
                            print(str_base + ' and matrix dissipation on conservative variables (diablo LF1)')
                            print(f'average={self.average}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            if self.dim == 1:
                                self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,1)
                            else:
                                self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,1)
                        elif self.average == 'roe' and self.entropy_fix=='diablo':
                            # equivalent to choice 'diablo2'
                            print(str_base + ' and matrix dissipation on conservative variables (diablo LF2)')
                            print(f'average={self.average}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            if self.dim == 1:
                                self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,2)
                            else:
                                self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,2)
                        else:
                            raise Exception('The following SAT combination is not currently possible:')
                            print(f'diss_type=cons, jac_type=mat, average={self.average}, entropy_fix={self.entropy_fix}')
                    
                    else:
                        raise Exception('Must have jac_type=sca or jac_type=mat on conservative variables.')

                elif self.diss_type=='ent' or self.diss_type=='entropy':
                    if self.jac_type == 'scasca':
                        raise Exception('Scalar-scalar dissipation on entropy variables is not coded up yet.')
                    
                    elif self.jac_type == 'scamat':
                        print(str_base + ' and scalar-matrix dissipation on entropy variables.')
                        print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                        self.dqdw_jump = solver.diffeq.dqdw_jump
                        self.entropy_var = solver.diffeq.entropy_var
                        if self.dim == 1:
                            self.diss = self.diss_ent_1d
                        elif self.dim == 2:
                            self.diss = self.diss_ent_2d

                    elif self.jac_type == 'matmat':
                        print(str_base + ' and matrix-matrix dissipation on entropy variables.')
                        print(f'average={self.average}, maxeig={self.maxeig_type}, coeff={self.coeff}, entropy_fix={self.entropy_fix}')
                        self.dqdw_jump = solver.diffeq.dqdw_jump
                        self.entropy_var = solver.diffeq.entropy_var
                        if self.dim == 1:
                            self.diss = self.diss_ent_1d
                        elif self.dim == 2:
                            self.diss = self.diss_ent_2d

                    else:
                        raise Exception('Must have jac_type = scasca, scamat, or matmat on conservative variables.')

       
                elif self.diss_type=='diablo1':
                    print(str_base + ' and matrix dissipation on conservative variables (diablo LF1)')
                    print(f'average=roe, coeff={self.coeff}, entropy_fix=hicken')
                    self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                    if self.dim == 1:
                        self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,1)
                    else:
                        self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,1)
                elif self.diss_type=='diablo2':
                    print(str_base + ' and matrix dissipation on conservative variables (diablo LF2)')
                    print(f'average=roe, coeff={self.coeff}, entropy_fix=diablo')
                    self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                    if self.dim == 1:
                        self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,2)
                    else:
                        self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,2)
                elif self.diss_type=='diablo3':
                    print(str_base + ' and scalar dissipation on conservative variables (diablo LF3)')
                    print(f'average=roe, maxeig=qavg, coeff={self.coeff}, entropy_fix=hicken')
                    print("TODO: although I'm not convinced this is actually scalar...")
                    self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                    if self.dim == 1:
                        self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,3)
                    else:
                        self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,3)
                elif self.diss_type=='diablo4':
                    print(str_base + ' and scalar dissipation on conservative variables (diablo LF4)')
                    print(f'average=roe, maxeig=qavg, coeff={self.coeff}, entropy_fix=False')
                    self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                    if self.dim == 1:
                        self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,4)
                    else:
                        self.diss = lambda qL,qR, idx: self.diss_dEndq_eig_abs_dq_nD(qL,qR,idx,4)
                
                else:
                    raise Exception("SAT type not understood. Try 'ec', 'symmetric', 'es', 'roe', 'lf', 'lf_ent', 'lf_cons3', 'lf_cons4', 'roe_cons1', 'roe_cons2'.")

            # TODO: default cons / ent type

            else:
                raise Exception('Choice of SAT not understood.')

            # TODO: Use hasattribute to check for additional parameters?
            # Set the method for the sat and dfdq_sat for the first derivative
            #self.calc_sat_der1 = getattr(self, self.diffeq.sat_type_der1)
            #self.calc_dfdq_sat_der1 = getattr(self, self.diffeq.dfdq_sat_type_der1)

        elif self.pde_order == 2:
            
            # TODO

            # Set the method for the sat and dfdq_sat for the various derivatives
            self.sat_der1 = getattr(self, self.diffeq.sat_type_der1)
            self.sat_der2 = getattr(self, self.diffeq.sat_type_der2)

            self.dfdq_sat_der1 = getattr(self, self.diffeq.dfdq_sat_type_der1)
            self.dfdq_sat_der2 = getattr(self, self.diffeq.dfdq_sat_type_der2)

            self.calc_sat = self.sat_der2_master
            self.calc_dfdq_sat = self.dfdq_sat_der2_master

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
            if self.method == 'central' or self.method == 'nondissipative' or self.method == 'symmetric':
                self.calc = lambda q,E,q_bdyL=None,q_bdyR=None: self.llf_div_1d_varcoeff(q, E, sigma=0, q_bdyL=q_bdyL, q_bdyR=q_bdyR,
                                                                 extrapolate_flux=solver.diffeq.extrapolate_bdy_flux)
            elif self.method == 'lf' or self.method == 'llf' or self.method == 'lax_friedrichs':
                self.calc = lambda q,E,q_bdyL=None,q_bdyR=None: self.llf_div_1d_varcoeff(q, E, sigma=1, q_bdyL=q_bdyL, q_bdyR=q_bdyR,
                                                                 extrapolate_flux=solver.diffeq.extrapolate_bdy_flux)
        
        
    def set_metrics_2d_x(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        self.metrics = []
        self.bdy_metrics = []
        for row in range(self.nelem[1]):
            self.metrics.append(np.repeat(metrics[:,:2,row::self.nelem[1]],self.neq_node,0)) # only want dx_ref/dx_phys and dx_ref/dy_phys
            self.bdy_metrics.append(np.repeat(bdy_metrics[:,:2,:2,row::self.nelem[1]],self.neq_node,0)) # facets 1 and 2, same matrix entries

    def set_metrics_2d_y(self, metrics, bdy_metrics):
        ''' create a list of metrics for each col '''  
        self.metrics = []
        self.bdy_metrics = []
        for col in range(self.nelem[0]):
            start = col*self.nelem[0]
            end = start + self.nelem[1]
            self.metrics.append(np.repeat(metrics[:,2:,start:end],self.neq_node,0)) # only want dy_ref/dx_phys and dy_ref/dy_phys
            self.bdy_metrics.append(np.repeat(bdy_metrics[:,2:,2:,start:end],self.neq_node,0)) # facets 3 and 4, same matrix entries
           
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
        self.metrics = []
        self.bdy_metrics = []
        skipx = self.nelem[1]*self.nelem[2]
        for row in range(skipx):
            self.metrics.append(np.repeat(metrics[:,:3,row::skipx],self.neq_node,0))
            self.bdy_metrics.append(np.repeat(bdy_metrics[:,:2,:3,row::skipx],self.neq_node,0))
    
    def set_metrics_3d_y(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        self.metrics = []
        self.bdy_metrics = []
        for coly in range(self.nelem[0]*self.nelem[2]):
            start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
            end = start + self.nelem[1]*self.nelem[2]
            self.metrics.append(np.repeat(metrics[:,3:6,start:end:self.nelem[2]],self.neq_node,0))
            self.bdy_metrics.append(np.repeat(bdy_metrics[:,2:4,3:6,start:end:self.nelem[2]],self.neq_node,0))
    
    def set_metrics_3d_z(self, metrics, bdy_metrics):
        ''' create a list of metrics for each row '''  
        self.metrics = []
        self.bdy_metrics = []
        for colz in range(self.nelem[0]*self.nelem[2]):
            start = colz*self.nelem[2]
            end = start + self.nelem[2]
            self.metrics.append(np.repeat(metrics[:,6:,start:end],self.neq_node,0))
            self.bdy_metrics.append(np.repeat(bdy_metrics[:,4:,6:,start:end],self.neq_node,0))

    


    ''' Define the 2-point flux dissipation functions. For a conservative flux + dissipation '''
    
    def diss_dEndq_eig_abs_dq_1D(self,qLf,qRf,flux_type):
        ''' calculates dissipation calculates -0.5*abs(A)@(q-qg) using diffeq's dEndq_eig_abs_dq function 
        accepts projected qL and qR, already padded '''
        # remember that metric have been repeated neq times, unecessary here
        metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])
        A_q_jump = self.dEndq_eig_abs_dq(metrics, qRf, qLf, flux_type)
        dissL = self.tL @ A_q_jump[:,:-1]
        dissR = self.tR @ A_q_jump[:,1:]

        diss = dissR - dissL # diablo function outputs 2 * negative of the convention used here
        return diss
    
    def diss_dEndq_eig_abs_dq_nD(self,qLf,qRf,idx,flux_type):
        ''' calculates dissipation calculates -0.5*abs(A)@(q-qg) using diffeq's dEndq_eig_abs_dq function 
        accepts projected qL and qR, already padded '''
        # remember that metric have been repeated neq times, unecessary here
        metrics = fn.pad_ndR(self.bdy_metrics[idx][::self.neq_node,0,:,:], self.bdy_metrics[idx][::self.neq_node,1,:,-1])
        A_q_jump = self.dEndq_eig_abs_dq(metrics, qRf, qLf, flux_type)
        dissL = (self.tL * self.Hperp) @ A_q_jump[:,:-1]
        dissR = (self.tR * self.Hperp) @ A_q_jump[:,1:]

        diss = dissR - dissL # diablo function outputs 2 * negative of the convention used here
        return diss

    def diss_cons_1d(self,qLf,qRf):
        ''' dissipation in conservative variables '''
        q_jump = qRf - qLf
        
        if self.jac_type == 'sca':
            if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                if self.average=='simple' or self.maxeig_type == 'lf':
                    qf_avg = (qLf + qRf)/2
                else:
                    raise Exception('Averaging method not understood.', self.average)
                maxeigs = self.maxeig_dExdq(qf_avg)
            elif self.maxeig_type == 'rusanov':
                maxeigs = np.maximum(self.maxeig_dExdq(qLf), self.maxeig_dExdq(qRf))
            else:
                raise Exception('maxeig type not understood.', self.maxeig_type)
            
            metrics = fn.pad_1dR(self.bdy_metrics[:,0,:], self.bdy_metrics[:,1,-1])
            Lambda = np.abs(maxeigs * metrics)
            absA_dq = fn.gdiag_gv(Lambda, q_jump)

        elif self.jac_type == 'mat':
            raise Exception('Not set up for matrix interface dissipation yet.')
        else:
            raise Exception('jac type not understood.', self.jac_type)
        
        dissL = self.tL @ absA_dq[:,:-1]
        dissR = self.tR @ absA_dq[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_ent_1d(self,qL,qR):
        ''' dissipation in entropy variables '''
        wLf = self.entropy_var(qL)
        wRf = self.entropy_var(qR)
        
        w_jump = wRf - wLf

        if self.jac_type == 'scamat':
            metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])

            if self.average == 'derigs':
                # averaging from Derigs et al 2017, scalar-matrix dissipation
                P = self.dqdw_jump(qL,qR)
                if self.maxeig_type == 'lf':
                    qavg = (qL + qR)/2
                    if self.entropy_fix == False:
                        lam = self.maxeig_dEndq(qavg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'rusanov':
                    if self.entropy_fix == False:
                        lam = np.maximum(self.maxeig_dEndq(qL,metrics), self.maxeig_dEndq(qR,metrics))
                    else:
                        #TODO: add row entropy fix or diablo or hicken
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'qavg':
                    raise Exception('maxeig=qavg, average=derigs not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            elif self.average == 'simple':
                qavg = (qL + qR)/2
                P = self.dqdw(qavg)
                if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                    if self.entropy_fix == False:
                        lam = self.maxeig_dEndq(qavg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'rusanov':
                    if self.entropy_fix == False:
                        lam = np.maximum(self.maxeig_dEndq(qL,metrics), self.maxeig_dEndq(qR,metrics))
                    else:
                        #TODO: add row entropy fix or diablo or hicken
                        raise Exception('Entropy fix not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            else: 
                raise Exception('average not understood.', self.average)

            absAP_dw = fn.gdiag_gv(fn.repeat_neq_gv(lam,self.neq_node), fn.gm_gv(P, w_jump))


        elif self.jac_type == 'matmat':
            metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])

            if self.average == 'simple':
                # simple arithmetic average, matrix-matrix dissipation (LF)
                qavg = (qL + qR)/2
                if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                    if self.entropy_fix == False:
                        absAP = self.dEndw_abs(q_avg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            else:
                raise Exception('average not understood.', self.average)

            absAP_dw = fn.gm_gv(absAP, w_jump)

        elif self.jac_type == 'scasca':
            raise Exception('Not set up for scalar-scalar interface dissipation yet.')
        else:
            raise Exception('jac type not understood.', self.jac_type)
        
        dissL = self.tL @ absAP_dw[:,:-1]
        dissR = self.tR @ absAP_dw[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
        
    def diss_cons_2d(self,qLf,qRf,idx):
        ''' dissipation in conservative variables '''
        q_jump = qRf - qLf

        if self.jac_type == 'sca':
            if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                if self.average=='simple' or self.maxeig_type == 'lf':
                    qf_avg = (qLf + qRf)/2
                else:
                    raise Exception('Averaging method not understood.', self.average)
                maxeigsx = self.maxeig_dExdq(q_avg)
                maxeigsy = self.maxeig_dEydq(q_avg)
            elif self.maxeig_type == 'rusanov':
                maxeigsx = np.maximum(self.maxeig_dExdq(qLf), self.maxeig_dExdq(qRf))
                maxeigsy = np.maximum(self.maxeig_dEydq(qLf), self.maxeig_dEydq(qRf))
            else:
                raise Exception('maxeig type not understood.', self.maxeig_type)
            
            metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
            metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
            H_Lambda = self.Hperp * np.abs(maxeigsx * metricsx + maxeigsy * metricsy)
            absA_dq = fn.gdiag_gv(H_Lambda, q_jump)

        elif self.jac_type == 'mat':
            raise Exception('Not set up for matrix interface dissipation yet.')
        else:
            raise Exception('jac type not understood.', self.jac_type)
        
        dissL = self.tL @ absA_dq
        dissR = self.tR @ absA_dq
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_ent_2d(self,qL,qR,idx):
        ''' dissipation in entropy variables '''
        wLf = self.entropy_var(qL)
        wRf = self.entropy_var(qR)
        
        w_jump = wRf - wLf

        if self.jac_type == 'scamat':
            # remember that metric have been repeated neq times, unecessary here
            metrics = fn.pad_ndR(self.bdy_metrics[idx][::self.neq_node,0,:,:], self.bdy_metrics[idx][::self.neq_node,1,:,-1])

            if self.average == 'derigs':
                # averaging from Derigs et al 2017, scalar-matrix dissipation
                P = self.dqdw_jump(qL,qR)

                if self.maxeig_type == 'lf':
                    qavg = (qL + qR)/2
                    if self.entropy_fix == False:
                        lam = self.maxeig_dEndq(qavg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'rusanov':
                    if self.entropy_fix == False:
                        lam = np.maximum(self.maxeig_dEndq(qL,metrics), self.maxeig_dEndq(qR,metrics))
                    else:
                        #TODO: add row entropy fix or diablo or hicken
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'qavg':
                    raise Exception('maxeig=qavg, average=derigs not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            elif self.average == 'simple':
                qavg = (qL + qR)/2
                P = self.dqdw(qavg)
                if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                    if self.entropy_fix == False:
                        lam = self.maxeig_dEndq(qavg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                elif self.maxeig_type == 'rusanov':
                    if self.entropy_fix == False:
                        lam = np.maximum(self.maxeig_dEndq(qL,metrics), self.maxeig_dEndq(qR,metrics))
                    else:
                        #TODO: add row entropy fix or diablo or hicken
                        raise Exception('Entropy fix not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            else: 
                raise Exception('average not understood.', self.average)

            absAP_dw = fn.gdiag_gv(fn.repeat_neq_gv(lam,self.neq_node), fn.gm_gv(P, w_jump))


        elif self.jac_type == 'matmat':
            # remember that metric have been repeated neq times, unecessary here
            metrics = fn.pad_ndR(self.bdy_metrics[idx][::self.neq_node,0,:,:], self.bdy_metrics[idx][::self.neq_node,1,:,-1])

            if self.average == 'simple':
                # simple arithmetic average, matrix-matrix dissipation (LF)
                qavg = (qL + qR)/2
                if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                    if self.entropy_fix == False:
                        absAP = self.dEndw_abs(q_avg,metrics)
                    else:
                        raise Exception('Entropy fix not set up yet.')
                else:
                    raise Exception('maxeig not understood.', self.maxeig_type)
            else:
                raise Exception('average not understood.', self.average)

            absAP_dw = fn.gm_gv(absAP, w_jump)

        elif self.jac_type == 'scasca':
            raise Exception('Not set up for scalar-scalar interface dissipation yet.')
        else:
            raise Exception('jac type not understood.', self.jac_type)
    
        dissL = self.tL @ absAP_dw[:,:-1]
        dissR = self.tR @ absAP_dw[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
    
    def diss_cons_3d(self,qLf,qRf,idx,flux_type=1):
        ''' dissipation in conservative variables '''
        q_jump = qRf - qLf

        if self.jac_type == 'sca':
            if self.maxeig_type == 'lf' or self.maxeig_type == 'qavg':
                if self.average=='simple' or self.maxeig_type == 'lf':
                    qf_avg = (qLf + qRf)/2
                else:
                    raise Exception('Averaging method not understood.', self.average)
                maxeigsx = self.maxeig_dExdq(q_avg)
                maxeigsy = self.maxeig_dEydq(q_avg)
                maxeigsz = self.maxeig_dEzdq(q_avg)
            elif self.maxeig_type == 'rusanov':
                maxeigsx = np.maximum(self.maxeig_dExdq(qLf), self.maxeig_dExdq(qRf))
                maxeigsy = np.maximum(self.maxeig_dEydq(qLf), self.maxeig_dEydq(qRf))
                maxeigsz = np.maximum(self.maxeig_dEzdq(qLf), self.maxeig_dEzdq(qRf))
            else:
                raise Exception('maxeig type not understood.', self.maxeig_type)
            
            metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
            metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
            metricsz = fn.pad_1dR(self.bdy_metrics[idx][:,0,2,:], self.bdy_metrics[idx][:,1,2,-1])
            H_Lambda = self.Hperp * np.abs(maxeigsx * metricsx + maxeigsy * metricsy + maxeigsz * metricsz)
            absA_dq = fn.gdiag_gv(H_Lambda, q_jump)

        elif self.jac_type == 'mat':
            raise Exception('Not set up for matrix interface dissipation yet.')
        else:
            raise Exception('jac type not understood.', self.jac_type)

        dissL = self.tL @ absA_dq[:,:-1]
        dissR = self.tR @ absA_dq[:,1:]
        
        diss = (dissL - dissR)/2
        return diss