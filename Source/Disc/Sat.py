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
        
        self.method = solver.surf_type
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

        if self.neq_node == 1:
            self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sca(q1, q2, flux))
            self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sca(q1, q2, flux))
        else:
            self.build_F = staticmethod(lambda q1, q2, flux: fn.build_F_sys(self.neq_node, q1, q2, flux))
            self.build_F_vol = staticmethod(lambda q1, q2, flux: fn.build_F_vol_sys(self.neq_node, q1, q2, flux))
        
        if self.dim == 1:
            self.tL = solver.tL
            self.tR = solver.tR
            self.calcEx = solver.diffeq.calcEx
            self.dExdq = solver.diffeq.dExdq
            self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            #self.maxeig_dExdq_cmplx = solver.diffeq.maxeig_dExdq_cmplx
            self.metrics = fn.repeat_neq_gv(solver.mesh.metrics[:,0,:],self.neq_node)
            self.bdy_metrics = np.repeat(np.reshape(solver.mesh.bdy_metrics, (1,2,self.nelem)),self.neq_node,0)
            self.had_flux_Ex = solver.had_flux_Ex
            
        elif self.dim == 2:
            eye = np.eye(self.nen*self.neq_node)
            self.Hperp = solver.H_perp #TODO: Flatten this?
            if self.direction == 'x': # computational direction, not physical direction
                self.tL = np.kron(solver.tL, eye)
                self.tR = np.kron(solver.tR, eye)
                self.set_metrics_2d_x(solver.mesh.metrics, solver.mesh.bdy_metrics)
            elif self.direction == 'y':
                self.tL = np.kron(eye, solver.tL)
                self.tR = np.kron(eye, solver.tR)
                self.set_metrics_2d_y(solver.mesh.metrics, solver.mesh.bdy_metrics)
            self.had_flux_Ex = solver.had_flux_Ex
            self.had_flux_Ey = solver.had_flux_Ey
            self.dExdq = solver.diffeq.dExdq
            self.dEydq = solver.diffeq.dEydq
            self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.d2Eydq2 = solver.diffeq.d2Eydq2
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.dEydq_eig_abs = solver.diffeq.dEydq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
        
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
            self.had_flux_Ex = solver.had_flux_Ex
            self.had_flux_Ey = solver.had_flux_Ey
            self.had_flux_Ez = solver.had_flux_Ez
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
            
            
        ''' save useful matrices so as not to calculate on each loop '''

        self.tLT = self.tL.T
        self.tRT = self.tR.T
        if self.dim == 1:
            self.Esurf = self.tR @ self.tRT - self.tL @ self.tLT
            self.vol_mat = fn.lm_gdiag(self.Esurf,self.metrics)
            self.taphys = fn.lm_gm(self.tL, fn.gdiag_lm(self.bdy_metrics[:,0,:],self.tRT))
            self.tbphys = fn.lm_gm(self.tR, fn.gdiag_lm(self.bdy_metrics[:,1,:],self.tLT))
    
        elif self.dim == 2:
            self.Esurf = self.tR @ np.diag(self.Hperp[:,0]) @ self.tRT - self.tL @ np.diag(self.Hperp[:,0]) @ self.tLT
            # for volume terms, matrices to contract with x_phys and y_phys flux matrices
            self.vol_x_mat = [fn.lm_gdiag(self.Esurf,metrics[:,0,:]) for metrics in self.metrics]
            self.vol_y_mat = [fn.lm_gdiag(self.Esurf,metrics[:,1,:]) for metrics in self.metrics]
            # for surface terms, matrices to contract with x_phys and y_phys flux matrices on a and b facets
            self.taphysx = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,0,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.taphysy = [fn.lm_gm(self.tL, fn.gdiag_lm((self.Hperp * bdy_metrics[:,0,1,:]), self.tRT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysx = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,0,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
            self.tbphysy = [fn.lm_gm(self.tR, fn.gdiag_lm((self.Hperp * bdy_metrics[:,1,1,:]), self.tLT)) for bdy_metrics in self.bdy_metrics]
        
        elif self.dim == 3:
            self.Esurf = self.tR @ np.diag(self.Hperp[:,0]) @ self.tRT - self.tL @ np.diag(self.Hperp[:,0]) @ self.tLT
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
                
            if self.method == 'central' or self.method == 'nondissipative' or self.method == 'symmetric':
                if self.disc_type == 'div':
                    print('Using the base central SAT and no dissipation.')
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
                    
            elif self.method == 'upwind':
                print('WARNING: upwind SATs are not provably stable because of metric terms. Use for example lf instead.')
                if self.disc_type == 'div':
                    print('Using the base upwind SAT.')
                    if self.dim == 1:
                        self.calc = self.upwind_div_1d
                    elif self.dim == 2:
                        self.calc = self.upwind_div_2d
                    elif self.dim == 3:
                        self.calc = self.upwind_div_3d
                elif self.disc_type == 'had':
                    raise Exception("upwind dissipation not coded up yet for Hadamard. Try surf_type='lf' instead")
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.upwind_diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.upwind_diss_cons_2d
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.upwind_diss_cons_3d
                    
            elif self.method == 'lf' or self.method == 'llf' or self.method == 'lax_friedrichs':
                if self.disc_type == 'div':
                    print('Using the base LF SAT.')
                    if self.dim == 1:
                        self.calc = self.llf_div_1d
                        #self.calc_dfdq = self.llf_div_1d_dfdq
                    elif self.dim == 2:
                        self.calc = self.llf_div_2d
                    elif self.dim == 3:
                        self.calc = self.llf_div_3d
                elif self.disc_type == 'had':
                    print(f'Using the base Hadamard SAT with {solver.had_flux} flux and LF dissipation on conservative variables.')
                    if self.dim == 1:
                        self.calc = self.base_had_1d
                        self.diss = self.lf_diss_cons_1d
                    elif self.dim == 2:
                        self.calc = self.base_had_2d
                        self.diss = self.lf_diss_cons_2d
                    elif self.dim == 3:
                        self.calc = self.base_had_3d
                        self.diss = self.lf_diss_cons_3d
            
            elif self.diffeq_name=='Burgers':
                if self.dim >= 2:
                    raise Exception('Burgers equation SATs only set up for 1D!')
                else:
                    if self.disc_type == 'had':
                        assert (self.method.lower()=='ec'),"Only entropy-conservative SATs set up for Hadamard formulation. Try surf_type='ec'."
                        print(f'Using the base Hadamard SAT with {solver.had_flux} flux and no dissipation.')
                        self.calc = self.base_had_1d
                        self.diss = lambda *x: 0
                    elif self.disc_type == 'div':
                        print('WARNING: This is not set up yet for curvilinear transformations.')
                        if self.method.lower()=='split':
                            print('Using a split form SAT mimicking the variable coefficient advection formulation.')
                            print('WARNING: The split form follows the Variable Coefficient formulation and is not entropy-stable.')
                            self.alpha = solver.diffeq.split_alpha
                            self.calc = lambda q,E: self.div_1d_burgers_split(q, E, q_bdyL=None, q_bdyR=None, sigma=0., extrapolate_flux=True)
                        elif self.method.lower()=='split_diss':
                            print('Using a split form SAT mimicking the variable coefficient advection formulation.')
                            print('WARNING: The split form follows the Variable Coefficient formulation and is not entropy-stable.')
                            self.alpha = solver.diffeq.split_alpha
                            self.calc = lambda q,E: self.div_1d_burgers_split(q, E, q_bdyL=None, q_bdyR=None, sigma=1., extrapolate_flux=True)
                        elif self.method.lower()=='ec':
                            print('Using an entropy-conservative SAT found in the SBP book (not the one recovered from the Hadamard form).')
                            self.calc = lambda q,E: self.div_1d_burgers_es(q, E, q_bdyL=None, q_bdyR=None, sigma=0.)
                        elif self.method.lower()=='es' or self.method.lower()=='diss':
                            print('Using an entropy-dissipative SAT found in the SBP book (not the one recovered from the Hadamard form).')
                            self.calc = lambda q,E: self.div_1d_burgers_es(q, E, q_bdyL=None, q_bdyR=None, sigma=1.)
                        elif self.method.lower()=='ec_had':
                            print('Using the entropy-conservative SAT recovered from the Hadamard form.')
                            self.calc = lambda q,E: self.div_1d_burgers_had(q, E, q_bdyL=None, q_bdyR=None, sigma=0.)
                        elif self.method.lower()=='es_had':
                            print('Using the entropy-dissipative SAT recovered from the Hadamard form.')
                            self.calc = lambda q,E: self.div_1d_burgers_had(q, E, q_bdyL=None, q_bdyR=None, sigma=1.)
                        else:
                            raise Exception("SAT type not understood. Try 'ec', 'es', 'ec_had', 'es_had', 'split', or 'split_diss'.")

            elif self.diffeq_name=='Quasi1dEuler':
                if self.dim >= 2:
                    raise Exception('Burgers equation SATs only set up for 1D!')
                else:
                    if self.disc_type == 'had':
                        if self.method.lower()=='ec':
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and no dissipation.')
                            self.calc = self.base_had_1d
                            self.diss = lambda *x: 0
                        elif self.method.lower()=='lf_cons3':
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and diablo LF3 on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.base_had_1d
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,3)
                        elif self.method.lower()=='lf_cons4':
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and diablo LF4 on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.base_had_1d
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,4)
                        elif self.method.lower()=='roe_cons1':
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and diablo Roe1 (Hicken fix) on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.base_had_1d
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,1)
                        elif self.method.lower()=='roe_cons2' or self.method.lower()=='roe':
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and diablo Roe2 (entropy fix) on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.base_had_1d
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D(qL,qR,2)
                        elif self.method.lower()=='es':
                            self.dEndw_abs = solver.diffeq.dEndw_abs
                            self.entropy_var = solver.diffeq.entropy_var
                            self.calc = self.base_had_1d
                            self.diss = self.lf_diss_ent_matmat_1d
                            print(f'Using the base Hadamard SAT with {solver.had_flux} flux and matrix-matrix dissipation on entropy variables.')
                        else:
                            raise Exception("SAT type not understood. Try 'ec', 'es', 'lf', 'lf_cons3', 'lf_cons4', 'roe_cons1', 'roe_cons2'.")
                        
                    else:
                        if self.method.lower()=='lf_cons3':
                            print(f'Using the base conservative SAT and diablo LF3 on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.central_div_1d_base
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D_2(qL,qR,3)
                        elif self.method.lower()=='lf_cons4':
                            print(f'Using the base conservative SAT and diablo LF4 on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.central_div_1d_base
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D_2(qL,qR,4)
                        elif self.method.lower()=='roe_cons1':
                            print(f'Using the base conservative SAT and diablo Roe1 (Hicken fix) on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.central_div_1d_base
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D_2(qL,qR,1)
                        elif self.method.lower()=='roe_cons2' or self.method.lower()=='roe':
                            print(f'Using the base conservative SAT and diablo Roe2 (entropy fix) on conservative variables.')
                            self.dEndq_eig_abs_dq = solver.diffeq.dEndq_eig_abs_dq
                            self.calc = self.central_div_1d_base
                            self.diss = lambda qL,qR: self.diss_dEndq_eig_abs_dq_1D_2(qL,qR,2)
                        else:
                            raise Exception("SAT type not understood. Try 'lf', 'lf_cons3', 'lf_cons4', 'roe_cons1', 'roe_cons2'.")

                    

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

    def diss_dEndq_eig_abs_dq_1D(self,qL,qR,flux_type):
        ''' calculates dissipation calculates -0.5*abs(A)@(q-qg) using diffeq's dEndq_eig_abs_dq function '''
        qLf = self.tRT @ qL
        qRf = self.tLT @ qR
        # remember that metric have been repeated neq times, unecessary here
        metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])
        A_q_jump = self.dEndq_eig_abs_dq(metrics, qRf, qLf, flux_type)
        dissL = self.tL @ A_q_jump[:,:-1]
        dissR = self.tR @ A_q_jump[:,1:]

        diss = dissR - dissL # diablo function outputs 2 * negative of the convention used here
        return diss
    
    def diss_dEndq_eig_abs_dq_1D_2(self,qLf,qRf,flux_type):
        ''' calculates dissipation calculates -0.5*abs(A)@(q-qg) using diffeq's dEndq_eig_abs_dq function '''
        # remember that metric have been repeated neq times, unecessary here
        metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])
        A_q_jump = self.dEndq_eig_abs_dq(metrics, qRf, qLf, flux_type)
        dissL = self.tL @ A_q_jump[:,:-1]
        dissR = self.tR @ A_q_jump[:,1:]

        diss = dissR - dissL # diablo function outputs 2 * negative of the convention used here
        return diss

    def lf_diss_cons_1d(self,qL,qR,avg='simple'):
        ''' lax-friedrichs dissipation in conservative variables '''
        qLf = self.tRT @ qL
        qRf = self.tLT @ qR
        
        q_jump = qRf - qLf
        
        if avg=='simple': # Alternatively, a Roe average can be used
            q_avg = (qLf + qRf)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        maxeigs = self.maxeig_dExdq(q_avg)
        metrics = fn.pad_1dR(self.bdy_metrics[:,0,:], self.bdy_metrics[:,1,-1])
        Lambda = np.abs(maxeigs * metrics)
        Lambda_q_jump = fn.gdiag_gv(Lambda, q_jump)
        dissL = self.tL @ Lambda_q_jump[:,:-1]
        dissR = self.tR @ Lambda_q_jump[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
    
    def lf_diss_ent_matmat_1d(self,qL,qR,avg='simple'):
        ''' lax-friedrichs dissipation in entropy variables, matrix-matrix form '''
        qLf = self.tRT @ qL
        qRf = self.tLT @ qR
        wLf = self.entropy_var(qLf)
        wRf = self.entropy_var(qRf)
        
        w_jump = wRf - wLf
        
        if avg=='simple': # Alternatively, a Roe average can be used
            q_avg = (qLf + qRf)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        # remember that metric have been repeated neq times, unecessary here
        metrics = fn.pad_1dR(self.bdy_metrics[::self.neq_node,0,:], self.bdy_metrics[::self.neq_node,1,-1])
        A = self.dEndw_abs(q_avg,metrics)
        A_w_jump = fn.gm_gv(A, w_jump)
        dissL = self.tL @ A_w_jump[:,:-1]
        dissR = self.tR @ A_w_jump[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
        
    def lf_diss_cons_2d(self,qL,qR,idx,avg='simple'):
        ''' lax-friedrichs dissipation in conservative variables '''
        qLf = self.tRT @ qL
        qRf = self.tLT @ qR
        if avg=='simple': # Alternatively, a Roe average can be used
            q_avg = (qLf + qRf)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        q_jump = qRf - qLf
        
        maxeigsx = self.maxeig_dExdq(q_avg)
        maxeigsy = self.maxeig_dEydq(q_avg)
        metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
        metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
        H_Lambda = self.Hperp * np.abs(maxeigsx * metricsx + maxeigsy * metricsy)
        Lambda_q_jump = fn.gdiag_gv(H_Lambda, q_jump)
        dissL = self.tL @ Lambda_q_jump[:,:-1]
        dissR = self.tR @ Lambda_q_jump[:,1:]
        
        diss = (dissL - dissR)/2
        return diss
    
    def lf_diss_cons_3d(self,qL,qR,idx,avg='simple'):
        ''' lax-friedrichs dissipation in conservative variables '''
        qLf = self.tRT @ qL
        qRf = self.tLT @ qR
        if avg=='simple': # Alternatively, a Roe average can be used
            q_avg = (qLf + qRf)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        q_jump = qRf - qLf
        
        maxeigsx = self.maxeig_dExdq(q_avg)
        maxeigsy = self.maxeig_dEydq(q_avg)
        maxeigsz = self.maxeig_dEzdq(q_avg)
        metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
        metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
        metricsz = fn.pad_1dR(self.bdy_metrics[idx][:,0,2,:], self.bdy_metrics[idx][:,1,2,-1])
        H_Lambda = self.Hperp * np.abs(maxeigsx * metricsx + maxeigsy * metricsy + maxeigsz * metricsz)
        Lambda_q_jump = fn.gdiag_gv(H_Lambda, q_jump)
        dissL = self.tL @ Lambda_q_jump[:,:-1]
        dissR = self.tR @ Lambda_q_jump[:,1:]
        
        diss = (dissL - dissR)/2
        return diss