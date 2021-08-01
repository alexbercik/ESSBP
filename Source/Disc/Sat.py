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


class Sat(SatDer1, SatDer2):

    # TODO: Finish the comments here
    '''
    The methods calc_sat_unstruc,
    '''

    def __init__(self, solver, direction):
        '''
        Sets up the SAT for a particular direction used in solver 

        Parameters
        ----------
        solver : class instance
            The solver class which contains all the important functions.
        direction : str
            the direction in computational space for SAT (xi).
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
        eye = np.eye(self.nen*self.neq_node)
        
        if self.dim == 1:
            self.tL = solver.tL
            self.tR = solver.tR
            self.dEdq = solver.diffeq.dEdq
            self.d2Edq2 = solver.diffeq.d2Edq2
            self.dEdq_eig_abs = solver.diffeq.dEdq_eig_abs
            self.maxeig_dEdq = solver.diffeq.maxeig_dEdq
            self.metrics = solver.mesh.metrics[:,0,:]
            self.bdy_metrics = np.reshape(solver.mesh.bdy_metrics, (1,2,self.nelem))
            
        elif self.dim == 2:
            self.Hperp = solver.H_perp #TODO: Flatten this
            if direction == 'x': # computational direction, not physical direction
                self.tL = np.kron(solver.tL, eye)
                self.tR = np.kron(solver.tR, eye)
                self.metrics = solver.mesh.metrics[:,:2,:] # only want dx_ref/dx_phys and dx_ref/dy_phys
                self.bdy_metrics = solver.mesh.bdy_metrics[:,:2,:2,:] # facets 1 and 2, same matrix entries
                self.set_metrics = self.set_metrics_2d_x
            elif direction == 'y':
                self.tL = np.kron(eye, solver.tL)
                self.tR = np.kron(eye, solver.tR)
                self.metrics = solver.mesh.metrics[:,2:,:] # only want dy_ref/dx_phys and dy_ref/dy_phys
                self.bdy_metrics = solver.mesh.bdy_metrics[:,2:,2:,:] # facets 1 and 2, same matrix entries
                self.set_metrics = self.set_metrics_2d_y
            self.dExdq = solver.diffeq.dExdq
            self.dEydq = solver.diffeq.dEydq
            self.d2Exdq2 = solver.diffeq.d2Exdq2
            self.d2Eydq2 = solver.diffeq.d2Eydq2
            self.dExdq_eig_abs = solver.diffeq.dExdq_eig_abs
            self.dEydq_eig_abs = solver.diffeq.dEydq_eig_abs
            self.maxeig_dExdq = solver.diffeq.maxeig_dExdq
            self.maxeig_dEydq = solver.diffeq.maxeig_dEydq
        
        elif self.dim == 3:
            self.Hperp = solver.H_perp
            if direction == 'x': 
                self.tL = np.kron(np.kron(solver.tL, eye), eye)
                self.tR = np.kron(np.kron(solver.tR, eye), eye)
                self.metrics = solver.mesh.metrics[:,:3,:]
                self.bdy_metrics = solver.mesh.bdy_metrics[:,:2,:3,:]
                self.set_metrics = self.set_metrics_3d_x
            elif direction == 'y':
                self.tL = np.kron(np.kron(eye, solver.tL), eye)
                self.tR = np.kron(np.kron(eye, solver.tR), eye)
                self.metrics = solver.mesh.metrics[:,3:6,:]
                self.bdy_metrics = solver.mesh.bdy_metrics[:,2:4,3:6,:]
                self.set_metrics = self.set_metrics_3d_y
            elif direction == 'z':
                self.tL = np.kron(eye, np.kron(eye, solver.tL))
                self.tR = np.kron(eye, np.kron(eye, solver.tR))
                self.metrics = solver.mesh.metrics[:,6:,:]
                self.bdy_metrics = solver.mesh.bdy_metrics[:,4:,6:,:]
                self.set_metrics = self.set_metrics_3d_z
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

        self.tLT = self.tL.T
        self.tRT = self.tR.T

        ''' Set the methods that will be used to calculate the SATs '''

        if solver.pde_order == 1:          
            
            if self.method == 'diffeq': # will this work for burgers? Do i need to transfer things like tL?
                self.calc = solver.diffeq.calc_sat
                self.calc_dfdq = solver.diffeq.calc_dfdq_sat
                
            elif self.method == 'central' or self.method == 'nondissipative':
                if self.disc_type == 'div':
                    if self.dim == 1:
                        self.calc = self.central_div_1d
                    elif self.dim == 2:
                        self.calc = self.central_div_2d
                    elif self.dim == 3:
                        self.calc = self.central_div_3d
                elif self.disc_type == 'had':
                    if self.dim == 1:
                        self.calc = self.central_had_1d
                    elif self.dim == 2:
                        self.calc = self.central_had_2d
                    elif self.dim == 3:
                        self.calc = self.central_had_3d
                if self.neq_node == 1:
                    if self.disc_type == 'div':
                        pass
                        #self.calc_dfdq = self.central_scalar_div_dfdq
                    elif self.disc_type == 'had':
                        self.calc_dfdq = self.central_scalar_had_dfdq
                else:
                    self.calc_dfdq = self.dfdq_complexstep
                    
            elif self.method == 'upwind':
                if self.disc_type == 'div':
                    if self.dim == 1:
                        self.calc = self.upwind_div_1d
                    elif self.dim == 2:
                        self.calc = self.upwind_div_2d
                    elif self.dim == 3:
                        self.calc = self.upwind_div_3d
                elif self.disc_type == 'had':
                    if self.dim == 1:
                        self.calc = self.upwind_had_1d
                    elif self.dim == 2:
                        self.calc = self.upwind_had_2d
                    elif self.dim == 3:
                        self.calc = self.upwind_had_3d
                if self.neq_node == 1:
                    if self.disc_type == 'div':
                        pass
                        #self.calc_dfdq = self.upwind_scalar_div_dfdq
                    elif self.disc_type == 'had':
                        self.calc_dfdq = self.upwind_scalar_had_dfdq
                else:
                    self.calc_dfdq = self.dfdq_complexstep
                    
            elif self.method == 'lf' or self.method == 'llf' or self.method == 'lax_friedrichs':
                if self.disc_type == 'div':
                    if self.dim == 1:
                        self.calc = self.llf_div_1d
                    elif self.dim == 2:
                        self.calc = self.llf_div_2d
                    elif self.dim == 3:
                        self.calc = self.llf_div_3d
                elif self.disc_type == 'had':
                    if self.dim == 1:
                        self.calc = self.llf_had_1d
                    elif self.dim == 2:
                        self.calc = self.llf_had_2d
                    elif self.dim == 3:
                        self.calc = self.llf_had_3d
                if self.neq_node == 1:
                    if self.disc_type == 'div':
                        pass
                        #self.calc_dfdq = self.llf_scalar_div_dfdq
                    elif self.disc_type == 'had':
                        self.calc_dfdq = self.llf_scalar_had_dfdq
                else:
                    self.calc_dfdq = self.dfdq_complexstep
                    

                    
            ######### TO DO
            
            elif self.method == 'upwind':
                if self.neq_node == 1:
                    self.calc = lambda qL,qR: self.der1_upwind(qL, qR, 1) 
                    self.calc_dfdq = lambda qL,qR: self.dfdq_der1_upwind_scalar(qL, qR, 1)
                else:
                    self.calc = lambda qL,qR: self.der1_upwind(qL, qR, 1) # use Roe average?
                    self.calc_dfdq = self.dfdq_der1_complexstep
            elif (self.method.lower()=='ec' and self.diffeq_name=='Burgers') or self.method.lower()=='burgers ec':
                    self.calc = self.der1_burgers_ec
                    self.calc_dfdq = self.dfdq_der1_burgers_ec
            elif (self.method.lower()=='ec' and self.diffeq_name=='Quasi1dEuler') or self.method.lower()=='crean ec':
                    self.calc = self.der1_crean_ec
                    #self.calc_dfdq = complex step?
            elif (self.method.lower()=='es' and self.diffeq_name=='Quasi1dEuler') or self.method.lower()=='crean es':
                    self.calc = self.der1_crean_es
                    #self.calc_dfdq = complex step?
            # TODO: Add 'try' if it is there, if not revert to complexstep
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
            
        
    def set_metrics_2d_x(self, row):
        ''' pick out the appropriate metrics for the given row '''
        metrics = self.metrics[:,:,row::self.nelem[1]] 
        bdy_metrics = self.bdy_metrics[:,:,:,row::self.nelem[1]]
        return metrics, bdy_metrics
    
    def set_metrics_2d_y(self, col):
        ''' pick out the appropriate metrics for the given column '''
        start = col*self.nelem[0]
        end = start + self.nelem[1]
        metrics = self.metrics[:,:,start:end] 
        bdy_metrics = self.bdy_metrics[:,:,:,start:end]
        return metrics, bdy_metrics

    def set_metrics_3d_x(self, row):
        ''' pick out the appropriate metrics for the given row '''
        skipx = self.nelem[1]*self.nelem[2]
        metrics = self.metrics[:,:,row::skipx]
        bdy_metrics = self.bdy_metrics[:,:,:,row::skipx]
        return metrics, bdy_metrics
    
    def set_metrics_3d_y(self, row):
        ''' pick out the appropriate metrics for the given row '''
        start = row + (row//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
        end = start + self.nelem[1]*self.nelem[2]
        metrics = self.metrics[:,:,start:end:self.nelem[2]]
        bdy_metrics = self.bdy_metrics[:,:,:,start:end:self.nelem[2]]
        return metrics, bdy_metrics
    
    def set_metrics_3d_z(self, row):
        ''' pick out the appropriate metrics for the given row '''
        start = row*self.nelem[2]
        end = start + self.nelem[2]
        metrics = self.metrics[:,:,start:end]
        bdy_metrics = self.bdy_metrics[:,:,:,start:end]
        return metrics, bdy_metrics