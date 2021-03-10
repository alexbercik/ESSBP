#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:28:38 2020

@author: andremarchildon
"""

import numpy as np

from Source.Disc.MakeMesh import MakeMesh
from Source.Disc.MakeSbpOp import MakeSbpOp
from Source.Disc.Sat import Sat
from Source.Solvers.PdeSolver import PdeSolver
import Source.Methods.Functions as fn


class PdeSolverSbp(PdeSolver, Sat):

    def init_disc_specific(self):
        
        self.energy = self.sbp_energy
        self.conservation = self.sbp_conservation
        self.entropy = self.sbp_entropy

        ''' Setup the discretization '''

        # Construct SBP operators
        if (self.disc_type.lower()=='csbp' and self.nelem<2):
            self.sbp = MakeSbpOp(self.p, self.disc_type, self.nn)
        elif self.nen>0:
            self.sbp = MakeSbpOp(self.p, self.disc_type, self.nen)
        elif self.disc_type.lower()=='csbp':
            self.nen = self.nn // self.nelem
            self.sbp = MakeSbpOp(self.p, self.disc_type, self.nen)
        else:
            self.sbp = MakeSbpOp(self.p, self.disc_type)

        self.nen = self.sbp.nn
        self.p = self.sbp.p
        self.xy_op = self.sbp.xy
        self.rrL = self.sbp.rr_all[0,:,:].T
        self.rrR = self.sbp.rr_all[1,:,:].T

        # Calculate the number of elements and total number of nodes
        if self.nelem == 0: # Option 1: nn and nen are used to calculate nelem
            self.nelem = self.nn // self.nen
            self.nn = self.nelem * self.nen
        elif self.nelem > 0: # Option 2: nelem and nen are used to calculate nn
            self.nn = self.nelem * self.nen
        else:
            raise Exception('Invalid input for nelem. Set nelem=0 to use nn to set nelem automatically.')

        ''' Setup the mesh '''

        self.mesh = MakeMesh(self.xmin, self.xmax, self.isperiodic, self.nelem, self.xy_op)
        assert self.nn == self.mesh.xy.shape[0], 'ERROR: self.nn not set exactly'

        ''' Apply required transformations to SBP operators '''

        # Calculate scaled SBP operators
	    # note: since linear mapping, jacobian is constant over domain
        # (see Yano notes 4.3.1) therefore we only compute the jacobian once
        # TODO: Generalize for non-linear mappings
        xy_elem = self.mesh.xy_elem[:,0,:]
        self.jac, self.detjac, self.invjac = MakeMesh.mesh_jac(xy_elem, self.sbp.dd)
        self.hh_phys, self.qq_phys, self.dd_phys \
            = self.sbp.ref_2_phys_op(self.detjac, self.invjac)
        self.hh_inv_phys = np.linalg.inv(self.hh_phys)

        # Apply kron products to SBP operators
        eye = np.eye(self.neq_node)
        self.hh_phys_unkronned = self.hh_phys
        self.hh_phys = np.kron(self.hh_phys, eye)
        self.hh_inv_phys = np.kron(self.hh_inv_phys, eye)
        self.qq_phys = np.kron(self.qq_phys, eye)
        self.dd_phys = np.kron(self.dd_phys, eye)
        self.rrL = np.kron(self.rrL, eye)
        self.rrR = np.kron(self.rrR, eye)

        ''' Modify solver approach '''

        self.diffeq.set_mesh(self.mesh)
        self.init_sat_class(self.sat_flux_type)
        self.diffeq.set_sbp_op(self.dd_phys, self.qq_phys, self.hh_inv_phys, self.rrL, self.rrR)
        self.dqdt = self.sbp_dqdt
        self.dfdq = self.sbp_dfdq

    
    def sbp_dqdt(self, q):
        
        dqdt_out = self.diffeq.dqdt(q)
        satL , satR = np.empty(q.shape) , np.empty(q.shape)
        satL[:,:-1] , satR[:,1:] = self.calc_sat(q[:,:-1], q[:,1:])
        
        if self.isperiodic:
            satL[:,[-1]] , satR[:,[0]] = self.calc_sat(q[:,[-1]], q[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            satL[:,[-1]] = self.calc_sat(self.diffeq.qL.reshape((self.neq_node,1)), q[:,[0]])[0]
            satR[:,[0]] = self.calc_sat(q[:,[-1]], self.diffeq.qR.reshape((self.neq_node,1)))[1]
            
        dqdt_out += self.hh_inv_phys @ ( satL + satR )

        return dqdt_out
    
    def sbp_dfdq(self, q):
        
        dfdq_diffeq = self.diffeq.dfdq(q)
        shape = dfdq_diffeq.shape
        
        # notation is: given an element q, SatL and SatR correspond to L and R interfaces
        # SatL has derivatives with respect to q and qL, the element to L
        # SatR has derivatives with respect to q and qR, the element to R
        # note that this is slightly different notation than the interface based one used in Sat class
        dSatLdq, dSatRdq, dSatLdqL, dSatRdqR = np.zeros(shape), np.empty(shape), np.zeros(shape), np.empty(shape)
        dSatRdq[:,:,:-1], dSatRdqR[:,:,:-1], dSatLdqL[:,:,1:], dSatLdq[:,:,1:] = self.calc_dfdq_sat(q[:,:-1], q[:,1:])                
        
        if self.isperiodic:
            dSatRdq[:,:,[-1]], dSatRdqR[:,:,[-1]], dSatLdqL[:,:,[0]], dSatLdq[:,:,[0]] = self.calc_dfdq_sat(q[:,[-1]], q[:,[0]])
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')
        
        # global 2d matrix blocks acting on element q blocks
        dfdq_q = dfdq_diffeq + fn.lm_gm(self.hh_inv_phys, dSatLdq + dSatRdq)
        # global 2d matrix blocks acting on element qL blocks (to the left of dfdq_q)
        # length nelem if periodic (first goes to top right, remaining under main diagonal)
        dfdq_qL = fn.lm_gm(self.hh_inv_phys, dSatLdqL)
        # global 2d matrix blocks acting on element qR blocks (to the right of dfdq_q)
        # length nelem if periodic (last goes to bottom left, remaining above main diagonal)
        dfdq_qR = fn.lm_gm(self.hh_inv_phys, dSatRdqR)
        
        if self.isperiodic:
            dfdq_2d = fn.glob_block_2d_mat_periodic(dfdq_qL,dfdq_q,dfdq_qR)
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')        
            
        return dfdq_2d


    def sbp_energy_elem(self,q):
        ''' compute the element-wise SBP energy of local solution vector q '''
        return (q.T @ self.hh_phys @ q)
    
    def sbp_energy(self,q):
        ''' compute the global SBP energy of global solution vector q '''
        return np.tensordot(q, self.hh_phys @ q)
    
    def sbp_conservation_elem(self,q):
        ''' compute the element-wise SBP conservation of local solution vector q '''
        return np.sum(self.hh_phys @ q , axis=0)

    def sbp_conservation(self,q):
        ''' compute the global SBP conservation of global solution vector q '''
        return np.sum(self.hh_phys @ q)
    
    def sbp_entropy(self,q):
        ''' compute the global SBP entropy of global solution vector q '''
        s = self.diffeq.entropy(q)
        return np.sum(self.hh_phys_unkronned @ s)


