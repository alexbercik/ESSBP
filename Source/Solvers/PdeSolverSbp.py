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


class PdeSolverSbp(PdeSolver, Sat):

    def init_disc_specific(self):
        
        self.energy = self.sbp_energy
        self.conservation = self.sbp_conservation

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
        satA , satB = np.empty(q.shape) , np.empty(q.shape)
        satA[:,:-1] , satB[:,1:] = self.calc_sat(q[:,:-1], q[:,1:])
        
        if self.isperiodic:
            satA[:,[-1]] , satB[:,[0]] = self.calc_sat(q[:,[-1]], q[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            satA[:,[-1]] = self.calc_sat(self.diffeq.qL.reshape((self.neq_node,1)), q[:,[0]])[0]
            satB[:,[0]] = self.calc_sat(q[:,[-1]], self.diffeq.qR.reshape((self.neq_node,1)))[1]
            
        dqdt_out += self.hh_inv_phys @ ( satA + satB )

        return dqdt_out
    
    def sbp_dfdq(self, q):
        
        dfdq_out = self.diffeq.dfdq(q)
        
        dsatAdq , dsatBdq = np.empty(q.shape) , np.empty(q.shape)
        dsatAdq[:,:-1] , dsatBdq[:,1:] = self.calc_sat(q[:,:-1], q[:,1:])
        
        if self.isperiodic:
            satA[:,[-1]] , satB[:,[0]] = self.calc_sat(q[:,[-1]], q[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            satA[:,[-1]] = self.calc_sat(self.diffeq.qL.reshape((self.neq_node,1)), q[:,[0]])[0]
            satB[:,[0]] = self.calc_sat(q[:,[-1]], self.diffeq.qR.reshape((self.neq_node,1)))[1]
            
        dfdq_out += self.hh_inv_phys @ ( satA + satB )

# =============================================================================
#     def sbp_dfdq_unstruc(self, q):
# 
#         n = q.size
#         dfdq_out = np.zeros((n,n))
# 
#         # Calculate the derivative of the sol at all the elements
#         for ie in range(self.nelem):
#             idx0, idx1 = self.get_elem_idx_1d(ie)
#             xy_idx0 = idx0 // self.neq_node
#             xy_idx1 = idx1 // self.neq_node
# 
#             q_elem = q[idx0:idx1]
#             dfdq_out[idx0:idx1, idx0:idx1] = self.diffeq_in.dfdq(q_elem, xy_idx0, xy_idx1).todense()
# 
#         # Calculate the numerical flux at each of the facets
#         for i_facet in range(self.mesh.nfacet):
#             idx_elems = self.mesh.gf2ge[:,i_facet]
# 
#             idx_elem_facetA = idx_elems[0]
#             idx_elem_facetB = idx_elems[1]
# 
#             # Calculate the value of q on each side of the interface
#             if idx_elem_facetA >= 0:
#                 idxA0, idxA1 = self.get_elem_idx_1d(idx_elem_facetA)
#                 qelemA = q[idxA0:idxA1]
#             else:
#                 # TODO: Generalize to include varying qL
#                 qelemA = self.diffeq_in.qL
# 
#             if idx_elem_facetB >= 0:
#                 idxB0, idxB1 = self.get_elem_idx_1d(idx_elem_facetB)
#                 qelemB = q[idxB0:idxB1]
#             else:
#                 # TODO:  Generalize to include varying qR
#                 qelemB = self.diffeq_in.qR
# 
#             # Calculate the SATs
#             dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB = self.diffeq_in.dfdq_sat(qelemA, qelemB)
# 
#             if idx_elem_facetA >= 0:
#                 dfdq_out[idxA0:idxA1, idxA0:idxA1] += self.hh_inv_phys @ dSatA_dqA
# 
#             if idx_elem_facetB >= 0:
#                 dfdq_out[idxB0:idxB1, idxB0:idxB1] += self.hh_inv_phys @ dSatB_dqB
# 
#             if idx_elem_facetA >=0 and idx_elem_facetB >= 0:
#                 dfdq_out[idxB0:idxB1, idxA0:idxA1] += self.hh_inv_phys @ dSatB_dqA
#                 dfdq_out[idxA0:idxA1, idxB0:idxB1] += self.hh_inv_phys @ dSatA_dqB
# 
#         return dfdq_out
# =============================================================================


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


