#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:28:38 2020

@author: andremarchildon
"""

import numpy as np

from Source.Disc.MakeMesh import MakeMesh
from Source.Disc.MakeSbpOp import MakeSbpOp
from Source.DiffEq.DiffEqBase import DiffEqOverwrite


class PdeSolverSbp():

    def sbp_init(self):

        ''' Setup the discretization '''

        self.neq_node = self.diffeq.neq_node

        # Construct SBP operators
        if (self.disc_type.lower()=='csbp' and self.nelem<2):
            self.sbp = MakeSbpOp(self.p, self.disc_type, self.nn)
        elif self.nen>0:
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

        self.diffeq_in = self.diffeq
        self.diffeq_in.set_mesh(self.mesh)
        self.diffeq_in.set_sat(self.sat_type)
        self.diffeq_in.set_sbp_op(self.dd_phys, self.qq_phys, self.hh_inv_phys, self.rrL, self.rrR)

        # Create a modified diffeq class with the dqdt, dfdq and dqds for SBP op
        self.diffeq = DiffEqOverwrite(self.diffeq_in, self.sbp_dqdt, self.sbp_dfdq, self.sbp_dfds,
                                           self.calc_cons_obj, self.n_cons_obj)



    
    def sbp_dqdt(self, q):
        
        dqdt_out = self.diffeq_in.dqdt(q)
        satA , satB = np.empty(q.shape) , np.empty(q.shape)
        satA[:,:-1] , satB[:,1:] = self.diffeq_in.calc_sat(q[:,:-1], q[:,1:])
        
        if self.isperiodic:
            satA[:,[-1]] , satB[:,[0]] = self.diffeq_in.calc_sat(q[:,[-1]], q[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            satA[:,[-1]] = self.diffeq_in.calc_sat(self.diffeq_in.qL.reshape((self.neq_node,1)), q[:,[0]])[0]
            satB[:,[0]] = self.diffeq_in.calc_sat(q[:,[-1]], self.diffeq_in.qR.reshape((self.neq_node,1)))[1]
            
        dqdt_out += self.hh_inv_phys @ ( satA + satB )

        return dqdt_out
    
    def sbp_dfdq(self, q):
        raise Exception('Not coded up yet.')

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

# =============================================================================
#     def sbp_dfds_unstruc(self, q):
# 
#         # n = q.size
#         dfds_out = np.zeros((self.len_q, self.npar))
# 
#         # Calculate the derivative of the sol at all the elements
#         for ie in range(self.nelem):
#             idx0, idx1 = self.get_elem_idx_1d(ie)
#             xy_idx0 = idx0 // self.neq_node
#             xy_idx1 = idx1 // self.neq_node
# 
#             q_elem = q[idx0:idx1]
#             dfds_out[idx0:idx1, idx0:idx1] = self.diffeq_in.dfds(q_elem, xy_idx0, xy_idx1).todense()
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
#             dSatA_ds, dSatB_ds = self.diffeq_in.dfds_sat(qelemA, qelemB)
# 
#             dfds_out[idxA0:idxA1, :] += self.hh_inv_phys @ dSatA_ds
#             dfds_out[idxB0:idxB1, :] += self.hh_inv_phys @ dSatB_ds
# 
#         return dfds_out
# =============================================================================
    
    def sbp_dfds(self, q):
        # TODO
        raise Exception('Not coded up yet')

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

    def sbp_calc_error(self, q=None, tf=None, method='H-norm'):
        '''
        Purpose
        ----------
        Calculate the error of the solution using a defined method. This calls
        on smaller specific functions also defined below.
        TODO: modify for use with FD

        Parameters
        ---------
        q : np array (optional)
            The global solution at the global nodes. If None, this uses the
            final solution determined by solve()
        tf : float (optional)
            The time at which to evaluate the error. If None, this uses the
            default final time of solve()
        method : string (optional)
            Determines which error to use. Options are:
                'H-norm' : the SBP error sqrt((q-q_ex).T @ H @ (q-q_ex))
                'Rms' : the standard root mean square error in L2 norm
                'Boundary' : The simple periodic boundary error | q_1 - q_N |
                'Truncation-SBP' : the SBP error but instead using er = dqdt(q-q_ex)
                'Truncation-Rms' : the Rms error but instead using er = dqdt(q-q_ex)
        '''
        if tf == None: tf = self.t_final
        if q == None:
            if self.q_sol.ndim == 2: q = self.q_sol
            elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]

        # determine error to use
        if method == 'H-norm' or method == 'Rms':
            q_exa = self.diffeq_in.exact_sol(tf)
            error = q - q_exa
        elif method == 'Boundary':
            error = abs(q[0]-q[-1])
        elif method == 'Truncation-SBP' or method == 'Truncation-Rms':
            q_exa = self.diffeq_in.exact_sol(tf)
            error = self.diffeq.dqdt(q - q_exa)
        else:
            raise Exception('Unknown error method. Use one of: H-norm, Rms, Boundary, Truncation-SBP, Truncation-Rms')

        # if we still need to apply a norm, do it
        if method == 'H-norm' or method == 'Truncation-SBP':
            error = self.sbp_norm(error)
        elif method == 'Rms' or method == 'Truncation-Rms':
            error = np.linalg.norm(error) / np.sqrt(self.nn)
        return error

