#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 00:26:58 2020

@author: bercik
"""
import numpy as np

from Source.Disc.MakeMesh import MakeMesh
from Source.Disc.MakeDgOp import MakeDgOp
from Source.Disc.NumFlux import NumFlux
from Source.Solvers.PdeSolver import PdeSolver

class PdeSolverDg(PdeSolver, NumFlux):
    
    weak_form = False # if False, uses strong form

    def init_disc_specific(self):
            
        self.energy = self.dg_energy
        self.conservation = self.dg_conservation

        ''' Setup the discretization '''

        if self.nen > 0:
            print('WARNING: Can not set nen explicitly for DG. Ignoring, using p.')

        # Construct DG operators
        self.dg = MakeDgOp(self.p)

        self.nen = self.dg.nn_sol # number of solution nodes per element
        self.p = self.dg.p 
        self.xy_op = self.dg.xy # solution nodes on reference element
        self.van = self.dg.van # maps solution nodes to flux nodes
        self.vanf = self.dg.vanf # entry [i] maps solution nodes to i'th facet nodes
        self.proj = self.dg.proj # maps flux nodes to solution nodes
        self.weight = self.dg.weight # volume integration over flux nodes
        self.weightf = self.dg.weightf # surface integration over facet nodes

        # Calculate the number of elements and total number of solution nodes
        if self.nelem == 0: # Option 1: nn and nen are used to calculate nelem
            self.nelem = self.nn // self.nen
            self.nn = self.nelem * self.nen
        elif self.nelem > 0: # Option 2: nelem and nen are used to calculate nn
            self.nn = self.nelem * self.nen
        else:
            raise Exception('Invalid input for nelem. Set nelem=0 to use nn to set nelem automatically.')

        ''' Setup the mesh (flux nodes, NOT solution nodes) '''

        self.mesh = MakeMesh(self.xmin, self.xmax, self.isperiodic, self.nelem, self.xy_op)
        assert self.nn == len(self.mesh.xy), 'ERROR: self.nn not set exactly'

        ''' Apply required transformations to SBP operators '''

        # Calculate DG (weak) operators that include mapping and flux quadrature
	    # note: since linear mapping, jacobian is constant over domain
        # (see Yano notes 4.3.1) therefore we only compute the jacobian once
        # TODO: Generalize for non-linear mappings
        xy_elem = self.mesh.xy_elem[:,0,:]
        self.jac, self.detjac, self.invjac = MakeMesh.mesh_jac(xy_elem, self.dg.dd)
        
        eye = np.eye(self.neq_node)
        
        if self.weak_form:
            self.mass, self.vol, self.surf = self.dg.weak_op(self.detjac,self.invjac)
            self.mass_inv = np.linalg.inv(self.mass)
            
            # Apply kron products to DG operators
            self.mass = np.kron(self.mass, eye)
            self.vol = np.kron(self.vol, eye)
            for i in range(len(self.surf)):
                self.surf[i] = np.kron(self.surf[i], eye)
        else:
            self.mass, self.dd, self.surf_num, self.surf_flux = self.dg.strong_op(self.detjac,self.invjac)
            self.mass_inv = np.linalg.inv(self.mass)
            self.vol = self.mass_inv @ self.van.T @ self.weight @ self.van
            
            # Apply kron products to DG operators
            self.mass = np.kron(self.mass, eye)
            self.vol = np.kron(self.vol, eye)
            self.dd = np.kron(self.dd, eye)
            for i in range(len(self.surf_num)):
                self.surf_num[i] = np.kron(self.surf_num[i], eye)
                self.surf_flux[i] = np.kron(self.surf_flux[i], eye)
                
            self.diffeq.set_dg_strong_op(self.dd)


        ''' Modify solver approach '''
        
        if self.weak_form:
            self.dqdt = self.dg_dqdt_weak
            self.dfdq = None
        else:
            self.dqdt = self.dg_dqdt_strong
            self.dfdq = None

        # TODO: this will cause problems if nodes not collocated (ex. plotting)
        self.diffeq.set_mesh(self.mesh)
        self.init_numflux_class(self.sat_flux_type)


    
    def dg_dqdt_strong(self, q):
        
        q_flux = self.van @ q
        q_facB = self.vanf[0] @ q
        q_facA = self.vanf[1] @ q
        
        dqdt_out = self.vol @ self.diffeq.dqdt(q_flux)
        
        nfluxA, nfluxB = np.empty((1,self.nelem)) , np.empty((1,self.nelem))
        nfluxA[:,:-1] , nfluxB[:,1:] = self.numflux(q_facA[:,:-1], q_facB[:,1:])
        
        if self.isperiodic:
            nfluxA[:,[-1]] , nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], q_facB[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            nfluxA[:,[-1]] = self.numflux(self.diffeq.qL, q_facB[:,[0]])
            nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], self.diffeq.qR)
            
        flux = self.diffeq.calcE(q_flux)
        dqdt_out -= self.mass_inv @ ( self.surf_num[1]@nfluxA - self.surf_flux[1]@flux \
                                    + self.surf_num[0]@nfluxB - self.surf_flux[0]@flux )

        return dqdt_out
    
    def dg_dfdq_strong(self, q):
        
        q_flux = self.van @ q
        q_facB = self.vanf[0] @ q
        q_facA = self.vanf[1] @ q
        
        dqdt_out = self.vol @ self.diffeq.dqdt(q_flux)
        
        nfluxA, nfluxB = np.empty((1,self.nelem)) , np.empty((1,self.nelem))
        nfluxA[:,:-1] , nfluxB[:,1:] = self.numflux(q_facA[:,:-1], q_facB[:,1:])
        
        if self.isperiodic:
            nfluxA[:,[-1]] , nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], q_facB[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            nfluxA[:,[-1]] = self.numflux(self.diffeq.qL, q_facB[:,[0]])
            nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], self.diffeq.qR)
            
        flux = self.diffeq.calcE(q_flux)
        dqdt_out -= self.mass_inv @ ( self.surf_num[1]@nfluxA - self.surf_flux[1]@flux \
                                    + self.surf_num[0]@nfluxB - self.surf_flux[0]@flux )

        return dqdt_out
    
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
    
    def dg_dqdt_weak(self, q):
        # ONLY WORKS FOR WEAK FORM IN DIVERGENCE FORM (NO SPLIT FORM DIFFEQ)
        # TODO: Make it compatible with Diffeq classes
        # TODO: Generalze below for higher dimensions (n_facet > 2)
        q_flux = self.van @ q
        q_facB = self.vanf[0] @ q
        q_facA = self.vanf[1] @ q
        # TODO: Generalize calcE to handle cases when passed x (physical flux nodes)
        # TODO: Gereralize calcE and vol to handle multiple dimensions
        dqdt_out = self.mass_inv @ (self.vol @ self.diffeq.calcE(q_flux) + self.proj @ self.diffeq.calcG(q_flux))

        nfluxA, nfluxB = np.empty((1,self.nelem)) , np.empty((1,self.nelem))
        nfluxA[:,:-1] , nfluxB[:,1:] = self.numflux(q_facA[:,:-1], q_facB[:,1:])
        
        if self.isperiodic:
            nfluxA[:,[-1]] , nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], q_facB[:,[0]])
        else:
            # TODO: Generalize to include varying qL and qR
            nfluxA[:,[-1]] = self.numflux(self.diffeq.qL, q_facB[:,[0]])
            nfluxB[:,[0]] = self.numflux(q_facA[:,[-1]], self.diffeq.qR)
            
        dqdt_out += self.mass_inv @ ( self.surf[1]@nfluxA + self.surf[0]@nfluxB )

        return dqdt_out

    def dg_energy_elem(self,q):
        ''' compute the element-wise SBP energy of local solution vector q '''
        # TODO: Clean these up... consider using methods in methods class?
        return (q.T @ self.mass @ q)
    
    def dg_energy(self,q):
        ''' compute the global SBP energy of global solution vector q '''
        return np.tensordot(q, self.mass @ q)
    
    def dg_conservation_elem(self,q):
        ''' compute the element-wise SBP conservation of local solution vector q '''
        return np.sum(self.mass @ q , axis=0)

    def dg_conservation(self,q):
        ''' compute the global SBP conservation of global solution vector q '''
        return np.sum(self.mass @ q)

