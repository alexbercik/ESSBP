#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:28:38 2020

@author: bercik
"""

import numpy as np
import gc

from Source.Disc.MakeMesh import MakeMesh
from Source.Disc.MakeSbpOp import MakeSbpOp
from Source.Disc.Sat import Sat
from Source.Disc.ADiss import ADiss
from Source.Solvers.PdeSolver import PdeSolver
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp


class PdeSolverSbp(PdeSolver):

    Vptop1 = None

    def init_disc_specific(self):
        
        self.energy = self.sbp_energy
        self.kinetic_energy = self.sbp_kinetic_energy
        self.conservation = self.sbp_conservation
        self.energy_der = self.sbp_energy_der
        self.conservation_der = self.sbp_conservation_der
        self.entropy = self.sbp_entropy
        self.entropy_der = self.sbp_entropy_der
        self.enstrophy = self.sbp_enstrophy
        
        # Construct SBP operators
        self.sbp = MakeSbpOp(self.p, self.disc_nodes, self.nen, print_progress=self.print_progress)
        self.nen = self.sbp.nn
        self.p = self.sbp.p
        self.x_op = self.sbp.x
        #self.tL = self.sbp.tL.reshape((self.nen,1))
        #self.tR = self.sbp.tR.reshape((self.nen,1))
        #self.tLT = self.tL.T
        #self.tRT = self.tR.T
        if self.dim == 1:
            self.nn = self.nelem * self.nen
            self.qshape = (self.nen * self.neq_node , self.nelem)
        elif self.dim == 2:
            self.nn = (self.nelem[0] * self.nen, self.nelem[1] * self.nen)
            self.qshape = (self.nen**2 * self.neq_node , self.nelem[0] * self.nelem[1])
            #Hperp = np.diag(self.sbp.H)
        elif self.dim == 3:
            self.nn = (self.nelem[0] * self.nen, self.nelem[1] * self.nen, self.nelem[2] * self.nen)
            self.qshape = (self.nen**3 * self.neq_node , self.nelem[0] * self.nelem[1] * self.nelem[2])  
            #H = np.diag(self.sbp.H) 
            #Hperp = np.kron(H,H)

        
        # decide whether to use sparse formulation or not
        if self.sparse is None:
            self.sparse = False
            if self.dim == 1: 
                if self.disc_nodes in ['csbp', 'upwind', 'hgtl', 'mattsson']:
                    self.sparse = True
                if (self.disc_type == 'had') and (self.neq_node > 1):
                    self.sparse = True
            elif self.dim == 2:
                self.sparse = True
                if self.disc_nodes in ['lgl','lg']:
                    self.sparse = False
                    self.satsparse = False
            elif self.dim == 3:
                self.sparse = True
        elif not self.sparse and self.sat_sparse is None:
            self.sat_sparse = False

        if self.sparse and not self.sat_sparse:
            print('NOTE: Overriding sat_sparse=False to sat_sparse=True since sparse=True')
            self.sat_sparse = True
        
        elif self.sat_sparse is None:
            self.sat_sparse = False
            if self.disc_nodes in ['csbp', 'upwind', 'hgtl', 'mattsson']:
                self.sat_sparse = True
            #if self.disc_nodes == 'lgl' and (self.dim > 1 or self.p > 3):
            #    self.sat_sparse = True


        ''' Setup the mesh and apply required transformations to SBP operators '''

        self.mesh = MakeMesh(self.dim, self.xmin, self.xmax, self.nelem, self.x_op, 
                             self.settings['warp_factor'], self.settings['warp_type'],
                             self.print_progress)

        #TODO: Should probably sparsify this
        self.mesh.get_jac_metrics(self.sbp, self.periodic,
                        metric_method = self.settings['metric_method'], 
                        bdy_metric_method = self.settings['bdy_metric_method'],
                        jac_method=self.settings['jac_method'],
                        use_optz_metrics = self.settings['use_optz_metrics'],
                        calc_exact_metrics = self.settings['calc_exact_metrics'],
                        optz_method = self.settings['metric_optz_method'],
                        had_metric_alpha = self.settings['had_alpha'],
                        had_metric_beta = self.settings['had_beta'])
        
        if self.settings['stop_after_metrics']:
            return
        
        # Set physical operators, and sparsify operators if needed
        if self.settings['skew_sym']: form = 'skew_sym'
        else: form = 'div'
        
        self.sbp.ref_2_phys(self.mesh, self.neq_node, form, self.disc_type, self.sparse, self.sat_sparse)
        self.H_phys, self.H_inv_phys = self.sbp.H_phys, self.sbp.H_inv_phys
        self.Dx, self.Volx = self.sbp.Dx, self.sbp.Volx
        if self.dim > 1:
            self.Dy, self.Voly = self.sbp.Dy, self.sbp.Voly
        if self.dim > 2:
            self.Dz, self.Volz = self.sbp.Dz, self.sbp.Volz

        self.volume = np.sum(self.H_phys)
        # no need to save tL, tR, Dx_unkronned, etc.
        # but if there is, we can access them from self.sat

        self.adiss = ADiss(self)
        self.dissipation = self.adiss.dissipation

        ''' Modify solver approach '''

        self.diffeq.set_mesh(self.mesh)
        if self.settings['stop_after_mesh']:
            return
        if self.dim == 1:
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_1d_div
                self.dfdq = self.dfdq_1d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_1d_had
                self.dfdq = self.dfdq_1d_had  
            self.sat = Sat(self, None, form)
        elif self.dim == 2:
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_2d_div
                self.dfdq = self.dfdq_2d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_2d_had
                self.dfdq = self.dfdq_2d_had  
            self.satx = Sat(self, 'x', form)
            self.saty = Sat(self, 'y', form)
        elif self.dim == 3:
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_3d_div
                self.dfdq = self.dfdq_3d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_3d_had
                self.dfdq = self.dfdq_3d_had  
            self.satx = Sat(self, 'x', form)
            self.saty = Sat(self, 'y', form)
            self.satz = Sat(self, 'z', form)

        # save sparsity information
        if self.sparse:
            self.lm_gv = staticmethod(sp.lm_gv)
            self.gm_gv = staticmethod(sp.gm_gv)
            self.gm_gm_had_diff = staticmethod(sp.gm_gm_had_diff)

        else:
            self.lm_gv = staticmethod(fn.lm_gv)
            self.gm_gv = staticmethod(fn.gm_gv)
            self.gm_gm_had_diff = staticmethod(fn.gm_gm_had_diff)
            
        # originally was just using this for if diffeq.use_diffeq_dExdx, 
        # but it is also useful for diffeq.calc_breaking_times
        self.diffeq.set_sbp_op_1d(self.Dx, self.gm_gv)

    def dqdt_1d_div(self, q, t):
        ''' the main dqdt function for divergence form in 1D '''
        E = self.diffeq.calcEx(q)
        if self.use_diffeq_dExdx:
            dExdx = self.diffeq.dExdx(q,E)
        else:
            dExdx = self.gm_gv(self.Dx, E, self.neq_node)
        
        if self.periodic:
            sat = self.sat.calc(q,E)
        elif self.bc == 'homogeneous':
            sat = self.sat.calc(q, E, q_bdyL=np.array([0]), q_bdyR=np.array([0]))
        elif self.bc == 'homogeneous-outflow':
            sat = self.sat.calc(q, E, q_bdyL=np.array([0]), q_bdyR='None')
        elif self.bc == 'dirichlet':
            sat = self.sat.calc(q, E, q_bdyL=self.diffeq.qL, q_bdyR=self.diffeq.qR, E_bdyL=self.diffeq.EL, E_bdyR=self.diffeq.ER)
        else:
            raise Exception('Not coded up yet')
    
        sat_term = fn.gdiag_gv(self.H_inv_phys, sat, self.neq_node)
        dqdt = - dExdx + self.diffeq.calcG(q,t) + sat_term + self.dissipation(q)
        return np.ascontiguousarray(dqdt)
        
    def dqdt_2d_div(self, q, t):
        ''' the main dqdt function for divergence form in 2D '''
        Ex = self.diffeq.calcEx(q)
        #dExdx = self.gm_gv(self.Dx, Ex, self.neq_node)
        dExdx = self.gm_gv(self.Volx, Ex, self.neq_node)
        Ey = self.diffeq.calcEy(q)
        #dEydy = self.gm_gv(self.Dy, Ey, self.neq_node)
        dEydy = self.gm_gv(self.Voly, Ey, self.neq_node)

        satx, saty = np.zeros_like(q), np.zeros_like(q)
        if self.periodic[0]:   # x sat (in ref space) 
            for row in range(self.nelem[1]):
                # starts at bottom left to bottom right, then next row up
                satx[:,row::self.nelem[1]] = self.satx.calc(q[:,row::self.nelem[1]],Ex[:,row::self.nelem[1]],Ey[:,row::self.nelem[1]],row)      
        else:
            raise Exception('Not coded up yet')
            
        if self.periodic[1]:   # y sat (in ref space) 
            for col in range(self.nelem[0]):
                # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                saty[:,start:end] = self.saty.calc(q[:,start:end],Ex[:,start:end],Ey[:,start:end],col)
        else:
            raise Exception('Not coded up yet')
        
        sat_term = fn.gdiag_gv(self.H_inv_phys, (satx + saty), self.neq_node)
        dqdt = - dExdx - dEydy + sat_term + self.diffeq.calcG(q,t) + self.dissipation(q)
        return np.ascontiguousarray(dqdt)

    def dqdt_3d_div(self, q, t):
        ''' the main dqdt function for divergence form in 3D '''
        Ex = self.diffeq.calcEx(q)
        dExdx = self.gm_gv(self.Dx, Ex, self.neq_node)
        Ey = self.diffeq.calcEy(q)
        dEydy = self.gm_gv(self.Dy, Ey, self.neq_node)
        Ez = self.diffeq.calcEz(q)
        dEzdz = self.gm_gv(self.Dz, Ez, self.neq_node)
        satx, saty, satz = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
        
        skipx = self.nelem[1]*self.nelem[2]
        skipz = self.nelem[0]*self.nelem[1]
        if self.periodic[0]:   # x sat (in ref space) 
            for rowx in range(skipx):
                satx[:,rowx::skipx] = self.satx.calc(q[:,rowx::skipx],Ex[:,rowx::skipx],Ey[:,rowx::skipx],Ez[:,rowx::skipx],rowx)
        else:
            raise Exception('Not coded up yet')
        if self.periodic[1]:   # y sat (in ref space) 
            for coly in range(self.nelem[0]*self.nelem[2]):
                    start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                    end = start + skipx
                    saty[:,start:end:self.nelem[2]] = self.saty.calc(q[:,start:end:self.nelem[2]],Ex[:,start:end:self.nelem[2]],Ey[:,start:end:self.nelem[2]],Ez[:,start:end:self.nelem[2]],coly)
        else:
            raise Exception('Not coded up yet')
        if self.periodic[2]:   # z sat (in ref space) 
            for colz in range(skipz):
                    start = colz*self.nelem[2]
                    end = start + self.nelem[2]
                    satz[:,start:end] = self.satz.calc(q[:,start:end],Ex[:,start:end],Ey[:,start:end],Ez[:,start:end],colz)      
        else:
            raise Exception('Not coded up yet')
        
        sat_term = fn.gdiag_gv(self.H_inv_phys, (satx + saty + satz), self.neq_node)
        dqdt = - dExdx - dEydy - dEzdz + sat_term + self.diffeq.calcG(q,t) + self.dissipation(q)
        return np.ascontiguousarray(dqdt)
        
    def dqdt_1d_had(self, q, t):
        ''' the main dqdt function for hadamard form in 1D '''
        #Fvol = self.build_F_vol(q)
        #dExdx = 2*self.gm_gm_had_diff(self.Dx, Fvol)
        #dqdt = - dExdx
        if self.sparse:
            dqdt = sp.Vol_had_Fvol_diff(self.Volx,q,self.calc_had_flux,self.neq_node)
        else:
            dqdt = fn.Vol_had_Fvol_diff(self.Volx,q,self.calc_had_flux,self.neq_node)
            #dqdt = - dExdx
        
        if self.periodic:
            sat = self.sat.calc(q,None)
            #sat = self.sat.calc(q,Fvol)
        elif self.bc == 'dirichlet':
            sat = self.sat.calc(q, q_bdyL=self.diffeq.qL, q_bdyR=self.diffeq.qR)
        else:
            raise Exception('Not coded up yet')
        
        sat_term = fn.gdiag_gv(self.H_inv_phys, sat, self.neq_node)
        dqdt += sat_term + self.diffeq.calcG(q,t) + self.dissipation(q)
        return np.ascontiguousarray(dqdt)
        
    def dqdt_2d_had(self, q, t):
        ''' the main dqdt function for hadamard form in 2D '''
        #Fxvol, Fyvol = self.build_F_vol(q)
        #dExdx = 2*self.gm_gm_had_diff(self.Dx, Fxvol)
        #dEydy = 2*self.gm_gm_had_diff(self.Dy, Fyvol)
        #dExdx = self.gm_gm_had_diff(self.Volx, Fxvol)
        #dEydy = self.gm_gm_had_diff(self.Voly, Fyvol)
        #dqdt = - dExdx - dEydy
        if self.sparse:
            dqdt = sp.VolxVoly_had_Fvol_diff(self.Volx,self.Voly,q,self.calc_had_flux,self.neq_node)
        else:
            dqdt = fn.VolxVoly_had_Fvol_diff(self.Volx,self.Voly,q,self.calc_had_flux,self.neq_node)

        satx, saty = np.zeros_like(q), np.zeros_like(q)
        if self.periodic[0]:   # x sat (in ref space) 
            for row in range(self.nelem[1]):
                # starts at bottom left to bottom right, then next row up
                satx[:,row::self.nelem[1]] = self.satx.calc(q[:,row::self.nelem[1]],row)    
        else:
            raise Exception('Not coded up yet')
            
        if self.periodic[1]:   # y sat (in ref space) 
            for col in range(self.nelem[0]):
                # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                saty[:,start:end] = self.saty.calc(q[:,start:end],col)
        else:
            raise Exception('Not coded up yet')
        
        sat_term = fn.gdiag_gv(self.H_inv_phys, (satx + saty), self.neq_node)
        dqdt += sat_term + self.diffeq.calcG(q,t) + self.dissipation(q)
        return np.ascontiguousarray(dqdt)
        
    def dqdt_3d_had(self, q, t):
        ''' the main dqdt function for hadamard form in 3D '''
        Fxvol, Fyvol, Fzvol = self.build_F_vol(q)
        dExdx = 2*self.gm_gm_had_diff(self.Dx, Fxvol)
        dEydy = 2*self.gm_gm_had_diff(self.Dy, Fyvol)
        dEzdz = 2*self.gm_gm_had_diff(self.Dz, Fzvol)
        satx, saty, satz, = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
        
        skipx = self.nelem[1]*self.nelem[2]
        skipz = self.nelem[0]*self.nelem[1]
        if self.periodic[0]:   # x sat (in ref space) 
            for rowx in range(skipx):
                satx[:,rowx::skipx] = self.satx.calc(q[:,rowx::skipx],Fxvol[:,:,rowx::skipx],Fyvol[:,:,rowx::skipx],Fzvol[:,:,rowx::skipx],rowx)
        else:
            raise Exception('Not coded up yet')
        if self.periodic[1]:   # y sat (in ref space) 
            for coly in range(self.nelem[0]*self.nelem[2]):
                    start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                    end = start + skipx
                    saty[:,start:end:self.nelem[2]] = self.saty.calc(q[:,start:end:self.nelem[2]],Fxvol[:,:,start:end:self.nelem[2]],Fyvol[:,:,start:end:self.nelem[2]],Fzvol[:,:,start:end:self.nelem[2]],coly)
        else:
            raise Exception('Not coded up yet')
        if self.periodic[2]:   # z sat (in ref space) 
            for colz in range(skipz):
                    start = colz*self.nelem[2]
                    end = start + self.nelem[2]
                    satz[:,start:end] = self.satz.calc(q[:,start:end],Fxvol[:,:,start:end],Fyvol[:,:,start:end],Fzvol[:,:,start:end],colz)      
        else:
            raise Exception('Not coded up yet')
        
        sat_term = fn.gdiag_gv(self.H_inv_phys, (satx + saty + satz), self.neq_node)
        dqdt = - dExdx - dEydy - dEzdz + sat_term + self.diffeq.calcG(q,t) + self.dissipation(q)
        return np.ascontiguousarray(dqdt)
    

    def dfdq_1d_div(self, q, t):
        ''' the main linearized RHS function for divergence form in 1D '''
        # TODO: this is not fully correct
        if not self.dissipation.type.lower() == 'nd':
            raise Exception('Not coded up')
        
        if self.use_diffeq_dExdx:
            raise Exception('Not coded up')
        else:
            A = self.diffeq.dExdq(q)
            vol = -fn.gm_gm(self.Dx, A) # TODO: Dx is a gm (possibly sparse), but A is a gbdiag

        if self.periodic:
            sat = self.sat.calc_dfdq(q, A)
        elif self.bc == 'homogeneous':
            sat = self.sat.calc_dfdq(q, A, q_bdyL=np.array([0]), q_bdyR=np.array([0]))
        elif self.bc == 'dirichlet':
            sat = self.sat.calc_dfdq(q, A, q_bdyL=self.diffeq.qL, q_bdyR=self.diffeq.qR)
        else:
            raise Exception('Not coded up yet')
        
        H_inv_kron = fn.repeat_neq_gv(self.H_inv_phys, self.neq_node)
        dfdq = fn.sp_block_diag(vol) + self.diffeq.dGdq(q) \
            + H_inv_kron.flatten('f')[:, np.newaxis] * sat
        return dfdq
    
    def dfdq_2d_div(self, q, t):
        ''' the main linearized RHS function for divergence form in 2D '''
        raise Exception('Not done yet.')
    
    def dfdq_3d_div(self, q, t):
        ''' the main linearized RHS function for divergence form in 3D '''
        raise Exception('Not done yet.') 
    
    def dfdq_1d_had(self, q, t):
        ''' the main linearized RHS function for hadamard form in 1D '''
        raise Exception('Not done yet.')
    
    def dfdq_2d_had(self, q, t):
        ''' the main linearized RHS function for hadamard form in 2D '''
        raise Exception('Not done yet.')
    
    def dfdq_3d_had(self, q, t):
        ''' the main linearized RHS function for hadamard form in 3D '''
        raise Exception('Not done yet.')
        
 
    
    def sbp_energy(self,q,nen=None):
        ''' compute the global SBP energy of global solution vector q '''
        if (nen == self.neq_node) or (nen is None):
            H_phys = self.H_phys
            local_neq = self.neq_node
        elif nen == 1:
            H_phys = self.H_phys
            local_neq = 1
        else:
            raise Exception('Something went wrong, nen = ',nen)
        if q.ndim == 2:
            energy = fn.norm_gv_neq(H_phys, q, local_neq)
        elif q.ndim == 3:
            energy = fn.norm_gv_neq_3d(H_phys, q, local_neq)
        else:
            raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return energy

    def sbp_conservation(self,q):
        ''' compute the global SBP conservation of global solution vector q '''
        H_phys = self.H_phys
        if q.ndim == 2:
            cons = fn.sum_gv_neq(H_phys, q, self.neq_node)
        elif q.ndim == 3:
            cons = fn.sum_gv_neq_3d(H_phys, q, self.neq_node)
        else:
            raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return cons
    
    def sbp_conservation_der(self,dqdt):
        ''' compute the derivative of the global SBP conservation. '''
        H_phys = self.H_phys
        if dqdt.ndim == 2:
            cons = fn.sum_gv_neq(H_phys, dqdt, self.neq_node)
        elif dqdt.ndim == 3:
            cons = fn.sum_gv_neq_3d(H_phys, dqdt, self.neq_node)
        else:
            raise Exception('Something went wrong, dqdt.ndim = ',dqdt.ndim)
        return cons
    
    def sbp_energy_der(self,q,dqdt):
        ''' compute the derivative of the global SBP energy . '''
        H_phys = self.H_phys
        local_neq = self.neq_node
        if q.ndim == 2:
            energy = fn.inner_product_gv_neq(H_phys, q, dqdt, local_neq)
        elif q.ndim == 3:
            energy = fn.inner_product_gv_neq_3d(H_phys, q, dqdt, local_neq)
        else:
            raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return 2 * energy
    
    def sbp_entropy(self,q):
        ''' compute the global SBP entropy of global solution vector q '''
        s = self.diffeq.entropy(q)
        if q.ndim == 2: ent = np.sum(self.H_phys * s)
        elif q.ndim == 3: ent = np.sum(self.H_phys[:,:,np.newaxis] * s, axis=(0,1))
        else: raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return ent
    
    def sbp_entropy_der(self,q,dqdt):
        ''' compute the derivative of the global SBP entropy . '''
        w = self.diffeq.entropy_var(q)
        H_phys = self.H_phys
        local_neq = self.neq_node
        if q.ndim == 2:
            entropy = fn.inner_product_gv_neq(H_phys, w, dqdt, local_neq)
        elif q.ndim == 3:
            entropy = fn.inner_product_gv_neq_3d(H_phys, w, dqdt, local_neq)
        else:
            raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return entropy
    
    def sbp_kinetic_energy(self,q):
        ''' compute the global SBP kinetic energy of global solution vector q '''
        # NOTE: Need to think hard for the case where H is not diagonal
        k = self.diffeq.kinetic_energy(q)
        if q.ndim == 2: energy = np.sum(self.H_phys * k) / self.volume
        elif q.ndim == 3: energy = np.sum(self.H_phys[:,:,np.newaxis] * k, axis=(0,1)) / self.volume
        else: raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return energy
    
    def sbp_enstrophy(self,q):
        ''' compute the global SBP enstrophy of global solution vector q '''
        # NOTE: Need to think hard for the case where H is not diagonal
        s = self.diffeq.enstropy(q)
        if q.ndim == 2: ent = np.sum(self.H_phys * s)
        elif q.ndim == 3: ent = np.sum(self.H_phys[:,:,np.newaxis] * s, axis=(0,1))
        else: raise Exception('Something went wrong, q.ndim = ',q.ndim)
        return ent


    ''' temporary functions '''
    def check_invariants(self,return_ers=False,return_max_only=True,returnRL=True):
        eye = np.eye(self.nen)
        if self.dim==1:
            pass
        elif self.dim==2:
            Hperp = np.diag(self.sbp.H)
            Dx = np.kron(self.sbp.D, eye)
            Dy = np.kron(eye, self.sbp.D)
            tLTx = np.kron(self.sbp.tL, eye)
            tRTx = np.kron(self.sbp.tR, eye)
            tLTy = np.kron(eye, self.sbp.tL)
            tRTy = np.kron(eye, self.sbp.tR)
            tLx = tLTx.T
            tRx = tRTx.T
            tLy = tLTy.T
            tRy = tRTy.T
            Hinv = np.linalg.inv(np.kron(self.sbp.H, self.sbp.H))
            LHS1 = Dx @ self.mesh.metrics[:,0,:] + Dy @ self.mesh.metrics[:,2,:]
            LHS2 = Dx @ self.mesh.metrics[:,1,:] + Dy @ self.mesh.metrics[:,3,:]    
            RHS1 = tRx @ ( Hperp[:,None] * ( tRTx @ self.mesh.metrics[:,0,:] - self.mesh.bdy_metrics[:,1,0,:] )) \
                 - tLx @ ( Hperp[:,None] * ( tLTx @ self.mesh.metrics[:,0,:] - self.mesh.bdy_metrics[:,0,0,:] )) \
                 + tRy @ ( Hperp[:,None] * ( tRTy @ self.mesh.metrics[:,2,:] - self.mesh.bdy_metrics[:,3,2,:] )) \
                 - tLy @ ( Hperp[:,None] * ( tLTy @ self.mesh.metrics[:,2,:] - self.mesh.bdy_metrics[:,2,2,:] )) 
            RHS2 = tRx @ ( Hperp[:,None] * ( tRTx @ self.mesh.metrics[:,1,:] - self.mesh.bdy_metrics[:,1,1,:] )) \
                 - tLx @ ( Hperp[:,None] * ( tLTx @ self.mesh.metrics[:,1,:] - self.mesh.bdy_metrics[:,0,1,:] )) \
                 + tRy @ ( Hperp[:,None] * ( tRTy @ self.mesh.metrics[:,3,:] - self.mesh.bdy_metrics[:,3,3,:] )) \
                 - tLy @ ( Hperp[:,None] * ( tLTy @ self.mesh.metrics[:,3,:] - self.mesh.bdy_metrics[:,2,3,:] ))
            tot1 = LHS1 - Hinv @ RHS1
            tot2 = LHS2 - Hinv @ RHS2
        else:
            #Hperp = np.diag(np.kron(self.sbp.H,self.sbp.H))
            h = np.diag(self.sbp.H)           # 1D weights
            Hperp = np.kron(h, h) 
            D = sp.lm_to_sp(self.sbp.D)
            Dx = sp.kron_lm_eye(sp.kron_lm_eye(D, self.nen), self.nen)
            Dy = sp.kron_lm_eye(sp.kron_eye_lm(D, self.nen), self.nen)
            Dz = sp.kron_eye_lm(sp.kron_eye_lm(D, self.nen), self.nen)
            tL = sp.lm_to_sp(self.sbp.tL.reshape(self.nen,1))
            tR = sp.lm_to_sp(self.sbp.tR.reshape(self.nen,1))
            tLx = sp.kron_lm_eye(sp.kron_lm_eye(tL, self.nen), self.nen)
            tRx = sp.kron_lm_eye(sp.kron_lm_eye(tR, self.nen), self.nen)
            tLy = sp.kron_lm_eye(sp.kron_eye_lm(tL, self.nen), self.nen)
            tRy = sp.kron_lm_eye(sp.kron_eye_lm(tR, self.nen), self.nen)
            tLz = sp.kron_eye_lm(sp.kron_eye_lm(tL, self.nen), self.nen)
            tRz = sp.kron_eye_lm(sp.kron_eye_lm(tR, self.nen), self.nen)
            tLTx = tLx.T()
            tRTx = tRx.T()
            tLTy = tLy.T()
            tRTy = tRy.T()
            tLTz = tLz.T()
            tRTz = tRz.T()
            Hinv = np.kron(np.kron(1/h, 1/h),1/h)
            LHS1 = sp.lm_gv(Dx, self.mesh.metrics[:,0,:]) + sp.lm_gv(Dy, self.mesh.metrics[:,3,:]) + sp.lm_gv(Dz, self.mesh.metrics[:,6,:])
            LHS2 = sp.lm_gv(Dx, self.mesh.metrics[:,1,:]) + sp.lm_gv(Dy, self.mesh.metrics[:,4,:]) + sp.lm_gv(Dz, self.mesh.metrics[:,7,:])
            LHS3 = sp.lm_gv(Dx, self.mesh.metrics[:,2,:]) + sp.lm_gv(Dy, self.mesh.metrics[:,5,:]) + sp.lm_gv(Dz, self.mesh.metrics[:,8,:])
            RHS1 = sp.lm_gv(tRx, ( Hperp[:,None] * ( sp.lm_gv(tRTx, self.mesh.metrics[:,0,:]) - self.mesh.bdy_metrics[:,1,0,:] ))) \
                 - sp.lm_gv(tLx, ( Hperp[:,None] * ( sp.lm_gv(tLTx, self.mesh.metrics[:,0,:]) - self.mesh.bdy_metrics[:,0,0,:] ))) \
                 + sp.lm_gv(tRy, ( Hperp[:,None] * ( sp.lm_gv(tRTy, self.mesh.metrics[:,3,:]) - self.mesh.bdy_metrics[:,3,3,:] ))) \
                 - sp.lm_gv(tLy, ( Hperp[:,None] * ( sp.lm_gv(tLTy, self.mesh.metrics[:,3,:]) - self.mesh.bdy_metrics[:,2,3,:] ))) \
                 + sp.lm_gv(tRz, ( Hperp[:,None] * ( sp.lm_gv(tRTz, self.mesh.metrics[:,6,:]) - self.mesh.bdy_metrics[:,5,6,:] ))) \
                 - sp.lm_gv(tLz, ( Hperp[:,None] * ( sp.lm_gv(tLTz, self.mesh.metrics[:,6,:]) - self.mesh.bdy_metrics[:,4,6,:] )))
            RHS2 = sp.lm_gv(tRx, ( Hperp[:,None] * ( sp.lm_gv(tRTx, self.mesh.metrics[:,1,:]) - self.mesh.bdy_metrics[:,1,1,:] ))) \
                 - sp.lm_gv(tLx, ( Hperp[:,None] * ( sp.lm_gv(tLTx, self.mesh.metrics[:,1,:]) - self.mesh.bdy_metrics[:,0,1,:] ))) \
                 + sp.lm_gv(tRy, ( Hperp[:,None] * ( sp.lm_gv(tRTy, self.mesh.metrics[:,4,:]) - self.mesh.bdy_metrics[:,3,4,:] ))) \
                 - sp.lm_gv(tLy, ( Hperp[:,None] * ( sp.lm_gv(tLTy, self.mesh.metrics[:,4,:]) - self.mesh.bdy_metrics[:,2,4,:] ))) \
                 + sp.lm_gv(tRz, ( Hperp[:,None] * ( sp.lm_gv(tRTz, self.mesh.metrics[:,7,:]) - self.mesh.bdy_metrics[:,5,7,:] ))) \
                 - sp.lm_gv(tLz, ( Hperp[:,None] * ( sp.lm_gv(tLTz, self.mesh.metrics[:,7,:]) - self.mesh.bdy_metrics[:,4,7,:] )))
            RHS3 = sp.lm_gv(tRx, ( Hperp[:,None] * ( sp.lm_gv(tRTx, self.mesh.metrics[:,2,:]) - self.mesh.bdy_metrics[:,1,2,:] ))) \
                 - sp.lm_gv(tLx, ( Hperp[:,None] * ( sp.lm_gv(tLTx, self.mesh.metrics[:,2,:]) - self.mesh.bdy_metrics[:,0,2,:] ))) \
                 + sp.lm_gv(tRy, ( Hperp[:,None] * ( sp.lm_gv(tRTy, self.mesh.metrics[:,5,:]) - self.mesh.bdy_metrics[:,3,5,:] ))) \
                 - sp.lm_gv(tLy, ( Hperp[:,None] * ( sp.lm_gv(tLTy, self.mesh.metrics[:,5,:]) - self.mesh.bdy_metrics[:,2,5,:] ))) \
                 + sp.lm_gv(tRz, ( Hperp[:,None] * ( sp.lm_gv(tRTz, self.mesh.metrics[:,8,:]) - self.mesh.bdy_metrics[:,5,8,:] ))) \
                 - sp.lm_gv(tLz, ( Hperp[:,None] * ( sp.lm_gv(tLTz, self.mesh.metrics[:,8,:]) - self.mesh.bdy_metrics[:,4,8,:] )))
            tot1 = LHS1 - Hinv[:,None] * RHS1
            tot2 = LHS2 - Hinv[:,None] * RHS2
            tot3 = LHS3 - Hinv[:,None] * RHS3
            
        if return_ers:
            if self.dim==2:
                if return_max_only:
                    maxval = np.max(np.nan_to_num([abs(tot1),abs(tot2)]))
                    if returnRL:
                        maxRHS = np.max(np.nan_to_num([abs(RHS1),abs(RHS2)]))
                        maxLHS = np.max(np.nan_to_num([abs(LHS1),abs(LHS2)]))
                        return maxval, maxRHS, maxLHS
                    else:
                        return maxval
                else:
                    return np.max(abs(LHS1)),np.mean(abs(LHS1)),np.max(abs(LHS2)),np.mean(abs(LHS2)),\
                            np.max(abs(RHS1)),np.mean(abs(RHS1)),np.max(abs(RHS2)),np.mean(abs(RHS2)),\
                            np.max(abs(tot1)),np.mean(abs(tot1)),np.max(abs(tot2)),np.mean(abs(tot2))
            else:
                if return_max_only:
                    maxval = np.max(np.nan_to_num([abs(tot1),abs(tot2),abs(tot3)]))
                    if returnRL:
                        maxRHS = np.max(np.nan_to_num([abs(RHS1),abs(RHS2),abs(RHS3)]))
                        maxLHS = np.max(np.nan_to_num([abs(LHS1),abs(LHS2),abs(LHS3)]))
                        return maxval, maxRHS, maxLHS
                    else:
                        return maxval
                else:
                    return np.max(abs(LHS1)),np.mean(abs(LHS1)),np.max(abs(LHS2)),np.mean(abs(LHS2)),np.max(abs(LHS3)),np.mean(abs(LHS3)),\
                            np.max(abs(RHS1)),np.mean(abs(RHS1)),np.max(abs(RHS2)),np.mean(abs(RHS2)),np.max(abs(RHS3)),np.mean(abs(RHS3)),\
                            np.max(abs(tot1)),np.mean(abs(tot1)),np.max(abs(tot2)),np.mean(abs(tot2)),np.max(abs(tot3)),np.mean(abs(tot3))
        else:
            print('Max error on LHSx =', np.max(abs(LHS1)))
            print('Avg error on LHSx =', np.mean(abs(LHS1)))
            print('Max error on RHSx =', np.max(abs(RHS1)))
            print('Avg error on RHSx =', np.mean(abs(RHS1)))
            print('Max error on total x =', np.max(abs(tot1)))
            print('Avg error on total x =', np.mean(abs(tot1)))
            print('Max error on LHSy =', np.max(abs(LHS2)))
            print('Avg error on LHSy =', np.mean(abs(LHS2)))
            print('Max error on RHSy =', np.max(abs(RHS2)))
            print('Avg error on RHSy =', np.mean(abs(RHS2)))
            print('Max error on total y =', np.max(abs(tot2)))
            print('Avg error on total y =', np.mean(abs(tot2)))
            if self.dim==3:
                print('Max error on LHSz =', np.max(abs(LHS3)))
                print('Avg error on LHSz =', np.mean(abs(LHS3)))
                print('Max error on RHSz =', np.max(abs(RHS3)))
                print('Avg error on RHSz =', np.mean(abs(RHS3)))
                print('Max error on total z =', np.max(abs(tot3)))
                print('Avg error on total z =', np.mean(abs(tot3)))
                
    def check_surf_invariants(self, returnval=False):
        maxval = 0.
        if self.dim==2:
            Hperp = np.diag(self.sbp.H)
            for phys_dir in range(2):
                if phys_dir == 0: # matrix entries for metric terms
                    term = 'x'
                    xm = 0 # l=x, m=x
                    ym = 2 # l=y, m=x
                else: 
                    term = 'y'
                    xm = 1 # l=x, m=y
                    ym = 3 # l=y, m=y
                    
                RHS = np.dot(Hperp, ( self.mesh.bdy_metrics[:,1,xm,:] - self.mesh.bdy_metrics[:,0,xm,:] \
                                     + self.mesh.bdy_metrics[:,3,ym,:] - self.mesh.bdy_metrics[:,2,ym,:] ))
                if np.max(abs(RHS))>maxval:
                    maxval = np.max(abs(RHS))
                if not returnval:
                    print('Metric Optz: '+term+' surface integral GCL constraints violated by a max of {0:.2g}'.format(np.max(abs(RHS))))
        elif self.dim==3:
            Hperp = np.diag(np.kron(self.sbp.H,self.sbp.H))
            for phys_dir in range(3):
                if phys_dir == 0: # matrix entries for metric terms
                    term = 'x'
                    xm = 0 # l=x, m=x
                    ym = 3 # l=y, m=x
                    zm = 6 # l=z, m=x
                elif phys_dir == 1: 
                    term = 'y'
                    xm = 1 # l=x, m=y
                    ym = 4 # l=y, m=y
                    zm = 7 # l=z, m=x
                else: 
                    term = 'z'
                    xm = 2 # l=x, m=z
                    ym = 5 # l=y, m=z
                    zm = 8 # l=z, m=z
                    
                RHS = np.dot(Hperp, ( self.mesh.bdy_metrics[:,1,xm,:] - self.mesh.bdy_metrics[:,0,xm,:] \
                                     + self.mesh.bdy_metrics[:,3,ym,:] - self.mesh.bdy_metrics[:,2,ym,:] \
                                     + self.mesh.bdy_metrics[:,5,zm,:] - self.mesh.bdy_metrics[:,4,zm,:] ))
                if np.max(abs(RHS))>maxval:
                    maxval = np.max(abs(RHS))
                if not returnval:
                    print('Metric Optz: '+term+' surface integral GCL constraints violated by a max of {0:.2g}'.format(np.max(abs(RHS))))
        if returnval:
            if np.isnan(maxval): maxval = 0.
            return maxval
            
    ''' temporary functions '''
    def check_chain_rule(self,theta, tau):
        eye = np.eye(self.nen)
        if self.dim==1:
            pass
        elif self.dim==2:
            Hperp = np.diag(self.sbp.H)
            Dx = np.kron(self.sbp.D, eye)
            Dy = np.kron(eye, self.sbp.D)
            tLx = np.kron(self.tL, eye)
            tRx = np.kron(self.tR, eye)
            tLy = np.kron(eye, self.tL)
            tRy = np.kron(eye, self.tR)
            tLTx = tLx.T
            tRTx = tRx.T
            tLTy = tLy.T
            tRTy = tRy.T
            Ex = tRx @ (Hperp * tRTx) - tLx @ (Hperp * tLTx)
            Ey = tRy @ (Hperp * tRTy) - tLy @ (Hperp * tLTy)
            one = np.ones(self.qshape[0])
            Hinv = np.linalg.inv(np.kron(self.sbp.H, self.sbp.H))
            LHS1 = Dx @ self.mesh.metrics[:,0,:] + Dy @ self.mesh.metrics[:,2,:]
            LHS2 = Dx @ self.mesh.metrics[:,1,:] + Dy @ self.mesh.metrics[:,3,:]    
            RHS1 = tRx @ ( Hperp * ( tRTx @ self.mesh.metrics[:,0,:] - (1 - tau) * self.mesh.bdy_metrics[:,1,0,:] )) \
                 - tLx @ ( Hperp * ( tLTx @ self.mesh.metrics[:,0,:] - (1 - tau) * self.mesh.bdy_metrics[:,0,0,:] )) \
                 + tRy @ ( Hperp * ( tRTy @ self.mesh.metrics[:,2,:] - (1 - tau) * self.mesh.bdy_metrics[:,3,2,:] )) \
                 - tLy @ ( Hperp * ( tLTy @ self.mesh.metrics[:,2,:] - (1 - tau) * self.mesh.bdy_metrics[:,2,2,:] )) \
                 - tau * fn.gm_lv(fn.gdiag_lm(self.mesh.metrics[:,0,:], Ex), one) \
                 - tau * fn.gm_lv(fn.gdiag_lm(self.mesh.metrics[:,2,:], Ey), one)
            RHS2 = tRx @ ( Hperp * ( tRTx @ self.mesh.metrics[:,1,:] - (1 - tau) * self.mesh.bdy_metrics[:,1,1,:] )) \
                 - tLx @ ( Hperp * ( tLTx @ self.mesh.metrics[:,1,:] - (1 - tau) * self.mesh.bdy_metrics[:,0,1,:] )) \
                 + tRy @ ( Hperp * ( tRTy @ self.mesh.metrics[:,3,:] - (1 - tau) * self.mesh.bdy_metrics[:,3,3,:] )) \
                 - tLy @ ( Hperp * ( tLTy @ self.mesh.metrics[:,3,:] - (1 - tau) * self.mesh.bdy_metrics[:,2,3,:] )) \
                 - tau * fn.gm_lv(fn.gdiag_lm(self.mesh.metrics[:,1,:], Ex), one) \
                 - tau * fn.gm_lv(fn.gdiag_lm(self.mesh.metrics[:,3,:], Ey), one)
            tot1 = LHS1 - theta * Hinv @ RHS1
            tot2 = LHS2 - theta * Hinv @ RHS2
            print('Max Errors in x are:', np.max(abs(tot1)))
            #print(tot1)
            print('Max Errors in y are:', np.max(abs(tot2)))
            #print(tot2)

        
    def check_cons(self,q=None,t=0.):
        ''' returns what I think is 1 @ H @ dqdt for 2D and 3D '''
        if q == None:
            q = self.diffeq.set_q0()
        dqdt = self.dqdt(q,t)
        cons = fn.sum_gv_neq(self.H_phys, dqdt, self.neq_node)
        
        Esurfxref = self.satx.tR @ np.diag(self.satx.Hperp[:,0]) @ self.satx.tRT \
            - self.satx.tL @ np.diag(self.satx.Hperp[:,0]) @ self.satx.tLT
        Esurfyref = self.saty.tR @ np.diag(self.saty.Hperp[:,0]) @ self.saty.tRT \
            - self.saty.tL @ np.diag(self.saty.Hperp[:,0]) @ self.saty.tLT
        Ex = self.diffeq.calcEx(q)
        Ey = self.diffeq.calcEy(q)
        if self.dim ==3:
            Esurfzref = self.satz.tR @ np.diag(self.satz.Hperp[:,0]) @ self.satz.tRT \
                - self.satz.tL @ np.diag(self.satz.Hperp[:,0]) @ self.satz.tLT
            Ez = self.diffeq.calcEz(q) 
        
        bdy1 = np.zeros_like(q)
        bdy2 = np.zeros_like(q)
        
        if self.dim == 2:
            Hperp = np.diag(self.sbp.H)
            for col in range(self.nelem[0]):
                # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                ExL = fn.shift_right(Ex[:,start:end])
                ExR = fn.shift_left(Ex[:,start:end])
                EyL = fn.shift_right(Ey[:,start:end])
                EyR = fn.shift_left(Ey[:,start:end])
                
                bdy1[:,start:end] += self.mesh.metrics[:,2,start:end] * (Esurfyref @ Ex[:,start:end]) \
                                    + self.mesh.metrics[:,3,start:end] * (Esurfyref @ Ey[:,start:end])
                bdy2[:,start:end] += (self.saty.tR * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,3,2,start:end] * (self.saty.tLT @ ExR)) \
                                    - (self.saty.tL * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,2,2,start:end] * (self.saty.tRT @ ExL)) \
                                    + (self.saty.tR * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,3,3,start:end] * (self.saty.tLT @ EyR)) \
                                    - (self.saty.tL * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,2,3,start:end] * (self.saty.tRT @ EyL))
                
            for row in range(self.nelem[1]):
                # starts at bottom left to bottom right, then next row up
                ExL = fn.shift_right(Ex[:,row::self.nelem[1]])
                ExR = fn.shift_left(Ex[:,row::self.nelem[1]])
                EyL = fn.shift_right(Ey[:,row::self.nelem[1]])
                EyR = fn.shift_left(Ey[:,row::self.nelem[1]])
                
                bdy1[:,row::self.nelem[1]] += self.mesh.metrics[:,0,row::self.nelem[1]] * (Esurfxref @ Ex[:,row::self.nelem[1]]) \
                        + self.mesh.metrics[:,1,row::self.nelem[1]] * (Esurfxref @ Ey[:,row::self.nelem[1]])
                bdy2[:,row::self.nelem[1]] += (self.satx.tR * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,1,0,row::self.nelem[1]] * (self.satx.tLT @ ExR)) \
                        - (self.satx.tL * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,0,0,row::self.nelem[1]] * (self.satx.tRT @ ExL)) \
                        + (self.satx.tR * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,1,1,row::self.nelem[1]] * (self.satx.tLT @ EyR)) \
                        - (self.satx.tL * Hperp[:,0]) @ (self.mesh.bdy_metrics[:,0,1,row::self.nelem[1]] * (self.satx.tRT @ EyL))

        elif self.dim == 3:
            skipx = self.nelem[1]*self.nelem[2]
            skipz = self.nelem[0]*self.nelem[1]
            for rowx in range(skipx):
                ExL = fn.shift_right(Ex[:,rowx::skipx])
                ExR = fn.shift_left(Ex[:,rowx::skipx])
                EyL = fn.shift_right(Ey[:,rowx::skipx])
                EyR = fn.shift_left(Ey[:,rowx::skipx])
                EzL = fn.shift_right(Ez[:,rowx::skipx])
                EzR = fn.shift_left(Ez[:,rowx::skipx])
                bdy1[:,rowx::skipx] += self.mesh.metrics[:,0,rowx::skipx] * (Esurfxref @ Ex[:,rowx::skipx]) \
                                    + self.mesh.metrics[:,1,rowx::skipx] * (Esurfxref @ Ey[:,rowx::skipx]) \
                                    + self.mesh.metrics[:,2,rowx::skipx] * (Esurfxref @ Ez[:,rowx::skipx])
                bdy2[:,rowx::skipx] += (self.satx.tR * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,1,0,rowx::skipx] * (self.satx.tLT @ ExR)) \
                                    - (self.satx.tL * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,0,0,rowx::skipx] * (self.satx.tRT @ ExL)) \
                                    + (self.satx.tR * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,1,1,rowx::skipx] * (self.satx.tLT @ EyR)) \
                                    - (self.satx.tL * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,0,1,rowx::skipx] * (self.satx.tRT @ EyL)) \
                                    + (self.satx.tR * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,1,2,rowx::skipx] * (self.satx.tLT @ EzR)) \
                                    - (self.satx.tL * self.satx.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,0,2,rowx::skipx] * (self.satx.tRT @ EzL))
                                    
            for coly in range(self.nelem[0]*self.nelem[2]):
                start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                end = start + skipx
                ExL = fn.shift_right(Ex[:,start:end:self.nelem[2]])
                ExR = fn.shift_left(Ex[:,start:end:self.nelem[2]])
                EyL = fn.shift_right(Ey[:,start:end:self.nelem[2]])
                EyR = fn.shift_left(Ey[:,start:end:self.nelem[2]])
                EzL = fn.shift_right(Ez[:,start:end:self.nelem[2]])
                EzR = fn.shift_left(Ez[:,start:end:self.nelem[2]])
                bdy1[:,start:end:self.nelem[2]] += self.mesh.metrics[:,3,start:end:self.nelem[2]] * (Esurfyref @ Ex[:,start:end:self.nelem[2]]) \
                                    + self.mesh.metrics[:,4,start:end:self.nelem[2]] * (Esurfyref @ Ey[:,start:end:self.nelem[2]]) \
                                    + self.mesh.metrics[:,5,start:end:self.nelem[2]] * (Esurfyref @ Ez[:,start:end:self.nelem[2]])
                bdy2[:,start:end:self.nelem[2]] += (self.saty.tR * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,3,3,start:end:self.nelem[2]] * (self.saty.tLT @ ExR)) \
                                    - (self.saty.tL * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,2,3,start:end:self.nelem[2]] * (self.saty.tRT @ ExL)) \
                                    + (self.saty.tR * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,3,4,start:end:self.nelem[2]] * (self.saty.tLT @ EyR)) \
                                    - (self.saty.tL * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,2,4,start:end:self.nelem[2]] * (self.saty.tRT @ EyL)) \
                                    + (self.saty.tR * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,3,5,start:end:self.nelem[2]] * (self.saty.tLT @ EzR)) \
                                    - (self.saty.tL * self.saty.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,2,5,start:end:self.nelem[2]] * (self.saty.tRT @ EzL))
            
            for colz in range(skipz):
                start = colz*self.nelem[2]
                end = start + self.nelem[2]
                ExL = fn.shift_right(Ex[:,start:end])
                ExR = fn.shift_left(Ex[:,start:end])
                EyL = fn.shift_right(Ey[:,start:end])
                EyR = fn.shift_left(Ey[:,start:end])
                EzL = fn.shift_right(Ez[:,start:end])
                EzR = fn.shift_left(Ez[:,start:end])
                bdy1[:,start:end] += self.mesh.metrics[:,6,start:end] * (Esurfzref @ Ex[:,start:end]) \
                                    + self.mesh.metrics[:,7,start:end] * (Esurfzref @ Ey[:,start:end]) \
                                    + self.mesh.metrics[:,8,start:end] * (Esurfzref @ Ez[:,start:end])
                bdy2[:,start:end] += (self.satz.tR * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,5,6,start:end] * (self.satz.tLT @ ExR)) \
                                    - (self.satz.tL * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,4,6,start:end] * (self.satz.tRT @ ExL)) \
                                    + (self.satz.tR * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,5,7,start:end] * (self.satz.tLT @ EyR)) \
                                    - (self.satz.tL * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,4,7,start:end] * (self.satz.tRT @ EyL)) \
                                    + (self.satz.tR * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,5,8,start:end] * (self.satz.tLT @ EzR)) \
                                    - (self.satz.tL * self.satz.Hperp[:,0]) @ (self.mesh.bdy_metrics[:,4,8,start:end] * (self.satz.tRT @ EzL)) 
            
        erbdy = -0.5*np.sum(bdy1 + bdy2)
        
        if self.dim == 2:
            Qxref = np.kron(self.sbp.Q, self.sbp.H)
            Qyref = np.kron(self.sbp.H, self.sbp.Q)
            erQT = 0.5*np.sum(self.mesh.metrics[:,0,:] * (Qxref.T @ Ex) + self.mesh.metrics[:,2,:] * (Qyref.T @ Ex) \
                         + self.mesh.metrics[:,1,:] * (Qxref.T @ Ey) + self.mesh.metrics[:,3,:] * (Qyref.T @ Ey))
            erQ_orig = -0.5*np.sum(self.mesh.metrics[:,0,:] * (Qxref @ Ex) + self.mesh.metrics[:,2,:] * (Qyref @ Ex) \
                         + self.mesh.metrics[:,1,:] * (Qxref @ Ey) + self.mesh.metrics[:,3,:] * (Qyref @ Ey))
        elif self.dim == 3:
            Qxref = np.kron(np.kron(self.sbp.Q, self.sbp.H), self.sbp.H)
            Qyref = np.kron(np.kron(self.sbp.H, self.sbp.Q), self.sbp.H)
            Qzref = np.kron(np.kron(self.sbp.H, self.sbp.H), self.sbp.Q)
            erQT = 0.5*np.sum(self.mesh.metrics[:,0,:] * (Qxref.T @ Ex) + self.mesh.metrics[:,3,:] * (Qyref.T @ Ex) + self.mesh.metrics[:,6,:] * (Qzref.T @ Ex) \
                         + self.mesh.metrics[:,1,:] * (Qxref.T @ Ey) + self.mesh.metrics[:,4,:] * (Qyref.T @ Ey) + self.mesh.metrics[:,7,:] * (Qzref.T @ Ey) \
                         + self.mesh.metrics[:,2,:] * (Qxref.T @ Ez) + self.mesh.metrics[:,5,:] * (Qyref.T @ Ez) + self.mesh.metrics[:,8,:] * (Qzref.T @ Ez))
            erQ_orig = -0.5*np.sum(self.mesh.metrics[:,0,:] * (Qxref @ Ex) + self.mesh.metrics[:,3,:] * (Qyref @ Ex) + self.mesh.metrics[:,6,:] * (Qzref @ Ex) \
                         + self.mesh.metrics[:,1,:] * (Qxref @ Ey) + self.mesh.metrics[:,4,:] * (Qyref @ Ey) + + self.mesh.metrics[:,7,:] * (Qzref @ Ey) \
                         + self.mesh.metrics[:,2,:] * (Qxref @ Ez) + self.mesh.metrics[:,5,:] * (Qyref @ Ez) + + self.mesh.metrics[:,8,:] * (Qzref @ Ez))
            
        print('Conservation in boundary 1 (er_bdyE) = {0:.4g}'.format(-0.5*np.sum(bdy1)))
        print('Conservation in boundary 2 (er_bdynum) = {0:.4g}'.format(-0.5*np.sum(bdy2)))
        print('Conservation in boundary (er_bdy) = {0:.4g}'.format(erbdy))
        print('Conservation in QT (er_QT) = {0:.4g}'.format(erQT))
        print('Conservation in Q_orig (= er_bdyE - er_QT) = {0:.5g}'.format(erQ_orig))
        print('Conservation in er_bdyE - er_QT (= Q_orig) = {0:.5g}'.format(-0.5*np.sum(bdy1) - erQT))
        print('Total estimated conservation = {0:.4g}'.format(erbdy + erQT))
        print('Actual conservation = {0:.4g}'.format(cons))
        print('Estimation off by {0:.4g}'.format(abs(cons - erbdy - erQT)))

    def zelalem_diss_coeff(self,q):
        ''' compute the Zelalem dissipation coefficients 
        NOTE: self.adiss.eps_type must have been set by init '''
        try:
            eps_type = self.adiss.eps_type # for easy access
        except:
            return 0.
        if self.dim == 1:
            if eps_type == 0 or eps_type == 1:
                return 0.
            if eps_type == 2 or eps_type == 21:
                "boundary jumps"
                v = q # TODO: could generalize this to residual or the derivative
                v_a = self.sat.lm_gv(self.sat.tLT, v)
                v_b = self.sat.lm_gv(self.sat.tRT, v)
                if not self.periodic: raise Exception('Need to implement non-periodic BCs')
                vf_L = fn.pad_1dL(v_b, v_b[:,-1]) # this is the solution to the left of the interface
                vf_R = fn.pad_1dR(v_a, v_a[:,0]) # this is the solution to the right of the interface
                vf_jump2 = (vf_R - vf_L)**2 # size (neq,nelem+1)
                vf_sum2 = (vf_R + vf_L)**2 # size (neq,nelem+1)
                denom = vf_sum2[:,1:]  + vf_sum2[:,:-1]
                if eps_type == 2:
                    # take the norm in case we have system
                    # returns shape (nelem)
                    coeff = np.linalg.norm(np.sqrt(np.divide((vf_jump2[:,1:] + vf_jump2[:,:-1]), denom,
                                                            out=np.zeros_like(denom, dtype=float), where=denom!=0)), axis=0)
                elif eps_type == 21:
                    # start with shape (neq,nelem)
                    coeff = np.sqrt(np.divide((vf_jump2[:,1:] + vf_jump2[:,:-1]), denom,
                                    out=np.zeros_like(denom, dtype=float), where=denom!=0))
                    # extend to shape (nen*neq,nelem)
                    coeff = fn.repeat_nen_gv(coeff, self.nen)
            elif eps_type == 3 or eps_type == 31:
                "difference between conservative and nonconservative forms"
                dEdx1 = self.gm_gv(self.Dx, self.diffeq.calcEx(q), self.neq_node)
                dEdx2 = self.diffeq.nonconservative_coeff(q) * self.gm_gv(self.Dx, q, self.neq_node)   
                if eps_type == 3:
                    # take the norm in case we have system
                    denom = np.linalg.norm(dEdx1 + dEdx2, axis=0)
                    coeff = np.divide(np.linalg.norm(dEdx1 - dEdx2, axis=0), denom,
                                        out=np.zeros_like(denom, dtype=float), where=denom!=0)
                else:
                    # returns shape (neq,nelem)
                    denom = fn.block_norm_gv_neq(dEdx1 + dEdx2, self.neq_node)
                    coeff = np.divide(fn.block_norm_gv_neq(dEdx1 - dEdx2, self.neq_node), denom,
                                        out=np.zeros_like(denom, dtype=float), where=denom!=0)
                    coeff = fn.repeat_nen_gv(coeff, self.nen)
            elif eps_type == 4 or eps_type == 41:
                "difference between degree p and degree p+1 flux derivatives"
                if self.Vptop1 is None:
                    # have not set up the p+1 derivative yet
                    assert self.disc_nodes in ['lgl','lg','nc'], 'ERROR: Interpolation only implemented for element-type.'
                    from Source.Disc.MakeDgOp import MakeDgOp
                    sbpp1 = MakeSbpOp(self.p+1, self.disc_nodes, 0, print_progress=False)
                    Vp1top = MakeDgOp.VandermondeLagrange1D(self.sbp.x, sbpp1.x)
                    self.Vptop1 = MakeDgOp.VandermondeLagrange1D(sbpp1.x, self.sbp.x)
                    self.Vp1top_Dp1 = Vp1top @ sbpp1.D
                    self.Dp = self.sbp.D
                    if self.sparse:
                        self.Vptop1 = sp.lm_to_sp(self.Vptop1)
                        self.Vp1top_Dp1 = sp.lm_to_sp(self.Vp1top_Dp1)
                        self.Dp = sp.lm_to_sp(self.Dp)
                # TODO: could generalize this to residual? otherwise can't use q for linear problems
                vp = self.lm_gv(self.Dp, self.diffeq.calcEx(q), self.neq_node)
                vp1 = self.lm_gv(self.Vp1top_Dp1, self.diffeq.calcEx(self.lm_gv(self.Vptop1, q, self.neq_node)), self.neq_node)
                if eps_type == 4:
                    # take the norm in case we have system
                    denom = np.linalg.norm(vp + vp1, axis=0)
                    coeff = np.divide(np.linalg.norm(vp - vp1, axis=0), denom,
                                        out=np.zeros_like(denom, dtype=float), where=denom!=0)
                else:
                    # returns shape (neq,nelem)
                    denom = fn.block_norm_gv_neq(vp + vp1, self.neq_node)
                    coeff = np.divide(fn.block_norm_gv_neq(vp - vp1, self.neq_node), denom,
                                        out=np.zeros_like(denom, dtype=float), where=denom!=0)
                    coeff = fn.repeat_nen_gv(coeff, self.nen)

            else:
                raise Exception('Desired type not implemented yet')
            #TODO: use something based on entropy flux like in "Entropy correction with SIAC filters, Picklo & Edoh"
            return coeff

        elif self.dim == 2:
            if eps_type == 4 or eps_type == 41:
                "difference between degree p and degree p+1 flux derivatives"
                if self.Vptop1 is None:
                    # have not set up the p+1 derivative yet
                    assert self.disc_nodes in ['lgl','lg','nc'], 'ERROR: Interpolation only implemented for element-type.'
                    from Source.Disc.MakeDgOp import MakeDgOp
                    sbpp1 = MakeSbpOp(self.p+1, self.disc_nodes, 0, print_progress=False)
                    Vp1top = MakeDgOp.VandermondeLagrange1D(self.sbp.x, sbpp1.x)
                    Vptop1 = MakeDgOp.VandermondeLagrange1D(sbpp1.x, self.sbp.x)
                    self.Vptop1 = np.kron(Vptop1, Vptop1)
                    self.Vp1top_Dp1xi = np.kron(Vp1top @ sbpp1.D, Vp1top)
                    self.Vp1top_Dp1eta = np.kron(Vp1top, Vp1top @ sbpp1.D)
                    self.Dpxi = np.kron(self.sbp.D, np.eye(self.nen))
                    self.Dpeta = np.kron(np.eye(self.nen), self.sbp.D)
                    if self.sparse:
                        self.Vptop1 = sp.lm_to_sp(self.Vptop1)
                        self.Vp1top_Dp1xi = sp.lm_to_sp(self.Vp1top_Dp1xi)
                        self.Vp1top_Dp1eta = sp.lm_to_sp(self.Vp1top_Dp1eta)
                        self.Dpxi = sp.lm_to_sp(self.Dpxi)
                        self.Dpeta = sp.lm_to_sp(self.Dpeta)
                # TODO: could generalize this to residual? otherwise can't use q for linear problems
                vpxi = self.lm_gv(self.Dpxi, self.diffeq.calcEx(q), self.neq_node)
                vp1xi = self.lm_gv(self.Vp1top_Dp1xi, self.diffeq.calcEx(self.lm_gv(self.Vptop1, q, self.neq_node)), self.neq_node)
                vpeta = self.lm_gv(self.Dpeta, self.diffeq.calcEy(q), self.neq_node)
                vp1eta = self.lm_gv(self.Vp1top_Dp1eta, self.diffeq.calcEy(self.lm_gv(self.Vptop1, q, self.neq_node)), self.neq_node)
                if eps_type == 4:
                    # take the norm in case we have system
                    denomxi = np.linalg.norm(vpxi + vp1xi, axis=0)
                    denometa = np.linalg.norm(vpeta + vp1eta, axis=0)
                    coeffxi = np.divide(np.linalg.norm(vpxi - vp1xi, axis=0), denomxi,
                                        out=np.zeros_like(denomxi, dtype=float), where=denomxi!=0)
                    coeffeta = np.divide(np.linalg.norm(vpeta - vp1eta, axis=0), denometa,
                                        out=np.zeros_like(denometa, dtype=float), where=denometa!=0)
                else:
                    # returns shape (neq,nelem)
                    denomxi = fn.block_norm_gv_neq(vpxi + vp1xi, self.neq_node)
                    denometa = fn.block_norm_gv_neq(vpeta + vp1eta, self.neq_node)
                    coeffxi = np.divide(fn.block_norm_gv_neq(vpxi - vp1xi, self.neq_node), denomxi,
                                        out=np.zeros_like(denomxi, dtype=float), where=denomxi!=0)
                    coeffeta = np.divide(fn.block_norm_gv_neq(vpeta - vp1eta, self.neq_node), denometa,
                                        out=np.zeros_like(denometa, dtype=float), where=denometa!=0)
                    coeffxi = fn.repeat_nen_gv(coeffxi, self.nen)
                    coeffeta = fn.repeat_nen_gv(coeffeta, self.nen)
            else:
                raise Exception('Desired type not implemented yet')

            return coeffxi, coeffeta
        elif self.dim == 3:
            raise Exception('Not implemented yet')
        