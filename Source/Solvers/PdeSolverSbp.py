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


class PdeSolverSbp(PdeSolver):

    def init_disc_specific(self):
        
        self.energy = self.sbp_energy
        self.conservation = self.sbp_conservation
        self.energy_der = self.sbp_energy_der
        self.conservation_der = self.sbp_conservation_der
        self.entropy = self.sbp_entropy
        
        # Construct SBP operators
        self.sbp = MakeSbpOp(self.p, self.disc_nodes, self.nen)
        self.nen = self.sbp.nn
        self.p = self.sbp.p
        self.x_op = self.sbp.x
        self.tL = self.sbp.tL.reshape((self.nen,1))
        self.tR = self.sbp.tR.reshape((self.nen,1))
        self.tLT = self.tL.T
        self.tRT = self.tR.T
        if self.dim == 1:
            self.nn = self.nelem * self.nen
            self.qshape = (self.nen * self.neq_node , self.nelem)
        elif self.dim == 2:
            self.nn = (self.nelem[0] * self.nen, self.nelem[1] * self.nen)
            self.qshape = (self.nen**2 * self.neq_node , self.nelem[0] * self.nelem[1])
            self.H_perp = np.diag(self.sbp.H)[:,None]
        elif self.dim == 3:
            self.nn = (self.nelem[0] * self.nen, self.nelem[1] * self.nen, self.nelem[2] * self.nen)
            self.qshape = (self.nen**3 * self.neq_node , self.nelem[0] * self.nelem[1] * self.nelem[2])   
            self.H_perp = np.diag(np.kron(self.sbp.H,self.sbp.H))[:,None]

        ''' Setup the mesh and apply required transformations to SBP operators '''

        self.mesh = MakeMesh(self.dim, self.xmin, self.xmax, self.nelem, self.x_op, 
                             self.settings['warp_factor'],self.settings['warp_type'])

        self.mesh.get_jac_metrics(self.sbp, self.periodic,
                        metric_method = self.settings['metric_method'], 
                        bdy_metric_method = self.settings['bdy_metric_method'],
                        use_optz_metrics = self.settings['use_optz_metrics'],
                        calc_exact_metrics = self.settings['calc_exact_metrics'],
                        optz_method = self.settings['metric_optz_method'],
                        had_metric_alpha = self.settings['had_alpha'],
                        had_metric_beta = self.settings['had_beta'])
        
        if self.dim == 1:
            self.H_phys, self.Dx_phys = self.sbp.ref_2_phys(self.mesh)
        elif self.dim == 2:
            self.H_phys, self.Dx_phys, self.Dy_phys = self.sbp.ref_2_phys(self.mesh)
        elif self.dim == 3:
            self.H_phys, self.Dx_phys, self.Dy_phys, self.Dz_phys = self.sbp.ref_2_phys(self.mesh)
        self.H_inv_phys = 1/self.H_phys


        # Apply kron products to SBP operators for nelem per node
        eye = np.eye(self.neq_node)
        self.H_phys_unkronned = self.H_phys
        self.H_phys = np.kron(self.H_phys, eye)
        self.H_inv_phys = np.kron(self.H_inv_phys, eye)
        self.tL = np.kron(self.tL, eye)
        self.tR = np.kron(self.tR, eye)
        self.tLT = np.kron(self.tLT, eye)
        self.tRT = np.kron(self.tRT, eye)

        self.Dx_phys_unkronned = self.Dx_phys
        self.Dx_phys = np.kron(self.Dx_phys, eye)
        if self.dim == 2 or self.dim == 3:
            self.Dy_phys_unkronned = self.Dy_phys
            self.Dy_phys = np.kron(self.Dy_phys, eye)
            self.H_perp = np.kron(self.H_perp, eye)
        if self.dim == 3:
            self.Dz_phys_unkronned = self.Dz_phys
            self.Dz_phys = np.kron(self.Dz_phys, eye)

        ''' Modify solver approach '''

        self.diffeq.set_mesh(self.mesh)
        if self.dim == 1:
            self.diffeq.set_sbp_op(self.H_inv_phys, self.Dx_phys) # do we need this any more?
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_1d_div
                self.dfdq = self.dfdq_1d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_1d_had
                self.dfdq = self.dfdq_1d_had  
            self.sat = Sat(self, None)
        elif self.dim == 2:
            self.diffeq.set_sbp_op(self.H_inv_phys, self.Dx_phys, self.Dy_phys)
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_2d_div
                self.dfdq = self.dfdq_2d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_2d_had
                self.dfdq = self.dfdq_2d_had  
            self.satx = Sat(self, 'x')
            self.saty = Sat(self, 'y')
        elif self.dim == 3:
            self.diffeq.set_sbp_op(self.H_inv_phys, self.Dx_phys, self.Dy_phys, self.Dz_phys)
            if self.disc_type == 'div':
                self.dqdt = self.dqdt_3d_div
                self.dfdq = self.dfdq_3d_div
            elif self.disc_type == 'had':
                self.dqdt = self.dqdt_3d_had
                self.dfdq = self.dfdq_3d_had  
            self.satx = Sat(self, 'x')
            self.saty = Sat(self, 'y')
            self.satz = Sat(self, 'z')

    def dqdt_1d_div(self, q):
        ''' the main dqdt function for divergence form in 1D '''
        E = self.diffeq.calcEx(q)
        dEdx = fn.gm_gv(self.Dx_phys, E)
        
        if self.periodic:
            sat = self.sat.calc(q,E)
        else:
            raise Exception('Not coded up yet')
        
        dqdt = - dEdx + (self.H_inv_phys * sat) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_1d_div(self, q):
        ''' the main linearized RHS function for divergence form in 1D '''
        raise Exception('Not done yet.')
        
    def dqdt_2d_div(self, q):
        ''' the main dqdt function for divergence form in 2D '''
        Ex = self.diffeq.calcEx(q)
        dExdx = fn.gm_gv(self.Dx_phys, Ex)
        Ey = self.diffeq.calcEy(q)
        dEydy = fn.gm_gv(self.Dy_phys, Ey)
        satx, saty = np.empty(self.qshape), np.empty(self.qshape)
 
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
        
        dqdt = - dExdx - dEydy + self.H_inv_phys * (satx + saty) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_2d_div(self, q):
        ''' the main linearized RHS function for divergence form in 2D '''
        raise Exception('Not done yet.')

    def dqdt_3d_div(self, q):
        ''' the main dqdt function for divergence form in 3D '''
        Ex = self.diffeq.calcEx(q)
        dExdx = fn.gm_gv(self.Dx_phys, Ex)
        Ey = self.diffeq.calcEy(q)
        dEydy = fn.gm_gv(self.Dy_phys, Ey)
        Ez = self.diffeq.calcEz(q)
        dEzdz = fn.gm_gv(self.Dz_phys, Ez)
        satx, saty, satz = np.empty(self.qshape), np.empty(self.qshape), np.empty(self.qshape)
        
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
        
        dqdt = - dExdx - dEydy - dEzdz + self.H_inv_phys * (satx + saty + satz) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_3d_div(self, q):
        ''' the main linearized RHS function for divergence form in 3D '''
        raise Exception('Not done yet.') 
        
    def dqdt_1d_had(self, q):
        ''' the main dqdt function for hadamard form in 1D '''
        Fvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ex)
        dEdx = 2*fn.gm_gm_had_diff(self.Dx_phys, Fvol)
        
        if self.periodic:
            sat = self.sat.calc(q,Fvol)
        else:
            raise Exception('Not coded up yet')
        
        dqdt = - dEdx + (self.H_inv_phys * sat) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_1d_had(self, q):
        ''' the main linearized RHS function for hadamard form in 1D '''
        raise Exception('Not done yet.')
        
    def dqdt_2d_had(self, q):
        ''' the main dqdt function for hadamard form in 2D '''
        Fxvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ex)
        Fyvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ey)
        dExdx = 2*fn.gm_gm_had_diff(self.Dx_phys, Fxvol)
        dEydy = 2*fn.gm_gm_had_diff(self.Dy_phys, Fyvol)
        satx, saty = np.empty(self.qshape), np.empty(self.qshape)
        
        if self.periodic[0]:   # x sat (in ref space) 
            for row in range(self.nelem[1]):
                # starts at bottom left to bottom right, then next row up
                satx[:,row::self.nelem[1]] = self.satx.calc(q[:,row::self.nelem[1]],Fxvol[:,:,row::self.nelem[1]],Fyvol[:,:,row::self.nelem[1]],row)      
        else:
            raise Exception('Not coded up yet')
            
        if self.periodic[1]:   # y sat (in ref space) 
            for col in range(self.nelem[0]):
                # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                saty[:,start:end] = self.saty.calc(q[:,start:end],Fxvol[:,:,start:end],Fyvol[:,:,start:end],col)
        else:
            raise Exception('Not coded up yet')
        
        dqdt = - dExdx -dEydy + (self.H_inv_phys * (satx + saty)) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_2d_had(self, q):
        ''' the main linearized RHS function for hadamard form in 2D '''
        raise Exception('Not done yet.')
        
    def dqdt_3d_had(self, q):
        ''' the main dqdt function for hadamard form in 3D '''
        Fxvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ex)
        Fyvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ey)
        Fzvol = fn.build_F_vol(q, self.neq_node, self.had_flux_Ez)
        dExdx = 2*fn.gm_gm_had_diff(self.Dx_phys, Fxvol)
        dEydy = 2*fn.gm_gm_had_diff(self.Dy_phys, Fyvol)
        dEzdz = 2*fn.gm_gm_had_diff(self.Dz_phys, Fzvol)
        satx, saty, satz, = np.empty(self.qshape), np.empty(self.qshape), np.empty(self.qshape)
        
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
        
        dqdt = - dExdx - dEydy - dEzdz + self.H_inv_phys * (satx + saty + satz) + self.diffeq.calcG(q)
        return dqdt
    
    def dfdq_3d_had(self, q):
        ''' the main linearized RHS function for hadamard form in 3D '''
        raise Exception('Not done yet.')
        
    
    ############################################################################
    '''' old functions '''

    
    def sbp_dfdq_1D(self, q):
        
        dfdq_diffeq = self.diffeq.dfdq(q)
        
        if self.bc == 'periodic':
            qpad = fn.pad_periodic_1D(q)
            # notation is: given an element q, SatL and SatR correspond to L and R of element (NOT L and R of interface)
            # SatL has derivatives with respect to q and qL, the element to L
            # SatR has derivatives with respect to q and qR, the element to R
            # note that this is slightly different notation than the interface based one used in Sat class
            dSatRdq, dSatRdqR, dSatLdqL, dSatLdq = self.sat.calc_dfdq(qpad[:,:-1], qpad[:,1:])
            dSatRdq = fn.fix_dsatL_1D(dSatRdq)
            dSatRdqR = fn.fix_dsatL_1D(dSatRdqR)
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')
        
        # global 2d matrix blocks acting on element q blocks
        dfdq_q = dfdq_diffeq + fn.gdiag_gm(self.H_inv_phys, dSatLdq + dSatRdq)
        # global 2d matrix blocks acting on element qL blocks (to the left of dfdq_q)
        # length nelem if periodic (first goes to top right, remaining under main diagonal)
        dfdq_qL = fn.gdiag_gm(self.H_inv_phys, dSatLdqL)
        # global 2d matrix blocks acting on element qR blocks (to the right of dfdq_q)
        # length nelem if periodic (last goes to bottom left, remaining above main diagonal)
        dfdq_qR = fn.gdiag_gm(self.H_inv_phys, dSatRdqR)
        
        if self.bc == 'periodic':
            dfdq = fn.gm_triblock_flat_periodic(dfdq_qL,dfdq_q,dfdq_qR)
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')        
            
        return dfdq

    def sbp_dqdt_2D_mod(self, q):
        
        dqdt_out = self.diffeq.dqdt(q)
        
        if self.bc == 'periodic':
            # notation is: given an element q, SatL and SatR correspond to L and R of element (NOT L and R of interface)
            # notation is: given an element q, SatU and SatD correspond to up and down of element (NOT up and down of interface)
            satL, satR, satD, satU = np.ones(self.qshape), np.ones(self.qshape), np.ones(self.qshape), np.ones(self.qshape)
            for col in range(self.nelem[0]):
                for nodecol in range(self.nen):
                    start = col*self.nelem[0] # starts at bottom left, then next iteration bottom row and 1 over right
                    end = start + self.nelem[1]
                    startcol = nodecol*self.nen
                    endcol = startcol + self.nen
                    qpad = fn.pad_periodic_1D(q[startcol:endcol,start:end])
                    satU_temp, satD[startcol:endcol,start:end] = self.saty.calc(qpad[:,:-1], qpad[:,1:])
                    satU[startcol:endcol,start:end] = fn.fix_satL_1D(satU_temp)
            for elemrow in range(self.nelem[1]):
                for noderow in range(self.nen):
                    qpad = fn.pad_periodic_1D(q[noderow::self.nen,elemrow::self.nelem[1]])
                    satR_temp, satL[noderow::self.nen,elemrow::self.nelem[1]] = self.satx.calc(qpad[:,:-1], qpad[:,1:])
                    satR[noderow::self.nen,elemrow::self.nelem[1]] = fn.fix_satL_1D(satR_temp)
        else:
            raise Exception('Not coded up yet for boundary SATs')
            
        dqdt_out += self.Hx_inv_phys * (satL + satR) + self.Hy_inv_phys * (satD + satU) # equivalent to fn.gv_gv

        return dqdt_out

    def sbp_dqdt_2D(self, q):
        
        dqdt_out = self.diffeq.dqdt(q)
        
        if self.bc == 'periodic':
            # notation is: given an element q, SatL and SatR correspond to L and R of element (NOT L and R of interface)
            # notation is: given an element q, SatU and SatD correspond to up and down of element (NOT up and down of interface)
            satL, satR, satD, satU = np.empty(self.qshape), np.empty(self.qshape), np.empty(self.qshape), np.empty(self.qshape)
            for col in range(self.nelem[0]):
                start = col*self.nelem[0] # starts at bottom left, then next iteration bottom row and 1 over right
                end = start + self.nelem[1]
                qpad = fn.pad_periodic_1D(q[:,start:end])
                satU_temp, satD[:,start:end] = self.saty.calc(qpad[:,:-1], qpad[:,1:])
                satU[:,start:end] = fn.fix_satL_1D(satU_temp)
            for row in range(self.nelem[1]):
                qpad = fn.pad_periodic_1D(q[:,row::self.nelem[1]])
                satR_temp, satL[:,row::self.nelem[1]] = self.satx.calc(qpad[:,:-1], qpad[:,1:])
                satR[:,row::self.nelem[1]] = fn.fix_satL_1D(satR_temp)
        else:
            raise Exception('Not coded up yet for boundary SATs')
            
        dqdt_out += self.Hx_inv_phys * (satL + satR) + self.Hy_inv_phys * (satD + satU) # equivalent to fn.gv_gv

        return dqdt_out
    
    def sbp_dfdq_2D(self, q):
        
        raise Exception('Not Coded up yet.')
        
        dfdq_diffeq = self.diffeq.dfdq(q)
        shape = dfdq_diffeq.shape
        dSatLdq, dSatRdq, dSatLdqL, dSatRdqR = np.zeros(shape), np.empty(shape), np.zeros(shape), np.empty(shape)
        dSatUdq, dSatDdq, dSatUdqU, dSatDdqD = np.zeros(shape), np.empty(shape), np.zeros(shape), np.empty(shape)
        
        if self.bc == 'periodic':
            for col in self.nelem[0]:
                start = col*self.nelem[0] # starts at bottom left, then next iteration bottom row and 1 over right
                end = start + self.nelem[1]
                qpad = fn.pad_periodic_1D(q[:,start:end])
                dSatUdq_temp, dSatUdqU_temp, dSatDdq[:,:,start:end], dSatDdqD[:,:,start:end] = self.saty.calc_dfdq(qpad[:,:-1], qpad[:,1:])
                dSatUdq[:,:,start:end] = fn.fix_dsatL_1D(dSatUdq_temp)
                dSatUdqU[:,:,start:end] = fn.fix_dsatL_1D(dSatUdqU_temp)
            for row in self.nelem[1]:
                end = row+self.nelem[0]
                qpad = fn.pad_periodic_1D(q[:,row:end:self.nelem[1]])
                dSatRdq_temp, dSatRdqR_temp, dSatLdqL[:,:,row:end:self.nelem[1]], dSatLdq[:,:,row:end:self.nelem[1]] = self.satx.calc_dfdq(qpad[:,:-1], qpad[:,1:])
                dSatRdq[:,:,row:end:self.nelem[1]] = fn.fix_dsatL_1D(dSatRdq_temp)
                dSatRdqR[:,:,row:end:self.nelem[1]] = fn.fix_dsatL_1D(dSatRdqR_temp)
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')
        
        # global 2d matrix blocks acting on element q blocks
        dfdq_q = dfdq_diffeq + fn.gdiag_gm(self.Hx_inv_phys, dSatLdq + dSatRdq) \
                             + fn.gdiag_gm(self.Hy_inv_phys, dSatDdq + dSatUdq)
        # global 2d matrix blocks acting on element qL blocks (to the left of dfdq_q)
        # length nelem if periodic (first goes to top right, remaining under main diagonal)
        dfdq_qL = fn.gdiag_gm(self.Hx_inv_phys, dSatLdqL)
        # global 2d matrix blocks acting on element qR blocks (to the right of dfdq_q)
        # length nelem if periodic (last goes to bottom left, remaining above main diagonal)
        dfdq_qR = fn.gdiag_gm(self.Hx_inv_phys, dSatRdqR)
        dfdq_qD = fn.gdiag_gm(self.Hy_inv_phys, dSatDdqD)
        dfdq_qU = fn.gdiag_gm(self.Hy_inv_phys, dSatUdqU)
        
        if self.isperiodic:
            dfdq = fn.gm_triblock_2D_flat_periodic(dfdq_qL,dfdq_q,dfdq_qR,dfdq_qD,dfdq_qU,self.nelem[1])
        else:
            raise Exception('Linearization not coded up yet for boundary SATs')        
            
        return dfdq
    
    def sbp_energy(self,q):
        ''' compute the global SBP energy of global solution vector q '''
        return np.tensordot(q, self.H_phys * q)

    def sbp_conservation(self,q):
        ''' compute the global SBP conservation of global solution vector q '''
        return np.sum(self.H_phys * q)
    
    def sbp_conservation_der(self,dqdt):
        ''' compute the derivative of the global SBP conservation . Note 
        that this is wildly inefficient as it calculates dqdt all over again. '''
        # TODO: Update the cons_obj functions to take dqdt as a function
        return np.sum(self.H_phys * dqdt)
    
    def sbp_energy_der(self,q,dqdt):
        ''' compute the derivative of the global SBP energy . Note 
        that this is wildly inefficient as it calculates dqdt all over again. '''
        # TODO: Update the cons_obj functions to take dqdt as a function
        return 2 * np.tensordot(q, self.H_phys * dqdt)
    
    def sbp_entropy(self,q):
        ''' compute the global SBP entropy of global solution vector q '''
        s = self.diffeq.entropy(q)
        return np.sum(self.H_phys_unkronned * s)


    ''' temporary functions '''
    def check_cons(self,q=None):
        ''' returns what I think is 1 @ H @ dqdt for 2D and 3D '''
        if q == None:
            q = self.diffeq.set_q0()
        dqdt = self.dqdt(q)
        cons = np.sum(self.H_phys * dqdt)
        
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
        
        bdy1 = np.zeros(self.qshape)
        bdy2 = np.zeros(self.qshape)
        
        if self.dim == 2:
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
                bdy2[:,start:end] += (self.saty.tR * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,3,2,start:end] * (self.saty.tLT @ ExR)) \
                                    - (self.saty.tL * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,2,2,start:end] * (self.saty.tRT @ ExL)) \
                                    + (self.saty.tR * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,3,3,start:end] * (self.saty.tLT @ EyR)) \
                                    - (self.saty.tL * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,2,3,start:end] * (self.saty.tRT @ EyL))
                
            for row in range(self.nelem[1]):
                # starts at bottom left to bottom right, then next row up
                ExL = fn.shift_right(Ex[:,row::self.nelem[1]])
                ExR = fn.shift_left(Ex[:,row::self.nelem[1]])
                EyL = fn.shift_right(Ey[:,row::self.nelem[1]])
                EyR = fn.shift_left(Ey[:,row::self.nelem[1]])
                
                bdy1[:,row::self.nelem[1]] += self.mesh.metrics[:,0,row::self.nelem[1]] * (Esurfxref @ Ex[:,row::self.nelem[1]]) \
                        + self.mesh.metrics[:,1,row::self.nelem[1]] * (Esurfxref @ Ey[:,row::self.nelem[1]])
                bdy2[:,row::self.nelem[1]] += (self.satx.tR * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,1,0,row::self.nelem[1]] * (self.satx.tLT @ ExR)) \
                        - (self.satx.tL * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,0,0,row::self.nelem[1]] * (self.satx.tRT @ ExL)) \
                        + (self.satx.tR * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,1,1,row::self.nelem[1]] * (self.satx.tLT @ EyR)) \
                        - (self.satx.tL * self.H_perp[:,0]) @ (self.mesh.bdy_metrics[:,0,1,row::self.nelem[1]] * (self.satx.tRT @ EyL))

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
        