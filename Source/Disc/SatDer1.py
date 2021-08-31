#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:02:07 2021

@author: bercik
"""
import numpy as np
import Source.Methods.Functions as fn
#from Source.DiffEq.Quasi1DEulerA import build_F_vol, build_F_int

class SatDer1:
    
    ##########################################################################
    ''' CENTRAL FLUXES '''
    ##########################################################################
    
    def central_div_1d(self, q, E, q_bdyL=None, q_bdyR=None):
        '''
        A non-dissipative central flux in 1D
        '''
        if q_bdyL is None: # periodic
            EL = fn.shift_right(E)
            ER = fn.shift_left(E)
        else:
            raise Exception('TODO: adding boundary condition.')
        
        Ephys = self.metrics * E
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        EnumL = 0.5*(self.bdy_metrics[:,0,:] * (self.tRT @ EL) + EphysL)
        EnumR = 0.5*(self.bdy_metrics[:,1,:] * (self.tLT @ ER) + EphysR)
        
        sat = self.tR @ (EphysR - EnumR) - self.tL @ (EphysL - EnumL)
        return sat
    
    def central_div_2d(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None):
        '''
        A non-dissipative central flux in 2D. 
        '''
        
        if q_bdyL is None: # periodic
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
        else:
            raise Exception('TODO: adding boundary condition.')
        
        Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        EnumL = 0.5*(self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL) \
                   + self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL) + EphysL)
        EnumR = 0.5*(self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR) \
                   + self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR) + EphysR)
        
        sat = self.tR @ (self.Hperp * (EphysR - EnumR)) - self.tL @ (self.Hperp * (EphysL - EnumL))
        return sat
    
    def central_div_2d_alt(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None):
        '''
        A non-dissipative central flux in 2D written in alternative form. 
        '''
        
        if q_bdyL is None: # periodic
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
        else:
            raise Exception('TODO: adding boundary condition.')
        
        Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey
        
        ExphysL = self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL)
        ExphysR = self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR)
        EyphysL = self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL)
        EyphysR = self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR)
        
        sat = 0.5*(self.Esurf @ Ephys - (self.tR @ (self.Hperp * (ExphysR + EyphysR)) - self.tL @ (self.Hperp * (ExphysL + EyphysL))))
        return sat
    
    def central_div_3d(self, q, Ex, Ey, Ez, idx, q_bdyL=None, q_bdyR=None):
        '''
        A non-dissipative central flux in 3D. 
        '''
        
        if q_bdyL is None: # periodic
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
            EzL = fn.shift_right(Ez)
            EzR = fn.shift_left(Ez)
        else:
            raise Exception('TODO: adding boundary condition.')
        
        Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey + self.metrics[idx][:,2,:] * Ez
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        EnumL = 0.5*(self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL) \
                   + self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL) \
                   + self.bdy_metrics[idx][:,0,2,:] * (self.tRT @ EzL) + EphysL)
        EnumR = 0.5*(self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR) \
                   + self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR) \
                   + self.bdy_metrics[idx][:,1,2,:] * (self.tLT @ EzR) + EphysR)
        
        sat = self.tR @ (self.Hperp * (EphysR - EnumR)) - self.tL @ (self.Hperp * (EphysL - EnumL))
        return sat
    
    ##########################################################################
    ''' LAX-FRIEDRICHS FLUXES '''
    ##########################################################################
    
    def llf_div_1d(self, q, E, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        A Local Lax-Fridriechs dissipative flux in 1D. sigma=0 turns off dissipation.
        '''
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
            EL = fn.shift_right(E)
            ER = fn.shift_left(E)
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = qf_R - qf_L
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        Ephys = self.metrics * E
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        maxeigs = self.maxeig_dEdq(qf_avg)
        dissL = np.abs(maxeigs[:,:-1] * self.bdy_metrics[:,0,:])
        dissR = np.abs(maxeigs[:,1:] * self.bdy_metrics[:,1,:])
        
        EnumL = 0.5*((self.bdy_metrics[:,0,:] * (self.tRT @ EL) + EphysL) \
                     - sigma*dissL*qf_jump[:,:-1])
        EnumR = 0.5*((self.bdy_metrics[:,1,:] * (self.tLT @ ER) + EphysR) \
                     - sigma*dissR*qf_jump[:,1:])
        
        sat = self.tR @ (EphysR - EnumR) - self.tL @ (EphysL - EnumL)
        return sat
    
    
    def llf_div_2d(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        A Local Lax-Fridriechs dissipative flux in 2D. sigma=0 turns off dissipation.
        '''
        
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = qf_R - qf_L
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        maxeigsx = self.maxeig_dExdq(qf_avg)
        maxeigsy = self.maxeig_dEydq(qf_avg)
        dissL = np.abs(maxeigsx[:,:-1] * self.bdy_metrics[idx][:,0,0,:]) \
              + np.abs(maxeigsy[:,:-1] * self.bdy_metrics[idx][:,0,1,:])
        dissR = np.abs(maxeigsx[:,1:] * self.bdy_metrics[idx][:,1,0,:]) \
              + np.abs(maxeigsy[:,1:] * self.bdy_metrics[idx][:,1,1,:])
        
        EnumL = 0.5*(self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL) \
                   + self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL) + EphysL \
                   - sigma*dissL*qf_jump[:,:-1])
        EnumR = 0.5*(self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR) \
                   + self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR) + EphysR \
                   - sigma*dissR*qf_jump[:,1:])
        
        sat = self.tR @ (self.Hperp * (EphysR - EnumR)) - self.tL @ (self.Hperp * (EphysL - EnumL))
        return sat
    
    def llf_div_3d(self, q, Ex, Ey, Ez, idx, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        A Local Lax-Fridriechs dissipative flux in 3D. sigma=0 turns off dissipation.
        '''
        
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
            EzL = fn.shift_right(Ez)
            EzR = fn.shift_left(Ez)
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = qf_R - qf_L
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey + self.metrics[idx][:,2,:] * Ez
        EphysL = self.tLT @ Ephys
        EphysR = self.tRT @ Ephys
        
        maxeigsx = self.maxeig_dExdq(qf_avg)
        maxeigsy = self.maxeig_dEydq(qf_avg)
        maxeigsz = self.maxeig_dEzdq(qf_avg)
        dissL = np.abs(maxeigsx[:,:-1] * self.bdy_metrics[idx][:,0,0,:]) \
              + np.abs(maxeigsy[:,:-1] * self.bdy_metrics[idx][:,0,1,:]) \
              + np.abs(maxeigsz[:,:-1] * self.bdy_metrics[idx][:,0,2,:])
        dissR = np.abs(maxeigsx[:,1:] * self.bdy_metrics[idx][:,1,0,:]) \
              + np.abs(maxeigsy[:,1:] * self.bdy_metrics[idx][:,1,1,:]) \
              + np.abs(maxeigsz[:,1:] * self.bdy_metrics[idx][:,1,2,:])
        
        EnumL = 0.5*(self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL) \
                   + self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL) \
                   + self.bdy_metrics[idx][:,0,2,:] * (self.tRT @ EzL) + EphysL \
                   - sigma*dissL*qf_jump[:,:-1])
        EnumR = 0.5*(self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR) \
                   + self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR) \
                   + self.bdy_metrics[idx][:,1,2,:] * (self.tLT @ EzR) + EphysR \
                   - sigma*dissR*qf_jump[:,1:])
        
        sat = self.tR @ (self.Hperp * (EphysR - EnumR)) - self.tL @ (self.Hperp * (EphysL - EnumL))
        return sat

    ##########################################################################
    ''' UPWIND FLUXES ''' # note: the same as llf for scalar case
    ##########################################################################
    
    def upwind_div_1d(self, q, E, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        An upwind dissipative flux in 1D. sigma=0 turns off dissipation.
        '''
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = -(qf_R - qf_L)
        bdy_metrics = fn.pad_1dR(self.bdy_metrics[:,0,:], self.bdy_metrics[:,1,-1])
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        A = self.dEdq(qf_avg)            
        A_abs = self.dEdq_eig_abs(A)
        
        # Upwinding flux
        A_upwind = (A + sigma*A_abs)/2 * bdy_metrics
        A_downwind = (A - sigma*A_abs)/2 * bdy_metrics
        
        sat = self.tR @ fn.gm_gv(A_downwind, qf_jump)[:,1:] \
            + self.tL @ fn.gm_gv(A_upwind, qf_jump)[:,:-1]
        return sat

    
    def upwind_div_2d(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        An upwind dissipative flux in 2D. sigma=0 turns off dissipation.
        '''
        
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = -(qf_R - qf_L)
        bdy_metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
        bdy_metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        Ax = self.dExdq(qf_avg)            
        Ax_abs = self.dExdq_eig_abs(Ax)
        Ay = self.dExdq(qf_avg)            
        Ay_abs = self.dEydq_eig_abs(Ay)
        
        # Upwinding flux
        A_upwind = (Ax + sigma*Ax_abs)/2 * bdy_metricsx + (Ay + sigma*Ay_abs)/2 * bdy_metricsy
        A_downwind = (Ax - sigma*Ax_abs)/2 * bdy_metricsx + (Ay - sigma*Ay_abs)/2 * bdy_metricsy
        
        sat = self.tR @ (self.Hperp * fn.gm_gv(A_downwind, qf_jump)[:,1:]) \
            + self.tL @ (self.Hperp * fn.gm_gv(A_upwind, qf_jump)[:,:-1])
        return sat
    
    def upwind_div_3d(self, q, Ex, Ey, Ez, idx, q_bdyL=None, q_bdyR=None, sigma=1, avg='simple'):
        '''
        An upwind dissipative flux in 3D. sigma=0 turns off dissipation.
        '''
        
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            raise Exception('TODO: adding boundary condition.')
        qf_jump = -(qf_R - qf_L)
        bdy_metricsx = fn.pad_1dR(self.bdy_metrics[idx][:,0,0,:], self.bdy_metrics[idx][:,1,0,-1])
        bdy_metricsy = fn.pad_1dR(self.bdy_metrics[idx][:,0,1,:], self.bdy_metrics[idx][:,1,1,-1])
        bdy_metricsz = fn.pad_1dR(self.bdy_metrics[idx][:,0,2,:], self.bdy_metrics[idx][:,1,2,-1])
        
        if avg=='simple': # Alternatively, a Roe average can be used
            qf_avg = (qf_L + qf_R)/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
        
        Ax = self.dExdq(qf_avg)            
        Ax_abs = self.dExdq_eig_abs(Ax)
        Ay = self.dEydq(qf_avg)            
        Ay_abs = self.dEydq_eig_abs(Ay)
        Az = self.dEzdq(qf_avg)            
        Az_abs = self.dEzdq_eig_abs(Az)
        
        # Upwinding flux
        A_upwind = (Ax + sigma*Ax_abs)/2 * bdy_metricsx \
                 + (Ay + sigma*Ay_abs)/2 * bdy_metricsy \
                 + (Az + sigma*Az_abs)/2 * bdy_metricsz
        A_downwind = (Ax - sigma*Ax_abs)/2 * bdy_metricsx \
                   + (Ay - sigma*Ay_abs)/2 * bdy_metricsy \
                   + (Az - sigma*Az_abs)/2 * bdy_metricsz
        
        sat = self.tR @ (self.Hperp * fn.gm_gv(A_downwind, qf_jump)[:,1:]) \
            + self.tL @ (self.Hperp * fn.gm_gv(A_upwind, qf_jump)[:,:-1])
        return sat


    ##########################################################################
    ''' Hadamard Fluxes ''' 
    ##########################################################################
    
    def base_had_1d(self, q, Fvol, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        vol = fn.gm_gm_had_diff(self.vol_mat, Fvol)
        
        # TODO: Modify build_F and hadamard functions to only consider non-zero entries
        
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qL = fn.pad_1dL(q, q[:,-1])
        else:
            qL = fn.pad_1dL(q, q_bdyL) # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        if q_bdyR is None:
            qR = fn.pad_1dR(q, q[:,0])
        else:
            qR = fn.pad_1dL(q, q_bdyR)  # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        
        Fsurf = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ex)
        
        surfa = fn.gm_gm_had_diff(self.taphys,np.transpose(Fsurf[:,:,:-1],(1,0,2)))
        surfb = fn.gm_gm_had_diff(self.tbphys,Fsurf[:,:,1:])
        
        diss = self.diss(qL,qR)
        
        sat = vol + surfa - surfb - diss 
        return sat
    
    def base_had_2d(self, q, Fxvol, Fyvol, idx, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        
        vol = fn.gm_gm_had_diff(self.vol_x_mat[idx], Fxvol) + fn.gm_gm_had_diff(self.vol_y_mat[idx], Fyvol)
        
        # TODO: Modify build_F and hadamard functions to only consider non-zero entries
        
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qL = fn.pad_1dL(q, q[:,-1])
        else:
            qL = fn.pad_1dL(q, q_bdyL) # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        if q_bdyR is None:
            qR = fn.pad_1dR(q, q[:,0])
        else:
            qR = fn.pad_1dL(q, q_bdyR)  # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        
        Fsurfx = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ex)
        Fsurfy = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ey)
        
        surfa = fn.gm_gm_had_diff(self.taphysx[idx],np.transpose(Fsurfx[:,:,:-1],(1,0,2))) + \
                fn.gm_gm_had_diff(self.taphysy[idx],np.transpose(Fsurfy[:,:,:-1],(1,0,2)))
        surfb = fn.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[:,:,1:]) + \
                fn.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[:,:,1:])
        
        diss = self.diss(qL,qR,idx)
        
        sat = vol + surfa - surfb - diss
        return sat
    
    def base_had_3d(self, q, Fxvol, Fyvol, Fzvol, idx, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        
        vol = fn.gm_gm_had_diff(self.vol_x_mat[idx], Fxvol) + \
              fn.gm_gm_had_diff(self.vol_y_mat[idx], Fyvol) + \
              fn.gm_gm_had_diff(self.vol_z_mat[idx], Fzvol)
        
        # TODO: Modify build_F and hadamard functions to only consider non-zero entries
        
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qL = fn.pad_1dL(q, q[:,-1])
        else:
            qL = fn.pad_1dL(q, q_bdyL) # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        if q_bdyR is None:
            qR = fn.pad_1dR(q, q[:,0])
        else:
            qR = fn.pad_1dL(q, q_bdyR)  # TODO: this definitely does not work
            raise Exception('TODO: adding boundary condition.')
        
        Fsurfx = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ex)
        Fsurfy = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ey)
        Fsurfz = fn.build_F(qL, qR, self.neq_node, self.had_flux_Ez)
        
        surfa = fn.gm_gm_had_diff(self.taphysx[idx],np.transpose(Fsurfx[:,:,:-1],(1,0,2))) + \
                fn.gm_gm_had_diff(self.taphysy[idx],np.transpose(Fsurfy[:,:,:-1],(1,0,2))) + \
                fn.gm_gm_had_diff(self.taphysz[idx],np.transpose(Fsurfz[:,:,:-1],(1,0,2)))
        surfb = fn.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[:,:,1:]) + \
                fn.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[:,:,1:]) + \
                fn.gm_gm_had_diff(self.tbphysz[idx],Fsurfz[:,:,1:])
        
        diss = self.diss(qL,qR,idx)
        
        sat = vol + surfa - surfb - diss 
        return sat


    ##########################################################################
    ''' OLD STUFF ''' 
    ##########################################################################

    def der1_upwind(self, q_L, q_R, sigma=1, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the upwind or central SAT for a first derivative term such as
        \frac{dE}{dx}) where E can be nonlinear
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        xy : str, 'x' or 'y'
            Determines whether to use SAT in x or y direction
        sigma: float (default=1)
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT
        Returns
        -------
        satL : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the left.
        satR : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the right.
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        if avg=='simple':
            qfacet = (q_fL + q_fR)/2 # Alternatively, a Roe average can be used
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
            
        qf_diff = q_fL - q_fR

        A = self.dEdq(qfacet)            
        A_abs = self.dEdq_eig_abs(A)
        
        # Upwinding flux
        A_upwind = (A + sigma*A_abs)/2
        A_downwind = (A - sigma*A_abs)/2

        satL = self.tR @ fn.gm_gv(A_downwind, qf_diff)  # SAT for the left of the interface
        satR = self.tL @ fn.gm_gv(A_upwind, qf_diff)    # SAT for the right of the interface

        return satL, satR
    
    def dfdq_der1_upwind_scalar(self, q_L, q_R, xy, sigma=1, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the derivative of the upwind SAT for a scalar equation.
        Used for implicit time marching.
        Parameters
        ----------
        q_fL : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        xy : str, 'x' or 'y'
            Determines whether to use SAT in x or y direction
        sigma: float (default=1)
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT
        Returns
        -------
        Given an interface, SatL and SatR are the contributions to either side
        qL are the solutions (NOT extrapolated) on either side. Therefore:
            dSatLdqL : derivative of the SAT in the left element wrt left solution
            dSatLdqR : derivative of the SAT in the left element wrt right solution
            dSatRdqL : derivative of the SAT in the right element wrt left solution
            dSatRdqR : derivative of the SAT in the right element wrt right solution
 
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        if avg=='simple':
            qfacet = (q_fL + q_fR)/2 # Alternatively, a Roe average can be used
            # derivative of qfacet wrt qL and qR
            dqfacetdqL = self.tRT / 2
            dqfacetdqR = self.tLT / 2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')

        qfacet = (q_fL + q_fR)/2 # Assumes simple averaging, will need to be modified otherwise
        qf_diff = q_fL - q_fR

        # First derivative of the flux wrt q
        A = self.dEdq(qfacet)
        # second derivative of the flux wrt q
        dAdq = self.d2Edq2(qfacet)

        #A_abs = self.diffeq.dEdq_eig_abs(A) # actually just absolute value (scalar in 3d format)
        A_abs = abs(A)
        sign_A = np.sign(A)
        
        
        # derivative of q_diff wrt qL and qR
        dqf_diffdqL = self.tRT
        dqf_diffdqR = - self.tLT
        
        factor_qL = fn.gm_lm(dAdq,dqfacetdqL)*qf_diff
        factor_qR = fn.gm_lm(dAdq,dqfacetdqR)*qf_diff
        sigmasignA = sigma*sign_A
        sigmaA_abs = sigma*A_abs
        psigsignA = 1+sigmasignA
        msigsignA = 1-sigmasignA
        ApsigmaA_abs = A + sigmaA_abs
        AmsigmaA_abs = A - sigmaA_abs
        # these do the same thing, but the second is a bit quicker (order of gm_lm needs to be fixed)
        # for derivation of below, see personal notes
        #dSatRdqL = 0.5*fn.lm_gm(self.tL, factor_qL + fn.gm_lm(A, dqf_diffdqL) + sigma*(sign_A*factor_qL + fn.gm_lm(A_abs, dqf_diffdqL)))
        #dSatRdqR = 0.5*fn.lm_gm(self.tL, factor_qR + fn.gm_lm(A, dqf_diffdqR) + sigma*(sign_A*factor_qR + fn.gm_lm(A_abs, dqf_diffdqR)))
        #dSatLdqL = 0.5*fn.lm_gm(self.tR, factor_qL + fn.gm_lm(A, dqf_diffdqL) - sigma*(sign_A*factor_qL + fn.gm_lm(A_abs, dqf_diffdqL)))
        #dSatLdqR = 0.5*fn.lm_gm(self.tR, factor_qR + fn.gm_lm(A, dqf_diffdqR) - sigma*(sign_A*factor_qR + fn.gm_lm(A_abs, dqf_diffdqR)))       
        dSatRdqL = 0.5*fn.lm_gm(self.tL,psigsignA*factor_qL + fn.gm_lm(ApsigmaA_abs, dqf_diffdqL))
        dSatRdqR = 0.5*fn.lm_gm(self.tL,psigsignA*factor_qR + fn.gm_lm(ApsigmaA_abs,dqf_diffdqR))
        dSatLdqL = 0.5*fn.lm_gm(self.tR,msigsignA*factor_qL + fn.gm_lm(AmsigmaA_abs, dqf_diffdqL))
        dSatLdqR = 0.5*fn.lm_gm(self.tR,msigsignA*factor_qR + fn.gm_lm(AmsigmaA_abs, dqf_diffdqR))

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def dfdq_der1_complexstep(self, q_L, q_R, xy, eps_imag=1e-30):
        '''
        Purpose
        ----------
        Calculates the derivative of the SATs using complex step. This is
        required for PDEs that are systems and nonlinear, such as the Euler eq.
        This function only works to calculate the derivatives of SATs for
        first derivatives but could be modified to handle higher derivatives.
        More details are given below.
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element to the facet.
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element to the facet.
        xy : str, 'x' or 'y'
            Determines whether to use SAT in x or y direction
        eps_imag : float, optional
            Size of the complex step
        Returns
        -------
        The derivative of the SAT contribution to the elements on both sides
        of the interface. shapes (nen*neq_node,nen*neq_node,nelem)
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        neq_node,nelem = q_fL.shape
        assert neq_node == self.neq_node,'neq_node does not match'
        satL_pert_qL = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satR_pert_qL = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satL_pert_qR = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satR_pert_qR = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        
        for neq in range(neq_node):
            pert = np.zeros((self.neq_node,nelem),dtype=complex)
            pert[neq,:] = eps_imag * 1j
            satL_pert_qL[:,neq,:], satR_pert_qL[:,neq,:] = self.calc(q_fL + pert, q_fR, xy)
            satL_pert_qR[:,neq,:], satR_pert_qR[:,neq,:] = self.calc(q_fL, q_fR + pert, xy)
        
        dSatLdqL = fn.gm_lm(np.imag(satL_pert_qL) / eps_imag , self.tRT)
        dSatLdqR = fn.gm_lm(np.imag(satL_pert_qR) / eps_imag , self.tLT)
        dSatRdqL = fn.gm_lm(np.imag(satR_pert_qL) / eps_imag , self.tRT)
        dSatRdqR = fn.gm_lm(np.imag(satR_pert_qR) / eps_imag , self.tLT)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def der1_burgers_ec(self, q_L, q_R, xy):
        '''
        Purpose
        ----------
        Calculate the entropy conservative SAT for Burgers equation
        
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).

        Returns
        -------
        satL : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the left.
        satR : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the right.
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        qLqR = q_fL * q_fR
        qL2 = q_fL**2
        qR2 = q_fR**2

        satL = self.tR @ (qL2/3 - qLqR/6 - qR2/6)    # SAT for the left of the interface
        satR = self.tL @ (-qR2/3 + qLqR/6 + qL2/6)    # SAT for the right of the interface

        return satL, satR

    def dfdq_der1_burgers_ec(self, q_L, q_R, xy):
        '''
        Purpose
        ----------
        Calculate the derivative of the entropy conservative SAT for Burgers equation.

        Parameters
        ----------
        q_fL : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).

        Returns
        -------
        Given an interface, SatL and SatR are the contributions to either side
        qL are the solutions (NOT extrapolated) on either side. Therefore:
            dSatLdqL : derivative of the SAT in the left element wrt left solution
            dSatLdqR : derivative of the SAT in the left element wrt right solution
            dSatRdqL : derivative of the SAT in the right element wrt left solution
            dSatRdqR : derivative of the SAT in the right element wrt right solution
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)

        dSatLdqL = fn.gs_lm((4*q_fL - q_fR)/6, self.tR @ self.tRT)
        dSatLdqR = fn.gs_lm(-(q_fL + 2*q_fR)/6, self.tR @ self.tLT)
        dSatRdqL = fn.gs_lm((q_fR + 2*q_fL)/6, self.tL @ self.tRT)
        dSatRdqR = fn.gs_lm((q_fL - 4*q_fR)/6, self.tL @ self.tLT)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def der1_crean_ec(self, q_L, q_R):
        '''
        Purpose
        ----------
        Calculate the SATs for the entropy consistent scheme by Crean et al 2018
        NOTE: ONLY WORKS FOR ELEMENTS WITH BOUNDARY NODES! Should use more
        general matrix formulations for other cases. See notes.
        
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).

        Returns
        -------
        satL : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the left.
        satR : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the right.
        '''
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        #TODO: Rework Ismail_Roe to accept shapes (neq_node,nelem) rather than (neq_node,)
        neq,nelem = q_fL.shape
        numflux = np.zeros((neq,nelem))
        for e in range(nelem):
            numflux[:,e] = self.ec_flux(q_fL[:,e], q_fR[:,e])
        
        satL = self.tR @ ( self.calcE(q_fL) - numflux )
        satR = self.tR @ ( numflux - self.calcE(q_fR) )
        
        #F_vol = build_F_vol(q, self.neq_node, self.diffeq.ec_flux)
        #build_F_int(q1, q2, neq, ec_flux)
        #build_F_vol(q, neq, ec_flux)

        return satL, satR   
    
    def der1_crean_es(self, qL, qR):
        '''
        Purpose
        ----------
        Calculate the SATs for the entropy dissipative scheme by Crean et al 2018
        NOTE: ONLY WORKS FOR ELEMENTS WITH BOUNDARY NODES! Should use more
        general matrix formulations for other cases. See notes.
        
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).

        Returns
        -------
        satL : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the left.
        satR : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the right.
        '''
        q_fL, q_fR = self.get_interface_sol(qL, qR)
        
        #TODO: Rework Ismail_Roe to accept shapes (neq_node,nelem) rather than (neq_node,)
        neq,nelem = q_fL.shape
        numflux = np.zeros((neq,nelem))
        for e in range(nelem):
            numflux[:,e] = self.ec_flux(q_fL[:,e], q_fR[:,e])
        
        qfacet = (q_fL + q_fR)/2 # Assumes simple averaging, can generalize
        # TODO: This will get all fucked up by svec
        # TODO: Move all of these smaller things to diffeq?
        rhoL, rhouL, eL = self.diffeq.decompose_q(qL)
        uL = rhouL / rhoL
        pL = (self.diffeq.g-1)*(eL - (rhoL * uL**2)/2)
        aL = np.sqrt(self.diffeq.g * pL/rhoL)
        rhoR, rhouR, eR = self.diffeq.decompose_q(qR)
        uR = rhouR / rhoR
        pR = (self.diffeq.g-1)*(eR - (rhoR * uR**2)/2)
        aR = np.sqrt(self.diffeq.g * pR/rhoR)
        LF_const = np.max([np.abs(uL)+aL,np.abs(uR)+aR],axis=(0,1))
        Lambda = self.diffeq.dqdw(qfacet)*LF_const
        
        w_fL= self.tRT @ self.diffeq.entropy_var(qL) 
        w_fR= self.tLT @ self.diffeq.entropy_var(qR)
        
        satL = self.tR @ ( self.diffeq.calcE(q_fL) - numflux - fn.gm_gv(Lambda, w_fL - w_fR))
        satR = self.tR @ ( numflux - self.diffeq.calcE(q_fR) - fn.gm_gv(Lambda, w_fR - w_fL))

        return satL, satR 