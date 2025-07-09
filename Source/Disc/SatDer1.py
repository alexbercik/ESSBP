#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:02:07 2021

@author: bercik
"""
import numpy as np
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp

class SatDer1:
    
    ##########################################################################
    ''' CENTRAL FLUXES '''
    ##########################################################################

    def base_div_1d(self, q, E, q_bdyL=None, q_bdyR=None, E_bdyL=None, E_bdyR=None):
        '''
        A non-dissipative central flux in 1D, that calls an external dissipation function.
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)

        EL = fn.shift_right(E)
        intL = self.lm_gv(self.ta, EL)
        ER = fn.shift_left(E)
        intR = self.lm_gv(self.tb, ER)
        
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
        else:
            # manually fix boundaries of EL to ensure proper boundary coupling
            if E_bdyL is None: E_bdyL = self.calcEx(q_bdyL)
            intL[:,0] = self.lm_lv(self.tL, E_bdyL)
            qf_L = fn.pad_1dL(q_b, q_bdyL)
        
        if q_bdyR is None:
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        elif np.any(q_bdyR == 'None'):
            # outflow boundary condition - do not apply a SAT here
            qf_R = fn.pad_1dR(q_a, np.zeros(self.neq_node))
            qf_L[:,-1] = 0.0
            # but careful I am doing E * flux, must also subtract this from SAT
            E_bdyR = E[-self.neq_node:,-1]
            intR[:,-1] = self.lm_lv(self.tR, E_bdyR)
        else:
            # manually fix boundaries of ER to ensure proper boundary coupling
            if E_bdyR is None: E_bdyR = self.calcEx(q_bdyR)
            intR[:,-1] = self.lm_lv(self.tR, E_bdyR)
            qf_R = fn.pad_1dR(q_a, q_bdyR)

        diss = self.coeff*self.diss(qf_L,qf_R)

        sat = 0.5*( self.lm_gv(self.Esurf, E) - intR + intL ) - diss
        return sat
    
    def central_div_1d(self, q, E, q_bdyL=None, q_bdyR=None, E_bdyL=None, E_bdyR=None):
        '''
        A non-dissipative central flux in 1D
        '''
        if q_bdyL is None: # periodic
            EL = fn.shift_right(E)
            ER = fn.shift_left(E)
            intR = self.lm_gv(self.tb, ER)
            intL = self.lm_gv(self.ta, EL)
        else:
            EL = fn.shift_right(E)
            ER = fn.shift_left(E)
            intR = self.lm_gv(self.tb, ER)
            intL = self.lm_gv(self.ta, EL)
            # manually fix boundaries of EL, ER to ensure proper boundary coupling
            if E_bdyL is None:
                E_bdyL = self.calcEx(q_bdyL)
                E_bdyR = self.calcEx(q_bdyR)
            intR[:,-1] = self.lm_lv(self.tR, E_bdyR)
            intL[:,0] = self.lm_lv(self.tL, E_bdyL)
        
# =============================================================================
#         # This is equivalent to below, but tested to be slightly slower
#         Ephys = self.metrics * E
#         EphysL = self.tLT @ Ephys
#         EphysR = self.tRT @ Ephys
#         
#         EnumL = 0.5*(self.bdy_metrics[:,0,:] * (self.tRT @ EL) + EphysL)
#         EnumR = 0.5*(self.bdy_metrics[:,1,:] * (self.tLT @ ER) + EphysR)
#         
#         sat = self.tR @ (EphysR - EnumR) - self.tL @ (EphysL - EnumL)
# =============================================================================
        
        # This is equivalent to above, but tested to be slightly faster
        sat = 0.5*( self.lm_gv(self.Esurf, E) - intR + intL )
        
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
        
# =============================================================================
#         # Option 1: equivalent to below, but tested to be slowest of 3
#         Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey
#         EphysL = self.tLT @ Ephys
#         EphysR = self.tRT @ Ephys
#         EnumL = 0.5*(self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL) \
#                    + self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL) + EphysL)
#         EnumR = 0.5*(self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR) \
#                    + self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR) + EphysR)
#         
#         sat = self.tR @ (self.Hperp * (EphysR - EnumR)) - self.tL @ (self.Hperp * (EphysL - EnumL))
# =============================================================================
        
        # Option 2: equivalent to above and below, but tested to be fastest of 3
        sat = 0.5*( #self.gm_gv(self.vol_x_mat[idx], Ex) + self.gm_gv(self.vol_y_mat[idx], Ey) 
                  - self.gm_gv(self.tbphysx[idx], ExR) - self.gm_gv(self.tbphysy[idx], EyR)
                  + self.gm_gv(self.taphysx[idx], ExL) + self.gm_gv(self.taphysy[idx], EyL))
        
# =============================================================================
#         # Option 3: equivalent to above, but tested to be slower than 2
#         Ephys = self.metrics[idx][:,0,:] * Ex + self.metrics[idx][:,1,:] * Ey
#         ExphysL = self.bdy_metrics[idx][:,0,0,:] * (self.tRT @ ExL)
#         ExphysR = self.bdy_metrics[idx][:,1,0,:] * (self.tLT @ ExR)
#         EyphysL = self.bdy_metrics[idx][:,0,1,:] * (self.tRT @ EyL)
#         EyphysR = self.bdy_metrics[idx][:,1,1,:] * (self.tLT @ EyR)
#         sat = 0.5*(self.Esurf @ Ephys - (self.tR @ (self.Hperp * (ExphysR + EyphysR)) - self.tL @ (self.Hperp * (ExphysL + EyphysL))))
# =============================================================================
        
        return sat
    
    def base_div_2d(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None, E_bdyL=None, E_bdyR=None):
        '''
        A non-dissipative central flux in 2D, that calls an external dissipation function..
        '''
        
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
        if q_bdyL is None: # periodic
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
            ExL = fn.shift_right(Ex)
            ExR = fn.shift_left(Ex)
            EyL = fn.shift_right(Ey)
            EyR = fn.shift_left(Ey)
        else:
            raise Exception('TODO: adding boundary condition.')
        
        diss = self.coeff*self.diss(qf_L,qf_R,idx)
        
        sat = 0.5*( #self.gm_gv(self.vol_x_mat[idx], Ex) + self.gm_gv(self.vol_y_mat[idx], Ey) 
                  - self.gm_gv(self.tbphysx[idx], ExR) - self.gm_gv(self.tbphysy[idx], EyR)
                  + self.gm_gv(self.taphysx[idx], ExL) + self.gm_gv(self.taphysy[idx], EyL)) - diss
        """
        # Debugging: ok
        sat2 = 0.5*( fn.gm_gv(self.vol_x_mat[idx], Ex) + fn.gm_gv(self.vol_y_mat[idx], Ey) 
                  - fn.gm_gv(self.tbphysx[idx], ExR) - fn.gm_gv(self.tbphysy[idx], EyR)
                  + fn.gm_gv(self.taphysx[idx], ExL) + fn.gm_gv(self.taphysy[idx], EyL)) - diss
        print('sat:', np.max(np.abs(sat-sat2)))
        """
        
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

        sat = 0.5*( self.gm_gv(self.vol_x_mat[idx], Ex) + self.gm_gv(self.vol_y_mat[idx], Ey) + self.gm_gv(self.vol_z_mat[idx], Ez) 
                  - self.gm_gv(self.tbphysx[idx], ExR) - self.gm_gv(self.tbphysy[idx], EyR) - self.gm_gv(self.tbphysz[idx], EzR)
                  + self.gm_gv(self.taphysx[idx], ExL) + self.gm_gv(self.taphysy[idx], EyL) + self.gm_gv(self.taphysz[idx], EzL))
        
        return sat
    

    ##########################################################################
    ''' UPWIND FLUXES ''' # note: the same as llf for scalar case without mesh warping
    # in general, these are NOT stable because of the treatment of metric terms
    ##########################################################################
    
    def upwind_div_1d(self, q, E, q_bdyL=None, q_bdyR=None, E_bdyL=None, E_bdyR=None):
        '''
        An upwind dissipative flux in 1D. self.coeff=0 turns off dissipation.
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
        qf_jump = -(qf_R - qf_L)
        
        qf_avg = self.calc_avgq(qf_L, qf_R)
        A = self.dExdq(qf_avg)            
        A_abs = self.dExdq_abs(qf_avg)
        
        # Upwinding flux
        A_upwind = (A + self.coeff*A_abs)/2
        A_downwind = (A - self.coeff*A_abs)/2
        
        sat = self.lm_gv(self.tR, fn.gbdiag_gv(A_downwind, qf_jump)[:,1:]) \
            + self.lm_gv(self.tL, fn.gbdiag_gv(A_upwind, qf_jump)[:,:-1])
        return sat

    
    def upwind_div_2d(self, q, Ex, Ey, idx, q_bdyL=None, q_bdyR=None):
        '''
        An upwind dissipative flux in 2D. self.coeff=0 turns off dissipation.
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
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
        
        qf_avg = self.calc_avgq(qf_L, qf_R)
        Ax = self.dExdq(qf_avg)            
        Ax_abs = self.dExdq_abs(qf_avg)
        Ay = self.dExdq(qf_avg)            
        Ay_abs = self.dEydq_abs(qf_avg)
        
        # Upwinding flux
        A_upwind = (Ax + self.coeff*Ax_abs)/2 * bdy_metricsx[:,None,None,:] + (Ay + self.coeff*Ay_abs)/2 * bdy_metricsy[:,None,None,:]
        A_downwind = (Ax - self.coeff*Ax_abs)/2 * bdy_metricsx[:,None,None,:] + (Ay - self.coeff*Ay_abs)/2 * bdy_metricsy[:,None,None,:]
        
        sat = self.lm_gv(self.tRHperp * fn.gbdiag_gv(A_downwind, qf_jump)[:,1:]) \
            + self.lm_gv(self.tLHperp * fn.gbdiag_gv(A_upwind, qf_jump)[:,:-1])
        return sat
    
    def upwind_div_3d(self, q, Ex, Ey, Ez, idx, q_bdyL=None, q_bdyR=None):
        '''
        An upwind dissipative flux in 3D. self.coeff=0 turns off dissipation.
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
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
        
        qf_avg = self.calc_avgq(qf_L, qf_R)
        Ax = self.dExdq(qf_avg)            
        Ax_abs = self.dExdq_abs(qf_avg)
        Ay = self.dEydq(qf_avg)            
        Ay_abs = self.dEydq_abs(qf_avg)
        Az = self.dEzdq(qf_avg)            
        Az_abs = self.dEzdq_abs(qf_avg)
        
        # Upwinding flux
        A_upwind = (Ax + self.coeff*Ax_abs)/2 * bdy_metricsx \
                 + (Ay + self.coeff*Ay_abs)/2 * bdy_metricsy \
                 + (Az + self.coeff*Az_abs)/2 * bdy_metricsz
        A_downwind = (Ax - self.coeff*Ax_abs)/2 * bdy_metricsx \
                   + (Ay - self.coeff*Ay_abs)/2 * bdy_metricsy \
                   + (Az - self.coeff*Az_abs)/2 * bdy_metricsz
        
        sat = self.lm_gv(self.tRHperp * fn.gbdiag_gv(A_downwind, qf_jump)[:,1:]) \
            + self.lm_gv(self.tLHperp * fn.gbdiag_gv(A_upwind, qf_jump)[:,:-1])
        return sat


    ##########################################################################
    ''' Hadamard Fluxes ''' 
    ##########################################################################
    
    #def base_had_1d(self, q, Fvol, q_bdyL=None, q_bdyR=None):
    def base_had_1d(self, q, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        """
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qL = fn.pad_1dL(q, q[:,-1])
        else:
            # TODO: I don't think this will be provably stable, but works for dissipation at least
            qbdy = np.repeat(q_bdyL, self.nen)
            qL = fn.pad_1dL(q, qbdy) 
        if q_bdyR is None:
            qR = fn.pad_1dR(q, q[:,0])
        else:
            # TODO: I don't think this will be provably stable, but works for dissipation at least
            qbdy = np.repeat(q_bdyR, self.nen)
            qR = fn.pad_1dL(q, qbdy) 
        
        vol = self.lm_Fvol_had_diff(self.Esurf, Fvol)
        
        Fsurf = self.build_F(qL, qR)
        if self.sparse:
            surfa = self.lmT_gm_had_diff(self.taT,Fsurf[:-1])
            surfb = self.lm_gm_had_diff(self.tb,Fsurf[1:])
        else:
            surfa = self.lm_gmT_had_diff(self.ta,Fsurf[:,:,:-1])
            surfb = self.lm_gm_had_diff(self.tb,Fsurf[:,:,1:])
        
        diss = self.coeff*self.diss(self.lm_gv(self.tRT,qL), self.lm_gv(self.tLT,qR))
        sat = vol + surfa - surfb - diss 
        """
        qa = self.lm_gv(self.tLT,q)
        qb = self.lm_gv(self.tRT,q)

        if (q_bdyL is None): #and (q_bdyR is None):
            sat = self.Fsat_diff_periodic(q)
            qL = fn.pad_1dL(qb, qb[:,-1])
            qR = fn.pad_1dR(qa, qa[:,0])
        else:
            sat = self.Fsat_diff_dirichlet(q,q_bdyL,q_bdyR)
            qL = fn.pad_1dL(qb, q_bdyL)
            qR = fn.pad_1dR(qa, q_bdyR)

        sat -= self.coeff*self.diss(qL, qR)
        return sat
    
    def base_had_2d(self, q, idx, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        """
        vol = self.gm_Fvol_had_diff(self.vol_x_mat[idx], Fxvol) \
            + self.gm_Fvol_had_diff(self.vol_y_mat[idx], Fyvol)

        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            qL = fn.pad_1dL(q, q[:,-1])
        else:
            qL = fn.pad_1dL(q, q_bdyL) # TODO: this definitely does not work (needs entire q, not extrapolation)
            raise Exception('TODO: adding boundary condition.')
        if q_bdyR is None:
            qR = fn.pad_1dR(q, q[:,0])
        else:
            qR = fn.pad_1dL(q, q_bdyR)  # TODO: this definitely does not work (needs entire q, not extrapolation)
            raise Exception('TODO: adding boundary condition.')
        
        Fsurfx, Fsurfy = self.build_F(qL, qR)
        if self.sparse:         
            surfa = self.gmT_gm_had_diff(self.taphysxT[idx],Fsurfx[:-1]) + \
                    self.gmT_gm_had_diff(self.taphysyT[idx],Fsurfy[:-1])
            surfb = self.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[1:]) + \
                    self.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[1:])
        else:         
            surfa = self.gm_gmT_had_diff(self.taphysx[idx],Fsurfx[:,:,:-1]) + \
                    self.gm_gmT_had_diff(self.taphysy[idx],Fsurfy[:,:,:-1])
            surfb = self.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[:,:,1:]) + \
                    self.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[:,:,1:])
        
        diss = self.coeff*self.diss(self.lm_gv(self.tRT,qL), self.lm_gv(self.tLT,qR), idx)
        sat = surfa - surfb - diss + vol
        """

        assert ((q_bdyL is None) and (q_bdyR is None)), 'base_had_2d SAT: Only periodic boundary conditions are implemented.'
        qa = self.lm_gv(self.tLT,q)
        qb = self.lm_gv(self.tRT,q)
        qL = fn.pad_1dL(qb, qb[:,-1])
        qR = fn.pad_1dR(qa, qa[:,0])

        sat = self.Fsat_diff_periodic(q,idx) - self.coeff*self.diss(qL, qR, idx)
        return sat
    
    def base_had_3d(self, q, Fxvol, Fyvol, Fzvol, idx, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in Hadamard Form. Then add dissipative term.
        '''
        
        vol = self.gm_Fvol_had_diff(self.vol_x_mat[idx], Fxvol) + \
              self.gm_Fvol_had_diff(self.vol_y_mat[idx], Fyvol) + \
              self.gm_Fvol_had_diff(self.vol_z_mat[idx], Fzvol)
        
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
        
        Fsurfx, Fsurfy, Fsurfz = self.build_F(qL, qR)
        
        if self.sparse:         
            surfa = self.gmT_gm_had_diff(self.taphysxT[idx],Fsurfx[:-1]) + \
                    self.gmT_gm_had_diff(self.taphysyT[idx],Fsurfy[:-1])+ \
                    self.gmT_gm_had_diff(self.taphyszT[idx],Fsurfz[:-1])
            surfb = self.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[1:]) + \
                    self.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[1:])+ \
                    self.gm_gm_had_diff(self.tbphysz[idx],Fsurfz[1:])
        else:         
            surfa = self.gm_gmT_had_diff(self.taphysx[idx],Fsurfx[:,:,:-1]) + \
                    self.gm_gmT_had_diff(self.taphysy[idx],Fsurfy[:,:,:-1])+ \
                    self.gm_gmT_had_diff(self.taphysz[idx],Fsurfz[:,:,:-1])
            surfb = self.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[:,:,1:]) + \
                    self.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[:,:,1:]) + \
                    self.gm_gm_had_diff(self.tbphysz[idx],Fsurfz[:,:,1:])
        
        diss = self.coeff*self.diss(self.lm_gv(self.tRT,qL), self.lm_gv(self.tLT,qR), idx)
        
        sat = vol + surfa - surfb - diss 
        return sat
    
    def base_generalized_had_2d(self, q, Fxvol, Fyvol, idx, q_bdyL=None, q_bdyR=None):
        '''
        The base conservative flux in a generalized Hadamard Form. Then add dissipative term.
        '''
        
        vol = self.had_alpha * (fn.gm_gm_had_diff(self.vol_x_mat[idx], Fxvol) + fn.gm_gm_had_diff(self.vol_y_mat[idx], Fyvol))
        vol2 = (1 - self.had_alpha) * self.had_beta * (fn.gm_gm_had_diff(self.vol_x_mat2[idx], Fxvol) + fn.gm_gm_had_diff(self.vol_y_mat2[idx], Fyvol))
        vol3 = (1 - self.had_alpha) * (1 - self.had_beta) * (fn.gm_gm_had_diff(self.vol_x_mat3[idx], Fxvol) + fn.gm_gm_had_diff(self.vol_y_mat3[idx], Fyvol))
        
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
        
        Fsurfx, Fsurfy = self.build_F(qL, qR)
        
        surfa = self.had_gamma * (fn.gm_gm_had_diff(self.taphysx[idx],np.transpose(Fsurfx[:,:,:-1],(1,0,2))) + \
                                  fn.gm_gm_had_diff(self.taphysy[idx],np.transpose(Fsurfy[:,:,:-1],(1,0,2))) )
        surfb = self.had_gamma * (fn.gm_gm_had_diff(self.tbphysx[idx],Fsurfx[:,:,1:]) + \
                                  fn.gm_gm_had_diff(self.tbphysy[idx],Fsurfy[:,:,1:]) )
        surfa2 = (1-self.had_gamma) * (fn.gm_gm_had_diff(self.taphysx2[idx],np.transpose(Fsurfx[:,:,:-1],(1,0,2))) + \
                                  fn.gm_gm_had_diff(self.taphysy2[idx],np.transpose(Fsurfy[:,:,:-1],(1,0,2))) )
        surfb2 = (1-self.had_gamma) * (fn.gm_gm_had_diff(self.tbphysx2[idx],Fsurfx[:,:,1:]) + \
                                  fn.gm_gm_had_diff(self.tbphysy2[idx],Fsurfy[:,:,1:]) )
        
        diss = self.coeff*self.diss(qL,qR,idx)
        
        sat = vol + vol2 + vol3 + surfa - surfb + surfa2 - surfb2 - diss
        return sat
    
    ##########################################################################
    ''' LINEARIZATIONS ''' 
    ##########################################################################

    def central_div_1d_dfdq(self, q, A, q_bdyL=None, q_bdyR=None):
        '''
        A non-dissipative central flux in 1D
        Assumes skew-symmetric form metrics.
        '''
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        # Here we work in terms of facets, starting from the left-most facet.
        # This is NOT the same as elements. i.e. qR is to the right of the
        # facet and qL is to the left of the facet, opposite of element-wise.
        if q_bdyL is None:
            # here AL is the flux-jac in the left-hand side element, AR is the element on the right
            # then intL is the coupling contribution from the left-interface to the current element (row i, column i-1)
            # and intR is the coupling contribution from the right-interface to the current element (row i, column i+1)
            AL = fn.shift_mat_right(A) # move the last elem to the first elem
            AR = fn.shift_mat_left(A) # move the first elem to the last elem

            intR = fn.lm_gm(self.tb, AR) # these should be placed in col R
            intL = fn.lm_gm(self.ta, AL) # these should be placed in col L
        else:
            AL = fn.shift_mat_right(A) # move the last elem to the first elem
            AR = fn.shift_mat_left(A) # move the first elem to the last elem
            intR = fn.lm_gm(self.tb, AR)
            intL = fn.lm_gm(self.ta, AL)
            # manually fix boundaries of AL, AR to ensure proper boundary coupling
            # TODO: assuming dirichlet boundaries (or not depending on solution)
            # else would need to fix boundary intR and intL more smartly
            intR[:,:,-1] = 0.
            intL[:,:,0] = 0.

        # Note: could also use gm_triblock_flat_periodic or gm_triblock_flat (works for 2D)
        dfdq = 0.5*(  fn.sparse_block_diag(fn.lm_gm(self.Esurf, A)) \
                    - fn.sparse_block_diag_R_1D(intR) \
                    + fn.sparse_block_diag_L_1D(intL)  )
        return dfdq
    
    def dfdq_complexstep(self, q_L, q_R, xy, eps_imag=1e-30):
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

    ##########################################################################
    ''' SPECIAL FUNCTIONS ''' 
    ##########################################################################
    
    def llf_div_1d_varcoeff(self, q, E, q_bdyL=None, q_bdyR=None):
        '''
        A Local Lax-Fridriechs dissipative flux in 1D, specific for the variable
        coefficient linear convection equation. self.coeff=0 turns off dissipation.
        '''
        extrapolate_flux = True

        sat = self.alpha * self.Esurf @ E + (1 - self.alpha) * self.a * (self.Esurf @ q)
        q_a = self.tLT @ q
        q_b = self.tRT @ q
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            # make sure boundaries have upwind SATs
            sigma = self.coeff * np.ones((1,self.nelem+1))
            sigma[0] = 1
            sigma[-1] = 1
        x_f = fn.pad_1dR(self.bdy_x[[0],:], self.bdy_x[[1],-1])
        a_f = self.afun(x_f)
        qf_jump = qf_R - qf_L
        if extrapolate_flux:
            E_a = self.tLT @ E
            E_b = self.tRT @ E
            if q_bdyL is None:
                Ef_L = fn.pad_1dL(E_b, E_b[:,-1])
                Ef_R = fn.pad_1dR(E_a, E_a[:,0])
            else:
                Ef_L = fn.pad_1dL(E_b, a_f[:,0] * q_bdyL)
                Ef_R = fn.pad_1dR(E_a, a_f[:,-1] * q_bdyR)
            f_avg = (Ef_L + Ef_R) / 2
        else:
            f_avg = a_f * (qf_L + qf_R) / 2
        numflux = f_avg - sigma * fn.cabs(a_f) * qf_jump / 2
        
        sat = sat - self.tR @ numflux[:,1:] + self.tL @ numflux[:,:-1]
        return sat
    
    def div_1d_burgers_split(self, q, E, q_bdyL=None, q_bdyR=None, extrapolate_flux=True):
        '''
        A general split form SAT for Burgers equation, treating it as a variable coefficient problem with a=u/2
        Note: the entropy conservative SAT is NOT recovered with self.split_alpha=2/3
        self.coeff=0 is conservative, self.coeff=1 is disspative
        '''        
        
        sat = self.alpha * self.lm_gv(self.Esurf, E) \
            + (1 - self.alpha) * 0.5 * q * self.lm_gv(self.Esurf, q)
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
        if q_bdyL is None:
            qf_L = fn.pad_1dL(q_b, q_b[:,-1])
            qf_R = fn.pad_1dR(q_a, q_a[:,0])
        else:
            qf_L = fn.pad_1dL(q_b, q_bdyL)
            qf_R = fn.pad_1dR(q_a, q_bdyR)
            # make sure boundaries have upwind SATs
            sigma = self.coeff * np.ones((1,self.nelem+1))
            sigma[0] = 1
            sigma[-1] = 1
        qf_jump = qf_R - qf_L
        qf_avg = self.calc_avgq(qf_L, qf_R)

        if extrapolate_flux:
            E_a = self.lm_gv(self.tLT, E)
            E_b = self.lm_gv(self.tRT, E)
            if q_bdyL is None:
                Ef_L = fn.pad_1dL(E_b, E_b[:,-1])
                Ef_R = fn.pad_1dR(E_a, E_a[:,0])
            else:
                Ef_L = fn.pad_1dL(E_b, 0.5 * q_bdyL**2)
                Ef_R = fn.pad_1dR(E_a, 0.5 * q_bdyR**2)
            f_avg = self.calc_avgq(Ef_L, Ef_R)
        else:  
            f_avg = (qf_avg)**2 / 2
        numflux = f_avg - sigma * fn.cabs(qf_avg) * qf_jump / 2
        
        sat = sat - self.lm_gv(self.tR, numflux[:,1:]) \
                  + self.lm_gv(self.tL, numflux[:,:-1])
        return sat
            
    def div_1d_burgers_es(self, q, E=None, q_bdyL=None, q_bdyR=None):
        '''
        Entropy-conservative/stable SATs for self.split_alpha=2/3 found in SBP book
        (uses extrapolation of the solution from the coupled elements)
        self.coeff=0 is conservative, self.coeff=1 is disspative
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
        if q_bdyL is None: # periodic
            q_L = fn.shift_right(q_b)
            q_R = fn.shift_left(q_a)
            sigma = self.coeff
        else:
            q_L = fn.pad_1dL(q_b, q_bdyL)[:,:-1]
            q_R = fn.pad_1dR(q_a, q_bdyR)[:,1:]
            # make sure boundaries have upwind SATs
            sigma = self.coeff * np.ones((1,self.nelem+1))
            sigma[0] = 1
            sigma[-1] = 1

        if E is None: E = q**2 / 2.
        sat = (1./6.) * ( self.lm_gv(self.tR, (4. * self.lm_gv(self.tRT, E) - q_b*q_R - q_R*q_R))
                        - self.lm_gv(self.tL, (4. * self.lm_gv(self.tLT, E) - q_a*q_L - q_L*q_L)) )
        
        if self.coeff != 0.:
            q_Rjump = q_b - q_R
            q_Ljump = q_a - q_L
            #NOTE: dividing by 2 below
            if self.maxeig_type == 'rusanov':
                q_Rlambda = np.maximum(fn.cabs(q_b), fn.cabs(q_R))
                q_Llambda = np.maximum(fn.cabs(q_a), fn.cabs(q_L))
            elif self.maxeig_type == 'lf':
                q_Rlambda = fn.cabs(q_b + q_R) / 2.
                q_Llambda = fn.cabs(q_a + q_L) / 2.
            else:
                raise Exception(f"maxeig_type {self.maxeig_type} not recognized. Try 'rusanov' or 'lf'.")
            
            sat -= 0.5*sigma*(self.lm_gv(self.tR, (q_Rlambda * q_Rjump)) + self.lm_gv(self.tL, (q_Llambda * q_Ljump)))
        
        return sat
    
    def div_1d_burgers_had(self, q, E, q_bdyL=None, q_bdyR=None):
        '''
        Entropy-conservative/stable SATs for self.split_alpha=2/3
        self.coeff=0 is conservative, self.coeff=1 is disspative
        Not the same SAT found in SBP book, but this is the SAT 
        one recovers from the hadamard formulation
        (uses extrapolation of the flux from the coupled elements)
        '''
        q_a = self.lm_gv(self.tLT, q)
        q_b = self.lm_gv(self.tRT, q)
        q2_a = self.lm_gv(self.tLT, q**2)
        q2_b = self.lm_gv(self.tRT, q**2)
        if q_bdyL is None: # periodic
            q_L = fn.shift_right(q_b)
            q_R = fn.shift_left(q_a)
            q2_L = fn.shift_right(q2_b)
            q2_R = fn.shift_left(q2_a)
        else:
            raise Exception('TODO: adding boundary condition.')

        # this is the correct form you recover from the hadamard form (sec 9.3.2 in SBP book)
        sat = (1./6.) * ( q * self.lm_gv(self.tR, q_b - q_R) + self.lm_gv(self.tR, q2_b - q2_R)
                        - q * self.lm_gv(self.tL, q_a - q_L) - self.lm_gv(self.tL, q2_a - q2_L) )
        
        # below is the volume term [ E \circ F(u,u) ] 1 you get from the Hadamard form fluxes
        #vol = (1./6.) * ( q**2 * self.tR[:] + q * (self.tR @ q_b) + self.tR @ q2_b  
        #                - q**2 * self.tL[:] - q * (self.tL @ q_a) - self.tL @ q2_a )  
        
        # below is the volume term you get from the divergence 2/3 split-form form fluxes
        #vol = (1./6.) * ( tR @ ( 2. * q2_b + q_b*q_b ) - tL @ ( 2. * q2_a + q_a*q_a ) )
        
        # below is supposedly the 1st option you get from the SBP book (sec 9.3.1), which seems to fail, as it mixes the volume term from 
        # the Hadamard formulation but the coupling terms from the coupling terms from the divergence split formulation,
        # which is just the entropy conservative 2-point flux using extrapolated q at the boundaries.
        #sat2 = (1./6.) * ( q**2 * self.tR[:] + q * (self.tR @ q_b) + self.tR @ q2_b - self.tR @ (q_b*q_b + q_b*q_R + q_R*q_R)
        #                 - q**2 * self.tL[:] - q * (self.tL @ q_a) - self.tL @ q2_a + self.tL @ (q_a*q_a + q_a*q_L + q_L*q_L) )
        
        if self.coeff != 0.:
            q_Rjump = q_b - q_R
            q_Ljump = q_a - q_L
            #NOTE: dividing by 2 below
            if self.maxeig_type == 'rusanov': # as in eq 24 of local linear stability paper
                q_Rlambda = np.maximum(fn.cabs(q_b), fn.cabs(q_R))
                q_Llambda = np.maximum(fn.cabs(q_a), fn.cabs(q_L))
            elif self.maxeig_type == 'lf':
                q_Rlambda = fn.cabs(q_b + q_R) / 2.
                q_Llambda = fn.cabs(q_a + q_L) / 2.
            else:
                raise Exception(f"maxeig_type {self.maxeig_type} not recognized. Try 'rusanov' or 'lf'.")
            
            sat -= 0.5*self.coeff*(self.lm_gv(self.tR, (q_Rlambda * q_Rjump)) + self.lm_gv(self.tL, (q_Llambda * q_Ljump)))
        
        return sat


    ##########################################################################
    ''' OLD STUFF ''' 
    ##########################################################################
    
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
        A = self.dExdq(qfacet)
        # second derivative of the flux wrt q
        dAdq = self.d2Exdq2(qfacet)

        #A_abs = self.diffeq.dExdq_abs(A) # actually just absolute value (scalar in 3d format)
        A_abs = fn.cabs(A)
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

