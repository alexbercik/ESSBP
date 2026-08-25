#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:42:05 2020
@author: bercik
"""

# Check if this is being run on SciNet
from sys import platform
if platform == "linux" or platform == "linux2": # True if on SciNet
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
rc('text', usetex=True)


import numpy as np
from os import path


from Source.Disc.FiniteDiff import FiniteDiff
from Source.Disc.MakeDgOp import MakeDgOp
from Source.Disc.MakeMesh import MakeMesh
import Source.Methods.Functions as fn
#import quadpy as qp

'''
The classes in this file are inheritated by the classes other PDEs.
PdeBase:
    -Provides an init function for the parameter and the names of the
    objectives functions.
    -Introduces various functions that should not be called directly.
    -Sets parameters for the initial solution (q0)
    -Provides an updated init file with more inputs
    -Provides functions required for all PDEs such as set_xy, set_q0
    -Has a function to plot the solution
    -Sets default methods to calculate dExdx as well as the source term and
    its derivative
The PDEs are solved in this form:
    The Diffeq:             dqdt = -dExdx + G + diss = f
    Time marching methods:  dqdt = f(q) = rhs(q)
    Linearization:          f(q) = dfdq @ q 
        * note: dfdq \neq dExdq , analytic flux jacobian
    Implicit Euler:         dq = q^{n+1} - q^{n}
                            dq = h*f(q^{n+1}) = h*(f(q^{n}) + dfdq(q^{n})@dq) + O(h^2)
        * note: of course if f(q) is linear (i.e. dfdq is constant), this time 
                linearization is unecessary and above trivially reduces to 
                dq = h*f(q^{n+1}) = h*(f(q^{n}) + f(q^{n+1}) - f(q^{n})) = h*(f(q^{n+1})
    
'''

class PdeBase:

    # Diffeq info
    x = None                # Initiate 1D nodal coordinates
    xy = None               # Initiate 2D nodal coordinates
    dim = None              # No. of dimensions
    has_exa_sol = False     # True if there is an exact solution for the DiffEq
    nn = None               # No. of nodes for the spatial discretization
    nen = None              # No. of nodes per element
    nelem = None            # No. of elements
    neq_node = None         # No. of equations per node
    xmin_fix = None         # If we need to ensure xmin is a certain value
    xmax_fix = None         # If we need to ensure xmax is a certain value
    steady = False          # Whether or not steady or transient flow
    check_resid_conv = False # Whether to check for residual convergence to stop sim
    nondimensionalize = False
    enforce_positivity = False

    # Ploting options
    plt_fig_size = (6,4)
    plt_style_exa_sol = {'color':'r','linestyle':'-','linewidth':2,'marker':''}
    plt_style_sol = [{'color':'b','linestyle':'-','linewidth':2,'marker':''},
                     {'color':'k','linestyle':'--','linewidth':2,'marker':''},
                     {'color':'r','linestyle':'-.','linewidth':2,'marker':''},
                     {'color':'g','linestyle':':','linewidth':2,'marker':''}]
    plt_label_font_size = 15
    var2plot_name = None
    plt_mesh_settings = {'label lines': True,   # if True, x and y ticks are based on grid lines
                         'plot nodes': True,    # whether or not to display the nodes
                         'node size': 4,        # markersize used on nodes
                         'node color': 'black'} # marker colour used for nodes
    plt_contour_settings = {'levels': 100,          # number of distinct contours
                            'cmap': 'inferno', #'jet'}  # colourmap
                            'rotation': 90,
                            'cbar_nticks': None,
                            'cbar_font_size': 12,
                            'cbar_tick_size': 12} 

    # Parameters for the initial solution
    q0_max_q = 1.0                 # Max value in the vector q0
    q0_gauss_wave_val_bc = np.exp(-625/32) #1e-10    # Value at the boundary for Gauss wave (now matches sbp_book)

    # Discrete initial condition storage (for interpolation-based set_q0)
    _q0_interpolator = None        # Interpolation function (UnivariateSpline)

    def __init__(self, para, q0_type=None):
        '''
        Parameters
        ----------
        para : np array or float
            Parameters of the differential equation
        q0_type : str
            The type of initial solution for the DiffEq.
        '''

        ''' Add inputs to the class '''

        self.para = para
        if not hasattr(self, 'para_fix'): self.para_fix = None
        self.q0_type = q0_type
        if self.q0_type == None:
            print("WARNING: No default q0_type given. Defaulting to 'gausswave'.")
            self.q0_type = 'gausswave'


        ''' Modify type for inputs '''

        # Make sure that para is stored as a numpy array
        if isinstance(self.para, int) or isinstance(self.para, float):
            self.para = np.atleast_1d(np.asarray(self.para))
            
        
    def var2plot(self,q,var2plot_name=None):
        ''' base method, only important for systems where this is redefined '''
        return q

    def set_mesh(self, mesh, H):
        '''
        Purpose
        ----------
        Needed to calculate the initial solution and to calculate source terms
        '''

        self.mesh = mesh
        self.H = H

        ''' Extract other parameters '''
        assert self.dim == self.mesh.dim,'Dimensions of DiffEq and Solver do not match.'
        if self.dim == 1:
            self.x = self.mesh.x
            self.x_elem = self.mesh.x_elem
        elif self.dim == 2:
            self.xy = self.mesh.xy
            self.xy_elem = self.mesh.xy_elem
        elif self.dim == 3:
            self.xyz = self.mesh.xyz
            self.xyz_elem = self.mesh.xyz_elem
        self.x_ref = self.mesh.x_op
        #self.dx = self.mesh.dx
        self.xmin = self.mesh.xmin
        self.xmax = self.mesh.xmax
        if self.xmin_fix is not None:
            assert ((self.xmin == self.xmin_fix) and (self.xmax == self.xmax_fix)),\
                    "xmin and xmax do not match required values. {0} ≠ {1} , {2} ≠ {3}".format(self.xmin,self.xmin_fix,self.xmax,self.xmax_fix)
        self.dom_len = self.mesh.dom_len
        self.nn = self.mesh.nn
        self.nelem = self.mesh.nelem
        self.nen = self.mesh.nen
        if self.dim == 1:
            self.qshape = (self.nen*self.neq_node,self.nelem)
        elif self.dim == 2:
            self.qshape = ((self.nen**2)*self.neq_node,self.nelem[0]*self.nelem[1])
        elif self.dim == 3:
            self.qshape = ((self.nen**3)*self.neq_node,self.nelem[0]*self.nelem[1]*self.nelem[2])

    def _eval_gassnersinwave_projected_at_x(self, q_mesh, xy_query):
        '''
        For ``*_coarse*`` Gassner IC only: evaluate q_mesh (LG→solution projection) at arbitrary x.
        '''
        assert self.dim == 1
        if not hasattr(self, 'mesh') or self.mesh is None:
            raise RuntimeError('set_mesh must be called before evaluating the projected Gassner IC at arbitrary x.')
        if q_mesh.shape != self.qshape:
            raise ValueError(f'q_mesh shape {q_mesh.shape} does not match qshape {self.qshape}.')

        from scipy.interpolate import BarycentricInterpolator
        from scipy.optimize import brentq

        def lag1d(nodes, values, xi_):
            return float(BarycentricInterpolator(nodes, values)(xi_))

        xy_query = np.asarray(xy_query, dtype=float)
        orig_shape = xy_query.shape
        xflat = xy_query.ravel()
        out = np.empty_like(xflat, dtype=float)

        verts = np.asarray(self.mesh.vertices, dtype=float)
        ne = int(self.nelem)
        elen = float(verts[1] - verts[0])
        xr = np.asarray(self.x_ref, dtype=float).ravel()
        dom = float(self.dom_len)
        x_template = verts[:ne] + elen * xr[:, np.newaxis]
        affine_ok = np.allclose(self.x_elem, x_template, rtol=0.0, atol=1e-10 * max(1.0, abs(self.xmax - self.xmin)))
        xi_lo, xi_hi = float(xr.min()), float(xr.max())

        for ii, xraw in enumerate(xflat):
            xq = np.mod(xraw - self.xmin, dom) + self.xmin
            e = int(np.searchsorted(verts, xq, side='right') - 1)
            e = max(0, min(ne - 1, e))
            qe = q_mesh[:, e]
            xe = self.x_elem[:, e]
            if affine_ok:
                xi = (xq - verts[e]) / elen
            else:
                try:
                    xi = brentq(lambda z: lag1d(xr, xe, z) - xq, xi_lo, xi_hi, xtol=1e-14, rtol=1e-14)
                except ValueError:
                    raise ValueError(
                        f'Could not invert reference coordinate for x={xq} in element {e}. '
                        'Try a non-warped mesh or check coordinates.'
                    ) from None
            out[ii] = lag1d(xr, qe, xi)

        return out.reshape(orig_shape)

    def set_q0(self, q0_type=None, xy=None, **kwargs):
        '''
        Parameters
        ----------
        q0_type : str, optional
            Indiactes the type of initial solution.
            The default is None, in which case the default q0_type for the
            DiffEq is used.
        xy : np array, optional
            If not provided then the xy provided from self.mesh is used. One
            reason to provide the input xy is to calculate the exact solution
            at a later time, this is done for the linear convection eq.
            The default is None.
        Returns
        -------
        q0 : np array of floats
            The initial solution at the nodes xy.
        NOTE: These options here only return scalar initial solutions.
        '''

        if q0_type is None:
            q0_type = self.q0_type
        q0_type = q0_type.lower()

        if xy is None:
            qshape = self.qshape
            if self.dim == 1:
                xy = self.x_elem # shape (self.nen, self.nelem)
            elif self.dim == 2:
                xy = self.xy_elem # shape (self.nen**2, 2, self.nelem[0]*self.nelem[1])
            elif self.dim == 3:
                xy = self.xyz_elem
        else:
            #if q0_type in ('gassnersinwave','gassnersinwave_coarse'):
            #    raise Exception('This q0_type does not work with xy provided.')
            qshape = np.shape(xy)

        if q0_type == 'gausswave' or q0_type == 'gausswave_0.25' or q0_type == 'gausswave_shift' or q0_type == 'gausswave_smallshift':
            if self.dim == 1:
                if q0_type == 'gausswave_0.25':
                    mid_point = 0.5*(self.xmax + self.xmin)
                    k = (16*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
                    xmod = np.mod((xy - self.xmin) + 0.5*mid_point, self.dom_len) + self.xmin
                else:
                    mid_point = 0.5*(self.xmax + self.xmin) # mean
                    k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
                    xmod = xy
                stdev2 = abs(self.dom_len**2/k) # standard deviation squared
                exp = -0.5*(xmod-mid_point)**2/stdev2
                q0 = self.q0_max_q * np.exp(exp)
            elif self.dim == 2:
                mid_pointx = 0.5*(self.xmax[0] + self.xmin[0]) # mean
                mid_pointy = 0.5*(self.xmax[1] + self.xmin[1])
                k = 8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q)
                stdev2x = abs(self.dom_len[0]**2/k) # standard deviation squared
                stdev2y = abs(self.dom_len[1]**2/k)
                exp = -0.5*((xy[:,0,:]-mid_pointx)**2/stdev2x + (xy[:,1,:]-mid_pointy)**2/stdev2y)
                q0 = self.q0_max_q * np.exp(exp)
            elif self.dim == 3:
                mid_pointx = 0.5*(self.xmax[0] + self.xmin[0]) # mean
                mid_pointy = 0.5*(self.xmax[1] + self.xmin[1])
                mid_pointz = 0.5*(self.xmax[2] + self.xmin[2])
                k = 8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q)
                stdev2x = abs(self.dom_len[0]**2/k) # standard deviation squared
                stdev2y = abs(self.dom_len[1]**2/k)
                stdev2z = abs(self.dom_len[2]**2/k)
                exp = -0.5*((xy[:,0,:]-mid_pointx)**2/stdev2x + (xy[:,1,:]-mid_pointy)**2/stdev2y + (xy[:,2,:]-mid_pointz)**2/stdev2z)
                q0 = self.q0_max_q * np.exp(exp) 
            if 'smallshift' in q0_type: q0 = q0 + 0.1
            elif 'shift' in q0_type: q0 = q0 + 0.5
        elif q0_type == 'gausswave_shift':
            assert self.dim==1,'only for dim=1'
            assert (self.xmax==1 and self.xmin==0)
            stdev2 = 0.08**2
            exp = -0.5*(xy-0.5)**2/stdev2
            q0 = np.exp(exp) + 0.5 #- 0.5
        elif q0_type == 'gausswave_dispersed_shift':
            assert self.dim==1,'only for dim=1'
            assert (self.xmax==1 and self.xmin==0)
            # fitted "numerical-solution-like" analytic guess (Gaussian + localized cosine ringing)
            b     = 0.49601635
            A     = 1.01882765
            x0    = 0.50655031
            sig   = 0.07309955
            eps   = -0.09054338
            sig2  = 0.21394410
            k     = 26.68566281
            phi   = -0.76228906
            q0 = b + A*np.exp(-0.5*((xy-x0)/sig)**2) + eps*np.exp(-0.5*((xy-x0)/sig2)**2)*np.cos(k*(xy-x0) + phi)
        elif q0_type == 'morlet_wavelet' or q0_type == 'morlet_wavelet_shift':
            assert self.dim==1,'only for dim=1'
            k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
            mid_point = 0.5*(self.xmax + self.xmin) # mean
            stdev2 = abs(self.dom_len**2/k) # standard deviation squared
            exp = -0.5*(xy-mid_point)**2/stdev2
            gauss = self.q0_max_q * np.exp(exp)
            k_pi = 10 # how many oscillations to include in the domain
            cos = np.cos(k_pi*2.*np.pi*(xy-mid_point)/self.dom_len)
            q0 = gauss * cos
            if 'shift' in q0_type: q0 = q0 + 1.0
        elif q0_type == 'gausswave_1d' or 'gausswave_debug' in q0_type:
            if 'y' in q0_type: xyz = 1
            elif 'z' in q0_type: xyz = 2
            else: xyz = 0
            assert self.dim!=1,'This is meant to be used in 2D or 3D to mimic a 1D problem in x, y, or z.'
            k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
            mid_pointx = 0.5*(self.xmax[xyz] + self.xmin[xyz]) # mean
            stdev2 = abs(self.dom_len[xyz]**2/k) # standard deviation squared
            exp = -0.5*((xy[:,xyz,:]-mid_pointx)**2/stdev2)
            q0 = self.q0_max_q * np.exp(exp)  
        elif 'gausswave' in q0_type and 'skew' in q0_type:
            assert self.dim == 1,'skew gausswave only works for dim = 1.'
            from scipy.special import erf
            mu = 0.5
            sigma = 0.08
            stdev2 = sigma**2
            exp = -0.5*(xy-mu)**2/stdev2
            gaussian = np.exp(exp)
            # alpha: skewness parameter (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
            alpha = 1.5
            skew = 0.5 * (1 + erf(alpha * (xy - mu) / (sigma * np.sqrt(2))))
            q0 = gaussian * skew
            if 'shift' in q0_type: q0 = q0 - 0.5
        elif q0_type == 'squarewave' or q0_type == 'squarewave_shift': 
            assert self.dim == 1,'square wave only works for dim = 1.' 
            dom_len = self.xmax - self.xmin
            x_scaled = (xy - self.xmin) / dom_len
            q0 = np.ones_like(xy)
            q0[x_scaled <= 0.25] = 0.
            q0[x_scaled >= 0.75] = 0.
            if 'shift' in q0_type: q0 = q0 + 0.5
        elif q0_type == 'dissipation_test':
            # Jiang & Shu, J. Comput. Phys. 126, 202–228 (1996), Sec. 8.1
            assert self.dim == 1, 'dissipation_test only works for dim = 1.'
            x_scaled = 2.0 * (xy - self.xmin) / self.dom_len - 1.0  # map to [-1, 1]
            a, z, delta, alpha = 0.5, -0.7, 0.005, 10.0
            beta = np.log(2.0) / (36.0 * delta**2)
            q0 = np.zeros_like(xy, dtype=float)
            mask = (x_scaled >= -0.8) & (x_scaled <= -0.6)
            q0[mask] = (np.exp(-beta*(x_scaled[mask]-(z-delta))**2)
                        + np.exp(-beta*(x_scaled[mask]-(z+delta))**2)
                        + 4.0*np.exp(-beta*(x_scaled[mask]-z)**2)) / 6.0
            q0[(x_scaled >= -0.4) & (x_scaled <= -0.2)] = 1.0
            mask = (x_scaled >= 0.0) & (x_scaled <= 0.2)
            q0[mask] = 1.0 - np.abs(10.0*(x_scaled[mask] - 0.1))
            mask = (x_scaled >= 0.4) & (x_scaled <= 0.6)
            q0[mask] = (np.sqrt(np.maximum(1.0 - alpha**2*(x_scaled[mask]-(a-delta))**2, 0.0))
                        + np.sqrt(np.maximum(1.0 - alpha**2*(x_scaled[mask]-(a+delta))**2, 0.0))
                        + 4.0*np.sqrt(np.maximum(1.0 - alpha**2*(x_scaled[mask]-a)**2, 0.0))) / 6.0
        elif ('sinwave' in q0_type) and not ('gassner' in q0_type) \
            or ('coswave' in q0_type) and not ('gassner' in q0_type):
            if self.dim == 1:
                if '4pi' in q0_type:
                    w = 4*np.pi
                elif '8pi' in q0_type:
                    w = 8*np.pi
                else:
                    w = 2*np.pi
                x_scaled = (xy - self.xmin) / self.dom_len
                if 'sinwave' in q0_type:
                    q0 = np.sin(w * x_scaled) * self.q0_max_q
                elif 'coswave' in q0_type:
                    q0 = np.cos(w * x_scaled) * self.q0_max_q
                if 'shift' in q0_type:
                    q0 = q0+1.5
                if 'perturb' in q0_type:
                    assert (self.xmax==1 and self.xmin==0)
                    stdev2 = 0.08**2
                    exp = -0.5*(xy-0.5)**2/stdev2
                    q0 = q0 + 0.01*np.exp(exp)
                if 'coarse' in q0_type:
                    ncoarse = 8 # number of linear pieces for the coarse IC
                    qshape = self.qshape
                    if ncoarse > 1:
                        xflat = xy.flatten('F')
                        qflat = q0.flatten('F')
                        # section boundaries
                        for i in range(ncoarse-1):
                            # endpoints
                            x0, x1 = xflat[(i*len(xflat))//ncoarse], xflat[((i+1)*len(xflat))//ncoarse]
                            y0, y1 = qflat[(i*len(xflat))//ncoarse], qflat[((i+1)*len(xflat))//ncoarse]
                            # linear interpolation
                            xmod = (xflat[(i*len(xflat))//ncoarse:((i+1)*len(xflat))//ncoarse] - x0 ) / (x1-x0)
                            qflat[(i*len(xflat))//ncoarse:((i+1)*len(xflat))//ncoarse] = xmod * (y1 - y0) + y0
                        q0 = qflat.reshape(qshape, order='F')
            elif self.dim == 2:
                x_scaled = (xy[:,0,:] - self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] - self.xmin[1]) / self.dom_len[1]
                if q0_type == 'sinwave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled)  
                elif q0_type == 'sinwave2' or q0_type == 'sinwavesum' or q0_type == 'sinwave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) )
                elif q0_type == 'sinwave_8pi':
                    q0 = self.q0_max_q * np.sin(8*np.pi * x_scaled) * np.sin(8*np.pi * y_scaled)  
            elif self.dim == 3:
                x_scaled = (xy[:,0,:] - self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] - self.xmin[1]) / self.dom_len[1]
                z_scaled = (xy[:,2,:] - self.xmin[2]) / self.dom_len[2]
                if q0_type == 'sinwave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled) * np.sin(2*np.pi * z_scaled) 
                elif q0_type == 'sinwave2' or q0_type == 'sinwavesum' or q0_type == 'sinwave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) + np.sin(2*np.pi * z_scaled) )
        elif q0_type == 'random' or q0_type == 'random_shift':
            if 'shift' in q0_type:
                # Random numbers between 1.0 and 2.0
                q0 = np.random.rand(*qshape) + 1
            else:
                # Random numbers between -1.0 and 1.0
                q0 = 2*np.random.rand(*qshape) - 1
        elif q0_type == 'constant':
            q0 = np.ones(qshape)
        elif q0_type == 'gassnersinwave_cont': # continuous
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            q0 = np.sin(np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type == 'gassnersinwave_cont_4pi': # continuous
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            q0 = np.sin(4 * np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type in ('gassnersinwave', 'gassnersinwave_4pi'):
            # Same smooth IC as *_cont* on the solution nodes (no LG sub-cell projection).
            assert self.dim == 1, 'Chosen q0 shape only works for dim = 1.'
            if '4pi' in q0_type:
                w = 4 * np.pi
            else:
                w = np.pi
            xy_arr = np.asarray(xy, dtype=float)
            q0 = np.sin(w * xy_arr - 0.7) + 2
        elif q0_type in ('gassnersinwave_coarse', 'gassnersinwave_coarse_4pi'):
            assert self.dim == 1, 'Chosen q0 shape only works for dim = 1.'
            if '4pi' in q0_type:
                w = 4 * np.pi
            else:
                w = np.pi
            xy_arr = np.asarray(xy, dtype=float)
            from Source.Disc.Quadratures.LG import LG_set
            LGp = 1
            xy_LG = LG_set(LGp + 1)[0]
            xy_LG = 0.5 * (xy_LG[:, None] + 1)
            wBary_LG = MakeDgOp.BaryWeights(xy_LG)
            van = MakeDgOp.VandermondeLagrange1D(self.x_ref, xy_LG, wBary_LG)
            verts = np.linspace(self.xmin, self.xmax, int(self.nelem) + 1)
            elen = float(verts[1] - verts[0])
            x_op_1d = np.asarray(xy_LG[:, 0], dtype=float).ravel()
            x_elem_coarse = verts[:-1] + elen * x_op_1d[:, np.newaxis]
            q0_coarse = np.sin(w * x_elem_coarse - 0.7) + 2
            q_mesh = van @ q0_coarse
            if xy_arr.shape == self.qshape:
                q0 = q_mesh
            else:
                q0 = self._eval_gassnersinwave_projected_at_x(q_mesh, xy_arr)
        elif q0_type == 'density_wave' or q0_type == 'density_wave_shift':
            x_scaled = (xy - self.xmin) / self.dom_len
            q0 = 1. + 0.98*np.sin(4*np.pi*x_scaled)
            if 'shift' in q0_type:
                q0 = q0 + 0.5
        elif q0_type == 'skewed_sin':
            # Lazy computation of coefficients on first call
            if not hasattr(self, '_skewed_sin_coeff_q0'):
                from scipy.special import comb
                self._skewed_sin_n_fourier_q0 = 5
                n_fourier = self._skewed_sin_n_fourier_q0
                binom_norm = comb(2 * n_fourier, n_fourier, exact=True)
                ks = np.arange(1, n_fourier + 1)
                # Use exact=False for array inputs (exact=True only works with scalars)
                comb_vals = comb(2 * n_fourier, n_fourier - ks, exact=False)
                self._skewed_sin_coeff_q0 = comb_vals / (binom_norm * ks)
            
            # Check if x is a scalar (Python float/int or numpy scalar)
            if np.isscalar(xy):
                # Scalar case: return scalar
                xmod = 2.*np.pi*(xy - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier_q0 + 1)
                kx = ks * xmod  # shape: (n_fourier,)
                sin_kx = np.sin(kx)  # shape: (n_fourier,)
                fourier_sum = np.dot(self._skewed_sin_coeff_q0, sin_kx)  # scalar result
                q0 = float(fourier_sum + 1.5)
            else:
                # Array case: preserve shape
                xmod = 2.*np.pi*(xy - self.xmin) / self.dom_len + 4.
                ks = np.arange(1, self._skewed_sin_n_fourier_q0 + 1)
                ks_shape = (len(ks),) + (1,) * xmod.ndim
                kx = ks.reshape(ks_shape) * xmod  # shape: (n_fourier, ...)
                sin_kx = np.sin(kx)  # shape: (n_fourier, ...)
                # Multiply by coefficients and sum over k dimension (axis 0)
                # Use einsum to sum over first dimension while preserving other dimensions
                fourier_sum = np.einsum('i,i...->...', self._skewed_sin_coeff_q0, sin_kx)
                q0 = fourier_sum + 1.5
        elif q0_type == 'rarefaction':
            uL = 0.1
            uR = 1.0
            xmid = 0.5 * (self.xmin + self.xmax)
            xmod = np.mod((xy - xmid), self.dom_len) / self.dom_len 
            q0 = uR - (uR - uL) * xmod
        else:
            print(f'q0_type = {q0_type}')
            raise Exception('Unknown q0_type for initial solution')
        
        return q0
    
    def set_q0_discrete(self, q0, xy=None, k=3, s=0, periodic=None):
        '''
        Set discrete initial condition using spline interpolation.
        Currently only supports 1D problems.
        
        Parameters
        ----------
        q0 : np array
            Discrete initial condition values. Shape should match either:
            - The shape of xy if xy is provided
            - The shape of self.qshape if xy is None (e.g., (nen*neq_node, nelem) for 1D)
        xy : np array, optional
            Coordinates corresponding to q0. Defaults to self.x_elem if not provided.
        k : int, optional
            Degree of the spline. Default is 3 (cubic).
        s : float, optional
            Smoothing factor. Default is 0 (exact interpolation at all points).
            If s > 0, the spline will smooth the data instead of interpolating exactly.
        periodic : bool, optional
            Whether to use periodic boundary conditions. Defaults to checking
            hasattr(self, 'periodic') and self.periodic, or False if not available.
            For periodic cases, the spline will wrap coordinates using modx.
        
        Returns
        -------
        None
            Overwrites self.set_q0 with the interpolator function.
        '''
        # Dimension check: only 1D supported
        assert self.dim == 1, "set_q0_discrete currently only supports 1D problems"
        
        # Determine if periodic
        if periodic is None:
            periodic = (hasattr(self, 'periodic') and self.periodic)
        else:
            periodic = bool(periodic)
        
        # Get coordinates
        if xy is None:
            xy = self.x_elem
        
        # Flatten q0 and xy using Fortran-style
        q0_flat = np.array(q0).flatten('F')
        xy_flat = np.array(xy).flatten('F')
        
        # Remove duplicate points (e.g., at element interfaces where nodes are shared)
        # Keep unique x-coordinates and corresponding q0 values
        # If duplicates exist, verify they have matching q0 values, then use the first occurrence
        tol = 1e-12
        unique_mask = np.ones(len(xy_flat), dtype=bool)
        for i in range(1, len(xy_flat)):
            if np.abs(xy_flat[i] - xy_flat[i-1]) < tol:
                # Check that q0 values also match for duplicate x-coordinates
                if np.abs(q0_flat[i] - q0_flat[i-1]) > tol:
                    raise ValueError(
                        f"Duplicate x-coordinate at {xy_flat[i]:.6e} has inconsistent q0 values: "
                        f"{q0_flat[i-1]:.6e} and {q0_flat[i]:.6e}. "
                        f"Cannot create valid interpolator for discontinuous data."
                    )
                unique_mask[i] = False
        
        xy_unique = xy_flat[unique_mask]
        q0_unique = q0_flat[unique_mask]
        
        if periodic:
            print(f"Using periodic spline interpolation (k={k}) to set initial condition.")
            from scipy.interpolate import make_interp_spline, UnivariateSpline
            
            # Check which endpoints are included
            left_endpoint_included = np.abs(xy_unique[0] - self.xmin) < tol
            right_endpoint_included = np.abs(xy_unique[-1] - self.xmax) < tol
            
            # Prepare data for spline construction
            if left_endpoint_included and right_endpoint_included:
                # Both endpoints included (SBP case)
                # For periodic splines, the endpoint values must match
                if np.abs(q0_unique[-1] - q0_unique[0]) > tol:
                    print(f"WARNING: Right endpoint value ({q0_unique[-1]:.6e}) does not match left endpoint ({q0_unique[0]:.6e}).")
                    print("  For periodic data, these should be equal. Forcing right endpoint to match left endpoint.")
                    qbdy_avg = 0.5*(q0_unique[-1] + q0_unique[0])
                    q0_unique[-1] = qbdy_avg
                    q0_unique[0] = qbdy_avg
                xy_spline = xy_unique
                q0_spline = q0_unique
                use_periodic_spline = True
                
            elif left_endpoint_included or right_endpoint_included:
                # One endpoint included - add the missing one
                if left_endpoint_included:
                    # Right endpoint missing - add it with value matching left endpoint
                    xy_spline = np.concatenate([xy_unique, [self.xmax]])
                    q0_spline = np.concatenate([q0_unique, [q0_unique[0]]])
                else:
                    # Left endpoint missing - add it with value matching right endpoint
                    xy_spline = np.concatenate([[self.xmin], xy_unique])
                    q0_spline = np.concatenate([[q0_unique[-1]], q0_unique])
                use_periodic_spline = True
                
            else:
                # Neither endpoint included - fallback to non-periodic spline with manual periodicity
                print("  Neither endpoint included. Using non-periodic spline with manual periodic wrapping.")
                endpoint_value = q0_unique[0]  # Use leftmost value for both endpoints
                xy_spline = np.concatenate([[self.xmin], xy_unique, [self.xmax]])
                q0_spline = np.concatenate([[endpoint_value], q0_unique, [endpoint_value]])
                use_periodic_spline = False
            
            # Create spline (periodic or non-periodic)
            if use_periodic_spline:
                spline = make_interp_spline(xy_spline, q0_spline, k=k, bc_type='periodic')
            else:
                spline = UnivariateSpline(xy_spline, q0_spline, k=k, s=0)
            
            # Create periodic interpolator function that wraps coordinates
            # This wrapper is the same regardless of spline type
            def _q0_interpolator_func(x):
                x = np.array(x, copy=False)
                # Wrap coordinates to [xmin, xmin+L) domain (periodic)
                if hasattr(self, 'modx'):
                    x_wrapped = self.modx(x)
                else:
                    x_wrapped = self.xmin + np.mod(x - self.xmin, self.dom_len)
                # Evaluate spline
                return spline(x_wrapped)
            
            self._q0_interpolator_func = _q0_interpolator_func
        else:
            print(f"Using spline interpolation (k={k}, s={s}) to set initial condition.")
            from scipy.interpolate import UnivariateSpline

            # Create spline with s=0 (exact interpolation) and specified degree k
            self._q0_interpolator_func = UnivariateSpline(xy_unique, q0_unique, k=k, s=s)
        
        # Store unique arrays for sanity check
        xy_check = xy_unique
        q0_check = q0_unique
        
        # Sanity check: verify interpolation is exact at original grid points
        q0_interp_at_grid = self._q0_interpolator_func(xy_check)
        max_error = np.max(np.abs(q0_interp_at_grid - q0_check))
        tol = 1e-12
        
        if max_error > tol:
            print(f"WARNING: Interpolation error at grid points: max error = {max_error:.2e} > tol = {tol:.2e}")
            if s > 0:
                print(f"  This is expected when s={s} > 0 (smoothing mode). For exact interpolation, use s=0.")
            else:
                print("  This should not happen for UnivariateSpline with s=0. Check implementation.")
        else:
            print(f"Interpolation sanity check passed: max error at grid points = {max_error:.2e}")
        
        # Create the interpolator function that matches set_q0 signature
        # This makes sure that self._q0_interpolator_func is called with a flat array
        def _q0_interpolator(self, q0_type=None, xy=None, **kwargs):
            """
            Interpolated version of set_q0 for discrete initial conditions.
            Only works for 1D (enforced in set_q0_discrete).
            """
            if xy is None:
                # Use default mesh coordinates (same as original set_q0)
                xy = self.x_elem
            
            # Store original shape for reshaping
            orig_shape = np.shape(xy)
            
            # Flatten xy for interpolation using Fortran-style
            xy_flat = np.array(xy).flatten('F')
            
            # Evaluate interpolator at flattened coordinates
            q0_flat = self._q0_interpolator_func(xy_flat)
            
            # Reshape to match input shape using Fortran-style
            if len(orig_shape) > 1:
                q0 = np.reshape(q0_flat, orig_shape, 'F')
            else:
                q0 = q0_flat
            
            return q0
        
        # Bind the method to the instance and overwrite set_q0
        import types
        self.set_q0 = types.MethodType(_q0_interpolator, self)

    # TODO: Make a separate function for interactive plots? is this even possible using free packages?
    def plot_sol(self, q, x=None, time=0., plot_exa=None, savefile=None,
                 show_fig=True, ymin=None, ymax=None, display_time=False, 
                 title=None, plot_mesh=False, save_format='png', dpi=300,
                 plot_only_exa=False, var2plot_name=None, legendloc=None, legend=True,
                 show_negative=False, time_round=2, figsize=None, ymin_negative=None,
                 label_axes=True, **kwargs):
        '''
        Purpose
        ----------
        Used to plot the solution
        
        '''

        if var2plot_name is None:
            var2plot_name = self.var2plot_name
        if legendloc is None:
            legendloc = 'best'

        if plot_exa and not self.has_exa_sol:
            print('WARNING: Exact solution not available, so ignoring plot_exa=True')
            plot_exa = False
        
        if self.dim == 1:
            if plot_exa is None: plot_exa = True

            if x is None: 
                x = self.x_elem
            else:
                assert np.ndim(x) == 2, 'x must be a 2D array.'

            num_sol = self.var2plot(q,var2plot_name)

            if figsize is None: figsize=self.plt_fig_size 
            fig = plt.figure(figsize=figsize)
            ax = plt.axes() 

            if plot_exa and self.has_exa_sol:
                exa_sol = self.var2plot(self.exact_sol(time,x=x,guess=q),var2plot_name)
                if np.shape(x[:, 0]) != np.shape(exa_sol[:, 0]):
                    print(f'WARNING: Shapes of x and exa_sol do not match ({x.shape} vs {exa_sol.shape}). Plotting only numerical solution.')
                    plot_exa = False
                if plot_exa:
                    ax.plot(x[:, 0], exa_sol[:, 0], **self.plt_style_exa_sol,label='Exact')
                    for elem in range(1,x.shape[1]):
                        ax.plot(x[:, elem], exa_sol[:, elem], **self.plt_style_exa_sol)
            
            ax.plot(x[:, 0], num_sol[:, 0], **self.plt_style_sol[0], label='Numerical')
            for elem in range(1,x.shape[1]):
                ax.plot(x[:, elem], num_sol[:, elem], **self.plt_style_sol[0])
        
            ax.set_ylim(ymin,ymax)
            plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
            if var2plot_name is None:
                plt.ylabel(r'$u$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
            else:
                plt.ylabel(var2plot_name,fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
        
        
        elif self.dim == 2:
            if plot_exa is None: plot_exa = False
            if x is None: 
                xy = None
                x = fn.reshape_to_meshgrid_2D(self.xy_elem[:,0,:],self.nen,self.nelem[0],self.nelem[1])
                y = fn.reshape_to_meshgrid_2D(self.xy_elem[:,1,:],self.nen,self.nelem[0],self.nelem[1])
                nen = self.nen
            else:
                xy = np.copy(x)
                nen = int(np.sqrt(xy.shape[0]))
                x = fn.reshape_to_meshgrid_2D(xy[:,0,:],nen,self.nelem[0],self.nelem[1])
                y = fn.reshape_to_meshgrid_2D(xy[:,1,:],nen,self.nelem[0],self.nelem[1])


            fig = plt.figure(figsize=(6,5.5*self.dom_len[1]/self.dom_len[0])) # scale figure properly
            ax = plt.axes()
            
            num_sol = fn.reshape_to_meshgrid_2D(self.var2plot(q,var2plot_name),nen,self.nelem[0],self.nelem[1])

            cmap = plt.get_cmap(self.plt_contour_settings['cmap'])
            cmap.set_bad(color='white')  
            if ymin is None: ymin = 0.0
            
            CS = ax.contourf(x,y,num_sol,levels=self.plt_contour_settings['levels'],
                                 vmin=ymin, vmax=ymax, cmap=cmap)
            ax.set_aspect('equal', adjustable='box') # adjusts the shape of the figure to make data in x and y scale equally
            
            if ymin is not None or ymax is not None:
                if ymin is None: ymin = np.min(num_sol)
                if ymax is None: ymax = np.max(num_sol)
                norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                if label_axes:
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.79, aspect=18)
                else:
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.87, aspect=18)
            else:
                if label_axes:
                    cbar = fig.colorbar(CS, ax=ax, shrink=0.79, aspect=18)
                else:
                    cbar = fig.colorbar(CS, ax=ax, shrink=0.87, aspect=18)
            if var2plot_name is not None:
                cbar.ax.set_ylabel(var2plot_name, rotation=self.plt_contour_settings['rotation'],
                                   fontsize=self.plt_contour_settings['cbar_font_size'])
            if self.plt_contour_settings['cbar_nticks'] is not None:
                cbar.set_ticks(np.linspace(ymin,ymax,self.plt_contour_settings['cbar_nticks'])) 
            cbar.ax.tick_params(labelsize=self.plt_contour_settings['cbar_tick_size'])   
                
            if plot_mesh:
                #ax = plt.axes(frameon=False) # turn off the frame
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                
                #ax.set_xlim(self.xmin[0]-self.dom_len[0]/100,self.xmax[0]+self.dom_len[0]/100)
                #ax.set_ylim(self.xmin[1]-self.dom_len[1]/100,self.xmax[1]+self.dom_len[1]/100)
                
                if self.plt_mesh_settings['plot nodes']:
                    ax.scatter(self.xy[:,0],self.xy[:,1],marker='o',
                               c=self.plt_mesh_settings['node color'],
                               s=self.plt_mesh_settings['node size'])
        
                for line in self.mesh.grid_lines:
                    ax.plot(line[0],line[1],color='black',lw=1)
                if self.plt_mesh_settings['label lines']:
                    ax.tick_params(axis='both',length=0,labelsize=self.plt_label_font_size) # hide ticks
                    edge_verticesx = np.linspace(self.xmin[0],self.xmax[0],self.nelem[0]+1)
                    edge_verticesy = np.linspace(self.xmin[1],self.xmax[1],self.nelem[1]+1)
                    ax.set_xticks(edge_verticesx) # label element boundaries
                    ax.set_yticks(edge_verticesy)

            if show_negative:
                if ymin_negative is None: 
                    ymin_negative = ymin
                # to really make it obvious, add squares where solution is less than ymin
                square_size = 0.005*np.sqrt((self.xmax[0] - self.xmin[0])**2 +
                                           (self.xmax[1] - self.xmin[1])**2 )  # physical size of the square
                import matplotlib.patches as patches
                for i in range(len(x)):
                    for j in range(len(y)):
                        if num_sol[i, j] < ymin_negative:
                            print(f'Adding square at {x[i,j]}, {y[i,j]}')
                            print(f'num_sol[i,j] = {num_sol[i,j]}')
                            print(f'ymin_negative = {ymin_negative}')
                            # You may need to adjust x[j] and y[i] depending on mesh alignment
                            ax.add_patch(patches.Rectangle(
                                        (x[i,j] - square_size/2, y[i,j] - square_size/2),
                                        square_size, square_size,
                                        facecolor='white', edgecolor='none'))

            if label_axes:
                plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
                plt.ylabel(r'$y$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        
        elif self.dim == 3:
            # TODO: add option for plotting cross sections?
            raise Exception('Plotting is not currently supported for dim>2.')
        
        if display_time and (time is not None):
            # define matplotlib.patch.Patch properties
            # TODO: Add a check to see whether to set alpha or not
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, f'$t={round(time,time_round)}$', transform=ax.transAxes, 
                    fontsize=self.plt_label_font_size, verticalalignment='top', bbox=props)
        
        if plt.title is not None:
            plt.title(title,fontsize=self.plt_label_font_size+1)
            if self.dim == 1:
                if legend:
                    plt.legend(loc=legendloc,fontsize=self.plt_label_font_size-1)

        plt.tight_layout()
        
        if savefile is not None:
            filename = savefile+'.'+save_format
            if path.exists(filename):
                print('WARNING: File name already exists. Using a temporary name instead.')
                plt.savefig(filename+'_RENAMEME', format=save_format, dpi=dpi)
            else: 
                plt.savefig(filename, format=save_format, dpi=dpi)
            
        if show_fig:
            plt.show()
        plt.close()
        
        if self.dim == 2 and plot_exa and not plot_only_exa:
            if savefile is not None:
                savefile = savefile + '_exa'
            if title is not None:
                title = 'Exact Solution'
            exa_sol = self.exact_sol(time,xy=xy,guess=q)
            self.plot_sol(exa_sol, time=time, plot_exa=True, savefile=savefile,
                 show_fig=show_fig, ymin=ymin, ymax=ymax, display_time=display_time, 
                 title=title, plot_mesh=plot_mesh, save_format=save_format, dpi=dpi,
                 plot_only_exa=True, var2plot_name=var2plot_name)

    def plot_slice(self, q, x=None, xslice=None, yslice=None, 
                 time=None,savefile=None, show_fig=True, ymin=None, ymax=None, display_time=False, 
                 title=None, save_format='png', dpi=300, show_plot=True,
                 var2plot_name=None, legendloc=None, legend=True,
                 time_round=2, figsize=None, return_slice=False,**kwargs):
        '''
        Purpose
        ----------
        Used to plot the solution: Note assumes an unwarped grid!
        
        '''

        if var2plot_name is None:
            var2plot_name = self.var2plot_name
        if legendloc is None:
            legendloc = 'best'

        if x is None: 
            xy = np.copy(self.xy_elem)
            nen = self.nen
        else:
            xy = np.copy(x)
            nen = int(np.sqrt(xy.shape[0]))

        # Now find the closest x or y to slice
        n_idcs_expected = int(np.sqrt(xy[:,0,:].size))
        if yslice is None:
            slice_idx = np.unravel_index(np.argmin(np.abs(xy[:,0,:] - xslice)), xy[:,0,:].shape)
            xslice = xy[slice_idx[0],0,slice_idx[1]]
            print(f'Slicing at x={xslice}')
            idcs = np.argwhere(np.abs(xy[:,0,:] - xslice)<1e-10)
            xy = xy[idcs[:,0],1,idcs[:,1]]
        else:
            slice_idx = np.unravel_index(np.argmin(np.abs(xy[:,1,:] - yslice)), xy[:,1,:].shape)
            yslice = xy[slice_idx[0],1,slice_idx[1]]
            print(f'Slicing at y={yslice}')
            idcs = np.argwhere(np.abs(xy[:,1,:] - yslice)<1e-10)
            xy = xy[idcs[:,0],0,idcs[:,1]]

        n_idcs = idcs.shape[0]
        if n_idcs != n_idcs_expected:
            # This may happen, for example, if you choose a slice along a boundary with doubled nodes
            raise Exception(f'Expected {n_idcs_expected} nodes within 1e-10 of the slice at x={xslice} or y={yslice}, but found {n_idcs}')
    
        num_sol = self.var2plot(q,var2plot_name)[idcs[:,0],idcs[:,1]]

        #ax.plot(xy, num_sol, linestyle='', marker='.')
        order = np.argsort(xy)
        x_sorted = xy[order]
        y_sorted = num_sol[order]
        if show_plot:
            if figsize is None: figsize=self.plt_fig_size 
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
            ax.plot(x_sorted, y_sorted, **self.plt_style_sol[0])
            plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
            if var2plot_name is None:
                plt.ylabel(r'$u$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
            else:
                plt.ylabel(var2plot_name,fontsize=self.plt_label_font_size,rotation=0,labelpad=15)

            if display_time and (time is not None):
                # define matplotlib.patch.Patch properties
                # TODO: Add a check to see whether to set alpha or not
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.05, 0.95, f'$t={round(time,time_round)}$', transform=ax.transAxes, 
                        fontsize=self.plt_label_font_size, verticalalignment='top', bbox=props)
            
            if plt.title is not None:
                plt.title(title,fontsize=self.plt_label_font_size+1)

            plt.tight_layout()
            
            if savefile is not None:
                filename = savefile+'.'+save_format
                if path.exists(filename):
                    print('WARNING: File name already exists. Using a temporary name instead.')
                    plt.savefig(filename+'_RENAMEME', format=save_format, dpi=dpi)
                else: 
                    plt.savefig(filename, format=save_format, dpi=dpi)
                
            if show_fig:
                plt.show()
            plt.close()
        if return_slice:
            return x_sorted, y_sorted

        
            

    ''' Terms for the first derivative: E '''
    
    # Note: since this is class dependent, it will throw errors with numba
    def central_flux(self,qL,qR):
        ''' a simple central 2-point flux, the default for the Hadamard form
        NOTE: Ideally this should NOT be used. Will be very slow. '''
        fx = fn.arith_mean(self.calcEx(qL),self.calcEx(qR))
        return fx

    def logarithmic_flux(self,qL,qR):
        ''' a logarithmic 2-point flux, the default for the Hadamard form
        NOTE: Ideally this should NOT be used. Will be very slow. '''
        fx = fn.log_mean(self.calcEx(qL),self.calcEx(qR))
        return fx

    def geometric_flux(self,qL,qR):
        ''' a geometric 2-point flux, the default for the Hadamard form
        NOTE: Ideally this should NOT be used. Will be very slow. '''
        fx = fn.geom_mean(self.calcEx(qL),self.calcEx(qR))
        return fx

    # Note: since this is class dependent, it will throw errors with numba    
    def central_fluxes(self,qL,qR):
        ''' a simple central 2-point flux, the default for the Hadamard form
        NOTE: Ideally this should NOT be used. Will be very slow. '''
        fx = fn.arith_mean(self.calcEx(qL),self.calcEx(qR))
        fy = fn.arith_mean(self.calcEy(qL),self.calcEy(qR))
        if self.dim == 3:
            fz = fn.arith_mean(self.calcEy(qL),self.calcEy(qR))
            return fx, fy, fz
        else:
            return fx, fy
        
    def maxeig_dExdq(self,q):
        print('WARNING: Using default maxeig_dExdq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dExdq matrix
            return np.abs(self.dExdq(q)[:,0,0,:])
        else: # system
            eig_val = fn.spec_rad(self.dExdq(q),self.neq_node)
            return eig_val
        
    def maxeig_dEydq(self,q):
        print('WARNING: Using default maxeig_dEydq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dExdq matrix
            return np.abs(self.dEydq(q)[:,0,0,:])
        else: # system
            eig_val = fn.spec_rad(self.dEydq(q),self.neq_node)
            return eig_val
        
    def maxeig_dEzdq(self,q):
        print('WARNING: Using default maxeig_dEzdq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dExdq matrix
            return np.abs(self.dEzdq(q)[:,0,0,:])
        else: # system
            eig_val = fn.spec_rad(self.dEzdq(q),self.neq_node)
            return eig_val 

    def dExdq_abs(self,q):
        # This is a base method and should not be used, as it in general will be slow
        dExdq = self.dExdq(q)
        return fn.abs_eig_mat(dExdq)
    
    def dEydq_abs(self,q):
        # This is a base method and should not be used, as it in general will be slow
        dEydq = self.dEydq(q)
        return fn.abs_eig_mat(dEydq) 
    
    def dEzdq_abs(self,q):
        # This is a base method and should not be used, as it in general will be slow
        dEzdq = self.dEzdq(q)
        return fn.abs_eig_mat(dEzdq) 
    
    def check_positivity(self,q):
        raise Exception('This base method should not be called.')

    ''' Source term '''

    def calcG(self, q, t):
        return 0

    def dGdq(self, q):
        return 0

    def clip_positivity_fn(self, q, pos_floor, pos_cut):
        ''' default clip positivity function, can be overwritten by the user 
            this implementation assumes a clipping everywhere, i.e. only
            suitable for scalar euqations or systems where all variables are positive '''
        return fn.clip_pos_smooth_vec(q, pos_floor, pos_cut)

    
    ''' functions setting up operators '''
    # TODO: Do I need these? At least for split forms, yes
    
    def set_sbp_op_1d(self, Dx, gm_gv):
        self.Dx = Dx
        self.gm_gv = gm_gv
    
    def set_dg_strong_op(self, dd_phys):
        #TODO
        
        self.der1 = dd_phys

    def set_fd_op(self, p, use_sparse=False):
        #TODO

        self.p = p

        # Construct the finite difference operator
# =============================================================================
#         der1, der1_bcL, der1_bcR = FiniteDiff.der1(self.p, self.nn, self.dx, self.isperiodic)
#         eye = sp.eye(self.neq_node, format="csr")
#         if der1_bcL is None: der1_bcL = 0
#         if der1_bcR is None: der1_bcR = 0
#         self.der1 = np.array(sp.kron(der1, eye).todense())
#         self.der1_bcL = np.array(sp.kron(der1_bcL, eye).todense())
#         self.der1_bcR = np.array(sp.kron(der1_bcR, eye).todense())
# =============================================================================
            
class DiffEqOverwrite:
# Allows you to overwrite the methods in the Diffeq class

    def __init__(self, diffeq_in, dqdt, dfdq, 
                       cons_obj, n_cons_obj):

        self.dqdt = dqdt
        self.dfdq = dfdq

        self.set_q0 = diffeq_in.set_q0
        self.plot_sol = diffeq_in.plot_sol
        self.calc_cons_obj = cons_obj

        self.n_cons_obj = n_cons_obj