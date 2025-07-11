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
                            'cmap': 'inferno'} #'jet'}          # colourmap

    # Parameters for the initial solution
    q0_max_q = 1.0                 # Max value in the vector q0
    q0_gauss_wave_val_bc = 1e-10    # Value at the boundary for Gauss wave


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

    def set_mesh(self, mesh):
        '''
        Purpose
        ----------
        Needed to calculate the initial solution and to calculate source terms
        '''

        self.mesh = mesh

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

    def set_q0(self, q0_type=None, xy=None):
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
            if q0_type in ('gassnersinwave','gassnersinwave_coarse'):
                raise Exception('This q0_type does not work with xy provided.')
            qshape = np.shape(xy)

        if q0_type == 'gausswave' or q0_type == 'gausswave_0.25':
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
                stdev2x = abs(self.dom_len[0]**2/k) # standard deviation squared
                stdev2y = abs(self.dom_len[1]**2/k)
                exp = -0.5*((xy[:,0,:]-mid_pointx)**2/stdev2x + (xy[:,1,:]-mid_pointy)**2/stdev2y)
                q0 = self.q0_max_q * np.exp(exp)
            elif self.dim == 3:
                mid_pointx = 0.5*(self.xmax[0] + self.xmin[0]) # mean
                mid_pointy = 0.5*(self.xmax[1] + self.xmin[1])
                mid_pointz = 0.5*(self.xmax[2] + self.xmin[2])
                stdev2x = abs(self.dom_len[0]**2/k) # standard deviation squared
                stdev2y = abs(self.dom_len[1]**2/k)
                stdev2z = abs(self.dom_len[2]**2/k)
                exp = -0.5*((xy[:,0,:]-mid_pointx)**2/stdev2x + (xy[:,1,:]-mid_pointy)**2/stdev2y + (xy[:,2,:]-mid_pointz)**2/stdev2z)
                q0 = self.q0_max_q * np.exp(exp) 
        elif q0_type == 'gausswave_sbpbook':
            assert self.dim==1,'only for dim=1'
            assert (self.xmax==1 and self.xmin==0)
            stdev2 = 0.08**2
            exp = -0.5*(xy-0.5)**2/stdev2
            q0 = np.exp(exp)
        elif q0_type == 'gausswave_shifted':
            assert self.dim==1,'only for dim=1'
            assert (self.xmax==1 and self.xmin==0)
            stdev2 = 0.08**2
            exp = -0.5*(xy-0.5)**2/stdev2
            q0 = np.exp(exp) - 0.5
        elif q0_type == 'morlet_wavelet':
            assert self.dim==1,'only for dim=1'
            k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
            mid_point = 0.5*(self.xmax + self.xmin) # mean
            stdev2 = abs(self.dom_len**2/k) # standard deviation squared
            exp = -0.5*(xy-mid_point)**2/stdev2
            gauss = self.q0_max_q * np.exp(exp)
            k_pi = 10 # how many oscillations to include in the domain
            cos = np.cos(k_pi*2.*np.pi*(xy-mid_point)/self.dom_len)
            q0 = gauss * cos
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
        elif q0_type == 'squarewave': 
            assert self.dim == 1,'square wave only works for dim = 1.' 
            dom_len = self.xmax - self.xmin
            x_scaled = (xy + self.xmin) / dom_len
            q0 = np.ones_like(xy)
            q0[x_scaled <= 0.25] = 0.
            q0[x_scaled >= 0.75] = 0.
        elif ('sinwave' in q0_type) and not ('gassner' in q0_type) \
            or ('coswave' in q0_type) and not ('gassner' in q0_type):
            if self.dim == 1:
                if '4pi' in q0_type:
                    w = 4*np.pi
                elif '8pi' in q0_type:
                    w = 8*np.pi
                else:
                    w = 2*np.pi
                x_scaled = (xy + self.xmin) / self.dom_len
                if 'sinwave' in q0_type:
                    q0 = np.sin(w * x_scaled) * self.q0_max_q
                elif 'coswave' in q0_type:
                    q0 = np.cos(w * x_scaled) * self.q0_max_q
                if 'shift' in q0_type:
                    q0 = q0+2
                if 'perturb' in q0_type:
                    assert (self.xmax==1 and self.xmin==0)
                    stdev2 = 0.08**2
                    exp = -0.5*(xy-0.5)**2/stdev2
                    q0 = q0 + 0.01*np.exp(exp)
            elif self.dim == 2:
                x_scaled = (xy[:,0,:] + self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] + self.xmin[1]) / self.dom_len[1]
                if q0_type == 'sinwave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled)  
                elif q0_type == 'sinwave2' or q0_type == 'sinwavesum' or q0_type == 'sinwave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) )
            elif self.dim == 3:
                x_scaled = (xy[:,0,:] + self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] + self.xmin[1]) / self.dom_len[1]
                z_scaled = (xy[:,2,:] + self.xmin[2]) / self.dom_len[2]
                if q0_type == 'sinwave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled) * np.sin(2*np.pi * z_scaled) 
                elif q0_type == 'sinwave2' or q0_type == 'sinwavesum' or q0_type == 'sinwave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) + np.sin(2*np.pi * z_scaled) )
        elif q0_type == 'random':
            # Random numbers between -0.5 and 0.5
            q0 = np.random.rand(*qshape) -0.5
        elif q0_type == 'constant':
            q0 = np.ones(qshape)
        elif q0_type == 'gassnersinwave_cont': # continuous
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            q0 = np.sin(np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type == 'gassnersinwave_cont_4pi': # continuous
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            q0 = np.sin(4 * np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type in ('gassnersinwave','gassnersinwave_coarse'): # discontinuous  
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            from Source.Disc.Quadratures.LG import LG_set # use this instead of quadpy
            if q0_type == 'gassnersinwave_coarse':
                #xy_LG = qp.c1.gauss_legendre(2).points # coarse LG nodes
                LGp = 1
                xy_LG = LG_set(LGp+1)[0]
            else:
                #xy_LG = qp.c1.gauss_legendre(self.nen).points # full degree LG nodes
                xy_LG = LG_set(self.nen)[0]
            xy_LG = 0.5*(xy_LG[:, None] + 1) # Convert from 1D to 2D array
            wBary_LG = MakeDgOp.BaryWeights(xy_LG) # Barycentric weights for LG nodes
            van = MakeDgOp.VandermondeLagrange1D(self.x_ref,xy_LG,wBary_LG)
            # The vandermonde maps from xy_LG coarse nodes to self.xy_elem solution nodes
            # now map the LG nodes to the entire domain
            mesh = MakeMesh(self.dim, self.xmin, self.xmax, self.nelem, xy_LG[:,0])
            # Now set the initial condition on the coarse nodes and map to solution nodes
            q0_coarse = np.sin(np.pi * mesh.x_elem - 0.7) + 2
            q0 = van @ q0_coarse
        else:
            print(f'q0_type = {q0_type}')
            raise Exception('Unknown q0_type for initial solution')
        
        return q0
    

    # TODO: Make a separate function for interactive plots? is this even possible using free packages?
    def plot_sol(self, q, x=None, time=0., plot_exa=None, savefile=None,
                 show_fig=True, ymin=None, ymax=None, display_time=False, 
                 title=None, plot_mesh=False, save_format='png', dpi=300,
                 plot_only_exa=False, var2plot_name=None, legendloc=None, legend=True,
                 show_negative=False, time_round=2, **kwargs):
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
            
            fig = plt.figure(figsize=self.plt_fig_size)
            ax = plt.axes() 

            if plot_exa and self.has_exa_sol:
                exa_sol = self.var2plot(self.exact_sol(time,x=x,guess=q),var2plot_name)
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
            if show_negative:
                cmap.set_bad(color='white')  
                cmap.set_under(color='white') 
                if ymin is None: ymin = 0.0
            
            CS = ax.contourf(x,y,num_sol,levels=self.plt_contour_settings['levels'],
                                 vmin=ymin, vmax=ymax,cmap=cmap)
            ax.set_aspect('equal', adjustable='box') # adjusts the shape of the figure to make data in x and y scale equally
            
            if ymin is not None or ymax is not None:
                if ymin is None: ymin = np.min(num_sol)
                if ymax is None: ymax = np.max(num_sol)
                norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(mappable, ax=ax, shrink=0.79, aspect=18)
            else:
                cbar = fig.colorbar(CS, ax=ax, shrink=0.79, aspect=18)
            if var2plot_name is not None:
                cbar.ax.set_ylabel(var2plot_name)     
                
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
                # to really make it obvious, add squares where solution is less than ymin
                square_size = 0.005*np.sqrt((self.xmax[0] - self.xmin[0])**2 +
                                           (self.xmax[1] - self.xmin[1])**2 )  # physical size of the square
                import matplotlib.patches as patches
                for i in range(len(x)):
                    for j in range(len(y)):
                        if num_sol[i, j] < ymin:
                            # You may need to adjust x[j] and y[i] depending on mesh alignment
                            ax.add_patch(patches.Rectangle(
                                        (x[i,j] - square_size/2, y[i,j] - square_size/2),
                                        square_size, square_size,
                                        facecolor='white', edgecolor='none'))
            
            plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
            plt.ylabel(r'$y$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
        
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
                 plot_only_exa=True)
            

    ''' Terms for the first derivative: E '''
    
    # Note: since this is class dependent, it will throw errors with numba
    def central_flux(self,qL,qR):
        ''' a simple central 2-point flux, the default for the Hadamard form
        NOTE: Ideally this should NOT be used. Will be very slow. '''
        fx = fn.arith_mean(self.calcEx(qL),self.calcEx(qR))
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