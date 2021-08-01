#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:42:05 2020
@author: andremarchildon
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
rc('text', usetex=True)


import numpy as np
from os import path


from Source.Disc.FiniteDiff import FiniteDiff
from Source.Disc.MakeDgOp import MakeDgOp
from Source.Disc.MakeMesh import MakeMesh
import Source.Methods.Functions as fn
import quadpy as qp

'''
The classes in this file are inheritated by the classes for ODEs and PDEs.
The order of inheritance is as follows:
        PdeBaseCons       <- PdeBase
        PdeBasePar       <- PdeBase
PdeBase:
    -Provides an init function for the parameter and the names of the
    objectives functions.
    -Introduces various functions that should not be called directly.
    -Sets parameters for the initial solution (q0)
    -Provides an updated init file with more inputs
    -Provides functions required for all PDEs such as set_xy, set_q0
    -Has a function to plot the solution of 1D PDEs
    -Sets default methods to calculate dEdx as well as the source term and
    its derivative
PdeBaseCons:
    -Builds from PdeBase by adding methods for PDEs that are of the
    form dqdt + dEdx = G, ie no second or higher derivatives
PdeBasePar:
    -Builds from PdeBase by adding methods for PDEs that are of the
    form dqdt + dEdx = d^2EVdx^2 + G, ie with second derivatives
The PDEs are solved in this form:
    The Diffeq:             dqdt = -dEdx + G + diss = f
    Time marching methods:  dqdt = f(q) = rhs(q)
    Linearization:          f(q) = dfdq @ q 
        * note: dfdq \neq dEdq , analytic flux jacobian
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

    # Ploting options
    plt_fig_size = (6,4)
    plt_style_exa_sol = {'color':'r','linestyle':'-','linewidth':2,'marker':''}
    plt_style_sol = [{'color':'b','linestyle':'-','linewidth':2,'marker':''},
                     {'color':'k','linestyle':'','linewidth':2,'marker':''},
                     {'color':'r','linestyle':'','linewidth':2,'marker':''},
                     {'color':'g','linestyle':'','linewidth':2,'marker':''}]
    plt_label_font_size = 15
    plt_var2plot_name = None
    plt_mesh_settings = {'label lines': True,   # if True, x and y ticks are based on grid lines
                         'plot nodes': True,    # whether or not to display the nodes
                         'node size': 4,        # markersize used on nodes
                         'node color': 'black'} # marker colour used for nodes
    plt_contour_settings = {'levels': 100,          # number of distinct contours
                            'cmap': 'jet'}          # colourmap

    # Parameters for the initial solution
    q0_max_q = 1.2                  # Max value in the vector q0
    q0_gauss_wave_val_bc = 1e-10    # Value at the boundary for Gauss wave


    def __init__(self, para, obj_name=None, q0_type=None):
        '''
        Parameters
        ----------
        s : np array or float
            Parameters of the differential equation that derivatives can be
            calculated with respect to.
        obj_name : str or tuple, optional
            The name or names of objectives to calculate.
            The default is None.
        q0_type : str
            The type of initial solution for the DiffEq.
        '''

        ''' Add inputs to the class '''

        self.para = para
        self.obj_name = obj_name
        self.q0_type = q0_type

        ''' Modify type for inputs '''

        # Make sure that para is stored as a numpy array
        if isinstance(self.para, int) or isinstance(self.para, float):
            self.para = np.atleast_1d(np.asarray(self.para))

        # Store the objective name(s) in a tuple
        if obj_name is None:
            self.n_obj = 0
            self.obj_name = None
        else:
            # Standardize format for the name of the objective(s)
            if isinstance(obj_name, str):
                self.n_obj = 1
                self.obj_name = (obj_name,)
            elif isinstance(obj_name, tuple):
                self.n_obj = len(obj_name)
                self.obj_name = obj_name
            else:
                raise Exception('The variable obj_name has to be either a string or a tuple of strings')
            
        
    def var2plot(self,q):
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

        if xy is None:
            qshape = self.qshape
            if self.dim == 1:
                xy = self.x_elem # shape (self.nen, self.nelem)
            elif self.dim == 2:
                xy = self.xy_elem # shape (self.nen**2, 2, self.nelem[0]*self.nelem[1])
            elif self.dim == 3:
                xy = self.xyz_elem
        else:
            shape = xy.shape
            if self.dim == 1:
                assert xy.ndim == self.x_elem.ndim, 'Provided xy has wrong dimensions. Given shape {0} while default is {1}'.format(shape,self.x_elem.shape)
                qshape = xy.shape
            elif self.dim == 2:
                assert xy.ndim == self.xy_elem.ndim, 'Provided xy has wrong dimensions. Given shape {0} while default is {1}'.format(shape,self.xy_elem.shape)
                qshape = (shape[0],shape[2])
            elif self.dim == 3:
                assert xy.ndim == self.xyz_elem.ndim, 'Provided xy has wrong dimensions. Given shape {0} while default is {1}'.format(shape,self.xyz_elem.shape)
                qshape = (shape[0],shape[2])

        if q0_type == 'GaussWave':
            k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
            if self.dim == 1:
                mid_point = 0.5*(self.xmax + self.xmin) # mean
                stdev2 = abs(self.dom_len**2/k) # standard deviation squared
                exp = -0.5*(xy-mid_point)**2/stdev2
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
        elif 'GaussWave_debug' in q0_type:
            if 'y' in q0_type: xyz = 1
            elif 'z' in q0_type: xyz = 2
            else: xyz = 0
            assert self.dim!=1,'This is meant to be used in 2D or 3D to mimic a 1D problem in x, y, or z.'
            k = (8*np.log(self.q0_gauss_wave_val_bc/self.q0_max_q))
            mid_pointx = 0.5*(self.xmax[xyz] + self.xmin[xyz]) # mean
            stdev2 = abs(self.dom_len[xyz]**2/k) # standard deviation squared
            exp = -0.5*((xy[:,xyz,:]-mid_pointx)**2/stdev2)
            q0 = self.q0_max_q * np.exp(exp)    
        elif 'SinWave' in q0_type:
            if self.dim == 1:
                x_scaled = (xy + self.xmin) / self.dom_len
                q0 = np.sin(2*np.pi * x_scaled) * self.q0_max_q
            elif self.dim == 2:
                x_scaled = (xy[:,0,:] + self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] + self.xmin[1]) / self.dom_len[1]
                if q0_type == 'SinWave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled)  
                elif q0_type == 'SinWave2' or q0_type == 'SinWavesum' or q0_type == 'SinWave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) )
            elif self.dim == 3:
                x_scaled = (xy[:,0,:] + self.xmin[0]) / self.dom_len[0]
                y_scaled = (xy[:,1,:] + self.xmin[1]) / self.dom_len[1]
                z_scaled = (xy[:,2,:] + self.xmin[2]) / self.dom_len[2]
                if q0_type == 'SinWave':
                    q0 = self.q0_max_q * np.sin(2*np.pi * x_scaled) * np.sin(2*np.pi * y_scaled) * np.sin(2*np.pi * z_scaled) 
                elif q0_type == 'SinWave2' or q0_type == 'SinWavesum' or q0_type == 'SinWave_sum':
                    q0 = self.q0_max_q * ( np.sin(2*np.pi * x_scaled) + np.sin(2*np.pi * y_scaled) + np.sin(2*np.pi * z_scaled) )
        elif q0_type == 'Random':
            # Random numbers between -0.5 and 0.5
            q0 = np.random.rand(*qshape) -0.5
        elif q0_type == 'Constant':
            q0 = np.ones(qshape)
        elif q0_type == 'GassnerSinWave_cont': # continuous
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            q0 = np.sin(np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type in ('GassnerSinWave','GassnerSinWave_coarse'): # discontinuous  
            assert self.dim == 1,'Chosen q0 shape only works for dim = 1.'
            if q0_type == 'GassnerSinWave_coarse':
                xy_LG = qp.c1.gauss_legendre(2).points # coarse LG nodes
            else:
                xy_LG = qp.c1.gauss_legendre(self.nen).points # full degree LG nodes
            xy_LG = 0.5*(xy_LG[:, None] + 1) # Convert from 1D to 2D array
            wBary_LG = MakeDgOp.BaryWeights(xy_LG) # Barycentric weights for LG nodes
            van = MakeDgOp.VandermondeLagrange1D(self.xy_ref,wBary_LG,xy_LG)
            # The vandermonde maps from xy_LG coarse nodes to self.xy_elem solution nodes
            # now map the LG nodes to the entire domain
            mesh = MakeMesh(self.xmin, self.xmax, self.isperiodic, self.nelem, xy_LG)
            # Now set the initial condition on the coarse nodes and map to solution nodes
            q0_coarse = np.sin(np.pi * mesh.xy_elem[:,:,0] - 0.7) + 2
            q0 = van @ q0_coarse
        else:
            print(f'q0_type = {q0_type}')
            raise Exception('Unknown q0_type for initial solution')
        
        return q0
    

    # TODO: Make a separate function for interactive plots? is this even possible using free packages?
    def plot_sol(self, q, time=None, plot_exa=True, plt_save_name=None,
                 show_fig=True, solmin=None, solmax=None, display_time=False, 
                 title=None, plot_mesh=False, save_format='png', dpi=1000,
                 plot_only_exa=False):
        '''
        Purpose
        ----------
        Used to plot the solution
        
        '''
        
        if self.dim == 1:
            num_sol = self.var2plot(q).flatten('F')
            
            fig = plt.figure(figsize=self.plt_fig_size)
            ax = plt.axes() 

            if plot_exa and self.has_exa_sol:
                exa_sol = self.var2plot(self.exact_sol(time)).flatten('F')
                ax.plot(self.x, exa_sol, **self.plt_style_exa_sol, label='Exact')
                
            ax.plot(self.x, num_sol, **self.plt_style_sol[0], label='Numerical')
        
            ax.set_ylim(solmin,solmax)
            plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
            if self.plt_var2plot_name is None:
                plt.ylabel(r'$u$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
            else:
                plt.ylabel(self.plt_var2plot_name,fontsize=self.plt_label_font_size)
        
        
        elif self.dim == 2:            
            fig = plt.figure(figsize=(6,5.5*self.dom_len[1]/self.dom_len[0])) # scale figure properly
            ax = plt.axes()
            
            if plot_mesh:
                ax = plt.axes(frameon=False) # turn off the frame
                
                ax.set_xlim(self.xmin[0]-self.dom_len[0]/100,self.xmax[0]+self.dom_len[0]/100)
                ax.set_ylim(self.xmin[1]-self.dom_len[1]/100,self.xmax[1]+self.dom_len[1]/100)
                
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
            
            x = fn.reshape_to_meshgrid(self.xy_elem[:,0,:],self.nen,self.nelem[0],self.nelem[1])
            y = fn.reshape_to_meshgrid(self.xy_elem[:,1,:],self.nen,self.nelem[0],self.nelem[1])
            num_sol = fn.reshape_to_meshgrid(self.var2plot(q),self.nen,self.nelem[0],self.nelem[1])
            
            CS = ax.contourf(x,y,num_sol,levels=self.plt_contour_settings['levels'],
                                 vmin=solmin, vmax=solmax,
                                 cmap=self.plt_contour_settings['cmap'])
            
            cbar = fig.colorbar(CS)
            if self.plt_var2plot_name is not None:
                cbar.ax.set_ylabel(self.plt_var2plot_name)                    
                
            plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)
            plt.ylabel(r'$y$',fontsize=self.plt_label_font_size,rotation=0,labelpad=15)
        
        elif self.dim == 3:
            # TODO: add option for plotting cross sections?
            raise Exception('Plotting is not currently supported for dim>2.')
        
        if display_time and (time is not None):
            # define matplotlib.patch.Patch properties
            # TODO: Add a check to see whether to set alpha or not
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, r'$t=$ '+str(round(time,2))+' s', transform=ax.transAxes, 
                    fontsize=self.plt_label_font_size, verticalalignment='top', bbox=props)
        
        fig.tight_layout()
        
        if plt.title is not None:
            plt.title(title,fontsize=self.plt_label_font_size+1)
            if self.dim == 1:
                plt.legend(loc='best',fontsize=self.plt_label_font_size-1)
        
        if plt_save_name is not None:
            filename = plt_save_name+'.'+save_format
            if path.exists(filename):
                print('WARNING: File name already exists. Using a temporary name instead.')
                plt.savefig(filename+'_RENAMEME', format=save_format, dpi=dpi)
            else: 
                plt.savefig(filename, format=save_format, dpi=dpi)
            
        if show_fig:
            plt.show()
        plt.close()
        
        if self.dim == 2 and plot_exa and not plot_only_exa:
            if plt_save_name is not None:
                plt_save_name = plt_save_name + '_exa'
            if title is not None:
                title = 'Exact Solution'
            exa_sol = self.var2plot(self.exact_sol(time))
            self.plot_sol(exa_sol, time=time, plot_exa=True, plt_save_name=plt_save_name,
                 show_fig=show_fig, solmin=solmin, solmax=solmax, display_time=display_time, 
                 title=title, plot_mesh=plot_mesh, save_format=save_format, dpi=dpi,
                 plot_only_exa=True)
            

    ''' Terms for the first derivative: E '''
    
    def maxeig_dEdq(self,q):
        ''' Calculate the constant for the Lax-Friedrichs flux, useful to set 
        the CFL number but should not be used for computations in the code
        since this is slow. Equal to max(abs(eigval(dEdq))). See Hesthaven pg. 33. 
        If q is given, use that. If not, use the initial condition.'''
        print('WARNING: Using default maxeig_dEdq. Should not be used for main code.')
        if self.neq_node == 1: # scalar
            return np.max(np.abs(self.dEdq(q)),axis=(0,1))
        else: # system
            dEdq_mod = np.transpose(self.dEdq(q),axes=(2,0,1))
            eig_val = np.linalg.eigvals(dEdq_mod)
            return np.max(np.abs(eig_val),axis=1)
        
    def maxeig_dExdq(self,q):
        print('WARNING: Using default maxeig_dExdq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dEdq matrix
            return np.max(np.abs(self.dExdq(q)),axis=(0,1))
        else: # system
            dEdq_mod = np.transpose(self.dExdq(q),axes=(2,0,1))
            eig_val = np.linalg.eigvals(dEdq_mod)
            return np.max(np.abs(eig_val),axis=1)
        
    def maxeig_dEydq(self,q):
        print('WARNING: Using default maxeig_dEydq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dEdq matrix
            return np.max(np.abs(self.dEydq(q)),axis=(0,1))
        else: # system
            dEdq_mod = np.transpose(self.dEydq(q),axes=(2,0,1))
            eig_val = np.linalg.eigvals(dEdq_mod)
            return np.max(np.abs(eig_val),axis=1)
        
    def maxeig_dEzdq(self,q):
        print('WARNING: Using default maxeig_dEzdq. Should not be used for main code.')
        if self.neq_node == 1: # scalar, so diagonal dEdq matrix
            return np.max(np.abs(self.dEzdq(q)),axis=(0,1))
        else: # system
            dEdq_mod = np.transpose(self.dEzdq(q),axes=(2,0,1))
            eig_val = np.linalg.eigvals(dEdq_mod)
            return np.max(np.abs(eig_val),axis=1)
        
    # TODO: I dont think I need these
    
    def dExdx_div(self, q):
        E = self.calcEx(q)
        dEdx = fn.gm_gv(self.Dx, E)
        return dEdx
    
    def dEydy_div(self,q):
        E = self.calcEy(q)
        dEdx = fn.gm_gv(self.Dy, E)
        return dEdx  

    def dEzdz_div(self,q):
        E = self.calcEz(q)
        dEdx = fn.gm_gv(self.Dz, E)
        return dEdx 

    def dExdx_had(self, q):
        F = self.Fx_mat(q) 
        dEdx = 2*np.sum(fn.lm_gm_hadamard(self.Dx, F),axis=1)
        return dEdx
    
    def dEydy_had(self,q):
        F = self.Fx_mat(q) 
        dEdx = 2*np.sum(fn.lm_gm_hadamard(self.Dy, F),axis=1)
        return dEdx  

    def dEzdz_had(self,q):
        F = self.Fx_mat(q) 
        dEdx = 2*np.sum(fn.lm_gm_hadamard(self.Dz, F),axis=1)
        return dEdx 
            

    ''' Source term '''

    def calcG(self, q):
        return 0

    def dGdq(self, q):
        return 0
    
    ''' functions setting up operators '''
    # TODO: Do I need these?
    
    def set_sbp_op(self, H_inv, Dx, Dy=None, Dz=None):
        self.Dx = Dx
        self.Dy = Dy
        self.Dz = Dz
        self.H_inv = H_inv
    
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


# TODO: I dont think I need this
class PdeBaseCons(PdeBase):
    
    def __init__(self, para=None, obj_name=None, q0_type='SinWave'):

        super().__init__(para, obj_name, q0_type)
        if self.dim == 1:
            if not hasattr(self, 'dqdt'):
                print('Using default 1D dqdt')
                self.dqdt = self.dqdt_1D
            else:
                print('Using dqdt defined in specific DiffEq file.')
            if not hasattr(self, 'dfdq'):
                if self.use_hadamard:
                    print('Using default 1D hadamard form dfdq')
                    self.dfdq = self.dfdq_1D_had
                else:
                    print('Using default 1D divergence form dfdq')
                    self.dfdq = self.dfdq_1D_div
            else:
                print('Using dfdq defined in specific DiffEq file.')
        elif self.dim == 2:
            if not hasattr(self, 'dqdt'):
                print('Using default 2D dqdt')
                self.dqdt = self.dqdt_2D
            else:
                print('Using dqdt defined in specific DiffEq file.')
            if not hasattr(self, 'dfdq'):
                if self.use_hadamard:
                    print('Using default 2D hadamard form dfdq')
                    self.dfdq = self.dfdq_2D_had
                else:
                    print('Using default 2D divergence form dfdq')
                    self.dfdq = self.dfdq_2D_div
            else:
                print('Using dfdq defined in specific DiffEq file.')
        elif self.dim == 3:
            if not hasattr(self, 'dqdt'):
                print('Using default 3D dqdt')
                self.dqdt = self.dqdt_3D
            else:
                print('Using dqdt defined in specific DiffEq file.')
            if not hasattr(self, 'dfdq'):
                if self.use_hadamard:
                    print('Using default 3D hadamard form dfdq')
                    self.dfdq = self.dfdq_3D_had
                else:
                    print('Using default 3D divergence form dfdq')
                    self.dfdq = self.dfdq_3D_div
            else:
                print('Using dfdq defined in specific DiffEq file.')

    ''' This base class is for PDEs of the form dqdt + dE_idx_i = G '''

    def dqdt_1D(self, q):

        dEdx = self.dExdx(q)
        G = self.calcG(q)

        dqdt = -dEdx + G
        return dqdt

    def dfdq_1D_div(self, q):
        # WARNING: Does not apply for explicit split forms

        A = self.dExdq(q)
        dGdq = self.dGdq(q)

        dfdq = - fn.gm_gm(self.Dx, A) + dGdq
        return dfdq
    
    def dfdq_1D_had(self, q):

        A = self.Ax_tens(q) 
        dGdq = self.dGdq(q)

        dfdq = - 2*np.sum(fn.lm_gt_hadamard(self.Dx, A),axis=1) + dGdq
        return dfdq
    
    def dqdt_2D(self, q):

        dExdx = self.dExdx(q)
        dEydy = self.dEydy(q)
        G = self.calcG(q)

        dqdt = -dExdx -dEydy + G
        return dqdt

    def dfdq_2D_div(self, q):
        # WARNING: Does not apply for explicit split forms

        Ax = self.dExdq(q)
        Ay = self.dEydq(q)
        dGdq = self.dGdq(q)

        dfdq = - fn.gm_gm(self.Dx, Ax) - fn.gm_gm(self.Dy, Ay) + dGdq
        return dfdq
    
    def dfdq_2D_had(self, q):

        Ax = self.Ax_tens(q) 
        Ay = self.Ay_tens(q) 
        dGdq = self.dGdq(q)

        dfdq = - 2*np.sum(fn.lm_gt_hadamard(self.Dx, Ax),axis=1) \
               - 2*np.sum(fn.lm_gt_hadamard(self.Dx, Ay),axis=1) + dGdq
        return dfdq
    
    def dqdt_3D(self, q):

        dExdx = self.dExdx(q)
        dEydy = self.dEydy(q)
        dEzdy = self.dEydy(q)
        G = self.calcG(q)

        dqdt = -dExdx -dEydy -dEzdy + G
        return dqdt

    def dfdq_3D_div(self, q):
        # WARNING: Does not apply for explicit split forms

        Ax = self.dExdq(q)
        Ay = self.dEydq(q)
        Az = self.dEzdq(q)
        dGdq = self.dGdq(q)

        dfdq = - fn.gm_gm(self.Dx, Ax) - fn.gm_gm(self.Dy, Ay) - fn.gm_gm(self.Dz, Az) + dGdq
        return dfdq
    
    def dfdq_3D_had(self, q):

        Ax = self.Ax_tens(q) 
        Ay = self.Ay_tens(q) 
        Az = self.Az_tens(q) 
        dGdq = self.dGdq(q)

        dfdq = - 2*np.sum(fn.lm_gt_hadamard(self.Dx, Ax),axis=1) \
               - 2*np.sum(fn.lm_gt_hadamard(self.Dx, Ay),axis=1) \
               - 2*np.sum(fn.lm_gt_hadamard(self.Dz, Az),axis=1) + dGdq
        return dfdq
    

        

            
class DiffEqOverwrite:
# Allows you to overwrite the methods in the Diffeq class

    def __init__(self, diffeq_in, f_dqdt, f_dfdq, f_dfds,
                       f_cons_obj, n_cons_obj):

        self.dqdt = f_dqdt
        self.dfdq = f_dfdq
        self.dfds = f_dfds

        self.dfds = diffeq_in.dfds

        self.djdq = diffeq_in.djdq
        self.djds = diffeq_in.djds

        self.set_q0 = diffeq_in.set_q0
        self.plot_sol = diffeq_in.plot_sol
        self.calc_obj = diffeq_in.calc_obj
        self.calc_cons_obj = f_cons_obj

        self.n_obj = diffeq_in.n_obj
        self.n_cons_obj = n_cons_obj