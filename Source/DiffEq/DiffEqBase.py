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
import scipy.sparse as sp

from Source.Disc.FiniteDiff import FiniteDiff
from Source.Disc.MakeDgOp import MakeDgOp
from Source.Disc.MakeMesh import MakeMesh
import Source.Methods.Functions as fn
import quadpy as qp

'''
The classes in this file are inheritated by the classes for ODEs and PDEs.
The order of inheritance is as follows:
        PdeBaseCons       <- PdeBase    <- DiffEqBase
        OdeBase           <- DiffEqBase
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
The PDEs are solved in this form:
    The Diffeq:             dqdt + r = 0
    Time marching methods   dqdt = f(q)
    Therefore:              f = -r = Aq
    Jacobian                A := dfdq = -drdq
'''

class PdeBase:

    # Diffeq info
    xy = None               # Initiate nodal coordinates
    dim = None              # No. of dimensions
    npar = None             # No. of design parameters
    para_names = None       # Names of all the parameters
    has_exa_sol = False     # True if there is an exact solution for the DiffEq
    steady_sol = None       # If array, this is the steady solution for the DiffEq
    nn = None               # No. of nodes for the spatial discretization
    nen = None              # No. of nodes per element
    nelem = None            # No. of elements
    neq_node = None         # No. of equations per node

    # Ploting options
    plt_fig_size = (8,6)
    plt_style_exa_sol = {'color':'r','linestyle':'','marker':'+'}
    plt_style_sol = [{'color':'b','linestyle':'','marker':'s'},
                     {'color':'k','linestyle':'','marker':'o'},
                     {'color':'r','linestyle':'','marker':'X'},
                     {'color':'g','linestyle':'','marker':'D'}]
    plt_label_font_size = 15
    plt_var2plot_name = None
    plt_t_pause = 0.05
    plt_ymin = -2
    plt_ymax = 2

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


    def dqdt(self, *argv):
        raise Exception('This base method should not be called')

    def calc_ff(self, *argv):
        raise Exception('This base method should not be called')

    def dfdq(self, *argv):
        raise Exception('This base method should not be called')

    def dfds(self, *argv):
        raise Exception('This base method should not be called')

    def calc_rr(self, *argv):
        raise Exception('This base method should not be called')

    def drdq(self, *argv):
        raise Exception('This base method should not be called')

    def drds(self, *argv):
        raise Exception('This base method should not be called')

    def calcE(self, q):
        raise Exception('This base method should not be called')

    def dEdq_eig_abs(self, *argv):
        raise Exception('This base method should not be called')

    def calc_obj(self, *argv):
        raise Exception('This base method should not be called')

    def djdq(self, *argv):
        raise Exception('This base method should not be called')

    def djds(self, *argv):
        raise Exception('This base method should not be called')

    def set_sbp_op(self, *argv):
        raise Exception('This base method should not be called')

    def set_fd_op(self, *argv):
        raise Exception('This base method should not be called')

    def calc_sat(self, *argv):
        raise Exception('This base method should not be called')

    def dfdq_sat(self, *argv):
        raise Exception('This base method should not be called')

    def dfds_sat(self, *argv):
        raise Exception('This base method should not be called')
        
    def var2plot(self):
         raise Exception('This base method should not be called')
        
    def calc_LF_const(self):
        ''' Calculate the constant for the Lax-Friedrichs flux, also useful
        to set the CFL number. Equal to max(dEdq). Note this is approximate
        as it only estimates the LF constant from the initial condition.'''
        q = fn.check_q_shape(self.set_q0())
        C = np.max(np.abs(self.dEdq(q)))
        if isinstance(C,float):
            return C
        else:
            return 1

    def set_mesh(self, mesh):
        '''
        Purpose
        ----------
        Needed to calculate the initial solution and to calculate source terms
        '''

        self.mesh = mesh

        ''' Extract other parameters '''

        self.xy = self.mesh.xy
        self.xy_elem = self.mesh.xy_elem
        self.xy_ref = self.mesh.xy_op
        self.dx = self.mesh.dx
        self.xmin = self.mesh.xmin
        self.xmax = self.mesh.xmax
        self.isperiodic = self.mesh.isperiodic

        self.dom_len = self.xmax - self.xmin
        self.nn = self.mesh.nn
        self.nelem = self.mesh.nelem
        self.nen = self.mesh.nen
        self.qshape = (self.nen*self.neq_node,self.nelem)

        self.len_q = self.nn * self.neq_node # Length of the sol vector q

    def set_q0(self, q0_type=None, xy=None):
        '''
        Parameters
        ----------
        q0_type : str, optional
            Indiactes the type of initial solution.
            The default is None, in which case the default q0_type for the
            DiffEq is used.
        xy : np array, optional
            If not provided then the xy provided from set_mesh is used. One
            reason to provide the input xy is to calculate the exact solution
            at a later time, this is done for the linear convection eq.
            The default is None.
        Returns
        -------
        q0 : np array of floats
            The initial solution at the nodes xy.
        '''

        assert self.dim==1, 'set_q0 only setup for 1D'

        if q0_type is None:
            q0_type = self.q0_type

        if xy is None:
            xy = self.xy

        if xy.ndim == 2:
            xy = xy[:,0]

        if q0_type == 'GaussWave':
            # The initial condition is a Gauss-like wave
            k = np.abs(np.log(self.q0_gauss_wave_val_bc) / 0.5**2) # 0.5 since this is the max val of xy_mod

            mid_point = 0.5*(self.xmax + self.xmin)
            xy_mod = (xy - mid_point) / self.dom_len
            exp = -k * xy_mod**2
            q0 = np.exp(exp) * self.q0_max_q
        elif q0_type == 'SinWave':
            xy_scaled = (xy + self.xmin) / self.dom_len
            q0 = np.sin(2*np.pi * xy_scaled) * self.q0_max_q
        elif q0_type == 'Random':
            # Random numbers between -0.5 and 0.5
            q0 = np.random.rand(xy.size) -0.5
        elif q0_type == 'Constant':
            q0 = np.ones(xy.size)
        elif q0_type == 'GassnerSinWave_cont': # continuous
            q0 = np.sin(np.pi * xy - 0.7) + 2 # note in the paper it is incorrectly written (np.pi * (xy - 0.7))
        elif q0_type in ('GassnerSinWave','GassnerSinWave_coarse'): # discontinuous  
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
        
        # restructure in shape (nen,nelem), i.e. columns are each element
        return np.reshape(q0,(self.nen*self.neq_node,self.nelem),'F')


    def plot_sol(self, q, time=None, fig_no=1, plot_exa=True, plt_save_name=None,
                 show_fig=True,ymin=None, ymax=None, display_time=False):
        '''
        Purpose
        ----------
        Used to plot the solution in the time marching class
        '''
        assert self.dim == 1, 'This function is only setup for 1D PDEs'

        if plt.fignum_exists(fig_no):
            plt.clf()
        else:
            plt.figure(fig_no)

        if self.plt_var2plot_name is None:
            num_sol = q.flatten('F')
        else:
            num_sol = self.var2plot(q.flatten('F'))

        plt.plot(self.xy, num_sol, **self.plt_style_sol[0])

        if plot_exa and self.has_exa_sol:
            q_exa = self.exact_sol(time)

            if self.plt_var2plot_name is None:
                exa_sol = q_exa.flatten('F')
            else:
                exa_sol = self.var2plot(q_exa.flatten('F'))

            plt.plot(self.xy, exa_sol, **self.plt_style_exa_sol)
            
        plt.xlabel(r'$x$',fontsize=self.plt_label_font_size)

        if self.plt_var2plot_name is None: 
            plt.ylabel(r'$u$',fontsize=self.plt_label_font_size,rotation=0,
                       labelpad=15)
        else:
            ax = plt.gca()
            ax.set_ylabel(self.plt_var2plot_name, fontsize=self.plt_label_font_size)
            
        
        if display_time and (time is not None):
            ax = plt.gca()
            # define matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='white')
            ax.text(0.05, 0.95, r'$t=$ '+str(round(time,2))+' s', transform=ax.transAxes, 
                    fontsize=self.plt_label_font_size, verticalalignment='top', bbox=props)

        plt.ylim(ymin,ymax)
        plt.tight_layout()
        
        if plt_save_name is not None:
            plt.savefig(plt_save_name+'.eps', format='eps')
            
        if show_fig:
            plt.show()
            plt.pause(self.plt_t_pause)
            
            

    ''' Terms for the first derivative: E '''

    def dEdx(self, q):

        E = self.calcE(q)
        dEdx = self.der1 @ E
        return dEdx

    ''' Source term '''

    def calcG(self, q):
        return np.zeros(q.shape)

    def dGdq(self, q):
        nq = q.shape[0]
        return np.zeros((nq, nq))


class PdeBaseCons(PdeBase):

    ''' This base class is for PDEs of the form dqdt + dEdx = G '''

    def dqdt(self, q):

        dEdx = self.dEdx(q)
        G = self.calcG(q)

        dqdt = -dEdx + G
        return dqdt

    def set_sbp_op(self, dd_phys, qq_phys, hh_inv_phys, rrL, rrR):

        self.der1 = dd_phys
        self.qq = qq_phys
        self.hh_inv = hh_inv_phys
        self.rrL = rrL
        self.rrR = rrR

        self.nn_elem = self.hh_inv.shape[0]  # No. of nodes per element
    
    def set_dg_strong_op(self, dd_phys):
        
        self.der1 = dd_phys

    def set_fd_op(self, p, use_sparse=False):

        self.p = p

        # Construct the finite difference operator
        der1, der1_bcL, der1_bcR = FiniteDiff.der1(self.p, self.nn, self.dx, self.isperiodic)
        eye = sp.eye(self.neq_node, format="csr")
        if der1_bcL is None: der1_bcL = 0
        if der1_bcR is None: der1_bcR = 0
        self.der1 = sp.kron(der1, eye).todense()
        self.der1_bcL = sp.kron(der1_bcL, eye).todense()
        self.der1_bcR = sp.kron(der1_bcR, eye).todense()
            
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

        self.len_q = diffeq_in.len_q
        self.n_obj = diffeq_in.n_obj
        self.n_cons_obj = n_cons_obj