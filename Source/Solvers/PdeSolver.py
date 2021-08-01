#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:01:45 2020

@author: andremarchildon
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from Source.TimeMarch.TimeMarching import TimeMarching

class PdeSolver:

    
    # Initialize parameters
    q_sol = None        # Solution of the Diffeq
    obj = None          # Objective(s)
    cons_obj = None     # Conservation Objective(s)
    keep_all_ts = True  # whether to keep all time steps on solve

    def __init__(self, diffeq, settings,                            # Diffeq
                 tm_method, dt, t_final,                    # Time marching
                 q0=None,                                   # Initial solution
                 dim=1, p=2, disc_type='div',               # Discretization
                 surf_type='upwind', diss_type=None,
                 nelem=0, nen=0,  disc_nodes='lgl',
                 bc=None, xmin=0, xmax=1,     # Domain
                 obj_name=None, cons_obj_name=None,         # Other
                 bool_plot_sol=False, print_sol_norm=False):
        '''
        Parameters
        ----------
        diffeq : class
            The class of the differential equation.
        tm_method : str
            The type of time marching method to use
        dt : float
            Size of the time step.
        t_final : float
            The solution is evaluated up to this point.
        q0 : np array, optional
            If provided, this initial solution is used to start the solution.
            If not provided, the initial solution for the diffeq class is used.
            The default is None.
        p : int, optional
            Degree of the spatial discretization.
            The default is 2.
        disc_type : str, optional
            Indicates the type of spatial discretization that is used.
            The default is 'lgl'.
        nelem : int, optional (Not used for FD)
            The number of elements in the mesh.
            The default is 0.
        nen : int, optional (Not used for FD)
            The number of nodes per element. The actual number of nodes per 
            element is set according to whichever of p or nen results in the 
            largest number of nodes (See SbpQuadRule).
            The default is 0.
        bc : str or tuple
            Indicates boundary conditions, ex. 'periodic' if the mesh is periodic.
            The default is None, in which case it searches diffeq for default.
        xmin : float or (float,float)
            Min coordinate of the mesh, either x in 1D or (x,y) in 2D
        xmax : float or (float,float)
             Max coordinate of the mesh, either x in 1D or (x,y) in 2D
        xmax : float
             Factor by which to warp the 2D mesh. Typically use 0.2
             The defualt is 0.
        bool_calc_obj : bool, optional
            The objective(s) is calculated if this is set to True.
            The default is True, unless n_obj=0.
        cons_obj_name : str or tuple, optional
            The name or names of the conservation objective(s) to calculate.
            If None, no conservation objectives are calculated.
            The default is None.
        bool_plot_sol : bool, optional
            The solution is plotted at each time step if this is set to True.
            The default is False.
        print_sol_norm : bool, optional
            The norm of the sol is printed at each time step if this is True.
            The default is False.
        '''

        ''' Add all inputs to the class '''

        self.diffeq = diffeq
        
        self.settings = settings
        
        ''' Set default settings if not given '''
        self.settings.setdefault('warp_factor',0.) 
        # called by Disc.MakeMesh. Warps / stretches mesh.
        self.settings.setdefault('warp_type','default')
        # called by Disc.MakeMesh. Options: 'defualt', 'papers', 'quad'
        self.settings.setdefault('metric_method','exact') 
        # called by Disc.MakeMesh.get_jac_metrics. Options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact'
        self.settings.setdefault('bdy_metric_method','exact') 
        # called by Disc.MakeMesh.get_jac_metrics. Options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact', 'interpolate'
        self.settings.setdefault('use_optz_metrics',True) 
        # called by Disc.MakeMesh.get_jac_metrics. Uses optimized metrics for free stream preservation.
        self.settings.setdefault('calc_exact_metrics',False) 
        # called by Disc.MakeMesh.get_jac_metrics. Calculates exact metrics alongside regular method..
        self.settings.setdefault('metric_optz_method','default') 
        # called by Disc.MakeMesh.get_jac_metrics. Define the metric optimization procedure.

        # Time marching
        self.tm_method = tm_method.lower()
        self.dt = dt
        self.t_final = t_final

        # Initial solution
        self.q0 = q0
        
        # Discretization
        self.dim = dim
        if disc_type.lower() == 'div' or disc_type.lower() == 'divergence':
            self.disc_type = 'div'
        elif disc_type.lower() == 'had' or disc_type.lower() == 'hadamard':
            self.disc_type = 'had'
        else: raise Exception('Discretization type not understood. Try div or had.')
        self.disc_nodes = disc_nodes.lower()
        self.neq_node = self.diffeq.neq_node
        self.surf_type = surf_type.lower()
        self.diss_type = diss_type
        self.pde_order = self.diffeq.pde_order

        assert isinstance(p, int), 'p must be an integer'
        self.p = p
        assert isinstance(nen, int), 'nen must be an integer'
        self.nen = nen
        
        if self.dim == 1:
            assert isinstance(nelem, int), 'nelem must be an integer'
            assert isinstance(xmin, float) or isinstance(xmin, int), 'xmin must be a float or integer'
            assert isinstance(xmax, float) or isinstance(xmax, int), 'xmax must be a float or integer'
            self.nelem = nelem
            self.xmin = xmin
            self.xmax = xmax

        elif self.dim == 2:
            if isinstance(nelem, int):
                self.nelem = (nelem, nelem)
            elif (isinstance(nelem, tuple) and len(nelem)==2):
                self.nelem = nelem
            else: 
                'nelem must be an integer or tuple of integers. Inputted: ', nelem
            if isinstance(xmin, float) or isinstance(xmin, int):
                self.xmin = (xmin, xmin)
            elif (isinstance(xmin, tuple) and len(xmin)==2):
                self.xmin = xmin
            else: 
                'xmin must be a float/integer or tuple of float/integers. Inputted: ', xmin
            if isinstance(xmax, float) or isinstance(xmax, int):
                self.xmax = (xmax, xmax)
            elif (isinstance(xmax, tuple) and len(xmax)==2):
                self.xmax = xmax
            else: 
                'xmax must be a float/integer or tuple of float/integers. Inputted: ', xmax
                
        elif self.dim == 3:
            if isinstance(nelem, int):
                self.nelem = (nelem, nelem, nelem)
            elif (isinstance(nelem, tuple) and len(nelem)==3):
                self.nelem = nelem
            else: 
                'nelem must be an integer or tuple of integers. Inputted: ', nelem
            if isinstance(xmin, float) or isinstance(xmin, int):
                self.xmin = (xmin, xmin, xmin)
            elif (isinstance(xmin, tuple) and len(xmin)==3):
                self.xmin = xmin
            else: 
                'xmin must be a float/integer or tuple of float/integers. Inputted: ', xmin
            if isinstance(xmax, float) or isinstance(xmax, int):
                self.xmax = (xmax, xmax, xmax)
            elif (isinstance(xmax, tuple) and len(xmax)==3):
                self.xmax = xmax
            else: 
                'xmax must be a float/integer or tuple of float/integers. Inputted: ', xmax
        else: raise Exception('Only set up currently for 1D, 2D, and 3D')

        # Other
        self.obj_name = obj_name
        self.cons_obj_name = cons_obj_name
        self.bool_plot_sol = bool_plot_sol
        self.print_sol_norm = print_sol_norm

        ''' Extract other attributes '''

        self.n_obj = self.diffeq.n_obj
        if self.n_obj == 0: self.obj_name = None
        
        if self.obj_name == None:
            self.bool_calc_obj = False
        else:
            self.bool_calc_obj = True

        if self.cons_obj_name is None:
            self.n_cons_obj = 0
            self.cons_obj_name = None
            self.bool_calc_cons_obj = False
        else:
            self.bool_calc_cons_obj = True
            # Standardize format for the name of the objective(s)
            if isinstance(self.cons_obj_name, str):
                self.n_cons_obj = 1
                self.cons_obj_name = (self.cons_obj_name.lower(),)
            elif isinstance(cons_obj_name, tuple):
                self.n_cons_obj = len(self.cons_obj_name)
            else:
                print(cons_obj_name)
                raise Exception('The variable cons_obj_name has to be either a string or a tuple of strings')

        self.diffeq.calc_cons_obj = self.calc_cons_obj
        self.diffeq.n_cons_obj = self.n_cons_obj

        ''' Check all inputs '''

        # Time marching data
        if self.t_final == None:
            print('No t_final given. Checking diffeq for a default t_final.')
            self.t_final = self.diffeq.t_final
        if isinstance(self.t_final, int) or isinstance(self.t_final, float):
            self.n_ts = int(np.round(self.t_final / self.dt))
            if abs(self.n_ts - (self.t_final / self.dt)) > 1e-10:
                dt_old = np.copy(self.dt)
                self.dt = self.t_final/self.n_ts
                print('WARNING: To ensure final time is exact, changing dt from {0} to {1}'.format(dt_old,self.dt))
        elif self.t_final == 'steady':
            print('Indicated steady problem. Will use a convergence criteria to stop time march.')
            raise Exception('Have not coded this up yet!')
        else:
            print('ERROR: No t_final found.') # unsure whether to throw exception error...

        # Domain
        if bc == None:
            print('No boundary conditions provided. Checking diffeq for a default.')
            self.bc = diffeq.bc
        else: self.bc = bc
        if self.dim == 1:
            assert isinstance(self.bc, str), 'bc must be a string'
            if self.bc == 'periodic': self.periodic = True
        elif self.dim == 2:
            if isinstance(self.bc, str):
                self.bc = (self.bc, self.bc)
            assert isinstance(self.bc, tuple) and len(self.bc)==2, 'bc must be a string or tuple of strings'
            self.periodic = [False,False]
            if self.bc[0] == 'periodic': self.periodic[0] = True
            if self.bc[1] == 'periodic': self.periodic[1] = True               
        elif self.dim == 3:
            if isinstance(self.bc, str):
                self.bc = (self.bc, self.bc, self.bc)
            assert isinstance(self.bc, tuple) and len(self.bc)==3, 'bc must be a string or tuple of strings'
            self.periodic = [False,False,False]
            if self.bc[0] == 'periodic': self.periodic[0] = True
            if self.bc[1] == 'periodic': self.periodic[1] = True
            if self.bc[2] == 'periodic': self.periodic[2] = True
            
        
        ''' Call the discretisation specific method to finish the class initialization '''
        self.init_disc_specific()        
        
        ''' Sanity Checks '''
        
        # time step stability
        q = self.diffeq.set_q0()
        if self.dim==1:
            LFconst = np.max(self.diffeq.maxeig_dEdq(q))
            dt = 0.1*(self.xmax-self.xmin)/self.nn/LFconst
            if self.dt > dt:
                print('WARNING: time step dt may not be small enough to remain stable.')
                print('Assuming CFL = 0.1 and max wave speed = {0:.2g}, try dt < {1:.2g}'.format(LFconst, dt))
        elif self.dim==2:
            LFconstx = np.max(self.diffeq.maxeig_dExdq(q))
            LFconsty = np.max(self.diffeq.maxeig_dEydq(q))
            const = np.sqrt(LFconstx**2 + LFconsty**2)
            dx = min((self.xmax[0]-self.xmin[0])/self.nn[0],(self.xmax[1]-self.xmin[1])/self.nn[1])
            dt = 0.1*dx/const
            if self.dt > dt:
                print('WARNING: time step dt may not be small enough to remain stable.')
                print('Assuming CFL = 0.1 and max wave speed = ({0:.2g}, {1:.2g}), try dt < {2:.2g}'.format(LFconstx, LFconsty, dt))
        elif self.dim==3:
            LFconstx = np.max(self.diffeq.maxeig_dExdq(q))
            LFconsty = np.max(self.diffeq.maxeig_dEydq(q))
            LFconstz = np.max(self.diffeq.maxeig_dEzdq(q))
            const = np.sqrt(LFconstx**2 + LFconsty**2 + LFconstz**2)
            dx = min((self.xmax[0]-self.xmin[0])/self.nn[0],(self.xmax[1]-self.xmin[1])/self.nn[1])
            dt = 0.1*dx/const
            if self.dt > dt:
                print('WARNING: time step dt may not be small enough to remain stable.')
                print('Assuming CFL = 0.1 and max wave speed = ({0:.2g}, {1:.2g}, {2:.2g}), try dt < {3:.2g}'.format(LFconstx, LFconsty, LFconstz, dt))
                
        # Free Stream Preservation
        test = self.dqdt(np.ones(self.qshape))
        if np.max(abs(test))>1e-12:
            print('WARNING: Free Stream is not preserved. Check Metrics and/or SAT discretization.')
            print('         Free Stream is violated by a maximum of {0:.2g}'.format(np.max(abs(test))))

    def solve(self, q0_in=None, q0_idx=None):

        ''' Solve to calculate the solution and the objective '''

        stat_time = time.time()
        
        if q0_in is not None:
            q0 = q0_in
        else:
            q0 = self.diffeq.set_q0()
        
        tm_class = TimeMarching(self.diffeq, self.tm_method, self.keep_all_ts,
                        bool_plot_sol = self.bool_plot_sol,
                        bool_calc_obj = self.bool_calc_obj,
                        bool_calc_cons_obj = self.bool_calc_cons_obj,
                        print_sol_norm = self.print_sol_norm,
                        dqdt=self.dqdt, dfdq=self.dfdq)
        
        self.q_sol =  tm_class.solve(q0, self.dt, self.n_ts)
        self.obj = tm_class.obj
        self.obj_all_iter = tm_class.obj_all_iter
        self.cons_obj = tm_class.cons_obj

        end_time = time.time()
        self.simulation_time = end_time - stat_time
    
    def calc_cons_obj(self, q):
        '''
        Purpose
        ----------
        Calculate the conservation objectives, such as energy, conservation,
        entropy, etc. This calls on smaller specific functions defined for
        each spatial discretization

        Parameters
        ----------
        q : np array
            The global solution at the global nodes.

        Returns
        -------
        A vector of size n_cons_obj with the desired conservation objectives
        '''

        cons_obj = np.zeros(self.n_cons_obj)
        
        if any('der' in name.lower() for name in self.cons_obj_name):
            dqdt = self.dqdt(q) #TODO: Wildly inneficient. Can i get this from tm class?

        for i in range(self.n_cons_obj):
            cons_obj_name_i = self.cons_obj_name[i].lower()

            if cons_obj_name_i == 'energy':
                cons_obj[i] = self.energy(q)
            elif cons_obj_name_i == 'entropy':
                cons_obj[i] = self.entropy(q)
            elif cons_obj_name_i == 'conservation':
                cons_obj[i] = self.conservation(q)
            elif cons_obj_name_i == 'energy_der':
                cons_obj[i] = self.energy_der(q,dqdt)
            elif cons_obj_name_i == 'conservation_der':
                cons_obj[i] = self.conservation_der(dqdt)
            else:
                raise Exception('Unknown conservation objective function')

        return cons_obj
    
    
    def norm(self, q):
        ''' calculate the energy norm q.T @ H @ q given a global q'''
        return np.sqrt(float(self.energy(q)))
    
    def calc_error(self, q=None, tf=None, method=None, use_all_t=False):
        '''
        Purpose
        ----------
        Calculate the error of the solution using a defined method. This calls
        on smaller specific functions also defined below.

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
                'SBP' : the SBP error sqrt((q-q_ex).T @ H @ (q-q_ex))
                'Rms' : the standard root mean square error in L2 norm
                'Boundary' : The simple periodic boundary error | q_1 - q_N |
                'Truncation-SBP' : the SBP error but instead using er = dqdt(q-q_ex)
                'Truncation-Rms' : the Rms error but instead using er = dqdt(q-q_ex)
                'max_diff' : np.max(abs(q-q_ex))
        use_all_t : bool (optional)
            If True, it calls calc_error recursively for all time steps 
            (len of q_sol) and stores the results in a 1D array.
            The default is False.
        '''
        if tf == None: tf = self.t_final
        if q is None:
            if self.q_sol.ndim == 2: q = self.q_sol
            elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
        if method == None:
            if self.disc_type == 'FD': method = 'Rms'
            else: method = 'SBP'
        if use_all_t:
            assert(self.q_sol.ndim == 3),'ERROR: There is only one time step for q_sol.'
            steps = np.shape(self.q_sol)[2]
            errors = np.zeros(steps)
            times = np.linspace(0,self.t_final,steps)
            # TODO: generalize this for varying time steps, if we ever implement this
            for i in range(steps):
                errors[i] = self.calc_error(self.q_sol[:,:,i],times[i],method=method)
            return errors

        # determine error to use
        if method == 'SBP' or method == 'Rms':
            q_exa = self.diffeq.exact_sol(tf)
            error = q - q_exa
        elif method == 'max diff':
            q_exa = self.diffeq.exact_sol(tf)
            error = np.max(abs(q-q_exa))
        elif method == 'Boundary':
            error = abs(q[0]-q[-1])
        elif method == 'Truncation-SBP' or method == 'Truncation-Rms':
            q_exa = self.diffeq.exact_sol(tf)
            error = self.diffeq.dqdt(q - q_exa)
        else:
            raise Exception('Unknown error method. Use one of: H-norm, Rms, Boundary, Truncation-SBP, Truncation-Rms')

        # if we still need to apply a norm, do it
        if method == 'SBP' or method == 'Truncation-SBP':
            error = self.norm(error)
        elif method == 'Rms' or method == 'Truncation-Rms':
            error = np.linalg.norm(error) / np.sqrt(self.nn)
        return error

    def reset(self,variables=[]):
        """
        Purpose
        ----------
        Allows one to reset the solver instance. It will read the parameters
        in variables, then call __init__ with the updated arguments.

        Parameters
        ----------
        variables : list of tuples
        The attributes to be updated for the solver. Default is empty.
        tuples must be in form ('name_of_attribute',value)
        """
        for i in range(len(variables)):
            attribute , value = variables[i]
            if hasattr(self, attribute):
                setattr(self, attribute, value)
            else:
                print("ERROR: solver has no attribute '{0}'. Ignoring.".format(attribute))

        self.__init__(self.diffeq, self.settings, 
                      self.tm_method, self.dt, self.t_final, 
                      q0=self.q0, 
                      dim=self.dim, p=self.p, disc_type=self.disc_type,
                      surf_type=self.surf_type, diss_type=self.diss_type,
                      nelem=self.nelem, nen=self.nen, disc_nodes=self.disc_nodes,
                      bc=self.bc, xmin=self.xmin, xmax=self.xmax,
                      obj_name=self.bool_calc_obj, cons_obj_name=self.cons_obj_name,
                      bool_plot_sol=self.bool_plot_sol, print_sol_norm=self.print_sol_norm)

    
    def check_eigs(self, q=None, plot_eigs=True, returnA=False, exact_dfdq=True,
                   step=1.0e-4, tol=1.0e-10, plt_save_name=None, ymin=None, ymax=None,
                   xmin=None, xmax=None, time=None, display_time=False, title=None, **kargs):
        '''
        Call on self.diffeq.dqdt to check the stability of the spatial operator
        at a particular state q using central finite differences (approximate!).
        A_ij \approx ( dqdt(q + e_j*tol) - dqdt(q - e_j*tol) ) / 2*tol

        Parameters
        ----------
        q : numpy array, optional
            Current state (must be compatible with self.diffeq.dqdt(q))
            Default is last solution.
        plot_eigs : boolean, optional
            Flag whether to plot eigenvalues. The default is True.
        returnA : boolean, optional
            Flag whether to return the spatial operator. The default is False.
        step : float, optional
            step size for finite difference approximation. The default is 1.0e-2.
        tol : float, optional
            The tolerance allowed for detecting postive eigenvaules. 
            The default is 1.0e-10.
        plt_save_name : str, optional
            name of saved file, without file extension
            The defualt is None. 
        ymin, ymax, xmin, xmax : str, optional
            Y and X axis limits.
            The defualt is None.
        **kargs : nothing
            This just absorbs additional unecessary keword arguments

        Returns
        -------
        A : 2D numpy array
            Approximate RHS spatial operator

        '''
        if q is None:
            if hasattr(self, 'q_sol'):
                if self.q_sol is not None:
                    if self.q_sol.ndim == 2: q = self.q_sol
                    elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
                else:
                    q = self.diffeq.set_q0()
            else:
                q = self.diffeq.set_q0()
        
        if exact_dfdq:
            try:  
                A = self.dfdq(q)
            except:
                exact_dfdq = False
                print('WARNING: self.dfdq(q) returned errors. Using finite diff approximation')
        if not exact_dfdq:
            nen,nelem = q.shape      
            A = np.zeros((nelem*nen,nelem*nen))      
            assert(self.nn*self.neq_node==q.size),"ERROR: sizes don't match"           
            for i in range(nen):
                for j in range(nelem):
                    ei = np.zeros((nen,nelem))
                    ei[i,j] = 1.*step
                    q_r = self.dqdt(q+ei).flatten('F')
                    q_l = self.dqdt(q-ei).flatten('F')
                    idx = np.where(ei.flatten('F')>step/10)[0][0]
                    A[:,idx] = (q_r - q_l)/(2*step)
        
        eigs = np.linalg.eigvals(A)
        print('Max real component =',max(eigs.real))
        if max(eigs.real) < tol:
            print('Max eigenvalue within tolerance. STABLE.')
        else:
            print('Max eigenvalue exceeds tolerance. UNSTABLE.')
            
        if plot_eigs:
            X = [x.real for x in eigs]
            Y = [x.imag for x in eigs]
            plt.scatter(X,Y, color='red')
            plt.axvline(x=0, linewidth=1, linestyle='--', color='black')
            plt.xlabel(r'Real Component ($x<0$ for stability)',fontsize=14)
            plt.ylabel(r'Imaginary Component',fontsize=14)
            if title is None:
                plt.title(r'Eigenvalues',fontsize=16)
            else:
                plt.title(title,fontsize=16)
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            if display_time and (time is not None):
                ax = plt.gca()
                # define matplotlib.patch.Patch properties
                #props = dict(boxstyle='round', facecolor='white')
                ax.text(0.05, 0.95, r'$t=$ '+str(round(time,2))+' s', transform=ax.transAxes, 
                        fontsize=14, verticalalignment='top')           
            if plt_save_name is not None:
                plt.savefig(plt_save_name+'.eps', format='eps')
            plt.show()
        
        if returnA:
            return A

    def plot_cons_obj(self,savefile=None):
        '''
        Plot the conservation objectives

        Parameters
        ----------
        savefile : string, optional
            File name under which to save plots. The default is None.
        '''
        for i in range(self.n_cons_obj):
            cons_obj_name_i = self.cons_obj_name[i].lower()
            norm = self.cons_obj[i,0]
            plt.figure(figsize=(6,4))
            plt.xlabel(r'Time $t$',fontsize=16)

            if cons_obj_name_i == 'energy':
                plt.title(r'Change in Energy',fontsize=18)
                plt.ylabel(r'$\vert \vert u(x,t)^2 \vert \vert_H$ - $\vert \vert u_0(x)^2 \vert \vert_H$',fontsize=16)
                plt.plot(np.linspace(0,self.t_final,len(self.cons_obj[i])),self.cons_obj[i]-norm) 
                #plt.ylabel(r'$- ( \vert \vert u(x,t)^2 \vert \vert_H$ - $\vert \vert u_0(x)^2 \vert \vert_H )$',fontsize=16)
                #plt.plot(np.linspace(0,self.t_final,len(self.cons_obj[i])),abs(self.cons_obj[i]-norm)) 
                #plt.yscale('log')
                #plt.gca().invert_yaxis()
                
            elif cons_obj_name_i == 'entropy':
                plt.title(r'Change in Entropy',fontsize=18)
                plt.ylabel(r'$ 1 H s(x,t) - 1 H s(x,0) $',fontsize=16)
                plt.plot(np.linspace(0,self.t_final,len(self.cons_obj[i])),self.cons_obj[i]-norm) 
    
            elif cons_obj_name_i == 'conservation':
                plt.title(r'Change in Conservation',fontsize=18)
                plt.ylabel(r'$\vert \vert u(x,t) \vert \vert_H$ - $\vert \vert u_0(x) \vert \vert_H$',fontsize=16)
                plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
                plt.plot(np.linspace(0,self.t_final,len(self.cons_obj[i])),self.cons_obj[i]-norm)              
            
            else:
                raise Exception('Unknown conservation objective function')
                
            if savefile is not None:
                plt.savefig(savefile+'_'+cons_obj_name_i+'.jpg',dpi=600)

    def force_steady_solution(self, q0_type=None):
        '''
        Modifies the dqdt function to force a steady manufactured solution 
        such that the forcing term is always the negative of the RHS residual
        of the initial condition. If the initial condition is then perturbed,
        this allows for analysis of linearization states. Otherwise, nothing
        should happen and the solution should be steady.
        
        Parameters
        ----------
        q0_type : str, optional
            Indiactes the type of initial solution.
            The default is None, in which case the default q0_type for the
            DiffEq is used.
        '''

        q0 = self.diffeq.set_q0(q0_type)
        
        if self.disc_type == 'fd':
            rhs = self.diffeq.dqdt(q0)
            self.dqdt = lambda q: self.diffeq.dqdt(q) - rhs
        elif self.disc_type == 'div' or self.disc_type == 'had':
            rhs = self.sbp_dqdt(q0)
            self.dqdt = lambda q: self.sbp_dqdt(q) - rhs           
        elif self.disc_type == 'dg':
            if self.weak_form:
                rhs = self.dg_dqdt_weak(q0)
                self.dqdt = lambda q: self.dg_dqdt_weak(q) - rhs  
            else:
                rhs = self.dg_dqdt_strong(q0)
                self.dqdt = lambda q: self.dg_dqdt_strong(q) - rhs  
        else:
            raise Exception('Invalid discretization type')
        
        self.diffeq.exact_sol = lambda *args: q0
        self.diffeq.has_exa_sol = True
        
    def perturb_q0(self, eigmode=True, randnoise=False, ampli=0.001):
        '''
        Modify self.diffeq.set_q0() to add a pertubation according to largest
        real eigenvalue of the approximated spatial operator. This is what is
        in Gassner et. al. 2020 (Stability issues of entropy-stable...). It
        also can add a random noise pertubation.

        Parameters
        ----------
        eigmode : bool, optional
            Whether to use pertubation of largest eigenmode. The default is True.
        randnoise : bool, optional
            Whether to use random noise pertubation. The default is False.
        ampli : float, optional
            Max amplitude of pertubation. The default is 0.001.
        '''
        q0 = self.diffeq.set_q0() # save initial condition
        pert = np.zeros(q0.shape) # initialize pertubation
        if eigmode and randnoise: # if we use both, adjust amplitude
            ampli = ampli/2
        
        # This part adds a pertubation based on the largest real eigenmode
        if eigmode:
            A = self.check_eigs(q=q0, plot_eigs=False,returnA=True)
            eigvals, eigvecs = np.linalg.eig(A)
            idx = np.argmax(eigvals.real)
            eigmode = eigvecs[:,idx]
            #np.savetxt('eigenmode.csv', eigmode.real, delimiter=',')
            #eigenmode = np.loadtxt('eigmode.csv', delimiter=',')        
            pert += ampli*np.reshape(eigmode.real,q0.shape,'F')/np.max(np.abs(eigmode.real))
        
        # This part adds a random noise pertubation '''
        if randnoise:
            pert += ampli*2*(np.random.rand(*q0.shape)-0.5)
        
        # Now overwrite initial condition function
        self.diffeq.set_q0 = lambda *args: q0 + pert
        
    def plot_sol(self, q=None, **kwargs):
        ''' simply calls the plotting function from diffeq for final sol '''
        if q is None:
            if self.q_sol.ndim == 2: q = self.q_sol
            elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
        if 'time' not in kwargs: kwargs['time']=self.t_final
        if 'plot_exa' not in kwargs: kwargs['plot_exa']=True
        self.diffeq.plot_sol(q, **kwargs)
        
    def plot_error(self, method=None, savefile=None, extra_fn=None, extra_label=None, title=None):
        ''' plot the error from all time steps '''
        errors = self.calc_error(method=method, use_all_t=True)
        steps = np.shape(self.q_sol)[2]
        times = np.linspace(0,self.t_final,steps)
        plt.figure(figsize=(6,4))
        plt.ylabel(r"{0} Error".format(method),fontsize=16)
        plt.xlabel(r"Time",fontsize=16)
        plt.plot(times,errors,label='Error')
        if extra_fn is not None:
            plt.plot(times,extra_fn(times),label=extra_label)
        plt.yscale('log')
        if extra_label is not None:
            plt.legend(loc='best',fontsize=14)
        if title is not None:
            plt.title(title,fontsize=18)
        if savefile is not None:
            plt.savefig(savefile+'.eps', format='eps')
    
        
        
        
        