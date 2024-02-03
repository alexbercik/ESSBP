#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:55:52 2020

@author: andremarchildon
"""

# Import the required modules
import time
import numpy as np

from Source.TimeMarch.TimeMarching import TimeMarching
from Source.DiffEq.DiffEqBase import DiffEqOverwrite


class OdeSolver:

    '''
    This class is used to solve ODEs and is also the base class for PdeSolver,
    which is the base class to solve PDEs using various discretizations.

    Method: solve
    --------------------------------------------------------------------------
    The method solve will be set to either calc_obj_w_para or
    calc_obj_no_para in the init method, depending on whether the ODE or PDE
    being solved is for an optmization problem that involves varying
    parameters.

    When the method solve is called varying parameters will be set if needed
    and then the method solve_main is called, which itself calls in order:
        1) solve_t_init
        2) solve_t_final
        3) calc_std_obj (if needed)

    The method solve_t_init uses the specified time marching method to march
    the soution in time by t_init. In this case, the solution is not saved at
    each time step and no objectives are calculated.

    The method solve_t_final will also march the solution in time and it will
    calculate any specified objectives.

    The method calc_std_obj is called if an estimate of the std of the
    objective(s) is requested when this class is initiated.

    Method: calc_obj
    --------------------------------------------------------------------------
    Calling this method will also end up calling the solve method but it
    returns the objective and the std or the objective. Meanwhile, if solve
    is called directly then the obj and its std can still be accessed with dot
    notation, ie NameOfClass.obj, where NameOfClass is the selected name of
    the class. This method is used for

    Method: calc_grad
    --------------------------------------------------------------------------
    This method should only be called after calling either the solve or
    calc_obj methods since the solution at each time step is required to
    calculate the gradient.
    '''

    # Initialize parameters
    q_sol = None        # Solution of the Diffeq
    obj = None          # Objective(s)
    time_t_final = None # Time to solve from t_init to t_final
    keep_all_ts = True  # whether to keep all time steps on solve

    def __init__(self,
            diffeq,                                 # Diffeq or functions to update diffeq
            tm_method, dt, t_final,                 # Time marching
            q0=None,                                # Initial solution
            obj_name=None, cons_obj_name=None,      # Other
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
        bool_calc_obj : bool, optional
            The objective(s) is calculated if this is set to True.
            The default is True, unless n_obj=0.
        cons_obj_name : str or tuple, optional
            The name or names of the conservation objective(s) to calculate.
            If None, no conservation objectives are calculated.
        bool_plot_sol : bool, optional
            The solution is plotted at each time step if this is set to True.
            The default is False.
        print_sol_norm : bool, optional
            The norm of the sol is printed at each time step is this is True.
            The default is False.
        '''

        ''' Add all inputs to the class '''

        self.diffeq = diffeq

        # Time marching
        self.tm_method = tm_method
        self.dt = dt
        self.t_final = t_final

        # Initial solution
        self.q0 = q0

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
                self.cons_obj_name = (self.cons_obj_name,)
            elif isinstance(cons_obj_name, tuple):
                self.n_cons_obj = len(self.cons_obj_name)
                # self.cons_obj_name = self.cons_obj_name
            else:
                print(cons_obj_name)
                raise Exception('The variable cons_obj_name has to be either a string or a tuple of strings')


        ''' Check all inputs '''

        # Time marching data
        if self.t_final == None:
            print('No t_final given. Checking diffeq for a default t_final.')
            self.t_final = self.diffeq.t_final
        if isinstance(self.t_final, int) or isinstance(self.t_final, float):
            self.n_ts_init = int(np.round(self.t_init / self.dt_init))
            self.n_ts = int(np.round(self.t_final / self.dt))
            if abs(self.n_ts - (self.t_final / self.dt)) > 1e-10:
                print('WARNING: Final time may not be exact! Check time step size.')
        elif self.t_final == 'steady':
            print('Indicated steady problem. Will use a convergence criteria to stop time march.')
            raise Exception('Have not coded this up yet!')
        else:
            print('ERROR: No t_final found.') # unsure whether to throw exception error...
            


    def solve_t_final(self, q_init):
        '''
        Parameters
        ----------
        q_init : np array
            The solution to start solving the DiffEq.

        Returns
        -------
        q_sol : np array
            Either the final solution or the solution at all time steps.
        obj : np array
            The evaluation of each objective over the entire simulation.
        obj_all_iter : np array
            The evaluation of each objective at each time step.
        cons_obj : np array
            The conserved variable such as conservation of energy.
        '''

        ''' Set up the solver '''


        tm_class = TimeMarching(self.diffeq, self.tm_method, self.keep_all_ts,
                                bool_plot_sol = self.bool_plot_sol,
                                bool_calc_obj = self.bool_calc_obj,
                                bool_calc_cons_obj = self.bool_calc_cons_obj,
                                print_sol_norm = self.print_sol_norm)

        ''' Solve '''

        q_sol =  tm_class.solve(q_init, self.dt, self.n_ts)
        obj = tm_class.obj
        obj_all_iter = tm_class.obj_all_iter
        cons_obj = tm_class.cons_obj

        return q_sol, obj, obj_all_iter, cons_obj

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
                        print_sol_norm = self.print_sol_norm)
        
        self.q_sol =  tm_class.solve(q0, self.dt, self.n_ts)
        self.obj = tm_class.obj
        self.obj_all_iter = tm_class.obj_all_iter
        self.cons_obj = tm_class.cons_obj

        end_time = time.time()
        self.simulation_time = end_time - stat_time

    
    def plot_cons_obj():
        print("ERROR: This method is not set up yet.")
        

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
        self.diffeq_unforced = self.diffeq
        q0 = self.diffeq_unforced.set_q0(q0_type)
        
        rhs = self.diffeq_unforced.dqdt(q0)
        new_dqdt = lambda q: self.diffeq_unforced.dqdt(q) - rhs
        
        self.diffeq = DiffEqOverwrite(self.diffeq, new_dqdt, self.diffeq.dfdq, self.diffeq.dfds,
                                           self.calc_cons_obj, self.n_cons_obj)
        
        self.diffeq_in.exact_sol = lambda *args: q0
        self.diffeq_in.has_exa_sol = True
    
        
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
                self.tm_solver = setattr(self, attribute, value)
            else:
                print("ERROR: solver has no attribute '{0}'. Ignoring.".format(attribute))

        self.__init__(self.diffeq,
                      self.tm_method, self.dt, self.t_final, t_init=self.t_init,
                      q0_set=self.q0_set, n_q0=self.n_q0,
                      n_int_ens=self.n_int_ens,
                      bool_calc_obj=self.bool_calc_obj, cons_obj_name=self.cons_obj_name,
                      bool_plot_sol=self.bool_plot_sol, print_sol_norm=self.print_sol_norm)
