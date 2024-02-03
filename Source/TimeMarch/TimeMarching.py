#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:56:20 2019

@author: andremarchildon
"""

import numpy as np

from Source.TimeMarch.TimeMarchingRk import TimeMarchingRk
from Source.TimeMarch.TimeMarchingLms import TimeMarchingLms
from Source.TimeMarch.TimeMarchingOneStep import TimeMarchingOneStep
import time as tm


class TimeMarching(TimeMarchingRk, TimeMarchingLms, TimeMarchingOneStep):

    idx_print = 1 # Calc norm after this number of iterations

    def __init__(self, diffeq, tm_method,
                 keep_all_ts=True,
                 bool_plot_sol=False, fun_plot_sol=None,
                 bool_calc_cons_obj=False, fun_calc_cons_obj=None,
                 print_sol_norm=False, print_residual=False,
                 dqdt=None, dfdq=None):
        '''
        Parameters
        ----------
        diffeq : class
            The class for the differential equation.
        tm_method : str
            Indicates the time marching method.
        keep_all_ts : bool, optional
            The solution at all time steps is stored if this flag is set to true.
            The default is True.
        bool_plot_sol : bool, optional
            The solution is plotted if this flag is true.
            The default is False.
        fun_plot_sol : method, optional
            If not provided, the default solution plotter from diffeq is used.
            The default is None.
        bool_calc_cons_obj : bool, optional
            Quantities like conservation, energy norms, and entropy are
            calculated and stored in self.cons_obj if this flag is true.
            The default is False.
        fun_calc_cons_obj : method, optional
            If not povided, the method from diffeq is used to calculate the
            conservation objectives.
            The default is None.
        print_sol_norm : bool, optional
            The norm of the solution is printed if this flag is true.
            The default is False.
        print_residual : bool, optional
            The norm of the residual is printed if this flag is true.
            The default is False.
        '''

        ''' Add inputs to the class '''

        self.diffeq = diffeq
        self.tm_method = tm_method
        self.keep_all_ts = keep_all_ts
        self.bool_plot_sol = bool_plot_sol
        self.fun_plot_sol = fun_plot_sol
        self.print_sol_norm = print_sol_norm
        self.print_residual = print_residual
        self.bool_calc_cons_obj = bool_calc_cons_obj
        self.fun_calc_cons_obj = fun_calc_cons_obj

        # TODO: Add this capability
        if self.print_residual:
            raise Exception('This option is not yet setup')

        ''' Extract other required parameters '''

        if dqdt is None:
            self.f_dqdt = self.diffeq.dqdt
        else:
            self.f_dqdt = dqdt
        if dfdq is None:
            self.f_dfdq = self.diffeq.dfdq
        else:
            self.f_dfdq = dfdq

        if self.bool_plot_sol and self.fun_plot_sol is None:
            self.fun_plot_sol = self.diffeq.plot_sol

        if self.bool_calc_cons_obj:
            self.n_cons_obj = self.diffeq.n_cons_obj
            if self.n_cons_obj == 0:
                self.bool_calc_cons_obj = False
        if self.bool_calc_cons_obj and self.fun_calc_cons_obj is None:
            self.fun_calc_cons_obj = self.diffeq.calc_cons_obj

        ''' Initiate variables '''

        self.cons_obj = None

        ''' Set the solver '''

        if hasattr(self, self.tm_method):
            self.tm_solver = getattr(self, self.tm_method)
        else:
            raise Exception('The requested time marching method is not available')

    def solve(self, q0, dt, n_ts):
        '''
        Parameters
        ----------
        q0 : numpy array
            The initial solution.
        dt : float
            The size of the time step (assumed to be all the same size).
        n_ts : int, optional
            The number of time steps.
        Returns
        -------
        q_vec : numpy array
            One or all time steps, where each time step is a column in the
            2D array.
        '''

        if self.bool_calc_cons_obj:
            self.cons_obj = np.zeros((self.n_cons_obj, n_ts+1))

        self.shape_q = q0.shape
        self.len_q = q0.size

        return self.tm_solver(q0, dt, n_ts)

    def common(self, q, t_idx, n_ts, dt):
        '''
        Parameters
        ----------
        q : np float array
            The solution.
        t_idx : int
            Indicates the time step number.
        n_ts : int
            The total number of time steps.
        dt : float
            Size of the time step.
        '''

        if self.bool_plot_sol:
            tf = t_idx * dt
            self.fun_plot_sol(q, tf)

        if self.print_sol_norm:
            if (t_idx % self.idx_print == 0) or (t_idx == n_ts):
                norm_q = np.linalg.norm(q) / np.sqrt(np.size(q))
                print(f'i = {t_idx:4}, ||q|| = {norm_q:3.4}')

        if self.bool_calc_cons_obj:
            self.cons_obj[:, t_idx] = self.fun_calc_cons_obj(q)
            
        if t_idx == 10:
            sim_time = tm.time() - self.start_time
            rem_time = sim_time/10*(n_ts-10)
            print('... Estimating {0}:{1:02d}:{2:02d} to run.'.format(int(rem_time//3600),
                                                        int((rem_time//60)%60),int(rem_time%60)))         
            
        if (t_idx*100/n_ts).is_integer():
            if t_idx == 0:
                print('--- Beginning Simulation ---')
                self.start_time = tm.time()
            elif t_idx <= 10:
                pass
            elif t_idx == n_ts:
                sim_time = tm.time() - self.start_time
                suf = 'Complete.'
                h,m,s = int(sim_time//3600),int((sim_time//60)%60),int(sim_time%60)
                printProgressBar(t_idx, n_ts, prefix = 'Progress:', suffix = suf)
                print('... Took {0}:{1:02d}:{2:02d} to run.'.format(h,m,s))
            else:
                sim_time = tm.time() - self.start_time
                rem_time = sim_time/t_idx*(n_ts-t_idx)
                h,m,s = int(rem_time//3600),int((rem_time//60)%60),int(rem_time%60)
                suf = 'Complete. Estimating {0}:{1:02d}:{2:02d} remaining.'.format(h,m,s)
                #print('... {0}% Done. Estimating {1}:{2:02d}:{3:02d} remaining.'.format(pct,h,m,s))
                printProgressBar(t_idx, n_ts, prefix = 'Progress:', suffix = suf)

        #TODO: Add convergence tests to stop time marching if solution is 
        # sufficiently converged (rhs=0) or blown up.

    
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()