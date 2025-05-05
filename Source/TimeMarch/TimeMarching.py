#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:56:20 2019

@author: bercik
"""

import numpy as np

from Source.TimeMarch.TimeMarchingRk import TimeMarchingRk
#from Source.TimeMarch.TimeMarchingLms import TimeMarchingLms
#from Source.TimeMarch.TimeMarchingOneStep import TimeMarchingOneStep
import time as tm


class TimeMarching(TimeMarchingRk):

    idx_print = 100 # Calc norm after this number of iterations
    return_failed_itn = False # include the last iteration in the return if it failed?

    def __init__(self, diffeq, tm_method,
                 keep_all_ts=True, skip_ts=0,
                 bool_plot_sol=False, fun_plot_sol=None,
                 bool_calc_cons_obj=False, fun_calc_cons_obj=None,
                 print_sol_norm=False, print_residual=False,
                 check_resid_conv=False,
                 dqdt=None, dfdq=None,
                 rtol=None, atol=None):
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
        skip_ts : int, optional
            If keep_all_ts = True, every how steps should we skip before saving the solution? 
            ex. if skipsteps=5, every 6th time step will be saved
            The default is 0.
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
        check_resid_conv : bool, optional
            Whether or not to check for residual convergence to exit.
            The default is False (as it should be for unsteady)
        '''

        ''' Add inputs to the class '''

        self.diffeq = diffeq
        self.tm_method = tm_method
        self.keep_all_ts = keep_all_ts
        self.skip_ts = skip_ts
        assert(isinstance(self.skip_ts, int)),"skip_ts must be an integer"
        self.bool_plot_sol = bool_plot_sol
        self.fun_plot_sol = fun_plot_sol
        self.print_sol_norm = print_sol_norm
        self.print_residual = print_residual
        self.bool_calc_cons_obj = bool_calc_cons_obj
        self.fun_calc_cons_obj = fun_calc_cons_obj
        self.check_resid_conv = check_resid_conv
        self.quitsim = False
        self.enforce_positivity = self.diffeq.enforce_positivity
        self.print_progress = True

        ''' Extract other required parameters '''

        if dqdt is None:
            self.dqdt = self.diffeq.dqdt
        else:
            self.dqdt = dqdt
        if dfdq is None:
            self.dfdq = self.diffeq.dfdq
        else:
            self.dfdq = dfdq

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
        
        if self.tm_method == 'rk8':
            if rtol is not None:
                self.rtol = rtol
            else:
                print('WARNING: rtol not set for rk8. Setting to 1e-12.')
                self.rtol = 1e-12
            if atol is not None:
                self.atol = atol
            else:
                print('WARNING: atol not set for rk8. Setting to 1e-12.')
                self.atol = 1e-12

            if self.keep_all_ts or self.bool_calc_cons_obj:
                assert self.diffeq.cons_obj_name is not None, \
                    'RK8 requires time to be in the cons_obj_name list if keep_all_ts=True.'
                # we need to track time as part of cons_obj
                assert any(name.lower() == 'time' for name in self.diffeq.cons_obj_name), \
                    'RK8 requires time to be in the cons_obj_name list.'

    def solve(self, q0, dt, n_ts, t0):
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
            3D array.
        '''

        self.qshape = q0.shape
        self.len_q = q0.size
        self.t_initial = t0
        self.t_final = t0 + dt*n_ts

        return self.tm_solver(q0, dt, n_ts, t0)

    def common(self, q, q_sol, t_idx, n_ts, dt, dqdt, time=None):
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
        if time is None: time = t_idx * dt + self.t_initial

        if np.any(np.isnan(q)):
            print('\n ERROR: there are undefined values for q at t =',time,'t_idx =', t_idx)
            self.quitsim = True
            self.failsim = True
            return

        if self.enforce_positivity:
            if self.diffeq.check_positivity(q):
                print('\n ERROR: there are negative values for q at t =',time,'t_idx =', t_idx)
                self.quitsim = True
                self.failsim = True
                return

        if self.keep_all_ts or self.bool_calc_cons_obj:
            if self.use_time_frames:
                timediff = abs(time - self.frame_target_time)
                if timediff < self.frame_current_timediff:
                    # we are closer to the target time. Save the current solution at current index.
                    self.frame_current_timediff = timediff
                else:
                    # we have started moving away from the target time. Update the target time.
                    self.frame_idx += 1
                    self.frame_target_time = self.t_final / self.nframes * self.frame_idx + self.t_initial
                    self.frame_current_timediff = abs(time - self.frame_target_time)

                if self.keep_all_ts:
                    q_sol[:, :, self.frame_idx] = q
                if self.bool_calc_cons_obj:
                    self.cons_obj[:, self.frame_idx] = self.fun_calc_cons_obj(q,time,dqdt)

            elif int(t_idx) % (self.skip_ts + 1) == 0:
                self.frame_idx = int(t_idx/(self.skip_ts+1))
                if self.keep_all_ts:
                    q_sol[:, :, self.frame_idx] = q
                if self.bool_calc_cons_obj:
                    self.cons_obj[:, self.frame_idx] = self.fun_calc_cons_obj(q,time,dqdt)

        if self.bool_plot_sol:
            self.fun_plot_sol(q, time)

        if self.print_sol_norm:
            if (t_idx % self.idx_print == 0) or (t_idx == n_ts):
                norm_q = np.linalg.norm(q) / np.sqrt(np.size(q))
                print(f'i = {t_idx:4}, ||q|| = {norm_q:3.4}')

        if self.check_resid_conv or self.print_residual:
            resid = np.linalg.norm(dqdt)
            resid = resid*resid # I actually want the norm squared

        if t_idx > 10 and self.print_progress:
            if t_idx % max(1,(n_ts // 100)) == 0:
                sim_time = tm.time() - self.start_time
                rem_time = sim_time/(t_idx-1)*(n_ts-t_idx-1)
                h,m,s = int(rem_time//3600),int((rem_time//60)%60),int(rem_time%60)
                suf = 'Complete. Estimating {0}:{1:02d}:{2:02d} remaining.'.format(h,m,s)
                #print('... {0}% Done. Estimating {1}:{2:02d}:{3:02d} remaining.'.format(pct,h,m,s))
                if self.print_residual:
                    suf += ' Resid = {0:.1E}'.format(resid)
                printProgressBar(t_idx, n_ts, prefix = 'Progress:', suffix = suf)
                
        elif t_idx == 0:
            print('--- Beginning Simulation ---')
        elif t_idx == 1:
            self.start_time = tm.time()
        elif t_idx == 10:
            sim_time = tm.time() - self.start_time
            rem_time = sim_time/9*(n_ts-9)
            print('... Estimating {0}:{1:02d}:{2:02d} to run.'.format(int(rem_time//3600),
                                                                int((rem_time//60)%60),int(rem_time%60)))

        if self.check_resid_conv:
            if (resid < 1E-10): 
                print('Reached a residual tolerance of 1E-10 (L2 square norm). Ending simulation.')
                self.quitsim = True
                self.failsim = False

    def final_common(self, q, q_sol, t_idx, n_ts, dt, dqdt, time=None):
        ''' Like common, but intended for the final iteration '''
        if time is None: time = t_idx * dt

        if not self.quitsim:
            # if we already indicated to quit, then we did everything we had to in previous common().
            if t_idx != n_ts or (time is not None and abs(time - self.t_final)>1e-12):
                print('ERROR: final_common is being called before (or after?) the final iteration.')
                print('       t =',time,'t_idx =', t_idx)

            if self.keep_all_ts or self.bool_calc_cons_obj:
                # append final time step
                if self.keep_all_ts:
                    q_sol[:, :, -1] = q
                if self.bool_calc_cons_obj:
                    self.cons_obj[:, -1] = self.fun_calc_cons_obj(q,time,dqdt)

            if self.bool_plot_sol:
                self.fun_plot_sol(q, time)

            if self.print_sol_norm:
                if (t_idx % self.idx_print == 0) or (t_idx == n_ts):
                    norm_q = np.linalg.norm(q) / np.sqrt(np.size(q))
                    print(f'i = {t_idx:4}, ||q|| = {norm_q:3.4}')

            if self.check_resid_conv or self.print_residual:
                resid = np.linalg.norm(dqdt)
                resid = resid*resid # I actually want the norm squared

            if self.print_progress:
                sim_time = tm.time() - self.start_time
                suf = 'Complete.'
                h,m,s = int(sim_time//3600),int((sim_time//60)%60),int(sim_time%60)
                printProgressBar(t_idx, n_ts, prefix = 'Progress:', suffix = suf)
                print('... Took {0}:{1:02d}:{2:02d} to run.'.format(h,m,s))
        
            if np.any(np.isnan(q)):
                print('ERROR: there are undefined values for q at final t =',time,'t_idx =', t_idx)

            if self.check_resid_conv:
                if (resid < 1E-10): 
                    print('Reached a residual tolerance of 1E-10 (L2 square norm) on final iteration. Lucky you!')


    def init_q_sol(self, q0, n_ts):
        ''' initiate the q_sol vector to store solutions '''

        if self.keep_all_ts or self.bool_calc_cons_obj:
            if self.nframes is not None:
                assert self.t_final is not None, 't_final must be set before using nframes'
                assert isinstance(self.nframes, int), 'nframes must be an integer'
                assert self.nframes > 1, 'nframes must be greater than 1'
                self.use_time_frames = True
                self.frame_target_time = self.t_final / self.nframes + self.t_initial
                self.frame_current_timediff = np.inf
            else:
                self.use_time_frames = False
                # set tm_frames to a size depending on n_ts and skip_ts
                if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                    # we will land exactly on the final frame
                    self.nframes = int(n_ts/(self.skip_ts+1)) 
                else:
                    print('WARNING: skip_ts in time marching is chosen so that you do not land exactly on the final')
                    print('         time step. Will manually append the final time step solution to q_sol.')
                    # will append the final frame manually
                    self.nframes = int(n_ts/(self.skip_ts+1)) + 1

        if self.keep_all_ts:
            q_sol = np.zeros([*self.qshape, self.nframes+1])
            q_sol[:, :, 0] = q0
        else:
            q_sol = None
            
        if self.bool_calc_cons_obj:
            self.cons_obj = np.zeros((self.n_cons_obj, self.nframes+1))

        self.frame_idx = 1
        return q_sol
        
    def return_q_sol(self,q,q_sol,t_idx,dt,dqdt,time=None):
        ''' prepare q or q_sol to be returned by the main function '''
        if self.quitsim:
            # NOTE: t_idx = i+1 from where the method determined q was bad
            if time is None: time = (t_idx-1) * dt
            if self.keep_all_ts:
                if self.failsim:
                    if self.use_time_frames:
                        self.t_final = time
                        if self.bool_calc_cons_obj:
                            self.cons_obj = self.cons_obj[:, :self.frame_idx+1]
                        return q_sol[:,:,:self.frame_idx+1]
                    else:
                        # simulation failed, meaning there are some NaNs in q.
                        # return up to *but not including* this current q.
                        mod_t_idx = (t_idx-1)/(self.skip_ts+1)
                        # recall we only saved q every mod_t_idx, but that may or many not be now
                        # since the program could have exited for a t_idx between these checkpoints
                        if abs(mod_t_idx - round(mod_t_idx)) < 1e-9:
                            # only return up to but not including this mod_t_idx 
                            self.t_final = (int(mod_t_idx)-1)*(self.skip_ts+1)*dt
                            print("... returning q's up to and including t =", self.t_final, "t_idx =", int(mod_t_idx)-1)
                            if self.bool_calc_cons_obj:
                                self.cons_obj = self.cons_obj[:, :int(mod_t_idx)]
                            return q_sol[:,:,:int(mod_t_idx)]
                        else:
                            # safe to return up to last mod_t_idx 
                            self.t_final = int(mod_t_idx)*(self.skip_ts+1)*dt
                            print("... returning q's up to and including t =", self.t_final, "t_idx =", int(mod_t_idx))
                            if self.bool_calc_cons_obj:
                                self.cons_obj = self.cons_obj[:, :int(mod_t_idx)+1]
                            return q_sol[:,:,:int(mod_t_idx)+1]
                else:
                    # the simulation ended because convergence was reached.
                    # We want to return this most recent q.
                    mod_t_idx = (t_idx-1)/(self.skip_ts+1)
                    self.t_final = t_idx*dt
                    # recall we only saved q every mod_t_idx, but that may or many not be now
                    # since the program could have exited for a t_idx between these checkpoints
                    if abs(mod_t_idx - round(mod_t_idx)) < 1e-9:
                        # return up to and including this mod_t_idx, which is the most recent t_idx
                        print("... returning q's up to and including t =", self.t_final, "t_idx =", int(mod_t_idx))
                        if self.bool_calc_cons_obj:
                            self.cons_obj = self.cons_obj[:, :int(mod_t_idx)+1]
                        return q_sol[:,:,:int(mod_t_idx)+1]
                    else:
                        # want to append final q even though it lies inbetween checkpoints
                        print("... returning q's up to and including t =", self.t_final, "t_idx =", int(mod_t_idx))
                        print("... WARNING: this final timestep checkpoint was appended, so is not the same dt as all the ones before it, rather a bit shorter.")
                        if self.bool_calc_cons_obj:
                            self.cons_obj = self.cons_obj[:, :int(mod_t_idx)+1]
                        q_sol[:,:,int(mod_t_idx)+1] = q
                        return q_sol[:,:,:int(mod_t_idx)+2]
            else:
                if self.failsim:
                    self.t_final = (t_idx-1)*dt
                    print('... returning solution q at t =', self.t_final, '. Note: will contain NaNs. Consider running with keep_all_ts = True')
                return q
        else:
            if self.keep_all_ts:
                if self.use_time_frames:
                    #self.t_final = time
                    if self.bool_calc_cons_obj:
                        self.cons_obj = self.cons_obj[:, :self.frame_idx+1]
                    return q_sol[:,:,:self.frame_idx+1]
                else:
                    # no need to append final solution, should have been done in final_common
                    return q_sol
            else:
                return q

    
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