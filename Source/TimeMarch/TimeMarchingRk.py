#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:28:45 2020

@author: bercik
"""

import numpy as np
from scipy.integrate import DOP853
import traceback
# import scipy.sparse as sp
# from scipy.sparse.linalg import spsolve

'''
The Runge-Kutta (RK) methods are divided into 3 types, explicit (ERK),
diagonaly implicit (DIRK), and fully implicit (IRK). Each of these RK types
has a general method (e.g. erk_general) which is provided with the 2D array
a_mat and the 1D array b_vec from another method (e.g. rk4). Only autonomous
sytems or time-invariant systems are considered and as such, the vector c for
RK methods is not utilized. Time-invariant methods are differential equations
that do not depend directly on time. For example, the Euler equations are
time-invariant since the PDE is the same at time t=0 and at time t=10.
'''

class TimeMarchingRk:

    ''' Specific RK methods '''

    def rk4(self, q, dt, n_ts):

        q_sol = self.init_q_sol(q, n_ts)
        quit = False

        # This method solves the 4th order explicit Runge-Kutta method.
        for i in range(0, n_ts):

            t = i * dt

            try:
                k1 = self.dqdt(q, t)
            except Exception as e:
                print("ERROR RK4: dqdt failed in first stage. Returning last q.")
                traceback.print_exc()
                quit = True
                break
            self.common(q, q_sol, i, n_ts, dt, k1)
            if self.quitsim: break

            q1 = q + 0.5*dt*k1 # first predictor q_{n+1/2}
            
            try:
                k2 = self.dqdt(q1, t+0.5*dt)
            except Exception as e:
                print("ERROR RK4: dqdt failed in second stage. Returning q at stage 1 as last q.")
                traceback.print_exc()
                quit = True
                q = q1
                break
            q2 = q + 0.5*dt*k2 # first corrector q_{n+1/2}
            
            try:
                k3 = self.dqdt(q2, t+0.5*dt)
            except Exception as e:
                print("ERROR RK4: dqdt failed in third stage. Returning q at stage 2 as last q.")
                traceback.print_exc()
                quit = True
                q = q2
                break
            q3 = q + dt*k3     # second predictor q_{n+1}
            
            try:
                k4 = self.dqdt(q3, t+dt)
            except Exception as e:
                print("ERROR RK4: dqdt failed in fourth stage. Returning q at stage 3 as last q.")
                traceback.print_exc()
                quit = True
                q = q3
                break
            
            q += dt*(k1 + 2*(k2+k3) + k4)/6 # final correction q_{n+1}
        
        # Congrats you reached the end
        i += 1
        if quit:
            k1 = np.empty_like(q)
            k1.fill(np.nan)
        else:
            k1 = self.dqdt(q, t)
        self.final_common(q, q_sol, i, n_ts, dt, k1)
        return self.return_q_sol(q,q_sol,i,dt,k1)

    def explicit_euler(self, q, dt, n_ts):

        q_sol = self.init_q_sol(q, n_ts)
        quit = False

        # This method solves the 1st order explicit Euler method.
        for i in range(0, n_ts):

            t = i * dt

            try:
                dqdt = self.dqdt(q, t)
            except Exception as e:
                print("ERROR RK1: dqdt failed. Returning last q.")
                traceback.print_exc()
                quit = True
                break
            self.common(q, q_sol, i, n_ts, dt, dqdt)
            if self.quitsim: break

            q += dt * dqdt

        # Congrats you reached the end
        i += 1
        if quit:
            k1 = np.empty_like(q)
            k1.fill(np.nan)
        else:
            k1 = self.dqdt(q, t)
        self.final_common(q, q_sol, i, n_ts, dt, k1)
        return self.return_q_sol(q,q_sol,i,dt,dqdt)
    
    def rk8(self, q, dt, n_ts):
        ''' uses the Explicit Runge-Kutta method of order 8.
        Calls the Scipy Implementation of “DOP853” algorithm.
        
        E. Hairer, S. P. Norsett G. Wanner, 
        “Solving Ordinary Differential Equations I: Nonstiff Problems”, Sec. II.'''

        def f(t, y):
            return self.dqdt(y.reshape(self.qshape,order='F'), t).flatten('F')

        tm_solver = DOP853(f, 0.0, q.flatten('F'), t_bound=self.t_final, 
                           first_step=dt, rtol=self.rtol, atol=self.atol)
        
        # we don't have access to the internal steps, so we can't get dqdt without doing exrtra work
        dqdt = np.zeros_like(q)
        q_sol = self.init_q_sol(q, n_ts)
        if self.keep_all_ts:
            self.keep_all_ts = False
            keep_all_ts_lcl = True
        else:
            keep_all_ts_lcl = False

        if self.bool_calc_cons_obj:
            self.bool_calc_cons_obj = False
            bool_calc_cons_obj_lcl = True
            self.cons_obj[:, 0] = self.fun_calc_cons_obj(q,0.0,dqdt)
        else:
            bool_calc_cons_obj_lcl = False

        #n_ts = int(self.t_final/dt)
        i = 0
        bad_dt = 0
        frame_skip = 0
        t_current = 0.
        y_current = q.flatten('F')
        n_ts = int(self.t_final/dt)
        while tm_solver.status == 'running':
            try:
                tm_solver.step()  # Advance one internal step
                t_current = tm_solver.t
                y_current = tm_solver.y
                i += 1
                # we need some estimate of i in relation to n_ts
                n_ts = int(i*self.t_final/t_current)+1
            except Exception as e:
                print("ERROR RK8: WARNING: DOP853 step failed. Returning last q.")
                traceback.print_exc()
                break
            q = y_current.reshape(self.qshape,order='F')
            self.common(q, q_sol, i, n_ts, dt, dqdt, time=t_current)

            # Check if step size is too small
            if tm_solver.h_abs < 1e-7:
                print(f"WARNING RK8: timestep has reached a very small dt = {tm_solver.h_abs:e} at t = {t_current}. Consider aborting.")
                bad_dt += 1
                if tm_solver.h_abs < 1e-10:
                    print(f"ERROR RK8: timestep has reached timestep threshold of 1e-10. Aborting.")
                    self.quitsim = True
                    self.failsim = True
                    break
                if bad_dt > 50:
                    print(f"ERROR RK8: too many very small time steps. Aborting.")
                    self.quitsim = True
                    self.failsim = True
                    break

            if keep_all_ts_lcl or bool_calc_cons_obj_lcl:
                if self.use_time_frames:
                    timediff = abs(t_current - self.frame_target_time)
                    if timediff < self.frame_current_timediff:
                        # we are closer to the target time. Save the current solution at current index.
                        self.frame_current_timediff = timediff
                    else:
                        # we have started moving away from the target time. Update the target time.
                        self.frame_idx += 1
                        self.frame_target_time = self.t_final / self.nframes * self.frame_idx
                        self.frame_current_timediff = abs(t_current - self.frame_target_time)

                    if keep_all_ts_lcl:
                        q_sol[:, :, self.frame_idx] = q
                    if bool_calc_cons_obj_lcl:
                        self.cons_obj[:, self.frame_idx] = self.fun_calc_cons_obj(q,t_current,dqdt)
                
                elif frame_skip == self.skip_ts or abs(t_current - self.t_final) < 1e-12:
                    if self.frame_idx == self.nframes - 1:
                        self.nframes = int(2 * self.nframes)
                        if keep_all_ts_lcl:
                            print('\n')
                            print('WARNING: q_sol is not large enough to hold all time steps.')
                            print('         Increasing the current size by a factor of 2.')
                            temp = np.zeros((*self.qshape, self.nframes))
                            temp[:, :, :self.frame_idx+1] = q_sol
                            q_sol = temp
                            del temp
                        if bool_calc_cons_obj_lcl:
                            if not keep_all_ts_lcl: ('\n')
                            print('WARNING: cons_obj is not large enough to hold all time steps.')
                            print('         Increasing the current size by a factor of 2.')
                            print('\n')
                            temp = np.zeros((self.n_cons_obj, self.nframes))
                            temp[:, :self.frame_idx+1] = self.cons_obj
                            self.cons_obj = temp
                            del temp

                    if keep_all_ts_lcl:
                            q_sol[:, :, self.frame_idx] = q
                    if bool_calc_cons_obj_lcl:
                        self.cons_obj[:, self.frame_idx] = self.fun_calc_cons_obj(q,t_current,dqdt)

                    self.frame_idx += 1
                    frame_skip = 0
                else:
                    frame_skip += 1

            if self.quitsim: break

            #TODO : make sure time is included in cons obj
        
        #t_current = tm_solver.t
        #y_current = tm_solver.y
        if t_current < self.t_final:
            self.quitsim = True
            self.failsim = True
            self.t_final = t_current
        else:
            n_ts = i
        q = y_current.reshape(self.qshape,order='F')
        self.final_common(q, q_sol, i, n_ts, dt, dqdt, time=t_current)
        if t_current < self.t_final:
            print("RK8: WARNING: Did not reach final time. t = %f" % t_current)
        else:
            print("RK8: Reached final time. t = %f" % t_current)
        print("RK8: used %d steps" % i)
        if keep_all_ts_lcl:
            if self.use_time_frames:
                q_sol = q_sol[:, :, :self.frame_idx+1]
            else:
                q_sol = q_sol[:, :, :self.frame_idx] # already added one
        else:
            q_sol = q
        if bool_calc_cons_obj_lcl:
            if self.use_time_frames:
                self.cons_obj = self.cons_obj[:, :self.frame_idx+1]
            else:
                self.cons_obj = self.cons_obj[:, :self.frame_idx]
        return q_sol
    
    def rk8_verner(self, q, dt, n_ts):
        ''' uses the Explicit Runge-Kutta method of order 8.
        Verner's RK8(7) coefficients (11-stage, embedded pair).
        
        Verner (2010): High Order Explicit Runge-Kutta Pairs with Low-Storage Implementations'''

        q_sol = self.init_q_sol(q, n_ts)
        quit = False

        c = np.array([
            0.0,
            0.092662,
            0.13122303617540176,
            0.19683455426310264,
            0.427173,
            0.485972,
            0.161915,
            0.985468,
            0.962697734860454,
            0.99626,
            0.997947,
            1.0,
            1.0
        ])

        # a coefficients: list of tuples (j, a_ij) for each stage i
        a_stages = {
            1: [(0, 0.092662)],
            2: [(0, 0.03830746548250284), (1, 0.09291557069289892)],
            3: [(0, 0.04920863856577566), (2, 0.14762591569732698)],
            4: [(0, 0.2743076085702487), (2, -0.9319887203102656), (3, 1.084854111740017)],
            5: [(0, 0.06461852970939692), (3, 0.26876292133689234), (4, 0.15259054895371074)],
            6: [(0, 0.07189155819773217), (3, 0.12212657833625497), (4, -0.07943550859198561), (5, 0.04733237205799848)],
            7: [(0, -6.073603893714329), (3, -73.8956), (4, 11.93985370695274), (5, -3.8392515414050546), (6, 72.85406972816664)],
            8: [(0, -4.868640079323569), (3, -59.18572799975646), (4, 9.230819319232425), (5, -2.676847914962526),
                (6, 58.45720009994686), (7, 0.00589430972372636)],
            9: [(0, -6.689861899320853), (3, -81.44271004053111), (4, 13.367788256983971), (5, -4.470777638416181),
                (6, 80.2332139216141), (7, -0.013136383362121816), (8, 0.011743783032194139)],
            10: [(0, -6.788841955800464), (3, -82.65639855934829), (4, 13.59973921874899), (5, -4.574464055350504),
                (6, 81.41943207216076), (7, -0.01416248014826418), (8, 0.013754415808352274), (9, -0.0011116560705810062)],
            11: [(0, -6.910189846402486), (3, -84.14495154176749), (4, 13.885121223789838), (5, -4.702458788144493),
                (6, 82.87411451529242), (7, -0.016454983371987802), (8, 0.016446639721625213),
                (9, 0.004275449370796531), (10, -0.005902668488222361)],
            12: [(0, -6.91197392119898), (3, -84.16635595878781), (4, 13.88834627565582), (5, -4.703463178409703),
                (6, 82.89518622207405), (7, -0.010203450162282603), (8, 0.014279004232303915), (9, -0.005814993403397981)]
        }

        # Sparse b coefficients: (s, b_s)
        b_terms = [
            (0, 0.04625543159712467),
            (5, 0.3706666165521011),
            (6, 0.25904408245527466),
            (7, -679.9841468175039),
            (8, 49.89161129042053),
            (9, 10271.235222137312),
            (10, -14782.196606356897),
            (11, 5141.377953616064)
        ]

        for i in range(n_ts):
            t = i * dt
            k = np.zeros((13, *self.qshape), order='F')

            try:
                for s in range(13):
                    if s == 0:
                        q_s = q
                    else:
                        q_s = q.copy()
                        for j, a_sj in a_stages.get(s, []):
                            q_s += dt * a_sj * k[j]
                    k[s] = self.dqdt(q_s, t + c[s]*dt)
            except Exception as e:
                print(f"ERROR rk8_verner: dqdt failed in stage {s+1}. Returning last q.")
                traceback.print_exc()
                quit = True
                break

            self.common(q, q_sol, i, n_ts, dt, k[0])
            if self.quitsim:
                break

            for s, bs in b_terms:
                q += dt * bs * k[s]

        i += 1
        if quit:
            k0 = np.empty_like(q)
            k0.fill(np.nan)
        else:
            k0 = self.dqdt(q, t)
        self.final_common(q, q_sol, i, n_ts, dt, k0)
        return self.return_q_sol(q, q_sol, i, dt, k0)

    
    # TODO: Do I want to put back any of the implicit methods? 
    # THis would include implicit Euler, Trapezoidal, etc







class TimeMarchingRk_old:

    ''' This is the class originally built by Andre, but I only ever used RK4,
    and it was just too bulky. So instead I spearate it here in case I ever in
    the future want to re-implement IRK or DIRK methods again. '''

    tol = 1e-12     # Used for IRK to check if the solution has converged
    n_iter_max = 10 # Max no. of iterations to converge the solution

    ''' Specific RK methods '''

    def rk4(self, q0, dt, n_ts):

        # The classical fourth order RK method

        a_mat = np.zeros((4,4))
        a_mat[1,0] = a_mat[2,1] = 1/2
        a_mat[3,2] = 1

        b_vec = np.array([1/6, 1/3, 1/3, 1/6])

        return self.erk_general(q0, dt, n_ts, a_mat, b_vec)

    def dirk_p4s3(self, q0, dt, n_ts):

        # Order 4 wth 3
        # From the Reivew of DIRK methods eq (214)

        a_mat = np.zeros((3,3))
        a_mat[1,0] = a_mat[1,1] = 1/4
        a_mat[2,1] = 1

        b_vec = np.array([1/6, 4/6, 1/6])

        return self.dirk_general(q0, dt, n_ts, a_mat, b_vec)

    def dirk_p4s4(self, q0, dt, n_ts):

        # Order 4 wth 4
        # From the Reivew of DIRK methods eq (215)
        # Stiffly accurate

        a_mat = np.array([[0,0,0,0], [1/6, 1/6, 0, 0],
                          [1/12, 1/2, 1/12, 0], [1/8, 3/8, 3/8, 1/8]])

        b_vec = np.array([1/8, 3/8, 3/8, 1/8])

        return self.dirk_general(q0, dt, n_ts, a_mat, b_vec)

    def irk_s2(self, q0, dt, n_ts):

        a_mat = np.array([[1/4, 1/4-np.sqrt(3)/6], [1/4+np.sqrt(3)/6, 1/4]])
        b_vec = np.array([1/2, 1/2])

        return self.irk_general(q0, dt, n_ts, a_mat, b_vec)

    def irk_s3(self, q0, dt, n_ts):

        a_mat = np.array([[5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
                          [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
                          [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]])
        b_vec = np.array([5/18, 4/9, 5/18])

        return self.irk_general(q0, dt, n_ts, a_mat, b_vec)

    ''' General methods '''

    def erk_general(self, q0, dt, n_ts, a_mat, b_vec):

        # This method solves any explicit Runge-Kutta method.

        n_step = b_vec.size

        # Check that the 2D array a_mat is strictly lower diagonal
        for i in range(n_step):
            max_abs_row = np.max(np.abs(a_mat[i,i:]))
            assert max_abs_row < 1e-10, 'The 2D array a_mat must be strickly lower diagonal to be an ERK method'

        def calc_time_step(q_in):

            f_mat = np.zeros((*self.qshape, n_step))

            # Solve each explicit step individually
            for i in range(n_step):

                # Vectorization of for loop over j to calculate: (dt*a_mat[i,j]) * f_mat[:,j]
                qi = q_in + np.sum((dt*a_mat[i,:i]) * f_mat[:,:,:i], axis=2)

                f_mat[:,:,i] = self.f_dqdt(qi)

            # Calculate the new time step using f at all the stages
            q_new = q_in*1.0

            for i in range(n_step):
                q_new += dt * b_vec[i] * f_mat[:,:,i]

            return q_new, f_mat[:,:,-1]

        q = q0 * 1.0
        self.common(q, 0, n_ts, dt, -1)

        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.qshape, frames])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):
            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q

            self.common(q, i, n_ts, dt, dqdt)
            if self.quitsim: break

        if self.keep_all_ts:
            if self.quitsim:
                return q_vec[:,:,:int(modi)]
            else:
                if modi.is_integer():
                    return q_vec
                else:
                    q_vec[:, :, -1] = q
                return q_vec
        else:
            return q

    def dirk_general(self, q0, dt, n_ts, a_mat, b_vec):

        # This method solves any diagonaly implicit Runge-Kutta method.
        # A second order approximation is made to solve the implicit problem
        # at each step and as such, the stage order cannot exceed two,
        # otherwise the method's order may be reduced

        I = np.eye(self.len_q)
        n_step = b_vec.size

        # Check that the 2D array a_mat is lower diagonal
        for i in range(n_step-1):
            max_abs_row = np.max(np.abs(a_mat[i,(i+1):]))
            assert max_abs_row < 1e-10, 'The 2D array a_mat must be lower diagonal to be a DIRK method'

        def calc_time_step(q_in):

            f_mat = np.zeros((self.len_q, n_step))

            f0 = self.f_dqdt(q_in)
            dfdq0 = self.f_dfdq(q_in)

            # Solve each implicit step individually
            for i in range(n_step):

                lhs = I - dt*a_mat[i,i]*dfdq0

                # Vectorization of for loop over j to calculate: (dt*a_mat[i,j]) * f_mat[:,j]
                rhs = lhs @ q_in + (dt*a_mat[i,i]) * f0 + np.sum((dt*a_mat[i,:i]) * f_mat[:,:i], axis=1)

                qi = np.linalg.solve(lhs, rhs)
                f_mat[:,i] = self.f_dqdt(qi)

            # Calculate the new time step using f at all the stages
            q_new = q_in*1.0

            for i in range(n_step):
                q_new += dt * b_vec[i] * f_mat[:,i]

            return q_new, f_mat[:,:,-1]

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt, -1)
        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.qshape, frames])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):
            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q

            self.common(q, i, n_ts, dt, dqdt)
            if self.quitsim: break

        if self.keep_all_ts:
            if self.quitsim:
                return q_vec[:,:,:int(modi)]
            else:
                if modi.is_integer():
                    return q_vec
                else:
                    q_vec[:, :, -1] = q
                return q_vec
        else:
            return q

    def irk_general(self, q0, dt, n_ts, a_mat, b_vec):

        # This method solves any implicit Runge-Kutta method.
        # A second order approximation is made to solve the implicit problem
        # at each step and as such, the stage order cannot exceed two,
        # otherwise the method's order may be reduced

        n_step = b_vec.size
        I = np.eye(self.len_q)

        def calc_f_mat(q_mat):

            f_mat = np.zeros((self.len_q, n_step))
            for i in range(n_step):
                f_mat[:,i] = self.f_dqdt(q_mat[:,i])

            return f_mat

        def calc_dfdq_all(q_mat):

            dfdq_array = np.zeros((n_step, self.len_q, self.len_q))

            for i in range(n_step):
                dfdq_array[i,:,:] = self.f_dfdq(q_mat[:,i])

            return dfdq_array

        def solve_q_mat(q_in, q_mat, f_array, dfdq_array):

            # Create the LHS and RHS of the linear problem
            n_size = self.len_q * n_step
            lhs = np.zeros((n_size, n_size))
            rhs = np.zeros(n_size)

            r2 = 0
            for i in range(n_step):
                r1 = r2
                r2 = r1 + self.len_q

                rhs[r1:r2] = q_in

                c2 = 0
                for j in range(n_step):
                    c1 = c2
                    c2 = c1 + self.len_q

                    rhs[r1:r2] += (dt*a_mat[i,j]) * (f_array[:,j] - dfdq_array[j, :,:] @ q_mat[:,j])
                    lhs[r1:r2, c1:c2] = (-dt*a_mat[i,j]) * dfdq_array[j, :,:]

                    if i == j:
                        lhs[r1:r2, c1:c2] += I

            # Reshape to have the solution for each stage in one column
            q_mat = np.linalg.solve(lhs, rhs).reshape((self.len_q, n_step), order='F')

            return q_mat

        def calc_time_step(q_in):

            q_new = q_in

            q_mat = np.zeros((self.len_q, n_step))
            f_array = np.zeros((self.len_q, n_step))
            dfdq_array = np.zeros((n_step, self.len_q, self.len_q))

            f0 = self.f_dqdt(q_in)
            dfdq0 = self.f_dfdq(q_in)

            for i in range(n_step):
                f_array[:,i] = f0
                dfdq_array[i,:,:] = dfdq0

            i_iter = 0
            not_done = True
            while not_done:
                i_iter += 1

                # Solve for the new time step
                q_mat = solve_q_mat(q_in, q_mat, f_array, dfdq_array)
                f_array = calc_f_mat(q_mat)

                # Calculate the new time step using f at all the stages
                q_old = q_new*1.0
                q_new = q_in*1.0

                for i in range(n_step):
                    q_new += dt * b_vec[i] * f_array[:,i]

                # Check if the solution has converged
                q_diff_norm = np.linalg.norm(q_new - q_old)

                if q_diff_norm < self.tol or i_iter == self.n_iter_max:
                    not_done = False
                else:
                    dfdq_array = calc_dfdq_all(q_mat)

            return q_new, f_array[:,-1]

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt, -1)

        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.qshape, frames])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):
            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q

            self.common(q, i, n_ts, dt, dqdt)
            if self.quitsim: break

        if self.keep_all_ts:
            if self.quitsim:
                return q_vec[:,:,:int(modi)]
            else:
                if modi.is_integer():
                    return q_vec
                else:
                    q_vec[:, :, -1] = q
                return q_vec
        else:
            return q