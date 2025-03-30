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
        frame = 1
        nframes = self.nframes
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

            if frame_skip == self.skip_ts or abs(t_current - self.t_final) < 1e-10:
                if frame == nframes - 1:
                    nframes = int(1.5 * frame)
                    if keep_all_ts_lcl:
                        print('\n')
                        print('WARNING: q_sol is not large enough to hold all time steps.')
                        print('         Increasing the current size by a factor of 1.5.')
                        temp = np.zeros((*self.qshape, nframes))
                        temp[:, :, :frame+1] = q_sol
                        q_sol = temp
                    if bool_calc_cons_obj_lcl:
                        if not keep_all_ts_lcl: ('\n')
                        print('WARNING: cons_obj is not large enough to hold all time steps.')
                        print('         Increasing the current size by a factor of 1.5.')
                        print('\n')
                        temp = np.zeros((self.n_cons_obj, nframes))
                        temp[:, :frame+1] = self.cons_obj
                        self.cons_obj = temp
                    del temp

                if keep_all_ts_lcl: q_sol[:, :, frame] = q
                if bool_calc_cons_obj_lcl: self.cons_obj[:, frame] = self.fun_calc_cons_obj(q,t_current,dqdt)
                frame += 1
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
        else:
            n_ts = i
        q = y_current.reshape(self.qshape,order='F')
        self.final_common(q, q_sol, i, n_ts, dt, dqdt)
        if t_current < self.t_final:
            print("RK8: WARNING: Did not reach final time. t = %f" % t_current)
        else:
            print("RK8: Reached final time. t = %f" % t_current)
        print("RK8: used %d steps" % i)
        if keep_all_ts_lcl:
            q_sol = q_sol[:, :, :frame]
        else:
            q_sol = q
        if bool_calc_cons_obj_lcl:
            self.cons_obj = self.cons_obj[:, :frame]
        return q_sol

    
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