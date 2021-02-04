#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:29:45 2020

@author: andremarchildon
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# from Source.TimeMarch.GetTimeMarchingClass import GetTimeMarchingClass

'''
The explicit and implicit linear multistep methods are solved in this class.
A genral method is provided for the explicit and implicit time marching
schemes. These methods must be provided with a vector a, which indicates the
weight used for the solution at the previous time steps, and a vector b, which
indicates the weight for the flux at the previous time steps.
'''

class TimeMarchingLms:

    def ab1(self, q0, dt, n_ts):

        # Explicit Euler
        a_vec = np.array([-1, 1])
        b_vec = np.array([1, 0])

        return self.linearmultistep(q0, dt, n_ts, a_vec, b_vec)

    def ab2(self, q0, dt, n_ts):

        a_vec = np.array([0, -1, 1])
        b_vec = np.array([-1/2, 3/2, 0])

        return self.linearmultistep(q0, dt, n_ts, a_vec, b_vec)

    def ab3(self, q0, dt, n_ts):

        # Leapfrog
        a_vec = np.array([0, 0, -1, 1])
        b_vec = np.array([5/12, -16/12, 23/12, 0])

        return self.linearmultistep(q0, dt, n_ts, a_vec, b_vec)

    def am3(self, q0, dt, n_ts):

        a_vec = np.array([0, -1, 1])
        b_vec = np.array([5/12, -16/12, 23/12])

        return self.linearmultistep(q0, dt, n_ts, a_vec, b_vec)

    def linearmultistep(self, q0, dt, n_ts, a_vec, b_vec):

        ''' Check if the method is consistent '''

        n_step = a_vec.size - 1
        test1 = np.sum(a_vec)
        test2a = np.sum(b_vec)

        test2b = 0
        for i in range(n_step+1):
            test2b += i*a_vec[i]

        # These conditions come from page 103 of Fundamentals of CFD by Lomax,
        # Pulliam and Zingg
        assert test1 == 0, 'The method is not consistent'
        assert np.abs(test2a - test2b) < 1e-4, 'The method is not consistent'
        assert a_vec.size == b_vec.size, 'a_vec and b_vec must be the same length'

        ''' Use 4th order RK to have sufficient steps to start the solution '''

        # Need to save all the time steps when calling the RK4 method
        keep_all_ts = self.keep_all_ts
        self.keep_all_ts = True

        n_ts_rk = n_step - 1 # Minus 1 since q0 is provided

        # Determine if the time marching is explicit or implicit
        if b_vec[-1] == 0:
            is_explicit = True
            sol_rk = self.rk4(q0, dt, n_ts_rk)
        else:
            is_explicit = False
            sol_rk = self.dirk_p4s3(q0, dt, n_ts_rk)
            I = np.eye(self.len_q)

        self.keep_all_ts = keep_all_ts

        if self.keep_all_ts:
            q_vec = np.zeros([self.len_q, n_ts+1])
            q_vec[:, :n_ts_rk+1] = sol_rk

        # Initiate the 2D arrays to hold the previous time steps
        q_mat = sol_rk * 1.0
        f_mat = np.zeros(q_mat.shape)

        for i in range(n_step):
            f_mat[:,i] = self.f_dqdt(q_mat[:,i])

        ''' Solve the time marching problem '''

        def calc_time_step(q_in, q_mat, f_mat):

            rhs = np.sum(-a_vec[:-1] * q_mat + (dt * b_vec[:-1]) * f_mat, axis=1)

            if is_explicit:
                q_new = rhs / a_vec[-1]
            else:
                dfdq = self.f_dfdq(q_mat[:,-1])
                lhs = I*a_vec[-1] - (dt * b_vec[-1]) * dfdq
                rhs += (dt * b_vec[-1]) * (f_mat[:,-1] + dfdq @ q_mat[:,-1])
                q_new = np.linalg.solve(lhs, rhs)

            return q_new

        for i in range(n_ts_rk+1, n_ts+1):

            q_new = calc_time_step(q_mat, q_mat, f_mat)

            # Update 2D array of previous time steps and fluxes
            q_mat = np.roll(q_mat, -1, axis=1)
            f_mat = np.roll(f_mat, -1, axis=1)

            q_mat[:,-1] = q_new
            f_mat[:,-1] = self.f_dqdt(q_new)

            if self.keep_all_ts:
                q_vec[:,i] = q_new

            self.common(q_new, i, n_ts, dt)

        if self.keep_all_ts:
            return q_vec
        else:
            return q_new

    def bdf2(self, q0, dt, n_ts):

        I = sp.eye(self.len_q)

        def calc_time_step(q_t, q_tn1):

            ff = self.f_dqdt(q_t)
            dfdq = self.f_dfdq(q_t)

            lhs = 3*I - 2*dt*dfdq
            rhs = q_t - q_tn1 + 2*dt*ff

            if sp.issparse(lhs):
                dq = spsolve(lhs, rhs)
            else:
                dq = np.linalg.solve(lhs, rhs)

            q_new = q_t + dq

            return q_new

        q_old = np.copy(q0)
        self.common(q_old, 0, n_ts, dt)

        # For the first time step use trapezoidal
        ff = self.f_dqdt(q_old)
        dfdq = self.f_dfdq(q_old)

        lhs = I - 0.5*dt*dfdq
        rhs = dt*ff

        if sp.issparse(lhs):
            dq = spsolve(lhs, rhs)
        else:
            dq = np.linalg.solve(lhs, rhs)

        q = q_old + dq
        self.common(q, 1, n_ts, dt)

        if self.keep_all_ts:
            q_vec = np.zeros([self.len_q, n_ts+1])
            q_vec[:, 0] = q_old
            q_vec[:, 1] = q

        for i in range(2, n_ts+1):

            q_new = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                q_vec[:,i] = q

            self.common(q, i, n_ts, dt)

        if self.keep_all_ts:
            return q_vec
        else:
            return q

