#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:24:35 2020

@author: andremarchildon
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

'''
This class contains time marching schemes for one-step and one-stage methods.
These methods can technically fit under both the linear multistep and the
Runge-Kutta methods.
'''

class TimeMarchingOneStep:

    def explicit_euler(self, q0, dt, n_ts):

        def calc_time_step(q_in):
            q_new = q_in + dt * self.f_dqdt(q_in)
            return q_new

        q = np.copy(q0)

        if self.keep_all_ts:
            q_vec = np.zeros([*self.shape_q, n_ts+1])
            q_vec[:, :, 0] = q

        self.common(q, 0, n_ts, dt)

        for i in range(1, n_ts+1):
            q = calc_time_step(q)

            if self.keep_all_ts:
                q_vec[:,:,i] = q

            self.common(q, i, n_ts, dt)

        if self.keep_all_ts:
            return q_vec
        else:
            return q

    def implicit_euler(self, q0, dt, n_ts):

        I = np.eye(self.len_q)

        # self.M = np.eye(self.len_q)

        def calc_time_step(q_in):

            shape = q_in.shape
            ff = self.f_dqdt(q_in)
            dfdq = self.f_dfdq(q_in)

            lhs = I - dt*dfdq
            rhs = dt*(ff.flatten('F'))

            dq = np.linalg.solve(lhs, rhs)

            q_new = q_in + np.reshape(dq, shape, 'F')

            # self.M = self.M @ (I + dfdq)

            return q_new

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt)

        if self.keep_all_ts:
            q_vec = np.zeros([*self.shape_q, n_ts+1])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):

            q = calc_time_step(q)

            if self.keep_all_ts:
                q_vec[:,:,i] = q

            self.common(q, i, n_ts, dt)

        # qq, rr = np.linalg.qr(self.M)
        # lya = np.log(np.abs(np.diagonal(rr))) / (n_ts*dt)
        # print(f'lya = {lya}')

        if self.keep_all_ts:
            return q_vec
        else:
            return q

    def trapezoidal(self, q0, dt, n_ts):

        self.len_q = q0.size
        I = np.eye(self.len_q)

        def calc_time_step(q_in):

            shape = q_in.shape
            ff = self.f_dqdt(q_in)
            dfdq = self.f_dfdq(q_in)

            lhs = I - 0.5*dt*dfdq
            rhs = dt*ff.flatten('F')

            dq = np.linalg.solve(lhs, rhs)

            q_new = q_in + np.reshape(dq, shape, 'F')

            return q_new

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt)

        if self.keep_all_ts:
            q_vec = np.zeros([*self.shape_q, n_ts+1])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):
            q = calc_time_step(q)

            if self.keep_all_ts:
                q_vec[:,:,i] = q

            self.common(q, i, n_ts, dt)

        if self.keep_all_ts:
            return q_vec
        else:
            return q