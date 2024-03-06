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

class TimeMarchingOneStep_old:

    def explicit_euler(self, q0, dt, n_ts):

        def calc_time_step(q_in):
            dqdt = self.f_dqdt(q_in)
            q_new = q_in + dt * self.f_dqdt(q_in)
            return q_new, dqdt

        q = np.copy(q0)

        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.shape_q, frames])
            q_vec[:, :, 0] = q

        self.common(q, 0, n_ts, dt, -1)

        for i in range(1, n_ts+1):
            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q_new

            self.common(q, i, n_ts, dt, dqdt)
            if self.quitsim: break

        if self.quitsim:
            # return up to but not including this current time step
            if self.keep_all_ts:
                # we only save for modi, but that may or many not be now
                if modi.is_integer():
                    # only return up to but not including this modi 
                    self.t_final = (int(modi)-1)*(self.skip_ts+1)*dt
                    print('... returning solutions up to and including t =', self.t_final)
                    return q_vec[:,:,:int(modi)]
                else:
                    # safe to return up to last modi
                    self.t_final = int(modi)*(self.skip_ts+1)*dt
                    print('... returning solutions up to and including t =', self.t_final)
                    return q_vec[:,:,:int(modi)+1]
            else:
                return q_mat[:,-2]
        else:
            if self.keep_all_ts:
                return q_vec
            else:
                return q_new

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

            return q_new, ff

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt, -1)

        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.shape_q, frames])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):

            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q_new

            self.common(q, i, n_ts, dt, dqdt)
            if self.quitsim: break

        # qq, rr = np.linalg.qr(self.M)
        # lya = np.log(np.abs(np.diagonal(rr))) / (n_ts*dt)
        # print(f'lya = {lya}')

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

            return q_new, ff

        q = np.copy(q0)
        self.common(q, 0, n_ts, dt, -1)

        if self.keep_all_ts:
            if n_ts/(self.skip_ts+1) == int(n_ts/(self.skip_ts+1)):
                # we will land exactly on the final frame
                frames = int(n_ts/(self.skip_ts+1)) + 1
            else:
                # append the final frame manually
                frames = int(n_ts/(self.skip_ts+1)) + 2
            q_vec = np.zeros([*self.shape_q, frames])
            q_vec[:, :, 0] = q

        for i in range(1, n_ts+1):
            q_new, dqdt = calc_time_step(q, q_old)
            q_old = np.copy(q)
            q = np.copy(q_new)

            if self.keep_all_ts:
                modi = i/(self.skip_ts+1)
                if modi.is_integer():
                    q_vec[:, :, int(modi)] = q_new

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