#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:06:42 2020

@author: andremarchildon
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEulerFd
from Source.DiffEq.Quasi1dEuler import Quasi1dEulerSbp
from Source.Solvers.PdeSolver import PdeSolver


'''
Solve the quasi-one-dimensional converging-diverging nozzle from the textbook
Fundamental Algorithms in Computational Fluid Dynamics by Pulliam and Zingg.
The exact solution is available along with the algorithm from Chapter 4.
'''

# Eq parameters
para = None
obj_name = None
flow_is_subsonic = True

# Time marching
tm_method = 'rk4' # 'implicit_euler', 'explicit_euler', 'trapezoidal', 'rk4', 'bdf2'
dt = 0.01
dt_init = dt
nts = 500
t_init = 0
tf = nts * dt # set to None to do automatically or use a convergence criterion
# note: can add option to pass None, then that triggers it to check diffeq, if not can pass 'steady' in which case it uses converged criteria

# Spatial discretization
disc_type = 'FD' # 'FD', 'Rd', 'Rdn1', 'R0'
nn = 99
nelem = 0 # optional, number of elements
nen = 0 # optional, number of nodes per element
p = 2
isperiodic = None # set to none so it is done automatically

# Initial solution
q0 = None
n_q0 = 1
q0_type = None

# Other
bool_plot_sol = True
print_sol_norm = True
cons_obj_name=('Energy','Conservation') # note: should I modify this for systems?

bool_norm_var = True

if flow_is_subsonic:
    sc = 0.8
    k2 = 1/2
    k4 = 1/50
else:
    sc = 1
    k2 = 1/2
    k4 = 1/50

''' Setup diffeq and solve '''

if disc_type == 'FD':
    DiffEq = Quasi1dEulerFd
else:
    DiffEq = Quasi1dEulerSbp

diffeq = DiffEq(para, obj_name, q0_type, flow_is_subsonic)


solver = PdeSolver(diffeq,
                   tm_method, dt, tf, t_init,
                   q0, n_q0, dt_init,
                   p, disc_type, nn, nelem, nen,
                   isperiodic,
                   cons_obj_name=cons_obj_name,
                   bool_plot_sol = bool_plot_sol,
                   print_sol_norm = print_sol_norm)

solver.solve()

# if disc_type == 'FD':

#     diffeq = Quasi1dEulerFd(para, obj_name, nn,
#                             flow_is_subsonic = flow_is_subsonic)

#     # Solve with internal solve method
#     q_sol, res_norm = diffeq.solve(nts, 'implicit_euler', cn=50)

#     # Initial solution
#     rho_init, u_init, e_init, p_init, a_init = diffeq.cons2prim(diffeq.q_init, diffeq.svec)

#     # Plot the numerical solution
#     rho_sol, u_sol, e_sol, p_sol, a_sol = diffeq.cons2prim(q_sol, diffeq.svec)

#     diffeq.plot_fun(q_sol)


