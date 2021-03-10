#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:25:19 2021

@author: bercik
"""
import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Quasi1dEulerA import Quasi1dEuler
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg


'''
Solve the quasi-one-dimensional converging-diverging nozzle from the textbook
Fundamental Algorithms in Computational Fluid Dynamics by Pulliam and Zingg.
The exact solution is available along with the algorithm from Chapter 4.
'''

# Eq parameters
para = None
obj_name = None
test_case = 'density wave' # subsonic, transonic, shock tube, density wave
nozzle_shape = 'constant' # book, constant, linear, smooth

# Time marching
tm_method = 'rk4' # 'implicit_euler', 'explicit_euler', 'trapezoidal', 'rk4', 'bdf2'
dt = 0.0001
dt_init = dt
nts = 500
t_init = 0
tf = 0.501 #nts * dt # set to None to do automatically or use a convergence criterion
# note: can add option to pass None, then that triggers it to check diffeq, if not can pass 'steady' in which case it uses converged criteria

# Spatial discretization
disc_type = 'lgl' 
nn = 99
nelem = 20 # optional, number of elements
nen = 0 # optional, number of nodes per element
p = 3
sat_flux_type='upwind'
vol_type='cons'
isperiodic = None # set to none so it is done automatically

# Initial solution
q0 = None
q0_type = 'linear'

# Other
bool_plot_sol = False
print_sol_norm = True
cons_obj_name=('Energy','Conservation','Entropy') # note: should I modify this for systems?

bool_norm_var = False
xmin = -1
xmax = 1


''' Set diffeq and solve '''

if disc_type == 'fd':
    c_solver = PdeSolverFd
elif disc_type == 'dg':
    c_solver = PdeSolverDg
else:
    c_solver = PdeSolverSbp

diffeq = Quasi1dEuler(para, obj_name, q0_type, test_case, nozzle_shape, bool_norm_var, vol_type)

solver = c_solver(diffeq,                              # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type, nn,                     # Discretization
                  nelem, nen, sat_flux_type,
                  isperiodic, xmin, xmax,               # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

solver.solve()
solver.plot_sol()
#solver.plot_cons_obj()