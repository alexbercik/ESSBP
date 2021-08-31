#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:02 2020

@author: bercik
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Set parameters for simultation '''

# Eq parameters
para = 1      # Wave speed a
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 1.00

# Domain
xmin = 0
xmax = 1
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had', 'dg'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = 10 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'lf'
had_flux = 'central_fix' # 2-point numerical flux used in hadamard form. 
# note: use 'central_fix' instead of 'central' for speed, but then fixes a=1
diss_type = None

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False

obj_name = None
cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

settings = {'warp_factor':0.2,               # Warps / stretches mesh.
            'warp_type': 'default',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'VinokurYee' and 'ThomasLombard'
            'use_optz_metrics':True,        # Uses optimized metrics for free stream preservation.
            'use_exact_metrics':True}      # Uses exact metrics instead of interpolation.}

''' Set diffeq and solve '''

if disc_type == 'fd':
    solver_c = PdeSolverFd
elif disc_type == 'dg':
    solver_c = PdeSolverDg
else:
    solver_c = PdeSolverSbp

diffeq = LinearConv(para, obj_name, q0_type)

solver1D = solver_c(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)


''' Analyze results '''

solver1D.solve()
solver1D.plot_sol()
#solver1D.plot_cons_obj()
#print('Final Error: ', solver1D.calc_error())

from Source.Methods.Analysis import run_convergence
schedule = [['disc_nodes','lg','lgl'],['p',3,4],['nelem',12,15,20,25,40]]
#run_convergence(solver1D,schedule_in=schedule)
