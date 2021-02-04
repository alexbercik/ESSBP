#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:02 2020

@author: andremarchildon
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConvSbp
from Source.DiffEq.LinearConv import LinearConvFd
from Source.DiffEq.LinearConv import LinearConvDg
from Source.Solvers.PdeSolver import PdeSolver

''' Run code '''

# Eq parameters
para = 1      # Wave speed a
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 2.0

# Domain
xmin = -1
xmax = 1
isperiodic = True

# Spatial discretization
disc_type = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
nn = 50
p = 2
nelem = 20 # optional, number of elements
nen = 0 # optional, number of nodes per element
sat_flux_type = 'upwind'

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False

obj_name = None
cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

''' Set diffeq and solve '''

if disc_type == 'fd':
    DiffEq = LinearConvFd
elif disc_type == 'dg':
    DiffEq = LinearConvDg
else:
    DiffEq = LinearConvSbp

diffeq = DiffEq(para, obj_name, q0_type)

solver = PdeSolver(diffeq,                              # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type, nn,                     # Discretization
                  nelem, nen, sat_flux_type,
                  isperiodic, xmin, xmax,               # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

solver.solve()
solver.plot_sol()