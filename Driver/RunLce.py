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

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 5.00

# Domain
xmin = 0
xmax = 1
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = 50 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'central'
had_flux = 'central' # 2-point numerical flux used in hadamard form. 
# note: use 'central_fix' instead of 'central' for speed, but then fixes a=1
diss_type = None

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False

cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

settings = {'warp_factor':0.0,               # Warps / stretches mesh.
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

diffeq = LinearConv(para, q0_type)

solver1D = solver_c(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)


''' Analyze results '''

solver1D.solve()
solver1D.plot_sol()
solver1D.plot_cons_obj()
#print('Final Error: ', solver1D.calc_error())

#from Source.Methods.Analysis import run_convergence
#schedule = [['disc_nodes','csbp'],['p',1,2,3,4],['nen',100,200,300,400]]
#dofs, errors, labels = run_convergence(solver1D,schedule_in=schedule,savefile='convergence.png',
#                title=r'Error Convergence', xlabel=r'Nodes',grid=True,return_conv=True,
#                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
#                labels=[r'$p=1$ SBP',r'$p=2$ SBP',r'$p=3$ SBP',r'$p=4$ SBP'])
