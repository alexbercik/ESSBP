#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:13:29 2021

@author: bercik
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv3D import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Run code '''

# Eq parameters
para = [1,1,1]     # Wave speed ax, ay, az
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 1.0

# Domain
xmin = (0,0,0)
xmax = (1,1,1)
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had', 'dg'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 3
nelem = (4,4,4) # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'lf'
had_flux = 'central_fix' # 2-point numerical flux used in hadamard form
diss_type = None

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
bool_plot_exa = True
print_sol_norm = False

obj_name = None
cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

settings = {'warp_factor':0.15,               # Warps / stretches mesh.
            'warp_type':'strong',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'VinokurYee','ThomasLombard','exact'
            'bdy_metric_method':'exact',   # Options: 'VinokurYee','ThomasLombard','interpolate','exact'
            'jac_method':'direct',      # Options: 'direct','match','deng','exact'
            'use_optz_metrics':False,        # Uses optimized metrics for free stream preservation.
            'calc_exact_metrics':True,      # calculate exact metrics alongside above choices.
            'metric_optz_method':'alex',    # Define the optimization procedure.
            'stop_after_metrics': True } # Do not set up physical operators, SATs, etc. only Mesh setup.

''' Set diffeq and solve '''

if disc_type == 'fd':
    solver_c = PdeSolverFd
elif disc_type == 'dg':
    solver_c = PdeSolverDg
else:
    solver_c = PdeSolverSbp

diffeq = LinearConv(para, obj_name, q0_type)

solver3D = solver_c(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)


''' Analyze results '''

#solver3D.solve()
#solver3D.plot_cons_obj()
#print('Final Error: ', solver3D.calc_error())

from Source.Methods.Analysis import run_convergence, run_jacobian_convergence
#schedule = [['disc_nodes','lg','lgl'],['p',3,4],['nelem',12,15,20]]
#run_convergence(solver,schedule_in=schedule)
schedule = [['disc_nodes','lg', 'lgl'],['p',3,4],['nelem',3,6,12,24,40]]
#schedule = [['disc_nodes','lg'],['p',3,4],['nelem',3,6,12]]
#run_convergence(solver2D,schedule_in=schedule)
dofs, avg_jacs, max_jacs, legend_strings = run_jacobian_convergence(solver3D,
                                schedule_in=schedule,return_conv=True,savefile='jac_deng_met_optz_3D')