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

from Source.DiffEq.LinearConv2D import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Run code '''

# Eq parameters
para = [1,1]     # Wave speed ax, ay
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 1

# Domain
xmin = (0,0)
xmax = (1,1)
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had', 'dg'
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = (40,40) # optional, number of elements
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

settings = {'warp_factor':0.99,               # Warps / stretches mesh.
            'warp_type': 'strong',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'calculate', 'exact'
            'bdy_metric_method':'exact',   # Options: 'calculate', 'exact', 'extrapolate'
            'use_optz_metrics':True,        # Uses optimized metrics for free stream preservation.
            'calc_exact_metrics':True,      # Calculates the exact metrics (useless if metric_method=exact).
            'metric_optz_method':'alex'} # Define the optimization procedure.

''' Set diffeq and solve '''

if disc_type == 'fd':
    solver_c = PdeSolverFd
elif disc_type == 'dg':
    solver_c = PdeSolverDg
else:
    solver_c = PdeSolverSbp

diffeq = LinearConv(para, obj_name, q0_type)

solver2D = solver_c(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)


''' Analyze results '''

#solver2D.check_conservation()
#solver2D.mesh.check_surface_metrics()

#solver2D.solve()
#solver2D.plot_sol(plot_exa=bool_plot_exa)
#solver2D.plot_cons_obj()
#print('Final Error: ', solver2D.calc_error())

from Source.Methods.Analysis import run_convergence, run_jacobian_convergence
#schedule = [['disc_nodes','lg','lgl'],['p',3,4],['nelem',12,15,20,25,40]]
schedule = [['disc_nodes','lg', 'lgl'],['p',3,4],['nelem',10,20,40,80,160]]
#run_convergence(solver2D,schedule_in=schedule)
#dofs, avg_jacs, max_jacs, legend_strings = run_jacobian_convergence(solver2D,
#                                schedule_in=schedule,return_conv=True,savefile='jac_convergence')