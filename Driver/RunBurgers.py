#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:48:54 2020

@author: andremarchildon
"""

import os
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Burgers import Burgers
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg


''' Run code '''

# Eq parameters
para = None
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
dt_init = dt
t_init = 0
tf = 10.00

# Domain
xmin = -1
xmax = 1
isperiodic = True

# Spatial discretization
disc_type = 'lgl' 
nn = 60
nelem = 10 # optional, number of elements
nen = 0 # optional, number of nodes per element
p = 3
use_split_form = True
sat_flux_type = 'ec'

# Initial solution
q0 = None
q0_type = 'GassnerSinWave_cont' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False
cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

''' Set diffeq and solve '''

if disc_type == 'fd':
    c_solver = PdeSolverFd
elif disc_type == 'dg':
    c_solver = PdeSolverDg
else:
    c_solver = PdeSolverSbp

diffeq = Burgers(para, obj_name, q0_type, use_split_form)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}
savefile = 'burgers_alpha_23'
title=r'Burgers Eqn, $\alpha=2/3$, EC flux'

solver = c_solver(diffeq,                              # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type, nn,                     # Discretization
                  nelem, nen, sat_flux_type,
                  isperiodic, xmin, xmax,               # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

#solver.weakform = True
solver.force_steady_solution()
solver.perturb_q0()
A = solver.check_eigs(plt_save_name=savefile+'_eigs',returnA=True,title='Eigenvalues: ' + title)
eigs = np.linalg.eigvals(A)
max_eig = max(eigs.real)
def theory_fn(time):
    return 0.001*np.exp(max_eig * time)

solver.solve()
diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
solver.plot_sol(plt_save_name=savefile+'_sol',title=title)
solver.plot_error(method='max diff',savefile=savefile+'_error', extra_fn=theory_fn, extra_label='Theory', title=title)
#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)

#solver.solve()
#solver.plot_sol()
solver.plot_cons_obj()