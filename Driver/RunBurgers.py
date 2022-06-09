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
dt = 0.001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 10

# Domain
xmin = 0
xmax = 1
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had', 'dg'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 3
nelem = 10 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'symmetric'
had_flux = 'ec' # 2-point numerical flux used in hadamard form
diss_type = None
use_split_form = True
split_alpha = 1/2

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GassnerSinWave' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
bool_plot_exa = True
print_sol_norm = False

obj_name = None
cons_obj_name = ('Energy','Conservation','Entropy') # 'Energy', 'Conservation', 'None'
settings = {}


''' Set diffeq and solve '''

if disc_type == 'fd':
    c_solver = PdeSolverFd
elif disc_type == 'dg':
    c_solver = PdeSolverDg
else:
    c_solver = PdeSolverSbp

diffeq = Burgers(para, obj_name, q0_type, use_split_form, split_alpha)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}
savefile = 'burgers_alpha1_symSAT'
title=r'Burgers Eqn, $\alpha=1$, sym flux'


solver = c_solver(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

#solver.weakform = True
solver.force_steady_solution()
solver.perturb_q0()
A = solver.check_eigs(plt_save_name=savefile+'_eigs',returnA=True,title='Eigenvalues: ' + title,exact_dfdq=True)
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
solver.plot_sol()
solver.plot_cons_obj()