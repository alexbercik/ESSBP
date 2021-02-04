#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:48:54 2020

@author: andremarchildon
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Burgers import BurgersSbp
from Source.DiffEq.Burgers import BurgersFd
from Source.DiffEq.Burgers import BurgersDg
from Source.Solvers.PdeSolver import PdeSolver


''' Run code '''

# Eq parameters
para = None
obj_name = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.002
dt_init = dt
t_init = 0
tf = 0.2

# Domain
xmin = -1
xmax = 1
isperiodic = True

# Spatial discretization
disc_type = 'lgl' 
nn = 60
nelem = 0 # optional, number of elements
nen = 0 # optional, number of nodes per element
p = 2
use_split_form = False
sat_flux_type = 'upwind'

# Initial solution
q0 = None
q0_type = 'GaussWave' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False
cons_obj_name = ('Energy','Conservation') # 'Energy', 'Conservation', 'None'

''' Set diffeq and solve '''

if disc_type == 'fd':
    DiffEq = BurgersFd
elif disc_type == 'dg':
    DiffEq = BurgersDg
else:
    DiffEq = BurgersSbp

diffeq = DiffEq(para, obj_name, q0_type, use_split_form)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':''}

solver = PdeSolver(diffeq,                              # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type, nn,                     # Discretization
                  nelem, nen, sat_flux_type,
                  isperiodic, xmin, xmax,               # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

#solver.force_steady_solution()
#solver.perturb_q0()
#solver.check_eigs()

solver.solve()
diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
solver.plot_sol()
#solver.plot_error(method='max diff')
#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)
