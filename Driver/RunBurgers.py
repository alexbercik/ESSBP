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
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


''' Run code '''

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
tf = 0.26

# Domain
xmin = -1.
xmax = 1.
bc = 'periodic' 

# Spatial discretization
disc_type = 'had' # 'div', 'had' (divergence or hadamard-product)
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp'
p = 3
nelem = 10 # optional, number of elementss
nen = 0 # optional, number of nodes per element (set to zero for element-type)
had_flux = 'ec' # 2-point numerical flux used in hadamard form (only 'ec' and 'central' set up)
surf_diss = {'diss_type':'ent', 'jac_type':'scasca', 'maxeig':'lf', 'coeff':1.0}
vol_diss = {'diss_type':'nd', 'jac_type':'scalar', 's':'p+1', 'coeff':1.0}
use_split_form = True
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative had form

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'SinWave' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False
skip_ts = 10

cons_obj_name = ('Energy','Conservation','Energy_der') # 'Energy', 'Conservation', 'A_Energy', 'Entropy'
settings = {'warp_factor':0.0,               # Warps / stretches mesh.
            'warp_type': 'none',             # Options: 'defualt', 'papers', 'quad'
            'use_optz_metrics':True,         # Uses optimized metrics for free stream preservation.
            'use_exact_metrics':True}        # Uses exact metrics instead of interpolation.}


''' Set diffeq and solve '''
diffeq = Burgers(None, q0_type, use_split_form, split_alpha)
solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_diss, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)
solver.skip_ts = skip_ts

diffeq.calc_breaking_time()
solver.solve()
solver.plot_sol()
solver.plot_cons_obj()
solver.check_eigs(q=diffeq.exact_sol(time=0.2))