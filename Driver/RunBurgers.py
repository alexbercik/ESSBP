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

# Eq parameters
para = None

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
t_init = 0
tf = 0.5

# Domain
xmin = 0
xmax = np.pi
bc = 'periodic' # no other boudnary conditions set up yet sorry...

# Spatial discretization
disc_type = 'div' # 'div', 'had' (divergence or hadamard-product)
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp'
p = 3
nelem = 20 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'ec' # 'ec' / 'es' / 'ec_had' / 'es_had' / 'split' / 'split_diss' (es is a dissipative version of ec, split follows variable coefficient advection splitting)
had_flux = 'ec' # 2-point numerical flux used in hadamard form (only 'ec' set up)
diss_type = None # not set up yet
use_split_form = True
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative had form

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'ShiftedSinWave' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
bool_plot_exa = True
print_sol_norm = False

cons_obj_name = ('Energy','Conservation','Entropy','Energy_der','A_Energy') # 'Energy', 'Conservation', 'None'
settings = {'warp_factor':0.0,               # Warps / stretches mesh.
            'warp_type': 'none',             # Options: 'defualt', 'papers', 'quad'
            'use_optz_metrics':True,         # Uses optimized metrics for free stream preservation.
            'use_exact_metrics':True}        # Uses exact metrics instead of interpolation.}


''' Set diffeq and solve '''
diffeq = Burgers(para, q0_type, use_split_form, split_alpha)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}
savefile = 'burgers_alpha1_symSAT'
title=r'Burgers Eqn, $\alpha={0:1.2}$, sym flux'.format(split_alpha)

solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

#solver.force_steady_solution()
#solver.perturb_q0()
#A = solver.check_eigs(plt_save_name=savefile+'_eigs',returnA=True,title='Eigenvalues: ' + title,exact_dfdq=True)
#eigs = np.linalg.eigvals(A)
#max_eig = max(eigs.real)
#def theory_fn(time):
#    return 0.001*np.exp(max_eig * time)

solver.solve()
solver.check_eigs()
diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
#solver.plot_sol(plt_save_name=savefile+'_sol',title=title)
solver.plot_sol()
#solver.plot_error(method='max diff',savefile=savefile+'_error', extra_fn=theory_fn, extra_label='Theory', title=title)
solver.plot_cons_obj()

#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)