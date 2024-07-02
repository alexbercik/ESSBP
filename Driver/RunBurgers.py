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
tf = 1.0

# Domain
xmin = -1.
xmax = 1.
bc = 'periodic' # no other boudnary conditions set up yet sorry...

# Spatial discretization
disc_type = 'div' # 'div', 'had' (divergence or hadamard-product)
disc_nodes = 'upwind' # 'lg', 'lgl', 'nc', 'csbp'
p = 6
nelem = 4 # optional, number of elements
nen = 20 # optional, number of nodes per element (set to zero for element-type)
surf_type = 'central' # 'ec' / 'es' / 'ec_had' / 'es_had' / 'split' / 'split_diss' (es is a dissipative version of ec, split follows variable coefficient advection splitting)
had_flux = 'ec_had' # 2-point numerical flux used in hadamard form (only 'ec' set up)
vol_diss = {'diss_type':'upwind', 'jac_type':'scalar', 's':'p', 'coeff':1.0}
use_split_form = True
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative had form

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GassnerSinWave' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
bool_plot_exa = True
print_sol_norm = False

cons_obj_name = ('Energy','Conservation','Energy_der') # 'Energy', 'Conservation', 'A_Energy', 'Entropy'
settings = {'warp_factor':0.0,               # Warps / stretches mesh.
            'warp_type': 'none',             # Options: 'defualt', 'papers', 'quad'
            'use_optz_metrics':True,         # Uses optimized metrics for free stream preservation.
            'use_exact_metrics':True}        # Uses exact metrics instead of interpolation.}


''' Set diffeq and solve '''
diffeq = Burgers(para, q0_type, use_split_form, split_alpha)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}
savefile = 'upwind_central'
title=r'Upwind LF Flux Vector Splitting w/ Central SATs, $\varepsilon = $'
#title=r"Entropy-Cons. + `Naive' Narrow Diss., $\varepsilon = 0.006$"
#title=r"Entropy-Cons. + Corrected Narrow Diss., $\varepsilon = 0.02$"
#title=r"Entropy-Cons. + Wide (Repeated D) Diss., $\varepsilon = 0.006$"
#title=r"Entropy-Conservative (No Dissipation)"

solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

#solver.force_steady_solution()
#solver.perturb_q0()
solver.check_eigs(savefile=savefile+'_eigs',returnA=False,title=title,exact_dfdq=True,xmin=-410,ymin=-240,ymax=240,xmax=30,overwrite=True)
#eigs = np.linalg.eigvals(A)
#max_eig = max(eigs.real)
#def theory_fn(time):
#    return 0.001*np.exp(max_eig * time)

solver.skip_ts = 100
#solver.check_eigs()
#solver.plot_sol(q=solver.diffeq.set_q0(),plot_exa=False)
#solver.solve()
#solver.check_eigs(title=r'Eigenvalues: LGL, Surface Dissipation (10 elem, $p=3$)',
#                  savefile='lgl_p3_es_nd', colour_by_k=True)
#diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
#solver.plot_sol(plt_save_name=savefile+'_sol',title=title)
#solver.plot_sol(q=solver.diffeq.set_q0(),plot_exa=False,title=r'Initial CSBP State, 2 elem',savefile='Linearization_State_csbp',legend=False)
#solver.plot_error(method='max diff',savefile=savefile+'_error', extra_fn=theory_fn, extra_label='Theory', title=title)
#solver.plot_cons_obj()
#solver.plot_sol()
#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)