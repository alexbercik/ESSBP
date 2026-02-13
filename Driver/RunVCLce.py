#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:02 2020

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

from Source.DiffEq.VarCoeffLinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp

# Eq parameters
alpha = 1.0    # Variable coefficient splitting parameter (0 to 1)
use_exact_der = False # whether to compute variable coefficient derivative exactly
extrapolate_bdy_flux = True

# Time marching
tm_method = 'rk8' # explicit_euler, rk4
dt = 0.001
# note: should set according to courant number C = a dt / dx
tf = 1.5

# Domain
xmin = 0.
xmax = 1
bc = 'periodic' # 'periodic' or 'homogeneous'

# Spatial discretization
flux_type = 'geometric' # 'product' or 'geometric'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = 10 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'es'
vol_diss = {'diss_type':'nd', 'use_H':False, 'bdy_fix':False, 
            'jac_type':'scalar', 's':int(p/2)+1, 'coeff':1./5**(int(p/2)+1)}

# Initial solution
q0_type = 'GaussWave_shift' # 'GaussWave', 'SinWave'
a_type = 'skewed_sin'

# Other
bool_plot_sol = False
print_sol_norm = False

#cons_obj_name = ('Energy','Conservation','A_Energy','Energy_der','Conservation_der','A_Energy_der') # 'Energy', 'Conservation', 'None'
cons_obj_name = ('Energy','Conservation','A_Energy','entropy','time')


settings = {'warp_factor':0,               # Warps / stretches mesh.
            'warp_type': 'default',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'calculate', 'exact'
            'jac_method':'exact'} 

''' Set diffeq and solve '''
    
diffeq = LinearConv(alpha, q0_type, a_type, flux_type) 
diffeq.use_exact_der = use_exact_der
diffeq.extrapolate_bdy_flux = extrapolate_bdy_flux

solver = PdeSolverSbp(diffeq, settings,             
                  tm_method, dt, tf,                  
                  p, 'div',           
                  surf_type, vol_diss, None,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,    
                  cons_obj_name,   
                  bool_plot_sol, print_sol_norm)
solver.tm_atol = 1e-10
solver.tm_rtol = 1e-10
solver.tm_nframes = 50

solver.solve()
solver.plot_sol()
solver.plot_cons_obj()
print(solver.calc_error())


from Source.Methods.Analysis import run_convergence
schedule = [['nelem',20,40,80,160,320]]
#schedule = [['nen',80,160,320,640]]
dofs, errors, labels = run_convergence(solver,schedule_in=schedule,plot=True,return_conv=True, figsize=(5,4))
solver.plot_sol()