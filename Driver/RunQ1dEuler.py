#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:25:19 2021

@author: bercik
"""
import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
#from Source.Methods.Analysis import animate


'''
Solve the quasi-one-dimensional converging-diverging nozzle from the textbook
Fundamental Algorithms in Computational Fluid Dynamics by Pulliam and Zingg.
The exact solution is available along with the algorithm from Chapter 4.
'''

# Eq parameters
para = [287,1.4] # [R, gamma]
test_case = 'density_wave' # subsonic_nozzle, transonic, shock_tube, density_wave, manufactured_soln
nozzle_shape = 'constant' # book, constant, linear, smooth
#TODO: transonic does not work

# Time marching
tm_method = 'rk8' # 'explicit_euler', 'rk4'
dt = 0.001
tf = 50.0 #nts * dt # set to None to do automatically or use a convergence criterion, or 'steady'
check_resid_conv = False

# Domain
xmin = -1.
xmax = 1.
bc = 'periodic' # 'periodic', 'dirichlet'
# Spatial discretization
disc_type = 'had' # 'div', 'had'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = 1 # number of elements
nen = 120 # optional, number of nodes per element
had_flux = 'chandrashekar' # 2-point numerical flux used in hadamard form
#surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1.0, 'P_derigs':True, 'A_derigs':True,
#    'entropy_fix':False, 'average':'none', 'maxeig':'none'}
surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'roe', 
            'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
vol_diss = {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':3.125/5**(p+1), 's':p+1, 
            'bdy_fix':True, 'use_H':False, 'entropy_fix':False, 'avg_half_nodes':True}
#vol_diss = {'diss_type':'nd', 'jac_type':'matmat', 's':'p+1', 'coeff':1, 'fluxvec':'sw', 'bdy_fix':True, 'use_H':True}
nondimensionalize = False

# output
#savefile = None
#title=r'1D Euler'
savefile = 'nd_ec'
#title=r'Upwind Lax-Friedrichs Splitting w/ LF SATs, $\varepsilon = 1$'
#title=r"Entropy-Diss. + `Naive' Narrow Diss., $\varepsilon = 0.01$"
#title=r"Entropy-Diss. + Corrected Narrow Diss., $\varepsilon = 0.004$"
#title=r"Entropy-Diss. + Wide (Repeated D) Diss., $\varepsilon = 0.02$"
title=r"Entropy-Conservative (No Dissipation)"

# Initial solution
q0_type = 'density_wave' #'gassnersinwave_cont'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = False
cons_obj_name=('time', 'Energy','Entropy') # note: should I modify this for systems?
settings = {'warp_factor':0.0,               # Warps / stretches mesh.
            'warp_type': 'default'} # extra things like for metrics
skip_ts = 0


''' Set diffeq and solve '''

diffeq = Quasi1dEuler(para, q0_type, test_case, nozzle_shape, bc, nondimensionalize)

diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}

if nen >= 60:
    sparse = True
else:
    sparse = False
solver = PdeSolverSbp(diffeq, settings, 
                  tm_method, dt, tf,                  
                  p, disc_type,      
                  surf_diss, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,     
                  cons_obj_name,      
                  bool_plot_sol, print_sol_norm,
                  print_residual, check_resid_conv,
                  sparse = True)
solver.skip_ts = skip_ts

if savefile is not None:
    eigs_savefile = savefile+'_eigs'
    sol_savefile = savefile+'_sol'
    err_savefile = savefile+'_error'
    cons_savefile = savefile+'_cons'
else:
    eigs_savefile = None
    sol_savefile = None
    err_savefile = None
    cons_savefile = None

import numpy as np
#q = np.random.rand(*solver.qshape) + 0.1

#solver.check_eigs(plot_eigs=True)
solver.tm_atol=1e-7
solver.tm_rtol=1e-7
solver.solve()
solver.plot_sol()

# # make external function for finite difference and complex step
# from Source.Methods.DebugTools import compare_Jacobian, calcJacobian_complex_step, calcJacobian_finite_diff
# import Source.Methods.Functions as fn
# import Source.DiffEq.EulerFunctions as efn
# #qL = np.random.rand(1,nelem+1) + 0.1
# #qR = np.random.rand(1,nelem+1) + 0.1

# def f(q):
#     sat = solver.dqdt(q,0.0)
#     return sat

# num = 1000
# i = 0
# ok = True
# while (i < num) and ok:
#     q = np.random.rand(*solver.qshape) + 0.1
#     if not diffeq.check_positivity(q):
#         A_complex, A_finite, ok = compare_Jacobian(f, q, h=1.0e-5, hi=1.0e-15, returnA=True, returnbool=True)
#         i += 1
#         if not ok:
#             if np.any(np.isnan(A_complex)):
#                 print('Complex step has nan entries.')
#             if np.any(np.isnan(A_finite)):
#                 print('Finite difference has nan entries.')