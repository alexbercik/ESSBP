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


'''
Solve the quasi-one-dimensional converging-diverging nozzle from the textbook
Fundamental Algorithms in Computational Fluid Dynamics by Pulliam and Zingg.
The exact solution is available along with the algorithm from Chapter 4.
'''

# Eq parameters
para = [287,1.4] # [R, gamma]
test_case = 'density wave' # subsonic, transonic, shock tube, density wave
nozzle_shape = 'constant' # book, constant, linear, smooth

# Time marching
tm_method = 'explicit euler' # 'implicit_euler', 'explicit_euler', 'trapezoidal', 'rk4', 'bdf2'
dt = 0.0001
dt_init = dt
nts = 500
t_init = 0
tf = 0.5 #nts * dt # set to None to do automatically or use a convergence criterion
# note: can add option to pass None, then that triggers it to check diffeq, if not can pass 'steady' in which case it uses converged criteria

# TODO: Add flag that stops sim when it hits negative pressures

# Domain
xmin = 0
xmax = 1
bc = 'periodic' # 'periodic', 'dirichlet', 'riemann'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 2
nelem = 50 # number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'lf'
had_flux = 'central' # 2-point numerical flux used in hadamard form
diss_type = None

# output
savefile = None
title=r'1D Euler'

# Initial solution
q0 = None
q0_type = 'linear'

# Other
bool_plot_sol = False
print_sol_norm = False
cons_obj_name=('Energy','Conservation','Entropy') # note: should I modify this for systems?

bool_norm_var = False # what is this??


''' Set diffeq and solve '''
c_solver = PdeSolverSbp

diffeq = Quasi1dEuler(para, q0_type, test_case, nozzle_shape, bool_norm_var, bc)

diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}

solver = PdeSolverSbp(diffeq,                               # Diffeq
                      tm_method, dt, tf,                    # Time marching
                      q0,                                   # Initial solution
                      p, disc_type, nn,                     # Discretization
                      nelem, nen, sat_flux_type,
                      xmin, xmax,               # Domain
                      cons_obj_name,              # Other
                      bool_plot_sol, print_sol_norm)

A = solver.check_eigs(plt_save_name=savefile+'_eigs',returnA=True,title='Eigenvalues: ' + title)
#import numpy as np
#eigs = np.linalg.eigvals(A)
#max_eig = max(eigs.real)
#def theory_fn(time):
#    return 0.001*np.exp(max_eig * time)

diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
solver.solve()
solver.plot_sol(plt_save_name=savefile+'_sol',title=title,display_time=True)
solver.plot_error(method='max diff',savefile=savefile+'_error', title=title)
solver.plot_cons_obj(savefile=savefile)
#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)

#solver.solve()
#solver.plot_sol()
#solver.plot_cons_obj()