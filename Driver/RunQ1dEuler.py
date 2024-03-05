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
test_case = 'density_wave' # subsonic_nozzle, transonic, shock_tube, density_wave
nozzle_shape = 'book' # book, constant, linear, smooth
#TODO: transonic does not work

# Time marching
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.0001
tf = 1 #nts * dt # set to None to do automatically or use a convergence criterion
check_resid_conv = False

# Domain
xmin = 0
xmax = 1
bc = 'periodic' # 'periodic', 'dirichlet', 'riemann'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 2
nelem = 5 # number of elements
nen = 10 # optional, number of nodes per element
surf_type = 'nondissipative'
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vol_diss = {'diss_type':'B', 'jac_type':'scalar', 's':p}

# output
savefile = None
title=r'1D Euler'

# Initial solution
q0 = None
q0_type = 'density_wave'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = False
cons_obj_name=('Energy','Conservation','Entropy') # note: should I modify this for systems?
settings = {} # extra things like for metrics



''' Set diffeq and solve '''

diffeq = Quasi1dEuler(para, q0_type, test_case, nozzle_shape, bc)

diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}

solver = PdeSolverSbp(diffeq, settings, 
                  tm_method, dt, tf,   
                  q0,                
                  p, disc_type,      
                  surf_type, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,     
                  cons_obj_name,      
                  bool_plot_sol, print_sol_norm,
                  print_residual, check_resid_conv)

if savefile is not None:
    eigs_savefile = savefile+'_eigs'
else:
    eigs_savefile = None
#A = solver.check_eigs(plt_save_name=eigs_savefile,returnA=True,title='Eigenvalues: ' + title,plot_eigs=False)
#import numpy as np
#eigs = np.linalg.eigvals(A)
#max_eig = max(eigs.real)
#def theory_fn(time):
#    return 0.001*np.exp(max_eig * time)

#diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
#solver.solve()
#solver.plot_sol(plt_save_name=savefile+'_sol',title=title,display_time=True)
#solver.plot_error(method='max diff',savefile=savefile+'_error', title=title)
#solver.plot_cons_obj(savefile=savefile)
#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)

#solver.check_eigs()
#solver.plot_sol(q=solver.diffeq.set_q0(),time=0.)
solver.solve()
solver.plot_sol()
solver.plot_cons_obj()
#solver.calc_error()