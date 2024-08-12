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

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


'''
Solve the 2D Euler equations
'''

# Eq parameters
para = [287,1.4] # [R, gamma]
test_case = 'density_wave_1d' # density_wave, vortex

# Time marching
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.0001
tf = 1.0 #nts * dt # set to None to do automatically or use a convergence criterion, or 'steady'
check_resid_conv = False

# Domain
xmin = (-1.,-1.)
xmax = (1.,1.)
bc = 'periodic' # 'periodic', 'dirichlet'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'lg' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 3
nelem = (4,4) # number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'central'
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vol_diss = {'diss_type':'nd', 'jac_type':'scalarmatrix', 's':'p', 'coeff':1.0, 'fluxvec':'sw'}

# Initial solution
q0 = None
q0_type = 'exact'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = False
cons_obj_name=('Energy','Entropy') # note: should I modify this for systems?
settings = {} # extra things like for metrics
skip_ts = 0


''' Set diffeq and solve '''

diffeq = Euler(para, q0_type, test_case, bc)

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
solver.skip_ts = skip_ts

solver.check_eigs()
solver.solve()
solver.plot_sol()#var2plot_name='mach')
solver.plot_cons_obj()
#print('Error: ', solver.calc_error())