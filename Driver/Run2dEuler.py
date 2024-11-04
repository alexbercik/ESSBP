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
test_case = 'vortex' # density_wave, vortex

# Time marching
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.005
# for vortex, maxeig < 1.6, so CFL = 0.1 gives dt = 0.16 * dx and CFL = 0.01 gives dt = 0.016 * dx
tf = 4.0 #nts * dt # set to None to do automatically or use a convergence criterion, or 'steady'
# for vortex, one period is t=20
check_resid_conv = False

# Domain
xmin = (-5.,-5.)
xmax = (5.,5.)
bc = 'periodic' # 'periodic', 'dirichlet'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = (1,1) # number of elements
nen = 30 # optional, number of nodes per element
surf_diss = {'diss_type':'nd', 'jac_type':'matmat', 'coeff':1., 'average':'simple', 
             'entropy_fix':True, 'P_derigs':True, 'A_derigs':True, 'maxeig':'rusanov'}
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vol_diss = {'diss_type':'nd', 'jac_type':'matmat', 's':'p+1', 'coeff':3.125/5**(p+1),
            'fluxvec':'dt', 'bdy_fix':True, 'use_H':True, 'entropy_fix':True, 'avg_half_nodes':True}

# Initial solution
q0 = None
q0_type = 'exact'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = False
cons_obj_name=('Energy','Entropy','Conservation') # note: should I modify this for systems?
settings = {'metric_method':'exact',
            'use_optz_metrics':False} # extra things like for metrics
skip_ts = 99


''' Set diffeq and solve '''

diffeq = Euler(para, q0_type, test_case, bc)

diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}

solver = PdeSolverSbp(diffeq, settings, 
                  tm_method, dt, tf,   
                  q0,                
                  p, disc_type,      
                  surf_diss, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,     
                  cons_obj_name,      
                  bool_plot_sol, print_sol_norm,
                  print_residual, check_resid_conv,
                  sparse=True,sat_sparse=True)
#solver.skip_ts = skip_ts

#eigs = solver.check_eigs(returneigs=True)
#solver.solve()
#print('Error: ', solver.calc_error())
#solver.plot_sol()#var2plot_name='entropy')
#solver.plot_cons_obj()
#solver.plot_sol()
#solver.check_conservation()

q0 = diffeq.set_q0()
dqdt = solver.dqdt(q0,0.)