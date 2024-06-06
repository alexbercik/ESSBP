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
from Source.Methods.Analysis import animate


'''
Solve the quasi-one-dimensional converging-diverging nozzle from the textbook
Fundamental Algorithms in Computational Fluid Dynamics by Pulliam and Zingg.
The exact solution is available along with the algorithm from Chapter 4.
'''

# Eq parameters
para = [287,1.4] # [R, gamma]
test_case = 'subsonic_nozzle' # subsonic_nozzle, transonic, shock_tube, density_wave
nozzle_shape = 'smooth' # book, constant, linear, smooth
#TODO: transonic does not work

# Time marching
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.0001
tf = 'steady' #nts * dt # set to None to do automatically or use a convergence criterion
check_resid_conv = True

# Domain
xmin = 0
xmax = 10
bc = 'dirichlet' # 'periodic', 'dirichlet', 'riemann'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 2
nelem = 6 # number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'upwind'
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vol_diss = {'diss_type':'dcp', 'jac_type':'scalar', 's':'p', 'coeff':0.1}

# output
savefile = None
title=r'1D Euler'

# Initial solution
q0 = None
q0_type = 'exact'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = True
cons_obj_name=('Energy','Conservation','Entropy') # note: should I modify this for systems?
settings = {} # extra things like for metrics
skip_ts = 9999


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

A = solver.check_eigs(savefile=eigs_savefile,returnA=True,title='Eigenvalues: ' + title,plot_eigs=True,exact_dfdq=False)
#import numpy as np
#eigs = np.linalg.eigvals(A)
#max_eig = max(eigs.real)
#def theory_fn(time):
#    return 0.001*np.exp(max_eig * time)

#diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
#solver.solve()
#solver.plot_sol(savefile=sol_savefile,title=title,display_time=True)
#solver.plot_error(method='max diff',savefile=err_savefile, title=title)
#solver.plot_cons_obj(savefile=savefile)

#solver.check_eigs()
#solver.plot_sol(q=solver.diffeq.set_q0(),time=0.)
#solver.solve()
#animate(solver, plotargs={'display_time':True, 'legendloc':'lower right'},skipsteps=0, last_frame=False)
#solver.plot_sol()#var2plot_name='mach')
#solver.plot_cons_obj()
#solver.calc_error()


"""
#rho = 0.02
#u = 0.1
#p = 20
rho = 1.
u = 150
p = 90000
rhou = rho*u
e = (p/0.4)+0.5*rho*u*u
q = np.array([rho,rhou,e]).reshape(3,1)

w = solver.diffeq.entropy_var(q)[:,0]
A = solver.diffeq.dExdq(q)[:,:,0]
P = solver.diffeq.dqdw(q)[:,:,0]
APabs = solver.diffeq.dExdw_abs(q)[:,:,0]

lambdas = np.linalg.eigvals(A)
lambda_max = np.max(np.abs(lambdas))
lambdas2 = np.linalg.eigvals(P)
lambda_max2 = np.max(np.abs(lambdas2))

np.set_printoptions(precision=1)
print('----- Variables & Matrices ------')
print('Conservative vars (rho, rho*u, e): {0:.1e}, {1:.1e}, {2:.1e}'.format(*q[:,0]))
print('Flux Jacobian A=dfdu:', '\t' + str(A).replace('\n', '\n\t\t\t'))
print('eigvals of A=dfdu: {0:.1e}, {1:.1e}, {2:.1e}'.format(*lambdas))
print('Entropy vars: {0:.1e}, {1:.1e}, {2:.1e}'.format(*w))
print('Symmetrizer dudw:', '\t' + str(P).replace('\n', '\n\t\t\t'))
print('eigvals of dudw: {0:.1e}, {1:.1e}, {2:.1e}'.format(*lambdas2))
print('----- Scalar Baseline ------')
print('dissipation ~ lambda_max = {0:.1e}'.format(lambda_max))
print('----- Scalar-Scalar Entropy ------')
print('dissipation ~ lambda_max * rho(dudw) = {0:.1e}'.format(lambda_max2))
print('----- Scalar-Matrix Entropy ------')
print('dissipation ~ lambda_max * dudw =', str(lambda_max*P).replace('\n', '\n\t\t\t\t'))
print('----- Matrix-Matrix Entropy ------')
print('dissipation ~ abs(dfdu * dudw) =', str(APabs).replace('\n', '\n\t\t\t\t'))
"""