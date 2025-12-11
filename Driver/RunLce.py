#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:02 2020

@author: bercik
"""

import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Set parameters for simultation '''

# Eq parameters
para = 1.0      # Wave speed a

# Time marching
tm_method = 'rk8' # explicit_euler, rk4
tm_method = 'rk8' # explicit_euler, rk4
dt = 0.001 # for convergence studies, try to choose at least C=0.02
# note: should set according to courant number C = a dt / dx
tf = 2

# Domain
xmin = -1.
xmax = 1.
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'lg_exp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd', 'upwind'
p = 0

nelem = 4 # optional, number of elements
nen = 3 # optional, number of nodes per element
surf_type = 'lf'

had_flux = 'central' # 2-point numerical flux used in hadamard form. 
vol_diss = {'diss_type':'nd', 'use_H':False, 'bdy_fix':True, 
            'jac_type':'scalar', 's':p+1, 'coeff':0.2*3.125/5**(p+1), 'beta':4, 
            'fluxvec':'lf', 'eps_type':2, 'D_type':'sbp'}
# Initial solution
q0_type = 'SinWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False

cons_obj_name = ('Energy','Conservation','Spec_Rad','time') # 'Energy', 'Conservation', 'None'

settings = {'warp_factor':0, #[0.05,1.0,40],               # Warps / stretches mesh.
            'warp_type': 'corners_periodic',         # Options: 'defualt', 'papers', 'quad'
            'jac_method':'exact'}   # Options: 'direct, 'exact'

''' Set diffeq and solve '''

diffeq = LinearConv(para, q0_type)

solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  p, disc_type,             # Discretization
                  surf_type, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)
solver.tm_nframes = 100
solver.tm_atol = 1e-8
solver.tm_rtol = 1e-8

''' Analyze results '''
#solver.mesh.plot()  # Plot mesh

#solver.skip_ts = 999
#solver.tm_nframes = 100
#solver.check_eigs()
solver.solve()
#solver.plot_sol()#q=solver.diffeq.set_q0(),plot_exa=False)
#solver.plot_cons_obj()

from Source.Methods.Analysis import run_convergence
#schedule = [['disc_nodes','csbp'],['p',1,2,3,4],['nen',100,200,300,400]]
#label = [r'$p=1$ SBP',r'$p=2$ SBP',r'$p=3$ SBP',r'$p=4$ SBP']
schedule1 = [['disc_nodes','csbp'],['nen',10,20,40,80],['p',2,3,4],['surf_type','central'],
            ['vol_diss',{'diss_type':'W', 'jac_type':'scalar', 's':'p', 'coeff':0.1},
                        {'diss_type':'W', 'jac_type':'scalar', 's':'p+1', 'coeff':0.1}]]
schedule2 = [['disc_nodes','csbp'],['nen',10,20,40,80],['p',2,3,4],['surf_type','central'],
            ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':'p', 'coeff':0.01},
                        {'diss_type':'dcp', 'jac_type':'scalar', 's':'p+1', 'coeff':0.01}]]
schedule3 = [['disc_nodes','csbp'],['nen',10,20,40,80],['p',2,3,4],['surf_type','central','upwind'],
            ['vol_diss',{'diss_type':'ND'}]]
schedule4 = [['disc_nodes','upwind'],['nen',10,20,40,80],['p',4,5,6,7,8,9],['surf_type','central'],
            ['vol_diss',{'diss_type':'upwind', 'fluxvec':'lf', 'coeff':1.}]]
schedule5 = [['disc_nodes','csbp'],['nen',10,20,40,80],['p',2,3,4],['surf_type','central'],
            ['vol_diss',{'diss_type':'upwind', 'fluxvec':'lf', 's':'2p-1', 'coeff':1.},
                        {'diss_type':'upwind', 'fluxvec':'lf', 's':'2p', 'coeff':1.},
                        {'diss_type':'upwind', 'fluxvec':'lf', 's':'2p+1', 'coeff':1.}]]
#label = None
label = [r'$p=2,s=2$',r'$p=2,s=3$',r'$p=3,s=3$',r'$p=3,s=4$',r'$p=4,s=4$',r'$p=4,s=5$']
label3 = [r'$p=2$, C',r'$p=2$, U',r'$p=3$, C',r'$p=3$, U',r'$p=4$, C',r'$p=4$, U']
label4 = [r'$p=4(2)$',r'$p=5(2)$',r'$p=6(3)$',r'$p=7(3)$',r'$p=8(4)$',r'$p=9(4)$']
label5 = [r'$p=2,s=3(1)$',r'$p=2,s=4(2)$',r'$p=2,s=5(2)$',r'$p=3,s=5(2)$',r'$p=3,s=6(3)$',r'$p=3,s=7(3)$',r'$p=4,s=7(3)$',r'$p=4,s=8(4)$',r'$p=4,s=9(4)$']

ylim=(1e-8,1e-1)
# TODO: check mattsson s=2p-2 or even lower to see when accuracy begins to degrade
# figure out what the hell is wrong with DCP

"""
dofs, errors, labels = run_convergence(solver,schedule_in=schedule1,savefile='CSBP_W_01.png',
                title=r'CSBP Wide (Repeated D) Dissipation $\epsilon=0.1$', xlabel=r'Num. Nodes',grid=True,return_conv=True,
                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
                labels=label, ylim=ylim)
"""
#solver = solver_c(diffeq,settings,tm_method,dt,tf,q0,p,disc_type,surf_type,vol_diss,had_flux,nelem,nen,disc_nodes,bc,xmin,xmax,cons_obj_name,bool_plot_sol,print_sol_norm)
dofs2, errors2, labels2 = run_convergence(solver,schedule_in=schedule2,savefile='CSBP_dcp_001.png',
                title=r"CSBP Narrow `Naive' Dissipation, $\epsilon=0.01$", xlabel=r'Num. Nodes',grid=True,return_conv=True,
                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
                labels=label, ylim=ylim, ignore_fail=True)
"""
#solver = solver_c(diffeq,settings,tm_method,dt,tf,q0,p,disc_type,surf_type,vol_diss,had_flux,nelem,nen,disc_nodes,bc,xmin,xmax,cons_obj_name,bool_plot_sol,print_sol_norm)
dofs3, errors3, labels3 = run_convergence(solver,schedule_in=schedule3,savefile='CSBP_ND.png',
                title=r"CSBP No (Volume) Dissipation", xlabel=r'Num. Nodes',grid=True,return_conv=True,
                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
                labels=label3, ylim=ylim)

#solver = solver_c(diffeq,settings,tm_method,dt,tf,q0,p,disc_type,surf_type,vol_diss,had_flux,nelem,nen,disc_nodes,bc,xmin,xmax,cons_obj_name,bool_plot_sol,print_sol_norm)
dofs4, errors4, labels4 = run_convergence(solver,schedule_in=schedule4,savefile='upwind_lf.png',
                title=r"Mattsson Upwind Ops", xlabel=r'Num. Nodes',grid=True,return_conv=True,
                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
                labels=label4, ylim=ylim)

#solver = solver_c(diffeq,settings,tm_method,dt,tf,q0,p,disc_type,surf_type,vol_diss,had_flux,nelem,nen,disc_nodes,bc,xmin,xmax,cons_obj_name,bool_plot_sol,print_sol_norm)
dofs5, errors5, labels5 = run_convergence(solver,schedule_in=schedule5,savefile='csbp_upwind_lf.png',
                title=r"CSBP + Mattsson Upwind Dissipation, $\epsilon = 1$", xlabel=r'Num. Nodes',grid=True,return_conv=True,
                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
                labels=label5, ylim=ylim)

#solver.check_eigs(title=r'Eigenvalues: CSBP No Dissipation (2 elem, 20 nodes, $p=3$)',
#                  savefile=None, colour_by_k=True)
"""