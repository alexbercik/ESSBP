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

# Time marching
tm_method = 'rk8' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
tf = 2 # final time / breaking time

# Domain
xmin = 0.
xmax = 1
bc = 'periodic' 

# Spatial discretization
disc_type = 'div' # 'div', 'had' (divergence or hadamard-product)
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp'
p = 4
nelem = 1 # optional, number of elements
nen = 200 # optional, number of nodes per element (set to zero for element-type)
had_flux = 'ec' # 2-point numerical flux used in hadamard form (only 'ec' and 'central' set up)
surf_diss = {'diss_type':'ent', 'jac_type':'scasca', 'maxeig':'rusanov', 'coeff':1.0}
vol_diss = {'diss_type':'nd', 'use_H':False, 'jac_type':'scalar', 'fluxvec':'burgers',
            'avg_half_nodes':True, 's':1, 'eps_type':4, 'coeff':1}#1.0}
use_split_form = True
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative had form

# Initial solution
q0_type = 'sinwave_shift' # 'GassnerSinWave', '..._cont', '..._coarse' 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False
skip_ts = 0

cons_obj_name = ('time', 'Energy', 'Conservation', 'Max_Eig')
settings = {'warp_factor':0.0,               # Warps / stretches mesh.
            'warp_type': 'none',             # Options: 'defualt', 'papers', 'quad'
            'use_optz_metrics':True,         # Uses optimized metrics for free stream preservation.
            'use_exact_metrics':True}        # Uses exact metrics instead of interpolation.}


''' Set diffeq and solve '''
diffeq = Burgers(None, q0_type, use_split_form, split_alpha)
solver = PdeSolverSbp(diffeq, settings,                     # Diffeq
                  tm_method, dt, None,                    # Time marching
                  p, disc_type,             # Discretization
                  surf_diss, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm,sparse=True)
solver.skip_ts = skip_ts
solver.t_final = diffeq.calc_breaking_time(print_res=False) * tf
solver.set_timestep()
solver.tm_rtol = 1e-13
solver.tm_atol = 1e-13
solver.tm_nframes = tf*80
solver.print_progress = False

#diffeq.calc_breaking_time()
#solver.check_eigs()
solver.solve()
solver.plot_sol()
#solver.plot_cons_obj()
import numpy as np
#solver.check_eigs(q=np.random.rand(*solver.qshape))
#solver.check_eigs(q=np.random.rand(*solver.qshape))
#solver.check_eigs(q=np.random.rand(*solver.qshape))
#solver.check_eigs() #q=diffeq.exact_sol(time=0.))

# for i in range(solver.q_sol.shape[2]):
#     solver.q_sol[:,:,i] = diffeq.exact_sol(time=solver.cons_obj[0,i])

# from Source.Methods.Analysis import animate
# diffeq.plt_style_sol = [{'color':'k','linestyle':'-','linewidth':2,'marker':''}]
# animate(solver, file_name='burgers', make_video=True, make_gif=False,
#             plotfunc='plot_sol',
#             plotargs={'display_time':True,
#                       'figsize':(6,4),
#                       'time_round':2,
#                         'ymin':0.4, 
#                         'ymax':2.6,
#                         'xmin':0.0,
#                         'xmax':1.0,
#                         'plot_exa':False,
#                         'legend':False,
#                         'normalize':True}, 
#             skipsteps=0,fps=30,last_frame=True,time=solver.cons_obj[0,:])

# from Source.Methods.Analysis import run_convergence
# dofs, errors, legend_strings = run_convergence(solver,
#                 schedule_in=[['nen',100,200,300],
#                              ['p',1,2,3,4],
#                              ['surf_diss',{'diss_type':'ec'}],
#                              ['vol_diss',{'diss_type':'nd'}]],#,
#                                         #{'diss_type':'dcp', 'use_H':False, 'use_B':True, 'jac_type':'scalar',
#                                         #  'avg_half_nodes':True, 's':'p+1', 'coeff':3.125/5**(5)}]],
#                 labels=[r'$p=1$', r'$p=2$',r'$p=3$', r'$p=4$'],#[r'CSBP ES $p=3$', r'CSBP ES + diss $p=3$',r'CSBP ES $p=4$', r'CSBP ES + diss $p=4$'],
#                 return_conv=True)

# if use_split_form:
#     eigs_split = solver.check_eigs(q=diffeq.set_q0(),returneigs=True)
# else:
#     eigs_div = solver.check_eigs(q=diffeq.set_q0(),returneigs=True)
# import matplotlib.pyplot as plt
# import os.path as ospath
# exmin, exmax = -1.4, 0.4
# eymin, eymax = 6000, -6000
# plt.figure(figsize=(5,4))
# X_split = [x.real for x in eigs_split]
# Y_split = [x.imag for x in eigs_split]
# X_div = [x.real for x in eigs_div]
# Y_div = [x.imag for x in eigs_div]
# plt.scatter(X_split,Y_split, color='tab:red',label='Entropy-stable')
# plt.scatter(X_div,Y_div, color='tab:blue',label='Central')
# plt.axvline(x=0, linewidth=1, linestyle='--', color='black')
# plt.xlabel(r'$\Re{(\lambda)}$',fontsize=14)
# plt.ylabel(r'$\Im{(\lambda)}$',fontsize=14)
# plt.title(fr'Eigenvalues of $\mathcal{{L}}$, {nen} nodes',fontsize=16)
# plt.ylim(eymin,eymax)
# plt.xlim(exmin,exmax)
# plt.legend(loc='lower center',fontsize=14)
# filename = 'eigvals'+f'{nen}n'+'.pdf'
# #if ospath.exists(filename):
# #    print('WARNING: File name already exists. Using a temporary name instead.')
# #    plt.savefig(filename+'_RENAMEME', format='png')
# plt.tight_layout()
# plt.savefig(filename, format='pdf')