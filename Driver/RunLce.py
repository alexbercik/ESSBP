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
dt = 0.001 # for convergence studies, try to choose at least C=0.02
# note: should set according to courant number C = a dt / dx
tf = 2

# Domain
xmin = -1.
xmax = 1.
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd', 'upwind'
p = 1

nelem = 1 # optional, number of elements
nen = 10 # optional, number of nodes per element
surf_type = 'lf'

had_flux = 'central' # 2-point numerical flux used in hadamard form. 
vol_diss = {'diss_type':'nd', 'use_H':False, 'bdy_fix':False, 
            'jac_type':'scalar', 's':p+1, 'coeff':3.125/5**(p+1), 'beta':4, 
            'fluxvec':'lf', 'eps_type':2, 'D_type':'sbp'}
# Initial solution
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

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
                  bool_plot_sol, print_sol_norm,
                  print_progress=False)
solver.tm_nframes = 100
solver.tm_atol = 1e-8
solver.tm_rtol = 1e-8

''' Analyze results '''
#solver.mesh.plot()  # Plot mesh

#solver.skip_ts = 999
#solver.tm_nframes = 100
solver.check_eigs(normalize=True,ymin=-1.8,ymax=1.8,xmin=-3.0,xmax=0.6,figsize=(5,5))
solver.plot_eigvecs(test_type='min real',plot_type='real',num_eigvecs=5)
#solver.solve()
#solver.plot_sol()
#solver.plot_sol()#q=solver.diffeq.set_q0(),plot_exa=False)
#solver.plot_cons_obj()

#from Source.Methods.Analysis import run_convergence
#schedule = [['nelem',8,16,32,64,128,256],['disc_nodes','lgl','lg','lg_exp','glaubitz_exp'],['p',3],['nen',0]]
#labels = ['LGL', 'LG', 'LG-Exp', 'Glaubitz-Exp']
#dofs, errors, labels = run_convergence(solver,schedule_in=schedule,plot=True,return_conv=True, figsize=(5,4), labels=labels)

# from Source.Methods.Analysis import run_convergence, plot_conv, plot_eigs
# schedule = [['nelem',8,16,32,64,128],['p',4],
#             ['vol_diss',{'diss_type':'nd'},
#                         {'diss_type':'zelalem', 'jac_type':'scalar', 's':'p', 'eps_type':0, 'coeff':0.05},
#                         {'diss_type':'zelalem', 'jac_type':'scalar', 's':'p', 'eps_type':1, 'coeff':5.0},
#                         {'diss_type':'zelalem', 'jac_type':'scalar', 's':1, 'eps_type':1, 'coeff':1000.0},
#                         {'diss_type':'zelalem', 'jac_type':'scalar', 's':1, 'eps_type':3, 'coeff':1000.0}]]
# label = [r'$\varepsilon = 0$',
#          r'$s=p$, $\varepsilon = 0.004$',
#          r'$s=p$, $\varepsilon = 0.1 h$',
#          r'$s=1$, $\varepsilon = h^p$',
#          r'$s=1$, $\varepsilon_k = \sqrt{\frac{\sum_\gamma \left( \tkgT \vec{u}_k - \tgkT \vec{u}_\gamma  \right)^2}{\sum_\gamma \left( \tkgT \vec{u}_k + \tgkT \vec{u}_\gamma \right)^2}}$']
# preamble = [r'\usepackage{amsmath}',
#             r'\newcommand{\norm}[1]{\left\lVert#1\right\rVert}',
#             r'\newcommand{\T}[0]{\mathrm{T}}',
#             r'\newcommand{\bsym}[1]{\ensuremath{\boldsymbol{#1}}}',
#             r'\newcommand{\fnc}[1]{\ensuremath{\mathcal{#1}}}',
#             r'\newcommand{\vecfnc}[1]{\ensuremath{\bsym{\mathcal{#1}}}}',
#             r'\renewcommand{\vec}[1]{\ensuremath{\bsym{#1}}}',
#             r'\newcommand{\mat}[1]{\ensuremath{\mathsf{#1}}}',
#             r'\newcommand{\U}[0]{\ensuremath{\fnc{U}}}',
#             r'\newcommand{\V}[0]{\ensuremath{\fnc{V}}}',
#             r'\newcommand{\zeros}[0]{\ensuremath{\vec{0}}}',
#             r'\newcommand{\ones}[0]{\ensuremath{\vec{1}}}',
#             r'\newcommand{\bu}[0]{\ensuremath{\vec{u}}}',
#             r'\newcommand{\tl}[0]{\vec{t}_{\mathrm{L}}}',
#             r'\newcommand{\tr}[0]{\vec{t}_{\mathrm{R}}}',
#             r'\newcommand{\tlT}[0]{\vec{t}_{\mathrm{L}}^\T}',
#             r'\newcommand{\trT}[0]{\vec{t}_{\mathrm{R}}^\T}',
#             r'\newcommand{\tkgT}[0]{\vec{t}_{k \gamma}^\T}',
#             r'\newcommand{\tgkT}[0]{\vec{t}_{\gamma k}^\T}',
#             r'\newcommand{\Idty}[0]{\mat{I}}',
#             r'\newcommand{\Hnrm}[0]{\mat{H}}',
#             r'\newcommand{\HI}[0]{\mat{H}^{-1}}']

# dofs, errors, labels = run_convergence(solver,schedule_in=schedule,plot=False,return_conv=True)
# savefile = None #'zelalemdiss_LGLp4.png'
# colors = ['tab:blue', 'm', 'tab:red', 'darkgoldenrod', 'tab:orange', 'k', 'tab:brown']
# markers = ['s', '^', 'o', 'v', 'x', '+', 'v']
# plot_conv(dofs, errors, label, 1, title=None, savefile=savefile, showslope=True,
#         extra_marker=None, skipfit=None, skip=None, title_size=16,
#         ylabel=r'$\norm{\bu - \U}_\mat{H}$', xlabel=r'Degrees of Freedom', 
#         ylim=(None,None),xlim=(None,None),grid=True,legendloc='lower left',convunc=False,
#         figsize=(5,4), tick_size=12, extra_xticks=True, scalar_xlabel=True, serif=False,
#         colors=colors, markers=markers, linestyles=None, legendsize=11.5, legendreorder=None,
#         remove_outliers=False, legend_anchor=None, put_legend_behind=True, preamble=preamble)

# diss_runs = [x[1:] for x in schedule if x[0]=='vol_diss'][0]
# p_runs = [x[1:] for x in schedule if x[0]=='p'][0]
# for elem in [8,16,32]:
#     for p in p_runs:
#         As = []
#         for diss in diss_runs:
#             solver = PdeSolverSbp(diffeq, settings,tm_method, dt, tf,
#                         p,surf_diss=surf_type, vol_diss=diss,
#                         nelem=elem, nen=nen, disc_nodes=disc_nodes,
#                         bc='periodic', xmin=xmin, xmax=xmax)
#             A = solver.calc_LHS()
#             A /= solver.nelem*(solver.nen-1)/(xmax-xmin)
#             As.append(A)

#         xlabel = r'$\Re(\lambda) h$'
#         ylabel = r'$\Im(\lambda) h$'
#         xlim=None #(-3.5,0.2) # normally use (-4.1,0.2) for large, (-3.5,0.2) for small, (-2.3,1.9) for main body
#         ylim=None # normally use (-1.5,1.5) for large, (-0.8,0.8) for small, (-1.2,1.2) for main body
#         plot_eigs(As,plot_hull=False,plot_individual_eigs=True,labels=label,savefile=savefile,
#                 line_width=2,equal_axes=True,title_size=16,legend_size=14,markersize=50, 
#                 markeredge=1.4, tick_size=12, colors=colors, linestyles=None, markers=markers,
#                 legend_loc='upper left', #legend_anchor=(0.0, 0.88), legend_anchor_type=('data','fig'),
#                 legend_alpha=0.9, left_space_pct=None, xlabel=xlabel, ylabel=ylabel,
#                 xlim=xlim, ylim=ylim,save_format=None, title=f'{elem} Elements', no_legend=True, figsize=(5,5))