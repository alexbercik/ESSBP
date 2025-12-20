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
from Source.Solvers.PdeSolver import PdeSolver


'''
Solve the 2D Euler equations
'''

# Eq parameters
para = [287,1.4] # [R, gamma]
test_case = 'kelvin-helmholtz' # density_wave, vortex, kelvin-helmholtz
nondimensionalize = False

# Time marching
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.002
# for vortex, maxeig < 1.6, so CFL = 0.1 gives dt = 0.16 * dx and CFL = 0.01 gives dt = 0.016 * dx
tf = 10 #nts * dt # set to None to do automatically or use a convergence criterion, or 'steady'
# for vortex, one period is t=20
check_resid_conv = False

# Domain
xmin = (-1.,-1.)
xmax = (1.,1.)
bc = 'periodic' # 'periodic', 'dirichlet'

# Spatial discretization
disc_type = 'had' # 'div', 'had'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 4
nelem = (1,1) # number of elements
nen = 200 # optional, number of nodes per element
surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'Roe', 
             'entropy_fix':True, 'P_derigs':True, 'A_derigs':True, 'maxeig':'rusanov'}
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vol_diss = {'diss_type':'entdcp', 'jac_type':'matmat', 's':p+1, 'coeff':0.2*3.125/5**(p+1),
            'fluxvec':'dt', 'bdy_fix':True, 'use_H':False, 'entropy_fix':True, 'avg_half_nodes':True, 'eps_type':4}

# Initial solution
q0_type = 'exact'

# Other
bool_plot_sol = False
print_sol_norm = False
print_residual = False
cons_obj_name=('time','energy','entropy') #('Energy','Entropy','Conservation','error','time') # note: should I modify this for systems?
settings = {'metric_method':'exact',
            'use_optz_metrics':False} # extra things like for metrics
skip_ts = 99


''' Set diffeq and solve '''

diffeq = Euler(para, q0_type, test_case, bc, nondimensionalize)

diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':2}
#diffeq.use_alternative_dEndw_abs()

solver = PdeSolverSbp(diffeq, settings, 
                  tm_method, dt, tf,                 
                  p, disc_type,      
                  surf_diss, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,     
                  cons_obj_name,      
                  bool_plot_sol, print_sol_norm,
                  print_residual, check_resid_conv,
                  sparse=True, sat_sparse=True)

solver.tm_nframes = int(tf * 10)
solver.tm_rtol = 1e-10 #1e-13
solver.tm_atol = 1e-10 #1e-13
solver.check_stability()
#solver.solve()
#solver.plot_cons_obj()

#eigs = solver.check_eigs(returneigs=True)
#solver.plot_sol()

#import numpy as np
#q = solver.diffeq.set_q0() + 0.01*np.random.rand(*solver.qshape)
#solver.check_conservation(q)
#solver.check_stability(q)

#eigs = solver.check_eigs(returneigs=True)
#solver.plot_sol()

#calc_eigs = True

# import numpy as np
# import matplotlib.pyplot as plt
# if disc_type == 'had':
#     if calc_eigs:
#         eigs_split = solver.check_eigs(q=solver.q_sol[:,:,2], returneigs=True)
#         lam_split = np.max(eigs_split.real) 
#         eigs_split = solver.check_eigs(q=solver.q_sol[:,:,2], returneigs=True)
#         lam_split = np.max(eigs_split.real)
#     max_er_split = solver.cons_obj[2,:]
#     er_split = solver.cons_obj[1,:]
#     t_split = solver.cons_obj[0,:]
# else:
#     if calc_eigs: 
#         eigs_div = solver.check_eigs(q=solver.q_sol[:,:,2], returneigs=True)
#         lam_div = np.max(eigs_div.real)
#     max_er_div = solver.cons_obj[2,:]
#     er_div = solver.cons_obj[1,:]
#     t_div = solver.cons_obj[0,:]


# plt.figure(figsize=(5,4))
# if calc_eigs: plt.semilogy(t_split,1e-14*np.exp(lam_split*t_split),label=r'Entropy-stable $ \varepsilon e^{\lambda t}$',linestyle='--',color='tab:orange',linewidth=2.5)
# plt.semilogy(t_split,max_er_split,label=r'Entropy-stable $\Vert \text{er}(t) \Vert_\infty $',color='tab:blue')
# if calc_eigs: plt.semilogy(t_div,1e-14*np.exp(lam_div*t_div),label=r'Central $ \varepsilon e^{\lambda t}$',linestyle='--',color='tab:green',linewidth=2.5)
# plt.semilogy(t_div,max_er_div,label=r'Central $\Vert \text{er}(t) \Vert_\infty $',color='tab:red')
# plt.grid(True, axis='y')
# plt.title(f'Solution Error (Perturbation Growth), {nen} nodes',fontsize=16)
# plt.xlabel(r'$t$',fontsize=14)
# plt.ylim(1e-12,1e2)
# filename = 'Euler_er'+f'{nen}n'+'.pdf'
# plt.tight_layout()
# plt.legend(fontsize=14, loc='upper left')
# plt.savefig(filename, format='pdf')

# if calc_eigs:
#     exmin, exmax = -2*1.7, 2*1.7
#     eymin, eymax = 2*170, -2*170
#     plt.figure(figsize=(5,4))
#     X_split = [x.real for x in eigs_split]
#     Y_split = [x.imag for x in eigs_split]
#     X_div = [x.real for x in eigs_div]
#     Y_div = [x.imag for x in eigs_div]
#     plt.scatter(X_split,Y_split, color='tab:blue',label='Entropy-stable', marker='o')
#     plt.scatter(X_div,Y_div, color='tab:red',label='Central', marker='x')
#     plt.axvline(x=0, linewidth=1, linestyle='--', color='black')
#     plt.xlabel(r'$\Re{(\lambda)}$',fontsize=14)
#     plt.ylabel(r'$\Im{(\lambda)}$',fontsize=14)
#     plt.title(fr'Eigenvalues of $\mathsf{{D}}_{{\mathcal{{U}}}} \mathcal{{L}}$, {nen} nodes',fontsize=16)
#     plt.ylim(eymin,eymax)
#     plt.xlim(exmin,exmax)
#     plt.legend(loc='lower left',fontsize=14)
#     filename = 'Euler_eigvals'+f'{nen}n'+'.pdf'
#     #if ospath.exists(filename):
#     #    print('WARNING: File name already exists. Using a temporary name instead.')
#     #    plt.savefig(filename+'_RENAMEME', format='png')
#     plt.tight_layout()
#     plt.savefig(filename, format='pdf')