import os
from sys import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tik
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
plt.rcParams['font.family'] = 'serif'

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import run_convergence, plot_conv

''' Set parameters for simultation 
'''
savefile = None #'LCEsolerr_CSBPp34e24n.png' # None for no save
a = 1.0 # wave speed 
cfl = 0.1
tf = 1.0 # final time
nelem = 4 # number of elements
nen = 24 # number of nodes per element, as a list
op = 'csbp' # operator type
p = 3 # polynomial degree
s = p+1 # dissipation degree
q0_type = 'GaussWave_sbpbook' # initial condition 
settings = {} # additional settings for mesh type, etc. Not needed.

# set up the differential equation
diffeq = LinearConv(a, q0_type)

# set different runs
run1 = {'diss':{'diss_type':'nd'},
        'sat':{'diss_type':'nd'},
        'label':'E.C.'}
run2 = {'diss':{'diss_type':'nd'},
        'sat':{'diss_type':'lf'},
        'label':r'$\sigma = 0$'}
run3 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125*5**(-s)},
        'sat':{'diss_type':'lf'},
        'label':f'$\\sigma = {3.125*5**(-s)}$'}
run4 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125*5**(-s)},
        'sat':{'diss_type':'lf'},
        'label':f'$\\sigma = {0.2*3.125*5**(-s)}$'}

# prepare the plot
title = None
xlabel = r'$x$'
ylabel = r'Solution Error $\bm{u} - \bm{u}_{\mathrm{ex}}$'
linear_thresh = 1e-5
colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red', 'tab:brown']
linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (1, 2, 3, 2))]

# initialize the runs and solve
if nen == 0: nen = p+1 # for LG/LGL
dx = 1./((nen-1)*nelem)
dt = cfl * dx / a
solver1 = PdeSolverSbp(diffeq, settings, 'rk4', dt, tf, p=p, surf_diss=run1['sat'], vol_diss=run1['diss'], nelem=nelem, nen=nen, disc_nodes=op, bc='periodic')
solver2 = PdeSolverSbp(diffeq, settings, 'rk4', dt, tf, p=p, surf_diss=run2['sat'], vol_diss=run2['diss'], nelem=nelem, nen=nen, disc_nodes=op, bc='periodic')
solver3 = PdeSolverSbp(diffeq, settings, 'rk4', dt, tf, p=p, surf_diss=run3['sat'], vol_diss=run3['diss'], nelem=nelem, nen=nen, disc_nodes=op, bc='periodic')
solver4 = PdeSolverSbp(diffeq, settings, 'rk4', dt, tf, p=p, surf_diss=run4['sat'], vol_diss=run4['diss'], nelem=nelem, nen=nen, disc_nodes=op, bc='periodic')
solver1.skip_ts, solver2.skip_ts, solver3.skip_ts, solver4.skip_ts = 100, 100, 100, 100 # don't save info on every iteration - unecessary
solver1.solve()
solver2.solve()
solver3.solve()
solver4.solve()
er1 = solver1.q_sol[:,:,-1] - solver1.diffeq.exact_sol(time=tf)
er2 = solver2.q_sol[:,:,-1] - solver2.diffeq.exact_sol(time=tf)
er3 = solver3.q_sol[:,:,-1] - solver3.diffeq.exact_sol(time=tf)
er4 = solver4.q_sol[:,:,-1] - solver4.diffeq.exact_sol(time=tf)
x = diffeq.x

# plot results
plt.figure(figsize=(6,4))
if title is not None: plt.title(title,fontsize=18)
plt.ylabel(ylabel,fontsize=16)
plt.xlabel(xlabel,fontsize=16)
plt.yscale('symlog',linthresh=linear_thresh)
plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')

plt.plot(x, er1.flatten('F'), color=colors[2], linestyle=linestyles[3], label=run1['label'], linewidth=2) 
plt.plot(x, er2.flatten('F'), color=colors[1], linestyle=linestyles[1], label=run2['label'], linewidth=2) 
plt.plot(x, er3.flatten('F'), color=colors[3], linestyle=linestyles[0], label=run3['label'], linewidth=2) 
plt.plot(x, er4.flatten('F'), color=colors[4], linestyle=linestyles[2], label=run4['label'], linewidth=2) 

plt.legend(loc='upper center',fontsize=12,  bbox_to_anchor=(0.5, 1.13), fancybox=True, shadow=False, ncol=4, columnspacing=1.5)
plt.tight_layout()
if savefile is not None: plt.savefig(savefile, dpi=600)