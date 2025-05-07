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
savefile = None # use a string like 'sol.png' or 'sol.pdf' to save the plot, None for no save
a = 1.0 # wave speed 
cfl = 0.01
tf = 1 # final time
nelem = 2 # number of elements
nen = 0 # number of nodes per element, as a list
op = 'lgl' # operator type
p = 8 # polynomial degree
linear_thresh = 1e-7 #1e-8 for gauss, 1e-7 for sin, 1e-5 for LGLp4, 1e-7 fpr LGLp8
max_thresh = 1e-4 #9e-3 for csbp gauss, 9e-4 for csbp sin, 3e-2 for LGLp4, 1e-4 for LGLp8
q0_type = 'sinwave_2pi' #'sinwave_4pi' #'GaussWave_sbpbook' 'sinwave_2pi' #'squarewave' # initial condition 
settings = {} # additional settings for mesh type, etc. Not needed.
plot_abs_error = False
zoom = 0 # how many nodes to zoom in on? Counts the left-most node to start in the frame
interp_num = 200 # for LG/LGL, number of nodes to interpolate to per element 

if op  in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    s = p + 1
    eps = 3.125/5**s
    useH = False
    bdy_fix = True
elif op in ['lg', 'lgl']:
    s = p
    if p == 2: eps = 0.02
    elif p == 3: eps = 0.01
    elif p == 4: eps = 0.004
    elif p == 5: eps = 0.002
    elif p == 6: eps = 0.0008
    elif p == 7: eps = 0.0004
    elif p == 8: eps = 0.0002
    else: raise Exception('No dissipation for this p')
    useH = False
    bdy_fix = False
else:
    raise Exception('No dissipation for this operator')

if zoom != 0:
    assert nelem == 2, 'Zoom only works for 2 elements so that you zoom in on the middle'

# set different runs
if op in ['lg', 'lgl']:
    #nelem_pm1 = int(nelem*(p+1)/p)
    #assert(nelem_pm1*p == nelem*(p+1)), 'choose nelem such that nelem*(p+1) is divisible by p'
    nelem_pm1 = nelem
    run1 = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'lf'},
            'label':r'$p={0}$, $\varepsilon = 0$'.format(int(p-1)),
            'p':p-1,'nelem':nelem_pm1,'nen':0}
    run2 = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'lf'},
            'label':r'$p={0}$, $\varepsilon = 0$'.format(int(p)),
            'p':p,'nelem':nelem,'nen':0}
    run3 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps},
            'sat':{'diss_type':'lf'},
            'label':f'$p={p}$, $\\varepsilon = {eps:.3g}$',
            'p':p,'nelem':nelem,'nen':0}
    run4 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':0.2*eps},
            'sat':{'diss_type':'lf'},
            'label':f'$p={p}$, $\\varepsilon = {0.2*eps:.3g}$',
            'p':p,'nelem':nelem,'nen':0}
else:
    nelem_pm1 = 0
    run1 = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'nd'},
            'label':r'Symmetric SAT $\varepsilon = 0$',
            'p':p,'nelem':nelem,'nen':nen}
    run2 = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'lf'},
            'label':r'Upwind SAT $\varepsilon = 0$',
            'p':p,'nelem':nelem,'nen':nen}
    run3 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps},
            'sat':{'diss_type':'lf'},
            'label':f'Upwind SAT $\\varepsilon = {eps:.3g}$',
            'p':p,'nelem':nelem,'nen':nen}
    run4 = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':0.2*eps},
            'sat':{'diss_type':'lf'},
            'label':f'Upwind SAT $\\varepsilon = {0.2*eps:.3g}$',
            'p':p,'nelem':nelem,'nen':nen}


# prepare the plot
title = None
xlabel = r'$x$'
ylabel = r'Solution Error $\bm{u} - \bm{u}_{\mathrm{ex}}$'
#colors = ['tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']
colors = ['tab:green', 'tab:orange', 'k',  'm', 'tab:brown']
if op in ['lg', 'lgl']: 
    linestyles = ['-','-','-','-']
    #linestyles = ['-','-','--',':']
    linewidths = [2.5,2.2,2.0,1.5]
else:
    linestyles = [(0, (1, 1.5)), (0, (2, 3)), '-',(0, (4, 3, 1, 3))]
    linewidths = [2.3,2,1.8,1.8]

# initialize the runs and solve
if nen == 0: 
    nen_tmp = p+1 # for LG/LGL
else:
    nen_tmp = nen
dx = 1./((nen_tmp-1)*nelem)
dt = cfl * dx / a
diffeq1 = LinearConv(a, q0_type)
diffeq1.q0_max_q = 1.
solver1 = PdeSolverSbp(diffeq1, settings, 'rk8', dt, tf, p=run1['p'], surf_diss=run1['sat'], vol_diss=run1['diss'], nelem=run1['nelem'], nen=run1['nen'], disc_nodes=op, bc='periodic')
diffeq2 = LinearConv(a, q0_type)
diffeq2.q0_max_q = 1.
solver2 = PdeSolverSbp(diffeq2, settings, 'rk8', dt, tf, p=run2['p'], surf_diss=run2['sat'], vol_diss=run2['diss'], nelem=run2['nelem'], nen=run2['nen'], disc_nodes=op, bc='periodic')
diffeq3 = LinearConv(a, q0_type)
diffeq3.q0_max_q = 1.
solver3 = PdeSolverSbp(diffeq3, settings, 'rk8', dt, tf, p=run3['p'], surf_diss=run3['sat'], vol_diss=run3['diss'], nelem=run3['nelem'], nen=run3['nen'], disc_nodes=op, bc='periodic')
diffeq4 = LinearConv(a, q0_type)
diffeq4.q0_max_q = 1.
solver4 = PdeSolverSbp(diffeq4, settings, 'rk8', dt, tf, p=run4['p'], surf_diss=run4['sat'], vol_diss=run4['diss'], nelem=run4['nelem'], nen=run4['nen'], disc_nodes=op, bc='periodic')
solver1.keep_all_ts, solver2.keep_all_ts, solver3.keep_all_ts, solver4.keep_all_ts = False, False, False, False # don't save info on every iteration - unecessary
solver1.solve()
solver2.solve()
solver3.solve()
solver4.solve()
if op in ['lg', 'lgl']:
    if interp_num == 0: interp_num = 20
    tmp, x1 = solver1.interpolate(return_mesh=True,num_nodes=interp_num)
    exa = solver1.diffeq.exact_sol(x=x1, time=tf)
    er1 = tmp - exa
    tmp, x2 = solver2.interpolate(return_mesh=True,num_nodes=interp_num)
    exa = solver2.diffeq.exact_sol(x=x2, time=tf)
    er2 = tmp - exa
    tmp, x3 = solver3.interpolate(return_mesh=True,num_nodes=interp_num)
    exa = solver3.diffeq.exact_sol(x=x3, time=tf)
    er3 = tmp - exa
    tmp, x4 = solver4.interpolate(return_mesh=True,num_nodes=interp_num)
    exa = solver4.diffeq.exact_sol(x=x4, time=tf)
    er4 = tmp - exa
    end = interp_num - zoom
else:
    er1 = solver1.q_sol - solver1.diffeq.exact_sol(time=tf)
    er2 = solver2.q_sol - solver2.diffeq.exact_sol(time=tf)
    er3 = solver3.q_sol - solver3.diffeq.exact_sol(time=tf)
    er4 = solver4.q_sol - solver4.diffeq.exact_sol(time=tf)
    x1 = solver1.diffeq.x_elem
    x2 = solver2.diffeq.x_elem
    x3 = solver3.diffeq.x_elem
    x4 = solver4.diffeq.x_elem
    end = nen - zoom

# plot results
plt.figure(figsize=(6,4.4))
if title is not None: plt.title(title,fontsize=18)
plt.ylabel(ylabel,fontsize=16)
plt.xlabel(xlabel,fontsize=16)
plt.yscale('symlog',linthresh=linear_thresh)
plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
plt.gca().tick_params(axis='both', labelsize=12) 
plt.ylim(-max_thresh,max_thresh)

plt.plot(x1[zoom:,0], er1[zoom:,0], color=colors[0], linestyle=linestyles[0], label=run1['label'], linewidth=linewidths[0]) 
plt.plot(x2[zoom:,0], er2[zoom:,0], color=colors[1], linestyle=linestyles[1], label=run2['label'], linewidth=linewidths[1]) 
plt.plot(x3[zoom:,0], er3[zoom:,0], color=colors[2], linestyle=linestyles[2], label=run3['label'], linewidth=linewidths[2]) 
plt.plot(x4[zoom:,0], er4[zoom:,0], color=colors[3], linestyle=linestyles[3], label=run4['label'], linewidth=linewidths[3]) 
for elem in range(1,nelem):
    plt.plot(x1[:end,elem], er1[:end,elem], color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0]) 
    plt.plot(x2[:end,elem], er2[:end,elem], color=colors[1], linestyle=linestyles[1], linewidth=linewidths[1]) 
    plt.plot(x3[:end,elem], er3[:end,elem], color=colors[2], linestyle=linestyles[2], linewidth=linewidths[2]) 
    plt.plot(x4[:end,elem], er4[:end,elem], color=colors[3], linestyle=linestyles[3], linewidth=linewidths[3]) 
for elem in range(nelem,nelem_pm1):
    plt.plot(x1[:end,elem], er1[:end,elem], color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0]) 

handles, labels = plt.gca().get_legend_handles_labels()
if op in ['lg', 'lgl']: 
    order = [0,1,2,3]
    anchor = (0.5, 1.248)
else:
    order = [0,3,1,2]
    anchor = (0.435, 1.248)
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper center',fontsize=14,  
           bbox_to_anchor=anchor, fancybox=True, shadow=False, ncol=2, columnspacing=1.5)
# bbox_to_anchor=(0.435, 1.248), bbox_to_anchor=(0.445, 1.248)
plt.tight_layout()
if savefile is not None: plt.savefig(savefile, dpi=600)

if plot_abs_error:
    plt.figure(figsize=(6,4.4))
    if title is not None: plt.title(title,fontsize=18)
    plt.ylabel(r'Solution Error $\left\lvert \bm{u} - \bm{u}_{\mathrm{ex}} \right\rvert$',fontsize=16)
    plt.xlabel(xlabel,fontsize=16)
    plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
    plt.gca().tick_params(axis='both', labelsize=12) 

    plt.semilogy(x1[zoom:,0], np.abs(er1[zoom:,0]), color=colors[0], linestyle=linestyles[0], label=run1['label'], linewidth=linewidths[0]) 
    plt.semilogy(x2[zoom:,0], np.abs(er2[zoom:,0]), color=colors[1], linestyle=linestyles[1], label=run2['label'], linewidth=linewidths[1]) 
    plt.semilogy(x3[zoom:,0], np.abs(er3[zoom:,0]), color=colors[2], linestyle=linestyles[2], label=run3['label'], linewidth=linewidths[2]) 
    plt.semilogy(x4[zoom:,0], np.abs(er4[zoom:,0]), color=colors[3], linestyle=linestyles[3], label=run4['label'], linewidth=linewidths[3]) 
    for elem in range(1,nelem):
        plt.semilogy(x1[:end,elem], np.abs(er1[:end,elem]), color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0]) 
        plt.semilogy(x2[:end,elem], np.abs(er2[:end,elem]), color=colors[1], linestyle=linestyles[1], linewidth=linewidths[1]) 
        plt.semilogy(x3[:end,elem], np.abs(er3[:end,elem]), color=colors[2], linestyle=linestyles[2], linewidth=linewidths[2]) 
        plt.semilogy(x4[:end,elem], np.abs(er4[:end,elem]), color=colors[3], linestyle=linestyles[3], linewidth=linewidths[3]) 
    for elem in range(nelem,nelem_pm1):
        plt.semilogy(x1[:end,elem], np.abs(er1[:end,elem]), color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0])


    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper center',fontsize=14,  
               bbox_to_anchor=anchor, fancybox=True, shadow=False, ncol=2, columnspacing=1.5)
    plt.tight_layout()
    if savefile is not None: plt.savefig('abs_' + savefile, dpi=600)