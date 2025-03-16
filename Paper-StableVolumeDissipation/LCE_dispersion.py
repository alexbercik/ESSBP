import os
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
import numpy as np
import matplotlib.pyplot as plt

# NOTE: This was not used in the paper. But have fun with it if you want.

''' Set parameters for simultation '''
savefile = None # None for no save
nen = 40 # number of nodes per element, as a list (multiple if traditional-refinement)
ops = ['upwind','csbp','hgtl','hgt','mattsson']#,'lgl','lg']
p = 4 # polynomial degree
include_nodiss = True # include runs without dissipation?
bdy_fix = True # include B? 
useH = False # include H? 
num_k = 500 # number of k-values (x-axis) for dispersion analysis
surf_diss = {'diss_type':'lf'} # SAT dissipation
sig_fix = 0.2

# these settings don't matter
colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red', 'tab:brown']
dt = 1e-5
tf = 1. # final time
nelem = 1 # number of elements, as a list (multiple if element-refinement)
q0_type = 'GaussWave_sbpbook' # initial condition 
a = 1.0 # wave speed 
diffeq = LinearConv(a, q0_type)
settings = {} # additional settings for mesh type, etc. Not needed.
tm_method = 'rk4'


Ks = np.linspace(0, np.pi, num_k, endpoint=True)
Omegas = []
Labels = []
Linestyles = []
Colors = []
for i,op in enumerate(ops):
    # set some default values
    if op == 'csbp':
        s = p+1 # dissipation degree
        eps = sig_fix*3.125/5**s
        label = 'CSBP'
        op_nen = nen
        op_p = p
    elif op == 'hgtl':
        s = p+1 # dissipation degree
        eps = sig_fix*3.125/5**s
        label = 'HGTL'
        op_nen = nen
        op_p = p
    elif op == 'hgt':
        s = p+1 # dissipation degree
        #eps = 0.8*3.125/5**s
        eps = sig_fix*3.125/5**s
        label = 'HGT'
        op_nen = nen
        op_p = p
    elif op == 'mattsson':
        s = p+1 # dissipation degree
        eps = sig_fix*3.125/5**s
        label = 'Mattsson'
        op_nen = nen
        op_p = p
    elif op == 'lg':
        s = p # dissipation degree
        if p == 2: eps = 0.02
        elif p == 3: eps = 0.01
        elif p == 4: eps = 0.004
        elif p == 5: eps = 0.002
        elif p == 6: eps = 0.0008
        elif p == 7: eps = 0.0004
        elif p == 8: eps = 0.0002
        else: raise Exception('No dissipation for this p')
        eps = sig_fix*eps
        if useH != False: print("WARNING: useH should be set to False for LG since element-type")
        if bdy_fix != False: print("WARNING: bdy_fix should be set to False for LG since element-type")
        label = 'LG'
        op_nen = 0
        op_p = p
    elif op == 'lgl':
        s = p # dissipation degree
        if p == 2: eps = 0.02
        elif p == 3: eps = 0.01
        elif p == 4: eps = 0.004
        elif p == 5: eps = 0.002
        elif p == 6: eps = 0.0008
        elif p == 7: eps = 0.0004
        elif p == 8: eps = 0.0002
        else: raise Exception('No dissipation for this p')
        eps = sig_fix*eps
        if useH != False: print("WARNING: useH should be set to False for LGL since element-type")
        if bdy_fix != False: print("WARNING: bdy_fix should be set to False for LGL since element-type")
        label = 'LGL'
        op_nen = 0
        op_p = p
    elif op == 'upwind':
        eps = 1.0
        label = f'UFD $p_u={2*p+1}$'
        op_nen = nen
        op_p = 2*p+1
    else:
        raise Exception('No dissipation for this operator')
    
    if include_nodiss and op != 'upwind':
        solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                        p=op_p, surf_diss=surf_diss, vol_diss=None,
                        nelem=nelem, nen=op_nen, disc_nodes=op,
                        bc='periodic',sparse=False)
        Omegas.append(solver.dispersion_analysis(num_k=num_k,all_modes=False,plot=False,return_omegas=True))
        #Labels.append(f'{op} $\\varepsilon=0$')
        Labels.append(None)
        Linestyles.append('--')
        Colors.append(colors[i])
    
    if op == 'upwind':
        diss = {'diss_type':'upwind', 'fluxvec':'lf', 'coeff':eps}
        solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                            p=op_p, surf_diss=surf_diss, vol_diss=diss,
                            nelem=nelem, nen=op_nen, disc_nodes=op,
                            bc='periodic',sparse=False)
        Omegas.append(solver.dispersion_analysis(num_k=num_k,all_modes=False,plot=False,return_omegas=True))
        Labels.append(label)
        Linestyles.append('-')
        Colors.append(colors[i])
    
    else:
        diss = {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps}
        solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                        p=op_p, surf_diss=surf_diss, vol_diss=diss,
                        nelem=nelem, nen=op_nen, disc_nodes=op,
                        bc='periodic',sparse=False)
        Omegas.append(solver.dispersion_analysis(num_k=num_k,all_modes=False,plot=False,return_omegas=True))
        Labels.append(f'{label} $\\varepsilon={eps}$')
        Linestyles.append('-')
        Colors.append(colors[i])


# prepare the plot

plt.figure()
plt.plot(Ks,Ks,'--',color='black')
for i in range(len(Omegas)):
    plt.plot(Ks,np.real(Omegas[i]),label=Labels[i],color=Colors[i],linestyle=Linestyles[i])
plt.xlabel(r'$K$',fontsize=16)
plt.ylabel(r'$\Re(\Omega)$',fontsize=16)
plt.title(f'Dispersion Relation p={p}',fontsize=18)
plt.legend(loc='upper left',fontsize=14)
if savefile is not None:
    plt.savefig(savefile + '_disp.png', format='png', dpi=600)

plt.figure()
plt.plot(Ks,np.zeros_like(Ks),'--',color='black')
for i in range(len(Omegas)):
    plt.plot(Ks,np.imag(Omegas[i]),label=Labels[i],color=Colors[i],linestyle=Linestyles[i])
plt.xlabel(r'$K$',fontsize=16)
plt.ylabel(r'$\Im(\Omega)$',fontsize=16)
plt.title(f'Dissipation Relation p={p}',fontsize=18)
plt.legend(loc='lower left',fontsize=14)
if savefile is not None:
    plt.savefig(savefile + '_diss.png', format='png', dpi=600)