import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


''' Set default parameters for simultation '''

para = [287,1.4] # [R, gamma]
nelem = 4 # number of elements
nen = 24 # number of nodes per element
op = 'csbp'
p = 4
disc_type = 'had' # 'div', 'had'
had_flux = 'chandrashekar' # 2-point numerical flux used in hadamard form: ismail_roe, chandrashekar, ranocha, central
tm_method = 'rk4' # 'explicit_euler', 'rk4'
dt = 0.0001
tf = 50.
diffeq = Quasi1dEuler(para, 'density_wave', 'density_wave', 'constant', 'periodic')
diffeq.nondimensionalize = False
sat = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1.0, 'P_derigs':True, 'A_derigs':True,
       'entropy_fix':False, 'average':'none', 'maxeig':'none'}
xmin = -1.
xmax = 1.

s = p+1 # dissipation degree
diss = [{'diss_type':'nd'},
        {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':True, 'entropy_fix':False},
        {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.625/5**s, 's':s, 'bdy_fix':False, 'use_H':True, 'entropy_fix':False},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':True, 'entropy_fix':False},
        {'diss_type':'entdcp', 'jac_type':'scamat','coeff':0.625/5**s, 's':s, 'bdy_fix':False, 'use_H':True, 'entropy_fix':False},
        {'diss_type':'dcp', 'jac_type':'mat', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':False, 'entropy_fix':False},
        {'diss_type':'dcp', 'jac_type':'mat', 'coeff':0.625/5**s, 's':s, 'bdy_fix':True, 'use_H':False, 'entropy_fix':False},
        {'diss_type':'dcp', 'jac_type':'sca', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':False, 'entropy_fix':False},
        {'diss_type':'dcp', 'jac_type':'sca', 'coeff':0.625/5**s, 's':s, 'bdy_fix':True, 'use_H':False, 'entropy_fix':False}]
labels = ['no volume diss','ent-matmat-3.125','ent-matmat-0.625','ent-scamat-3.125','ent-scamat-0.625',
          'mat-3.125','mat-0.625','sca-3.125','sca-0.625']

maxeigs = []
spec_rad = []
tfinal = []
for vdiss in(diss):
     solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                           p=p,surf_diss=sat, vol_diss=vdiss,  
                           had_flux=had_flux,disc_type=disc_type, 
                           nelem=nelem, nen=nen, disc_nodes=op, 
                           bc='periodic', xmin=xmin, xmax=xmax)
     solver.keep_all_ts = False
     
     eigs = solver.check_eigs(plot_eigs=False, returneigs=True)
     maxeigs.append(np.max(eigs.real))
     spec_rad.append(np.max(np.abs(eigs)))

     solver.solve()
     tfinal.append(solver.t_final)
     
     # if it completed, plot the final solution (only rho, by default, but can play with it)
     if solver.t_final == tf: solver.plot_sol(time=solver.t_final)

data = [(label, f"{maxeig:.4g}", f"{rad:.4g}", round(time, len(str(dt))))
        for label, maxeig, rad, time in zip(labels, maxeigs, spec_rad, tfinal)]
headers = ["Dissipation", "Max Re(\u03BB)", "Spec Rad", "Quit Time"]
print(tabulate(data, headers=headers, tablefmt="pretty"))


# plot initial condition
nelem = 1 # number of elements
nen = 100 # number of nodes per element
savefile = None
solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                           p=p,surf_diss=sat, vol_diss='nd',
                           had_flux=had_flux,disc_type=disc_type, 
                           nelem=nelem, nen=nen, disc_nodes=op, 
                           bc='periodic', xmin=xmin, xmax=xmax)

q = diffeq.set_q0()
x = diffeq.x
rho = diffeq.var2plot(q, 'rho').flatten('F')
w1 = diffeq.var2plot(q, 'w1').flatten('F')


fig, ax1 = plt.subplots(figsize=(6,4))

color = 'tab:blue'
ax1.set_xlabel(r'$x$',fontsize=16)
ax1.set_ylabel(r'$\rho$', fontsize=16)
ax1.plot(x, rho, color=color, linestyle='-', linewidth=2, label=r'$\rho$')
#ax1.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')

color = 'tab:orange'
ax2 = ax1.twinx() 
ax2.set_ylabel(r'$w_1 = \frac{\gamma - s}{\gamma-1} - \frac{\rho u^2}{2p}$', fontsize=16) 
ax2.plot(x, w1, color=color, linestyle=':', linewidth=2, label=r'$w_1$')
#ax2.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')

fig.legend(loc='lower left',fontsize=12,bbox_to_anchor=(0.125, 0.15),
           fancybox=True,shadow=False,ncol=1,columnspacing=1.5)
fig.tight_layout()
if savefile is not None: fig.savefig(savefile + '_vars.png', dpi=600)


solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                           p=p,surf_diss=sat, 
                           vol_diss={'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':True, 'entropy_fix':False},
                           had_flux=had_flux,disc_type=disc_type, 
                           nelem=nelem, nen=nen, disc_nodes=op, 
                           bc='periodic', xmin=xmin, xmax=xmax)
diss1 = solver.adiss.dissipation(q)[::3].flatten('F')
solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                           p=p,surf_diss=sat, 
                           vol_diss={'diss_type':'dcp', 'jac_type':'sca', 'coeff':3.125/5**s, 's':s, 'bdy_fix':True, 'use_H':False, 'entropy_fix':False},  
                           had_flux=had_flux,disc_type=disc_type, 
                           nelem=nelem, nen=nen, disc_nodes=op, 
                           bc='periodic', xmin=xmin, xmax=xmax)
diss2 = solver.adiss.dissipation(q)[::3].flatten('F')

plt.figure(figsize=(6,4))

plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d \rho}{d t}$', fontsize=16)

color = 'tab:blue'
ax1.set_xlabel(r'$x$',fontsize=16)
ax1.set_ylabel(r'$\rho$', fontsize=16)
plt.plot(x, diss2, color='tab:blue', linestyle='-', linewidth=2, label='Cons. Sca')
plt.plot(x, diss1, color='tab:orange', linestyle=':', linewidth=2, label='Ent. Sca-Mat')
plt.yscale('symlog',linthresh=1e-5)
plt.grid(which='major',axis='y',linestyle='--',linewidth='1')

plt.legend(loc='lower left',fontsize=12)
plt.tight_layout()
if savefile is not None: plt.savefig(savefile + '_diss.png', dpi=600)