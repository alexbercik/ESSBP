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
run_sims = False # actually run all the sims? Or just print the figures?
check_eigs = True # check the eigenvalues of each discretization?
show_final_sol = False # show the plots for runs that finish?
show_dissipation = True # show the dissipation plots?
savefile = None # use a string like 'CSBPp4' to save the plot, None for no save. Note: '.png' added automatically at end

nelem = 1 # number of elements
nen = 80 # number of nodes per element
op = 'csbp'
p = 4
s = p+1 # dissipation degree
disc_type = 'had' # 'div', 'had'
had_flux = 'chandrashekar' # 2-point numerical flux used in hadamard form: ismail_roe, chandrashekar, ranocha, central
cfl = 1.0
tf = 50.
tm_method = 'rk8' # 'explicit_euler', 'rk4'
para = [287,1.4] # [R, gamma]
diffeq = Quasi1dEuler(para, 'density_wave', 'density_wave', 'constant', 'periodic')
diffeq.nondimensionalize = False
sat = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1.0, 'P_derigs':True, 'A_derigs':True,
    'entropy_fix':False, 'average':'none', 'maxeig':'none'}
xmin = -1.
xmax = 1.
include_nodiss = True
both_dissipation = True
use_1_fifth_diss = True
use_scasca = True
consvar2plot = 0 # 0 for rho, 1 for rho*u, 2 for e

if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    eps = 0.2*3.125/5**s
    useH = False
    bdy_fix = True
    avg_half_node = True
    dx = (xmax-xmin)/((nen-1)*nelem)
elif op == 'lg':
    assert (s == p)
    if p == 1: eps = 0.05
    elif p == 2: eps = 0.02
    elif p == 3: eps = 0.0085
    elif p == 4: eps = 0.0038
    else: raise Exception('No dissipation for this p')
    useH = False
    bdy_fix = False
    avg_half_node = False
    dx = (xmax-xmin)/(p*nelem)
elif op == 'lgl':
    assert (s == p)
    if p == 1: eps = 0.06
    elif p == 2: eps = 0.023
    elif p == 3: eps = 0.0095
    elif p == 4: eps = 0.004
    elif p == 5: eps = 0.0017
    elif p == 6: eps = 0.0007
    elif p == 7: eps = 0.00031
    elif p == 8: eps = 0.000135
    # can approximate this well with eps = 0.1253*np.exp(-0.8716*s) +0.0076*s**(-2.8234)
    else: raise Exception('No dissipation for this p')
    useH = False
    bdy_fix = False
    avg_half_node = False
    dx = (xmax-xmin)/(p*nelem)
else:
    raise Exception('No dissipation for this operator')

diss = [{'diss_type':'nd'},
        {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'dcp', 'jac_type':'mat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'dcp', 'jac_type':'mat', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'dcp', 'jac_type':'sca', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'dcp', 'jac_type':'sca', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node}]
labels = ['no volume diss',
        f'ent-matmat-{eps:g}',
        f'ent-matmat-{0.2*eps:g}',
        f'ent-scamat-{eps:g}',
        f'ent-scamat-{0.2*eps:g}',
        f'mat-{eps:g}',
        f'mat-{0.2*eps:g}',
        f'sca-{eps:g}',
        f'sca-{0.2*eps:g}']

if include_nodiss == False:
    diss = diss[1:]
    labels = labels[1:]
    
if both_dissipation == False:
    diss = diss[:5]
    labels = labels[:5]

if use_scasca == False:
    diss = diss[:3] 
    labels = labels[:3]

if use_1_fifth_diss == False:
    if include_nodiss:
        tmp = [diss[0]]
        tmp = tmp + diss[1::2]
        diss = tmp
        tmp = [labels[0]]
        tmp = tmp + labels[1::2]
        labels = tmp
    else:
        diss = diss[::2]
        labels = labels[::2]

# Manual overwrite

diss = [{'diss_type':'nd'},
        {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':0.001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0005, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0002, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0005, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0002, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
        {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node}]
labels = ['no volume diss',
        f'ent-matmat-{0.001}',
        f'ent-matmat-{0.0005}',
        f'ent-matmat-{0.0002}',
        f'ent-matmat-{0.0001}',
        f'ent-scamat-{0.001}',
        f'ent-scamat-{0.0005}',
        f'ent-scamat-{0.0002}',
        f'ent-scamat-{0.0001}']


dt = cfl*dx/(35.5) # for this problem, assuming max eig is 35.5
maxeigs = []
spec_rad = []
tfinal = []
if run_sims or check_eigs:
    for i,vdiss in enumerate(diss):
        print("===============================================")
        print(f'Running {i+1} of {len(diss)}: ' + labels[i])
        solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                        p=p,surf_diss=sat, vol_diss=vdiss,  
                        had_flux=had_flux,disc_type=disc_type, 
                        nelem=nelem, nen=nen, disc_nodes=op, 
                        bc='periodic', xmin=xmin, xmax=xmax)
        solver.keep_all_ts = False
        solver.tm_atol = 1e-13
        solver.tm_rtol = 3e-13
        if check_eigs:
            eigs = solver.check_eigs(plot_eigs=False, returneigs=True)
            maxeigs.append(np.max(eigs.real))
            spec_rad.append(np.max(np.abs(eigs)))
        else:
            maxeigs.append(0.0)
            spec_rad.append(0.0)
        if run_sims:
            solver.solve()
            tfinal.append(solver.t_final)
            # if it completed, plot the final solution (only rho, by default, but can play with it)
            if solver.t_final == tf and show_final_sol: solver.plot_sol(time=solver.t_final)
        else:
            tfinal.append(0.0)
    data = [(label, f"{maxeig:.4g}", f"{rad:.4g}", round(time, len(str(dt))))
                for label, maxeig, rad, time in zip(labels, maxeigs, spec_rad, tfinal)]
    headers = ["Dissipation", "Max Re(\u03BB)", "Spec Rad", "Quit Time"]
    print('Operator=' + op + f' p={p}' + f' s={s}' + f' nelem={nelem}' + f' nen={nen}' + f' had_flux={had_flux}')
    print(tabulate(data, headers=headers, tablefmt="pretty"))

####### plot initial condition #######
if show_dissipation:
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
    ax1.plot(x, rho, color=color, linestyle='-', linewidth=2.5, label=r'$\rho$')
    #ax1.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')
    color = 'tab:orange'
    ax2 = ax1.twinx() 
    ax2.set_ylabel(r'$w_1 = \frac{\gamma - s}{\gamma-1} - \frac{\rho u^2}{2p}$', fontsize=16) 
    ax2.plot(x, w1, color=color, linestyle=':', linewidth=2.5, label=r'$w_1$')
    #ax2.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')
    ax1.tick_params(axis='both', labelsize=12) 
    ax2.tick_params(axis='both', labelsize=12) 
    fig.legend(loc='lower left',fontsize=14,bbox_to_anchor=(0.125, 0.15),
            fancybox=True,shadow=False,ncol=1,columnspacing=1.5)
    fig.tight_layout()
    if savefile is not None: fig.savefig(savefile + '_vars.png', dpi=600)

    ####### plot contributions to RHS #######
    solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                            p=p,surf_diss=sat, 
                            vol_diss={'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False},
                            had_flux=had_flux,disc_type=disc_type, 
                            nelem=nelem, nen=nen, disc_nodes=op, 
                            bc='periodic', xmin=xmin, xmax=xmax)
    diss1 = solver.adiss.dissipation(q)[consvar2plot::3].flatten('F')
    solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                            p=p,surf_diss=sat, 
                            vol_diss={'diss_type':'dcp', 'jac_type':'mat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False},  
                            had_flux=had_flux,disc_type=disc_type, 
                            nelem=nelem, nen=nen, disc_nodes=op, 
                            bc='periodic', xmin=xmin, xmax=xmax,
                            sparse=False)
    diss2 = solver.adiss.dissipation(q)[consvar2plot::3].flatten('F')

    plt.figure(figsize=(6,4))
    plt.xlabel(r'$x$',fontsize=16)
    if consvar2plot == 0: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d \rho}{d t}$', fontsize=16)
    elif consvar2plot == 1: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d \rho u}{d t}$', fontsize=16)
    elif consvar2plot == 2: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d e}{d t}$', fontsize=16)
    else: raise Exception('Invalid consvar2plot')
    color = 'tab:blue'
    #plt.plot(x, diss2, color='tab:blue', linestyle='-', linewidth=2.5, label='Cons. Sca')
    #plt.plot(x, diss1, color='tab:orange', linestyle=':', linewidth=2.5, label='Ent. Sca-Mat')
    plt.plot(x, diss2, color='tab:blue', linestyle='-', linewidth=2.5, label='Conservative Matrix')
    plt.plot(x, diss1, color='tab:orange', linestyle=':', linewidth=2.5, label='Entropy Matrix-Matrix')
    plt.yscale('symlog',linthresh=1e-6)
    #plt.ylim(-5,5)
    plt.grid(which='major',axis='y',linestyle='--',linewidth='1')
    plt.gca().tick_params(axis='both', labelsize=12) 
    #plt.legend(loc='lower left',fontsize=14, bbox_to_anchor=(-0.015, -0.02))
    plt.legend(loc='upper center',fontsize=14,  bbox_to_anchor=(0.472, 1.155), fancybox=True, shadow=False, ncol=2, columnspacing=1.5)
    plt.tight_layout()
    if savefile is not None: plt.savefig(savefile + '_diss.png', dpi=600)
