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
run_sims = True # actually run all the sims? Or just print the figures?
check_eigs = True # check the eigenvalues of each discretization?
show_final_sol = False # show the plots for runs that finish?
show_dissipation = False # show the dissipation plots?
plot_aggregated = False # plot the aggregated results at the end?
savefile = None # use a string like 'CSBPp4' to save the plot, None for no save. Note: '.png' added automatically at end

nelem = 6 # number of elements
nen = 0 # number of nodes per element
op = 'lg'
p = 8
s = p # dissipation degree
disc_type = 'had' # 'div', 'had'
had_flux = 'chandrashekar' # 2-point numerical flux used in hadamard form: ismail_roe, chandrashekar, ranocha, central
cfl = 1.0
tf = 50.
tm_method = 'rk8' # 'explicit_euler', 'rk4'
para = [287,1.4] # [R, gamma]
nondimensionalize = False
diffeq = Quasi1dEuler(para, 'density_wave', 'density_wave', 'constant', 'periodic', nondimensionalize)
sat = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1.0, 'P_derigs':True, 'A_derigs':True,
    'entropy_fix':False, 'average':'none', 'maxeig':'none'}
xmin = -1.
xmax = 1.
include_nodiss = True
both_dissipation = False
use_1_fifth_diss = True
use_scamat = True
consvar2plot = 0 # 0 for rho, 1 for rho*u, 2 for e

if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    eps = 3.125/5**s
    useH = False
    bdy_fix = True
    avg_half_node = True
    dx = (xmax-xmin)/((nen-1)*nelem)
elif op == 'lgl' or op == 'lg':
    assert (s == p)
    if p == 2: eps = 0.02
    elif p == 3: eps = 0.01
    elif p == 4: eps = 0.004
    elif p == 5: eps = 0.002
    elif p == 6: eps = 0.0008
    elif p == 7: eps = 0.0004
    elif p == 8: eps = 0.0002
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

if use_scamat == False:
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
""" diss = [{'diss_type':'nd'},
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
        f'ent-scamat-{0.0001}'] """


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
    if savefile is not None: fig.savefig(savefile + '_vars.pdf', dpi=600)

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
    if savefile is not None: plt.savefig(savefile + '_diss.pdf', dpi=600)


########### plot the aggregated results (logged manually, can verify from above) ############
# NOTE: This is for CSBP
if plot_aggregated:
    from matplotlib.transforms import blended_transform_factory

    nodes = np.array([20,40,80,160,320])
    max_eigs_nodiss = np.array([2.825,0.6195,0.3792,2.225,4.017])
    max_eigs_matmat1 = np.array([25.26,2.177,0.3587,1.818e-4,5.369e-10])
    max_eigs_matmat02 = np.array([5.376,0.1388,0.002949,2.278e-06,1.06e-09])
    max_eigs_scamat1 = np.array([7.597,0.7511,0.004984,1.611e-06,5.453e-10])
    max_eigs_scamat02 = np.array([2.598,0.2984,0.001997,7.83e-07,9.319e-10])
    spec_rad_nodiss = np.array([516.2,968.2,2124,4311,9297])
    spec_rad_matmat1 = np.array([1801,1820,2013,4071,8976])
    spec_rad_matmat02 = np.array([511.6,966.3,2120,4301,9283])
    spec_rad_scamat1 = np.array([1823,1939,2009,4069,8976])
    spec_rad_scamat02 = np.array([512.4,964.3,2120,4301,9283])
    crash_time_nodiss = np.array([17.937061,33.792154,3.206111,2.497499,2.618005])
    crash_time_matmat1 = np.array([50,50,50,50,50])
    crash_time_matmat02 = np.array([50,50,50,50,50])
    crash_time_scamat1 = np.array([50,50,50,50,50])
    crash_time_scamat02 = np.array([50,50,50,50,50])

    max_eigs = np.array([max_eigs_nodiss,max_eigs_matmat1,max_eigs_matmat02,max_eigs_scamat1,max_eigs_scamat02])
    spec_rad = np.array([spec_rad_nodiss,spec_rad_matmat1,spec_rad_matmat02,spec_rad_scamat1,spec_rad_scamat02])
    crash_time = np.array([crash_time_nodiss, crash_time_matmat1, crash_time_matmat02, crash_time_scamat1, crash_time_scamat02])
    labels = [r'$\varepsilon = 0$', r'Mat.-Mat. $\varepsilon = 0.001$', r'Mat.-Mat. $\varepsilon = 0.0002$', r'Sca.-Mat. $\varepsilon = 0.001$', r'Sca.-Mat. $\varepsilon = 0.0002$']
    xlabel = r'Degrees of Freedom'
    savefile = None
    colors = ['darkgoldenrod', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    markers = ['v', 's', '^', 'd', 'o']
    linestyles = ['-', '-', '-', (1, (2,1)), (1, (2,1))]
    markersizes = [12, 11, 11, 10, 10]
    linewidth = 3
    xticks = {20 : r'$20$',
                40 : r'$40$',
                80 : r'$80$',
                200 : r'$200$'}

    for i in range(3):
        if i == 0:
            data = max_eigs
            ylabel = r'max $\Re (\lambda)$'
            yticks = None
            if savefile is not None: savefile_ = savefile + '_max_eig.pdf'
            else: savefile_ = None
            legendloc = 'lower left'
            legendanchor = None
            log = True
            ylim = (1e-10,1e2)
            legendsize = 13.8
        elif i == 1:
            data = spec_rad
            ylabel = r'max $\vert \lambda \vert$'
            yticks = {5e2 : r'$5 \times 10^2$',
                    1e3 : r'$10^3$',
                    2e3 : r'$2 \times 10^3$',
                    5e3 : r'$5 \times 10^3$',
                    1e4 : r'$10^4$'}
            ylabel_adjust = -0.2
            if savefile is not None: savefile_ = savefile + '_spec_rad.pdf'
            else: savefile_ = None
            legendloc = 'upper left'
            legendanchor = None
            log = True
            ylim = (4e2,1.3e4)
            legendsize = 12.5
        elif i == 2:
            data = crash_time
            ylabel = r'Crash Time $t_f$'
            yticks = {2 : r'$2$',
                    5 : r'$5$',
                    10 : r'$10$',
                    50 : r'$>50$'}
            ylabel_adjust = -0.10
            if savefile is not None: savefile_ = savefile + '_crash_time.pdf'
            else: savefile_ = None
            legendloc = 'upper right'
            legendanchor = (1.0,0.93)
            log = True
            ylim = (2.0, 60)
            legendsize = 12.5
        
        fig = plt.figure(figsize=(5.0,4.5))
        for j in range(len(data)):
            plt.plot(nodes, data[j], color=colors[j], linestyle=linestyles[j], 
                marker=markers[j], label=labels[j], linewidth=linewidth, 
                markersize=markersizes[j], markerfacecolor='none', markeredgewidth=linewidth,zorder=3)
        # again so that markers are ontop
        for j in range(len(data)):
            plt.plot(nodes, data[j], color=colors[j], linestyle='', 
                marker=markers[j], label=None, linewidth=None, 
                markersize=markersizes[j], markerfacecolor='none', markeredgewidth=linewidth,zorder=4)

        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.ylim(ylim)
        plt.grid(which='major', axis='y', linestyle='--', linewidth=1)
        ax = plt.gca()
        ax.tick_params(axis='x', which='both', labelbottom=False)
        ax.xaxis.set_label_coords(0.5, -0.1)
        trans = blended_transform_factory(ax.transData, fig.transFigure)
        for tick in xticks:
            ax.text(tick, 0.132, xticks[tick], transform=trans, ha='center', va='top', fontsize=14)
        if yticks is not None:
            ax.tick_params(axis='y', which='both', labelleft=False)
            ax.yaxis.set_label_coords(ylabel_adjust, 0.5)
            xmin = ax.get_xlim()[0]
            xlim = xmin - 0.05*xmin
            for tick in yticks:
                ax.text(xlim, tick, yticks[tick], ha='right', va='center', fontsize=14)
        else:
            ax.tick_params(axis='y', labelsize=14)
        legend = plt.legend(loc=legendloc, bbox_to_anchor=legendanchor, fontsize=legendsize, 
                            fancybox=True, shadow=False, ncol=1, columnspacing=1, markerscale=0.8)
        for leg_line in legend.get_lines():
            leg_line.set_linewidth(linewidth*0.7)
            leg_line.set_markeredgewidth(linewidth*0.7)
        legend.set_zorder(2)
        legend.get_frame().set_alpha(0.85)
        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.2)
        if savefile is not None: 
            plt.savefig(savefile_, dpi=600)
        else:
            plt.show()