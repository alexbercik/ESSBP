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
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
plt.rcParams['font.family'] = 'serif'

''' Set parameters for simultation 
combinations used for paper: p=1,2,3,4
'''
savefile = None #'burgers_p4_rus_4e' # None for no save, '.png' added automatically at end
tm_method = 'rk4'
dt = 0.00005
tf = 0.155 # final time
nelem = 4 # number of elements
nen = 24 # number of nodes per element
p = 2 # polynomial degree
s = p+1 # dissipation degree
op = 'csbp' # operator type
coeff_fix = 1. # additional coefficient by which to modify 3.125/5**s and 0.625/5**s
q0_type = 'SinWave' # initial condition 
q0_amplitude = 1. # amplitude of initial condition
xmin = 0.
xmax = 1.
bc = 'periodic' 
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative scheme
settings = {} # additional settings for mesh type, etc. Not needed.
cons_obj = ('Energy','Conservation','Max_Eig','Spec_Rad','time') # what quantities to track, make sure time is included
# in the paper, we use 'Energy','Conservation','Max_Eig','Spec_Rad','time'
skip_ts = 4 # number of time steps to skip for plotting / saving quantities. makes runs quicker.

# instantiate Burgers Diffeq object
diffeq = Burgers(None, q0_type, True, split_alpha)
diffeq.q0_max_q = q0_amplitude

n_runs = 6
results = []
labels = []
assert cons_obj[-1] == 'time', 'time must be the last element of cons_obj'
for i in range(n_runs):
    # set the different solver settings
    if i == 0: # upwind 2p
        op_ = 'upwind'
        p_op = int(2*p)
        sat = {'diss_type':'cons', 'jac_type':'sca', 'maxeig':'rusanov'}
        diss = {'diss_type':'upwind', 'coeff':1., 'fluxvec':'lf'}
        use_split_form = False
        labels.append(f'Upwind $p={p_op}$')
    elif i == 1: # upwind 2p+1
        op_ = 'upwind'
        p_op = int(2*p+1)
        sat = {'diss_type':'cons', 'jac_type':'sca', 'maxeig':'rusanov'}
        diss = {'diss_type':'upwind', 'coeff':1., 'fluxvec':'lf'}
        use_split_form = False
        labels.append(f'Upwind $p={p_op}$')
    elif i == 2: # entropy-conservative
        op_ = op
        p_op = p
        sat = {'diss_type':'ec'}
        diss = {'diss_type':'nd'}
        use_split_form = True
        labels.append('E.C.')
    elif i == 3: # entropy-disipative SATs only
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':'rusanov'}
        diss = {'diss_type':'nd'}
        use_split_form = True
        labels.append(r'$\sigma=0$')
    elif i == 4: # entropy-dissipative with comparable sigma
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':'rusanov'}
        diss = {'diss_type':'dcp', 'jac_type':'sca', 's':'p+1', 'coeff':coeff_fix*3.125/5**s, 
                'bdy_fix':True, 'use_H':True, 'avg_half_nodes':True}
        use_split_form = True
        labels.append(f"$\\sigma={diss['coeff']}$")
    elif i == 5: # entropy-dissipative with lower sigma
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':'rusanov'}
        diss = {'diss_type':'dcp', 'jac_type':'sca', 's':'p+1', 'coeff':coeff_fix*0.625/5**s, 
                'bdy_fix':True, 'use_H':True, 'avg_half_nodes':True}
        use_split_form = True
        labels.append(f"$\\sigma={diss['coeff']}$")


    # set solver
    diffeq.use_split_form = use_split_form
    solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf, 
                        p=p_op, surf_diss=sat, vol_diss=diss,
                        nelem=nelem, nen=nen, disc_nodes=op_,
                        bc='periodic', xmin=xmin, xmax=xmax,
                        cons_obj_name=cons_obj)
    solver.skip_ts = skip_ts

    # solve PDE
    diffeq.calc_breaking_time()
    solver.solve()

    # save results
    results.append(solver.cons_obj)


# plot results
colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red', 'tab:brown']
linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (1, 2, 3, 2))]
for i in range(len(cons_obj)-1):
    
    # general figure settings
    plt.figure(figsize=(6,4))
    if cons_obj[i].lower() == 'energy':
        #plt.title(r'Change in Energy',fontsize=18)
        use_norm = True
        if use_norm:
            plt.ylabel(r'Energy Change $\left( \Vert \bm{u} \Vert_\mathsf{H}^2 - \Vert \bm{u}_0 \Vert_\mathsf{H}^2 \right)$',fontsize=16)
            plt.yscale('symlog',linthresh=1e-14)
        else:
            plt.ylabel(r'Energy $\left( \Vert \bm{u} \Vert_\mathsf{H}^2 \right)$',fontsize=16)
            plt.yscale('log')
        if p == 1 or p == 2:
            legend_loc = 'upper center'
        else:
            legend_loc = 'lower left'
        grid = True

    elif cons_obj[i].lower() == 'conservation':
        #plt.title(r'Change in Conservation',fontsize=18)
        plt.ylabel(r'Conservation $\left( \bm{1}^\mathsf{T} \mathsf{H} \bm{u} - \bm{1}^\mathsf{T} \mathsf{H} \bm{u}_0 \right)$',fontsize=16)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
        use_norm = True
        legend_loc = 'lower left'
        grid = False

    elif cons_obj[i].lower() == 'max_eig':
        #plt.title(r'Maximum Real Eigenvalue of LHS',fontsize=18)
        plt.ylabel(r'Linear Instability $\left( \max \ \Re(\lambda) \right)$',fontsize=16)
        plt.yscale('symlog',linthresh=1e-5)
        use_norm = False
        if p == 1 or p == 2:
            legend_loc = 'center right'
        else:
            legend_loc = 'center left'
        grid = False

    elif cons_obj[i].lower() == 'spec_rad':
        #plt.title(r'Spectral Radius of LHS',fontsize=18)
        plt.ylabel(r'Spectral Radius $\left( \max \ \vert \lambda \vert \right)$',fontsize=16)
        use_norm = False
        legend_loc = 'upper right'
        grid = False

    # plot!
    for j in range(n_runs):
        if use_norm:
            norm = results[j][i][0]
        else:
            norm = 0
        plt.plot(results[j][-1], results[j][i] - norm, color=colors[j], 
                 linestyle=linestyles[j], label=labels[j]) 

    plt.xlabel(r'Time $t$',fontsize=16)
    plt.legend(loc=legend_loc,fontsize=12)
    plt.tight_layout()
    if grid:
        plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
    if cons_obj[i].lower() == 'energy' and use_norm:
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
        negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
        custom_ticks = negative_ticks[::-1] + positive_ticks
        ax.set_yticks(custom_ticks)
    if savefile is not None:
        plt.savefig(savefile+'_'+cons_obj[i].lower()+'.png',dpi=600)