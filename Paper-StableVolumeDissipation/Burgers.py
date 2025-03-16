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
'''
savefile = None # use a string like 'CSBPp4' to save the plot, None for no save. Note: '.png' added automatically at end
tm_method = 'rk4' # must use rk4, because rk8 does not track conservation objectives (energy, max eigs, etc)
cfl = 0.01
tf = 1./(2*np.pi) # final time
nelem = 1 # number of elements
nen = 40 # number of nodes per element
p = 4 # polynomial degree
op = 'csbp' # operator type
coeff_fix = 1.0 # additional coefficient by which to modify 3.125/5**s and 0.625/5**s
maxeig = 'rusanov' # maxeig for entropy-conservative SATs, e.g. 'rusanov', 'lf'
q0_type = 'SinWave' # initial condition 
q0_amplitude = 1. # amplitude of initial condition
xmin = 0.
xmax = 1.
bc = 'periodic' 
split_alpha = 2./3. # splitting parameter, 2/3 to recover entropy-conservative scheme
settings = {} # additional settings for mesh type, etc. Not needed.
cons_obj = ('Energy','Max_Eig','Spec_Rad','Conservation','Conservation_der','time') # what quantities to track, make sure time is included
# in the paper, we use 'Energy','Max_Eig','time'
skip_ts = 10 # number of time steps to skip for plotting / saving quantities. makes runs slightly quicker.
include_upwind = True
plot_solution = False # plot the solution at the end?
print_errors = False # print errors at the end?
plot_errors = False # plot the solution errors at the end?
show_dissipation = False # show the dissipation plots?
plot_markers = True # plot markers on the line plots?


# instantiate Burgers Diffeq object
diffeq = Burgers(None, q0_type, True, split_alpha)
diffeq.q0_max_q = q0_amplitude

if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    s = p + 1
    eps = 3.125/5**s
    useH = False
    bdy_fix = True
    dx = (xmax-xmin)/((nen-1)*nelem)
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
    dx = (xmax-xmin)/(p*nelem)
else:
    raise Exception('No dissipation for this operator')

if include_upwind:
    n_runs = 6
else:
    n_runs = 4
results = []
labels = []
assert cons_obj[-1] == 'time', 'time must be the last element of cons_obj'
for i in range(n_runs):
    # set the different solver settings
    if (i == 0 and include_upwind): # FD upwind 2p / spectral upwind
        if op == 'lg' or op == 'lgl':
            op_ = op
            p_op = p
            coeff = coeff_fix*eps
            labels.append(f'USE $\\varepsilon={coeff:g}$')
        else:
            op_ = 'upwind'
            p_op = int(2*p)
            coeff = 1.
            labels.append(f'UFD $p_\\text{{u}}={p_op}$')
        op_ = 'upwind_m'
        sat = {'diss_type':'cons', 'jac_type':'sca', 'maxeig':maxeig}
        diss = {'diss_type':'upwind', 'coeff':coeff, 'fluxvec':'lf'}
        use_split_form = False
    elif (i == 1 and include_upwind): # FD upwind 2p+1 / spectral upwind
        if op == 'lg' or op == 'lgl':
            op_ = op
            p_op = p
            coeff = coeff_fix*0.2*eps
            labels.append(f'USE $\\varepsilon={coeff:g}$')
        else:
            op_ = 'upwind'
            p_op = int(2*p+1)
            coeff = 1.
            labels.append(f'UFD $p_\\text{{u}}={p_op}$')
        sat = {'diss_type':'cons', 'jac_type':'sca', 'maxeig':maxeig}
        diss = {'diss_type':'upwind', 'coeff':coeff, 'fluxvec':'lf'}
        use_split_form = False
    elif (i == 2 and include_upwind) or (i==0 and not include_upwind): # entropy-conservative
        op_ = op
        p_op = p
        sat = {'diss_type':'ec'}
        diss = {'diss_type':'nd'}
        use_split_form = True
        labels.append(r'E.C. $\varepsilon=0$')
    elif (i == 3 and include_upwind) or (i==1 and not include_upwind): # entropy-disipative SATs only
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':maxeig}
        diss = {'diss_type':'nd'}
        use_split_form = True
        labels.append(r'E.D. $\varepsilon=0$')
    elif i == 4 or (i==2 and not include_upwind): # entropy-dissipative with comparable sigma
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':maxeig}
        diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':coeff_fix*eps, 
                'bdy_fix':True, 'use_H':True, 'avg_half_nodes':True}
        use_split_form = True
        labels.append(f"E.D. $\\varepsilon={diss['coeff']:g}$")
    elif (i == 5 and include_upwind) or (i==3 and not include_upwind): # entropy-dissipative with lower sigma
        op_ = op
        p_op = p
        sat = {'diss_type':'es', 'jac_type':'sca', 'maxeig':maxeig}
        diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':coeff_fix*0.2*eps, 
                'bdy_fix':True, 'use_H':True, 'avg_half_nodes':True}
        use_split_form = True
        labels.append(f"E.D. $\\varepsilon={diss['coeff']:g}$")


    # set solver
    dt = cfl * dx / (1.0) # using initial condition = sin wave, max eigenvalue = 1
    diffeq.use_split_form = use_split_form
    solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf, 
                        p=p_op, surf_diss=sat, vol_diss=diss,
                        nelem=nelem, nen=nen, disc_nodes=op_,
                        bc='periodic', xmin=xmin, xmax=xmax,
                        cons_obj_name=cons_obj)
    solver.skip_ts = skip_ts
    solver.tm_atol = 1e-13
    solver.tm_rtol = 3e-13

    # solve PDE
    diffeq.calc_breaking_time()
    solver.solve()

    # save results
    results.append(solver.cons_obj)

    # plot solution
    if plot_solution:
        solver.plot_sol(title=labels[i])
    
    if print_errors:
        print(f'Solution Error: {solver.calc_error():.3g}')

    if plot_errors:
        plt.figure(figsize=(6,4))
        plt.title(labels[i],fontsize=18)
        plt.ylabel(r'Solution Error $\bm{u} - \bm{u}_{\mathrm{ex}}$',fontsize=16)
        plt.xlabel(r'$x$',fontsize=16)
        er = (solver.q_sol[:,:,-1]-diffeq.exact_sol(tf))
        x = solver.diffeq.x_elem
        plt.yscale('symlog',linthresh=0.5*np.min(abs(er)))
        plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
        plt.gca().tick_params(axis='both', labelsize=12) 

        plt.plot(x[:,0], er[:,0], color='tab:blue', linewidth=2) 
        for elem in range(1,nelem):
            plt.plot(x[:,elem], er[:,elem], color='tab:blue', linewidth=2) 

    if show_dissipation and solver.adiss.type != 'nd':
        q = solver.q_sol[:,:,-1]
        diss = solver.adiss.dissipation(q)
        plt.figure(figsize=(6,4))
        plt.xlabel(r'$x$',fontsize=16)
        plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d u}{d t}$', fontsize=16)
        plt.plot(x[:,0], diss[:,0], color='tab:blue', linewidth=2) 
        for elem in range(1,nelem):
            plt.plot(x[:,elem], diss[:,elem], color='tab:blue', linewidth=2) 
        plt.yscale('symlog',linthresh=0.5*np.min(abs(diss)))
        plt.grid(which='major',axis='y',linestyle='--',linewidth='1')
        plt.gca().tick_params(axis='both', labelsize=12) 


# plot results
linewidth=2
#colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:brown']
colors = ['tab:red', 'tab:orange', 'tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']
if plot_markers:
    linestyles = ['-','--','-',':','-',':']
    markers = ['o', '^', 's', 'v', 'x', '+', '*']
    marker_start = [0.01, 0.015, 0.0, 0.005, 0.01, 0.01] # where to start markers (in time units)
    n_markers = 8 # Number of markers per line
else:
    #linestyles = [(-1, (3,1)),(0, (1,2,3,2,1,3)),'-',':',(0,(1,1,1,3)),(-1,(2,4))] # this works well when 4 dissipations overlap
    linestyles = [(1, (2,1)),(0, (1,2,2,1)),'-',(0, (1,1)),(0,(1,2)),(-1,(2,4))] # this works well when 3 dissipations overlap
    markers = ['','','','','','']
    marker_start = [0,0,0,0,0,0] # where to start markers (in time units)
    nmarkers = 0
for i in range(len(cons_obj)-1):
    
    # general figure settings
    plt.figure(figsize=(6,4))
    if cons_obj[i].lower() == 'energy':
        #plt.title(r'Change in Energy',fontsize=18)
        use_norm = True
        if use_norm:
            plt.ylabel(r'Energy Change $\Vert \bm{u} \Vert_\mathsf{H}^2 - \Vert \bm{u}_0 \Vert_\mathsf{H}^2 $',fontsize=16)
            plt.yscale('symlog',linthresh=1e-14)
        else:
            plt.ylabel(r'Energy $\Vert \bm{u} \Vert_\mathsf{H}^2 $',fontsize=16)
            plt.yscale('log')
        if p == 1 or p == 2:
            legend_loc = 'upper center'
        elif p == 3:
            legend_loc = 'upper left'
        else:
            legend_loc = 'lower left'
        #legend_loc = 'upper center'
        #legend_anchor = (0.55,0.895) #(0.525,0.925)
        #legend_anchor = (0.565,0.875) #(0.565,0.895)
        legend_loc = 'best'
        legend_anchor = None
        grid = True

    elif cons_obj[i].lower() == 'conservation':
        #plt.title(r'Change in Conservation',fontsize=18)
        plt.ylabel(r'Conservation $\left( \bm{1}^\mathsf{T} \mathsf{H} \bm{u} - \bm{1}^\mathsf{T} \mathsf{H} \bm{u}_0 \right)$',fontsize=16)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
        use_norm = True
        legend_loc = 'lower left'
        legend_anchor = None
        grid = False

    elif cons_obj[i].lower() == 'max_eig':
        #plt.title(r'Maximum Real Eigenvalue of LHS',fontsize=18)
        plt.ylabel(r'$\max \ \Re(\lambda) $',fontsize=16)
        plt.yscale('symlog',linthresh=1e-5)
        use_norm = False
        #legend_loc = 'center'
        #legend_anchor = (0.5, 0.35)
        #legend_anchor = (0.5, 0.4)
        legend_loc = 'best'
        legend_anchor = None
        grid = False
        print("!!! SANITY CHECK !!! Max eig...")
        for j in range(n_runs):
            print('... for '+labels[j]+' = ',np.max(results[j][i]))

    elif cons_obj[i].lower() == 'spec_rad':
        #plt.title(r'Spectral Radius of LHS',fontsize=18)
        plt.ylabel(r'Spectral Radius $\left( \max \ \vert \lambda \vert \right)$',fontsize=16)
        use_norm = False
        #legend_loc = 'upper right'
        legend_loc = 'best'
        legend_anchor = None
        grid = False

    if cons_obj[i].lower() == 'conservation_der':
        use_norm = False
        if use_norm:
            plt.ylabel(r'Conservation Derivative $\left( \bm{1}^\mathsf{T} \mathsf{H} \frac{\mathrm{d} \bm{u}}{\mathrm{d} t} - \frac{\mathsf{d} \bm{u}_0}{\mathsf{d} t} \right)$',fontsize=16)
        else:
            plt.ylabel(r'Conservation Derivative $\left( \bm{1}^\mathsf{T} \mathsf{H} \frac{\mathrm{d} \bm{u}}{\mathrm{d} t} \right)$',fontsize=16)
        plt.yscale('symlog',linthresh=1e-15)
        legend_loc = 'best'
        legend_anchor = None
        grid = True

    # plot!
    for j in range(n_runs):
        if use_norm:
            norm = results[j][i][0]
        else:
            norm = 0
        data = results[j][i] - norm
        time = results[j][-1]
        
        # Define the spacing interval
        marker_spacing = (time[-1] - time[0]) / (n_markers - 1)
        # Generate marker positions before shifting
        marker_positions = time[0] + np.arange(n_markers) * marker_spacing
        # Apply the offset
        marker_positions += marker_start[j]
        # Remove markers beyond the last time point
        marker_positions = marker_positions[marker_positions <= time[-1]]
        # Get corresponding indices
        marker_indices = np.searchsorted(time, marker_positions)

                
        plt.plot(time, data, color=colors[j], linestyle=linestyles[j], 
             marker=markers[j], markevery=marker_indices, label=labels[j], linewidth=linewidth, 
             markersize=9, markerfacecolor='none', markeredgewidth=linewidth,zorder=2)
        
        # again so that markers are ontop
        if plot_markers:
            plt.plot(time, data, color=colors[j], linestyle='', 
                marker=markers[j], markevery=marker_indices, label=None, linewidth=None, 
                markersize=8, markerfacecolor='none', markeredgewidth=linewidth,zorder=3)
        
    plt.xlabel(r'Time $t$',fontsize=16)
    plt.legend(loc=legend_loc,fontsize=13, 
                    bbox_to_anchor=legend_anchor)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12) 
    if grid:
        plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
    if cons_obj[i].lower() == 'energy' and use_norm:
        ymin, ymax = ax.get_ylim()
        positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
        negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
        custom_ticks = negative_ticks[::-1] + positive_ticks
        ax.set_yticks(custom_ticks)

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile+'_'+cons_obj[i].lower()+'.png',dpi=600)