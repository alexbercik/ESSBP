import os
from sys import path
import numpy as np

# Define the base path to ESSBP (assuming it’s always under the home directory)
#base_dir = os.path.join(os.path.expanduser("~"), "ESSBP")
base_dir = os.path.join(os.path.expanduser("~"), "Desktop/UTIAS/ESSBP")

# Add the base directory to sys.path if it’s not already there
if base_dir not in path:
    path.append(base_dir)

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import run_convergence, plot_conv

# Simultation parameters
savefile_in = 'Vortex_Results/Euler2dVortex_LGp4Div_data.npz' # input .npz data file
savefile_out = None #'app_Euler2dVortex_HGTp3Had' # use a string like 'CSBPp4' to save, None for no save. Note: '.png' added automatically at end
tm_method = 'rk8'
cfl = 1.0 # if rk4, sets timestep. If rk8, sets max timestep (adaptive).
tf = 20.0 # final time. For vortex, one period is t=20
op = 'lgl' # 'lg', 'lgl', 'csbp', 'hgtl', 'hgt', 'mattsson', 'upwind'
nelem = [3] # number of elements in each direction, as a list
nen = [20,40,80,160] # number of nodes per element in each direction, as a list
p = 4 # polynomial degree
s = p+1 # dissipation degree
# trad: p+1, elem: p
disc_type = 'div' # 'div' for divergence form, 'had' for entropy-stable form
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vars2plot = ['p'] #['rho', 'entropy','q','p'] # can be any of these 4

nthreads = 1 # number of threads for batch runs
include_upwind = True # include upwind operators as a reference
include_bothdiss = True # include both cons. and ent. volume dissipation
savedata = False # save results of simulation? (ignored if reading in data)
loaddata = True # skip the actual simulation and just try to load and plot the data
plot = True
verbose_output = False # will always be false for nthreads > 1
put_legend_behind = True
shorten_legend = True

# Problem parameters
para = [287,1.4] # [R, gamma]
test_case = 'vortex' # density_wave, vortex
q0_type = 'vortex' # initial condition 
xmin = (-5.,-5.)
xmax = (5.,5.)
bc = 'periodic'
settings = {'metric_method':'exact',
            'use_optz_metrics':False} # extra things like for metrics, mesh warping, etc.

# set up the differential equation
diffeq = Euler(para, q0_type, test_case, bc)

if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    eps = 3.125/5**s
    opu = 'upwind'
    useH = False
    bdy_fix = True
    avg_nodes = True
    dx = (xmax[0]-xmin[0])/((nen[0]-1)*nelem[0])
elif op == 'lgl' or op == 'lg':
    if p == 2: eps = 0.02
    elif p == 3: eps = 0.01
    elif p == 4: eps = 0.004
    elif p == 5: eps = 0.002
    elif p == 6: eps = 0.0008
    elif p == 7: eps = 0.0004
    elif p == 8: eps = 0.0002
    else: raise Exception('No dissipation for this p')
    opu = op
    useH = False
    bdy_fix = False
    avg_nodes = False
    dx = (xmax[0]-xmin[0])/(p*nelem[0])
else:
    raise Exception('No dissipation for this operator')

# set schedules for convergence tests and set dissipations
if disc_type == 'div':
    surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
    
    if include_upwind:
        if op == 'lgl' or op == 'lg':
            schedule1 = [['disc_nodes',opu],['nen',*nen],['nelem',*nelem],['p',int(p)],
                        ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':eps, 'use_H':False, 'bdy_fix':False, 's':s},
                                    {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':0.2*eps, 'use_H':False, 'bdy_fix':False, 's':s}],
                        ['disc_type','div']]
            labels1 = [f'UFD $\\varepsilon={eps:g}$', f'UFD $\\varepsilon={0.2*eps:g}$']
        else:
            schedule1 = [['disc_nodes',opu],['nen',*nen],['nelem',*nelem],['p',int(2*p), int(2*p+1)],
                        ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}],
                        ['disc_type','div']]
            labels1 = [f'UFD $p_\\text{{u}}={int(2*p)}$', f'UFD $p_\\text{{u}}={int(2*p+1)}$']
    else:
        schedule1 = []
        labels1 = []

    schedule2 = []
    labels2 = []

    schedule3 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem],['p',p],['disc_type','div'],
                ['vol_diss',{'diss_type':'nd'},
                            {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':eps},
                            {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps},
                            {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':eps},
                            {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps}]]
    labels3 = [f'$\\varepsilon=0$', f'Cons. Sca. $\\varepsilon={eps:g}$', f'Cons. Sca. $\\varepsilon={0.2*eps:g}$',
                               f'Cons. Mat. $\\varepsilon={eps:g}$', f'Cons. Mat. $\\varepsilon={0.2*eps:g}$']
    

elif disc_type == 'had':
    surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
             'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
    
    if include_upwind:
        if op == 'lgl' or op == 'lg':
            schedule1 = [['disc_nodes',opu],['nen',*nen],['nelem',*nelem],['p',int(p)],
                        ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':eps, 'use_H':False, 'bdy_fix':False, 's':s},
                                    {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':0.2*eps, 'use_H':False, 'bdy_fix':False, 's':s}],
                        ['disc_type','div']]
            labels1 = [f'USE $\\varepsilon={eps:g}$', f'USE $\\varepsilon={0.2*eps:g}$']
        else:
            schedule1 = [['disc_nodes',opu],['nen',*nen],['nelem',*nelem],['p',int(2*p), int(2*p+1)],
                        ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}],
                        ['disc_type','div']]
            labels1 = [f'UFD $p_\\text{{u}}={int(2*p)}$', f'UFD $p_\\text{{u}}={int(2*p+1)}$']
    else:
        schedule1 = []
        labels1 = []

    if include_bothdiss:
        schedule2 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem],['p',p],['disc_type','had'],
                    ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps},
                                {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps}]]
        labels2 = [f'Cons. Sca. $\\varepsilon={0.2*eps:g}$', f'Cons. Mat. $\\varepsilon={0.2*eps:g}$']
    else:
        schedule2 = []
        labels2 = []

    schedule3 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem],['p',p],['disc_type','had'],
                ['vol_diss',{'diss_type':'nd'},
                            {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':eps},
                            {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps},
                            {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':eps},
                            {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'avg_half_nodes':avg_nodes, 'coeff':0.2*eps}]]
    labels3 = [f'$\\varepsilon=0$', f'Ent. Sca.-Mat. $\\varepsilon={eps:g}$', f'Ent. Sca.-Mat. $\\varepsilon={0.2*eps:g}$',
                               f'Ent. Mat.-Mat. $\\varepsilon={eps:g}$', f'Ent. Mat.-Mat. $\\varepsilon={0.2*eps:g}$']

if __name__ == '__main__':
    if loaddata:
        # Load the saved file
        data = np.load(savefile_in, allow_pickle=True)
        dofs1, errors1, outlabels1 = data['dofs1'], data['errors1'], data.get('labels1',None)
        dofs2, errors2, outlabels2 = data['dofs2'], data['errors2'], data.get('labels2',None)
        dofs3, errors3, outlabels3 = data['dofs3'], data['errors3'], data.get('labels3',None)
        if not include_upwind:
            dofs1, errors1 = None, None
        if not include_bothdiss:
            dofs2, errors2 = None, None

    else:
        if savedata:
            datafile = savefile_out + '_data.npz'
            if os.path.exists(datafile):
                for i in range(20):
                    print(f"Warning: The file '{datafile}' already exists and will be overwritten.")

        # initialize solver with some default values
        dt = cfl * dx / (1.56) # using a wavespeed of 1.56 from initial condition
        solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                            p=p, disc_type=disc_type,   
                            surf_diss=surf_diss, vol_diss=None, had_flux=had_flux,
                            nelem=nelem[0], nen=nen[0], disc_nodes=op,
                            bc='periodic', xmin=xmin, xmax=xmax)
        solver.print_progress = verbose_output
        solver.tm_atol = 1e-13
        solver.tm_rtol = 1e-13

        if len(schedule1) > 0: 
            dofs1, errors1, outlabels1 = run_convergence(solver,schedule_in=schedule1,return_conv=True,plot=False,vars2plot=vars2plot,nthreads=nthreads)
        else:
            dofs1, errors1, outlabels1 = None, None, []
        if len(schedule2) > 0: 
            dofs2, errors2, outlabels2 = run_convergence(solver,schedule_in=schedule2,return_conv=True,plot=False,vars2plot=vars2plot,nthreads=nthreads)
        else:
            dofs2, errors2, outlabels2 = None, None, []
        if len(schedule3) > 0: 
            dofs3, errors3, outlabels3 = run_convergence(solver,schedule_in=schedule3,return_conv=True,plot=False,vars2plot=vars2plot,nthreads=nthreads)
        else:
            dofs3, errors3, outlabels3 = None, None, []

        if savedata:
            # Save arrays in binary format just in case
            print('Saving simulation results to data file...')
            np.savez(datafile, dofs1=dofs1, errors1=errors1, labels1=outlabels1,
                               dofs2=dofs2, errors2=errors2, labels2=outlabels2, 
                               dofs3=dofs3, errors3=errors3, labels3=outlabels3, allow_pickle=True)

    if plot:
        if (outlabels1 is not None) or (outlabels2 is not None) or (outlabels3 is not None):
            print ('---------')
            print('Sanity check: ensure that these labels match:')
            if include_upwind and outlabels1 is not None: assert len(labels1) == len(outlabels1)
            if include_bothdiss and outlabels2 is not None: assert len(labels2) == len(outlabels2)
            if outlabels3 is not None: assert len(labels3) == len(outlabels3)
            if include_bothdiss and outlabels2 is not None:
                for i in range(len(labels2)):
                    print ('---------') 
                    print(labels2[i])
                    print(outlabels2[i])
            if include_upwind and outlabels1 is not None:
                for i in range(len(labels1)):
                    print ('---------') 
                    print(labels1[i])
                    print(outlabels1[i])
            if outlabels3 is not None: 
                for i in range(len(labels3)):
                    print ('---------') 
                    print(labels3[i])
                    print(outlabels3[i])
            print ('---------')
        else:
            print('WARNING: No labels to compare. Be careful you have the right data file!')

        # plot results
        arrays = [dofs for dofs in (dofs1, dofs3, dofs2) if dofs is not None and \
                not (isinstance(dofs, np.ndarray) and dofs.size == 1 and dofs[()] is None)]
        dofs = np.vstack(arrays)
        arrays = [errors for errors in (errors1, errors3, errors2) if errors is not None and \
                not (isinstance(errors, np.ndarray) and errors.size == 1 and errors[()] is None)]
        errors = np.vstack(arrays)
        labels = labels1 + labels3 + labels2

        if shorten_legend:
            labels = [label.replace("Sca.", "S.").replace("Mat.", "M.") for label in labels]
            labels = [label.replace("Cons.", "C.").replace("Ent.", "E.") for label in labels]

        title = None
        xlabel = r'$\surd$ Degrees of Freedom'

        if include_upwind:
            colors = ['tab:red', 'tab:orange']
            markers = ['o', '^']
        else:
            colors = []
            markers = []
        if disc_type == 'had':
            colors = colors + ['darkgoldenrod', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
            markers = markers + ['v', 's', '^', 'd', 'o']
        elif disc_type == 'div':
            colors = colors + ['darkgoldenrod', 'tab:green', 'k', 'tab:blue', 'm']
            markers = markers + ['v', 's', 'x', 'd', '+']
        else:
            raise Exception('Something went wrong')
        if disc_type == 'had' and include_bothdiss:
            colors = colors + ['k', 'm']
            markers = markers + ['x', '+']

        if disc_type == 'had':
            reorder = [0,5,6,1,2,3,4]
        elif disc_type == 'div':
            reorder = None

        for var in vars2plot:

            # adjust manually, because Had data files saved all 4 but Div only saved p
            if var == 'rho': varidx = 0
            elif var == 'entropy': varidx = 1
            elif var == 'q': varidx = 2
            elif var == 'p': varidx = -1
            else: raise Exception('Unknown variable to plot. Try one of [rho, q, p, entropy]')
            if var in ['rho', 'q', 'entropy'] and disc_type == 'div': raise Exception('Sorry, divergence form only saved pressure, not rho, q, or entropy.')

            if p==1 or p==2 or p==3: loc = 'lower left'
            elif p==4: loc = 'lower left'
            else: loc = 'best'

            ylim=None

            if var == 'rho':
                ylabel = r'Density Error $\Vert \rho - \rho_{\mathrm{ex}} \Vert_\mathsf{H}$'
                if op == 'csbp': ylim = (2e-10,2e-3) #ylim = (1e-10,5e-3)
            elif var == 'q': 
                ylabel = r'Solution Error $\Vert \bm{u} - \bm{u}_{\mathrm{ex}} \Vert_\mathsf{H}$'
                if op == 'csbp': ylim = (1e-9,5e-3)
            elif var == 'e':
                ylabel = r'Internal Energy Error $\Vert e - e_{\mathrm{ex}} \Vert_\mathsf{H}$'
                if op == 'csbp': ylim = (4e-11,5e-3)
            elif var == 'p':
                ylabel = r'Pressure Error $\Vert p - p_{\mathrm{ex}} \Vert_\mathsf{H}$'
                if op == 'csbp': ylim = (2e-10,1.5e-3) #ylim = (1e-10,5e-3)
            elif var == 'entropy':
                ylabel = r'Entropy Error $\Vert \mathcal{S} - \mathcal{S}_{\mathrm{ex}} \Vert_\mathsf{H}$'
                if op == 'csbp': ylim = (1e-10,5e-3)

            #figsize=(6,4)
            figsize=(5,4.5)
            ylim=(1e-12,2e-3)
            #ylim=(1e-10,2e-3)
            loc = 'lower left' #'best' #'lower left' #'best'

            if savefile_out is not None:
                savefile_var = savefile_out + '_' + var + '.png'
            else:
                savefile_var = None
            plot_conv(dofs, errors[:,:,varidx], labels, 2, 
                    title=title, savefile=savefile_var, xlabel=xlabel, ylabel=ylabel, 
                    ylim=ylim,xlim=(50,600), grid=True, legendloc=loc,
                    figsize=figsize, convunc=False, extra_xticks=True, scalar_xlabel=False,
                    serif=True, colors=colors, markers=markers, legendsize=12, legendreorder=reorder,
                    title_size=16, tick_size=13, put_legend_behind=put_legend_behind)
            

###### this is the code I ran to generate the comparison plots
# run these first lines separately.
""" varidx = -1
errors_comp = np.zeros((4, dofs.shape[1]))
errors_comp[0, :] = errors[-1,:,varidx] # run this for csbp
errors_comp[1, :] = errors[-1,:,varidx] # run this for hgtl
errors_comp[2, :] = errors[-1,:,varidx] # run this for hgt
errors_comp[3, :] = errors[-1,:,varidx] # run this for mattsson

labels = ['CSBP', 'HGTL', 'HGT', 'Mattsson']
dofs_comp = np.array([dofs[-1,:], dofs[-1,:], dofs[-1,:], dofs[-1,:]])
colors = ['m', 'tab:green', 'tab:blue', 'tab:orange']
markers = ['+', '^', 's', 'o']
savefile_var = 'Euler2dVortex_p4Had_comparison.png'
ylabel = r'Pressure Error $\Vert p - p_{\mathrm{ex}} \Vert_\mathsf{H}$'
figsize=(5,4.5)
ylim=(1e-12,2e-3)
loc = 'lower left'
plot_conv(dofs_comp, errors_comp, labels, 2, 
                    title=None, savefile=savefile_var, xlabel=xlabel, ylabel=ylabel, 
                    ylim=ylim,xlim=(50,600), grid=True, legendloc=loc,
                    figsize=figsize, convunc=False, extra_xticks=True, scalar_xlabel=False,
                    serif=True, colors=colors, markers=markers, legendsize=12, legendreorder=None,
                    title_size=16, tick_size=13, put_legend_behind=put_legend_behind) """