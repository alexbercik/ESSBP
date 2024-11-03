import os
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import run_convergence, plot_conv

# Simultation parameters
savefile = 'Euler2dVortex_CSBPp14eDiv' # will add extension + .png automatically
tm_method = 'rk4'
cfl = 0.05
tf = 20. # final time. For vortex, one period is t=20
op = 'csbp' # 'lg', 'lgl', 'csbp', 'hgtl', 'hgt', 'mattsson', 'upwind'
nelem = 4 # number of elements
nen = [20,40,80,160] # number of nodes per element in each direction, as a list
p = 1 # polynomial degree
s = p+1 # dissipation degree
disc_type = 'div' # 'div' for divergence form, 'had' for entropy-stable form
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
vars2plot = ['rho','entropy','q','p'] # must be the same as loaded data

include_upwind = True # include upwind operators as a reference
include_bothdiss = True # include both cons. and ent. volume dissipation
savedata = True
loaddata = False # skip the actual simulation and just try to load and plot the data

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

# set schedules for convergence tests and set dissipations
if disc_type == 'div':
    surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
    
    if include_upwind:
        schedule1 = [['disc_nodes','upwind'],['nen',*nen],['p',int(2*p), int(2*p+1)],
                    ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}],
                    ['disc_type','div']]
        labels1 = [f'Upwind $p_\\text{{u}}={int(2*p)}$', f'Upwind $p_\\text{{u}}={int(2*p+1)}$']
    else:
        schedule1 = []
        labels1 = []

    schedule2 = []
    labels2 = []

    schedule3 = [['disc_nodes','csbp'],['nen',*nen],['p',p],['disc_type','div'],
                ['vol_diss',{'diss_type':'nd'},
                            {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125/5**s},
                            {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s},
                            {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125/5**s},
                            {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s}]]
    labels3 = [f'$\\sigma=0$', f'Cons. Sca. $\\sigma={0.625/5**p:g}$', f'Cons. Sca. $\\sigma={0.2*0.625/5**p:g}$',
                               f'Cons. Mat. $\\sigma={0.625/5**p:g}$', f'Cons. Mat. $\\sigma={0.2*0.625/5**p:g}$']
    

elif disc_type == 'had':
    surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
             'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
    
    if include_upwind:
        schedule1 = [['disc_nodes','upwind'],['nen',*nen],['p',int(2*p), int(2*p+1)],
                    ['vol_diss',{'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}],
                    ['disc_type','div']]
        labels1 = [f'Upwind $p_\\text{{u}}={int(2*p)}$', f'Upwind $p_\\text{{u}}={int(2*p+1)}$']
    else:
        schedule1 = []
        labels1 = []

    if include_bothdiss:
        schedule2 = [['disc_nodes','csbp'],['nen',*nen],['p',p],['disc_type','had'],
                    ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s},
                                {'diss_type':'dcp', 'jac_type':'matrix', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s}]]
        labels2 = [f'Cons. Sca. $\\sigma={0.2*0.625/5**p:g}$', f'Cons. Mat. $\\sigma={0.2*0.625/5**p:g}$']

    schedule3 = [['disc_nodes','csbp'],['nen',*nen],['p',p],['disc_type','had'],
                ['vol_diss',{'diss_type':'nd'},
                            {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125/5**s},
                            {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s},
                            {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125/5**s},
                            {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s}]]
    labels3 = [f'$\\sigma=0$', f'Ent. Sca.-Mat. $\\sigma={0.625/5**p:g}$', f'Ent. Sca.-Mat. $\\sigma={0.2*0.625/5**p:g}$',
                               f'Ent. Mat.-Mat. $\\sigma={0.625/5**p:g}$', f'Ent. Mat.-Mat. $\\sigma={0.2*0.625/5**p:g}$']


if loaddata:
    # Load the saved file
    data = np.load(savefile + '_data.npz', allow_pickle=True)
    dofs1, errors1, dofs2, errors2, dofs3, errors3 = data['dofs1'], data['errors1'], data['dofs2'], data['errors2'], data['dofs3'], data['errors3']
    if not include_upwind:
        dofs1, errors1 = None, None
    if not include_bothdiss:
        dofs2, errors2 = None, None

else:
    if savedata:
        datafile = savefile + '_data.npz'
        if os.path.exists(datafile):
            print(f"Warning: The file '{datafile}' already exists and will be overwritten.")

    # initialize solver with some default values
    dx = 10./((nen[0]-1)*nelem)
    dt = cfl * dx / (1.56) # using a wavespeed of 1.56 from initial condition
    solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                        p=p, surf_diss=surf_diss, vol_diss=None,
                        nelem=nelem, nen=nen[0], disc_nodes='csbp',
                        bc='periodic', xmin=xmin, xmax=xmax)

    if len(schedule1) > 0: 
        dofs1, errors1, outlabels1 = run_convergence(solver,schedule_in=schedule1,return_conv=True,plot=False,vars2plot=vars2plot)
    else:
        dofs1, errors1, outlabels1 = None, None, []
    if len(schedule2) > 0: 
        dofs2, errors2, outlabels2 = run_convergence(solver,schedule_in=schedule2,return_conv=True,plot=False,vars2plot=vars2plot)
    else:
        dofs2, errors2, outlabels2 = None, None, []
    if len(schedule3) > 0: 
        dofs3, errors3, outlabels3 = run_convergence(solver,schedule_in=schedule3,return_conv=True,plot=False,vars2plot=vars2plot)
    else:
        dofs3, errors3, outlabels3 = None, None, []

    print ('---------')
    print('Sanity check: ensure that these labels match:')
    assert len(labels1) == len(outlabels1)
    assert len(labels2) == len(outlabels2)
    assert len(labels3) == len(outlabels3)
    for i in range(len(labels2)):
        print ('---------') 
        print(labels2[i])
        print(outlabels2[i])
    for i in range(len(labels1)):
        print ('---------') 
        print(labels1[i])
        print(outlabels1[i])
    for i in range(len(labels3)):
        print ('---------') 
        print(labels3[i])
        print(outlabels3[i])
    print ('---------')

    if savedata:
        # Save arrays in binary format just in case
        print('Saving simulation results to data file...')
        np.savez(datafile, dofs1=dofs1, errors1=errors1, dofs2=dofs2, errors2=errors2, dofs3=dofs3, errors3=errors3, allow_pickle=True)

# plot results
arrays = [dofs for dofs in (dofs1, dofs2, dofs3) if dofs is not None and \
          not (isinstance(dofs, np.ndarray) and dofs.size == 1 and dofs[()] is None)]
dofs = np.vstack(arrays)
arrays = [errors for errors in (errors1, errors2, errors3) if errors is not None and \
          not (isinstance(errors, np.ndarray) and errors.size == 1 and errors[()] is None)]
errors = np.vstack(arrays)
labels = labels1 + labels2 + labels3

title = None
xlabel = r'$\surd$ Degrees of Freedom'
colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red', 'tab:brown']
markers = ['o', '^', 's', 'd','x', '+', 'v']

for varidx, var in enumerate(vars2plot):

    if p==1 or p==2 or p==3: loc = 'lower left'
    elif p==4: loc = 'upper right'
    else: loc = 'best'

    if var == 'rho':
        ylabel = r'Density Error $\Vert \rho - \rho_{\mathrm{ex}} \Vert_\mathsf{H}$'
        ylim = (4e-11,9.5e-2)
    elif var == 'q': 
        ylabel = r'Solution Error $\Vert \bm{u} - \bm{u}_{\mathrm{ex}} \Vert_\mathsf{H}$'
        ylim = (4e-11,9.5e-2)
    elif var == 'e':
        ylabel = r'Internal Energy Error $\Vert e - e_{\mathrm{ex}} \Vert_\mathsf{H}$'
        ylim = (4e-11,9.5e-2)
    elif var == 'p':
        ylabel = r'Pressure Error $\Vert p - p_{\mathrm{ex}} \Vert_\mathsf{H}$'
        ylim = (4e-11,9.5e-2)
    elif var == 'entropy':
        ylabel = r'Entropy Error $\Vert \mathcal{S} - \mathcal{S}_{\mathrm{ex}} \Vert_\mathsf{H}$'
        ylim = (4e-11,9.5e-2)

    savefile_var = savefile + '_' + var + '.png'
    plot_conv(dofs, errors[:,:,varidx], labels, 2, 
            title=title, savefile=savefile_var, xlabel=xlabel, ylabel=ylabel, 
            ylim=ylim,xlim=(68,760), grid=True, legendloc=loc,
            figsize=(6,4), convunc=False, extra_xticks=True, scalar_xlabel=False,
            serif=True, colors=colors, markers=markers)
