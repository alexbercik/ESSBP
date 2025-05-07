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
from Source.Methods.Analysis import run_convergence, plot_conv

''' Set parameters for simultation 
'''
savefile = None # use a string like 'conv.png' or 'conv.pdf' to save the plot, None for no save
cfl = 0.01
tm_method = 'rk8' # if rk4, use cfl=0.01 at least. if rk8, can use cfl=0.1 (it is adaptive anyway)
tf = 1.0 # final time
nelem = [1] # number of elements, as a list (multiple if element-refinement)
nen = [40,60,80,120,160,240] # number of nodes per element, as a list (multiple if traditional-refinement)
op = 'csbp'
p = 4 # polynomial degree
s = p+1 # dissipation degree
eps = 3.125/5**s # volume dissipation coefficient for CSBP, HGTL, HGT, Mattsson (LGL/LG will be set automatically)
include_upwind = True # include upwind flux dissipation in the convergence test?
compare_formulations = False # compare different Artificial Dissipation formulations? (e.g. including H, B)
compare_operators = False # compare different operators? (e.g. CSBP, HGT, Mattsson)
bdy_fix = True # include B? Only needed if compare_formulations = False
useH = False # include H? Only needed if compare_formulations = False
include_spectral_p = True # if lgl/lg, plot dissipation against higher order? i.e. against p baseline
include_spectral_pm1 = True # if lgl/lg, plot dissipation against same order? i.e. against p-1 baseline
compare_spectral_equalp = True # if lgl/lg, plot FD dissipation against same order? i.e. against p-1 baseline
put_legend_behind = False

# set up the differential equation, plus more less important settings
print_sanity_check = False
q0_type = 'GaussWave_sbpbook' #'GaussWave_sbpbook' # initial condition 
a = 1.0 # wave speed 
diffeq = LinearConv(a, q0_type)
settings = {} # additional settings for mesh type, etc. Not needed.

assert not (compare_formulations and compare_operators), 'Can only compare one thing at a time'
if compare_operators: 
    #ops = ['csbp','hgtl','hgt','mattsson','lg','lgl']
    ops = ['csbp','hgtl','hgt','mattsson']
    assert len(nelem)==1 and len(nen)>1, 'For comparison, set nelem to one value and nen to multiple, as spectral-element settings will be set automatically'
else: 
    ops = [op]

tot_dofs = []
tot_errors = []
tot_labels = []
for op in ops:
    # set some default values
    if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
        #eps = 0.2*3.125/5**s
        dx = 1./((nen[0]-1)*nelem[0])
        if op == 'csbp': op_name = 'CSBP'
        elif op == 'hgtl': op_name = 'HGTL'
        elif op == 'hgt': op_name = 'HGT'
        elif op == 'mattsson': op_name = 'Mattsson'
    elif op in ['lg', 'lgl']:
        #eps = 0.1*2.25**(-p)
        if p == 2: eps = 0.02
        elif p == 3: eps = 0.01
        elif p == 4: eps = 0.004
        elif p == 5: eps = 0.002
        elif p == 6: eps = 0.0008
        elif p == 7: eps = 0.0004
        elif p == 8: eps = 0.0002
        else: raise Exception('No dissipation for this p')
        eps_vals = [0.02,0.01,0.004,0.002,0.0008,0.0004,0.0002]
        if useH != False: print("WARNING: useH should be set to False for LG since element-type")
        if bdy_fix != False: print("WARNING: bdy_fix should be set to False for LG since element-type")
        dx = 1./(p*nelem[0])
        if op == 'lg': op_name = 'LG'
        elif op == 'lgl': op_name = 'LGL'
    else:
        raise Exception('No dissipation for this operator')

    # set schedules for convergence tests
    if compare_formulations:
        schedule1 = [['disc_nodes',op],['nen',*nen],['p',p],['nelem',*nelem],['surf_diss',{'diss_type':'lf'}],
                    ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':False, 'use_H':False, 'coeff':eps},
                                {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':False, 'use_H':True, 'coeff':eps},
                                {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':False, 'coeff':eps},
                                {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':eps}]]
        labels1 = [r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$' ]
    elif not compare_operators:
        if op in ['lgl','lg']:
            if include_spectral_p:
                schedule1 = [['disc_nodes',op],['nen',*nen],['p',p],['nelem',*nelem],['surf_diss',{'diss_type':'lf'}],
                            ['vol_diss',{'diss_type':'nd'},
                                        {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps},
                                        {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':0.2*eps}]]
                labels1 = [f'$p={p}$, $\\varepsilon=0$',f'$p={p}$, $\\varepsilon={eps:g}$',f'$p={p}$, $\\varepsilon={0.2*eps:g}$']
            else:
                schedule1 = [['disc_nodes',op],['nen',*nen],['p',p],['nelem',*nelem],['surf_diss',{'diss_type':'lf'}],
                            ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps},
                                        {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':0.2*eps}]]
                labels1 = [f'$p={p}$, $\\varepsilon={eps:g}$',f'$p={p}$, $\\varepsilon={0.2*eps:g}$']
        else:
            schedule1 = [['disc_nodes',op],['nen',*nen],['p',p],['nelem',*nelem],['surf_diss',{'diss_type':'lf'}],
                        ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps},
                            {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':0.2*eps}]]
            labels1 = [f'$\\varepsilon={eps:g}$', f'$\\varepsilon={0.2*eps:g}$']
    else:
        if op in ['lg','lgl']:
            if compare_spectral_equalp:
                p_compare = p-1
            else:
                p_compare = p
            nelem_new = [round(n*nelem[0]/(p_compare+1)) for n in nen]
            dx = 1./(p_compare*nelem_new[0])
            eps_new = eps_vals[p_compare-2]
            schedule1 = [['disc_nodes',op],['nen',0],['p',p_compare],['nelem',*nelem_new],['surf_diss',{'diss_type':'lf'}],
                ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':p_compare, 'bdy_fix':False, 'use_H':False, 'coeff':eps_new}]]
            labels1 = [op_name + f' $\\varepsilon={eps_new:g}$']
        else:
            schedule1 = [['disc_nodes',op],['nen',*nen],['p',p],['nelem',*nelem],['surf_diss',{'diss_type':'lf'}],
                ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps}]]
            labels1 = [op_name]

    if include_upwind and (not compare_operators):
        # make sure upwind nen and nelem are FD-compatible
        if len(nen) == 1:
            nenu = [20,40,80,160]
        else:
            nenu = nen
        if len(nelem) > 1:
            nelemu = [1]
        else:
            nelemu = nelem
        schedule2 = [['disc_nodes','upwind'],['nen',*nenu],['nelem',*nelemu],['p',int(2*p), int(2*p+1)],
                    ['surf_diss',{'diss_type':'lf'}],['vol_diss',{'diss_type':'upwind', 'fluxvec':'lf', 'coeff':1.}]]
        labels2 = [f'UFD $p_\\text{{u}}={int(2*p)}$', f'UFD $p_\\text{{u}}={int(2*p+1)}$']
    else:
        schedule2 = []
        labels2 = []

    # include one runs with only SAT dissipation and no dissipation at all
    if compare_formulations:
        schedule3 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem],['p',p],['vol_diss',{'diss_type':'nd'}],
                    ['surf_diss',{'diss_type':'lf'}]]
        labels3 = [f'$\\varepsilon=0$']
    elif compare_operators:
        schedule3 = []
        labels3 = []
    elif op in ['lg','lgl']:
        if include_spectral_pm1:
            nelem_pm1 = [int(n*(p+1)/p) for n in nelem]
            schedule3 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem_pm1],['p',p-1],['vol_diss',{'diss_type':'nd'}],
                        ['surf_diss',{'diss_type':'lf'}]]
            labels3 = [f'$p={p-1}$, $\\varepsilon=0$']
        else:
            schedule3 = []
            labels3 = []
    else:
        schedule3 = [['disc_nodes',op],['nen',*nen],['nelem',*nelem],['p',p],['vol_diss',{'diss_type':'nd'}],
                ['surf_diss',{'diss_type':'nd'},{'diss_type':'lf'}]]
        labels3 = [f'Sym. $\\varepsilon=0$',f'$\\varepsilon=0$']

    # initialize solver with some default values
    dt = cfl * dx / a
    solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                        p=p, surf_diss='lf', vol_diss=None,
                        nelem=nelem[0], nen=nen[0], disc_nodes=op,
                        bc='periodic')
    solver.tm_atol = 1e-13
    solver.tm_rtol = 1e-13

    if labels1 != []:
        dofs1, errors1, outlabels1 = run_convergence(solver,schedule_in=schedule1,return_conv=True,plot=False)
    else:
        dofs1, errors1, outlabels1 = np.array([]), np.array([]), []
    if labels2 != []:
        dofs2, errors2, outlabels2 = run_convergence(solver,schedule_in=schedule2,return_conv=True,plot=False)
    else:
        dofs2, errors2, outlabels2 = np.array([]), np.array([]), []
    if labels3 != []:
        dofs3, errors3, outlabels3 = run_convergence(solver,schedule_in=schedule3,return_conv=True,plot=False)
    else:
        dofs3, errors3, outlabels3 = np.array([]), np.array([]), []
    if print_sanity_check:
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

    # plot results
    if (labels1 != []) and (labels2 != []) and (labels3 != []):
        dofs = np.vstack((dofs2, dofs3, dofs1))
        errors = np.vstack((errors2, errors3, errors1))
    elif (labels1 != []) and (labels2 != []):
        dofs = np.vstack((dofs2, dofs1))
        errors = np.vstack((errors2, errors1))
    elif (labels1 != []) and (labels3 != []):
        dofs = np.vstack((dofs3, dofs1))
        errors = np.vstack((errors3, errors1))
    elif (labels2 != []) and (labels3 != []):
        dofs = np.vstack((dofs2, dofs3))
        errors = np.vstack((errors2, errors3))
    elif labels1 != []:
        dofs = np.copy(dofs1)
        errors = np.copy(errors1)
    elif labels2 != []:
        dofs = np.copy(dofs2)
        errors = np.copy(errors2)
    elif labels3 != []:
        dofs = np.copy(dofs3)
        errors = np.copy(errors3)
    labels = labels2 + labels3 + labels1

    if compare_operators and len(tot_errors)>0:
        tot_errors = np.vstack((tot_errors,errors))
        tot_dofs = np.vstack((tot_dofs,dofs))
        tot_labels.append(*labels)
    else:
        tot_errors = np.copy(errors)
        tot_dofs = np.copy(dofs)
        tot_labels = labels.copy()

# prepare the plot
title = None
xlabel = 'Degrees of Freedom'
ylabel = r'Solution Error $\Vert \bm{u} - \bm{u}_{\mathrm{ex}} \Vert_\mathsf{H}$'
xtick = False
legendanc=None
if op in ['lg','lgl'] and include_spectral_p and include_spectral_pm1:
    colors = ['tab:green', 'tab:orange', 'k',  'm']
    markers = ['s', 'v', 'x', '+']
elif include_upwind:
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']
    markers = ['o', '^', 's', 'v', 'x', '+', 'v']
elif compare_formulations:
    colors = ['darkgoldenrod', 'tab:green', 'tab:blue', 'k',  'm']
    markers = ['v', 's', 'd', 'x', '+']
elif compare_operators:
    colors = ['m', 'tab:green', 'tab:blue', 'tab:orange', 'k', 'tab:brown']
    markers = ['+', '^', 's','o', 'x', 'v']
else:
    #colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'k',  'm', 'tab:brown']
    #markers = ['o', '^', 's', 'd', 'x', '+', 'v']
    colors = ['tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']
    markers = ['s', 'v', 'x', '+', 'v']
if p==1 or p==2: loc = 'lower left'
elif p==3: loc = 'lower left'
elif p==4: loc = 'lower left' #'upper right'
else: loc = 'best'
if compare_operators or compare_formulations: 
    figsize=(5,4.5)
    ylim=(5e-11,1.5e-2)
else:
    if op == 'csbp': ylim=(6e-11,5e-2) #ylim=(6e-11,1e-3) #ylim=(4e-11,9.5e-2) #ylim=(6e-11,1e-3)
    elif op == 'lgl' or op == 'lg': ylim=(4e-12,9.5e-2)
    else: ylim=(4e-12,9.5e-3)
    #figsize=(6,4)
    #ylim=(2e-11,3e-2)
    ylim=(5e-11,1.5e-2) # this is used for the main body
    #ylim=(5e-11,3e-2)
    figsize=(5,4.5)
    legendanc = None
if np.min(tot_dofs) == 80 and np.max(tot_dofs) == 640: 
    xlim=(68,760)
    xtick=True
else:
    xlim=None
    xtick=False
if op in ['lg','lgl']:
    ylim=(4e-11,1e-2)
    xlim=(34,750)
    xtick=True
plot_conv(tot_dofs, tot_errors, tot_labels, 1,
          title=title, savefile=savefile, xlabel=xlabel, ylabel=ylabel, put_legend_behind=put_legend_behind,
          ylim=ylim,xlim=xlim, grid=True, legendloc=loc, legend_anchor=legendanc, title_size=18,
          figsize=figsize, convunc=False, extra_xticks=xtick, scalar_xlabel=False,
          serif=True, colors=colors, markers=markers, tick_size=13, legendsize=13)
