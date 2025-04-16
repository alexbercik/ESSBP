import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import plot_eigs

''' Set default parameters for simultation '''

savefile = None # use a string like 'eigs.png' to save the plot, None for no save
p = 4 # SBP polynomial degree
s = p+1 # dissipation order
coeff = 0.2*3.125*5**(-s) # volume dissipation coefficient
nelem = 1 # number of elements
nen = 80 # number of nodes per element
sat = 'lf' # surface dissipation type
op = 'csbp' # operator type - try 'csbp', 'hgtl', 'hgt', 'mattsson', 'lg', 'lgl'
nelemFD = 1 # if using element-type, number of elements for finite difference comparison
nenFD = 120 # if using element-type, number of nodes for finite difference comparison

plot_convex_hull = False # plot convex hull of eigenvalues?
plot_individual_eigs = True # plot individual eigenvalues?
include_upwind_2p = True # include upwind 2*p for comparison?
include_upwind_2p1 = True # include upwind 2*p+1 for comparison?
include_nodiss = True # include surface dissipation only for comparison?
normalize = True # plot h*lambda instead of lambda?

q0_type = 'GaussWave' # doesn't actually matter here
settings = {} # warp mesh?
para = 1.0 # wave speed - doesn't actually matter here
tm_method = 'rk4' # doesn't actually matter here
dt = 0.001 # doesn't actually matter here
tf = 1. # doesn't actually matter here
xmin = 0. # In our paper, we use [0,1]. In DG-USBP paper, they use [-1,1]
xmax = 1.


# Initialize Diffeq Object
diffeq = LinearConv(para, q0_type)

if op == 'lg' or op == 'lgl':
    assert p==s, f'For purely element type, p must equal s, {p} != {s}'
    if include_upwind_2p or include_upwind_2p1:
        if nen == 0: nen = p+1
        assert nelemFD*nenFD == nelem*nen, f'For upwind comparison, nelemFD*nenFD must equal nelem*nen, {nelemFD*nenFD} != {nelem*nen}'
        pu = p - 1
else:
    nelemFD = nelem
    nenFD = nen
    pu = p

runs, labels = [], []
if include_upwind_2p:
    runs.append({'op':'upwind', 'p':2*pu, 'sat':sat, 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}, 'nen':nenFD, 'nelem':nelemFD})
    labels.append(f'UFD $p_\\text{{u}}={int(2*pu)}$')
if include_upwind_2p1:
    runs.append({'op':'upwind', 'p':2*pu+1, 'sat':sat, 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}, 'nen':nenFD, 'nelem':nelemFD})
    labels.append(f'UFD $p_\\text{{u}}={int(2*pu+1)}$')
if include_nodiss:
    runs.append({'op':op, 'p':p, 'sat':sat, 'diss':'nd', 'nen':nen, 'nelem':nelem})
    labels.append(r'$\varepsilon = 0$')

if op == 'lg' or op == 'lgl':
    runs.extend([   {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':False, 'use_H':False}, 'nen':nen, 'nelem':nelem},
                    {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':0.2*coeff, 's':s, 'bdy_fix':False, 'use_H':False}, 'nen':nen, 'nelem':nelem} ])
    labels.extend([ r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
                    r'$-\frac{1}{5} \varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$', ])
else:
    runs.extend([   {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':False, 'use_H':False}, 'nen':nen, 'nelem':nelem},
                    {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':False, 'use_H':True}, 'nen':nen, 'nelem':nelem},
                    {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':True, 'use_H':False}, 'nen':nen, 'nelem':nelem},
                    {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':True, 'use_H':True}, 'nen':nen, 'nelem':nelem} ])
    labels.extend([ r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
                    r'$-\varepsilon \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$' ])

As = []
for run, label in zip(runs, labels):
    solver = PdeSolverSbp(diffeq, settings,tm_method, dt, tf,
                  p=run['p'],surf_diss=run['sat'], vol_diss=run['diss'],
                  nelem=run['nelem'], nen=run['nen'], disc_nodes=run['op'],
                  bc='periodic', xmin=xmin, xmax=xmax)
    A = solver.calc_LHS()
    if normalize:
        A /= solver.nelem*(solver.nen-1)/(xmax-xmin)
    As.append(A)

if include_upwind_2p and include_upwind_2p1:
    if include_nodiss:
        colors = ['tab:red', 'tab:orange', 'tab:brown', 'tab:green', 'tab:blue', 'k',  'm']
        markers = ['o', '^', 'v', 's', 'd', 'x', '+']
    else:
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'k',  'm', 'tab:brown']
        markers = ['o', '^', 's', 'd', 'x', '+', 'v']
elif include_upwind_2p and include_nodiss:
    colors = ['tab:red', 'tab:brown', 'tab:green', 'tab:blue', 'k',  'm']
    markers = ['o', 'v', 's', 'd', 'x', '+']
elif include_upwind_2p1 and include_nodiss:
    colors = ['tab:orange', 'tab:brown', 'tab:green', 'tab:blue', 'k',  'm']
    markers = ['^', 'v', 's', 'd', 'x', '+']
elif include_nodiss:
    colors = ['tab:brown', 'tab:green', 'tab:blue', 'k',  'm', 'tab:brown']
    markers = ['v', 's', 'd', 'x', '+', 'v']
else:
    colors = ['tab:green', 'tab:blue', 'k',  'm', 'tab:brown']
    markers = ['s', 'd', 'x', '+', 'v']
linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (1, 2, 3, 2))]

if normalize:
    xlabel = r'$\Re(\lambda) \Delta x$'
    ylabel = r'$\Im(\lambda) \Delta x$'
else:
    xlabel = r'$\Re(\lambda)$'
    ylabel = r'$\Im(\lambda)$'

xlim=(-3.5,0.5) # normally use (-4.1,0.2) for large, (-3.5,0.2) for small, (-2.3,1.9) for main body
ylim=None # normally use (-1.5,1.5) for large, (-0.8,0.8) for small, (-1.2,1.2) for main body
plot_eigs(As,plot_convex_hull,plot_individual_eigs,labels=labels,savefile=savefile,
          line_width=2,equal_axes=True,title_size=16,legend_size=14,markersize=50, 
          markeredge=1.4, tick_size=12, colors=colors, linestyles=linestyles, markers=markers,
          legend_loc='upper left', #legend_anchor=(0.0, 0.88), legend_anchor_type=('data','fig'),
          legend_alpha=0.9, left_space_pct=None, xlabel=xlabel, ylabel=ylabel,
          xlim=xlim, ylim=ylim)

# Quick sanity check
import numpy as np
minleft = 0.0
for A in As:
    eigs = np.linalg.eigvals(A)
    minleft = min(minleft, np.min(np.real(eigs)))
print(f'Minimum real part of all discretizations: {minleft}')

                        
# plotting format for main body:
""" savefile = None
xlim=(-3.0,0.2) # normally use (-4.1,0.2) for large, (-3.5,0.2) for small, (-2.3,1.9) for main body
ylim=(-2.09,2.09) # normally use (-1.5,1.5) for large, (-0.8,0.8) for small, (-1.2,1.2) for main body
plot_eigs(As,plot_convex_hull,plot_individual_eigs,labels=labels,savefile=savefile,
          line_width=2,equal_axes=True,title_size=16,legend_size=14,markersize=50, 
          markeredge=1.4, tick_size=12, colors=colors, linestyles=linestyles, markers=markers,
          legend_loc='lower left', #legend_anchor=(0.0, 0.88), legend_anchor_type=('data','fig'),
          legend_alpha=0.85, left_space_pct=None, xlabel=xlabel, ylabel=ylabel,
          xlim=xlim, ylim=ylim, adjust_axes=False, tick_interval=0.5) """