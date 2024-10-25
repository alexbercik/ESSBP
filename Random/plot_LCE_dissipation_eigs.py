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

p = 3 # SBP polynomial degree
s = p+1 # dissipation order
coeff = 0.001 # volume dissipation coefficient
# for csbp/hgtl: 3.125*5**(-s)
# for hgt: 0.6*3.125*5**(-s)
# for mattsson: 0.02 for p2, 0.001 for p3
nelem = 4 # number of elements
nen = 24 # number of nodes per element
sat = 'lf' # surface dissipation type
op = 'mattsson' # operator type - try 'csbp', 'hgtl', 'hgt', 'mattsson', 'lg', 'lgl'

plot_convex_hull = False # plot convex hull of eigenvalues?
plot_individual_eigs = True # plot individual eigenvalues?
include_upwind_2p = True # include upwind 2*p for comparison?
include_upwind_2p1 = False # include upwind 2*p+1 for comparison?
include_nodiss = True # include surface dissipation only for comparison?

q0_type = 'GaussWave_sbpbook' # doesn't actually matter here
settings = {} # warp mesh?
para = 1.0 # wave speed - doesn't actually matter here
tm_method = 'rk4' # doesn't actually matter here
dt = 0.001 # doesn't actually matter here
tf = 1. # doesn't actually matter here

# Initialize Diffeq Object
diffeq = LinearConv(para, q0_type)

savefile = None
savefile = 'LCEeigs_0001_Mattssonp3lf4e24n.png'

runs, labels = [], []
if include_upwind_2p:
    runs.append({'op':'upwind', 'p':2*p, 'sat':sat, 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}})
    labels.append(f'Upwind $p={int(2*p)}$')
if include_upwind_2p1:
    runs.append({'op':'upwind', 'p':2*p+1, 'sat':sat, 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}})
    labels.append(f'Upwind $p={int(2*p+1)}$')
if include_nodiss:
    runs.append({'op':op, 'p':p, 'sat':sat, 'diss':'nd'})
    labels.append(r'$\sigma = 0$')

runs.extend([   {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':False, 'use_H':False}},
                {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':False, 'use_H':True}},
                {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':True, 'use_H':False}},
                {'op':op, 'p':p, 'sat':sat, 'diss':{'diss_type':'dcp', 'coeff':coeff, 's':s, 'bdy_fix':True, 'use_H':True}} ])
labels.extend([ r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
                r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
                r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
                r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$' ])

As = []
for run, label in zip(runs, labels):
    solver = PdeSolverSbp(diffeq, settings,tm_method, dt, tf,
                  p=run['p'],surf_diss=run['sat'], vol_diss=run['diss'],
                  nelem=nelem, nen=nen, disc_nodes=run['op'],
                  bc='periodic')
    A = solver.calc_LHS()
    As.append(A)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red']
markers = ['o', '^', 's', 'd', 'x', '+']
linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (1, 2, 3, 2))]

plot_eigs(As,plot_convex_hull,plot_individual_eigs,labels=labels,savefile=savefile,
          line_width=2,equal_axes=True,title_size=14,legend_size=12,markersize=40, 
          markeredge=1.5, colors=colors, linestyles=linestyles, markers=markers)