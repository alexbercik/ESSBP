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

para = 1.0  
tm_method = 'rk4' # doesn't actually matter here
dt = 0.001 # doesn't actually matter here
tf = 1. # doesn't actually matter here
bc = 'periodic'
nelem = 4 # number of elements
nen = 24 # number of nodes per element
q0_type = 'GaussWave_sbpbook' # doesn't actually matter here
settings = {}
diffeq = LinearConv(para, q0_type)



#savefile = None

savefile = 'LCEeigs_0001_CSBPp4lf4e24n.png'
runs = [{'op':'upwind', 'p':8, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':9, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=8$', r'Upwind $p=9$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

As = []
for run, label in zip(runs, labels):
    solver = PdeSolverSbp(diffeq, settings,tm_method, dt, tf,
                  p=run['p'],surf_diss=run['sat'], vol_diss=run['diss'],
                  nelem=nelem, nen=nen, disc_nodes=run['op'],
                  bc=bc)
    A = solver.get_LHS()
    As.append(A)

plot_eigs(As,labels=labels,savefile=savefile,line_width=2,equal_axes=True,title_size=14,legend_size=12)



# The following combinations were used in the paper:
"""
savefile = 'LCEeigs_0125_CSBPp1lf4e24n.png'
runs = [{'op':'upwind', 'p':2, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':3, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':1, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.125, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':1, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.125, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':1, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.125, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':1, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.125, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=2$', r'Upwind $p=3$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

savefile = 'LCEeigs_0025_CSBPp2lf4e24n.png'
runs = [{'op':'upwind', 'p':4, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':5, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':2, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':2, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':2, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':2, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=4$', r'Upwind $p=5$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

savefile = 'LCEeigs_0005_CSBPp3lf4e24n.png'
runs = [{'op':'upwind', 'p':6, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':7, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=6$', r'Upwind $p=7$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

savefile = 'LCEeigs_0001_CSBPp4lf4e24n.png'
runs = [{'op':'upwind', 'p':8, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':9, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=8$', r'Upwind $p=9$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']
"""


# The following combinations were NOT used in the paper:
"""
savefile = 'LCEeigs_0025_CSBPp2central4e48n.png'
runs = [{'op':'upwind', 'p':4, 'sat':'central', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':5, 'sat':'central', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':2, 'sat':'central', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':2, 'sat':'central', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':2, 'sat':'central', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':2, 'sat':'central', 'diss':{'diss_type':'dcp', 'coeff':0.025, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=4$', r'Upwind $p=5$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

savefile = 'LCEeigs_CSBPp3lf4e24n.png'
runs = [{'op':'upwind', 'p':6, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':7, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0075, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0025, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=6$', r'Upwind $p=7$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}, \ \sigma = 0.005$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 0.005$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 0.075$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 0.025$']

savefile = 'LCEeigs_0005_CSBPp3lf4e48n.png'
runs = [{'op':'upwind', 'p':6, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':7, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':3, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.005, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=6$', r'Upwind $p=7$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

savefile = 'LCEeigs_CSBPp4lf4e24n.png'
runs = [{'op':'upwind', 'p':8, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':9, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0015, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0005, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=6$', r'Upwind $p=7$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}, \ \sigma = 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 1.5 \times 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 5.0 \times 10^{-4}$']

savefile = 'LCEeigs_CSBPp4lf4e48n.png'
runs = [{'op':'upwind', 'p':8, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':9, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0015, 's':'p+1', 'bdy_fix':True, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.0005, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=6$', r'Upwind $p=7$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}, \ \sigma = 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 1.5 \times 10^{-3}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}, \ \sigma = 5.0 \times 10^{-4}$']

savefile = 'LCEeigs_0001_CSBPp4lf4e48n.png'
runs = [{'op':'upwind', 'p':8, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'upwind', 'p':9, 'sat':'lf', 'diss':{'diss_type':'upwind', 'coeff':1.0, 'fluxvec':'lf'}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':False, 'use_H':True}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':False}},
        {'op':'csbp', 'p':4, 'sat':'lf', 'diss':{'diss_type':'dcp', 'coeff':0.001, 's':'p+1', 'bdy_fix':True, 'use_H':True}},]
labels = [r'Upwind $p=8$', r'Upwind $p=9$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \mathsf{B} \tilde{\mathsf{D}}$',
        r'$\sigma \mathsf{H}^{-1} \tilde{\mathsf{D}}^\mathsf{T} \tilde{\mathsf{H}} \mathsf{B} \tilde{\mathsf{D}}$']

"""