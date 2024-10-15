import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import plot_eigs

''' Set default parameters for simultation '''

para = [287,1.4] # [R, gamma]
test_case = 'density_wave'
nelem = 4 # number of elements
nen = 24 # number of nodes per element
disc_type = 'div' # 'div', 'had'
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
xmin = -1.
xmax = 1.
settings = {}
diffeq = Quasi1dEuler(para, None, test_case, 'constant', 'periodic')



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
    solver = PdeSolverSbp(diffeq, settings,'rk4', 0.001, 1.0,
                  p=run['p'],surf_type=run['sat'], vol_diss=run['diss'], 
                  had_flux=had_flux,disc_type=disc_type,
                  nelem=nelem, nen=nen, disc_nodes=run['op'],
                  bc='periodic', xmin=xmin, xmax=xmax)
    #A = solver.calc_LHS()
    #As.append(A)
    solver.check_eigs(title=label)

#plot_eigs(As,labels=labels,savefile=savefile,line_width=2,equal_axes=True,title_size=14,legend_size=12)



# The following combinations were used in the paper:
"""

"""


# The following combinations were NOT used in the paper:
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