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
combinations used for paper: p=1, p=2, p=3, p=4
'''
savefile = None # None for no save
a = 1.0 # wave speed 
tm_method = 'rk4'
cfl = 0.1
tf = 1. # final time
bc = 'periodic'
nelem = 4 # number of elements
nen = [20,40,80,160] # number of nodes per element, as a list
p = 3 # polynomial degree
s = p+1 # dissipation degree
surf_diss = {'diss_type':'lf'} # SAT dissipation
q0_type = 'GaussWave_sbpbook' # initial condition 
settings = {} # additional settings for mesh type, etc. Not needed.

# set up the differential equation
diffeq = LinearConv(a, q0_type)

# set schedules for convergence tests
schedule1 = [['disc_nodes','csbp'],['nen',*nen],
            ['vol_diss',{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.0},
                        {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':3.125/5**s},
                        {'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':True, 'use_H':True, 'coeff':0.2*3.125/5**s}]]
schedule2 = [['disc_nodes','upwind'],['nen',*nen],['p',int(2*p), int(2*p+1)],
             ['vol_diss',{'diss_type':'upwind', 'fluxvec':'lf', 'coeff':1.}]]

# prepare the plot
title = None
xlabel = 'Degrees of Freedom'
ylabel = r'Solution Error $\Vert \bm{u} - \bm{u}_{\mathrm{ex}} \Vert_\mathsf{H}$'
labels1 = [f'$\\sigma=0$', f'$\\sigma={0.625/5**p:g}$', f'$\\sigma={0.2*0.625/5**p:g}$']
labels2 = [f'Upwind $p={int(2*p)}$', f'Upwind $p={int(2*p+1)}$']

# initialize solver with some default values
dx = 1./((nen[0]-1)*nelem)
dt = cfl * dx / a
solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                    p=p, surf_diss=surf_diss, vol_diss=None,
                    nelem=nelem, nen=nen[0], disc_nodes='csbp',
                    bc=bc)

dofs1, errors1, outlabels1 = run_convergence(solver,schedule_in=schedule1,return_conv=True,plot=False)
dofs2, errors2, outlabels2 = run_convergence(solver,schedule_in=schedule2,return_conv=True,plot=False)
print ('---------')
print('Sanity check: ensure that these labels match:')
assert len(labels1) == len(outlabels1)
assert len(labels2) == len(outlabels2)
for i in range(len(labels2)):
    print ('---------') 
    print(labels2[i])
    print(outlabels2[i])
for i in range(len(labels1)):
    print ('---------') 
    print(labels1[i])
    print(outlabels1[i])
print ('---------')

# plot results
dofs = np.vstack((dofs2, dofs1))
errors = np.vstack((errors2, errors1))
labels = labels2 + labels1

plot_conv(dofs, errors, labels, 1, 
          title=title, savefile=savefile, xlabel=xlabel, ylabel=ylabel, 
          ylim=(4e-11,9.5e-2),xlim=(68,760), grid=True, legendloc='lower left',
          figsize=(6,4), convunc=False, extra_xticks=True, scalar_xlabel=False,
          serif=False)