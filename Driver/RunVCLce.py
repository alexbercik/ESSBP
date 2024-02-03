#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:02 2020

@author: andremarchildon
"""

import os
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.VarCoeffLinearConv import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Run code '''

# Eq parameters
para = 0.2    # Variable coefficient splitting parameter (0 to 1)
use_exact_der = True # whether to compute variable coefficient derivative exactly
extrapolate_bdy_flux = True

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.001
# note: should set according to courant number C = a dt / dx
dt_init = dt
t_init = 0
tf = 10

# Domain
xmin = -1
xmax = 1
bc = 'periodic' # 'periodic' or 'homogeneous'

# Spatial discretization
disc_type = 'div' # 'div', 'had', 'dg'
disc_nodes = 'lgl' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 3
nelem = 10 # optional, number of elements
nen = 0 # optional, number of nodes per element
surf_type = 'central'
had_flux = 'central_fix' # 2-point numerical flux used in hadamard form
diss_type = None

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'
a_type = 'shifted Gaussian'

# Other
bool_plot_sol = False
print_sol_norm = False

#cons_obj_name = ('Energy','Conservation','A_Energy','Energy_der','Conservation_der','A_Energy_der') # 'Energy', 'Conservation', 'None'
cons_obj_name = ('Energy','Conservation','A_Energy','A_Conservation')


settings = {'warp_factor':0,               # Warps / stretches mesh.
            'warp_type': 'default',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'calculate', 'exact'
            'jac_method':'exact'} 

''' Set diffeq and solve '''

if disc_type == 'fd':
    solver_c = PdeSolverFd
elif disc_type == 'dg':
    solver_c = PdeSolverDg
else:
    solver_c = PdeSolverSbp
    
diffeq = LinearConv(para, q0_type, a_type)
diffeq.use_exact_der = use_exact_der
diffeq.extrapolate_bdy_flux = extrapolate_bdy_flux

solver1D = solver_c(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, diss_type, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

solver1D.solve()
solver1D.plot_sol()
solver1D.plot_cons_obj()
#solver1D.plot_sol(savefile='alpha={0}_'.format(para)+surf_type)
#solver1D.plot_cons_obj(savefile='alpha={0}_'.format(para)+surf_type)



'''

diffeq = DiffEq(para, obj_name, q0_type)
diffeq.plt_style_exa_sol = {'color':'r','linestyle':'-','marker':'','linewidth':3}
diffeq.alpha = 2/3 # constant to determine split form (1 conservative, 1/2 skeq symmetric, 2/3 mimics burgers eqn)
savefile = 'var_coeff_alpha_23'
title=r'Variable Coefficient Advection Eqn, $\alpha=2/3$'

solver = PdeSolver(diffeq,                              # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type, nn,                     # Discretization
                  nelem, nen, sat_flux_type,
                  isperiodic, xmin, xmax,               # Domain
                  obj_name, cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)

if disc_type =='dg':
    def new_dg_dqdt_strong(self, q):
        
        q_flux = self.van @ q
        x_flux = self.van @ self.mesh.xy_elem[:,:,0]
        q_facB = self.vanf[0] @ q
        x_facB = self.vanf[0] @ self.mesh.xy_elem[:,:,0]
        q_facA = self.vanf[1] @ q
        x_facA = self.vanf[1] @ self.mesh.xy_elem[:,:,0]
        
        dqdt_out = self.vol @ self.diffeq_in.dqdt(q_flux,x_flux)
        
        nfluxA, nfluxB = np.empty((1,self.nelem)) , np.empty((1,self.nelem))
        nfluxA[:,:-1] , nfluxB[:,1:] = self.diffeq_in.numflux(q_facA[:,:-1], q_facB[:,1:],x_facA[:,:-1],x_facB[:,1:])
        nfluxA[:,[-1]] , nfluxB[:,[0]] = self.diffeq_in.numflux(q_facA[:,[-1]], q_facB[:,[0]],x_facA[:,[-1]],x_facB[:,[0]])

        flux = self.diffeq_in.calcF(q_flux,x_flux)
        dqdt_out -= self.mass_inv @ ( self.surf_num[1]@nfluxA - self.surf_flux[1]@flux \
                                    + self.surf_num[0]@nfluxB - self.surf_flux[0]@flux )

        return dqdt_out
    solver.dg_dqdt = MethodType(new_dg_dqdt_strong, solver)
    solver.diffeq = DiffEqOverwrite(solver.diffeq_in, solver.dg_dqdt, solver.dg_dfdq, solver.dg_dfds,
                                           solver.calc_cons_obj, solver.n_cons_obj)
    
else:
    def new_sbp_dqdt(self, q):   
        dqdt_out = self.diffeq_in.dqdt(q,self.mesh.xy_elem[:,:,0])
        satA , satB = np.empty(q.shape) , np.empty(q.shape)
        satA[:,:-1] , satB[:,1:] = self.diffeq_in.calc_sat(q[:,:-1], q[:,1:],self.mesh.xy_elem[:,:-1,0],self.mesh.xy_elem[:,1:,0])  
        satA[:,[-1]] , satB[:,[0]] = self.diffeq_in.calc_sat(q[:,[-1]], q[:,[0]],self.mesh.xy_elem[:,[-1],0],self.mesh.xy_elem[:,[0],0])       
        dqdt_out += self.hh_inv_phys @ ( satA + satB )
        return dqdt_out
    solver.sbp_dqdt = MethodType(new_sbp_dqdt, solver)
    solver.diffeq = DiffEqOverwrite(solver.diffeq_in, solver.sbp_dqdt, solver.sbp_dfdq, solver.sbp_dfds,
                                               solver.calc_cons_obj, solver.n_cons_obj)

solver.force_steady_solution()
solver.perturb_q0()
A = solver.check_eigs(plt_save_name=savefile+'_eigs',returnA=True,title='Eigenvalues: ' + title)
eigs = np.linalg.eigvals(A)
max_eig = max(eigs.real)
def theory_fn(time):
    return 0.001*np.exp(max_eig * time)

solver.solve()
diffeq.plt_style_sol[0] = {'color':'b','linestyle':'-','marker':'','linewidth':3}
solver.plot_sol(plt_save_name=savefile+'_sol',title=title)
solver.plot_error(method='max diff',savefile=savefile+'_error', extra_fn=theory_fn, extra_label='Theory', title=title)
solver.plot_cons_obj()

#from Methods.Analysis import animate
#animate(solver, plotargs={'display_time':True},skipsteps=100)

'''