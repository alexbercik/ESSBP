import os
from sys import path
import numpy as np

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

import Source.DiffEq.LinearConv3D as LC3D
import Source.DiffEq.LinearConv2D as LC2D
from Source.Solvers.PdeSolverSbp import PdeSolverSbp

dim = 3
p = 4
disc_nodes = 'lg'
nelems = [10,15,20]
settings = {'warp_factor':0.1,               # Warps / stretches mesh.
            'warp_type':'strong',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'ThomasLombard',   # Options: 'VinokurYee','ThomasLombard','exact'
            'bdy_metric_method':'extrapolate',   # Options: 'VinokurYee','ThomasLombard','interpolate','exact'
            'jac_method':'exact',      # Options: 'direct','match','deng','exact'
            'use_optz_metrics':False,        # Uses optimized metrics for free stream preservation.
            'calc_exact_metrics':True,      # calculate exact metrics alongside above choices.
            'metric_optz_method':'alex',    # Define the optimization procedure.
            'stop_after_metrics': True } # Do not set up physical operators, SATs, etc. only Mesh setup.
nen = 0 
disc_type = 'div'


dof = np.zeros(len(nelems))
surf_er = np.zeros(len(nelems))
vol_er = np.zeros(len(nelems))
surfinv_er = np.zeros(len(nelems))
volinv_er = np.zeros(len(nelems))
volinvR_er = np.zeros(len(nelems))
volinvL_er = np.zeros(len(nelems))
for i,nelem in enumerate(nelems):
    
    if dim == 2:
        diffeq = LC2D.LinearConv([1,1])
        xmin = (0,0)
        xmax = (1,1)
        nelem = (nelem,nelem)
    elif dim == 3:
        diffeq = LC3D.LinearConv([1,1,1])
        xmin = (0,0,0)
        xmax = (1,1,1)
        nelem = (nelem,nelem,nelem)

    solver = PdeSolverSbp(diffeq, settings,   
                        'rk4', 0.00001, 0.00001, None,                                   # Initial solution
                        p, disc_type,
                        'lf', None, 'central',
                        nelem, nen, disc_nodes,
                        'periodic', xmin, xmax)
    
    dof[i] = solver.nn[0]
    surf_er[i] = np.max(np.nan_to_num(abs(solver.mesh.bdy_metrics - solver.mesh.bdy_metrics_exa)))
    vol_er[i] = np.max(abs(solver.mesh.metrics - solver.mesh.metrics_exa))
    surfinv_er[i] = solver.check_surf_invariants(returnval = True)
    volinv_er[i], volinvR_er[i], volinvL_er[i] = solver.check_invariants(return_ers=True,return_max_only=True,returnRL=True)

print('===============================================')
print('degree p =', p)
print('dimension d =', dim)
print('quadrature degree =', solver.sbp.check_diagH(solver.sbp.x, solver.sbp.H, returndegree=True))
logx = np.log(dof)
logx_plus = np.vstack([logx, np.ones_like(logx)]).T

if vol_er[0] < 1e-10:
    print('Volume Metrics Exact.')
else:
    logy = np.log(vol_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Volume Metric Convergence: {0:.2f}'.format(conv))

if surf_er[0] < 1e-10:
    print('Surface Metrics Exact.')
else:
    logy = np.log(surf_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Surface Metric Convergence: {0:.2f}'.format(conv))

if volinv_er[0] < 1e-10:
    print('Volume Invariants Exact.')
else:
    logy = np.log(volinv_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Volume Invariants Convergence: {0:.2f}'.format(conv))

if volinvL_er[0] < 1e-10:
    print('Volume Invariants (Volume Portion / Divergence Dx + d) Exact.')
else:
    logy = np.log(volinvL_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Volume Invariants (Volume Portion / Divergence Dx + d) Convergence: {0:.2f}'.format(conv))

if volinvR_er[0] < 1e-10:
    print('Volume Invariants (Extrapolation Portion / Extrap Dx + d) Exact.')
else:
    logy = np.log(volinvR_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Volume Invariants (Extrapolation Portion / Extrap Dx + d) Convergence: {0:.2f}'.format(conv))

if surfinv_er[0] < 1e-10:
    print('Surface Invariants Exact.')
else:
    logy = np.log(surfinv_er)
    conv, _ = -np.linalg.lstsq(logx_plus, logy, rcond=None)[0]
    print('Surface Invariants Convergence: {0:.2f}'.format(conv))