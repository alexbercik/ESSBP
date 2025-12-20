import os
from sys import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tik
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
plt.rcParams['font.family'] = 'serif'

''' Make some animations that show the effect of the boundary stencil on solution error,
    i.e. why volume dissipation is benficial to damp spurious modes. '''

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
savefile = None # e.g. 'animation'
a = 1.0 # wave speed 
cfl = 0.001
tf = 1.0 # final time
nelem = 1 # number of elements
nen = 40 # number of nodes per element, as a list
op = 'csbp' # operator type
p = 4 # polynomial degree
linear_thresh = 1e-7 #1e-8 for gauss, 1e-7 for sin, 1e-5 for LGLp4, 1e-7 fpr LGLp8
max_thresh = 9e-4 #9e-3 for csbp gauss, 9e-4 for csbp sin, 3e-2 for LGLp4, 1e-4 for LGLp8
q0_type = 'sinwave_2pi' #'sinwave_4pi' #'GaussWave_sbpbook' 'sinwave_2pi' #'squarewave' # initial condition 
settings = {} #{'warp_factor':[0.05,1.0,40],'warp_type': 'corners_periodic','jac_method':'exact'} #{} # additional settings for mesh type, etc. Not needed.
interp_num = 200 # for LG/LGL, number of nodes to interpolate to per element 
logscale = True
upwind_SAT = False
dissipation = False
eps_fix = 1.

if op  in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    s = p + 1
    eps = 3.125/5**s
    useH = False
    bdy_fix = True
elif op in ['lg', 'lgl']:
    s = p
    if p == 2: eps = 0.02
    elif p == 3: eps = 0.01
    elif p == 4: eps = 0.004
    elif p == 5: eps = 0.002
    elif p == 6: eps = 0.0008
    elif p == 7: eps = 0.0004
    elif p == 8: eps = 0.0002
    else: raise Exception('No dissipation for this p')
    useH = False
    bdy_fix = False
elif op in ['circulant']:
    s = int(p/2) + 1
    eps = 3.125/5**s
    useH = False
    bdy_fix = False
else:
    raise Exception('No dissipation for this operator')

# set run
if upwind_SAT and dissipation:
    run = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps_fix*eps},
            'sat':{'diss_type':'lf'},
            'label':f'$\\varepsilon = {eps_fix*eps:.3g}$',
            'p':p,'nelem':nelem,'nen':nen}
elif upwind_SAT and not dissipation:
    run = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'lf'},
            'label':r'$\varepsilon = 0$',
            'p':p,'nelem':nelem,'nen':nen}
elif not upwind_SAT and dissipation:
    run = {'diss':{'diss_type':'dcp', 'jac_type':'scalar', 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'coeff':eps_fix*eps},
            'sat':{'diss_type':'nd'},
            'label':f'$\\varepsilon = {eps_fix*eps:.3g}$',
            'p':p,'nelem':nelem,'nen':nen}
elif not upwind_SAT and not dissipation:
    run = {'diss':{'diss_type':'nd'},
            'sat':{'diss_type':'nd'},
            'label':r'$\varepsilon = 0$',
            'p':p,'nelem':nelem,'nen':nen}

# initialize the runs and solve
if nen == 0: 
    nen_tmp = p+1 # for LG/LGL
else:
    nen_tmp = nen
dx = 1./((nen_tmp-1)*nelem)
dt = cfl * dx / a
diffeq = LinearConv(a, q0_type)
diffeq.q0_max_q = 1.
solver = PdeSolverSbp(diffeq, settings, 'rk4', dt, tf, p=run['p'], 
                      surf_diss=run['sat'], vol_diss=run['diss'], 
                      nelem=run['nelem'], nen=run['nen'], disc_nodes=op, 
                      bc='periodic', cons_obj_name=('time'), sparse=True)
solver.keep_all_ts = True
solver.tm_nframes = 180
solver.solve()


from Source.Methods.Analysis import animate

#animate(solver, file_name=savefile, make_video=True, make_gif=False,
#               plotfunc='plot_sol_error',
#               plotargs={'display_time':True,
#                         'linear_thresh':linear_thresh, 
#                         'max_thresh':max_thresh,
#                         'color':'tab:blue', 
#                         'linestyle':'-', 
#                         'linewidth':2}, 
#               skipsteps=0,fps=24,last_frame=True,time=solver.cons_obj[0,:])

diffeq.plt_style_sol = [{'color':'k','linestyle':'-','linewidth':2,'marker':''}]
animate(solver, file_name=savefile+'sol', make_video=True, make_gif=False,
               plotfunc='plot_sol',
               plotargs={'display_time':False,
                         'ymin':-1.1, 
                         'ymax':1.1,
                         'plot_exa':False,
                         'legend':False}, 
               skipsteps=0,fps=24,last_frame=True,time=solver.cons_obj[0,:])