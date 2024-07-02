import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverFd import PdeSolverFd
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverDg import PdeSolverDg

''' Set parameters for simultation '''

# Eq parameters
para = 1      # Wave speed a

# Time marching
tm_method = 'rk4' # explicit_euler, rk4
dt = 0.0001
# note: should set according to courant number C = a dt / dx
tf = 1.00

# Domain
xmin = 0
xmax = 1
bc = 'periodic'

# Spatial discretization
disc_type = 'div' # 'div', 'had'
disc_nodes = 'csbp' 
p = 4
nelem = 20 # optional, number of elements
nen = 20 # 9, 13, 17
surf_type = 'llf'
vol_diss = None
had_flux = None

# Initial solution
q0 = None # can overwrite q0_type from DiffEq
q0_type = 'GaussWave' # 'GaussWave', 'SinWave'

# Other
bool_plot_sol = False
print_sol_norm = False

cons_obj_name = (None) # 'Energy', 'Conservation', 'None'
settings = {}

''' Set diffeq and solve '''
solver = PdeSolverSbp
diffeq = LinearConv(para, q0_type)
solver1D = solver(diffeq, settings,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  q0,                                   # Initial solution
                  p, disc_type,             # Discretization
                  surf_type, vol_diss, had_flux,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,         # Domain
                  cons_obj_name,              # Other
                  bool_plot_sol, print_sol_norm)


''' Analyze results '''

#solver1D.solve()
#print('Final Error: ', solver1D.calc_error())

from Source.Methods.Analysis import run_convergence
# sigmoid
schedule = [['disc_nodes','csbp','optz'],['p',2,3,4],['nelem',4,8,16,32,64],['nen',9,13,17]]
# corners
#schedule = [['disc_nodes','csbp','optz'],['p',1,2,3,4],['nelem',2,4,8,16,32],['nen',20,20,20,30]]
#dofs, errors, labels = run_convergence(solver1D,schedule_in=schedule,savefile='convergence_sigmoid.png',
#                title=r'Error Convergence', xlabel=r'Nodes',grid=True,return_conv=True,
#                ylabel=r'$\vert \vert u - u_{ex} \vert \vert_H$',convunc=False,
#                labels=[r'$p=2$ CSBP',r'$p=3$ CSBP',r'$p=4$ CSBP',r'$p=2$ optz',r'$p=3$ optz',r'$p=4$ optz'])