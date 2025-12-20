import os
from sys import path
import numpy as np
from tabulate import tabulate

''' Load the saved npz data files from the KHI simulations and plot '''

# Define the base path to ESSBP (assuming it’s always under the home directory)
#base_dir = os.path.join(os.path.expanduser("~"), "ESSBP")
base_dir = os.path.join(os.path.expanduser("~"), "Desktop/UTIAS/ESSBP")

# Add the base directory to sys.path if it’s not already there
if base_dir not in path:
    path.append(base_dir)

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Disc.MakeDgOp import MakeDgOp
from Source.Disc.SbpQuadRule import SbpQuadRule
from Source.Disc.MakeMesh import MakeMesh
from Source.Disc.CSbpOp import HGTOp

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
plt.rcParams['font.family'] = 'serif'

plot_solution = True
plot_entropy = False
plot_eigs = False
plot_slice = False
plot_markers = False
final_time_marker = True
savefile_dir = 'KelvinHelmholtz_Results' # where to read from
save_sol_figs = True
entropy_savefile = None # e.g. 'filename.png'
time_to_plot = 3.7
op = 'csbp'
nelem = '1'
nen = '480'
p = '4'
load_solver_once = True
cases = ['had_ent1_matmat','had_ent02_matmat'] # e.g. ['had_ent1_matmat','had_ent02_matmat']
eps = 0.0004 # only ued for labels below
labels = [f'$Ent. Mat.-Mat. \\varepsilon={eps:g}$', f'$Ent. Mat.-Mat. \\varepsilon={0.2*eps:g}$']
rho_min = 0.4
rho_max = 2.4
sol_figs_title = None #e.g. r'CSBP $p=4$ $480^2$ nodes ent-stable mat-mat '
slice_x = -0.42
slice_savefile = None #'CSBP_p4_nen480_had_ent_matmat_slice.png'
skip_load = False # if you are running in a jupyter notebook, you can avoid reloading the solver

if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
    eps = 3.125/5**(int(p)+1)
    s = int(p)+1
    use_H = False
    bdy_fix = True
    avg_nodes = True
elif op == 'lgl' or op == 'lg':
    s = int(p)
    if p == '2': eps = 0.02
    elif p == '3': eps = 0.01
    elif p == '4': eps = 0.004
    elif p == '5': eps = 0.002
    elif p == '6': eps = 0.0008
    elif p == '7': eps = 0.0004
    elif p == '8': eps = 0.0002
    use_H = False
    bdy_fix = False
    avg_nodes = False
elif op == 'upwind':
    pass
else:
    raise Exception('No dissipation for this operator')



linewidth=2
colors = ['k', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'm', 'tab:brown']
#colors = ['tab:red', 'tab:orange', 'tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']
markers = ['o', '^', 'v', 'x', '+', 'd', 's']
marker_start = [0, 0, 0, 0, 0, 0, 0] #[0, 0, 0.5, 1, 1, 1.5, 1.5]
linestyles = ['-','-','-','--',':',':',]
n_markers = 8

time_data = []
entropy_data = []
slice_data = []
for casei, case in enumerate(cases):

    try:
        if nen == '0':
            # When nen is 0, we don't include it in the filename.
            file = savefile_dir + '/' + op + '_p' + p + '_nelem' + nelem + '_' + case + '.npz'
        else:
            file = savefile_dir + '/' + op + '_p' + p + '_nen' + nen + '_nelem' + nelem + '_' + case + '.npz'
        print('Processing case:', case)
        print('           file:', file)
        file_data = np.load(file, allow_pickle=True)
    
    except:
        print('ERROR: File {} not found. Ignoring.'.format(file))
        continue
    
    if 'save_settings' in file_data.files:
        # pull relevant settings information directly from the npz file
        save_settings = file_data['save_settings'].item()
    else:
        save_settings = {}
        
    if 'settings' in save_settings.keys():
        settings = save_settings['settings']
    else:
        settings = {'metric_method':'exact',
                    'use_optz_metrics':False}
    if plot_eigs:
        settings['stop_after_mesh'] = False
    else:
        settings['stop_after_mesh'] = True

    if 'nondimensionalize' in save_settings.keys():
        nondimensionalize = save_settings['nondimensionalize']
    else:
        nondimensionalize = False

    if 'tm_method' in save_settings.keys():
        tm_method = save_settings['tm_method']
    else:
        tm_method = 'rk8'

    if 'p' in save_settings.keys():
        assert int(p) == int(save_settings['p']), 'p in save_settings does not match p in case'
    
    if 'had_flux' in save_settings.keys():
        had_flux = save_settings['had_flux']
    else:
        had_flux = 'ranocha'

    if 'disc_type' in save_settings.keys():
        disc_type = save_settings['disc_type']
    else:
        # Determine disc_type based on case
        if 'had_nodiss' in case or 'had_ent' in case or 'had_cons' in case:
            disc_type = 'had'
        elif 'upwind' in case or 'div_nodiss' in case or 'div_cons' in case:
            disc_type = 'div'
        else:
            # Only raise exception if we need to set something but don't recognize the case
            if surf_diss is None or vol_diss is None:
                raise Exception('Unknown case {}. Ignoring.'.format(case))

    if 'dt' in save_settings.keys():
        dt = save_settings['dt']
    else:
        dt = 0.01

    if 'tf' in save_settings.keys():
        tf = save_settings['tf']
    else:
        tf = 15.0

    if 'para' in save_settings.keys():
        para = save_settings['para']
    else:
        para = [287,1.4] # [R, gamma]

    if 'test_case' in save_settings.keys():
        test_case = save_settings['test_case']
    else:
        test_case = 'kelvin-helmholtz'

    if 'q0_type' in save_settings.keys():
        q0_type = save_settings['q0_type']
    else:
        q0_type = 'kelvin-helmholtz'

    if 'xmin' in save_settings.keys():
        xmin = save_settings['xmin']
    else:
        xmin = (-1.,-1.)

    if 'xmax' in save_settings.keys():
        xmax = save_settings['xmax']
    else:
        xmax = (1.,1.)

    if 'bc' in save_settings.keys():
        bc = save_settings['bc']
    else:
        bc = 'periodic'

    if 'surf_diss' in save_settings.keys():
        surf_diss = save_settings['surf_diss']
    else:
        print(case,': surf_diss not found in save_settings.keys(): setting manually.')
        if 'had_nodiss' in case or 'had_ent' in case or 'had_cons' in case:
            # Entropy-stable surface dissipation
            if '_lfsat' in case:
                surf_diss = {'diss_type':'ent', 'jac_type':'scamat', 'coeff':1., 'average':'none', 
                        'entropy_fix':False, 'P_derigs':True, 'A_derigs':False, 'maxeig':'rusanov'}
            else:
                surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
                            'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
        elif 'upwind' in case or 'div_nodiss' in case or 'div_cons' in case:
            # Conservative surface dissipation
            if '_lfsat' in case:
                surf_diss = {'diss_type':'cons', 'jac_type':'sca', 'coeff':1., 'average':'none', 
                        'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'rusanov'}
            else:
                surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                            'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}

    if 'vol_diss' in save_settings.keys():
        vol_diss = save_settings['vol_diss']
    else:
        print(case,': vol_diss not found in save_settings.keys(): setting manually.')
        if 'had_nodiss' in case:
            vol_diss = {'diss_type':'nd'}
        elif 'had_ent' in case:
            # Extract coefficient and jac_type from case name
            if 'ent0008' in case:
                coeff_val = 0.008*eps
            elif 'ent004' in case:
                coeff_val = 0.04*eps
            elif 'ent02' in case:
                coeff_val = 0.2*eps
            elif 'ent1' in case:
                coeff_val = eps
            else:
                raise Exception('Unknown case {}. Ignoring.'.format(case))
            # Determine jac_type from case name
            if 'scamat' in case:
                jac_type = 'scamat'
            elif 'matmat' in case:
                jac_type = 'matmat'
            else:
                raise Exception('Unknown jac_type in case {}. Ignoring.'.format(case))
            vol_diss = {'diss_type':'entdcp', 'jac_type':jac_type, 's':s, 'coeff':coeff_val,
                    'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
        elif 'had_cons' in case or 'div_cons' in case:
            if 'cons1_sca' in case: # Divergence form with large scalar conservative dissipation
                vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':eps,
                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
            elif 'cons1_mat' in case: # Divergence form with large matrix conservative dissipation
                vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':eps,
                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
            elif 'cons02_sca' in case: # Divergence form with small scalar conservative dissipation
                vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':0.2*eps,
                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
            elif 'cons02_mat' in case: # Divergence form with small matrix conservative dissipation
                vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':0.2*eps,
                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
            else:
                raise Exception('Unknown case {}. Ignoring.'.format(case))
        elif 'upwind' in case:
            if op == 'lgl' or op == 'lg':
                if 'upwind02' in case:
                    vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':0.2*eps}
                else:
                    vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':eps}
            else:
                if 'upwind02' in case: run = False
                vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}
        elif 'div_nodiss' in case:
            vol_diss = {'diss_type':'nd'}

    if 'cons_obj_name' in file_data.files:
        # pull relevant settings information directly from the npz file
        cons_obj_name = tuple[str](file_data['cons_obj_name'])
    else:
        cons_obj_name = None

    if (casei == 0 or not load_solver_once) and not skip_load:
        diffeq = Euler(para, q0_type, test_case, bc, nondimensionalize)
        solver = PdeSolverSbp(diffeq, settings, tm_method, dt, tf,
                            p=int(p), disc_type=disc_type,   
                            surf_diss=surf_diss, vol_diss=vol_diss, had_flux=had_flux,
                            nelem=int(nelem), nen=int(nen), disc_nodes=op,
                            bc='periodic', xmin=xmin, xmax=xmax, cons_obj_name=cons_obj_name,
                            print_progress=False)

    if 'cons_obj' in file_data.files:
        solver.cons_obj = file_data['cons_obj']
        time_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() == 'time'][0]
        final_time = solver.cons_obj[time_idx, -1]
    else:
        solver.cons_obj = None
        final_time = None
    if 'q_sol' in file_data.files:
        solver.q_sol = file_data['q_sol']
    else:
        raise Exception('ERROR: q_sol not found in file {}.'.format(file))

    rho = solver.q_sol[::4,:,-1]
    if op in ['lgl', 'lg']:
        # perform interpolation to a finer grid (can be negative inbetween grid nodes)
        xnew = np.linspace(0,1,40)
        if op == 'lgl':
            quad = SbpQuadRule(int(p), sbp_fam='R0', nn=0, quad_rule='lgl')
        elif op == 'lg':
            quad = SbpQuadRule(int(p), sbp_fam='Rd', nn=0, quad_rule='lg')
        xsbp = quad.xq[:,0]
        mesh = MakeMesh(2,xmin,xmax,(int(nelem),int(nelem)),xsbp,print_progress=False)
        V = MakeDgOp.VandermondeLagrange1D(xnew,xsbp)
        V = np.kron(V,V)
        xy = np.zeros((int(len(xnew)**2),2,int(nelem)**2))
        xy[:,0,:] = V @ mesh.xy_elem[:,0,:]
        xy[:,1,:] = V @ mesh.xy_elem[:,1,:]
        rho = V @ rho
    elif op in ['hgt']:
        _, _, _, _, _, _, _, tL, tR = HGTOp(int(p),int(nen))
        tLT = tL.reshape(1,int(nen))
        tRT = tR.reshape(1,int(nen))
        rhoL = np.kron(tLT, np.eye(int(nen))) @ rho
        rhoR = np.kron(tRT, np.eye(int(nen))) @ rho
        rhoT = np.kron(np.eye(int(nen)), tRT) @ rho
        rhoB = np.kron(np.eye(int(nen)), tLT) @ rho
        rho = np.concatenate((rho,rhoL,rhoR,rhoT,rhoB), axis=0)
    print(case + ' min rho (total):', np.min(solver.q_sol[::4,:,:]))
    print(case + ' min rho (final time):', np.min(rho))
    print(case + f' final time: {final_time:.3g}')

    if plot_solution or plot_eigs or plot_slice:
        if time_to_plot is None:
            time_to_plot_apprx = final_time
            plot_idx = -1
        else:
            plot_idx = np.argmin(np.abs(solver.cons_obj[time_idx, :] - time_to_plot))
            time_to_plot_apprx = solver.cons_obj[time_idx, plot_idx]
        q_plot = solver.q_sol[:,:,plot_idx]
        print(f'Plotting at t={time_to_plot_apprx} (index {plot_idx})')

        if save_sol_figs:
            savefile_plot = op + '_p' + p + '_nen' + nen + '_nelem' + nelem + '_' + case + '_soln'
        else:
            savefile_plot = None
    
    if plot_solution:
        if sol_figs_title is not None:
            sol_title = sol_figs_title + labels[casei]
        else:
            sol_title = None
        solver.diffeq.plt_contour_settings = {'levels': 100, 'cmap': 'coolwarm', 'rotation': 0, 
                                              'cbar_nticks': 6, 'cbar_font_size': 16, 'cbar_tick_size': 12}
        solver.plot_sol(q=q_plot, plot_exa=False, savefile=savefile_plot, save_format='png',
                        time=time_to_plot_apprx, display_time=True, ymin=rho_min, ymax=rho_max,
                        show_negative=True, ymin_negative=0.01, title=sol_title, dpi=600,
                        label_axes=False)

    if plot_eigs:
        solver.check_eigs(q=q_plot, plot_eigs=True, savefile=None,
                            normalize=True)

    if plot_entropy:
        time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
        entropy_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'entropy'][0]
        time_data.append(solver.cons_obj[time_idx, :])
        entropy_data.append(solver.cons_obj[entropy_idx, :])

    if plot_slice:
        x_slice, y_slice = solver.plot_slice(q=q_plot, xslice=slice_x, return_slice=True, show_plot=False)
        slice_data.append((x_slice, y_slice))

if plot_entropy:

    plt.figure(figsize=(5,4.5))

    plt.ylabel(r'Entropy Change $\boldsymbol{1}^\mathsf{T} \mathsf{H} \left( \mathcal{S} \left(\boldsymbol{u}\right) -  \mathcal{S} \left(\boldsymbol{u}_0\right) \right) $',fontsize=16)
    plt.xlabel(r'Time $t$',fontsize=16)
    plt.yscale('symlog',linthresh=1e-6)
    #plt.ylim(-1e-3, 1e-8)
    plt.xlim(-0.5, 15.5)
    legend_loc = 'best'
    legend_anchor = None
    #legend_loc = 'upper center'
    #legend_anchor = (0.55,0.895) 

    for casei in range(len(cases)):
        entropy = entropy_data[casei][:] - entropy_data[casei][0] 
        time = time_data[casei]
        
        # Define the spacing interval
        marker_spacing = 15 / (n_markers - 1)
        # Generate marker positions before shifting
        marker_positions = np.arange(n_markers) * marker_spacing
        # Apply the offset
        marker_positions += marker_start[casei]
        # Remove markers beyond the last time point
        marker_positions = marker_positions[marker_positions <= np.max(time)]
        # Get corresponding indices
        marker_indices = np.searchsorted(time, marker_positions)
                
        plt.plot(time, entropy, color=colors[casei], linestyle=linestyles[casei], 
                marker='', label=labels[casei], linewidth=linewidth, 
                markerfacecolor='none', zorder=2)
        
        # again so that markers are ontop
        if plot_markers:
            plt.plot(time, entropy, color=colors[casei], linestyle='', 
                marker=markers[casei], markevery=marker_indices, label=None, linewidth=None, 
                markersize=8, markerfacecolor='none', markeredgewidth=linewidth,zorder=3)
        
        elif final_time_marker:
            if time[-1] != 15:
                plt.plot(time[-1], entropy[-1], color=colors[casei], linestyle='', 
                marker='x', label=None, linewidth=None, 
                markersize=8, markerfacecolor='none', markeredgewidth=linewidth,zorder=3)

        
    plt.legend(loc=legend_loc,fontsize=14, 
                    bbox_to_anchor=legend_anchor)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelsize=13) 
    plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')

    if time_to_plot is not None:
        if time_to_plot != 15:
            plt.axvline(x=time_to_plot, color='k', linestyle='--', linewidth=1)

    #positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
    #negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
    #custom_ticks = negative_ticks[::-1] + positive_ticks
    #ax.set_yticks(custom_ticks)

    plt.tight_layout()
    if entropy_savefile is not None:
        plt.savefig(entropy_savefile,format='png', dpi=300)

if plot_slice:
    plt.figure(figsize=(5,4.5))
    plt.ylabel(r'Density $\rho$',fontsize=16)
    plt.xlabel(r'$y$',fontsize=16)
    plt.xlim(-1, 1)
    plt.title(sol_figs_title + fr'slice at $x={slice_x}$',fontsize=12)
    legend_loc = 'best'
    legend_anchor = None
    #legend_loc = 'upper center'
    #legend_anchor = (0.55,0.895) 

    for casei in range(len(cases)):
        x_slice, y_slice = slice_data[casei]
        plt.plot(x_slice, y_slice, color=colors[casei], linestyle='-', 
                marker='', label=labels[casei], linewidth=1, 
                markerfacecolor='none', zorder=2)
    plt.legend(loc=legend_loc,fontsize=10,bbox_to_anchor=legend_anchor)
    
    plt.tight_layout()
    if slice_savefile is not None:
        plt.savefig(slice_savefile,format='png', dpi=600)