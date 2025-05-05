import os
from sys import path
import numpy as np

# Define the base path to ESSBP (assuming it’s always under the home directory)
base_dir = os.path.join(os.path.expanduser("~"), "ESSBP")
#base_dir = os.path.join(os.path.expanduser("~"), "Desktop/UTIAS/ESSBP")

# Add the base directory to sys.path if it’s not already there
if base_dir not in path:
    path.append(base_dir)

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Methods.Analysis import run_convergence, plot_conv
from concurrent.futures import ProcessPoolExecutor

# Simultation parameters
savefile = 'Euler2dVortex_CSBPp4' # .npz extension will be added automatically
op = 'lgl' # 'lg', 'lgl', 'csbp', 'hgtl', 'hgt', 'mattsson', 'upwind'
nelem = [3] # number of elements in each direction, as a list
nen = [20,40,80,160] # number of nodes per element in each direction, as a list
p = 4 # polynomial degree
vars2plot = ['rho', 'entropy','q','p'] # can be any of these 4
nthreads = 10 # number of threads for batch runs
include_upwind = True # include upwind operators as a reference
include_bothdiss = True # include both cons. and ent. volume dissipation
cases = ['upwind1', 'upwind02', 
         'div_nodiss', 'div_cons1_sca', 'div_cons02_sca', 'div_cons1_mat', 'div_cons02_mat',
         'had_cons02_sca', 'had_cons02_mat',
         'had_nodiss', 'had_ent1_scamat', 'had_ent02_scamat', 'had_ent1_matmat', 'had_ent02_matmat'] # cases to run

def read_errors(savefile, case, op, p, nen, nelem, label):
    target_header = f'COMPLETE: {case} op: {op} p: {p} nen: {nen} nelem: {nelem}'
    target_label = f'    {label}'
    errors = []

    with open(savefile, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines[:-2]):  # ensure room to access line + 2
        if line.strip() == target_header and lines[i + 1].strip() == target_label.strip():
            dof_line = lines[i + 2].strip()
            if dof_line.startswith('dof:'):
                try:
                    dof = int(float(dof_line.split(':')[1].strip()))
                except ValueError:
                    print('Could not parse DOF. Skipping Read.')
                    print(dof_line)
                    return None

            else:
                print('Expected "dof:" line not found. Skipping Read.')
                return None

            # Now read the next len(vars2plot) lines for error values
            for j in range(len(vars2plot)):
                line_idx = i + 3 + j
                if line_idx >= len(lines):
                    print('Unexpected end of file while reading errors. Skipping Read.')
                    return None
                err_line = lines[line_idx].strip()
                try:
                    err_val = float(err_line.split(':')[1].strip())
                    errors.append(err_val)
                except (IndexError, ValueError):
                    print(f'Error parsing line: {err_line}. Skipping Read.')
                    return None

            return errors, dof
    return None

def run_simulation(case, p, nelem, nen, savefile, op, 
                   disc_type, surf_diss, vol_diss, label):
    
    #if os.path.exists(savefile + '.npz'):
    #    savefile += '1'

    errors = []

    if os.path.exists(savefile):
        res = read_errors(savefile, case, op, p, nen, nelem, label)
    else:
        res = None

    if res is not None:
        print('---------------------------------------------------')
        print('READING:', case, 'op:', op, 'p:', p, 'nen:', nen, 'nelem:', nelem)
        print('   ', label)
        errors, dof = res[0], res[1]
        print('dof:', dof)
        for i,var in enumerate(vars2plot):
            print(var + ' error:', errors[i])
    else:
        print('---------------------------------------------------')
        print('RUNNING:', case, 'op:', op, 'p:', p, 'nen:', nen, 'nelem:', nelem)
        print('   ', label, flush=True)

        #Set up the differential equation and solver
        diffeq = Euler([287,1.4], 'vortex', 'vortex', 'periodic', False)
        settings = {'metric_method':'exact',
                    'use_optz_metrics':False}
        solver = PdeSolverSbp(diffeq, settings, 'rk8', 1e-3, 20.0,
                            p=p, disc_type=disc_type,   
                            surf_diss=surf_diss, vol_diss=vol_diss, had_flux='ranocha',
                            nelem=nelem, nen=nen, disc_nodes=op,
                            bc='periodic', xmin=(-5.,-5.), xmax=(5.,5.), cons_obj_name=None)
        solver.print_progress = False
        solver.tm_atol = 1e-12
        solver.tm_rtol = 1e-12 
        solver.tm_nframes = 0
        solver.keep_all_ts = False

        solver.solve()
        for var in vars2plot:
            errors.append(solver.calc_error(method='SBP', var2plot_name=var))
        dof = np.sqrt(solver.nn[0] * solver.nn[1])

        print('---------------------------------------------------')
        print('COMPLETE:', case, 'op:', op, 'p:', p, 'nen:', nen, 'nelem:', nelem)
        print('   ', label)
        print('dof:', dof)
        for i,var in enumerate(vars2plot):
            print(var + ' error:', errors[i])
        print('---------------------------------------------------', flush=True)

        with open(savefile, 'a') as f:
            print('---------------------------------------------------', file=f)
            print('COMPLETE:', case, 'op:', op, 'p:', p, 'nen:', nen, 'nelem:', nelem, file=f)
            print('   ', label, file=f)
            print('dof:', dof, file=f)
            for i,var in enumerate(vars2plot):
                print(var + ':', errors[i], file=f)

    return errors, dof, case, label


if __name__ == '__main__':
    futures = []
    with ProcessPoolExecutor(max_workers=nthreads) as executor:  
        for nen_ in nen:
            for nelem_ in nelem:
                for case in cases:

                    if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
                        s = p+1
                        eps = 3.125/5**s
                        use_H = False
                        bdy_fix = True
                        avg_nodes = True
                    elif op == 'lgl' or op == 'lg':
                        s = p
                        if p == 2: eps = 0.02
                        elif p == 3: eps = 0.01
                        elif p == 4: eps = 0.004
                        elif p == 5: eps = 0.002
                        elif p == 6: eps = 0.0008
                        elif p == 7: eps = 0.0004
                        elif p == 8: eps = 0.0002
                        else: raise Exception('No dissipation for this p')
                        use_H = False
                        bdy_fix = False
                        avg_nodes = False
                    elif op == 'upwind':
                        pass
                    else:
                        raise Exception('No dissipation for this operator')
                    
                    if 'had' in case: # Entropy-stable with no dissipation
                        run = True
                        disc_type = 'had'
                        surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
                                    'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
                        
                        if 'nodiss' in case:
                            vol_diss = {'diss_type':'nd'}
                            label = f'$\\varepsilon=0$'
                        elif 'ent1_scamat' in case: # Entropy-stable with large scalar-matrix dissipation
                            vol_diss = {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Ent. Sca.-Mat. $\\varepsilon={eps:g}$'
                        elif 'ent02_scamat' in case: # Entropy-stable with small scalar-matrix dissipation
                            vol_diss = {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Ent. Sca.-Mat. $\\varepsilon={0.2*eps:g}$'
                        elif 'ent1_matmat' in case: # Entropy-stable with large matrix-matrix dissipation
                            vol_diss = {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Ent. Mat.-Mat. $\\varepsilon={eps:g}$'
                        elif 'ent02_matmat' in case: # Entropy-stable with small matrix-matrix dissipation
                            vol_diss = {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Ent. Mat.-Mat. $\\varepsilon={0.2*eps:g}$'
                        elif 'cons1_sca' in case: # Divergence form with large scalar conservative dissipation
                            run = include_bothdiss
                            vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Sca. $\\varepsilon={eps:g}$'
                        elif 'cons1_mat' in case: # Divergence form with large matrix conservative dissipation
                            run = include_bothdiss
                            vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Mat. $\\varepsilon={eps:g}$'
                        elif 'cons02_sca' in case: # Divergence form with small scalar conservative dissipation
                            run = include_bothdiss
                            vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Sca. $\\varepsilon={0.2*eps:g}$'
                        elif 'cons02_mat' in case: # Divergence form with small matrix conservative dissipation
                            run = include_bothdiss
                            vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Mat. $\\varepsilon={0.2*eps:g}$'
                        else:
                            print('ERROR: Unknown case {}. Ignoring.'.format(case))
                            run = False 
                    elif 'upwind' in case: # Upwind dissipation
                        run = include_upwind
                        disc_type = 'div'
                        surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                    'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                        if op == 'lgl' or op == 'lg':
                            if 'upwind02' in case:
                                vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':0.2*eps}
                                label = f'USE $\\varepsilon={0.2*eps:g}$'
                            else:
                                vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':eps}
                                label = f'USE $\\varepsilon={eps:g}$'
                        else:
                            if 'upwind02' in case: run = False
                            vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}
                            label = f'UFD $p_\\text{{u}}={int(p)}$'
                        if op not in ['lgl', 'lg', 'upwind']: run = False
                    elif 'div' in case: # Divergence form with no dissipation
                        disc_type = 'div'
                        surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                    'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                        if 'nodiss' in case:
                            vol_diss = {'diss_type':'nd'}
                            label = f'$\\varepsilon=0$'
                        elif 'cons1_sca' in case: # Divergence form with large scalar conservative dissipation
                            vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Sca. $\\varepsilon={eps:g}$'
                        elif 'cons1_mat' in case: # Divergence form with large matrix conservative dissipation
                            vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Mat. $\\varepsilon={eps:g}$'
                        elif 'cons02_sca' in case: # Divergence form with small scalar conservative dissipation
                            vol_diss = {'diss_type':'dcp', 'jac_type':'sca', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Sca. $\\varepsilon={0.2*eps:g}$'
                        elif 'cons02_mat' in case: # Divergence form with small matrix conservative dissipation
                            vol_diss = {'diss_type':'dcp', 'jac_type':'mat', 's':s, 'coeff':0.2*eps,
                                        'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                            label = f'Cons. Mat. $\\varepsilon={0.2*eps:g}$'
                        else:
                            print('ERROR: Unknown case {}. Ignoring.'.format(case))
                            run = False 
                    else:
                        print('ERROR: Unknown case {}. Ignoring.'.format(case))
                        run = False
                    
                    if run:

                        future = executor.submit(run_simulation, case, p, nelem_, nen_, savefile, op, 
                                                                    disc_type, surf_diss, vol_diss, label)
                        futures.append(future)

    print()
    print('---------------------------------------------------')
    print('COMPLETE: All simulations submitted')
    print('---------------------------------------------------')
    print()
    if include_upwind:
        div_1_data = np.zeros((2, len(nelem)*len(nen), len(vars2plot))) # upwind: 2 cases (upwind1, upwind02)
        div_1_dofs = np.zeros((2, len(nelem)*len(nen)))
        div_1_labels = ['None','None']
    else:
        div_1_data = None
        div_1_dofs = None
        div_1_labels = []
    div_2_data = None
    div_2_dofs = None
    div_2_labels = []
    div_3_data = np.zeros((5, len(nelem)*len(nen), len(vars2plot))) # div: 5 cases (div_nodiss, div_cons1_sca, div_cons02_sca, div_cons1_mat, div_cons02_mat)
    div_3_dofs = np.zeros((5, len(nelem)*len(nen)))
    div_3_labels = ['None','None','None','None','None']

    had_1_data = None
    had_1_dofs = None
    had_1_labels = []
    if include_bothdiss:
        had_2_data = np.zeros((2, len(nelem)*len(nen), len(vars2plot))) # div dissipation: 2 cases (had_cons02_sca, had_cons02_mat)
        had_2_dofs = np.zeros((2, len(nelem)*len(nen)))
        had_2_labels = ['None','None']
    else:
        had_2_data = None
        had_2_dofs = None
        had_2_labels = []
    had_3_data = np.zeros((5, len(nelem)*len(nen), len(vars2plot))) # had: 5 cases (had_nodiss, had_ent1_scamat, had_ent02_scamat, had_ent1_matmat, had_ent02_matmat)
    had_3_dofs = np.zeros((5, len(nelem)*len(nen)))
    had_3_labels = ['None','None','None','None','None']

    if op in ['lgl', 'lg']:
        runs = np.array(nelem)*(p+1)
    else:
        mesh = np.meshgrid(nelem,nen)
        runs = (mesh[0] * mesh[1]).flatten()

    for future in futures:
        try:
            errors, dof, case, label = future.result()
            runidx = int(np.where(abs(runs - dof)<1e-10)[0][0])

            if case == 'upwind1':
                div_1_data[0,runidx] = errors
                div_1_dofs[0,runidx] = dof
                div_1_labels[0] = label
            elif case == 'upwind02':
                div_1_data[1,runidx] = errors
                div_1_dofs[1,runidx] = dof
                div_1_labels[1] = label
            elif case == 'div_nodiss':
                div_3_data[0,runidx] = errors
                div_3_dofs[0,runidx] = dof
                div_3_labels[0] = label
            elif case == 'div_cons1_sca':
                div_3_data[1,runidx] = errors
                div_3_dofs[1,runidx] = dof
                div_3_labels[1] = label
            elif case == 'div_cons02_sca':
                div_3_data[2,runidx] = errors
                div_3_dofs[2,runidx] = dof
                div_3_labels[2] = label
            elif case == 'div_cons1_mat':
                div_3_data[3,runidx] = errors
                div_3_dofs[3,runidx] = dof
                div_3_labels[3] = label
            elif case == 'div_cons02_mat':
                div_3_data[4,runidx] = errors
                div_3_dofs[4,runidx] = dof
                div_3_labels[4] = label
            elif case == 'had_nodiss':
                had_3_data[0,runidx] = errors
                had_3_dofs[0,runidx] = dof
                had_3_labels[0] = label
            elif case == 'had_ent1_scamat':
                had_3_data[1,runidx] = errors
                had_3_dofs[1,runidx] = dof
                had_3_labels[1] = label
            elif case == 'had_ent02_scamat':
                had_3_data[2,runidx] = errors
                had_3_dofs[2,runidx] = dof
                had_3_labels[2] = label
            elif case == 'had_ent1_matmat':
                had_3_data[3,runidx] = errors
                had_3_dofs[3,runidx] = dof
                had_3_labels[3] = label
            elif case == 'had_ent02_matmat':
                had_3_data[4,runidx] = errors
                had_3_dofs[4,runidx] = dof
                had_3_labels[4] = label
            elif case == 'had_cons02_sca':
                had_2_data[0,runidx] = errors
                had_2_dofs[0,runidx] = dof
                had_2_labels[0] = label
            elif case == 'had_cons02_mat':
                had_2_data[1,runidx] = errors
                had_2_dofs[1,runidx] = dof
                had_2_labels[1] = label

        except Exception as e:
            print(f"Simulation FAILED with error: {e}")

    print('Saving simulation results to data file...')
    np.savez(savefile + 'Div.npz', dofs1=div_1_dofs, errors1=div_1_data, labels1=div_1_labels,
                                    dofs2=div_2_dofs, errors2=div_2_data, labels2=div_2_labels,
                                    dofs3=div_3_dofs, errors3=div_3_data, labels3=div_3_labels, allow_pickle=True)
    np.savez(savefile + 'Had.npz', dofs1=had_1_dofs, errors1=had_1_data, labels1=had_1_labels,
                                    dofs2=had_2_dofs, errors2=had_2_data, labels2=had_2_labels,
                                    dofs3=had_3_dofs, errors3=had_3_data, labels3=had_3_labels, allow_pickle=True)