import os
from sys import path
import numpy as np
from tabulate import tabulate

# Define the base path to ESSBP (assuming it’s always under the home directory)
#base_dir = os.path.join(os.path.expanduser("~"), "ESSBP")
base_dir = os.path.join(os.path.expanduser("~"), "Desktop/UTIAS/ESSBP")

# Add the base directory to sys.path if it’s not already there
if base_dir not in path:
    path.append(base_dir)

from Source.DiffEq.Euler2d import Euler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from concurrent.futures import ProcessPoolExecutor

# Simultation parameters
savefile_dir = 'KelvinHelmholtz_Results' # output directory for reading / saving results (name is created automatically)
# the savefile out is created automatically
tf = 15.0 # final time.
op = 'lg' # 'lg', 'lgl', 'csbp', 'hgtl', 'hgt', 'mattsson', 'upwind'
nelem = [4,8,16,32] # number of elements in each direction, as a list
nen = [0] #[20,40,80,160] #[30,60,120,240,480] # number of nodes per element in each direction, as a list. Use [0] for LG/LGL.
p = [3,4,5,6,7,8] # polynomial degree, as a list
#s = p+1 # dissipation degree. trad: p+1, elem: p - NOTE: Overwritten below.
had_flux = 'ranocha' # 2-point numerical flux used in hadamard form
rk8_atol = 1e-7 # absolute tolerance for RK8
rk8_rtol = 1e-7 # relative tolerance for RK8
tm_nframes = 300 # number of solution snapshots to save simulation
cons_obj_name=('time','entropy') # note: what to track? make sure to include 'time'

nthreads = 5 # number of threads for batch runs
include_upwind = True # include upwind operators as a reference (only relevant for lg and lgl)
include_bothdiss = False # include both cons. and ent. volume dissipation on entropy-stable schemes?
include_energy = True # include energy-stable schemes?
loaddata = False # skip the actual simulation and just try to load and display the data
use_scalar_sat = False # use a scalar rusanov sat instead of the matrix dissipation
verbose_output = False 
nondimensionalize = False
skip_if_exists = False # skip the simulation if the savefile already exists. Otherwise will retry with either rk4 OR larger rk8 tolerance.
minimum_density = 1e-3 # minimum density for the simulation to be considered a crash
rerun_with_rk4 = True # rerun with rk4 if the simulation fails with rk8. If False, will use rk8 with larger tolerance.

# Problem parameters
tm_method = 'rk8'
para = [287,1.4] # [R, gamma]
test_case = 'kelvin-helmholtz'
q0_type = 'kelvin-helmholtz'
xmin = (-1.,-1.)
xmax = (1.,1.)
bc = 'periodic'
settings = {'metric_method':'exact',
            'use_optz_metrics':False} # extra things like for metrics, mesh warping, etc.



def check_run(savefile_base, extension='.npz'):
    ''' Returns the final time, and exit code 0,1,2 for success, density crash, or error. '''
    ''' If exit code is 1 or 2, it also returns the minimum density.'''
    data = np.load(savefile_base + extension, allow_pickle=True)
    cons_obj = data['cons_obj']
    cons_obj_name = data['cons_obj_name']
    
    # Find the index for 'time'
    time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
    final_time = cons_obj[time_idx, -1]

    if abs(final_time - tf) < 1e-10:
        # SUCCESS: reached final time
        return final_time, 0, None
    
    else:
        rho = data['q_sol'][::4,:,-1]
        minrho = np.min(rho)
        if np.min(rho) < minimum_density:
            # DENSITY CRASH
            return final_time, 1, minrho
        else:
            # ERROR: did not reach final time, but no density crash
            return final_time, 2, minrho
    
# Set up arrays for storing results
extension = ".npz"
nruns = len(nen)*len(nelem)
if op == 'upwind':
    cases = ['upwind']
else:
    cases = ['had_nodiss',
             'had_ent1_matmat','had_ent02_matmat',
             'had_ent1_scamat','had_ent02_scamat',
             'had_cons02_sca','had_cons02_mat', 
             'upwind','upwind02', 
             'div_nodiss', 
             'div_cons1_mat','div_cons02_mat',
             'div_cons1_sca','div_cons02_sca']
ncases = len(cases)

if use_scalar_sat: cases = [case + '_lfsat' for case in cases]

def run_simulation(case, p, nelem, nen, savefile_dir, op, para, q0_type, test_case, bc, nondimensionalize,
                   settings, tm_method, tf, disc_type, surf_diss, vol_diss, had_flux, xmin, xmax, cons_obj_name,
                   rk8_atol, rk8_rtol, tm_nframes, verbose_output, skip_if_exists):
    ''' Runs the simulation, saves the results, and returns:
        - savefile_base: the base name of the savefile
        - success: True if the simulation was successful or had a density crash
        - final_time: the final time of the simulation
        - final_time2: the final time of the second run with either rk4 or larger rk8 tolerances (if applicable)
    '''

    # Build the savefile name
    if nondimensionalize and 'nondim' not in case:
        case = case + '_nondim'
    if nen == 0:
        savefile_base = os.path.join(savefile_dir, f"{op}_p{p}_nelem{nelem}_{case}")
    else:
        savefile_base = os.path.join(savefile_dir, f"{op}_p{p}_nen{nen}_nelem{nelem}_{case}")

    # Loop to find a filename that doesn't exist yet
    savefile = savefile_base + extension
    if os.path.exists(savefile):

        if skip_if_exists:
            print(f"File {savefile} already exists. Skipping simulation.")
            return savefile_base, True, 'skip', None
        
        else:
            print(f"File {savefile} already exists. Checking result...")
            # check if previous run completed succesfully or crashed "sucessfuly",
            # otherwise rerun with a larger rk8 tolerance, because likely was too stiff

            final_time, exit_code, minrho = check_run(savefile_base, extension)

            if exit_code == 0:
                print(f"... {savefile_base} ran to completion. Skipping.")
                return savefile_base, True, final_time, None
            
            elif exit_code == 1:
                print(f"... {savefile_base} CRASH due to minimum density =", minrho)
                return savefile_base, True, final_time, None
            
            else:
                print(f'... ERROR: {savefile_base} only reached minimum density =', minrho)
                if rerun_with_rk4:
                    savefile_base2 = savefile_base + '_rk4'
                else:
                    savefile_base2 = savefile_base + '_largedt'
                savefile2 = savefile_base2 + extension
                
                if os.path.exists(savefile2):
                    print(f"... however there exists a file with larger rk8 tolerances. Checking result...")
                    final_time2, exit_code, minrho = check_run(savefile_base2, extension)

                    if exit_code == 0:
                        print(f"... {savefile_base2} ran to completion.")
                        return savefile_base, True, final_time, final_time2
                    
                    elif exit_code == 1:
                        print(f"... {savefile_base2} CRASH due to minimum density =", minrho)
                        return savefile_base, True, final_time, final_time2
                    
                    else:
                        print(f"... ERROR: {savefile_base2} only reached minimum density =", minrho)
                        print('... **** Something went wrong (system is probably too stiff) **** ')
                        return savefile_base, False, final_time, final_time2
            
                else:
                    if rerun_with_rk4:
                        print(f"... RERUNNING {savefile_base} with rk4 and CFL ~ 0.1")
                        tol_fac = 0
                        tm_method2 = 'rk4'
                        if nen == 0:
                            dt2 = (0.03) * (2 / (nelem * p))
                        else:
                            dt2 = (0.03) * (2 / (nelem * (nen-1)))
                    else:
                        print(f"... RERUNNING {savefile_base} with larger rk8 tolerances by factor of", tol_fac)
                        tol_fac = 100
                        tm_method2 = tm_method
                        dt2 = 0.0001
                    try:
                        print('RERUNNING:', case, 'p:', p, 'nen:', nen, 'nelem:', nelem)

                        # Set up the differential equation and solver
                        diffeq = Euler(para, q0_type, test_case, bc, nondimensionalize)
                        solver = PdeSolverSbp(diffeq, settings, tm_method2, dt2, tf,
                                            p=p, disc_type=disc_type,   
                                            surf_diss=surf_diss, vol_diss=vol_diss, had_flux=had_flux,
                                            nelem=nelem, nen=nen, disc_nodes=op,
                                            bc='periodic', xmin=xmin, xmax=xmax, cons_obj_name=cons_obj_name)
                        solver.print_progress = verbose_output
                        solver.tm_atol = rk8_atol * tol_fac # increase rk8 tolerances
                        solver.tm_rtol = rk8_rtol * tol_fac
                        solver.tm_nframes = tm_nframes

                        solver.solve()
                        minrho = np.min(solver.q_sol[::4,:,-1])

                        print('---------------------------------------------------')
                        print('COMPLETE:', case, 'p:', p, 'nen:', nen, 'nelem:', nelem)
                        print('Final time:', solver.t_final)
                        print('Final minimum density:', minrho)
                        print('saving simulation results to ', savefile2)
                        print('---------------------------------------------------')

                        save_settings = {'settings': settings, 'tm_method': tm_method2,
                                        'p': p, 'disc_type': disc_type, 'had_flux': had_flux,
                                        'surf_diss': surf_diss, 'vol_diss': vol_diss,
                                        'nelem': nelem, 'nen': nen, 'op': op,
                                        'rk8_atol': rk8_atol * tol_fac, 'rk8_rtol': rk8_rtol * tol_fac,
                                        'nondimensionalize': nondimensionalize}

                        np.savez(savefile2, q_sol=solver.q_sol, 
                                cons_obj=solver.cons_obj, cons_obj_name=solver.cons_obj_name,
                                save_settings=save_settings, allow_pickle=True)
                        
                        # return the savefile base name and the final time
                        if abs(solver.t_final - tf) < 1e-10:
                            # SUCCESS: reached final time
                            return savefile_base, True, final_time, solver.t_final
                        
                        else:
                            if minrho < minimum_density:
                                return savefile_base, True, final_time, solver.t_final
                            else:
                                return savefile_base, False, final_time, solver.t_final
                    
                    except Exception as e:
                        print('ERROR:', case, 'nen:', nen, 'nelem:', nelem)
                        print('Simulation failed with error:', e)
                        return savefile_base, False, final_time, 'N/A'
    
    else:
        try:
            print('RUNNING:', case, 'p:', p, 'nen:', nen, 'nelem:', nelem)

            # Set up the differential equation and solver
            diffeq = Euler(para, q0_type, test_case, bc, nondimensionalize)
            solver = PdeSolverSbp(diffeq, settings, tm_method, 0.0001, tf,
                                p=p, disc_type=disc_type,   
                                surf_diss=surf_diss, vol_diss=vol_diss, had_flux=had_flux,
                                nelem=nelem, nen=nen, disc_nodes=op,
                                bc='periodic', xmin=xmin, xmax=xmax, cons_obj_name=cons_obj_name)
            solver.print_progress = verbose_output
            solver.tm_atol = rk8_atol
            solver.tm_rtol = rk8_rtol
            solver.tm_nframes = tm_nframes

            solver.solve()
            minrho = np.min(solver.q_sol[::4,:,-1])

            print('---------------------------------------------------')
            print('COMPLETE:', case, 'p:', p, 'nen:', nen, 'nelem:', nelem)
            print('Final time:', solver.t_final)
            print('Final minimum density:', minrho)
            print('saving simulation results to ', savefile)
            print('---------------------------------------------------')

            save_settings = {'settings': settings, 'tm_method': tm_method,
                            'p': p, 'disc_type': disc_type, 'had_flux': had_flux,
                            'surf_diss': surf_diss, 'vol_diss': vol_diss,
                            'nelem': nelem, 'nen': nen, 'op': op,
                            'rk8_atol': rk8_atol, 'rk8_rtol': rk8_rtol,
                            'nondimensionalize': nondimensionalize}

            np.savez(savefile, q_sol=solver.q_sol, 
                    cons_obj=solver.cons_obj, cons_obj_name=solver.cons_obj_name,
                    save_settings=save_settings, allow_pickle=True)
                        
            # return the savefile base name and the final time
            if abs(solver.t_final - tf) < 1e-10:
                # SUCCESS: reached final time
                return savefile_base, True, solver.t_final, None
            
            else:
                if minrho < minimum_density:
                    return savefile_base, True, solver.t_final, None
                else:
                    return savefile_base, False, solver.t_final, None
        
        except Exception as e:
            print('ERROR:', case, 'nen:', nen, 'nelem:', nelem)
            print('Simulation failed with error:', e)
            return savefile_base, False, 'N/A'

if __name__ == '__main__':
    if loaddata:
        for case in cases:
            # Gather final times for all combinations in a dictionary keyed by (p, nen, nelem)
            final_times = {}
            if nondimensionalize and 'nondim' not in case:
                case = case + '_nondim'
            for p_ in p:
                for nen_ in nen:
                    for nelem_ in nelem:
                        if nen_ == 0:
                            # When nen is 0, we don't include it in the filename.
                            savefile_base = os.path.join(savefile_dir, f"{op}_p{p_}_nelem{nelem_}_{case}")
                            savefile = savefile_base + extension
                        else:
                            savefile_base = os.path.join(savefile_dir, f"{op}_p{p_}_nen{nen_}_nelem{nelem_}_{case}")
                            savefile = savefile_base + extension
                        
                        if os.path.exists(savefile):

                            final_time1, exit_code, minrho = check_run(savefile_base, extension)

                            if exit_code == 0:
                                #final_time = round(final_time1,2)
                                final_time = f"{final_time1:.2f}"
                            
                            elif exit_code == 1:
                                print(f"{savefile_base} CRASH due to minimum density =", minrho)
                                final_time = f"{final_time1:.2f}"
                            
                            else:
                                print(f"{savefile_base} FAIL, only reached minimum density =", minrho)
                                if rerun_with_rk4:
                                    savefile_base2 = savefile_base + '_rk4'
                                else:
                                    savefile_base2 = savefile_base + '_largedt'
                                savefile2 = savefile_base2 + extension
                                
                                if os.path.exists(savefile2):

                                    final_time2, exit_code, minrho = check_run(savefile_base2, extension)

                                    if exit_code == 0:
                                        final_time = f"{final_time1:.2f}({final_time2:.2f})"
                                    
                                    elif exit_code == 1:
                                        print(f"{savefile_base2} CRASH due to minimum density =", minrho)
                                        final_time = f"{final_time1:.2f}({final_time2:.2f})"
                                    
                                    else:
                                        print(f"{savefile_base2} FAIL, only reached minimum density =", minrho)
                                        final_time = f"{final_time1:.2f}({final_time2:.2f})(F)"
                            
                                else:
                                    print("... consider rerunning with either rk4 or larger rk8 tolerances.")
                                    final_time = f"{final_time1:.2f}(F)"

                        else:
                            print('ERROR: File {} not found. Ignoring.'.format(savefile))
                            final_time = 'N/A'
                        
                        final_times[(p_, nen_, nelem_)] = final_time

            # --- Decide on the table layout based on the parameters ---
            # Case 1: nen is [0]: use rows = nelem and columns = p.
            if nen == [0]:
                headers = [f"p={p_}" for p_ in p]
                table_data = []
                for nelem_ in nelem:
                    row = []
                    for p_ in p:
                        row.append(final_times[(p_, 0, nelem_)])
                    table_data.append([f"nelem={nelem_}"] + row)
                print(f"\n Final times: {case}")
                print(tabulate(table_data, headers=["nelem / p"] + headers, tablefmt="github", floatfmt=".2f"))

            # Case 2: nen is not just [0]
            # use nen as the rows
            else:
                # If both nelem and p vary:
                if len(nelem) > 1 and len(p) > 1:
                    # Use rows = nen, columns = (nelem, p) combinations.
                    headers = []
                    # Order: for each nelem, then for each p.
                    for ne in nelem:
                        for p_ in p:
                            headers.append(f"nelem={ne}, p={p_}")
                    table_data = []
                    for nen_ in nen:
                        row = []
                        for ne in nelem:
                            for p_ in p:
                                row.append(final_times[(p_, nen_, ne)])
                        table_data.append([f"nen={nen_}"] + row)
                    print(f"\n Final times: {case}")
                    print(tabulate(table_data, headers=["nen"] + headers, tablefmt="github", floatfmt=".2f"))

                # If only nelem varies (p is singular):
                elif len(nelem) > 1:
                    fixed_p = p[0]
                    headers = [f"nen={nen_}" for nen_ in nen]
                    table_data = []
                    for ne in nelem:
                        row = []
                        for nen_ in nen:
                            row.append(final_times[(fixed_p, nen_, ne)])
                        table_data.append([f"nelem={ne}"] + row)
                    print(f"\n Final times: p={p_} {case}")
                    print(tabulate(table_data, headers=["nelem / nen"] + headers, tablefmt="github", floatfmt=".2f"))

                # If only p varies (nelem is singular):
                elif len(p) > 1:
                    fixed_nelem = nelem[0]
                    headers = [f"p={p_}" for p_ in p]
                    table_data = []
                    for nen_ in nen:
                        row = []
                        for p_ in p:
                            row.append(final_times[(p_, nen_, fixed_nelem)])
                        table_data.append([f"nen={nen_}"] + row)
                    print(f"\n Final times: nelem={nelem_} {case}")
                    print(tabulate(table_data, headers=["nen / p"] + headers, tablefmt="github", floatfmt=".2f"))

                elif len(nen) > 1 and len(p) == 1 and len(nelem) == 1:
                    fixed_p = p[0]
                    fixed_nelem = nelem[0]
                    # Use a single row with columns for each nen value.
                    headers = [f"nen={nen_}" for nen_ in nen]
                    # Create a one-row table with the fixed parameter combination.
                    row = [final_times[(fixed_p, nen_, fixed_nelem)] for nen_ in nen]
                    table_data = [["p=" + str(fixed_p) + ", nelem=" + str(fixed_nelem)] + row]
                    print(f"\nFinal times: {case}")
                    print(tabulate(table_data, headers=["p/nelem"] + headers, tablefmt="github", floatfmt=".2f"))

                # Otherwise, all parameters are singular.
                else:
                    fixed_p = p[0]
                    fixed_nen = nen[0]
                    fixed_nelem = nelem[0]
                    table_data = [[final_times[(fixed_p, fixed_nen, fixed_nelem)]]]
                    header = [f"p={fixed_p}, nen={fixed_nen}, nelem={fixed_nelem}"]
                    print(f"\n{case}")
                    print(tabulate(table_data, headers=header, tablefmt="github", floatfmt=".2f"))
                
                print('')


    else:
        futures = []
        with ProcessPoolExecutor(max_workers=nthreads) as executor:  
            for case in cases:
                for p_ in p:
                    for nen_ in nen:
                        for nelem_ in nelem:

                            if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
                                s = p_+1
                                eps = 3.125/5**s
                                use_H = False
                                bdy_fix = True
                                avg_nodes = True
                            elif op == 'lgl' or op == 'lg':
                                s = p_
                                if p_ == 2: eps = 0.02
                                elif p_ == 3: eps = 0.01
                                elif p_ == 4: eps = 0.004
                                elif p_ == 5: eps = 0.002
                                elif p_ == 6: eps = 0.0008
                                elif p_ == 7: eps = 0.0004
                                elif p_ == 8: eps = 0.0002
                                else: raise Exception('No dissipation for this p')
                                use_H = False
                                bdy_fix = False
                                avg_nodes = False
                            elif op == 'upwind':
                                pass
                            else:
                                raise Exception('No dissipation for this operator')
                            
                            if 'had_nodiss' in case: # Entropy-stable with no dissipation
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'scamat', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':True, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
                                                'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
                                vol_diss = {'diss_type':'nd'}
                                disc_type = 'had'
                                run = True
                            elif 'had_ent' in case:
                                run = True
                                disc_type = 'had'
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'scamat', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':True, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
                                                'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
                                if 'ent1_scamat' in case: # Entropy-stable with large scalar-matrix dissipation
                                    vol_diss = {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'coeff':eps,
                                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                                elif 'ent02_scamat' in case: # Entropy-stable with small scalar-matrix dissipation
                                    vol_diss = {'diss_type':'entdcp', 'jac_type':'scamat', 's':s, 'coeff':0.2*eps,
                                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                                elif 'ent1_matmat' in case: # Entropy-stable with large matrix-matrix dissipation
                                    vol_diss = {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'coeff':eps,
                                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                                elif 'ent02_matmat' in case: # Entropy-stable with small matrix-matrix dissipation
                                    vol_diss = {'diss_type':'entdcp', 'jac_type':'matmat', 's':s, 'coeff':0.2*eps,
                                            'bdy_fix':bdy_fix, 'use_H':use_H, 'entropy_fix':False, 'avg_half_nodes':avg_nodes}
                                else:
                                    print('ERROR: Unknown case {}. Ignoring.'.format(case))
                                    run = False 
                            elif 'had_cons' in case:
                                run = include_bothdiss
                                disc_type = 'had'
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'scamat', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':True, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1., 'average':'none', 
                                                'entropy_fix':False, 'P_derigs':True, 'A_derigs':True, 'maxeig':'none'}
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
                                    print('ERROR: Unknown case {}. Ignoring.'.format(case))
                                    run = False
                            elif 'upwind' in case: # Upwind dissipation
                                run = include_upwind
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'sca', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                                'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                                if op == 'lgl' or op == 'lg':
                                    if 'upwind02' in case:
                                        vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':0.2*eps}
                                    else:
                                        vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':eps}
                                else:
                                    if 'upwind02' in case: run = False
                                    vol_diss = {'diss_type':'upwind', 'fluxvec':'dt', 'coeff':1.}
                                disc_type = 'div'
                                if op not in ['lgl', 'lg', 'upwind']: run = False
                            elif 'div_nodiss' in case: # Divergence form with no dissipation
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'sca', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                                'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                                vol_diss = {'diss_type':'nd'}
                                disc_type = 'div'
                                run = include_energy
                            elif 'div_cons' in case:
                                run = include_energy
                                disc_type = 'div'
                                if '_lfsat' in case:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'sca', 'coeff':1., 'average':'none', 
                                            'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'rusanov'}
                                else:
                                    surf_diss = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                                'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                                
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
                                    print('ERROR: Unknown case {}. Ignoring.'.format(case))
                                    run = False 
                            else:
                                print('ERROR: Unknown case {}. Ignoring.'.format(case))
                                run = False

                            if nondimensionalize: case = case + '_nondim'
                            
                            if run:

                                future = executor.submit(run_simulation, case, p_, nelem_, nen_, savefile_dir, op,
                                                para, q0_type, test_case, bc, nondimensionalize,
                                                settings, tm_method, tf, disc_type, surf_diss, vol_diss, had_flux,
                                                xmin, xmax, cons_obj_name, rk8_atol, rk8_rtol, tm_nframes,
                                                verbose_output, skip_if_exists)
                                futures.append(future)

        print()
        print('---------------------------------------------------')
        print('COMPLETE: All simulations submitted')
        print('---------------------------------------------------')
        print()
        for future in futures:
            try:
                savefile_base, success, final_time, final_time2 = future.result()

                if final_time2 is None:
                    if success:
                        print(f"Simulation completed: {savefile_base}, Final time: {final_time}")
                    else:
                        print(f"Simulation FAILED: {savefile_base}, Final time: {final_time}")
                        print("... Consider rerun with larger rk8 tolerances?")
                elif final_time2 is not None:
                    if rerun_with_rk4: suffix = '_rk4'
                    else: suffix = '_largedt'
                    if success:
                        print(f"Simulation completed: {savefile_base}, Final time: {final_time}")
                        print(f"Second run completed: {savefile_base + suffix}, Final time: {final_time2}")
                    else:
                        print(f"Simulation FAILED: {savefile_base}, Final time: {final_time}")
                        print(f"Second run FAILED: {savefile_base + suffix}, Final time: {final_time2}")

            except Exception as e:
                print(f"Simulation FAILED with error: {e}")




# ############################################################
# # if we want to plot the entropy dissipation
# include_upwind = True
# plot_markers = False
# savefile = None
# savefile_dir = 'KelvinHelmholtz_Results' # where to read from
# op = 'csbp'
# nelem = '1'
# nen = '240'
# p = '4'
# if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
#     eps = 3.125/5**(int(p)+1)
# elif op == 'lgl' or op == 'lg':
#     if p == '2': eps = 0.02
#     elif p == '3': eps = 0.01
#     elif p == '4': eps = 0.004
#     elif p == '5': eps = 0.002
#     elif p == '6': eps = 0.0008
#     elif p == '7': eps = 0.0004
#     elif p == '8': eps = 0.0002
# cases = ['had_nodiss', 'had_ent1_scamat', 'had_ent02_scamat', 'had_ent1_matmat', 'had_ent02_matmat']
# labels = [f'$\\varepsilon=0$', f'Ent. Sca.-Mat. $\\varepsilon={eps:g}$', f'Ent. Sca.-Mat. $\\varepsilon={0.2*eps:g}$',
#           f'Ent. Mat.-Mat. $\\varepsilon={eps:g}$', f'Ent. Mat.-Mat. $\\varepsilon={0.2*eps:g}$']
# linewidth=2
# #colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:brown']
# colors = ['tab:red', 'tab:orange', 'tab:blue', 'darkgoldenrod', 'k',  'm', 'tab:brown']

# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
# plt.rcParams['font.family'] = 'serif'
# if include_upwind and op == 'csbp':
#     data = np.zeros((len(cases)+2, 2, 301))
#     idx_start = 2
#     colors = ['tab:red', 'tab:orange', 'darkgoldenrod', 'k',  'm', 'tab:green', 'tab:blue']
#     markers = ['o', '^', 'v', 'x', '+', 'd', 's']
#     marker_start = [0, 0, 0.5, 1, 1, 1.5, 1.5]
#     linestyles = ['-','--',':','-','-','--','--']

#     file = savefile_dir + '/upwind_p' + str(2*int(p)) + '_nen' + nen + '_nelem' + nelem + '_upwind.npz'
#     try:
#         file_data = np.load(file, allow_pickle=True)
#         cons_obj = file_data['cons_obj']
#         cons_obj_name = file_data['cons_obj_name']
#         time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
#         entropy_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'entropy'][0]
#         tot_idx = len(cons_obj[time_idx, :])
#         data[0, 0, :tot_idx] = cons_obj[time_idx, :]
#         data[0, 1, :tot_idx] = cons_obj[entropy_idx, :]
#         data[0, :, tot_idx:] = np.nan
#     except:
#         print('ERROR: File {} not found. Ignoring.'.format(file))
#         data[0, :, :] = np.nan
    
#     try:
#         file = savefile_dir + '/upwind_p' + str(2*int(p)+1) + '_nen' + nen + '_nelem' + nelem + '_upwind.npz'
#         file_data = np.load(file, allow_pickle=True)
#         cons_obj = file_data['cons_obj']
#         cons_obj_name = file_data['cons_obj_name']
#         time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
#         entropy_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'entropy'][0]
#         tot_idx = len(cons_obj[time_idx, :])
#         data[1, 0, :tot_idx] = cons_obj[time_idx, :]
#         data[1, 1, :tot_idx] = cons_obj[entropy_idx, :]
#         data[1, :, tot_idx:] = np.nan
#     except:
#         print('ERROR: File {} not found. Ignoring.'.format(file))
#         data[1, :, :] = np.nan

#     labels = [f'UFD $p_u = {2*int(p)}$', f'UFD $p_u = {2*int(p)+1}$'] + labels

# elif include_upwind and op in ['lgl', 'lg']:
#     data = np.zeros((len(cases)+1, 2, 300))
#     idx_start = 1
#     colors = ['tab:red', 'darkgoldenrod', 'k',  'm', 'tab:green', 'tab:blue']
#     markers = ['o', 'v', 'x', '+', 'd', 's']
#     marker_start = [0.01, 0.015, 0.0, 0.005, 0.01, 0.01]
#     linestyles = ['-',':','-','-','--','--']

#     try:
#         file = savefile_dir + '/' + op + '_p' + p + '_nen' + nen + '_nelem' + nelem + '_upwind.npz'
#         file_data = np.load(file, allow_pickle=True)
#         cons_obj = file_data['cons_obj']
#         cons_obj_name = file_data['cons_obj_name']
#         time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
#         entropy_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'entropy'][0]
#         tot_idx = len(cons_obj[time_idx, :])
#         data[0, 0, :tot_idx] = cons_obj[time_idx, :]
#         data[0, 1, :tot_idx] = cons_obj[entropy_idx, :]
#         data[0, :, tot_idx:] = np.nan
#     except:
#         print('ERROR: File {} not found. Ignoring.'.format(file))
#         data[0, :, :] = np.nan

#     labels = [f'USE $p_u = {int(p)-1}$'] + labels
# else:
#     data = np.zeros((len(cases), 2, 300))
#     idx_start = 0
#     colors = ['darkgoldenrod', 'k',  'm', 'tab:green', 'tab:blue']
#     markers = ['v', 'x', '+', 'd', 's']
#     marker_start = [0.01, 0.0, 0.005, 0.01, 0.01]
#     linestyles = [':','-','-','--','--']

# for idx in range(len(cases)):
#     try:
#         file = savefile_dir + '/' + op + '_p' + p + '_nen' + nen + '_nelem' + nelem + '_' + cases[idx] + '.npz'
#         file_data = np.load(file, allow_pickle=True)
#         cons_obj = file_data['cons_obj']
#         cons_obj_name = file_data['cons_obj_name']
#         time_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'time'][0]
#         entropy_idx = [j for j in range(len(cons_obj_name)) if cons_obj_name[j].lower() == 'entropy'][0]
#         tot_idx = len(cons_obj[time_idx, :])
#         data[idx + idx_start, 0, :tot_idx] = cons_obj[time_idx, :]
#         data[idx + idx_start, 1, :tot_idx] = cons_obj[entropy_idx, :]
#         data[idx + idx_start, :, tot_idx:] = np.nan
#     except:
#         print('ERROR: File {} not found. Ignoring.'.format(file))
#         data[idx + idx_start, :, :] = np.nan

# if plot_markers:
#     n_markers = 8 # Number of markers per line
# else:
#     markers = ['','','','','','','']
#     marker_start = [0,0,0,0,0,0,0] # where to start markers (in time units)
#     nmarkers = 0

# plt.figure(figsize=(5,4.5))

# plt.ylabel(r'Entropy Change $\boldsymbol{1}^\mathsf{T} \mathsf{H} \left( \mathcal{S} \left(\boldsymbol{u}\right) -  \mathcal{S} \left(\boldsymbol{u}_0\right) \right) $',fontsize=16)
# plt.xlabel(r'Time $t$',fontsize=16)
# plt.yscale('symlog',linthresh=1e-3)
# #plt.ylim(-1e-3, 1e-8)
# plt.xlim(-0.5, 15.5)
# legend_loc = 'best'
# legend_anchor = None
# #legend_loc = 'upper center'
# #legend_anchor = (0.55,0.895) 

# for casei in range(len(labels)):
#     entropy = data[casei,1,:] - data[casei,1,0] 
#     time = data[casei,0] 
    
#     # Define the spacing interval
#     marker_spacing = 15 / (n_markers - 1)
#     # Generate marker positions before shifting
#     marker_positions = np.arange(n_markers) * marker_spacing
#     # Apply the offset
#     marker_positions += marker_start[casei]
#     # Remove markers beyond the last time point
#     marker_positions = marker_positions[marker_positions <= np.max(time)]
#     # Get corresponding indices
#     marker_indices = np.searchsorted(time, marker_positions)
            
#     plt.plot(time, entropy, color=colors[casei], linestyle=linestyles[casei], 
#             marker=markers[casei], markevery=marker_indices, label=labels[casei], linewidth=linewidth, 
#             markersize=9, markerfacecolor='none', markeredgewidth=linewidth,zorder=2)
    
#     # again so that markers are ontop
#     if plot_markers:
#         plt.plot(time, entropy, color=colors[casei], linestyle='', 
#             marker=markers[casei], markevery=marker_indices, label=None, linewidth=None, 
#             markersize=8, markerfacecolor='none', markeredgewidth=linewidth,zorder=3)
    
# plt.legend(loc=legend_loc,fontsize=14, 
#                 bbox_to_anchor=legend_anchor)
# ax = plt.gca()
# ax.tick_params(axis='both', which='both', labelsize=13) 
# plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')

# #positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
# #negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
# #custom_ticks = negative_ticks[::-1] + positive_ticks
# #ax.set_yticks(custom_ticks)

# plt.tight_layout()
# if savefile is not None:
#     plt.savefig(savefile,dpi=600)
