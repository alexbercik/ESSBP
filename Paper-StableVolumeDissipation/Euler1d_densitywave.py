import os
import time
from sys import path
n_nested_folder = 1
folder_path, _ = os.path.split(__file__)
for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)
path.append(folder_path)

from Source.DiffEq.Quasi1dEuler import Quasi1dEuler
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

''' Set default parameters for simultation '''
run_sims = False # actually run all the sims? Or just print the figures?
check_eigs = False # check the eigenvalues of each discretization?
run_sims = False # actually run all the sims? Or just print the figures?
check_eigs = False # check the eigenvalues of each discretization?
show_final_sol = False # show the plots for runs that finish?
show_dissipation = False # show the dissipation plots?
plot_aggregated = True # plot the aggregated results at the end?
plot_aggregated = True # plot the aggregated results at the end?
show_plots = False
track_in_time = True # track the values in time, not just initial condition
check_for_fail = False # if True, check existing data files for failed simulations (missing or short time_array) and rerun them
savefile = None # use a string like 'CSBPp4' to save the plot, None for no save. Note: '.png' added automatically at end

nelem = 1 # number of elements
nen = [20,40,80,160,320] # number of nodes per element
nen = [20,40,80,160,320] # number of nodes per element
op = 'csbp'
p = 4
s = p+1 # dissipation degree
disc_type = 'had' # 'div', 'had'
had_flux = 'chandrashekar' # 2-point numerical flux used in hadamard form: ismail_roe, chandrashekar, ranocha, central
cfl = 1.0
tf = 50.
tm_method = 'rk8' # 'explicit_euler', 'rk4'
para = [287,1.4] # [R, gamma]
nondimensionalize = False
sat = {'diss_type':'ent', 'jac_type':'matmat', 'coeff':1.0, 'P_derigs':True, 'A_derigs':True,
    'entropy_fix':False, 'average':'none', 'maxeig':'none'}
xmin = -1.
xmax = 1.
include_nodiss = True
include_upwind = True
both_dissipation = False
use_1_fifth_diss = True
extra_coeff_vals = False
use_scamat = True
consvar2plot = 0 # 0 for rho, 1 for rho*u, 2 for e
nthreads = 1 # number of threads for batch runs


def run_simulation(op, p, nen, nelem, disc_type, vdiss, sat, label, nruns, irun, cons_obj, datafile=None, do_append=False, ident=None):

    if op in ['csbp', 'hgtl', 'hgt', 'mattsson', 'upwind']:
        dx = (xmax-xmin)/((nen-1)*nelem)
    elif op in ['lgl', 'lg']:
        dx = (xmax-xmin)/(p*nelem)
    dt = cfl*dx/(35.5) # for this problem, assuming max eig is 35.5

    if nen >= 80: 
        sparse = True
    else:
        sparse = False
    print("===============================================")
    print(f'Running {irun+1} of {nruns}: ' + label + f', nelem={nelem}, nen={nen}', flush=True)
    print(f'Running {irun+1} of {nruns}: ' + label + f', nelem={nelem}, nen={nen}', flush=True)
    diffeq = Quasi1dEuler(para, 'density_wave', 'density_wave', 'constant', 'periodic', nondimensionalize)
    solver = PdeSolverSbp(diffeq, {}, tm_method, dt, tf, 
                        p=p,surf_diss=sat, vol_diss=vdiss,  
                        had_flux=had_flux,disc_type=disc_type, 
                        nelem=nelem, nen=nen, disc_nodes=op, 
                        bc='periodic', xmin=xmin, xmax=xmax,
                        sparse=sparse, cons_obj_name=cons_obj)
    
    solver.keep_all_ts = False
    if track_in_time: solver.tm_nframes = round(300*(tf/10))
    solver.tm_atol = 1e-10 #1e-13 #1e-10
    solver.tm_rtol = 1e-10 #3e-13 #1e-10

    if nthreads > 1: solver.print_progress = False

    if check_eigs:
        eigs = solver.check_eigs(plot_eigs=False, returneigs=True)
        maxeig = np.max(eigs.real)
        spec_rad = np.max(np.abs(eigs))
    else:
        maxeig = 0.0
        spec_rad = 0.0

    if run_sims:
        solver.solve()
        tfinal = solver.t_final
        # if it completed, plot the final solution (only rho, by default, but can play with it)
        if solver.t_final == tf and show_final_sol: solver.plot_sol(time=solver.t_final)
    else:
        tfinal = 0.0

    if run_sims and track_in_time:
    if run_sims and track_in_time:
        if show_plots: solver.plot_cons_obj()
        idx = [j for j in range(len(cons_obj)) if cons_obj[j].lower() == 'time'][0]
        time_array = solver.cons_obj[idx, :]
        idx = [j for j in range(len(cons_obj)) if cons_obj[j].lower() == 'max_eig'][0]
        maxeigs_array = solver.cons_obj[idx, :]
        idx = [j for j in range(len(cons_obj)) if cons_obj[j].lower() == 'spec_rad'][0]
        spec_rad_array = solver.cons_obj[idx, :]
        idx = [j for j in range(len(cons_obj)) if cons_obj[j].lower() == 'entropy'][0]
        entropy_array = solver.cons_obj[idx, :]
    else:
        time_array = None
        maxeigs_array = None
        spec_rad_array = None
        entropy_array = None

    dof = solver.nn
    final_sol = solver.q_sol
    if run_sims:
        print('---------------------------------------------------')
        print(f'COMPLETE: {irun+1} of {nruns}: ' + label)
        print('          p:', p, 'nen:', nen, 'nelem:', nelem)
        print('Final time:', solver.t_final)
        print('---------------------------------------------------', flush=True)

    # Optionally append from worker process with a simple lock to serialize writes
    if do_append and datafile is not None and ident is not None:
        lock_path = _lock_path_for(datafile)
        got_lock = _acquire_file_lock(lock_path)
        try:
            if got_lock:
                records = _load_records(datafile)
                if not _record_exists(records, ident):
                    record = {
                        **ident,
                        'tfinal': float(tfinal),
                        'maxeig': float(maxeig) if maxeig is not None else None,
                        'spec_rad': float(spec_rad) if spec_rad is not None else None,
                        'dof': int(dof) if isinstance(dof, (int, np.integer)) else dof,
                        'final_sol': final_sol,
                        'time_array': time_array,
                        'maxeigs_array': maxeigs_array,
                        'spec_rad_array': spec_rad_array,
                        'entropy_array': entropy_array,
                    }
                    records.append(record)
                    _save_records(datafile, records)
        finally:
            if got_lock:
                _release_file_lock(lock_path)

    return tfinal, maxeig, spec_rad, dof, final_sol, \
        time_array, maxeigs_array, spec_rad_array, entropy_array


def _identifier_dict(label, nelem, nen, op, p, s, had_flux, diss, sat):
    # Build a unique identifier by exact match on each provided field
    return {
        'label': label,
        'nelem': int(nelem) if isinstance(nelem, (int, np.integer)) else nelem,
        'nen': int(nen) if isinstance(nen, (int, np.integer)) else nen,
        'op': op,
        'p': int(p) if isinstance(p, (int, np.integer)) else p,
        's': int(s) if isinstance(s, (int, np.integer)) else s,
        'had_flux': had_flux,
        'diss': diss,
        'sat': sat,
    }


def _load_records(datafile_path):
    if os.path.exists(datafile_path):
        try:
            data = np.load(datafile_path, allow_pickle=True)
            if 'records' in data.files:
                arr = data['records']
                # Ensure Python list of dicts
                return list(arr.tolist())
        except Exception:
            pass
    return []


def _save_records(datafile_path, records):
    np.savez(datafile_path, records=np.array(records, dtype=object), allow_pickle=True)


def _record_exists(records, ident):
    for rec in records:
        try:
            if (rec.get('label') == ident['label'] and
                rec.get('nelem') == ident['nelem'] and
                rec.get('nen') == ident['nen'] and
                rec.get('op') == ident['op'] and
                rec.get('p') == ident['p'] and
                rec.get('s') == ident['s'] and
                rec.get('had_flux') == ident['had_flux'] and
                rec.get('diss') == ident['diss'] and
                rec.get('sat') == ident['sat']):
                return True
        except Exception:
            continue
    return False


def _record_is_valid(record):
    """Check if a record has a valid time_array (exists and has length >= 3)."""
    if record is None:
        return False
    time_array = record.get('time_array')
    if time_array is None:
        return False
    try:
        # Handle numpy arrays and lists
        length = len(time_array)
        return length >= 3
    except (TypeError, AttributeError):
        return False


def _find_record(records, ident):
    """Find a record matching the identifier, or return None."""
    for rec in records:
        try:
            if (rec.get('label') == ident['label'] and
                rec.get('nelem') == ident['nelem'] and
                rec.get('nen') == ident['nen'] and
                rec.get('op') == ident['op'] and
                rec.get('p') == ident['p'] and
                rec.get('s') == ident['s'] and
                rec.get('had_flux') == ident['had_flux'] and
                rec.get('diss') == ident['diss'] and
                rec.get('sat') == ident['sat']):
                return rec
        except Exception:
            continue
    return None


def _lock_path_for(datafile_path):
    return datafile_path + '.lock'


def _acquire_file_lock(lock_path, timeout_seconds=30.0, poll_interval_seconds=0.1):
    start_time = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - start_time > timeout_seconds:
                return False
            time.sleep(poll_interval_seconds)


def _release_file_lock(lock_path):
    try:
        os.remove(lock_path)
    except Exception:
        pass


if __name__ == '__main__':

    if op in ['csbp', 'hgtl', 'hgt', 'mattsson', 'upwind']:
        eps = 3.125/5**s
        useH = False
        bdy_fix = True
        avg_half_node = True
        run_var_name = 'nen'
        run_var = nen
    elif op == 'lgl' or op == 'lg':
        assert (s == p)
        if p == 2: eps = 0.02
        elif p == 3: eps = 0.01
        elif p == 4: eps = 0.004
        elif p == 5: eps = 0.002
        elif p == 6: eps = 0.0008
        elif p == 7: eps = 0.0004
        elif p == 8: eps = 0.0002
        # can approximate this well with eps = 0.1253*np.exp(-0.8716*s) +0.0076*s**(-2.8234)
        else: raise Exception('No dissipation for this p')
        useH = False
        bdy_fix = False
        avg_half_node = False
        run_var_name = 'nelem'
        run_var = nelem
    else:
        raise Exception('No dissipation for this operator')
    
    if extra_coeff_vals:
        diss = [{'diss_type':'nd'},
                {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.04*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.008*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.04*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat','coeff':0.008*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'mat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'mat', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'sca', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'sca', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node}]
        labels = ['no volume diss',
                f'ent-matmat-{eps:g}',
                f'ent-matmat-{0.2*eps:g}',
                f'ent-matmat-{0.04*eps:g}',
                f'ent-matmat-{0.008*eps:g}',
                f'ent-scamat-{eps:g}',
                f'ent-scamat-{0.2*eps:g}',
                f'ent-scamat-{0.04*eps:g}',
                f'ent-scamat-{0.008*eps:g}',
                f'mat-{eps:g}',
                f'mat-{0.2*eps:g}',
                f'sca-{eps:g}',
                f'sca-{0.2*eps:g}']
    else:
        diss = [{'diss_type':'nd'},
                {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'entdcp', 'jac_type':'scamat','coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'mat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'mat', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'sca', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
                {'diss_type':'dcp', 'jac_type':'sca', 'coeff':0.2*eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node}]
        labels = ['no volume diss',
                f'ent-matmat-{eps:g}',
                f'ent-matmat-{0.2*eps:g}',
                f'ent-scamat-{eps:g}',
                f'ent-scamat-{0.2*eps:g}',
                f'mat-{eps:g}',
                f'mat-{0.2*eps:g}',
                f'sca-{eps:g}',
                f'sca-{0.2*eps:g}']

    if include_nodiss == False:
        diss = diss[1:]
        labels = labels[1:]
        
    if both_dissipation == False:
        if extra_coeff_vals:
            diss = diss[:9]
            labels = labels[:9]
        else:
            diss = diss[:5]
            labels = labels[:5]

    if use_scamat == False:
        if extra_coeff_vals:
            diss = diss[:5] 
            labels = labels[:5]
        else:
            diss = diss[:3] 
            labels = labels[:3]

    if use_1_fifth_diss == False and extra_coeff_vals == False:
        if include_nodiss:
            tmp = [diss[0]]
            tmp = tmp + diss[1::2]
            diss = tmp
            tmp = [labels[0]]
            tmp = tmp + labels[1::2]
            labels = tmp
        else:
            diss = diss[::2]
            labels = labels[::2]

    if include_upwind:
        if op == 'lgl' or op == 'lg':
            raise Exception('TODO')
        else:
            diss += [{'diss_type':'upwind', 'fluxvec':'sw', 'coeff':1., 'p_u':int(2*p)},
                     {'diss_type':'upwind', 'fluxvec':'sw', 'coeff':1., 'p_u':int(2*p+1)}]
            labels += [f'UFD $p_\\text{{u}}={int(2*p)}$', f'UFD $p_\\text{{u}}={int(2*p+1)}$']

    # Manual overwrite
    """ diss = [{'diss_type':'nd'},
            {'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':0.001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0005, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0002, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'matmat','coeff':0.0001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0005, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0002, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node},
            {'diss_type':'entdcp', 'jac_type':'scamat', 'coeff':0.0001, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False, 'avg_half_nodes':avg_half_node}]
    labels = ['no volume diss',
            f'ent-matmat-{0.001}',
            f'ent-matmat-{0.0005}',
            f'ent-matmat-{0.0002}',
            f'ent-matmat-{0.0001}',
            f'ent-scamat-{0.001}',
            f'ent-scamat-{0.0005}',
            f'ent-scamat-{0.0002}',
            f'ent-scamat-{0.0001}'] """

    if track_in_time: 
        cons_obj = ('Entropy','Max_Eig','Spec_Rad','time')
        maxeigs_time = [[] for _ in range(len(run_var))]
        spec_rad_time = [[] for _ in range(len(run_var))]
        entropy_time = [[] for _ in range(len(run_var))]
    else:
        cons_obj = None

    if run_sims:
        final_sols = [[] for _ in range(len(run_var))]
    else:
        final_sols = None

    if run_sims:
        final_sols = [[] for _ in range(len(run_var))]
    else:
        final_sols = None

    if run_sims or check_eigs:
        # Prepare per-run appendable data file only when tracking in time
        if track_in_time:
            if savefile is not None:
                datafile = savefile + '_data.npz'
            else:
                print('WARNING: No savefile specified, using default name')
                datafile = 'RENAMEME_densitywave_data.npz'
            records = _load_records(datafile)
        else:
            datafile = None
            records = []

        futures = [[None for _ in range(len(diss))] for _ in range(len(run_var))]
        nruns = len(diss)*len(run_var)
        submitted_runs = []  # Track which runs we actually submitted

        with ProcessPoolExecutor(max_workers=nthreads) as executor:  
            for dof_i in range(len(run_var)):

                if run_var_name == 'nen': 
                    nen_ = nen[dof_i]
                    nelem_ = nelem
                elif run_var_name == 'nelem': 
                    nen_ = nen
                    nelem_ = nelem[dof_i]
                else:
                    raise Exception('Invalid run_var_name')

                for diss_i,vdiss in enumerate(diss):

                    if vdiss['diss_type'] == 'upwind':
                        if op in ['csbp', 'hgtl', 'hgt', 'mattsson']:
                            op_ = 'upwind'
                            p_ = vdiss['p_u']
                            disc_type_ = 'div' # upwind is always div
                            sat_ = {'diss_type':'cons', 'jac_type':'mat', 'coeff':1., 'average':'roe', 
                                    'entropy_fix':False, 'P_derigs':False, 'A_derigs':False, 'maxeig':'none'}
                        else: 
                            op_ = op
                            p_ = p
                            disc_type_ = disc_type
                            sat_ = sat
                    else:
                        op_ = op
                        p_ = p
                        disc_type_ = disc_type
                        sat_ = sat

                    # Check if this run already exists before submitting (only when tracking)
                    ident = _identifier_dict(labels[diss_i], nelem_, nen_, op_, p_, s, had_flux, vdiss, sat_)
                    if track_in_time and _record_exists(records, ident):
                        # If check_for_fail is enabled, validate the record
                        if check_for_fail:
                            existing_record = _find_record(records, ident)
                            if existing_record is not None and not _record_is_valid(existing_record):
                                # Record exists but is invalid (failed simulation), remove it and rerun
                                print(f"Found failed simulation (invalid time_array): {labels[diss_i]}, nelem={nelem_}, nen={nen_}. Removing and rerunning...", flush=True)
                                records.remove(existing_record)
                                # Save updated records to file (remove the failed record)
                                lock_path = _lock_path_for(datafile)
                                got_lock = _acquire_file_lock(lock_path)
                                try:
                                    if got_lock:
                                        _save_records(datafile, records)
                                finally:
                                    if got_lock:
                                        _release_file_lock(lock_path)
                                # Continue to submit the simulation (don't skip)
                            else:
                                # Record exists and is valid, skip it
                                print(f"Skipping existing run: {labels[diss_i]}, nelem={nelem_}, nen={nen_}", flush=True)
                                continue
                        else:
                            # check_for_fail is False, skip existing records as before
                            print(f"Skipping existing run: {labels[diss_i]}, nelem={nelem_}, nen={nen_}", flush=True)
                            continue

                    irun = dof_i*len(diss) + diss_i
                    # When tracking in time, pass datafile and ident for worker-side appends
                    future = executor.submit(
                        run_simulation,
                        op_, p_, nen_, nelem_, disc_type_, vdiss, sat_, labels[diss_i], nruns, irun, cons_obj,
                        datafile=datafile if track_in_time else None,
                        do_append=bool(track_in_time),
                        ident=ident if track_in_time else None,
                    )
                    futures[dof_i][diss_i] = future
                    submitted_runs.append((dof_i, diss_i, ident))  # Track what we submitted
        
        print()
        print('---------------------------------------------------')
        print('COMPLETE: All simulations submitted')
        print('---------------------------------------------------', flush=True)
        print('---------------------------------------------------', flush=True)
        print()

        # Process results as they complete and append to data file (only when tracking)
        from concurrent.futures import as_completed
        
        # Create a mapping from future to run info
        future_to_run_info = {}
        for dof_i, diss_i, ident in submitted_runs:
            future_to_run_info[futures[dof_i][diss_i]] = (dof_i, diss_i, ident)
        
        # Store results as they complete to avoid calling .result() multiple times
        completed_results = {}
        
        # Process futures as they complete (not in submission order)
        for i, future in enumerate(as_completed([futures[dof_i][diss_i] for dof_i, diss_i, ident in submitted_runs])):
            dof_i, diss_i, ident = future_to_run_info[future]
            print(f"Processing completed simulation {i+1}/{len(submitted_runs)}: {ident['label']}, nelem={ident['nelem']}, nen={ident['nen']}", flush=True)
            
            tfinal, maxeig, spec_rad, dof, final_sol, \
            time_array, maxeigs_array, spec_rad_array, entropy_array = future.result()
            
            # Store results for later use
            completed_results[(dof_i, diss_i)] = (tfinal, maxeig, spec_rad, dof, final_sol, 
                                                 time_array, maxeigs_array, spec_rad_array, entropy_array)

        # Rebuild the old-style arrays for compatibility with existing plotting code
        for dof_i in range(len(run_var)):
            maxeigs = []
            spec_rads = []
            tfinals = []
            dofs = []
            for diss_i,vdiss in enumerate(diss):
                if futures[dof_i][diss_i] is not None:  # Only process runs that were actually submitted
                    # Get results from stored completed results
                    if (dof_i, diss_i) in completed_results:
                        tfinal, maxeig, spec_rad, dof, final_sol, \
                        time_array, maxeigs_array, spec_rad_array, entropy_array = completed_results[(dof_i, diss_i)]

                        maxeigs.append(maxeig)
                        spec_rads.append(spec_rad)
                        tfinals.append(tfinal)
                        dofs.append(dof)
                        if run_sims:
                            final_sols[dof_i].append(final_sol)
                            if track_in_time:
                                maxeigs_time[dof_i].append(maxeigs_array)
                                spec_rad_time[dof_i].append(spec_rad_array)
                                entropy_time[dof_i].append(entropy_array)
                                time_arrays[dof_i].append(time_array)
                    else:
                        # Fallback: call .result() if not in completed_results (shouldn't happen)
                        tfinal, maxeig, spec_rad, dof, final_sol, \
                        time_array, maxeigs_array, spec_rad_array, entropy_array = futures[dof_i][diss_i].result()

                        maxeigs.append(maxeig)
                        spec_rads.append(spec_rad)
                        tfinals.append(tfinal)
                        dofs.append(dof)
                        if run_sims:
                            final_sols[dof_i].append(final_sol)
                            if track_in_time:
                                maxeigs_time[dof_i].append(maxeigs_array)
                                spec_rad_time[dof_i].append(spec_rad_array)
                                entropy_time[dof_i].append(entropy_array)
                                time_arrays[dof_i].append(time_array)
                else:
                    # Load existing data for skipped runs
                    ident = _identifier_dict(labels[diss_i], nelem_, nen_, op, p, s, had_flux, vdiss, sat)
                    existing_record = None
                    for rec in records:
                        try:
                            if (rec.get('label') == ident['label'] and
                                rec.get('nelem') == ident['nelem'] and
                                rec.get('nen') == ident['nen'] and
                                rec.get('op') == ident['op'] and
                                rec.get('p') == ident['p'] and
                                rec.get('s') == ident['s'] and
                                rec.get('had_flux') == ident['had_flux'] and
                                rec.get('diss') == ident['diss'] and
                                rec.get('sat') == ident['sat']):
                                existing_record = rec
                                break
                        except Exception:
                            continue
                    
                    if existing_record is not None:
                        maxeigs.append(existing_record.get('maxeig'))
                        spec_rads.append(existing_record.get('spec_rad'))
                        tfinals.append(existing_record.get('tfinal'))
                        dofs.append(existing_record.get('dof'))
                        if run_sims:
                            final_sols[dof_i].append(existing_record.get('final_sol'))
                            if track_in_time:
                                maxeigs_time[dof_i].append(existing_record.get('maxeigs_array'))
                                spec_rad_time[dof_i].append(existing_record.get('spec_rad_array'))
                                entropy_time[dof_i].append(existing_record.get('entropy_array'))
                                time[dof_i].append(existing_record.get('time_array'))
                    else:
                        # Fallback to None if record not found (shouldn't happen)
                        maxeigs.append(None)
                        spec_rads.append(None)
                        tfinals.append(None)
                        dofs.append(None)

            if run_var_name == 'nen': 
                nen_ = nen[dof_i]
                nelem_ = nelem
            elif run_var_name == 'nelem': 
                nen_ = nen
                nelem_ = nelem[dof_i]

            # Compute time-maximum metrics if available; else None
            time_max_re = []
            time_spec_rad = []
            for diss_i in range(len(diss)):
                if track_in_time:
                    try:
                        arr_eig = maxeigs_time[dof_i][diss_i]
                        arr_rad = spec_rad_time[dof_i][diss_i]
                    except Exception:
                        arr_eig, arr_rad = None, None
                    if arr_eig is not None:
                        try:
                            time_max_re.append(float(np.max(arr_eig)))
                        except Exception:
                            time_max_re.append(None)
                    else:
                        time_max_re.append(None)
                    if arr_rad is not None:
                        try:
                            time_spec_rad.append(float(np.max(arr_rad)))
                        except Exception:
                            time_spec_rad.append(None)
                    else:
                        time_spec_rad.append(None)
                else:
                    time_max_re.append(None)
                    time_spec_rad.append(None)

            # Filter out None values for core metrics; include time-based columns even if None
            valid_data = [(label,
                           f"{eig:.4g}",
                           f"{rad:.4g}",
                           (f"{eig_t:.4g}" if eig_t is not None else None),
                           (f"{rad_t:.4g}" if rad_t is not None else None),
                           round(tf, 6))
                        for label, eig, rad, eig_t, rad_t, tf in zip(labels, maxeigs, spec_rads, time_max_re, time_spec_rad, tfinals)
                        if eig is not None and rad is not None and tf is not None]
            headers = ["Dissipation", "Max Re(\u03BB) (0)", "Spec Rad (0)", "Max Re(\u03BB)(t)", "Spec Rad(t)", "Quit Time"]
            print('Operator=' + op + f' p={p}' + f' s={s}' + f' nelem={nelem_}' + f' nen={nen_}' + f' had_flux={had_flux}')
            print(tabulate(valid_data, headers=headers, tablefmt="pretty"))

    # Deprecated: end-of-batch save. Data is now appended per-run to *_data.npz
        
        #TODO: Some plotting?
        if False:
            data = np.load(datafile, allow_pickle=True)
            time_arrays, maxeigs, spec_rad, entropy, dof = data['time'], data['maxeigs'], data['spec_rad'], data['entropy'], data['dof']
            final_sols = data['final_sol']
            plt.plot(final_sols[0][0].flatten('F'), label='Final Sol')
        

    ####### plot initial condition #######
    if show_dissipation:
        diffeq = Quasi1dEuler(para, 'density_wave', 'density_wave', 'constant', 'periodic', nondimensionalize)
        solver = PdeSolverSbp(diffeq, {}, tm_method, 1e-8, tf, 
                                p=p,surf_diss=sat, vol_diss='nd',
                                had_flux=had_flux,disc_type=disc_type, 
                                nelem=nelem, nen=nen, disc_nodes=op, 
                                bc='periodic', xmin=xmin, xmax=xmax)
        q = diffeq.set_q0()
        x = diffeq.x
        rho = diffeq.var2plot(q, 'rho').flatten('F')
        w1 = diffeq.var2plot(q, 'w1').flatten('F')

        fig, ax1 = plt.subplots(figsize=(6,4))
        color = 'tab:blue'
        ax1.set_xlabel(r'$x$',fontsize=16)
        ax1.set_ylabel(r'$\rho$', fontsize=16)
        ax1.plot(x, rho, color=color, linestyle='-', linewidth=2.5, label=r'$\rho$')
        #ax1.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')
        color = 'tab:orange'
        ax2 = ax1.twinx() 
        ax2.set_ylabel(r'$w_1 = \frac{\gamma - s}{\gamma-1} - \frac{\rho u^2}{2p}$', fontsize=16) 
        ax2.plot(x, w1, color=color, linestyle=':', linewidth=2.5, label=r'$w_1$')
        #ax2.grid(which='major',axis='y',linestyle='--',color=color,linewidth='1')
        ax1.tick_params(axis='both', labelsize=12) 
        ax2.tick_params(axis='both', labelsize=12) 
        fig.legend(loc='lower left',fontsize=14,bbox_to_anchor=(0.125, 0.15),
                fancybox=True,shadow=False,ncol=1,columnspacing=1.5)
        fig.tight_layout()
        if savefile is not None: fig.savefig(savefile + '_vars.pdf', dpi=600)

        ####### plot contributions to RHS #######
        solver = PdeSolverSbp(diffeq, {}, tm_method, 1e-8, tf, 
                                p=p,surf_diss=sat, 
                                vol_diss={'diss_type':'entdcp', 'jac_type':'matmat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False},
                                had_flux=had_flux,disc_type=disc_type, 
                                nelem=nelem, nen=nen, disc_nodes=op, 
                                bc='periodic', xmin=xmin, xmax=xmax)
        diss1 = solver.adiss.dissipation(q)[consvar2plot::3].flatten('F')
        solver = PdeSolverSbp(diffeq, {}, tm_method, 1e-8, tf, 
                                p=p,surf_diss=sat, 
                                vol_diss={'diss_type':'dcp', 'jac_type':'mat', 'coeff':eps, 's':s, 'bdy_fix':bdy_fix, 'use_H':useH, 'entropy_fix':False},  
                                had_flux=had_flux,disc_type=disc_type, 
                                nelem=nelem, nen=nen, disc_nodes=op, 
                                bc='periodic', xmin=xmin, xmax=xmax,
                                sparse=False)
        diss2 = solver.adiss.dissipation(q)[consvar2plot::3].flatten('F')

        plt.figure(figsize=(6,4))
        plt.xlabel(r'$x$',fontsize=16)
        if consvar2plot == 0: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d \rho}{d t}$', fontsize=16)
        elif consvar2plot == 1: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d \rho u}{d t}$', fontsize=16)
        elif consvar2plot == 2: plt.ylabel(r'Contribution of $\mathsf{A}_\mathsf{D}$ to $\frac{d e}{d t}$', fontsize=16)
        else: raise Exception('Invalid consvar2plot')
        color = 'tab:blue'
        #plt.plot(x, diss2, color='tab:blue', linestyle='-', linewidth=2.5, label='Cons. Sca')
        #plt.plot(x, diss1, color='tab:orange', linestyle=':', linewidth=2.5, label='Ent. Sca-Mat')
        plt.plot(x, diss2, color='tab:blue', linestyle='-', linewidth=2.5, label='Conservative Matrix')
        plt.plot(x, diss1, color='tab:orange', linestyle=':', linewidth=2.5, label='Entropy Matrix-Matrix')
        plt.yscale('symlog',linthresh=1e-6)
        #plt.ylim(-5,5)
        plt.grid(which='major',axis='y',linestyle='--',linewidth='1')
        plt.gca().tick_params(axis='both', labelsize=12) 
        #plt.legend(loc='lower left',fontsize=14, bbox_to_anchor=(-0.015, -0.02))
        plt.legend(loc='upper center',fontsize=14,  bbox_to_anchor=(0.472, 1.155), fancybox=True, shadow=False, ncol=2, columnspacing=1.5)
        plt.tight_layout()
        if savefile is not None: plt.savefig(savefile + '_diss.pdf', dpi=600)


    ########### plot the aggregated results (logged manually, can verify from above) ############
    if plot_aggregated:
        assert op == 'csbp', 'Data only saved for CSBP operator'
        from matplotlib.transforms import blended_transform_factory

        nodes = np.array([20,40,80,160,320])
        if p == 4:
            max_eigs_nodiss = np.array([2.825,0.6195,0.3792,2.225,4.017])
            max_eigs_matmat1 = np.array([25.26,2.177,0.3587,1.818e-4,5.369e-10])
            max_eigs_matmat02 = np.array([5.376,0.1388,0.002949,2.278e-06,1.06e-09])
            max_eigs_matmat004 = np.array([1.262,0.2278,0.01004,1.837e-07,3.908e-07])
            max_eigs_matmat0008 = np.array([2.363,0.377,0.07466,1.665,2.789])
            max_eigs_scamat1 = np.array([7.597,0.7511,0.004984,1.611e-06,5.453e-10])
            max_eigs_scamat02 = np.array([2.598,0.2984,0.001997,7.83e-07,9.319e-10])
            max_eigs_scamat004 = np.array([2.094,0.104,0.001344,7.42e-07,3.874e-07])
            max_eigs_scamat0008 = np.array([2.37,0.3739,0.04053,1.665,2.789])
            max_eigs_pu2p_sw = np.array([4.831,5.446,2.034,0.0691,0.0005763])
            max_eigs_pu2p1_sw = np.array([3.451,3.488,1.195,0.03207,0.0002351])
            max_eigs_pu2p_sw = np.array([4.831,5.446,2.034,0.0691,0.0005763])
            max_eigs_pu2p1_sw = np.array([3.451,3.488,1.195,0.03207,0.0002351])
            spec_rad_nodiss = np.array([516.2,968.2,2124,4311,9297])
            spec_rad_matmat1 = np.array([1801,1820,2013,4071,8976])
            spec_rad_matmat02 = np.array([511.6,966.3,2120,4301,9283])
            spec_rad_matmat004 = np.array([516.0,968.1,2124,4310,9297])
            spec_rad_matmat0008 = np.array([516.2,968.2,2124,4311,9297])
            spec_rad_scamat1 = np.array([1823,1939,2009,4069,8976])
            spec_rad_scamat02 = np.array([512.4,964.3,2120,4301,9283])
            spec_rad_scamat004 = np.array([516.2,967.8,2124,4310,9297])
            spec_rad_scamat0008 = np.array([516.2,968.1,2124,4311,9297])
            spec_rad_pu2p_sw = np.array([419.9,887.1,2119,5021,1.055e+04])
            spec_rad_pu2p1_sw = np.array([403,863.2,1938,4288,9472])
            spec_rad_pu2p_sw = np.array([419.9,887.1,2119,5021,1.055e+04])
            spec_rad_pu2p1_sw = np.array([403,863.2,1938,4288,9472])
            crash_time_nodiss = np.array([17.937061,33.792154,3.206111,2.497499,2.618005])
            crash_time_matmat1 = np.array([50,50,50,50,50])
            crash_time_matmat02 = np.array([50,50,50,50,50])
            crash_time_matmat004 = np.array([1.5621999,50,50,50,50])
            crash_time_matmat0008 = np.array([1.5936532,13.105304,2.401841,2.866652,50])
            crash_time_scamat1 = np.array([50,50,50,50,50])
            crash_time_scamat02 = np.array([50,50,50,50,50])
            crash_time_scamat004 = np.array([1.5621998,50,50,50,50])
            crash_time_scamat0008 = np.array([50,4.0493814,2.352266,50,50])
            crash_time_pu2p_sw = np.array([0.403328,0.271765,0.602812,12.967872,50])
            crash_time_pu2p1_sw = np.array([0.308718,0.309748,0.90709,26.772701,50])
            crash_time_pu2p_sw = np.array([0.403328,0.271765,0.602812,12.967872,50])
            crash_time_pu2p1_sw = np.array([0.308718,0.309748,0.90709,26.772701,50])
        elif p == 3:
            max_eigs_nodiss = np.array([])
            max_eigs_matmat1 = np.array([])
            max_eigs_matmat02 = np.array([])
            max_eigs_matmat004 = np.array([])
            max_eigs_matmat0008 = np.array([])
            max_eigs_scamat1 = np.array([])
            max_eigs_scamat02 = np.array([])
            max_eigs_scamat004 = np.array([])
            max_eigs_scamat0008 = np.array([])
            spec_rad_nodiss = np.array([])
            spec_rad_matmat1 = np.array([])
            spec_rad_matmat02 = np.array([])
            spec_rad_matmat004 = np.array([])
            spec_rad_matmat0008 = np.array([])
            spec_rad_scamat1 = np.array([])
            spec_rad_scamat02 = np.array([])
            spec_rad_scamat004 = np.array([])
            spec_rad_scamat0008 = np.array([])
            crash_time_nodiss = np.array([])
            crash_time_matmat1 = np.array([])
            crash_time_matmat02 = np.array([])
            crash_time_matmat004 = np.array([])
            crash_time_matmat0008 = np.array([])
            crash_time_scamat1 = np.array([])
            crash_time_scamat02 = np.array([])
            crash_time_scamat004 = np.array([])
            crash_time_scamat0008 = np.array([])
        else:
            raise Exception('No saved data for this p')
        
        if include_upwind:
            max_eigs = np.array([max_eigs_nodiss,max_eigs_scamat1,max_eigs_scamat02,max_eigs_matmat1,max_eigs_matmat02,max_eigs_pu2p_sw,max_eigs_pu2p1_sw])
            spec_rad = np.array([spec_rad_nodiss,spec_rad_scamat1,spec_rad_scamat02,spec_rad_matmat1,spec_rad_matmat02,spec_rad_pu2p_sw,spec_rad_pu2p1_sw])
            crash_time = np.array([crash_time_nodiss, crash_time_scamat1, crash_time_scamat02, crash_time_matmat1, crash_time_matmat02, crash_time_pu2p_sw, crash_time_pu2p1_sw])
            labels = [r'$\varepsilon = 0$', r'Sca.-Mat. $\varepsilon = 0.001$', r'Sca.-Mat. $\varepsilon = 0.0002$', r'Mat.-Mat. $\varepsilon = 0.001$', r'Mat.-Mat. $\varepsilon = 0.0002$', r'UFD SW $p_u=8$', r'UFD SW $p_u=9$']
            #colors = ['darkgoldenrod', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
            #markers = ['v', 's', '^', 'd', 'o']
            colors = ['tab:orange', 'tab:green', 'darkgoldenrod', 'k', 'm', 'tab:red', 'tab:blue']
            markers = ['v', '*', 'd', 'x', '+', 'o', '^']
            linestyles = ['-', '-', '-', (1, (2,1)), (1, (2,1)), ':', ':']
            markersizes = [12, 11, 11, 10, 10, 9, 9]
        elif extra_coeff_vals:
            max_eigs = np.array([max_eigs_nodiss,max_eigs_matmat1,max_eigs_matmat02,max_eigs_matmat004,max_eigs_matmat0008,max_eigs_scamat1,max_eigs_scamat02,max_eigs_scamat004,max_eigs_scamat0008])
            spec_rad = np.array([spec_rad_nodiss,spec_rad_matmat1,spec_rad_matmat02,spec_rad_matmat004,spec_rad_matmat0008,spec_rad_scamat1,spec_rad_scamat02,spec_rad_scamat004,spec_rad_scamat0008])
            crash_time = np.array([crash_time_nodiss,crash_time_matmat1,crash_time_matmat02,crash_time_matmat004,crash_time_matmat0008,crash_time_scamat1,crash_time_scamat02,crash_time_scamat004,crash_time_scamat0008])
            labels = [r'$\varepsilon = 0$', r'Mat.-Mat. $\varepsilon = 0.001$', r'Mat.-Mat. $\varepsilon = 0.0002$', r'Mat.-Mat. $\varepsilon = 4 \times 10^{-5}$', r'Mat.-Mat. $\varepsilon = 8 \times 10^{-6}$', 
                                            r'Sca.-Mat. $\varepsilon = 0.001$', r'Sca.-Mat. $\varepsilon = 0.0002$', r'Sca.-Mat. $\varepsilon = 4 \times 10^{-5}$', r'Sca.-Mat. $\varepsilon = 8 \times 10^{-6}$']
            colors = ['darkgoldenrod', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
            markers = ['v', 's', '^', '>', '<', 'd', 'o', 'x', '*']
            linestyles = ['-', '-', '-', '-', '-', (1, (2,1)), (1, (2,1)), (1, (2,1)), (1, (2,1))]
            markersizes = [12, 11, 11, 10, 10, 10, 10, 9, 9]
        else:
            max_eigs = np.array([max_eigs_nodiss,max_eigs_matmat1,max_eigs_matmat02,max_eigs_scamat1,max_eigs_scamat02])
            spec_rad = np.array([spec_rad_nodiss,spec_rad_matmat1,spec_rad_matmat02,spec_rad_scamat1,spec_rad_scamat02])
            crash_time = np.array([crash_time_nodiss, crash_time_matmat1, crash_time_matmat02, crash_time_scamat1, crash_time_scamat02])
            labels = [r'$\varepsilon = 0$', r'Mat.-Mat. $\varepsilon = 0.001$', r'Mat.-Mat. $\varepsilon = 0.0002$', r'Sca.-Mat. $\varepsilon = 0.001$', r'Sca.-Mat. $\varepsilon = 0.0002$']
            colors = ['darkgoldenrod', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
            markers = ['v', 's', '^', 'd', 'o']
            linestyles = ['-', '-', '-', (1, (2,1)), (1, (2,1))]
            markersizes = [12, 11, 11, 10, 10]
        xlabel = r'Degrees of Freedom'
        #savefile = None
        #savefile = None
        linewidth = 3
        xticks = {20 : r'$20$',
                    40 : r'$40$',
                    80 : r'$80$',
                    200 : r'$200$'}
        for i in range(3):
            leg_order = 2
            leg_alpha = 0.85
            if i == 0:
                data = max_eigs
                ylabel = r'max $\Re (\lambda)$'
                yticks = None
                if savefile is not None: savefile_ = savefile + '_max_eig.pdf'
                else: savefile_ = None
                legendloc = 'lower left'
                legendanchor = None
                log = True
                ylim = (1e-10,1e2)
                if include_upwind:
                    legendsize = 12
                else:
                    legendsize = 13.8
            elif i == 1:
                data = spec_rad
                ylabel = r'max $\vert \lambda \vert$'
                yticks = {5e2 : r'$5 \times 10^2$',
                        1e3 : r'$10^3$',
                        2e3 : r'$2 \times 10^3$',
                        5e3 : r'$5 \times 10^3$',
                        1e4 : r'$10^4$'}
                ylabel_adjust = -0.2
                if savefile is not None: savefile_ = savefile + '_spec_rad.pdf'
                else: savefile_ = None
                legendloc = 'upper left'
                legendanchor = None
                legend=False
                log = True
                if include_upwind:
                    legendsize = 10.5
                    ylim = (300,1.3e4)
                else:
                    legendsize = 12.5
                    ylim = (4e2,1.3e4)
                if include_upwind:
                    legendsize = 10.5
                    ylim = (300,1.3e4)
                else:
                    legendsize = 12.5
                    ylim = (4e2,1.3e4)
            elif i == 2:
                data = crash_time
                ylabel = r'Crash Time $t_f$'
                ylabel_adjust = -0.10
                if savefile is not None: savefile_ = savefile + '_crash_time.pdf'
                else: savefile_ = None
                log = True
                if include_upwind:
                    legendsize = 11
                    ylim = (0.2, 60)
                    yticks = {0.2 : r'$0.2$',
                              0.5 : r'$0.5$',
                              1 : r'$1$',
                              2 : r'$2$',
                        5 : r'$5$',
                        10 : r'$10$',
                        50 : r'$>50$'}
                    legendanchor = (0.58,-0.1)
                    legendloc = 'lower left'
                    leg_order = 3
                    leg_alpha = 0.95
                else:
                    legendsize = 12.5
                    ylim = (2.0, 60)
                    yticks = {2 : r'$2$',
                        5 : r'$5$',
                        10 : r'$10$',
                        50 : r'$>50$'}
                    legendanchor = (1.0,0.93)
                    legendloc = 'upper right'
            if extra_coeff_vals: legendsize = 10
            
            if include_upwind:
                fig = plt.figure(figsize=(6.2,4.5))
                ax = fig.add_axes([0.2, 0.15, 0.55, 0.75]) 
            else:
                fig = plt.figure(figsize=(5.0,4.5))
            if include_upwind:
                fig = plt.figure(figsize=(6.2,4.5))
                ax = fig.add_axes([0.2, 0.15, 0.55, 0.75]) 
            else:
                fig = plt.figure(figsize=(5.0,4.5))
            for j in range(len(data)):
                plt.plot(nodes, data[j], color=colors[j], linestyle=linestyles[j], 
                    marker=markers[j], label=labels[j], linewidth=linewidth, 
                    markersize=markersizes[j], markerfacecolor='none', markeredgewidth=linewidth,zorder=3)
            # again so that markers are ontop
            for j in range(len(data)):
                plt.plot(nodes, data[j], color=colors[j], linestyle='', 
                    marker=markers[j], label=None, linewidth=None, 
                    markersize=markersizes[j], markerfacecolor='none', markeredgewidth=linewidth,zorder=4)

            if log:
                plt.xscale('log')
                plt.yscale('log')
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.ylim(ylim)
            plt.grid(which='major', axis='y', linestyle='--', linewidth=1)
            ax = plt.gca()
            ax.tick_params(axis='x', which='both', labelbottom=False)
            ax.xaxis.set_label_coords(0.5, -0.1)
            trans = blended_transform_factory(ax.transData, fig.transFigure)
            for tick in xticks:
                ax.text(tick, 0.132, xticks[tick], transform=trans, ha='center', va='top', fontsize=14)
            if yticks is not None:
                ax.tick_params(axis='y', which='both', labelleft=False)
                ax.yaxis.set_label_coords(ylabel_adjust, 0.5)
                xmin = ax.get_xlim()[0]
                xlim = xmin - 0.05*xmin
                for tick in yticks:
                    ax.text(xlim, tick, yticks[tick], ha='right', va='center', fontsize=14)
            else:
                ax.tick_params(axis='y', labelsize=14)
            legend = plt.legend(loc=legendloc, bbox_to_anchor=legendanchor, fontsize=legendsize, 
                                fancybox=True, shadow=False, ncol=1, columnspacing=1, markerscale=0.8)
            for leg_line in legend.get_lines():
                leg_line.set_linewidth(linewidth*0.7)
                leg_line.set_markeredgewidth(linewidth*0.7)
            legend.set_zorder(leg_order)
            legend.get_frame().set_alpha(leg_alpha)
            #plt.tight_layout()
            if not include_upwind:
                plt.subplots_adjust(bottom=0.15, left=0.2)
            if not include_upwind:
                plt.subplots_adjust(bottom=0.15, left=0.2)
            if savefile is not None: 
                plt.savefig(savefile_, dpi=600)
            else:
                plt.show()
