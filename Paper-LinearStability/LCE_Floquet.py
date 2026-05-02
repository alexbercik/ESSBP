"""
Linear Convection Equation (LCE): 
Plots Floquet growth rates and instantaneous eigenvalue spectra, 
reproducing figures from section 3.2 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals, svd
from scipy.linalg import expm

import Floquet_Analysis  # noqa: F401  # extends sys.path for Source.*
from Floquet_Analysis import baseflow_u, operator_L_from_u, run_floquet
from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


T = 1.0
q0_type = 'GaussWave_shift'  # 'GaussWave_shift', 'SinWave_shift'

# Spatial discretization
disc_nodes = 'circulant'  # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd', 'upwind'
p = 4
nen = 40
nelem = 1
had_flux = 'geometric'  # 'central', 'geometric', 'logarithmic'

surf_type = {'diss_type': 'nd', 'fluxvec': 'lf', 'coeff': 1.}
vol_diss = {'diss_type': 'nd', 'use_H': False, 'bdy_fix': False,
            'jac_type': 'scalarscalar', 's': p, 'coeff': 0.004}

cons_obj_name = None
use_exact_sol = True


diffeq = LinearConv(1.0, q0_type, had_flux=had_flux)
solver = PdeSolverSbp(diffeq, None,
                        'rk8', 0.001, T,
                        p, 'had',
                        surf_type, vol_diss, had_flux,
                        nelem, nen, disc_nodes,
                        'periodic', 0., 1.,
                        cons_obj_name,
                        print_progress=True)
solver.tm_atol = 1e-12
solver.tm_rtol = 1e-12
solver.keep_all_ts = False
if not use_exact_sol:
    solver.tm_print_nothing = True

nen2 = 60
nelem2 = 1
diffeq2 = LinearConv(1.0, q0_type, had_flux=had_flux)
solver2 = PdeSolverSbp(diffeq2, None,
                        'rk8', 0.001, T,
                        p, 'had',
                        surf_type, vol_diss, had_flux,
                        nelem2, nen2, disc_nodes,
                        'periodic', 0., 1.,
                        cons_obj_name,
                        print_progress=False)
solver2.tm_atol = 1e-12
solver2.tm_rtol = 1e-12
solver2.keep_all_ts = False
if not use_exact_sol:
    solver2.tm_print_nothing = True

nen3 = 100
nelem3 = 1
diffeq3 = LinearConv(1.0, q0_type, had_flux=had_flux)
solver3 = PdeSolverSbp(diffeq3, None,
                        'rk8', 0.001, T,
                        p, 'had',
                        surf_type, vol_diss, had_flux,
                        nelem3, nen3, disc_nodes,
                        'periodic', 0., 1.,
                        cons_obj_name,
                        print_progress=False)
solver3.tm_atol = 1e-12
solver3.tm_rtol = 1e-12
solver3.keep_all_ts = False
if not use_exact_sol:
    solver3.tm_print_nothing = True

results = run_floquet(T, solver, K=1000, use_H=True, use_exact_sol=use_exact_sol)
results2 = run_floquet(T, solver2, K=1000, use_H=True, use_exact_sol=use_exact_sol)
results3 = run_floquet(T, solver3, K=1000, use_H=True, use_exact_sol=use_exact_sol)
L = operator_L_from_u(baseflow_u(0.0, solver), solver)
L2 = operator_L_from_u(baseflow_u(0.0, solver2), solver2)
L3 = operator_L_from_u(baseflow_u(0.0, solver3), solver3)
eigs = eigvals(L)
eigs2 = eigvals(L2)
eigs3 = eigvals(L3)

print("Max Floquet Multiplier:", np.max(np.abs(results['rho'])))
print("Max Growth Rate:", np.max(results['growth_rates']))
print("Max Singular Value:", np.max(results['svals_M']))
print("Max Sigma Max Phi:", np.max(results['sigma_max_Phi']))
print("")
print("Comparison to using Eigenvalues of frozen L at t=0:")
print("Max Floquet Multiplier (frozen):", np.exp(T * np.max(eigs.real)))
print("Max Growth Rate (frozen):", np.max(eigs.real))
svals_frozen = svd(expm(L * T), compute_uv=False)
print("Max Singular Value (frozen):", svals_frozen[0])
mu2 = np.max(eigvals(0.5 * (L + L.T)).real)
print("mu2(L) (log norm bound rate):", mu2)
print("Bound on sigma_max(exp(L*T)):", np.exp(mu2 * T))

solver.check_eigs(A=results['M'], print_nothing=True)

order = [2, 1, 0]

normalize_eigs = False
if normalize_eigs:
    dx = solver.mesh.x[1] - solver.mesh.x[0]
    dx2 = solver2.mesh.x[1] - solver2.mesh.x[0]
    dx3 = solver3.mesh.x[1] - solver3.mesh.x[0]
    eigs = eigs * dx
    eigs2 = eigs2 * dx2
    eigs3 = eigs3 * dx3

plt.figure(figsize=(4.5, 4))
ax = plt.gca()
if normalize_eigs:
    plt.xlabel(r'$\Re{(\lambda \Delta x)}$', fontsize=14)
    plt.ylabel(r'$\Im{(\lambda \Delta x)}$', fontsize=14)
else:
    plt.xlabel(r'$\Re{(\lambda)}$', fontsize=14)
    plt.ylabel(r'$\Im{(\lambda)}$', fontsize=14)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-230, 230)
plt.scatter(eigs3.real, eigs3.imag, color='tab:blue',
            label=r'$N=100$', marker='o', s=20)
plt.scatter(eigs2.real, eigs2.imag, color='tab:orange',
            label=r'$N=60$', marker='x', s=20)
plt.scatter(eigs.real, eigs.imag, color='tab:red',
            label=r'$N=40$', marker='+', s=20)
handles, labels = ax.get_legend_handles_labels()
plt.legend([handles[i] for i in order], [labels[i] for i in order],
            fontsize=14, loc='lower left')
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.figure(figsize=(4.5, 4))
ax = plt.gca()
plt.ylabel(r'$\Im{(\lambda)}$', fontsize=14)
if normalize_eigs:
    plt.xlabel(r'$\Re{(\lambda \Delta x)}$', fontsize=14)
    plt.scatter(results3['exponents'].real * dx3, results3['exponents'].imag,
                    color='tab:blue', label=r'$N=100$', marker='o', s=20)
    plt.scatter(results2['exponents'].real * dx2, results2['exponents'].imag,
                    color='tab:orange', label=r'$N=60$', marker='x', s=20)
    plt.scatter(results['exponents'].real * dx, results['exponents'].imag,
                    color='tab:red', label=r'$N=40$', marker='+', s=20)
else:
    plt.xlabel(r'$\Re{(\lambda)}$', fontsize=14)
    plt.scatter(results3['exponents'].real, results3['exponents'].imag,
                    color='tab:blue', label=r'$N=100$', marker='o', s=20)
    plt.scatter(results2['exponents'].real, results2['exponents'].imag,
                    color='tab:orange', label=r'$N=60$', marker='x', s=20)
    plt.scatter(results['exponents'].real, results['exponents'].imag,
                    color='tab:red', label=r'$N=40$', marker='+', s=20)
handles, labels = ax.get_legend_handles_labels()
plt.legend([handles[i] for i in order], [labels[i] for i in order],
            fontsize=14, loc='lower left')
plt.tick_params(axis='both', labelsize=14)
plt.ylim(-3.3, 3.3)
plt.ticklabel_format(axis='x', style='sci', scilimits=(-4, 4))
ax.xaxis.get_offset_text().set_fontsize(14)
ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_yticklabels(
    [r'$-\pi$', r'$\displaystyle -\frac{\pi}{2}$', r'$0$',
        r'$\displaystyle \frac{\pi}{2}$', r'$\pi$'])
plt.tight_layout()
plt.show()