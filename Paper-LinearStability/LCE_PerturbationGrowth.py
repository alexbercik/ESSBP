"""
Linear Convection Equation (LCE): 
Solves the LCE with three different schemes, then adds a perturbation
and solves again. The perturbation evolution is isolated, 
reproducing figures from section 3.2 of the paper.
"""

import os
from sys import path
import numpy as np
import matplotlib.pyplot as plt

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.LinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp

''' Set parameters for simultation '''

# Initial solution
q0_type = 'GaussWave_shift' #'GaussWave_shift' # 'GaussWave_shift', 'SquareWave_shift'

# Time marching
tm_method = 'rk4' # use rk4 or rk8_verner to ensure that times line up
dt = 0.0001
tf = 10.0

# Spatial discretization
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd', 'upwind'
p = 1
nelem = 1 # number of elements
nen1 = 100 # nodes per element for solver 1
nen2 = 100 # nodes per element for solver 2
nen3 = 100 # nodes per element for solver 3
label1 = 'Central'
label2 = 'Logarithmic'
label3 = 'Geometric'
savefile = None #'lce_csbp_p1_100n_nd'

had_flux1 = 'central' # 2-point numerical flux used in hadamard form.
had_flux2 = 'logarithmic' # 2-point numerical flux used in hadamard form.
had_flux3 = 'geometric' # 2-point numerical flux used in hadamard form.

plot_prediction = False
perturbation = 1 # if 0, use nothing. If 1, use random noise. If 2, use eigenmode from log. If 3, use eigenmode from geom.

surf_type = {'diss_type':'nd', 'fluxvec':'lf', 'coeff':1.}
vol_diss = {'diss_type':'nd', 'use_H':False, 'bdy_fix':False, 
            'jac_type':'scalar', 's':p, 'coeff':3.125/5**(p+1), 'beta':4, 
            'fluxvec':'lf', 'eps_type':2, 'D_type':'sbp'}

cons_obj_name = ('Energy', 'error', 'max_error', 'entropy', 'time')

''' Set diffeq and solve '''

diffeq1 = LinearConv(1.0, q0_type, had_flux=had_flux1)
solver1 = PdeSolverSbp(diffeq1, None,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  p, 'had',             # Discretization
                  surf_type, vol_diss, had_flux1,
                  nelem, nen1, disc_nodes,
                  'periodic', 0., 1.,         # Domain
                  cons_obj_name,              # Other
                  print_progress=True)
solver1.tm_nframes = 1000
solver1.tm_atol = 1e-14
solver1.tm_rtol = 1e-14

diffeq2 = LinearConv(1.0, q0_type, had_flux=had_flux2)
solver2 = PdeSolverSbp(diffeq2, None,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  p, 'had',             # Discretization
                  surf_type, vol_diss, had_flux2,
                  nelem, nen2, disc_nodes,
                  'periodic', 0., 1.,         # Domain
                  cons_obj_name,              # Other
                  print_progress=True)
solver2.tm_nframes = 1000
solver2.tm_atol = 1e-14
solver2.tm_rtol = 1e-14

diffeq3 = LinearConv(1.0, q0_type, had_flux=had_flux3)
solver3 = PdeSolverSbp(diffeq3, None,                     # Diffeq
                  tm_method, dt, tf,                    # Time marching
                  p, 'had',             # Discretization
                  surf_type, vol_diss, had_flux3,
                  nelem, nen3, disc_nodes,
                  'periodic', 0., 1.,         # Domain
                  cons_obj_name,              # Other
                  print_progress=True)
solver3.tm_nframes = 1000
solver3.tm_atol = 1e-14
solver3.tm_rtol = 1e-14

''' Analyze results '''
solver1.check_eigs(plot_maxvec=True, test_type='max real', num_vecs=1)
eigvals2, eigvecs2 = solver2.check_eigs(plot_maxvec=True, test_type='max real', num_vecs=1, returnvecs=True, returneigs=True)
eigvals3, eigvecs3 = solver3.check_eigs(plot_maxvec=True, test_type='max real', num_vecs=1, returnvecs=True, returneigs=True)

q0_1 = diffeq1.set_q0()
q0_2 = diffeq2.set_q0()
q0_3 = diffeq3.set_q0()
ampli = 1e-3
pred = None
pert_seed = 0

def random_noise(qshape, seed=pert_seed):
    """Uniform noise in [-1, 1]. Same seed + same shape => identical array."""
    rng = np.random.default_rng(seed=seed)
    return 2 * rng.random(qshape) - 1.

def resample_to_solver(q_src, solver_src, solver_dst):
    """Copy if the grids match; otherwise interpolate periodically onto solver_dst."""
    x_src = solver_src.diffeq.x_elem
    x_dst = solver_dst.diffeq.x_elem
    if solver_src.qshape == solver_dst.qshape and np.allclose(x_src, x_dst):
        return np.copy(q_src)
    xmin, xmax = solver_dst.xmin, solver_dst.xmax
    L = xmax - xmin
    x_s = np.asarray(x_src).reshape(-1, order='F')
    v_s = np.asarray(q_src).reshape(-1, order='F')
    x_d = np.asarray(x_dst).reshape(-1, order='F')
    order = np.argsort(x_s)
    x_sorted, v_sorted = x_s[order], v_s[order]
    x_ext = np.concatenate(([x_sorted[-1] - L], x_sorted, [x_sorted[0] + L]))
    v_ext = np.concatenate(([v_sorted[-1]], v_sorted, [v_sorted[0]]))
    x_d_mod = np.mod(x_d - xmin, L) + xmin
    v_d = np.interp(x_d_mod, x_ext, v_ext)
    return v_d.reshape(solver_dst.qshape, order='F')

def apply_perturbation(q0, eigvec):
    scale = np.max(np.abs(eigvec))
    if scale == 0:
        return q0
    return q0 + ampli * eigvec / scale

if perturbation == 1:
    eigvec1 = random_noise(solver1.qshape)
    eigvec2 = random_noise(solver2.qshape)
    eigvec3 = random_noise(solver3.qshape)
elif perturbation == 0:
    eigvec1 = eigvec2 = eigvec3 = 0.
elif perturbation == 2:
    eig_idx = np.argmax(eigvals2.real)
    eigvec2 = eigvecs2[:, eig_idx].real.reshape(solver2.qshape)
    eigvec1 = resample_to_solver(eigvec2, solver2, solver1)
    eigvec3 = resample_to_solver(eigvec2, solver2, solver3)
elif perturbation == 3:
    eig_idx = np.argmax(eigvals3.real)
    eigvec3 = eigvecs3[:, eig_idx].real.reshape(solver3.qshape)
    eigvec1 = resample_to_solver(eigvec3, solver3, solver1)
    eigvec2 = resample_to_solver(eigvec3, solver3, solver2)
else:
    raise ValueError(f"Invalid perturbation: {perturbation}")

q0_1 = apply_perturbation(q0_1, eigvec1)
q0_2 = apply_perturbation(q0_2, eigvec2)
q0_3 = apply_perturbation(q0_3, eigvec3)

solver1.solve(q0=q0_1)
#solver1.plot_cons_obj()
solver2.solve(q0=q0_2)
#solver2.plot_cons_obj()
solver3.solve(q0=q0_3)
#solver3.plot_cons_obj()


plt.figure(figsize=(5,4))
plt.ylabel('Energy (with perturbation)', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[0, :], label=label1, color='tab:blue')
plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[0, :], label=label2, color='tab:orange')
plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[0, :], label=label3, color='tab:green')
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(5,4))
plt.ylabel('Error (with perturbation)', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[1, :], label=label1, color='tab:blue')
plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[1, :], label=label2, color='tab:orange')
plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[1, :], label=label3, color='tab:green')
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(5,4))
plt.ylabel('Max Error (with perturbation)', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[2, :], label=label1, color='tab:blue')
plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[2, :], label=label2, color='tab:orange')
plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[2, :], label=label3, color='tab:green')
plt.legend(fontsize=14)
plt.show()

#D = solver2.Dx[:,:,0]
#w = np.sqrt(q0.flatten('F'))
#Theta_geom = (D @ np.diag(w) - np.diag(w) @ D - np.diag(D @ w)) @ np.diag(1/w)
#print(r"|| \Theta_geom ||_2 =", np.linalg.norm(Theta_geom,2))
#print(r"|| \diag{ Dw } ||_2 =", np.linalg.norm(np.diag(D @ w),2))
#print(r"|| \diag{ Dw } + \Theta_geom ||_2 =", np.linalg.norm(D @ np.diag(w) - np.diag(w) @ D,2))

if tm_method == 'rk4' or tm_method == 'rk8_verner':
    solver1_pert_qsol = np.copy(solver1.q_sol)
    solver2_pert_qsol = np.copy(solver2.q_sol)
    solver3_pert_qsol = np.copy(solver3.q_sol)
    # run again with no perturbation
    solver1.solve(q0=None)
    solver2.solve(q0=None)
    solver3.solve(q0=None)

    plt.figure(figsize=(5,4))
    #plt.ylabel('Energy (without perturbation)', fontsize=16)
    plt.ylabel(r'$\| \boldsymbol{u} \|_\mathsf{H}^2$ - $\| \boldsymbol{u}_0 \|_\mathsf{H}^2$', fontsize=16, labelpad=5)
    plt.xlabel(r'$t$', fontsize=16, labelpad=-5)
    plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[0, :] - solver1.cons_obj[0, 0], label=label1, color='tab:blue')
    plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[0, :] - solver2.cons_obj[0, 0], label=label2, color='tab:orange')
    plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[0, :] - solver3.cons_obj[0, 0], label=label3, color='tab:green')
    plt.yscale('symlog',linthresh=1e-11)
    #plt.ylim(-3e-10,3e-10)
    #plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if savefile is not None: plt.savefig(savefile + '_energy.pdf')
    plt.show()

    plt.figure(figsize=(5,4))
    ax = plt.gca()
    #plt.ylabel('Error (without perturbation)', fontsize=16)
    plt.ylabel(r'$\| \boldsymbol{\mathcal{E}} \|_\mathsf{H}$', rotation=0, fontsize=16, labelpad=22)
    plt.xlabel(r'$t$', fontsize=16, labelpad=-5)
    plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[1, :], label=label1, color='tab:blue')
    plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[1, :], label=label2, color='tab:orange')
    plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[1, :], label=label3, color='tab:green')
    #plt.legend(fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if savefile is not None: plt.savefig(savefile + '_error.pdf')
    plt.show()

    plt.figure(figsize=(5,4))
    plt.ylabel('Max Error (without perturbation)', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.plot(solver1.cons_obj[-1, :], solver1.cons_obj[2, :], label=label1, color='tab:blue')
    plt.plot(solver2.cons_obj[-1, :], solver2.cons_obj[2, :], label=label2, color='tab:orange')
    plt.plot(solver3.cons_obj[-1, :], solver3.cons_obj[2, :], label=label3, color='tab:green')
    #plt.legend(fontsize=14)
    plt.show()

    # Handle mismatched k dimensions due to early timeouts
    # Find minimum k dimension for each solver pair and truncate accordingly
    k1_min = min(solver1_pert_qsol.shape[2], solver1.q_sol.shape[2])
    k2_min = min(solver2_pert_qsol.shape[2], solver2.q_sol.shape[2])
    k3_min = min(solver3_pert_qsol.shape[2], solver3.q_sol.shape[2])
    
    solver1_pert = solver1_pert_qsol[:, :, :k1_min] - solver1.q_sol[:, :, :k1_min]
    solver2_pert = solver2_pert_qsol[:, :, :k2_min] - solver2.q_sol[:, :, :k2_min]
    solver3_pert = solver3_pert_qsol[:, :, :k3_min] - solver3.q_sol[:, :, :k3_min]

    plt.figure(figsize=(5,4))
    ax = plt.gca()
    #plt.ylabel('Perturbation H-Norm (perturbation)', fontsize=16)
    plt.ylabel(r'$\| \boldsymbol{v} \|_\mathsf{H}$', rotation=0, fontsize=16, labelpad=22)
    plt.xlabel(r'$t$', fontsize=16, labelpad=-5)
    plt.plot(solver1.cons_obj[-1, :k1_min], np.sqrt(solver1.energy(solver1_pert)), label=label1, color='tab:blue')
    plt.plot(solver2.cons_obj[-1, :k2_min], np.sqrt(solver2.energy(solver2_pert)), label=label2, color='tab:orange')
    plt.plot(solver3.cons_obj[-1, :k3_min], np.sqrt(solver3.energy(solver3_pert)), label=label3, color='tab:green')
    if pred is not None and plot_prediction:
        plt.plot(solver1.cons_obj[-1, :], np.sqrt(solver2.energy(solver2_pert[:,:,0]))*np.exp(pred*solver1.cons_obj[-1, :]), linestyle=':', label='Prediction', color='tab:red')
    #plt.legend(fontsize=14, loc='upper left')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if savefile is not None: plt.savefig(savefile + '_pert_H.pdf')

    plt.figure(figsize=(5,4))
    ax = plt.gca()
    #plt.ylabel('Perturbation L_infty norm', fontsize=16)
    plt.ylabel(r'$\| \boldsymbol{v} \|_{\infty}$', rotation=0, fontsize=16, labelpad=22)
    plt.xlabel(r'$t$', fontsize=16, labelpad=-5)
    plt.plot(solver1.cons_obj[-1, :k1_min], np.max(abs(solver1_pert), axis=0)[0], label=label1, color='tab:blue')
    plt.plot(solver2.cons_obj[-1, :k2_min], np.max(abs(solver2_pert), axis=0)[0], label=label2, color='tab:orange')
    plt.plot(solver3.cons_obj[-1, :k3_min], np.max(abs(solver3_pert), axis=0)[0], label=label3, color='tab:green')
    #plt.legend(fontsize=14, loc='upper right')
    plt.ylim(ymin=0.001)#,ymax=0.0027)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if savefile is not None: plt.savefig(savefile + '_pert_inf.pdf')
    plt.show()

    # Create a separate figure for the legend only
    fig_legend = plt.figure(figsize=(6, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    # Create dummy lines for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:blue', label=label1),
        Line2D([0], [0], color='tab:orange', label=label2),
        Line2D([0], [0], color='tab:green', label=label3)
    ]
    # Create legend with 3 columns (horizontal layout)
    legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=3, fontsize=16, frameon=True)
    plt.tight_layout()
    if savefile is not None: plt.savefig('lce_legend.pdf')
    plt.show()

    