"""
Variable-Coefficient Linear Convection Equation (VCLCE): 
- computes the eigenspectra, coloured by boundary content
- plots the unstable eigenvectors and their local growth contributions
- plots the local growth contributions for fourier modes
- solves the VCLCE and tracks the errors, plotting alongside predictions from eigenspectra
This reproduces figures from section 2.6 and 2.7 of the paper.
"""

import os
from sys import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

plt.rcParams.update({
    "text.usetex": True,              # use full LaTeX
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from Source.DiffEq.VarCoeffLinearConv import LinearConv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp
from Source.Solvers.PdeSolverCSbp import PdeSolverCSbp

run_sim = True
plot_eigs = True
plot_resolved = True
include_2nd_mode = False
savefile = None #'vce_csbp_nd_p1_40n_alpha00'

# Eq parameters
alpha = 0. # Variable coefficient splitting parameter (0 to 1)
use_exact_der = False # whether to compute variable coefficient derivative exactly
extrapolate_bdy_flux = True

# Time marching
tm_method = 'rk8' # explicit_euler, rk4
dt = 0.001
tf = 4.0

# Domain
xmin = 0.
xmax = 1.
bc = 'periodic' # 'periodic' or 'homogeneous'

# Spatial discretization
flux_type = 'product' # 'product' or 'geometric'
disc_nodes = 'csbp' # 'lg', 'lgl', 'nc', 'csbp', 'dg', 'fd'
p = 1
nelem = 1 # optional, number of elements
nen = 40 # optional, number of nodes per element
surf_type = {'diss_type':'nd', 'coeff':1.0, 'alpha':1.0, 'p':p}
vol_diss = {'diss_type':'nd', 'use_H':False, 'bdy_fix':False, 
            #'jac_type':'scalar', 's':int(p/2)+1, 'coeff':0.625/5**(int(p/2)+1)}
            'jac_type':'scalar', 's':p+1, 'coeff':0.625/5**(p+1)}

# Initial solution
q0_type = 'GaussWave' #'small_gaussian' # 'perturbation', 'random_shift' #'GaussWave_shift' # 'GaussWave', 'SinWave'
a_type = 'skewed_sin'

# Other
cons_obj_name = ('Error','Max_Error','Norm','Linf_Norm','time')
#cons_obj_name = ('Norm','Linf_Norm','time')
settings = {'warp_factor':0,               # Warps / stretches mesh.
            'warp_type': 'default',         # Options: 'defualt', 'papers', 'quad'
            'metric_method':'exact',   # Options: 'calculate', 'exact'
            'jac_method':'exact'} 

''' Set diffeq and solve '''
    
diffeq = LinearConv(alpha, q0_type, a_type, flux_type) 
diffeq.use_exact_der = use_exact_der
diffeq.extrapolate_bdy_flux = extrapolate_bdy_flux

solver = PdeSolverSbp(diffeq, settings,             
                  tm_method, dt, tf,                  
                  p, 'div',           
                  surf_type, vol_diss, None,
                  nelem, nen, disc_nodes,
                  bc, xmin, xmax,    
                  cons_obj_name,   
                  False, False)
solver.tm_atol = 1e-12
solver.tm_rtol = 1e-12

J = solver.calc_RHS_jac(q=np.ones(solver.qshape))
#if alpha == 1.0:
#    J1 = np.copy(J)
#if alpha == 0.0:
#   J0 = np.copy(J)
#try:
#    Theta = J1 - J0
#except:
#    Theta = None
#    print('WARNING: J1 and J0 are not properly defined yet')

if plot_eigs:
    H = solver.H_phys.flatten('F')
    HA = solver.H_phys.flatten('F') * solver.diffeq.a.flatten('F')
    S = 0.5 * (J.T * H[None,:] + H[:,None] * J)
    SA = 0.5 * (J.T * HA[None,:] + HA[:,None] * J)
    #S1 = 0.5 * (J1.T * H[None,:] + H[:,None] * J1)
    #if Theta is not None:
    #    S_theta = 0.5 * (Theta.T * H[None,:] + H[:,None] * Theta)


# Compute both left and right eigenvectors simultaneously
# This is numerically stable and efficient
eigs, eigvecs_left, eigvecs_right = eig(J, left=True, right=True)

# Normalize so that left^H @ right = 1 (biorthonormal)
for i in range(len(eigs)):
    denom = np.vdot(eigvecs_left[:, i], eigvecs_right[:, i])
    if abs(denom) < 1e-300:
        raise RuntimeError(f"Mode {i}: w^H v is ~0; can't normalize.")
    eigvecs_right[:, i] /= denom  
# sanity check:
#B = eigvecs_left.conj().T @ eigvecs_right                         # should be ~ I on diagonal
#print("max |diag(B)-1| =", np.max(np.abs(np.diag(B) - 1)))
#print("max |offdiag(B)| =", np.max(np.abs(B - np.diag(np.diag(B)))))    

# Find index of eigenvalue with max real part
idx_max = np.argmax(np.real(eigs))
eigval = eigs[idx_max]
max_real_eig = np.real(eigval)
max_imag_eig = np.imag(eigval)
eigvec_right = eigvecs_right[:, idx_max]
eigvec_left = eigvecs_left[:, idx_max] 

if include_2nd_mode:
    idx_sorted = np.argsort(np.real(eigs))   # ascending
    idx_third = idx_sorted[-3]              # 3rd largest
    eigval2 = eigs[idx_third]
    max_imag_eig2 = np.imag(eigval2)
    max_real_eig2 = np.real(eigval2)
    eigvec_right2 = eigvecs_right[:, idx_third]
    eigvec_left2 = eigvecs_left[:, idx_third]

def make_symmetric_ax(ax):
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    yabs = max(abs(ymin), abs(ymax))
    ax.set_ylim(-yabs, yabs)

def align_zero_lines(ax1, ax2):
    """
    Align the y=0 horizontal line of two y-axes (e.g. twin axes)
    without forcing symmetry or fixed bounds.
    """
    # Ensure limits reflect plotted data
    for ax in (ax1, ax2):
        ax.relim()
        ax.autoscale_view()

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    # Only makes sense if zero is inside both ranges
    if not (ymin1 < 0 < ymax1 and ymin2 < 0 < ymax2):
        return
    # Relative zero position on ax1
    r = (0 - ymin1) / (ymax1 - ymin1)
    # Adjust ax2 so zero has same relative position
    yrange2 = ymax2 - ymin2
    new_ymin2 = -r * yrange2
    new_ymax2 = (1 - r) * yrange2
    ax2.set_ylim(new_ymin2, new_ymax2)

if plot_eigs:
    if savefile is not None: savefile1 = savefile + '_eigvals'
    else: savefile1 = None
    solver.check_eigs(A=J, plot_maxvec=False, test_type='max real', num_vecs=3, 
                        colour_by_bdy=True, log_colourbar=True, figsize=(5,4),
                        #ymax=140, ymin=-140, xmin=-0.08, xmax=0.2,
                        ymax=140, ymin=-140, xmin=-1.2, xmax=3.4,
                        #ymax=140, ymin=-140, xmin=-21, xmax=1.5,
                        xlabel=r'$\Re{(\lambda)}$', ylabel=r'$\Im{(\lambda)}$',
                        label_fontsize=14, title=None, legend_fontsize=14,
                        savefile=savefile1, save_format='pdf', overwrite=True)
    separate_axes = False

    # Plot eigenvector, contribution to real eigenvalue, and baseflow all on same plot
    norm = np.max(np.abs(eigvec_right))
    denom = np.real( np.vdot(eigvec_right, H * eigvec_right) )
    g = np.real( np.conjugate(eigvec_right) * ( S @ eigvec_right ) ) / denom
    ga = np.real( np.conjugate(eigvec_right) * ( SA @ eigvec_right ) ) / np.real( np.vdot(eigvec_right, HA * eigvec_right) )
    g_exact = -0.5 * solver.diffeq.ader.flatten('F') * np.real( np.conjugate(eigvec_right) * eigvec_right ) * H / denom
    #g1 = np.real( np.conjugate(eigvec_right) * ( S1 @ eigvec_right ) ) / denom
    #if Theta is not None:
    #    g_theta = np.real( np.conjugate(eigvec_right) * ( S_theta @ eigvec_right ) ) / denom
    fig, ax1 = plt.subplots(figsize=(5,4))
    ax2 = ax1.twinx()
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_xlim(solver.mesh.xmin, solver.mesh.xmax)
    ax2.set_ylabel(r"$a$", fontsize=15, rotation=0)
    ax2.set_ylim(0.49,2.51)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.plot(solver.mesh.x, solver.diffeq.a.flatten('F'), color='k', linestyle='--', linewidth=1, label=r'$a(x)$')
    ax1.set_ylabel(r"$\boldsymbol{\phi}$", fontsize=15, rotation=0)
    ax1.set_ylim(-1.1,1.1)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.plot(solver.mesh.x, np.abs(eigvec_right) / norm, label=r'$\lvert \boldsymbol{\phi} \rvert$', color='tab:red', linestyle='--', linewidth=2)
    ax1.plot(solver.mesh.x, eigvec_right.real / norm, label=r'$\Re{(\boldsymbol{\phi})}$', color='tab:blue', linestyle=(0, (1, 1)), linewidth=2)
    ax1.plot(solver.mesh.x, eigvec_right.imag / norm, label=r'$\Im{(\boldsymbol{\phi})}$', color='tab:orange', linestyle=(1, (1, 1)), linewidth=2)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax2.legend(handles, labels, loc="upper left", fontsize=14)
    fig.tight_layout()
    if savefile is not None: fig.savefig(savefile + '_eigvecs1.pdf')
    
    fig, ax1 = plt.subplots(figsize=(5,4))
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_xlim(solver.mesh.xmin, solver.mesh.xmax)
    ax1.set_ylabel(r"$\boldsymbol{g}$", fontsize=15, rotation=0)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.plot(solver.mesh.x, g, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='-', color='tab:orange', linewidth=2)
    ax1.plot(solver.mesh.x, g_exact, label=r'$\frac{-\tfrac{1}{2} a_x \lvert \phi_i \rvert^2 \mathsf{H}_i}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='dashed', color='tab:red', linewidth=2)
    #ax1.plot(solver.mesh.x, g1, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{\mathsf{H}}^{\alpha=1} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='--', color='tab:purple', linewidth=2)
    #if Theta is not None:
    #    ax1.plot(solver.mesh.x, g_theta, label=r'$\frac{\Re{(\phi_i (\mathsf{\Theta}^\mathsf{S}_{\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='--', color='tab:green', linewidth=2)
    if separate_axes:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\boldsymbol{g}_a$", fontsize=18, rotation=0)
        ax2.tick_params(axis='both', labelsize=13)
        ax2.plot(solver.mesh.x, ga, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{a\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{a\mathsf{H}}}$', linestyle='--', color='tab:blue', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax2.legend(handles, labels, loc="best", fontsize=14)
        make_symmetric_ax(ax1)
        make_symmetric_ax(ax2)
        #align_zero_lines(ax1, ax2)
    else:
        ax1.plot(solver.mesh.x, ga, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{a\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{a\mathsf{H}}}$', linestyle='--', color='tab:blue', linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
        #ax1.legend(loc="best", fontsize=14)
        ax1.legend(loc="upper left", fontsize=14)
    fig.tight_layout()
    if savefile is not None: fig.savefig(savefile + '_eigval_contrib1.pdf')

    if include_2nd_mode:
        # Do it again for the 3rd largest eigenvalue (skipping complex conjugate)
        norm = np.max(np.abs(eigvec_right2))
        denom = np.real( np.vdot(eigvec_right2, H * eigvec_right2) )
        g2 = np.real( np.conjugate(eigvec_right2) * ( S @ eigvec_right2 ) ) / denom
        ga2 = np.real( np.conjugate(eigvec_right2) * ( SA @ eigvec_right2 ) ) / np.real( np.vdot(eigvec_right2, HA * eigvec_right2) )
        g_exact2 = -0.5 * solver.diffeq.ader.flatten('F') * np.real( np.conjugate(eigvec_right2) * eigvec_right2 ) * H  / denom
        fig, ax1 = plt.subplots(figsize=(5,4))
        ax2 = ax1.twinx()
        ax1.set_xlabel(r'$x$', fontsize=15)
        ax1.set_xlim(solver.mesh.xmin, solver.mesh.xmax)
        ax2.set_ylabel(r"$a$", fontsize=15, rotation=0)
        ax2.set_ylim(0.49,2.51)
        ax2.tick_params(axis='both', labelsize=13)
        ax2.plot(solver.mesh.x, solver.diffeq.a.flatten('F'), color='k', linestyle='--', linewidth=1, label=r'$a(x)$')
        ax1.set_ylabel(r"$\boldsymbol{\phi}$", fontsize=15, rotation=0)
        ax1.set_ylim(-1.1,1.1)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.plot(solver.mesh.x, np.abs(eigvec_right2) / norm, label=r'$\lvert \boldsymbol{\phi} \rvert$', color='tab:red', linestyle='--', linewidth=2)
        ax1.plot(solver.mesh.x, eigvec_right2.real / norm, label=r'$\Re{(\boldsymbol{\phi})}$', color='tab:blue', linestyle=(0, (1, 1)), linewidth=2)
        ax1.plot(solver.mesh.x, eigvec_right2.imag / norm, label=r'$\Im{(\boldsymbol{\phi})}$', color='tab:orange', linestyle=(1, (1, 1)), linewidth=2)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax2.legend(handles, labels, loc="best", fontsize=14)
        fig.tight_layout()
        if savefile is not None: fig.savefig(savefile + '_eigvecs2.pdf')
        
        fig, ax1 = plt.subplots(figsize=(5,4))
        ax1.set_xlabel(r'$x$', fontsize=15)
        ax1.set_xlim(solver.mesh.xmin, solver.mesh.xmax)
        ax1.set_ylabel(r"$\boldsymbol{g}$", fontsize=15, rotation=0)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.plot(solver.mesh.x, g2, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='-', color='tab:orange', linewidth=2)
        ax1.plot(solver.mesh.x, g_exact2, label=r'$\frac{-\tfrac{1}{2} a_x \lvert \phi_i \rvert^2 \mathsf{H}_i}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', linestyle='dashed', color='tab:red', linewidth=2)
        if separate_axes:
            ax2 = ax1.twinx()
            ax2.set_ylabel(r"$\boldsymbol{g}_a$", fontsize=18, rotation=0)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.plot(solver.mesh.x, ga2, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{a\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{a\mathsf{H}}}$', linestyle='--', color='tab:blue', linewidth=2)
            ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            labels = labels1 + labels2
            ax2.legend(handles, labels, loc="best", fontsize=14)
            make_symmetric_ax(ax1)
            make_symmetric_ax(ax2)
            #align_zero_lines(ax1, ax2)
        else:
            ax1.plot(solver.mesh.x, ga2, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{a\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{a\mathsf{H}}}$', linestyle='--', color='tab:blue', linewidth=2)
            ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax1.legend(loc="best", fontsize=14)
        fig.tight_layout()
        if savefile is not None: fig.savefig(savefile + '_eigval_contrib2.pdf')

    # Do it for a Fourier mode (commented out is a quasi-well-resolved mode)
    if plot_resolved:
        m = 10
        k = 2*np.pi*m/solver.mesh.dom_len
        #u = 0.25*np.sin(k*np.pi*solver.mesh.x) + 1
        u = np.exp(1j*k*solver.mesh.x) #np.sin(k*np.pi*solver.mesh.x)
        #ux = 10*np.pi*0.25*np.cos(10*np.pi*solver.mesh.x)
        #uxx = -10*np.pi*10*np.pi*0.25*np.sin(10*np.pi*solver.mesh.x)
        denom = np.real( np.vdot(u, H * u) )
        g = np.real( np.conjugate(u) * ( S @ u ) ) / denom
        ga = np.real( np.conjugate(u) * ( SA @ u ) ) / np.real( np.vdot(u, HA * u) )
        g_exact = -0.5 * solver.diffeq.ader.flatten('F') * np.real( np.conjugate(u) * u ) * H / denom
        #g_theta = np.real( np.conjugate(u) * ( S_theta @ u ) ) / denom
        fig, ax1 = plt.subplots(figsize=(5,4))
        ax2 = ax1.twinx()
        ax1.set_xlabel(r'$x$', fontsize=15)
        ax1.set_xlim(solver.mesh.xmin, solver.mesh.xmax)
        ax2.set_ylabel(r"$a$", fontsize=15, rotation=0)
        ax2.set_ylim(0.49,2.51)
        ax2.tick_params(axis='both', labelsize=13)
        ax2.plot(solver.mesh.x, solver.diffeq.a.flatten('F'), color='k', linestyle='--', linewidth=1, label=r'$a(x)$')
        #ax1.plot(solver.mesh.x, u, color='k', linestyle=':', linewidth=1, label=r'$\boldsymbol{v}$')
        ax1.set_ylabel(r"$\boldsymbol{g}$", fontsize=15, rotation=0)
        ax1.set_ylim(-0.3,0.13)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax1.plot(solver.mesh.x, g, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{\mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', color='tab:orange', linestyle='-', linewidth=2)
        ax1.plot(solver.mesh.x, g_exact, label=r'$\frac{-\tfrac{1}{2} a_x \lvert \phi_i \rvert^2 \mathsf{H}_i}{\|\boldsymbol{\phi}\|^2_{\mathsf{H}}}$', color='tab:red', linestyle='--', linewidth=2)
        #if Theta is not None:
        #    ax2.plot(solver.mesh.x, g_theta, label=r'$\frac{\Re{(u_i (\mathsf{\Theta}^\mathsf{S}_{\mathsf{H}} \boldsymbol{u})_i )}}{\|\boldsymbol{u}\|^2_{\mathsf{H}}}$', color='tab:purple', linestyle='--', linewidth=2)
        ax1.plot(solver.mesh.x, ga, label=r'$\frac{\Re{(\phi_i (\mathsf{S}_{a \mathsf{H}} \boldsymbol{\phi})_i )}}{\|\boldsymbol{\phi}\|^2_{a\mathsf{H}}}$', color='tab:blue', linestyle='--', linewidth=2)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax2.legend(handles, labels, loc="lower right", fontsize=14, framealpha=0.8)
        fig.tight_layout()
        if savefile is not None: fig.savefig(savefile + '_mode.pdf')

        # plt.figure()
        # g_avg = np.zeros_like(g)
        # g_exact_avg = np.zeros_like(g_exact)
        # g_theta_avg = np.zeros_like(g_theta)
        # w = nen//m + 1 # window length (odd is nicest); choose e.g. ~ points per wavelength
        # half = w // 2
        # for k in range(-half, half + 1):
        #     g_avg += np.roll(g, -k)   # shift so neighbors align
        #     g_exact_avg += np.roll(g_exact, -k)
        #     g_theta_avg += np.roll(g_theta, -k)
        # g_avg /= w
        # g_exact_avg /= w
        # g_theta_avg /= w
        # plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
        # plt.plot(solver.mesh.x, g_avg, label=r'$\frac{\Re{(u_i (\mathsf{S}_{\mathsf{H}} \boldsymbol{u})_i )}}{\|\boldsymbol{u}\|^2_{\mathsf{H}}}$', color='tab:orange', linestyle='-', linewidth=2)
        # plt.plot(solver.mesh.x, g_exact_avg, label=r'$\frac{-\tfrac{1}{2} a_x \lvert u_i \rvert^2 \mathsf{H}_i}{\|\boldsymbol{u}\|^2_{\mathsf{H}}}$', color='tab:red', linestyle='--', linewidth=2)
        # plt.plot(solver.mesh.x, g_theta_avg, label=r'$\frac{\Re{(u_i (\mathsf{\Theta}^\mathsf{S}_{\mathsf{H}} \boldsymbol{u})_i )}}{\|\boldsymbol{u}\|^2_{\mathsf{H}}}$', color='tab:purple', linestyle='--', linewidth=2)
        # plt.legend(fontsize=14)


if run_sim:
    if q0_type == 'perturbation':
        # Take real part of right eigenvector for reshaping
        pert = eigvec_right.real * 1e-3 / np.max(eigvec_right.real)
        solver.diffeq.set_q0_discrete(np.reshape(pert, solver.qshape, 'F'), periodic=False)
    elif q0_type == 'small_gaussian':
        solver.diffeq.q0_max_q = 0.01
        solver.diffeq.q0_gauss_wave_val_bc = 1e-12
        solver.diffeq.q0_type = 'GaussWave'
    
    solver.solve()
    #solver.plot_cons_obj(plot_change=False)

    # Find indices automatically from cons_obj_name
    norm_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() == 'norm'][0]
    linf_norm_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() in ['linf_norm', 'max']][0]
    time_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() == 'time'][0]
    error_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() == 'error'][0]
    max_error_idx = [j for j in range(len(solver.cons_obj_name)) if solver.cons_obj_name[j].lower() == 'max_error'][0]

    # Extract data
    norm = solver.cons_obj[norm_idx, :]
    linf_norm = solver.cons_obj[linf_norm_idx, :]
    time = solver.cons_obj[time_idx, :]
    error = solver.cons_obj[error_idx, :]
    max_error = solver.cons_obj[max_error_idx, :]

    # Predict eigenvalue growth:
    # After long times, we can say q(t) = \alpha e^{lambda t} v + \alpha^* e^{lambda^* t} v^*
    # where v is the right eigenvector, and v^* is its complex conjugate.
    # This simplifies to q(t) = 2 * Re(\alpha e^{i Im(\lambda) t} v)
    # If I ignore the complex oscillations, we further simplify to q(t) = 2 * Re(\alpha) e^{Re(\lambda) t} Re(v)
    # then \norm{q} = 2 * Re(\alpha) e^{Re(\lambda) t} \norm{Re(v)}
    # or on a semi-log y plot, \log(\norm{q}) = C + Re(\lambda) t, where C = \log{ 2 * Re(\alpha) \norm{Re(v)} }
    # How do I find \alpha though?
    # introduce the left eigenvectors w s.t. < w_i, v_j > = delta_ij
    # then for any q_0 = \sum_i \alpha_i v_i + \alpha_i^* v_i^*
    # we have < w_i, q_0 > = \sum_j \alpha_j < w_i, v_j > + \alpha_j^* < w_i, v_j^* > = \alpha_i 
    # therefore to calculate, use \alpha_i = < w_i, q_0 >

    # Compute projection coefficient: \alpha_i = < w_i, q_0 >
    projection_coeff = np.vdot(eigvec_left, solver.q_sol[:,:,0].flatten('F')) 
    if include_2nd_mode:
        projection_coeff2 = np.vdot(eigvec_left2, solver.q_sol[:,:,0].flatten('F')) 


    # Compute exact prediction vectors
    coeff_a = projection_coeff.real
    coeff_b = projection_coeff.imag
    vec_u = coeff_a * eigvec_right.real - coeff_b * eigvec_right.imag
    vec_w = coeff_a * eigvec_right.imag + coeff_b * eigvec_right.real
    if include_2nd_mode:
        coeff_a2 = projection_coeff2.real
        coeff_b2 = projection_coeff2.imag
        vec_u2 = coeff_a2 * eigvec_right2.real - coeff_b2 * eigvec_right2.imag
        vec_w2 = coeff_a2 * eigvec_right2.imag + coeff_b2 * eigvec_right2.real
    # Plot norms and exact prediction
    plt.figure(figsize=(5, 4))
    plt.semilogy(time, norm, label=r'$\| \boldsymbol{u} \|_\mathsf{H}$', linestyle='-', color='tab:blue', linewidth=3)
    plt.semilogy(time, linf_norm, label=r'$\| \boldsymbol{u} \|_\infty$', linestyle='-', color='tab:red', linewidth=3)

    # Compute exact prediction
    time_extended = np.linspace(0, time.max(), 100)
    theory_line_norm_exact = np.zeros_like(time_extended)
    theory_line_linf_exact = np.zeros_like(time_extended)
    for ti,t in enumerate(time_extended):
        vec = vec_u * np.cos(max_imag_eig * t) - vec_w * np.sin(max_imag_eig * t)
        theory_line_norm_exact[ti] = 2 * np.exp(max_real_eig * t) * solver.norm(np.reshape(vec.real, solver.qshape, 'F'))
        theory_line_linf_exact[ti] = 2 * np.exp(max_real_eig * t) * np.max(np.abs(vec.real))
    if include_2nd_mode:
        theory_line_norm_exact2 = np.zeros_like(time_extended)
        theory_line_linf_exact2 = np.zeros_like(time_extended)
        for ti,t in enumerate(time_extended):
            vec2 = vec_u2 * np.cos(max_imag_eig2 * t) - vec_w2 * np.sin(max_imag_eig2 * t)
            theory_line_norm_exact2[ti] = 2 * np.exp(max_real_eig2 * t) * solver.norm(np.reshape(vec2.real, solver.qshape, 'F'))
            theory_line_linf_exact2[ti] = 2 * np.exp(max_real_eig2 * t) * np.max(np.abs(vec2.real))
        # Create masks to stop plotting second mode when first mode becomes larger
        mask_norm2 = theory_line_norm_exact2 >= theory_line_norm_exact
        mask_linf2 = theory_line_linf_exact2 >= theory_line_linf_exact
    
    plt.semilogy(time_extended, theory_line_norm_exact, label=r'$\| \boldsymbol{u}_\mathrm{pred} \|_\mathsf{H}$', linestyle=':', color='tab:blue', linewidth=3)
    plt.semilogy(time_extended, theory_line_linf_exact, label=r'$\| \boldsymbol{u}_\mathrm{pred} \|_\infty$', linestyle=':', color='tab:red', linewidth=3)
    if include_2nd_mode:
        plt.semilogy(time_extended[mask_norm2], theory_line_norm_exact2[mask_norm2], linestyle=':', color='tab:blue', linewidth=3)
        plt.semilogy(time_extended[mask_linf2], theory_line_linf_exact2[mask_linf2], linestyle=':', color='tab:red', linewidth=3)
    plt.semilogy(time, error, linestyle='-.', color='tab:green', label=r'$\| \boldsymbol{\mathcal{E}} \|_\mathsf{H}$', linewidth=2)
    plt.semilogy(time, max_error, linestyle='-.',color='tab:orange', label=r'$\| \boldsymbol{\mathcal{E}} \|_\infty$', linewidth=2)

    plt.xlabel(r'$t$', fontsize=14)
    #plt.ylabel('Norm (log scale)', fontsize=12)
    #plt.title('Perturbation Growth', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.ylim(ymin=min(1e-3, theory_line_norm_exact[0]/10))
    if savefile is not None: plt.savefig(savefile + '_error.pdf')
    plt.show()


# time_10 = np.copy(time)
# error_10 = np.copy(error)
# max_error_10 = np.copy(max_error)
# norm_10 = np.copy(norm)
# linf_norm_10 = np.copy(linf_norm)
# plt.figure(figsize=(5, 4))
# time0 = 20
# time1 = 25
# idx0 = np.argmin(np.abs(time_00-time0))
# idx1 = np.argmin(np.abs(time_00-time1))
# plt.semilogy(time_00[idx0:idx1+1],error_00[idx0:idx1+1],label=r'$\alpha=0$', linewidth=2, linestyle='-', color='tab:red')
# idx0 = np.argmin(np.abs(time_05-time0))
# idx1 = np.argmin(np.abs(time_05-time1))
# plt.semilogy(time_05[idx0:idx1+1],error_05[idx0:idx1+1],label=r'$\alpha=\tfrac{1}{2}$', linewidth=2, linestyle='-', color='tab:orange')
# idx0 = np.argmin(np.abs(time_23-time0))
# idx1 = np.argmin(np.abs(time_23-time1))
# plt.semilogy(time_23[idx0:idx1+1],error_23[idx0:idx1+1],label=r'$\alpha=\tfrac{2}{3}$', linewidth=2, linestyle='-',color='tab:green')
# idx0 = np.argmin(np.abs(time_10-time0))
# idx1 = np.argmin(np.abs(time_10-time1))
# plt.semilogy(time_10[idx0:idx1+1],error_10[idx0:idx1+1],label=r'$\alpha=1$', linewidth=2, linestyle='-',color='tab:blue')
# plt.xlabel(r'$t$', fontsize=14)
# plt.ylabel(r'$\| \boldsymbol{\mathcal{E}} \|_\mathsf{H}$', rotation=0, fontsize=14)
# plt.legend(loc='center left', fontsize=13)
# plt.tight_layout()
# plt.savefig('vce_mattsson_lf_p4_100n_error.pdf')