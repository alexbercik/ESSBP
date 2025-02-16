#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:17:18 2021

@author: bercik
"""
import subprocess # see https://www.youtube.com/watch?v=2Fp1N6dof0Y for tutorial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tik
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm}'
from sympy import nsimplify
from os import path
import itertools
from tabulate import tabulate
import scipy.optimize as sc
import copy
import traceback

def animate(solver, file_name='animation', make_video=True, make_gif=False,
               plotfunc='plot_sol',plotargs={}, skipsteps=0,fps=24,
               last_frame=False,tfinal=None):
    '''
    Note: This uses bash calls with ffmpeg and ImageMagick, so have those installed
    Parameters
    An example of how to call this in the driver file:
    from Methods.Analysis import animate
    animate(solver, plotargs={'display_time':True},skipsteps=100)
    ----------
    solver : class instance
        the solver class instance of solved ODE/PDE
    file_name : str, optional
        Name of the file that is saved, either an mp4 or the individual frakes.
        The name can include a specific path (ex. ../../folder/filename) and
        should not include the file extension.
        The default is 'animation'.
    make_video : bool, optional
        If true, an mp4 is created with ffmpeg with all of the saved frames.
        The default is True.
    make_gif : bool, optional
        If true, a gif is created with ffmpeg with all of the saved frames.
        The default is True.
    plotfunc: str, optional
        Name of the method within solver used to plot. This method should take
        at least the following (keyword) arguments: 
            q : solution
            time (optional) : time at which to plot solution
            plt_save_name (optional) : name of saved file, without file extension
            show_fig (optional) : Boolean to be set to False
            ymin & ymax (optional) : y-axis limits
        The default is solver.plot_sol (calls solver.diffeq.plot_sol)
    plotargs: dict, optional
        keyword arguments to pass to solver.plotfunc.
        The default is {} (empty, no arguments).
    skipsteps: int, optional
        Defines how many time steps to skip in the animation of the solution.
        ex. if skipsteps=5, every 6th time step will be used as a frame
        The default is 0.
    fps: int, optional
        The frame rate of the video. Only used if make_video=True.
        The default is 24.
    last_frame: bool, optional
        Force the last frame to be a part of the animation, even if it doesn't
        land exactly on skipsteps.
        The default is False.
    tfinal : int, optional
        The final time to run the animation until. If None, uses default 
        solver.t_final. Must be less than solver.t_final.
        The default is None.
    '''
    # check that dimension of solution is correct
    assert(solver.q_sol.ndim == 3),"q_sol of inputted solver instance wrong shape"
    
    # check other inputs for correct format
    assert(isinstance(plotfunc, str)),"plotfunc must be a string"
    assert(isinstance(plotargs, dict)),"plotargs must be a dictionary"
    assert('savefile' not in plotargs),"savefile should not be set in plotargs"
    assert(isinstance(skipsteps, int)),"skipsteps must be an integer"
    assert(isinstance(fps, int)),"fps must be an integer"
    
    # set the plotting function
    plot = getattr(solver, plotfunc)
    
    qshape = np.shape(solver.q_sol)
    steps = qshape[2]
    if tfinal is None:
        tfinal = solver.t_final
        frames = int((steps-1)/(skipsteps+1)) + 1 # note int acts as a floor fn
    else:
        assert(tfinal <= solver.t_final),"tfinal must not exceed simulation time"
        frames = int(tfinal/((skipsteps+1)*solver.dt)) + 1 # note int acts as a floor fn
    numfill = len(str(frames)) # format appending number for saved files
        
    plotargs['show_fig']=False
    # get maximum and minimum y values along entire simulation
    # note: if you want axes to be set dynamically, input 'ymin'=None and 
    #       'ymax'=None in plotargs
    fig, ax = plt.subplots()
    if 'var2plot_name' in plotargs: 
        var2plot_name = plotargs['var2plot_name']
    else:
        var2plot_name = solver.diffeq.var2plot_name
    ax.plot([0,0],[np.min(solver.diffeq.var2plot(np.min(solver.q_sol,axis=2),var2plot_name)),np.max(solver.diffeq.var2plot(np.max(solver.q_sol,axis=2),var2plot_name))])
    ymin,ymax= ax.get_ylim()
    plt.close()
    if 'ymin' not in plotargs: plotargs['ymin']=ymin
    if 'ymax' not in plotargs: plotargs['ymax']=ymax
    
    # Make directory in which files will be saved
    if path.exists(file_name):
        print('WARNING: Folder name already exists. Using a temporary name instead.')
        file_name += '_RENAME'
    cmd = subprocess.run('mkdir '+file_name, shell=True, capture_output=True, text=True)
    assert(cmd.returncode==0),"Not able to make directory. Error raised: "+cmd.stderr
    
    print('...Making Frames') 
    suf = 'Complete.'  
    printProgressBar(0, frames, prefix = 'Progress:', suffix = suf) 
    for i in range(frames):
        stepi = i*(skipsteps+1)
        timei =  solver.dt*(solver.skip_ts+1)*stepi # or tfinal/steps*stepi
        # note, final time *may* be a little off due to how program was terminated
        # and when solution checkpoints were saved, but it will be the only one 
        # that is off. So fix the final time.
        if i == frames - 1:
            timei = min(timei, solver.t_final)
        printProgressBar(i+1, frames, prefix = 'Progress:', suffix = suf)
        
        # call plotting function from solver module
        plot(solver.q_sol[:,:,stepi], **plotargs, time=timei,
             savefile=file_name+'/'+'frame'+str(i).zfill(numfill))
    
    if last_frame:
        # if i == frames - 1, then this may double plot the final
        # frame, but I also don't really care if it does.
        plot(solver.q_sol[:,:,-1], **plotargs, time=tfinal,
             savefile=file_name+'/'+'frame'+str(frames))

    if (make_video or make_gif):
        # if images are saved as eps, convert to png with resolution given by -density (dpi)
        cmd = subprocess.run('test -f '+file_name+'/frame'+'0'.zfill(numfill)+'.eps', shell=True)
        if cmd.returncode==0: # files with extension .eps exist, must convert to png
            print('...Converting to png')
            cmd_str = 'for image in '+file_name+'/*.eps; do convert -density 300 "$image" "${image%.eps}.png"; done'
            cmd = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
            assert(cmd.returncode==0),"Not able to convert frames to png. Error raised: "+cmd.stderr
        
    if make_gif:
        print('...Making GIF')
        # delay inbetween frames (1/100s, or centiseconds) set by -delay. (4 is 25 fps)
        cmd_str = 'convert -delay 4 '+file_name+'/frame*.png '+file_name+'/animation.gif'
        # to set a longer delay on the last frame, can try:
        # 'convert -delay 4 '+file_name+'/frame%0'+numfill+'d.png -delay 100 '+file_name+'/frame'+str(frames-1)+'.png '+file_name+'/animation.gif'
        # see ImageMagick for more options
        cmd = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        assert(cmd.returncode==0),"Not able to create GIF. Error raised: "+cmd.stderr

    if make_video:
        print('...Making mp4')
        cmd_str = 'ffmpeg -framerate '+str(fps)+' -i '+file_name+'/frame%0'+str(numfill)+'d.png -c:v libx264 -pix_fmt yuv420p '+file_name+'/animation.mp4'
        # see more options here: https://trac.ffmpeg.org/wiki/Slideshow or https://symbols.hotell.kau.se/2017/09/12/converting-images/
        cmd = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        assert(cmd.returncode==0),"Not able to create mp4. Error raised: "+cmd.stderr
    
    print('All Done! Results are saved in '+file_name+' folder.') 

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
    
def plot_sparsity(mat_in,savefile=None,markersize=2,figsize=(6,6)):
    ''' prints the sparsity of the 2D matrix '''
    mat = np.copy(mat_in)
    plt.figure(figsize=figsize)
    #plt.spy(mat,precision=1e-14,markersize=markersize,color='black')
    mat[abs(mat)<1e-14] = 0
    mat[abs(mat)!=0] = 1
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.imshow(mat,origin='upper',cmap='binary')
    if savefile != None:
        plt.savefig(savefile,dpi=600)
    


def eigvec_history(solver, A=None, tfinal=None, save_file=None, window_size=1,
                   plot_difference=False, individual_plots=False, use_percent=True,
                   fit_data=True, plot_theory=False):
    '''
    Decompose a solution in an eigendecomposition of either a given matrix or
    the linearized RHS operator of the initial state. Takes a matrix A, finds
    it's eigendecomposition X, then decomposes every solution vector from q_sol
    into eigenbasis. Plots the evolution of the coefficients in time.

    Parameters
    ----------
    solver : class instance
        the solver class instance of solved ODE/PDE
    A : numpy 2d array, optional
        The matrix we use to find a decomposition of. Shape (nen*neq_node,nen*neq_node)
        If None, uses solver.check_eigs(q=solver.q_sol[:,:,0], plot_eigs=False, returnA=True)
        The default is None.
    tfinal : int, optional
        The final time to run the animation until. If None, uses default 
        solver.t_final. Must be less than solver.t_final.
        The default is None.
    save_file : str, optional
        If given, name of the file to save (all plots will add to this base).
        The name can include a specific path (ex. ../../folder/filename) and
        should not include the file extension.
        The default is None.
    window_size : int, optional
        Perform a moving average on the quantities using this given window size.
        If =1, no moving average is performed. The default is 1.
    plot_difference : bool, optional
        If False, plots the absolute value of the coefficients
        If True, plots the change in the coefficients
        The Default is False
    individual_plots : bool, optional
        If True, plots every eigenvector evolution separately. The default is False.
    use_percent : bool, optional
        If True, plots the percent change in the coefficients The default is False.

    '''
    if A is None:
        A = solver.check_eigs(q=solver.q_sol[:,:,0], plot_eigs=False, returnA=True)
    if tfinal is None:
        tfinal = solver.t_final
    else:
        assert(tfinal <= solver.t_final),"tfinal must not exceed simulation time"
    time = np.arange(0,tfinal,step=solver.dt*(solver.skip_ts+1))
    if (time[-1] != tfinal) and (tfinal == solver.t_final):
        time = np.append(time,tfinal)
    steps = len(time)

    assert(solver.q_sol.ndim == 3),'solver.q_sol of wrong dimension'
    assert(A.shape==(solver.nn,solver.nn)),'The solution array {0} does not match operator shape {1}'.format(solver.nn,A.shape)
    sols = solver.q_sol[:,:,:steps+1].reshape((solver.nn,steps),order='F')
    
    if use_percent: plot_difference=True
    if plot_theory: individual_plots=True
    
    V, X = np.linalg.eig(A)
    # column X[:,i] is the eigenvector corresponding to eigenvalue V[i]
    
    C = np.zeros((solver.nn,steps),dtype=complex) # store coefficients of eigenvector decomposition
    for i in range(steps):
        C[:,i] = X@sols[:,i] # C[j,i] is the coefficient of eigenvector X[:,j] at step i
        
    C_theory = np.zeros((solver.nn,steps),dtype=complex)
    for j in range(solver.nn):
        C_theory[j,:] = C[j,0]*np.exp(time*V[j])

    i = np.argmax(V.real)
    print('Eigenvalue with max real part is {0} with value {1:.3f}'.format(i,V[i]))
    
    C_fit = np.zeros((solver.nn,steps),dtype=complex)
    if fit_data:
        fit_vals = np.zeros(solver.nn,dtype=complex)
        for i in range(solver.nn):
            def fit_func(t,real,imag):
                return abs(C[i,0] * np.exp((real+1j*imag)*t))
            popt, pcov = sc.curve_fit(fit_func, time, abs(C[i,:]),p0=(0,np.imag(C[i,0])))
            C_fit[i,:] = fit_func(time, *popt)
            fit_vals[i] = popt[0] + 1j*popt[1]
    
    # apply moving average - if window_size=1 then this does nothing
    i = 0
    C_avg = []          # will be shape (steps-window_size+1,solver.nn)
    C_theory_avg = []   # will be shape (steps-window_size+1,solver.nn)
    C_fit_avg = []      # will be shape (steps-window_size+1,solver.nn)
    time_avg = []       # will be shape (steps-window_size+1)
    while i < steps - window_size + 1:
        C_window = C[:,i : i + window_size]
        C_theory_window = C_theory[:,i : i + window_size]
        C_fit_window = C_fit[:,i : i + window_size]
        C_avg.append(np.sum(C_window,axis=1) / window_size)
        C_theory_avg.append(np.sum(C_theory_window,axis=1) / window_size)
        C_fit_avg.append(np.sum(C_fit_window,axis=1) / window_size)
        time_avg.append((time[i]+time[i+window_size-1])/2)
        i += 1
    C_avg = np.array(C_avg).T
    C_theory_avg = np.array(C_theory_avg).T
    C_fit_avg = np.array(C_fit_avg).T
    time_avg = np.array(time_avg)
    
    
    # Plot the overall behaviour of all eigenvectors
    plt.figure()
    for i in range(solver.nn):
        if plot_difference:
            C0 = abs(C[i,0])
            if use_percent:
                plt.plot(time,100*(abs(C[i,:])-C0)/C0)
                plt.ylabel(r'\% Change in Coefficient',fontsize=14)
            else:
                plt.plot(time,abs(C[i,:])-C0)
                plt.ylabel(r'Change in Coefficient',fontsize=14)
        else:
            plt.plot(time,abs(C[i,:]))
            plt.ylabel(r'Coefficient',fontsize=14)
    plt.xlabel(r'Time',fontsize=14)
    plt.title(r'Eigenvector Decomposition of Solution',fontsize=16)
    if save_file is not None:
        plt.savefig(save_file+'_all.eps', format='eps')
    else:
        plt.show()
    plt.close()
    
    # Plot overall moving averages of all eigenvectors
    if window_size >1:
        plt.figure()
        for i in range(solver.nn):
            if plot_difference:
                C0 = abs(C_avg[i,0])
                if use_percent:
                    plt.plot(time_avg,100*(abs(C_avg[i,:])-C0)/C0)
                    plt.ylabel(r'\% Change in Coefficient',fontsize=14)
                else:
                    plt.plot(time_avg,abs(C_avg[i,:])-C0)
                    plt.ylabel(r'Change in Coefficient',fontsize=14)
            else:
                plt.plot(time_avg,abs(C_avg[i,:]))
                plt.ylabel(r'Coefficient',fontsize=14)
        plt.xlabel(r'Average Time (window size {0:.2f}s)'.format(solver.dt*window_size),fontsize=14)
        plt.title(r'Moving Averages of Eigenvector Decomposition',fontsize=16)
        if save_file is not None:
            plt.savefig(save_file+'_all_avg.eps', format='eps')
        else:
            plt.show()
        plt.close()
            
    # Plot fits
    if fit_data:
        plt.figure()
        for i in range(solver.nn):
            if plot_difference:
                C0 = abs(C_fit_avg[i,0])
                if use_percent:
                    plt.plot(time_avg,100*(abs(C_fit_avg[i,:])-C0)/C0)
                    plt.ylabel(r'\% Change in Coefficient',fontsize=14)
                else:
                    plt.plot(time_avg,abs(C_fit_avg[i,:])-C0)
                    plt.ylabel(r'Change in Coefficient',fontsize=14)
            else:
                plt.plot(time_avg,abs(C_fit_avg[i,:]))
                plt.ylabel(r'Fit to Coefficient',fontsize=14)
        if window_size >1:
            plt.xlabel(r'Average Time (window size {0:.2f}s)'.format(solver.dt*window_size),fontsize=14)
        else:
            plt.xlabel(r'Time',fontsize=14)
        plt.title(r'Coefficient Fit Eigenvector Decomposition',fontsize=16)
        if save_file is not None:
            plt.savefig(save_file+'_all_fits.eps', format='eps')
        else:
            plt.show()
        plt.close()
    
    # same plots as above but now for each individual eigenvector
    if individual_plots:
        for i in range(solver.nn):
            plt.figure()
            plt.title('Eigenvector {0}, Eigval {1:.2f}'.format(i,V[i]),fontsize=16)
            if plot_difference:
                C0 = abs(C_avg[i,0])
                C0_theory = abs(C_theory_avg[i,0])
                C0_fit = abs(C_fit_avg[i,0])
                if use_percent:
                    plt.plot(time_avg,100*(abs(C_avg[i,:])-C0)/C0,label='Numerical')
                    if plot_theory:
                        plt.plot(time_avg,100*(abs(C_theory_avg[i,:])-C0_theory)/C0_theory,label='Theory')
                    if fit_data:
                        plt.plot(time_avg,100*(abs(C_fit_avg[i,:])-C0_fit)/C0_fit,label=r'Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                    plt.ylabel(r'\% Change in Coefficient',fontsize=14)
                else:
                    plt.plot(time_avg,abs(C_avg[i,:])-C0,label='Numerical')
                    if plot_theory:
                        plt.plot(time_avg,abs(C_theory_avg[i,:])-C0_theory,label='Theory')
                    if plot_theory:
                        plt.plot(time_avg,abs(C_fit_avg[i,:])-C0_fit,label=r'Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                    plt.ylabel(r'Change in Coefficient',fontsize=14)
            else:
                plt.plot(time_avg,abs(C_avg[i,:]),label='Numerical')
                if plot_theory:
                    plt.plot(time_avg,abs(C_theory_avg[i,:]),label='Theory')
                if plot_theory:
                    plt.plot(time_avg,abs(C_fit_avg[i,:]),label=r'Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                plt.ylabel(r'Coefficient',fontsize=14)
            if plot_theory or fit_data:
                plt.legend(loc='upper left',fontsize=12)
            if window_size >1:
                plt.xlabel(r'Average Time (window size {0:.2f}s)'.format(solver.dt*window_size),fontsize=14)
                if save_file is not None:
                    plt.savefig(save_file+'_eig{0}_avg.eps'.format(i), format='eps')
                else:
                    plt.show()
            else: 
                plt.xlabel(r'Time',fontsize=14)
                if save_file is not None:
                    plt.savefig(save_file+'_eig{0}.eps'.format(i), format='eps')
                else:
                    plt.show()
            plt.close()


def symbolic(A):
    ''' Use sympy to make a matrix or vector symbolic '''
    shape = A.shape
    A1 = np.copy(A)
    A = np.zeros(shape,dtype=object)
    
    if len(shape) ==1:
        for i in range(shape[0]):
                A[i] = nsimplify(A1[i],tolerance=1e-10,rational=True)
    
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                A[i,j] = nsimplify(A1[i,j],tolerance=1e-10,rational=True)
                
    elif len(shape) ==3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    A[i,j,k] = nsimplify(A1[i,j,k],tolerance=1e-10,rational=True)
    
    return np.array(A)


def run_convergence(solver, schedule_in=None, error_type='SBP',
             scale_dt=True, return_conv=False, savefile=None, labels=None,
             title=None, ylabel=None, xlabel=None, grid=False, convunc=True, 
             ylim=None, xlim=(None,None), ignore_fail=False, plot=True, vars2plot=None,
             nthreads=1, extra_marker=None, skipfit=None, skip=None, title_size=16,
             legendloc=None, figsize=(6,4), tick_size=12, extra_xticks=False, scalar_xlabel=False, 
             serif=False, colors=None, markers=None, linestyles=None, legendsize=12, legendreorder=None):
    '''
    Purpose
    ----------
    Runs a convergence analysis for a given problem. The convergence is
    always done in terms of DOF (solver.nen or solver.nelem).

    Parameters
    ----------
    solver:
    schedule_in: ex. [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    scale_dt : boolean (optional)
        Determines whether to scale the time step (approximately) according
        to the relative mesh spacing as the mesh refines.
        The default is True.
    error_type : string (optional)
        Determines which error to use. Options are:
            'H-norm' : the SBP error sqrt((q-q_ex).T @ H @ (q-q_ex))
            'Rms' : the standard root mean square error in L2 norm
            'Boundary' : The simple periodic boundary error | q_1 - q_N |
            'Truncation-SBP' : the SBP error but instead using er = dqdt(q-q_ex)
            'Truncation-Rms' : the Rms error but instead using er = dqdt(q-q_ex)
            'max diff' : np.max(abs(q-q_ex))
        The default is Rms.
    return_conv : bool (optional)
        Flag whether to return dofs, errors, and legend_strings.
        The default is False.
    savefile: string (optional)
        file name under which to save the plot. The default is None.
    labels: list of strings (optional)
        labels to use in the legends for the different runs
    '''
    print('---------------------------------------------------------')
    
    if schedule_in==None:
        schedule = [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    else:
        schedule=schedule_in.copy()

    if scale_dt:
        base_dt = solver.dt
        if base_dt is None:
            raise Exception('input solver must have a set dt value, not None')
        if solver.dim == 1:
            xmin, xmax, nn = solver.xmin, solver.xmax, solver.nn
        else:
            # only estimate new dt based on scaling of x coordinate
            xmin, xmax, nn = solver.xmin[0], solver.xmax[0], solver.nn[0]
        base_dx = (xmax - xmin) / nn

    if nthreads > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

    ''' Unpack Schedule '''
    # Check that we either use 'nen' or 'nelem'
    # to refine. We can't use both.
    runs_nen = []
    if any(i[0]=='nen' for i in schedule) and any(i[0]=='nelem' for i in schedule):
        nens = next((x for x in schedule if x[0] == 'nen'), None)
        nelems = next((x for x in schedule if x[0] == 'nelem'), None)
        if len(nens) == 2:
            # was specifying a specific nens, but wanted to do element refinement
            solver.nen = nens[1]
            schedule = [x for x in schedule if not ('nen' in x)]
        elif len(nelems) == 2:
            # was specifying a specific nelems, but wanted to do node refinement
            solver.nelem = nelems[1]
            schedule = [x for x in schedule if not ('nelem' in x)]
        else:
            print('WARNING: Can not do a refinement specifying both nen and nelem.')
            print('     ... Using only nelem values, and atempting to taking nen for each run or for each p.')
            runs_nen = [x[1:] for x in schedule if x[0] == 'nen'][0]
            # remove 'nelem' and 'nen' lists
            schedule = [x for x in schedule if not ('nen' in x)]

    # Otherwise, we now have combinations of attributes to run, and we
    # calculate convergence according to either nn or nelem refinement.
    # Note that nen can be used both with nn or nelem refinement.
    match_nen_to_nelem = False
    match_nen_to_p = False
    if any(i[0]=='nen' for i in schedule) and not any(i[0]=='nelem' for i in schedule):
        print('Performing classical refinement.')
        runs_nen = [x[1:] for x in schedule if x[0] == 'nen'][0]
        schedule.remove([x for x in schedule if x[0] == 'nen'][0])
        runs_nelem = [solver.nelem] * len(runs_nen) # reset these as well
        match_nen_to_nelem = True
    elif any(i[0]=='nelem' for i in schedule):
        print('Performing element refinement.')
        runs_nelem = [x[1:] for x in schedule if x[0] == 'nelem'][0]
        schedule.remove([x for x in schedule if x[0] == 'nelem'][0])
        runs_p = [x[1:] for x in schedule if x[0] == 'p'][0]
        if len(runs_nen)==len(runs_nelem):
            print('... Match nen to nelem for each run')
            match_nen_to_nelem = True
        elif len(runs_nen)==1:
            print('... Set the same nen each run')
            runs_nen = runs_nen * len(runs_nelem)
            match_nen_to_nelem = True
        elif len(runs_nen) == len(runs_p):
            print('.. Match nen to p for each run')
            match_nen_to_p = True
        else:
            print('.. Match nen to nelem for each run using default nen')
            runs_nen = [solver.nen] * len(runs_nelem) # reset these as well
            match_nen_to_nelem = True
    else:
        raise Exception('Convergence schedule must contain either nen or nelem refinement.')
    
    # unpack remaining attributes in schedule, store combinations in cases
    attributes = []
    values = []
    for item in schedule:
        attributes.append(item[0])
        values.append(item[1:])
    cases = list(itertools.product(*values)) # all possible combinations

    ''' Run cases for various runs, store errors '''
    n_cases = len(cases)
    n_runs = len(runs_nelem)
    n_tot = n_cases*n_runs
    n_attributes = len(attributes)
    if vars2plot is None:
        errors = np.zeros((n_cases,n_runs)) # store errors here
    else:
        errors = np.zeros((n_cases,n_runs,len(vars2plot)))
    dofs = np.zeros((n_cases,n_runs)) # store degrees of freedom here
    legend_strings = ['']*n_cases # store labels for cases here
    if labels is not None:
        assert(len(labels)==n_cases),'labels must be a list of length = n_cases'

    if scale_dt: variables_base = [None]*(3+n_attributes) # initiate list to pass to reset()
    else: variables_base = [None]*(2+n_attributes)
    tm_rtol = solver.tol
    tm_atol = solver.atol

    def set_variables(casei,runi,variables_base):
        variables = variables_base.copy() # for each run (refinement)
        variables[0] = ('nelem',runs_nelem[runi])
        if match_nen_to_nelem:
            run_neni = runi
        elif match_nen_to_p:
            p = cases[casei][attributes.index('p')]
            run_neni = runs_p.index(p)
        else:
            raise Exception('Something went wrong')
        variables[1] = ('nen',runs_nen[run_neni])
        if scale_dt:
            variables[-1] = ('dt',None)
            
        # add a few default things to save time
        variables.append(('print_sol_norm', False))
        variables.append(('bool_plot_sol', False))
        variables.append(('cons_obj_name', None))

        # TODO : add flag to also calculate conservation objectives?
        return variables

    n_toti = 1
    if nthreads == 1: # run in serial
        # set a couple useful time-saving settings
        solver.keep_all_ts = False
        #solver.print_progress = False
        for casei in range(n_cases): # for each case
            for atti in range(n_attributes):
                variables_base[atti+2] = (attributes[atti],cases[casei][atti]) # assign attributes
                legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
            legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting
            
            try:
                for runi in range(n_runs):
                    variables = set_variables(casei,runi,variables_base)

                    ''' solve run for case, store results '''
                    print('Running for:', variables)
                    if labels is not None:
                        print('label:', labels[casei])

                    dofs[casei,runi], errors[casei,runi], nn = run_single_case(solver, variables, scale_dt, 
                                                                    base_dt, base_dx, tm_rtol, tm_atol, 
                                                                    error_type, vars2plot)
                    
                    print('Convergence Progress: run {0} of {1} complete.'.format(n_toti,n_tot))
                    print('Final Error: ', solver.calc_error())
                    print('Total number of nodes: ', nn)
                    print('---------------------------------------------------------')
                    n_toti += 1
            except Exception as e:
                if ignore_fail:
                    errors[casei,:] = None
                    dofs[casei,:] = None
                    print('---------------------------------------------------------')
                    print('---------------------------------------------------------')
                    print('---------------------------------------------------------')
                    print('WARNING: run {0} of {1} encountered errors.'.format(n_toti,n_tot))
                    print('         Ignoring this case (all runs) and moving on.')
                    print('---------------------------------------------------------')
                    print('WARNING: run {0} of {1} encountered errors.'.format(n_toti,n_tot))
                    print('         Ignoring this case (all runs) and moving on.')
                    print('---------------------------------------------------------')
                    print('WARNING: run {0} of {1} encountered errors.'.format(n_toti,n_tot))
                    print('         Ignoring this case (all runs) and moving on.')
                    print('---------------------------------------------------------')
                    print('---------------------------------------------------------')
                    print('---------------------------------------------------------')
                    n_toti += n_runs - runi
                else:
                    traceback.print_exc()  # Print the traceback
                    raise  # Re-raise the original exception
    else:
        # Run in parallel mode with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = {}
            for casei in range(n_cases):
                for atti in range(n_attributes):
                    variables_base[atti+2] = (attributes[atti],cases[casei][atti]) # assign attributes
                    legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
                legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting

                for runi in range(n_runs):
                    # Prepare a copy of solver and set up variables for this run
                    variables = set_variables(casei,runi,variables_base)  # set up run-specific variables
                    solver_kwargs, diffeq_args = prep_new_solver_instance(solver, variables)
                    diffeq_class = type(solver.diffeq)
                    solver_class = type(solver)
                    
                    # Submit each run as a separate task and store with its indices
                    future = executor.submit(
                        run_parallel_case,
                        solver_class, diffeq_class, solver_kwargs, diffeq_args,
                        scale_dt, base_dt, base_dx, tm_rtol, tm_atol, error_type, vars2plot
                    )
                    futures[future] = (casei, runi)
            
            # Gather results with preserved order
            for future in as_completed(futures):
                casei, runi = futures[future]  # Retrieve original indices
                try:
                    result_dofs, result_errors, nn = future.result()
                    dofs[casei, runi] = result_dofs
                    errors[casei, runi] = result_errors

                except Exception as e:
                    print(f"Error in parallel execution for case {casei}, run {runi}: {e}")
                    import traceback
                    traceback.print_exc()
                    dofs[casei, runi] = np.nan
                    errors[casei, runi] = np.nan

                # Progress update immediately after each task completes
                print('Convergence Progress: run {0} of {1} complete.'.format(n_toti,n_tot))
                print('Final Errors for casei={}, runi={}:'.format(casei,runi), errors[casei,runi])
                print('Total number of nodes: ', nn)
                print('---------------------------------------------------------')
                n_toti += 1
                        
    if labels is not None:
        # overwrite legend_strings with labels
        legend_strings = labels

    if return_conv:
        # keep the original arrays so that we can return arrays including NaNs
        dofs_ret, errors_ret, legend_strings_ret = np.copy(dofs), np.copy(errors), np.copy(legend_strings)
        
    if np.any(np.isnan(errors)) or np.any(np.equal(errors, None)):
        errors_to_keep = []
        dofs_to_keep = []
        labels_to_keep = []
        for i, row in enumerate(errors):
            if not (np.any(np.isnan(row)) or np.any(np.equal(row, None))):
                errors_to_keep.append(row)
                dofs_to_keep.append(dofs[i])
                labels_to_keep.append(legend_strings[i])
        errors, dofs, legend_strings = np.array(errors_to_keep), np.array(dofs_to_keep), labels_to_keep
    
    ''' Analyze convergence '''
    print('---------------------------------------------------------')
    if vars2plot is None:
        print(error_type + ' Error Convergence Rates:')
        conv_vec, avg_conv = calc_conv_rate(dofs, errors, solver.dim, legend_strings=legend_strings)
    else:
        for varidx, var in enumerate(vars2plot):
            print(error_type + f' Error Convergence Rates for {var}:')
            conv_vec, avg_conv = calc_conv_rate(dofs, errors[:,:,varidx], solver.dim, legend_strings=legend_strings)

    ''' Plot Results '''
    if plot:
        if vars2plot is None:
            if title == None:
                title = r"Convergence of " + error_type + " Error"
            plot_conv(dofs, errors, legend_strings, solver.dim, title=title, savefile=savefile,
                        ylabel=ylabel,xlabel=xlabel,grid=grid,convunc=convunc,ylim=ylim,xlim=xlim,
                    extra_marker=extra_marker, skipfit=skipfit, skip=skip, title_size=title_size,
                    legendloc=legendloc, figsize=figsize, tick_size=tick_size, 
                    extra_xticks=extra_xticks, scalar_xlabel=scalar_xlabel, serif=serif, colors=colors, 
                    markers=markers, linestyles=linestyles, legendsize=legendsize, legendreorder=legendreorder)
        else:
            for varidx, var in enumerate(vars2plot):
                if title == None:
                    title = r"Convergence of " + error_type + ' ' + var + " Error"
                plot_conv(dofs, errors[:,:,varidx], legend_strings, solver.dim, title, savefile,
                            ylabel=ylabel,xlabel=xlabel,grid=grid,convunc=convunc,ylim=ylim,xlim=xlim,
                            extra_marker=extra_marker, skipfit=skipfit, skip=skip, title_size=title_size,
                            legendloc=legendloc, figsize=figsize, tick_size=tick_size, 
                            extra_xticks=extra_xticks, scalar_xlabel=scalar_xlabel, serif=serif, colors=colors, 
                            markers=markers, linestyles=linestyles, legendsize=legendsize, legendreorder=legendreorder)
    
    if return_conv:
        return dofs_ret, errors_ret, legend_strings_ret
    
def run_single_case(solver, variables, scale_dt, base_dt, base_dx, 
                    tm_rtol, tm_atol,
                    error_type, vars2plot, reset=True):
    ''' runs a single case for convergence
    NOTE: solver should be a new object, not the original object, if running in parallel '''
    # Reset solver if needed (i.e. unless solver is a completely new object)
    if reset: solver.reset(variables)
    
    # Scale time step if needed
    if scale_dt:
        if solver.dim == 1:
            xmin, xmax, nn = solver.xmin, solver.xmax, solver.nn
        else:
            xmin, xmax, nn = solver.xmin[0], solver.xmax[0], solver.nn[0]
        new_dx = (xmax - xmin) / nn
        new_dt = base_dt * new_dx / base_dx
        solver.set_timestep(new_dt)
        print('set timestep to dt =', new_dt)
    
    # Run the solver
    solver.tm_atol = tm_atol
    solver.tm_rtol = tm_rtol
    solver.solve()

    # Calculate errors
    if vars2plot is None:
        errors = solver.calc_error(method=error_type)
    else:
        errors = np.zeros(len(vars2plot))
        for varidx, var in enumerate(vars2plot):
            try:
                errors[varidx] = solver.calc_error(method=error_type, var2plot_name=var)
            except Exception as e:
                print(f"ERROR: solver.calc_error(var2plot_name='{var}') returned errors. Skipping.")
                errors[varidx] = 0.
    
    # Gather DOFs
    if solver.dim == 1:
        nn = solver.nn
        dofs = nn
    elif solver.dim == 2:
        nn = solver.nn[0] * solver.nn[1]
        dofs = np.sqrt(nn)
    elif solver.dim == 3:
        nn = solver.nn[0] * solver.nn[1] * solver.nn[2]
        dofs = np.cbrt(nn)
    
    return dofs, errors, nn

def prep_new_solver_instance(base_solver, variables):
    ''' prepare arguments for a new solver and diffeq instance '''
    solver_copy = copy.copy(base_solver)
    for i in range(len(variables)):
        attribute , value = variables[i]
        if hasattr(solver_copy, attribute):
            setattr(solver_copy, attribute, value)
        else:
            print("ERROR: solver has no attribute '{0}'. Ignoring.".format(attribute))
    
    if solver_copy.diffeq.diffeq_name == 'Euler2d':
        diffeq_args = [[solver_copy.diffeq.R,solver_copy.diffeq.g],
                solver_copy.diffeq.q0_type,
                solver_copy.diffeq.test_case,
                solver_copy.diffeq.bc]
    else:
        raise Exception('TODO: must manually code up this Diffeq.')
    
    solver_kwargs = {'settings':solver_copy.settings, 
                'tm_method':solver_copy.tm_method, 'dt':solver_copy.dt, 't_final':solver_copy.t_final, 
                'q0':solver_copy.q0, 
                'p':solver_copy.p, 'disc_type':solver_copy.disc_type,
                'surf_diss':solver_copy.surf_diss, 'vol_diss':solver_copy.vol_diss, 'had_flux':solver_copy.had_flux,
                'nelem':solver_copy.nelem, 'nen':solver_copy.nen, 'disc_nodes':solver_copy.disc_nodes,
                'bc':solver_copy.bc, 'xmin':solver_copy.xmin, 'xmax':solver_copy.xmax,
                'cons_obj_name':solver_copy.cons_obj_name,
                'bool_plot_sol':solver_copy.bool_plot_sol, 'print_sol_norm':solver_copy.print_sol_norm,
                'print_residual':solver_copy.print_residual, 'check_resid_conv':solver_copy.check_resid_conv}
    return solver_kwargs, diffeq_args

def run_parallel_case(solver_class, diffeq_class, solver_kwargs, diffeq_args,
                       scale_dt, base_dt, base_dx, tm_rtol, tm_atol, 
                       error_type, vars2plot):
    ''' Create a new solver & diffeq instance with the given variables '''
    diffeq = diffeq_class(*diffeq_args)
    solver = solver_class(diffeq, **solver_kwargs)
    
    # set a couple useful time-saving settings
    solver.print_progress = False
    solver.keep_all_ts = False
    dofs, errors, nn = run_single_case(solver, None, 
                                       scale_dt, base_dt, base_dx,
                                       tm_rtol, tm_atol, 
                                       error_type, vars2plot, reset=False)
    return dofs, errors, nn
    
    
    
def calc_conv_rate(dof_vec, err_vec, dim, n_points=None,
                   print_conv=True, legend_strings=None):
    '''
    Parameters
    ----------
    dof_vec : numpy array
        Indicates the no. of DOF at each iteration
    err_vec : numpy array
        Indicates the error at each iteration.
    n_points : int, optional
        The first no. of points to use to calculate the average convergence.
        The default is to use all of the data points.
    print_conv : bool, optional
        Determines whether or not to print a table with convergence rates.
        The default is True.
    legend_strings : list of strings, optional
        Labels to use when printing table. Defaults is None.

    Returns
    -------
    conv_vec : numpy array
        The convergence rate between all two data points.
    avg_conv : float
        The least squares slope (convergence rate) using the last n_points data
        points.
    '''
    assert dof_vec.shape==err_vec.shape,"The two inputted arrays are not the same shape!"
    if dof_vec.ndim>1:
        n_cases, n_runs = dof_vec.shape
    else:
        n_cases, n_runs = 1, dof_vec.size
        dof_vec = np.reshape(dof_vec,(n_cases, n_runs))
        err_vec = np.reshape(err_vec,(n_cases, n_runs))
    assert n_runs>1,"ERROR: Not enough grids to perform convergence."

    conv_vec = np.zeros((n_cases, n_runs))
    avg_conv = np.zeros(n_cases)
    for casei in range(n_cases):
        logx = np.log(dof_vec[casei])
        logy = np.log(err_vec[casei])

        # Calculate the convergence between every two sets of points
        conv_vec[casei,1:] = -(logy[1:] - logy[:-1]) / (logx[1:] - logx[:-1])

        # Calculate the least squares solution
        if (n_points == None) or n_points > n_runs:
            n_points = n_runs

        logx_plus = np.vstack([logx[:n_points], np.ones(n_points)]).T
        avg_conv[casei], _ = -np.linalg.lstsq(logx_plus, logy[:n_points], rcond=None)[0]

        if print_conv:
            if legend_strings is not None:
                print('Case: ' + legend_strings[casei])
            print('Average Convergence: {:.3f}'.format(avg_conv[casei]))
            data = np.array([dof_vec[casei],err_vec[casei],conv_vec[casei]]).T
            if dim == 1: Ndof = 'Ndof'
            elif dim == 2: Ndof = u'\u221A' + 'Ndof'
            elif dim == 3: Ndof = u'\u221B' + 'Ndof'
            print(tabulate((data), headers=[Ndof, 'Error','Convergence'], tablefmt='orgtbl'))

    return conv_vec, avg_conv

def plot_conv(dof_vec, err_vec, legend_strings, dim, title=None, savefile=None,
              extra_marker=None, skipfit=None, skip=None, ylabel=None, xlabel=None, title_size=16,
              ylim=(None,None),xlim=(None,None),grid=False,legendloc=None,convunc=True,
              figsize=(6,4), tick_size=12, extra_xticks=False, scalar_xlabel=False, serif=False,
              colors=None, markers=None, linestyles=None, legendsize=12, legendreorder=None,
              remove_outliers=False, legend_anchor=None, put_legend_behind=False):
    '''
    Parameters
    ----------
    dof_vec : numpy array
        Indicates the no. of DOF at each iteration
    err_vec : numpy array
        Indicates the error at each iteration.
    legend_strings : list of strings
        Labels to use when printing table.
    title : string, optional
        Title of plot. The default is None.
    savefile : string, optional
        file name under which to save the plot. The default is None.
    extra_marker : list of lists, optional
        if True, mark the point with an additional marker X
    skipfit : list of ints, optional
        for each case, decide how many initial runs to skip in fit, AFTER skip
        Note: does not account for the entries removed if nan
    skip : list of ints, optional
        for each case, decide how many initial runs to skip from plotting.
        Note: does account for the entries removed

    Returns
    -------
    None
    '''

    if serif:
        plt.rcParams['font.family'] = 'serif'
    
    assert dof_vec.shape==err_vec.shape,"The two inputted arrays are not the same shape!"
    if dof_vec.ndim>1:
        n_cases, n_runs = dof_vec.shape
    else:
        n_cases, n_runs = 1, dof_vec.size
        dof_vec = np.reshape(dof_vec,(n_cases, n_runs))
        err_vec = np.reshape(err_vec,(n_cases, n_runs))
    assert n_runs>1,"ERROR: Not enough grids to perform convergence."
    
    assert len(legend_strings)==n_cases,"ERROR: legend_strings do not match n_cases"
    
    if skipfit == None:
        skipfit = [0] * n_cases
    else:
        assert(len(skipfit)==n_cases),"ERROR: skipfit does not match n_cases"
        
    if skip == None:
        skip = [0] * n_cases
    else:
        assert(len(skip)==n_cases),"ERROR: skipfit does not match n_cases"
    
    if extra_marker != None:
        assert np.shape(extra_marker)==(n_cases,n_runs),"ERROR: extra_marker shape should match dof_vec and err_vec"

    if put_legend_behind:
        zorder = 3
    else:
        zorder = 2

    fig = plt.figure(figsize=figsize)
    if title != None:
        plt.title(title,fontsize=18)
    if ylabel == None:
        ylabel = r"Error"
    if xlabel == None:
        if dim == 1: xlabel = r"Degrees of Freedom"
        elif dim == 2: xlabel = r"$\sqrt{} $ Degrees of Freedom"
        elif dim == 3: xlabel = r"$\sqrt[3]{}$ Degrees of Freedom"      
    plt.ylabel(ylabel,fontsize=title_size)
    plt.xlabel(xlabel,fontsize=title_size)
    
    # Do a curve fit to find the slope on log-log plot (order of convergence)
    def fit_func(x, a, b): 
        return -a*x + b
    
    if colors == None:
        colors=['blue','red','green','magenta','orange','purple','brown']
    else: 
        assert(isinstance(colors, list)), "colors must be a list"
    if markers == None:
        #markers=['o','^','s','D','>','<','8']
        markers = ['o', '^', 's', 'd','x', '+']
    else: 
        assert(isinstance(markers, list)), "markers must be a list"
    if linestyles == None:
        linestyles = ['--']
    else:
        assert(isinstance(linestyles, list)), "linestyles must be a list"
    
    for i in range(n_cases):
        dof_mod = np.copy(dof_vec[i][skip[i]:])
        err_mod = np.copy(err_vec[i][skip[i]:])
        ######### comment this out if you want to print things <1e-16 ########
        k = 0
        for j in range(len(dof_vec[i][skip[i]:])):
            if (err_mod[j-k] < 1e-16) or np.isnan(err_mod[j-k]) or np.isinf(err_mod[j-k]):
                dof_mod = np.delete(dof_mod,j-k)
                err_mod = np.delete(err_mod,j-k)
                k += 1
        ######################################################################
        string = legend_strings[i].replace("disc_nodes=","")
        if len(dof_mod) > 2:
            x_data = np.log(dof_mod[skipfit[i]:])
            y_data = np.log(err_mod[skipfit[i]:])
            if remove_outliers:
                from sklearn.linear_model import RANSACRegressor
                # Reshape x for sklearn
                x_data_reshaped = x_data.reshape(-1, 1)
                # Fit using RANSAC to remove outliers
                ransac = RANSACRegressor()
                ransac.fit(x_data_reshaped, y_data)
                inlier_mask = ransac.inlier_mask_
                x_data = x_data[inlier_mask]
                y_data = y_data[inlier_mask]
            p_opt, p_cov = sc.curve_fit(fit_func, x_data, y_data,(2,1)) # fit
            if convunc:
                acc = int(np.floor(np.log10(np.sqrt(p_cov[0,0]))))
                unc = np.round(np.sqrt(p_cov[0,0]),abs(acc))
                acc = int(np.floor(np.log10(unc)))
                if acc >=0:
                    slope = r" $({0} \pm {1})$".format(int(p_opt[0]),int(unc))
                elif acc==-1:
                    slope = r" $({0:9.1f} \pm {1:6.1f})$".format(p_opt[0],unc)
                elif acc==-2:
                    slope = r" $({0:9.2f} \pm {1:6.2f})$".format(p_opt[0],unc)
                elif acc==-3:
                    slope = r" $({0:9.3f} \pm {1:6.3f})$".format(p_opt[0],unc)
                else:
                    slope = r" $({0:9.4f} \pm {1:6.1g})$".format(p_opt[0],unc)
            else:
                slope = r" $({0:9.2f})$".format(p_opt[0])
            plt.loglog(dof_mod,err_mod,markers[i%len(markers)],markersize=8, color=colors[i%len(colors)],
                       markerfacecolor = 'none', markeredgewidth=2, label=string+slope, zorder=zorder)
            plt.loglog(np.linspace(dof_mod[skipfit[i]],dof_mod[-1]), # plot
                       np.exp(fit_func(np.log(np.linspace(dof_mod[skipfit[i]],dof_mod[-1])), *p_opt)), 
                       linewidth=1, linestyle = linestyles[i%len(linestyles)], color=colors[i%len(colors)], zorder=zorder)
        elif len(dof_mod) == 2:
            slope = r" $({0:9.3})$".format(-(np.log(err_mod[1])-np.log(err_mod[0]))/(np.log(dof_mod[1])-np.log(dof_mod[0])))
            plt.loglog(dof_mod,err_mod,markers[i%len(markers)],markersize=8, color=colors[i%len(colors)],
                       markerfacecolor = 'none', markeredgewidth=2, label=string+slope, zorder=zorder)
            plt.loglog(dof_mod, err_mod, linewidth=1, linestyle = linestyles[i%len(linestyles)], color=colors[i%len(colors)], zorder=zorder)
        if extra_marker != None:
            for j in range(n_runs):
                if extra_marker[i][j] == True:
                    plt.plot(dof_vec[i][j],err_vec[i][j],'x',color='black',markersize=12,linewidth=2, zorder=zorder)
    if legendloc == None:
        legendloc = 'best'
    bbox_transform = plt.gca().transAxes
    if legendreorder != None:
        legend = plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        if isinstance(legendreorder, list):
            if len(legendreorder) == n_cases:
                handles = [handles[i] for i in legendreorder]
                labels = [labels[i] for i in legendreorder]
            else:
                print('WARNING: legendreorder must be a list of length n_cases. Ignoring.', len(legendreorder))
        else:
            print('WARNING: legendreorder must be a list. Ignoring.')
        legend = plt.legend(handles, labels, loc=legendloc, fontsize=legendsize,
                   bbox_to_anchor=legend_anchor, bbox_transform=bbox_transform)
    else:
        legend = plt.legend(loc=legendloc, fontsize=legendsize,
                   bbox_to_anchor=legend_anchor, bbox_transform=bbox_transform)
    if put_legend_behind: legend.set_zorder(2)
    if grid:
        plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
    plt.ylim(ylim)
    plt.xlim(xlim)
    #plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    ax = plt.gca()  # Get current axis
    # Conditionally set scalar labels for major ticks if scalar_xlabel is True
    if scalar_xlabel:
        ax.xaxis.set_major_formatter(tik.ScalarFormatter())  # Major ticks in scalar format

    # Conditionally add extra minor ticks with labels if extra_xticks is True
    if extra_xticks:
        tick_labels = ax.get_xticklabels()  # Get major tick labels
        label_ypos = ax.get_ylim()[0]
        xmin, xmax = ax.get_xlim()

        for label in tick_labels:
            if label.get_text():  # Only consider visible labels
                bbox = label.get_window_extent(renderer=fig.canvas.get_renderer())
                # Convert bbox to data coordinates to get y-position of the label
                inverse = ax.transData.inverted()
                label_position = inverse.transform((bbox.x0, bbox.y0))[1]
                if label_position < label_ypos:
                    label_ypos = label_position

        # Add manual labels for 5×10^n
        if scalar_xlabel:
            extra_labels = {5 : r'$5$',
                            50 : r'$50$',
                            500 : r'$500$',
                            5000 : r'$5000$'}
        else:
            extra_labels = {5 : r'$5\times10^0$',
                            20 : r'$2\times10^1$',
                            50 : r'$5\times10^1$',
                            200 : r'$2\times10^2$',
                            500 : r'$5\times10^2$',
                            5000 : r'$5\times10^3$',
                            50000 : r'$5\times10^4$', 
                            500000 : r'$5\times10^5$' }
        for label in extra_labels:
            if label < xmax and label > xmin:
                 ax.text(label, label_ypos, extra_labels[label], va='bottom', ha='center')

    ax.tick_params(axis='both', labelsize=tick_size) 
    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.2)
    if savefile is not None:
        plt.savefig(savefile,dpi=600)
        
        
def run_jacobian_convergence(solver, schedule_in=None,
             return_conv=False, savefile=None, labels=None, 
             vol_metrics=False, surf_metrics=False, jacs=False, jac_ratios=False, backout_jacs=False):
    '''
    Purpose
    ----------
    Runs a convergence analysis of either:
        jac_ratio=True:  the value 1 - (J/J'), where J is the metric 
                         jacobian used for the time marching, and J' is 
                         the value backed out from the metric terms.
        jac=True:   The value (abs(J-J_ex)) where J_ex is the exact value
        backout_jacs=True: The value (abs(J-J_ex)) where J is the backed out jacobian
        vol_metrics:     The value (abs(M-M_ex)) where M_ex is the exact value
        surf_metrics:    The value (abs(M-M_ex)) where M_ex is the exact value
                         
    Parameters
    ----------
    solver:
    schedule_in: ex. [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    return_conv : bool (optional)
        Flag whether to return dofs, jacobian ratio, and legend_strings.
        The default is False.
    savefile: string (optional)
        file name under which to save the plot. The default is None.
    labels: list of strings (optional)
        labels to use in the legends for the different runs
    '''
    assert(vol_metrics or surf_metrics or jacs or jac_ratios or backout_jacs),'Must have at least one of (vol_metrics, surf_metrics, jacs, jac_ratios) = True'
    print('---------------------------------------------------------')
    
    if schedule_in==None:
        schedule = [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    else:
        schedule=schedule_in.copy()

    ''' Unpack Schedule '''
    # Check that we either use 'nen' or 'nelem'
    # to refine. We can't use both.
    if any(i[0]=='nen' for i in schedule) and any(i[0]=='nelem' for i in schedule):
        print('WARNING: Can not do a refinement specifying both nen and nelem. Using only nelem values.')
        # remove 'nelem' and 'nen' lists
        schedule = [x for x in schedule if not ('nen' in x)]

    # Otherwise, we now have combinations of attributes to run, and we
    # calculate convergence according to either nn or nelem refinement.
    # Note that nen can be used both with nn or nelem refinement.
    if any(i[0]=='nen' for i in schedule):
        runs_nen = [x[1:] for x in schedule if x[0] == 'nen'][0]
        schedule.remove([x for x in schedule if x[0] == 'nen'][0])
        runs_nelem = [solver.nelem] * len(runs_nen) # reset these as well
    elif any(i[0]=='nelem' for i in schedule):
        runs_nelem = [x[1:] for x in schedule if x[0] == 'nelem'][0]
        schedule.remove([x for x in schedule if x[0] == 'nelem'][0])
        runs_nen = [0] * len(runs_nelem) # reset these as well
    else:
        raise Exception('Convergence schedule must contain either nen or nelem refinement.')
    
    # unpack remaining attributes in schedule, store combinations in cases
    attributes = []
    values = []
    for item in schedule:
        attributes.append(item[0])
        values.append(item[1:])
    cases = list(itertools.product(*values)) # all possible combinations

    ''' Run cases for various runs, store errors '''
    n_cases = len(cases)
    n_runs = len(runs_nelem)
    n_tot = n_cases*n_runs
    n_attributes = len(attributes)
    dofs = np.zeros((n_cases,n_runs)) # store degrees of freedom here
    if jacs:
        avg_jacs = np.zeros((n_cases,n_runs)) # store jacobian ratios here
        max_jacs = np.zeros((n_cases,n_runs))
    if jac_ratios:
        avg_jac_ratios = np.zeros((n_cases,n_runs)) # store jacobian ratios here
        max_jac_ratios = np.zeros((n_cases,n_runs))
    if backout_jacs:
        avg_backout_jacs = np.zeros((n_cases,n_runs))
        max_backout_jacs = np.zeros((n_cases,n_runs))
        backout_neg_jacs = [[False]*n_runs]*n_cases
    if vol_metrics:
        avg_vol_mets = np.zeros((n_cases,n_runs))
        max_vol_mets = np.zeros((n_cases,n_runs))
    if surf_metrics:
        avg_surf_mets = np.zeros((n_cases,n_runs))
        max_surf_mets = np.zeros((n_cases,n_runs))
    neg_jacs = [[False]*n_runs]*n_cases
    legend_strings = ['']*n_cases # store labels for cases here
    if labels is not None:
        assert(len(labels)==n_cases),'labels must be a list of length = n_cases'

    variables = [None]*(2+n_attributes)

    n_toti = 1
    solver.settings['stop_after_metrics'] = True
    solver.settings['calc_exact_metrics'] = True
    for casei in range(n_cases): # for each case
        for atti in range(n_attributes):
            variables[atti+2] = (attributes[atti],cases[casei][atti]) # assign attributes
            legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
        legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting

        for runi in range(n_runs): # for each run (refinement)
            variables[0] = ('nelem',runs_nelem[runi])
            variables[1] = ('nen',runs_nen[runi])
            variables.append(('print_sol_norm', False))

            ''' set case, store results '''
            solver.reset(variables=variables)
            if solver.dim == 1:
                nn = solver.nn
                dofs[casei,runi] = nn
            elif solver.dim == 2:
                nn = solver.nn[0]*solver.nn[1]
                dofs[casei,runi] = np.sqrt(nn)
            elif solver.dim == 3:
                nn = solver.nn[0]*solver.nn[1]*solver.nn[2]
                dofs[casei,runi] = np.cbrt(nn)
            if np.any(solver.mesh.det_jac <= 0):
                neg_jacs[casei][runi] = True
            if jacs:
                temp = abs(solver.mesh.det_jac-solver.mesh.det_jac_exa)
                avg_jacs[casei,runi] = np.mean(temp)
                max_jacs[casei,runi] = np.max(temp)
            if (jac_ratios or backout_jacs):
                if solver.dim == 2:
                    back_jacs = solver.mesh.metrics[:,0,:] * solver.mesh.metrics[:,3,:] - \
                                    solver.mesh.metrics[:,1,:] * solver.mesh.metrics[:,2,:]
                elif solver.dim == 3:
                    back_jacs = np.sqrt( solver.mesh.metrics[:,8,:] * (solver.mesh.metrics[:,0,:] * solver.mesh.metrics[:,4,:] - solver.mesh.metrics[:,1,:] * solver.mesh.metrics[:,3,:]) \
                                       -solver.mesh.metrics[:,7,:] * (solver.mesh.metrics[:,0,:] * solver.mesh.metrics[:,5,:] - solver.mesh.metrics[:,2,:] * solver.mesh.metrics[:,3,:]) \
                                       +solver.mesh.metrics[:,6,:] * (solver.mesh.metrics[:,1,:] * solver.mesh.metrics[:,5,:] - solver.mesh.metrics[:,2,:] * solver.mesh.metrics[:,4,:]))
                    back_jacs = np.nan_to_num(back_jacs, copy=False)
                if jac_ratios:
                    temp = abs( 1 - (back_jacs / solver.mesh.det_jac) )
                    avg_jac_ratios[casei,runi] = np.mean(temp)
                    max_jac_ratios[casei,runi] = np.max(temp)
                if backout_jacs:
                    if np.any(back_jacs <= 0):
                        backout_neg_jacs[casei][runi] = True
                    temp = abs(back_jacs-solver.mesh.det_jac_exa)
                    avg_backout_jacs[casei,runi] = np.mean(temp)
                    max_backout_jacs[casei,runi] = np.max(temp)
            if vol_metrics:
                temp = abs(solver.mesh.metrics-solver.mesh.metrics_exa)
                avg_vol_mets[casei,runi] = np.mean(temp)
                max_vol_mets[casei,runi] = np.max(temp)
            if surf_metrics:
                solver.mesh.ignore_surface_metrics()
                temp = abs(solver.mesh.bdy_metrics-solver.mesh.bdy_metrics_exa)
                temp = np.ma.masked_invalid(temp)
                avg_surf_mets[casei,runi] = temp.mean()
                max_surf_mets[casei,runi] = temp.max()

            print('Progress: run {0} of {1} complete.'.format(n_toti,n_tot))
            if neg_jacs[casei][runi]:
                print('There were negative jacobians.')
            if jacs:
                print('Max Jac Error: ', max_jacs[casei,runi])
                print('Avg Jac Error: ', avg_jacs[casei,runi])
            if jac_ratios:
                print('Max Jac Ratio: ', max_jac_ratios[casei,runi])
                print('Avg Jac Ratio: ', avg_jac_ratios[casei,runi])
            if backout_jacs:
                print('Max Backout Jac Error: ', max_backout_jacs[casei,runi])
                print('Avg Backout Jac Error: ', avg_backout_jacs[casei,runi])
            if vol_metrics:
                print('Max Vol Metrics Error: ', max_vol_mets[casei,runi])
                print('Avg Vol Metrics Error: ', avg_vol_mets[casei,runi])
            if surf_metrics:
                print('Max Surf Metrics Error: ', max_surf_mets[casei,runi])
                print('Avg Surf Metrics Error: ', avg_surf_mets[casei,runi])
            print('Total number of nodes: ', nn)
            print('---------------------------------------------------------')
            n_toti += 1
    if labels is not None:
        # overwrite legend_strings with labels
        legend_strings = labels
    
    ''' Analyze convergence '''
    if jacs:
        print('---------------------------------------------------------')
        print('Average Jacobian Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, avg_jacs, solver.dim,
                                                    legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Maximum Jacobian Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, max_jacs, solver.dim,
                                                    legend_strings=legend_strings)
    if jac_ratios:
        print('---------------------------------------------------------')
        print('Average Jacobian Ratio Convergence Rates:')
        _,_ = calc_conv_rate(dofs, avg_jac_ratios, solver.dim,
                                                    legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Maximum Jacobian Ratio Convergence Rates:')
        _,_ = calc_conv_rate(dofs, max_jac_ratios, solver.dim,
                                                    legend_strings=legend_strings)
    if backout_jacs:
        print('---------------------------------------------------------')
        print('Average Backed-out Jacobian Convergence Rates:')
        _,_ = calc_conv_rate(dofs, avg_backout_jacs, solver.dim,
                                                    legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Maximum Backed-out Jacobian Convergence Rates:')
        _,_ = calc_conv_rate(dofs, max_backout_jacs, solver.dim,
                                                    legend_strings=legend_strings)
    if vol_metrics:
        print('---------------------------------------------------------')
        print('Vol Metrics: Average Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, avg_vol_mets, solver.dim,
                                                    legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Vol Metrics: Maximum Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, max_vol_mets, solver.dim,
                                                legend_strings=legend_strings)
    if surf_metrics:
        print('---------------------------------------------------------')
        print('Surf Metrics: Average Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, avg_surf_mets, solver.dim,
                                                    legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Surf Metrics: Maximum Error Convergence Rates:')
        _,_ = calc_conv_rate(dofs, max_surf_mets, solver.dim,
                                                legend_strings=legend_strings)

    ''' Plot Results '''
        
    if jacs:
        if savefile != None:
            savefile1 = 'jac_' + savefile + '_avg'
            savefile2 = 'jac_' + savefile + '_max'
        else:
            savefile1 = None
            savefile2 = None
        
        title = r"Average Metric Jacobian Error $\left\vert J_h - J_{ex} \right\vert $"
        plot_conv(dofs, avg_jacs, legend_strings, solver.dim, title, savefile1, extra_marker=neg_jacs)
        
        title = r"Maximum Metric Jacobian Error $\left\vert J_h - J_{ex} \right\vert $"
        plot_conv(dofs, max_jacs, legend_strings, solver.dim, title, savefile2, extra_marker=neg_jacs)
        
    if jac_ratios:
        if savefile != None:
            savefile1 = 'jac_ratio_' + savefile + '_avg'
            savefile2 = 'jac_ratio_' + savefile + '_max'
        else:
            savefile1 = None
            savefile2 = None
        
        title = r"Average Metric Jacobian Ratio $\left\vert 1-\frac{J_h}{J_h'} \right\vert $"
        plot_conv(dofs, avg_jac_ratios, legend_strings, solver.dim, title, savefile1, extra_marker=neg_jacs)
        
        title = r"Maximum Metric Jacobian Ratio $\left\vert 1-\frac{J_h}{J_h'} \right\vert $"
        plot_conv(dofs, max_jac_ratios, legend_strings, solver.dim, title, savefile2, extra_marker=neg_jacs)
        
    if backout_jacs:
        if savefile != None:
            savefile1 = 'jac_match_' + savefile + '_avg'
            savefile2 = 'jac_match_' + savefile + '_max'
        else:
            savefile1 = None
            savefile2 = None
        
        title = r"Average Metric Jacobian Error $\left\vert J_h - J_{ex} \right\vert $"
        plot_conv(dofs, avg_backout_jacs, legend_strings, solver.dim, title, savefile1, extra_marker=backout_neg_jacs)
        
        title = r"Maximum Metric Jacobian Error $\left\vert J_h - J_{ex} \right\vert $"
        plot_conv(dofs, max_backout_jacs, legend_strings, solver.dim, title, savefile2, extra_marker=backout_neg_jacs)
    
    if vol_metrics:
        if savefile != None:
            savefile1 = 'met_' + savefile + '_avg'
            savefile2 = 'met_' + savefile + '_max'
        else:
            savefile1 = None
            savefile2 = None
            
        title = r"Average Volume Metrics Error"
        plot_conv(dofs, avg_vol_mets, legend_strings, solver.dim, title, savefile1)
        
        title = r"Maximum Volume Metrics Error"
        plot_conv(dofs, max_vol_mets, legend_strings, solver.dim, title, savefile2)
    
    if surf_metrics:
        if savefile != None:
            savefile1 = 'surf_' + savefile + '_avg'
            savefile2 = 'surf_' + savefile + '_max'
        else:
            savefile1 = None
            savefile2 = None
            
        title = r"Average Surface Metrics Error"
        plot_conv(dofs, avg_surf_mets, legend_strings, solver.dim, title, savefile1)
        
        title = r"Maximum Surface Metrics Error"
        plot_conv(dofs, max_surf_mets, legend_strings, solver.dim, title, savefile2)
        
    
    if return_conv:
        avg_ers = []
        max_ers = []
        if jacs:
            avg_ers.append(avg_jacs)
            max_ers.append(max_jacs)
        if jac_ratios:
            avg_ers.append(avg_jac_ratios)
            max_ers.append(max_jac_ratios)
        if backout_jacs:
            avg_ers.append(avg_backout_jacs)
            max_ers.append(max_backout_jacs)
        if vol_metrics:
            avg_ers.append(avg_vol_mets)
            max_ers.append(max_vol_mets)
        if surf_metrics:
            avg_ers.append(avg_surf_mets)
            max_ers.append(max_surf_mets)
        return dofs, avg_ers, max_ers, legend_strings
    
    
def run_invariants_convergence(solver, schedule_in=None, labels=None):
    '''
    Purpose
    ----------
    Runs a convergence analysis for a given problem. The convergence is
    always done in terms of DOF (solver.nen or solver.nelem).

    Parameters
    ----------
    solver:
    schedule_in: ex. [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    return_conv : bool (optional)
        Flag whether to return dofs, errors, and legend_strings.
        The default is False.
    labels: list of strings (optional)
        labels to use in the legends for the different runs
    '''
    print('---------------------------------------------------------')
    
    if schedule_in==None:
        schedule = [['disc_nodes','lgl','lg'],['p',3,4],['nelem',5,20,100]]
    else:
        schedule=schedule_in.copy()

    ''' Unpack Schedule '''
    # Check that we either use 'nen' or 'nelem'
    # to refine. We can't use both.
    if any(i[0]=='nen' for i in schedule) and any(i[0]=='nelem' for i in schedule):
        print('WARNING: Can not do a refinement specifying both nen and nelem. Using only nelem values.')
        # remove 'nelem' and 'nen' lists
        schedule = [x for x in schedule if not ('nen' in x)]

    # Otherwise, we now have combinations of attributes to run, and we
    # calculate convergence according to either nn or nelem refinement.
    # Note that nen can be used both with nn or nelem refinement.
    if any(i[0]=='nen' for i in schedule):
        runs_nen = [x[1:] for x in schedule if x[0] == 'nen'][0]
        schedule.remove([x for x in schedule if x[0] == 'nen'][0])
        runs_nelem = [solver.nelem] * len(runs_nen) # reset these as well
    elif any(i[0]=='nelem' for i in schedule):
        runs_nelem = [x[1:] for x in schedule if x[0] == 'nelem'][0]
        schedule.remove([x for x in schedule if x[0] == 'nelem'][0])
        runs_nen = [0] * len(runs_nelem) # reset these as well
    else:
        raise Exception('Convergence schedule must contain either nen or nelem refinement.')
    
    # unpack remaining attributes in schedule, store combinations in cases
    attributes = []
    values = []
    for item in schedule:
        attributes.append(item[0])
        values.append(item[1:])
    cases = list(itertools.product(*values)) # all possible combinations

    ''' Run cases for various runs, store errors '''
    n_cases = len(cases)
    n_runs = len(runs_nelem)
    n_tot = n_cases*n_runs
    n_attributes = len(attributes)
    if solver.dim==1:
        raise Exception('Only set up for 2D or 3D')
    else:  
        avg_xRHS = np.zeros((n_cases,n_runs)) 
        max_xRHS = np.zeros((n_cases,n_runs))
        avg_yRHS = np.zeros((n_cases,n_runs)) 
        max_yRHS = np.zeros((n_cases,n_runs))
        avg_xLHS = np.zeros((n_cases,n_runs)) 
        max_xLHS = np.zeros((n_cases,n_runs))
        avg_yLHS = np.zeros((n_cases,n_runs)) 
        max_yLHS = np.zeros((n_cases,n_runs))
        avg_xtot = np.zeros((n_cases,n_runs)) 
        max_xtot = np.zeros((n_cases,n_runs))
        avg_ytot = np.zeros((n_cases,n_runs)) 
        max_ytot = np.zeros((n_cases,n_runs))
    if solver.dim==3:
        avg_zRHS = np.zeros((n_cases,n_runs)) 
        max_zRHS = np.zeros((n_cases,n_runs))
        avg_zLHS = np.zeros((n_cases,n_runs)) 
        max_zLHS = np.zeros((n_cases,n_runs))
        avg_ztot = np.zeros((n_cases,n_runs)) 
        max_ztot = np.zeros((n_cases,n_runs))
    dofs = np.zeros((n_cases,n_runs)) # store degrees of freedom here
    legend_strings = ['']*n_cases # store labels for cases here
    if labels is not None:
        assert(len(labels)==n_cases),'labels must be a list of length = n_cases'

    variables = [None]*(2+n_attributes)

    n_toti = 1
    for casei in range(n_cases): # for each case
        for atti in range(n_attributes):
            variables[atti+2] = (attributes[atti],cases[casei][atti]) # assign attributes
            legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
        legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting

        for runi in range(n_runs): # for each run (refinement)
            variables[0] = ('nelem',runs_nelem[runi])
            variables[1] = ('nen',runs_nen[runi])

            # add a few default things to save time
            # TODO : add flag to also calculate conservation objectives
            variables.append(('print_sol_norm', False))
            variables.append(('bool_plot_sol', False))
            variables.append(('cons_obj_name', None))

            ''' solve run for case, store results '''
            solver.reset(variables=variables)
            solver.settings['calc_exact_metrics'] = True
            
            if solver.dim==2:
                max_xLHS[casei,runi],avg_xLHS[casei,runi],max_yLHS[casei,runi],avg_yLHS[casei,runi],\
                max_xRHS[casei,runi],avg_xRHS[casei,runi],max_yRHS[casei,runi],avg_yRHS[casei,runi],\
                max_xtot[casei,runi],avg_xtot[casei,runi],max_ytot[casei,runi],avg_ytot[casei,runi]=\
                    solver.check_invariants(return_ers=True)
            else:
                max_xLHS[casei,runi],avg_xLHS[casei,runi],max_yLHS[casei,runi],avg_yLHS[casei,runi],max_zLHS[casei,runi],avg_zLHS[casei,runi],\
                max_xRHS[casei,runi],avg_xRHS[casei,runi],max_yRHS[casei,runi],avg_yRHS[casei,runi],max_zRHS[casei,runi],avg_zRHS[casei,runi],\
                max_xtot[casei,runi],avg_xtot[casei,runi],max_ytot[casei,runi],avg_ytot[casei,runi],max_ztot[casei,runi],avg_ztot[casei,runi]=\
                    solver.check_invariants(return_ers=True)

            if solver.dim == 1:
                nn = solver.nn
                dofs[casei,runi] = nn
            elif solver.dim == 2:
                nn = solver.nn[0]*solver.nn[1]
                dofs[casei,runi] = np.sqrt(nn)
            elif solver.dim == 3:
                nn = solver.nn[0]*solver.nn[1]*solver.nn[2]
                dofs[casei,runi] = np.cbrt(nn)

            print('Convergence Progress: run {0} of {1} complete.'.format(n_toti,n_tot))
            print('Total number of nodes: ', nn)
            print('---------------------------------------------------------')
            n_toti += 1
        if np.all(max_xLHS[casei,:] < 1e-13): max_xLHS[casei,:]=0
        if np.all(avg_xLHS[casei,:] < 1e-13): avg_xLHS[casei,:]=0
        if np.all(max_xRHS[casei,:] < 1e-13): max_xRHS[casei,:]=0
        if np.all(avg_xRHS[casei,:] < 1e-13): avg_xRHS[casei,:]=0
        if np.all(max_xtot[casei,:] < 1e-13): max_xtot[casei,:]=0
        if np.all(avg_xtot[casei,:] < 1e-13): avg_xtot[casei,:]=0
        if np.all(max_yLHS[casei,:] < 1e-13): max_yLHS[casei,:]=0
        if np.all(avg_yLHS[casei,:] < 1e-13): avg_yLHS[casei,:]=0
        if np.all(max_yRHS[casei,:] < 1e-13): max_yRHS[casei,:]=0
        if np.all(avg_yRHS[casei,:] < 1e-13): avg_yRHS[casei,:]=0
        if np.all(max_ytot[casei,:] < 1e-13): max_ytot[casei,:]=0
        if np.all(avg_ytot[casei,:] < 1e-13): avg_ytot[casei,:]=0
        if solver.dim==3:
            if np.all(max_zLHS[casei,:] < 1e-13): max_zLHS[casei,:]=0
            if np.all(avg_zLHS[casei,:] < 1e-13): avg_zLHS[casei,:]=0
            if np.all(max_zRHS[casei,:] < 1e-13): max_zRHS[casei,:]=0
            if np.all(avg_zRHS[casei,:] < 1e-13): avg_zRHS[casei,:]=0
            if np.all(max_ztot[casei,:] < 1e-13): max_ztot[casei,:]=0
            if np.all(avg_ztot[casei,:] < 1e-13): avg_ztot[casei,:]=0
    if labels is not None:
        # overwrite legend_strings with labels
        legend_strings = labels
    
    ''' Analyze convergence '''
    print('---------------------------------------------------------')
    print('Max x LHS')
    _,conv_max_xLHS = calc_conv_rate(dofs, max_xLHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg x LHS')
    _,conv_avg_xLHS = calc_conv_rate(dofs, avg_xLHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Max x RHS')
    _,conv_max_xRHS = calc_conv_rate(dofs, max_xRHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg x RHS')
    _,conv_avg_xRHS = calc_conv_rate(dofs, avg_xRHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Max x tot')
    _,conv_max_xtot = calc_conv_rate(dofs, max_xtot, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg x tot')
    _,conv_avg_xtot = calc_conv_rate(dofs, avg_xtot, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Max y LHS')
    _,conv_max_yLHS = calc_conv_rate(dofs, max_yLHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg y LHS')
    _,conv_avg_yLHS = calc_conv_rate(dofs, avg_yLHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Max y RHS')
    _,conv_max_yRHS = calc_conv_rate(dofs, max_yRHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg y RHS')
    _,conv_avg_yRHS = calc_conv_rate(dofs, avg_yRHS, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Max y tot')
    _,conv_max_ytot = calc_conv_rate(dofs, max_ytot, solver.dim, legend_strings=legend_strings)
    print('---------------------------------------------------------')
    print('Avg y tot')
    _,conv_avg_ytot = calc_conv_rate(dofs, avg_ytot, solver.dim, legend_strings=legend_strings)
    if solver.dim==3:
        print('---------------------------------------------------------')
        print('Max z LHS')
        _,conv_max_zLHS = calc_conv_rate(dofs, max_zLHS, solver.dim, legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Avg z LHS')
        _,conv_avg_zLHS = calc_conv_rate(dofs, avg_zLHS, solver.dim, legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Max z RHS')
        _,conv_max_zRHS = calc_conv_rate(dofs, max_zRHS, solver.dim, legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Avg z RHS')
        _,conv_avg_zRHS = calc_conv_rate(dofs, avg_zRHS, solver.dim, legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Max z tot')
        _,conv_max_ztot = calc_conv_rate(dofs, max_ztot, solver.dim, legend_strings=legend_strings)
        print('---------------------------------------------------------')
        print('Avg z tot')
        _,conv_avg_ztot = calc_conv_rate(dofs, avg_ztot, solver.dim, legend_strings=legend_strings)
    

    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('                    SUMMARY:                             ')
    print('---------------------------------------------------------')
    for casei in range(n_cases): # for each case
        print('---------------------------------------------------------')
        print('Convergence for ', legend_strings[casei], ': ')
        if solver.dim==2:
            res = [np.mean([conv_max_xLHS[casei],conv_max_yLHS[casei]]),\
                   np.mean([conv_avg_xLHS[casei],conv_avg_yLHS[casei]]),\
                   np.mean([conv_max_xRHS[casei],conv_max_yRHS[casei]]),\
                   np.mean([conv_avg_xRHS[casei],conv_avg_yRHS[casei]]),\
                   np.mean([conv_max_xtot[casei],conv_max_ytot[casei]]),\
                   np.mean([conv_avg_xtot[casei],conv_avg_ytot[casei]])]
            for i in range(len(res)):
                if np.isnan(res[i]):
                    res[i] = 'exact'
            print('Max LHS: ', res[0])
            print('Avg LHS: ', res[1])
            print('Max RHS: ', res[2])
            print('Avg RHS: ', res[3])
            print('Max tot: ', res[4])
            print('Avg tot: ', res[5])
        else:
            res = [np.mean([conv_max_xLHS[casei],conv_max_yLHS[casei],conv_max_zLHS[casei]]),\
                   np.mean([conv_avg_xLHS[casei],conv_avg_yLHS[casei],conv_avg_zLHS[casei]]),\
                   np.mean([conv_max_xRHS[casei],conv_max_yRHS[casei],conv_max_zRHS[casei]]),\
                   np.mean([conv_avg_xRHS[casei],conv_avg_yRHS[casei],conv_avg_zRHS[casei]]),\
                   np.mean([conv_max_xtot[casei],conv_max_ytot[casei],conv_max_ztot[casei]]),\
                   np.mean([conv_avg_xtot[casei],conv_avg_ytot[casei],conv_avg_ztot[casei]])]
            for i in range(len(res)):
                if np.isnan(res[i]):
                    res[i] = 'exact'
            print('Max LHS: ', res[0])
            print('Avg LHS: ', res[1])
            print('Max RHS: ', res[2])
            print('Avg RHS: ', res[3])
            print('Max tot: ', res[4])
            print('Avg tot: ', res[5])
            



def read_from_diablo(filename=None):
    'read data from diablo and turn it into something for plotting'
    if filename == None:
        filename = '../../../../jac_data.npz'
    file = np.load(filename)
    rho_er = file['rho']
    rhou_er = file['rhou']
    rhov_er = file['rhov']
    e_er = file['e']
    p_er = file['p']
    ent_er = file['ent']
    nodes = file['nodes']
    nodes_verify = file['nodes_verify']
    cases = file['cases']
    ops = file['ops']
    shape = file['shape']
    
    ops = ops.tolist()
    for i in range(len(ops)):
        ops[i] = ops[i].decode("utf-8")
        
    cases = cases.tolist()
    for i in range(len(cases)):
        cases[i] = cases[i].decode("utf-8")
        
    shape = shape.tolist()
    for i in range(len(shape)):
        shape[i] = shape[i].decode("utf-8")
    
    dofs = np.sqrt(nodes)


def plot_eigs(A, plot_hull=True, plot_individual_eigs=False, labels=None, savefile=None,
              save_format='png', dpi=600, line_width=1.5, equal_axes=False, 
              title_size=12, legend_size=12, markersize=16, markeredge=2,
              tick_size=12, serif=True, left_space_pct=None,
              colors=None, markers=None, linestyles=None, legend_loc='best', 
              legend_anchor=None, legend_anchor_type=None, legend_alpha=None):
    if plot_hull:
        from scipy.spatial import ConvexHull

    if serif:
        plt.rcParams['font.family'] = 'serif'
    else:
        plt.rcParams['font.family'] = 'sans-serif'

    # Define colors and linestyles for the hulls
    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'm', 'tab:red', 'tab:brown']
    if markers is None:
        markers = ['o', '^', 's', 'd','x', '+']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (1, 2, 3, 2))]
    
    # Check if A is a list of matrices or a single matrix
    if isinstance(A, list):
        if labels is None:
            labels = [f'A{i+1}' for i in range(len(A))]
        elif len(labels) != len(A):
            raise ValueError("Length of labels must match the number of matrices in A.")
        
        print(f"Calculating eigenvalues for {', '.join(labels)}")
        
        eig_values_list = [np.linalg.eigvals(matrix) for matrix in A]
    else:
        # Single matrix case
        if labels is not None:
            raise ValueError("Labels should be None when A is a single matrix.")
        
        print("Calculating eigenvalues for matrix A")
        eig_values_list = [np.linalg.eigvals(A)]
        labels = ["A"]  # Default label for a single matrix
    
    plt.figure(figsize=(6, 6))
    
    # Loop through each matrix's eigenvalues
    for idx, (eig_values, label) in enumerate(zip(eig_values_list, labels)):
        real_part = eig_values.real
        imag_part = eig_values.imag
        
        # Check if eigenvalues are all purely imaginary (within tolerance)
        if np.all(np.abs(real_part) < 1e-12):
            # Plot a line along the imaginary axis
            plt.plot(np.zeros_like(imag_part), imag_part, color=colors[idx % len(colors)], 
                     linestyle=linestyles[idx % len(linestyles)], linewidth=line_width)
        else:
            # Plot individual eigenvalues as scatter if plot_individual_eigs is True
            if plot_individual_eigs:
                if markers[idx % len(markers)] in ['x', '+', '|', '_']:  # Markers without a face color
                    plt.scatter(real_part, imag_part, s=markersize,
                                marker=markers[idx % len(markers)], facecolors=colors[idx % len(colors)],
                                linewidths=markeredge)
                else:
                    plt.scatter(real_part, imag_part, s=markersize,
                                marker=markers[idx % len(markers)], facecolors='none', edgecolors=colors[idx % len(colors)], 
                                linewidths=markeredge)
            
            # Plot convex hull around the outermost eigenvalues
            if plot_hull and len(eig_values) > 2:  # ConvexHull needs at least 3 points
                points = np.column_stack((real_part, imag_part))
                hull = ConvexHull(points)
                
                # Plot the convex hull with unique color and linestyle
                #for simplex in hull.simplices:
                #    plt.plot(points[simplex, 0], points[simplex, 1], color=colors[idx % len(colors)], 
                #             linestyle=linestyles[idx % len(linestyles)], linewidth=line_width)
                
                # Extract the vertices of the convex hull and add the first point at the end to close the loop
                hull_vertices = np.append(hull.vertices, hull.vertices[0])

                # Plot the convex hull as a single continuous line
                plt.plot(points[hull_vertices, 0], points[hull_vertices, 1], 
                        color=colors[idx % len(colors)], 
                        linestyle=linestyles[idx % len(linestyles)], 
                        linewidth=line_width)
                
            # Add a dummy artist to create a combined legend entry for both scatter and hull
            if plot_individual_eigs and plot_hull:
                plt.plot([], [], color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)], 
                        linewidth=line_width, marker=markers[idx % len(markers)], markersize=np.sqrt(markersize),
                        markerfacecolor='none', markeredgewidth=markeredge, label=label)
            elif plot_individual_eigs:
                if markers[idx % len(markers)] in ['x', '+', '|', '_']:  # Markers without a face color
                    plt.scatter([], [], s=markersize,
                                marker=markers[idx % len(markers)], facecolors=colors[idx % len(colors)],
                                linewidths=markeredge, label=label)
                else:
                    plt.scatter([], [], s=markersize,
                                marker=markers[idx % len(markers)], facecolors='none', edgecolors=colors[idx % len(colors)], 
                                linewidths=markeredge, label=label)
            elif plot_hull:
                plt.plot([], [], color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)], 
                        linewidth=line_width, label=label)
    
    # Add grid and labels
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real Part', fontsize=title_size)
    plt.ylabel('Imaginary Part', fontsize=title_size)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=tick_size) 
    plt.grid(True)

    if left_space_pct is not None:
        ax_min,ax_max = ax.get_xlim()
        ay_min,ay_max = ax.get_ylim()
        all_reals = np.concatenate([np.real(eig_values) for eig_values in eig_values_list])
        xmin = np.min(all_reals)
        if xmin < 0:
            ax_min = xmin - left_space_pct * abs(xmin)
        ax.set_xlim(ax_min, ax_max)
    
    # Fix axes ratio if equal_axes is True
    if equal_axes:
        if left_space_pct is not None:
            side = max(ax_max-ax_min, ay_max-ay_min)
            # For x, we want to keep the left side fixed:
            ax.set_xlim(ax_min, ax_min + side)
            # For y, we can center the data vertically:
            y_center = (ay_min + ay_max) / 2. # Should be zero anyway...
            ax.set_ylim(y_center - side/2, y_center + side/2)
        else:
            #ax.set_aspect('equal', adjustable='box') # This will change the shape of the plot
            ax.set_aspect('equal', adjustable='datalim') # This will keep the shape of the plot

    # Show legend only if A is a list
    if isinstance(A, list):
        # Create a blended transform
        from matplotlib.transforms import blended_transform_factory
        if legend_anchor_type == 'data' or legend_anchor_type == ('data','data'):
            bbox_transform = ax.transData
        elif legend_anchor_type == 'fig' or legend_anchor_type == ('fig','fig'):
            bbox_transform = plt.gcf().transFigure
        elif legend_anchor_type == ('data','fig'):
            bbox_transform = blended_transform_factory(ax.transData, plt.gcf().transFigure)
        elif legend_anchor_type == ('fig','data'):
            bbox_transform = blended_transform_factory(plt.gcf().transFigure, ax.transData)
        elif legend_anchor_type == None:
            bbox_transform = ax.transData
        else:
            print("Invalid legend_anchor_type. Try of the format ('data','fig'). Using default 'data' type.")
            bbox_transform = ax.transData
        
        legend = plt.legend(fontsize=legend_size,loc=legend_loc, 
                   bbox_to_anchor=legend_anchor, bbox_transform=bbox_transform)
        
        if legend_alpha is not None:
            legend.get_frame().set_alpha(legend_alpha)

    # Save the plot if a savefile is provided
    if savefile is not None:
        plt.savefig(savefile, format=save_format, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {savefile}")
    else:
        plt.show()





'''
plot_conv(dofs[0,:,:], p_er[0,:,:], ops, 2, title=cases[0], ylabel=r'Pressure Error $\vert \vert p \vert \vert_H$', skip=[3,3,2,3,3,2])
plot_conv(dofs[1,:,:], p_er[1,:,:], ops, 2, title=cases[1], ylabel=r'Pressure Error $\vert \vert p \vert \vert_H$', skip=[1,3,2,3,3,2])
plot_conv(dofs[2,:,:], p_er[2,:,:], ops, 2, title=cases[2], ylabel=r'Pressure Error $\vert \vert p \vert \vert_H$', skip=[3,3,2,3,3,2])

  
      
filename = '../../../../met_data.npz'
file = np.load(filename)
rho_er = file['rho']
rhou_er = file['rhou']
rhov_er = file['rhov']
e_er = file['e']
p_er = file['p']
ent_er = file['ent']
fun_er = file['fun']
avg_met_er = file['avg_mets']
max_met_er = file['max_mets']
l2freestream = file['l2freestream']
linf_surfgcl = file['linf_surfgcl']
nodes = file['nodes']
cases = file['cases']
ops = file['ops']
shape = file['shape']

ops = ops.tolist()
for i in range(len(ops)):
    ops[i] = ops[i].decode("utf-8")
    
cases = cases.tolist()
for i in range(len(cases)):
    cases[i] = cases[i].decode("utf-8")
    
shape = shape.tolist()
for i in range(len(shape)):
    shape[i] = shape[i].decode("utf-8")

dofs = np.sqrt(nodes)
    
'''