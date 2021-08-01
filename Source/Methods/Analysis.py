#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:17:18 2021

@author: bercik
"""
import subprocess # see https://www.youtube.com/watch?v=2Fp1N6dof0Y for tutorial
import numpy as np
import matplotlib.pyplot as plt
from sympy import nsimplify
from os import path
import itertools
from tabulate import tabulate
import scipy.optimize as sc

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
    assert('plt_save_name' not in plotargs),"plt_save_name should not be set in plotargs"
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
    ax.plot([0,0],[np.min(solver.q_sol),np.max(solver.q_sol)])
    ymin,ymax= ax.get_ylim()
    plt.close()
    if 'solmin' not in plotargs: plotargs['solmin']=ymin
    if 'solmax' not in plotargs: plotargs['solmax']=ymax
    
    # Make directory in which files will be saved
    if path.exists(file_name):
        print('WARNING: Folder name already exists. Using a temporary name instead.')
        file_name += '_RENAME'
    cmd = subprocess.run('mkdir '+file_name, shell=True, capture_output=True, text=True)
    assert(cmd.returncode==0),"Not able to make directory. Error raised: "+cmd.stderr
    
    print('...Making Frames')    
    for i in range(frames):
        stepi = i*(skipsteps+1)
        timei =  solver.dt*stepi # or tfinal/steps*stepi
        
        # call plotting function from solver module
        plot(solver.q_sol[:,:,stepi], **plotargs, time=timei,
             plt_save_name=file_name+'/'+'frame'+str(i).zfill(numfill))
    
    if last_frame:
        plot(solver.q_sol[:,:,-1], **plotargs, time=tfinal,
             plt_save_name=file_name+'/'+'frame'+str(frames))

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
    time = np.arange(0,tfinal,step=solver.dt)
    steps = len(time)

    assert(solver.q_sol.ndim == 3),'solver.q_sol of wrong dimension'
    assert(A.shape==(solver.nn,solver.nn)),'The solution array {0} does not match operator shape {1}'.format(solver.nn,A.shape)
    sols = solver.q_sol[:,:,:steps].reshape((solver.nn,steps),order='F')
    
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
                        plt.plot(time_avg,100*(abs(C_fit_avg[i,:])-C0_fit)/C0_fit,label='Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                    plt.ylabel(r'\% Change in Coefficient',fontsize=14)
                else:
                    plt.plot(time_avg,abs(C_avg[i,:])-C0,label='Numerical')
                    if plot_theory:
                        plt.plot(time_avg,abs(C_theory_avg[i,:])-C0_theory,label='Theory')
                    if plot_theory:
                        plt.plot(time_avg,abs(C_fit_avg[i,:])-C0_fit,label='Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                    plt.ylabel(r'Change in Coefficient',fontsize=14)
            else:
                plt.plot(time_avg,abs(C_avg[i,:]),label='Numerical')
                if plot_theory:
                    plt.plot(time_avg,abs(C_theory_avg[i,:]),label='Theory')
                if plot_theory:
                    plt.plot(time_avg,abs(C_fit_avg[i,:]),label='Fit $\lambda = {0:.2f}$'.format(fit_vals[i]))
                plt.ylabel(r'Coefficient',fontsize=14)
            if plot_theory or fit_data:
                plt.legend(loc='upper left',fontsize=12)
            if window_size >1:
                plt.xlabel(r'Average Time (window size {0:.2f}s)'.format(solver.dt*window_size),fontsize=14)
                if save_file is not None:
                    plt.savefig(save_file+'_eig{0}_avg.eps'.format(i), format='eps')
            else: 
                plt.xlabel(r'Time',fontsize=14)
                if save_file is not None:
                    plt.savefig(save_file+'_eig{0}.eps'.format(i), format='eps')


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
             scale_dt=True, return_conv=False, savefile=None, labels=None):
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
        if solver.dim == 1:
            xmin, xmax, nn = solver.xmin, solver.xmax, solver.nn
        else:
            # only estimate new dt based on scaling of x coordinate
            xmin, xmax, nn = solver.xmin[0], solver.xmax[0], solver.nn[0]
        base_dx = (xmax - xmin) / nn

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
    errors = np.zeros((n_cases,n_runs)) # store errors here
    dofs = np.zeros((n_cases,n_runs)) # store degrees of freedom here
    legend_strings = ['']*n_cases # store labels for cases here
    if labels is not None:
        assert(len(labels)==n_cases),'labels must be a list of length = n_cases'

    if scale_dt: variables = [None]*(3+n_attributes) # initiate list to pass to reset()
    else: variables = [None]*(2+n_attributes)

    n_toti = 1
    for casei in range(n_cases): # for each case
        for atti in range(n_attributes):
            variables[atti+2] = (attributes[atti],cases[casei][atti]) # assign attributes
            legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
        legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting
        # if we change p, estimate that nen will also change by the same amount
        if 'p' in attributes:
                    diff_p = cases[casei][attributes.index('p')] - solver.p
        for runi in range(n_runs): # for each run (refinement)
            variables[0] = ('nelem',runs_nelem[runi])
            variables[1] = ('nen',runs_nen[runi])
            if scale_dt:
                if runs_nen[runi] == 0: neni = solver.nen
                else: neni = runs_nen[runi]
                if solver.dim == 1:
                    xmin, xmax, nelem = solver.xmin, solver.xmax, runs_nelem[runi]
                else:
                    # only estimate new dt based on scaling of x coordinate
                    xmin, xmax = solver.xmin[0], solver.xmax[0]
                    if isinstance(runs_nelem[runi], tuple):
                        nelem = runs_nelem[runi][0]
                    else: nelem = runs_nelem[runi]
                new_dx = (xmax - xmin) / (nelem*(neni+diff_p))
                new_dt = base_dt * new_dx / base_dx
                variables[-1] = ('dt',new_dt)
            # add a few default things to save time
            # TODO : add flag to also calculate conservation objectives
            variables.append(('print_sol_norm', False))
            variables.append(('bool_plot_sol', False))
            variables.append(('cons_obj_name', None))

            ''' solve run for case, store results '''
            solver.reset(variables=variables)
            solver.solve()
            errors[casei,runi] = solver.calc_error(method=error_type)
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
            print('Final Error: ', solver.calc_error())
            print('Total number of nodes: ', nn)
            print('---------------------------------------------------------')
            n_toti += 1
    if labels is not None:
        # overwrite legend_strings with labels
        legend_strings = labels
        
    ##########################################################################
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

    def plot_conv(dof_vec, err_vec, legend_strings, dim, title=None, savefile=None):
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

        Returns
        -------
        None
        '''
        
        assert dof_vec.shape==err_vec.shape,"The two inputted arrays are not the same shape!"
        if dof_vec.ndim>1:
            n_cases, n_runs = dof_vec.shape
        else:
            n_cases, n_runs = 1, dof_vec.size
            dof_vec = np.reshape(dof_vec,(n_cases, n_runs))
            err_vec = np.reshape(err_vec,(n_cases, n_runs))
        assert n_runs>1,"ERROR: Not enough grids to perform convergence."
        
        assert len(legend_strings)==n_cases,"ERROR: legend_strings do not match n_cases"

        plt.figure(figsize=(6,4))
        if title is not None:
            plt.title(title,fontsize=18)
        plt.ylabel(r"Error",fontsize=16)
        if dim == 1: plt.xlabel(r"Degrees of Freedom",fontsize=16)
        elif dim == 2: plt.xlabel(r"$\sqrt{}$ Degrees of Freedom",fontsize=16)
        elif dim == 3: plt.xlabel(r"$\sqrt[3]{}$ Degrees of Freedom",fontsize=16)
        
        # Do a curve fit to find the slope on log-log plot (order of convergence)
        def fit_func(x, a, b): 
            return -a*x + b
        
        colors=['blue','red','green','magenta','orange','purple','brown']
        marks=['o','^','s','D','>','<','8']
        
        for i in range(n_cases):
            string = legend_strings[i].replace("disc_nodes=","")
            p_opt, p_cov = sc.curve_fit(fit_func, np.log(dof_vec[i]),np.log(err_vec[i]),(2,1)) # fit
            acc = int(np.floor(np.log10(np.sqrt(p_cov[0,0]))))
            unc = np.round(np.sqrt(p_cov[0,0]),abs(acc))
            acc = int(np.floor(np.log10(unc)))
            if acc >=0:
                slope = r": {0} $\pm$ {1}".format(int(p_opt[0]),int(unc))
            elif acc==-1:
                slope = r": {0:9.1f} $\pm$ {1:6.1f}".format(p_opt[0],unc)
            elif acc==-2:
                slope = r": {0:9.2f} $\pm$ {1:6.2f}".format(p_opt[0],unc)
            elif acc==-3:
                slope = r": {0:9.3f} $\pm$ {1:6.3f}".format(p_opt[0],unc)
            else:
                slope = r": {0:9.4f} $\pm$ {1:6.1g}".format(p_opt[0],unc)
            plt.loglog(dof_vec[i],err_vec[i],marks[i],markersize=8, color=colors[i],
                       label=string+slope)
            plt.loglog(np.linspace(dof_vec[i][0],dof_vec[i][-1]), # plot
                       np.exp(fit_func(np.log(np.linspace(dof_vec[i][0],dof_vec[i][-1])), *p_opt)), 
                       linewidth=1, linestyle = '--', color=colors[i])
        plt.legend(loc='best', fontsize=12)
        #plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        #plt.gca().xaxis.set_major_formatter(tik.ScalarFormatter())
        #plt.gca().xaxis.set_minor_formatter(tik.ScalarFormatter())
        if savefile is not None:
            plt.savefig(savefile,dpi=600)
    
    ##########################################################################
    
    ''' Analyze convergence '''
    print('---------------------------------------------------------')
    print(error_type + ' Error Convergence Rates:')
    conv_vec, avg_conv = calc_conv_rate(dofs, errors, solver.dim,
                                                    legend_strings=legend_strings)

    ''' Plot Results '''
    # use diffeq.plot_obj?
    title = r"Convergence of " + error_type + " Error"
    plot_conv(dofs, errors, legend_strings, solver.dim, title, savefile)
    
    if return_conv:
        return dofs, errors, legend_strings