#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:17:18 2021

@author: bercik
"""
import subprocess # see https://www.youtube.com/watch?v=2Fp1N6dof0Y for tutorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    if 'ymin' not in plotargs: plotargs['ymin']=ymin
    if 'ymax' not in plotargs: plotargs['ymax']=ymax
    
    # Make directory in which files will be saved
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
            popt, pcov = curve_fit(fit_func, time, abs(C[i,:]),p0=(0,np.imag(C[i,0])))
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
         