#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:17:18 2021

@author: bercik
"""
import subprocess # see https://www.youtube.com/watch?v=2Fp1N6dof0Y for tutorial
import numpy as np
import matplotlib.pyplot as plt

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
            plt_save_name (optional) : name of saved file, without file extension,
            show_fig (optional) : Boolean to be set to False
            ymin & ymax (optional) : y-axis limits
        The default is solver.diffeq.plot_sol
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
        frames = int((tfinal-solver.t_init)/((skipsteps+1)*solver.dt)) + 1 # note int acts as a floor fn
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
        timei = solver.t_init + solver.dt*stepi # or (tfinal-solver.t_init)/steps*stepi
        
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

