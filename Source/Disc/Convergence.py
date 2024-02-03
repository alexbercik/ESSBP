#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:52:32 2020

@author: alexbercik
"""

from sys import platform
import numpy as np
import itertools
from tabulate import tabulate
import scipy.optimize as sc

# Check if this is being run on SciNet
if platform == "linux" or platform == "linux2": # True if on SciNet
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


class Convergence:

    def run_conv(solver, schedule_in=None, error_type='Rms',
                 scale_dt=True, return_conv=False, savefile=None):
        '''
        Purpose
        ----------
        Runs a convergence analysis for a given problem. The convergence is
        always done in terms of DOF (solver.nn). The inputs for any

        Parameters
        ----------
        solver:
        schedule_in:
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
        '''
        if schedule_in==None:
            schedule = [['spat_disc_type','Rd','R0'],['p',3,4],['nn',50,100,200,400]]
        else:
            schedule=schedule_in.copy()

        if scale_dt:
            base_dt = solver.dt
            base_dx = (solver.xmax - solver.xmin) / solver.nn

        ''' Unpack Schedule '''
        # Check that we either use 'nn' or 'nelem'
        # to refine. We can't use both.
        if any(i[0]=='nn' for i in schedule) and any(i[0]=='nelem' for i in schedule):
            print('WARNING: Can not do a refinement specifying both nn and nelem. Using only nn values.')
            # remove 'nelem' and 'nen' lists
            schedule = [x for x in schedule if not ('nelem' in x)]

        # Otherwise, we now have combinations of attributes to run, and we
        # calculate convergence according to either nn or nelem refinement.
        # Note that nen can be used both with nn or nelem refinement.
        if any(i[0]=='nn' for i in schedule):
            runs_nn = [x[1:] for x in schedule if x[0] == 'nn'][0]
            schedule.remove([x for x in schedule if x[0] == 'nn'][0])
            runs_nelem = [0] * len(runs_nn) # reset these as well
        elif any(i[0]=='nelem' for i in schedule):
            runs_nelem = [x[1:] for x in schedule if x[0] == 'nelem'][0]
            schedule.remove([x for x in schedule if x[0] == 'nelem'][0])
            runs_nn = [0] * len(runs_nelem) # reset these as well
        else:
            raise Exception('Concergence schedule must contain either nn or nelem refinement.')
        if any(i[0]=='nen' for i in schedule):
            runs_nen = [x[1:] for x in schedule if x[0] == 'nen'][0]
            schedule.remove([x for x in schedule if x[0] == 'nen'][0])
        else:
            runs_nen = [0] * len(runs_nn)

        # unpack remaining attributes in schedule, store combinations in cases
        attributes = []
        values = []
        for item in schedule:
            attributes.append(item[0])
            values.append(item[1:])
        cases = list(itertools.product(*values)) # all possible combinations

        ''' Run cases for various runs, store errors '''
        n_cases = len(cases)
        n_runs = len(runs_nn)
        n_tot = n_cases*n_runs
        n_attributes = len(attributes)
        errors = np.zeros((n_cases,n_runs)) # store errors here
        dofs = np.zeros((n_cases,n_runs)) # store degrees of freedom here
        legend_strings = ['']*n_cases # store labels for cases here

        if scale_dt: variables = [None]*(4+n_attributes) # initiate list to pass to reset()
        else: variables = [None]*(3+n_attributes)

        n_toti = 1
        for casei in range(n_cases): # for each case
            for atti in range(n_attributes):
                variables[atti+3] = (attributes[atti],cases[casei][atti]) # assign attributes
                legend_strings[casei] += '{0}={1}, '.format(attributes[atti],cases[casei][atti])
            legend_strings[casei] = legend_strings[casei].strip().strip(',') # formatting
            for runi in range(n_runs): # for each run (refinement)
                variables[0] = ('nn',runs_nn[runi])
                variables[1] = ('nelem',runs_nelem[runi])
                variables[2] = ('nen',runs_nen[runi])
                if scale_dt:
                    new_dx = (solver.xmax - solver.xmin) / max(runs_nn[runi],runs_nelem[runi]*runs_nen[runi])
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
                dofs[casei,runi] = solver.nn

                print('Progress: run {0} of {1} complete.'.format(n_toti,n_tot))
                n_toti += 1

        ''' Analyze convergence '''
        print(error_type + ' Error Convergence Rates')
        print('------------------------------------------')
        conv_vec, avg_conv = Convergence.calc_conv_rate(dofs, errors,
                                                        legend_strings=legend_strings)

        ''' Plot Results '''
        title = r"Convergence of " + error_type + " Error"
        Convergence.plot_conv(dofs, errors,legend_strings,title,savefile)
        
        if return_conv:
            return dofs, errors, legend_strings


    def calc_conv_rate(dof_vec, err_vec, n_points=None,
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
                print(tabulate((data), headers=['Ndof', 'Error','Convergence'], tablefmt='orgtbl'))

        return conv_vec, avg_conv

    def plot_conv(dof_vec, err_vec,legend_strings,title=None,savefile=None):
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
        plt.xlabel(r"Degrees of Freedom",fontsize=16)
        
        # Do a curve fit to find the slope on log-log plot (order of convergence)
        def fit_func(x, a, b): 
            return -a*x + b
        
        colors=['blue','red','green','magenta','orange','purple','brown']
        marks=['o','^','s','D','>','<','8']
        
        for i in range(n_cases):
            string = legend_strings[i].replace("spat_disc_type=","")
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
