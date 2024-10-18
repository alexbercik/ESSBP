#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:33:50 2020

@author: bercik
"""

from Source.Disc.MakeMesh import MakeMesh
from Source.Solvers.PdeSolver import PdeSolver

class PdeSolverFd(PdeSolver):

    def init_disc_specific(self):

        # TODO: create these functions for FD and uncomment lines below
        # self.energy = self.fd_energy
        # self.conservation = self.fd_conservation
        # self.calc_error = self.fd_calc_error

        if self.cons_obj_name is not None:
            self.cons_obj_name = None
            self.bool_calc_cons_obj = False
            self.n_cons_obj = 0
            print('WARNING: No conservation objectives currently defined for Finite Difference. Ignoring.')

        ''' Verify inputs '''

        assert self.disc_type.lower() == 'fd', 'Invalid spatial discretization type'

        ''' Calculate other parameters '''

        self.mesh = MakeMesh(self.xmin, self.xmax, self.isperiodic,
                                 nn = self.nn)

        self.xy = self.mesh.xy
        self.diffeq.set_mesh(self.mesh)
        self.diffeq.set_fd_op(self.p)
        self.dfdq = None # use default diffeq ones
        self.dqdt = None
        
