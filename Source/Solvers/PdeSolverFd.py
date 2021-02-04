#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:33:50 2020

@author: andremarchildon
"""

from Source.Disc.MakeMesh import MakeMesh

class PdeSolverFd():

    def fd_init(self):

        ''' Verify inputs '''

        assert self.disc_type == 'fd', 'Invalid spatial discretization type'

        ''' Calculate other parameters '''

        self.mesh = MakeMesh(self.xmin, self.xmax, self.isperiodic,
                                 nn = self.nn)

        self.xy = self.mesh.xy
        self.diffeq.set_mesh(self.mesh)
        self.diffeq.set_fd_op(self.p,False)
        self.diffeq_in = self.diffeq
