#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:54:23 2020

@author: andremarchildon
"""

import numpy as np
from Source.Disc.BasisFun import BasisFun
from Source.Disc.RefSimplexElem import RefSimplexElem


class MakeMesh:

    dim = 1

    def __init__(self, xmin, xmax, isperiodic,
                 nelem=1, xy_op=None,   # Set for elem mesh (SBP op)
                 nn=None):              # Set for uniform mesh (FD op)
        '''
        Parameters
        ----------
        xmin : int
            Min coordinate of the mesh.
        xmax : int
             Max coordinate of the mesh..
        isperiodic : bool
            True if the mesh is periodic.
        nelem : int, optional
            Only specify for an elem mesh (ie if using SBP operators).
            No. of elements in the mesh
            The default is 1.
        xy_op : TYPE, optional
            Only specify for an elem mesh (ie if using SBP operators).
            Indicates the nodal locations for one element.
            The default is None.
        nn : int, optional
            Only specify for a uniform mesh (ie if using FD).
            The default is None.
        '''

        ''' Add all inputs to the class '''

        self.xmin = xmin
        self.xmax = xmax
        self.isperiodic = isperiodic
        self.nelem = nelem
        self.xy_op = xy_op
        self.nn = nn

        ''' Additional terms '''

        self.dom_len = self.xmax - self.xmin

        ''' Call appropriate mesh constructor '''

        if self.xy_op is None:
            assert nn is not None, 'nn must be set for a uniform mesh (FD op)'
            assert nelem == 1, 'For a uniform mesh nelem must be one'

            self.build_uniform_mesh()
        else:
            assert nn is None, 'nn should not be specified for an elem mesh'
            self.dx = None # dx is not constant bewteen nodes in an element
            self.build_elem_mesh()

    def build_uniform_mesh(self):

        if self.isperiodic:
            self.dx = self.dom_len / self.nn
            xL = self.xmin
            xR = self.xmax - self.dx
        else:
            self.dx = self.dom_len / (self.nn + 1)
            xL = self.xmin + self.dx
            xR = self.xmax - self.dx

        self.xy = np.linspace(xL, xR, self.nn)
        self.xy_elem = self.xy.reshape((self.nn,1))
        self.nen = self.nn

    def build_elem_mesh(self):

        ''' Extract required info '''

        self.dx = None
        self.nen, self.dim = self.xy_op.shape    # No. of nodes per elem and no. of dim
        self.nn = self.nen * self.nelem     # Total no. of nodes

        ''' Create mesh '''

        vert_op = RefSimplexElem.ref_vertices(self.dim)

        # Get the locations of the vertices
        self.nvert = self.nelem+1 # No. of vertices in the mesh
        self.xy_vert = np.linspace(self.xmin, self.xmax, self.nvert)
        self.xy_vert = self.xy_vert[:, None] # Convert from 1 to 2D array

        # Get the location of all the nodes in the element
        self.xy = np.zeros((self.nn, self.dim)) # Nodal locations in 2D array
        self.xy_elem = np.zeros((self.nen, self.nelem, self.dim)) # Each slice is for one elem

        for i in range(self.nelem):
            vert_new = self.xy_vert[i:(i+2),:]

            xy_new, _ = BasisFun.map2new_elem(self.xy_op, vert_op, vert_new)

            a = self.nen * i
            b = a + self.nen
            self.xy[a:b, :] = xy_new

            self.xy_elem[:,i,:] = xy_new

        # Make the mapping from the global edge no. to the global elem no.
        if self.isperiodic:
            self.nfacet = self.nelem
            self.gf2ge = np.zeros((2, self.nfacet), dtype=int)
            self.gf2ge[0,:] = np.arange(-1,self.nelem-1, dtype=int)
            self.gf2ge[0,0] = self.nelem -1
            self.gf2ge[1,:] = np.arange(0,self.nelem, dtype=int)
        else:
            self.nfacet = self.nelem + 1
            self.gf2ge = np.zeros((2, self.nfacet), dtype=int)
            self.gf2ge[0,:] = np.arange(-1, self.nelem, dtype=int)
            self.gf2ge[1,:] = np.arange(0, self.nelem+1, dtype=int)
            self.gf2ge[1,-1] = -1

    @staticmethod
    def mesh_jac(xquad, shpx):
        '''
        Parameters
        ----------
        xquad : np float array
            Coordinates of the nodes for one element.
        shpx : np float array
            Derivative operator.

        Returns
        -------
        jac : np float array
            The Jacobian of the transformation for the mesh.
        det_jac : float
            Determinant of the Jacobian.
        ijac : np float array
            Inverse of the Jacobian.
        '''

        nn, dim = xquad.shape

        jac = np.zeros((nn, dim, dim))
        ijac = np.zeros((nn, dim, dim))

        for d in range(dim):
            jac[:,:,d] = shpx[d,:,:] @ xquad

        if dim == 1:
            det_jac = np.diag(jac[:,0,0])
            ijac = 1 / jac
        else:
            raise Exception('mesh_jac for d>1 is not available')

        return jac, det_jac, ijac



''' Test out the code '''

# nelem = 5
# xy_op = np.array([0, 0.5, 1])
# xy_op = xy_op[:, None]
# isperiodic = True

# c = MakeMesh(nelem, xy_op, isperiodic)


