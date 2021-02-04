#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:11:13 2020

@author: andremarchildon
"""

import numpy as np
import scipy.sparse as sp


class FiniteDiff:

    def __init__(self, der_no, order, n, dx, periodic=False):
        '''
        Parameters
        ----------
        der_no : int
            Indicate if this is for a first, second etc derivative
        order : int
            Order of accuracy for the derivative.
        n : int
            Number of nodes in the mesh.
        dx : float
            Size of the time step.
        periodic : bool, optional
            Indicate if the derivative operator is for a periodic domain.
            The default is False.
        '''

        self.der_no = der_no
        self.order = order
        self.n = n
        self.dx
        self.periodic = periodic

        if periodic:
            raise Exception('Can not do periodic operators yet')
        if self.der_no == 1:
            self.der = FiniteDiff.der1(self.order, self.n, self.dx, self.periodic)
        elif self.der_no == 2:
            self.der = FiniteDiff.der2(self.order, self.n, self.dx, self.periodic)
        elif self.der_no == 4:
            self.der = FiniteDiff.der4(self.order, self.n, self.dx, self.periodic)
        else:
            raise Exception('Requested derivative is not available')

    @staticmethod
    def der1(order, n, dx, periodic=False):

        bcL = np.zeros(n)
        bcR = np.zeros(n)

        if order == 2:
            t1 = np.ones(n-1) / (2*dx)
            der1 = sp.diags((-t1, t1), [-1, 1])

            if periodic:
                data = np.array([-1, 1]) / (2*dx)
                row = [0, n-1]
                col = [n-1, 0]
                bc_per = sp.csc_matrix((data, (row, col)), shape=(n, n))
                der1 += bc_per

                # return der1, None, None
            else:
                # bcL = np.zeros(n)
                bcL[0] = -1/(2*dx)

                # bcR = np.zeros(n)
                bcR[-1] = 1/(2*dx)

                # return der1, bcL, bcR

        elif order == 4:
            t1 = (2/3)*np.ones(n-1) / dx
            t2 = (-1/12)*np.ones(n-2) / dx
            der1 = sp.diags((-t2, -t1, t1, t2), [-2, -1, 1, 2])

            if periodic:
                data = np.array([1/12, -2/3, 1/12, -1/12, 2/3, -1/12]) / dx
                row = [0, 0, 1, n-2, n-1, n-1]
                col = [n-2, n-1, n-1, 0, 0, 1]
                bc_per = sp.csc_matrix((data, (row, col)), shape=(n, n))
                der1 += bc_per

                return der1, None, None
            else:
                raise Exception('Need to contruct different stencil for the boundary')

                # return der1, bcL, bcR
        else:
            raise Exception('Requested order of derivative is not available')

        return der1, bcL, bcR

    @staticmethod
    def der2(order, n, dx, periodic=False):

        bcL = np.zeros(n)
        bcR = np.zeros(n)

        if order == 2:
            t0 = -2*np.ones(n) / dx**2
            t1 = np.ones(n-1) / dx**2
            der2 = sp.diags((t1, t0, t1), [-1, 0, 1])

            if periodic:
                data = np.array([1, 1]) / dx**2
                row = [0, n-1]
                col = [n-1, 0]
                bc_per = sp.csc_matrix((data, (row, col)), shape=(n, n))
                der2 += bc_per

                # return der2, None, None
            else:
                # bcL = np.zeros(n)
                bcL[0] = 1 / dx**2

                # bcR = np.zeros(n)
                bcR[-1] =  1 / dx**2

                # return der2, bcL, bcR

        else:
            raise Exception('Requested order of derivative is not available')

        return der2, bcL, bcR

    @staticmethod
    def der4(order, n, dx, periodic=False, use_b_der1=False):

        assert not (periodic and use_b_der1), 'Cannot have both periodic and use_b_der1 be true'

        bcL = np.zeros(n)
        bcR = np.zeros(n)

        if order == 2:
            t0 = 6*np.ones(n) / dx**4
            t1 = -4*np.ones(n-1) / dx**4
            t2 = np.ones(n-2) / dx**4

            if use_b_der1:
                t0[0] = 7 / dx**4
                t0[-1] = 7 / dx**4
                der4 = sp.diags((t2, t1, t0, t1, t2), [-2, -1, 0, 1, 2])

                # bcL = np.zeros(n)
                bcL[:2] = np.array([-4, 1]) / dx**4

                # bcR = np.zeros(n)
                bcR[-2:] = np.array([1, -4]) / dx**4

                bc_derL = np.zeros(n)
                bc_derL[0] = -2 / dx**3

                bc_derR = np.zeros(n)
                bc_derR[-1] = 2 / dx**3

                return der4, bcL, bcR, bc_derL, bc_derR

            else:
                der4 = sp.diags((t2, t1, t0, t1, t2), [-2, -1, 0, 1, 2])

                if periodic:
                    data = np.array([1, -4, 1, 1, 1, -4]) / dx**4
                    row = [0, 0, 1, n-2, n-1, n-1]
                    col = [n-2, n-1, n-1, 0, 0, 1]
                    bc_per = sp.csc_matrix((data, (row, col)), shape=(n, n))
                    der4 += bc_per

                    return der4, bcL, bcR, None, None
                else:
                    raise Exception('Need to contruct different stencil for the boundary')

        else:
            raise Exception('Requested order of derivative is not available')

