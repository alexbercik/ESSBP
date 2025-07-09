#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:31:11 2020

@author: bercik
"""

import sympy as sp
import numpy as np
from Source.Methods.Analysis import printProgressBar

def profiler(command, filename="profile.stats", n_stats=50, verbose=False,
             sortby='cumulative'):
    """Profiler for a python program

    Runs cProfile and outputs ordered statistics that describe
    how often and for how long various parts of the program are executed.

    Stats can be visualized with `!snakeviz profile.stats`.

    Parameters
    ----------
    command: str
        Command string to be executed.
    filename: str
        Name under which to store the stats.
    n_stats: int or None
        Number of top stats to show.
    verbose: bool
        Whether to print full file names
    sortby: str
        How to sort the data
    """
    import cProfile, pstats
    from numba import config
    config.PROFILING_MODE = True

    cProfile.run(command, filename)
    
    if verbose:
        # use this line to strip only the file name (ignore path)
        stats = pstats.Stats(filename).sort_stats(sortby)
    else:
        # use this line to include the full directory path
        stats = pstats.Stats(filename).strip_dirs().sort_stats(sortby)
    return stats.print_stats(n_stats or {})

class TaylorSeries1D:
    """Class for symbolic Taylor series."""
    def __init__(self, f, num_terms=4):
        self.f = f
        self.N = num_terms
        # Introduce symbols for the derivatives
        self.df = [f]
        self.h = sp.Symbol(r'h_\xi')
        for i in range(1, self.N+1):
            self.df.append(sp.Symbol(f.name + '_x'*i))

    def __call__(self,h):
        """Return the truncated Taylor series at x+h."""
        terms = self.f
        for i in range(1, self.N+1):
            terms += sp.Rational(1, sp.factorial(i))*self.df[i]*h**i
        return terms
    
def round_to_nearest_simple_decimal(expr, decimals=10, threshold=1e-9):
        """Round numbers in expr close to simple decimals to those simple decimals."""
        rounded_expr = expr
        for number in expr.atoms(sp.Number):
            num_float = float(number)
            nearest_integer = round(num_float)
            nearest_decimal = round(num_float, 3)
            printing_decimal = round(num_float, decimals)
            # Check if the number is close to an integer
            if abs(num_float - nearest_integer) < threshold:
                rounded_expr = rounded_expr.subs(number, nearest_integer)
            # Check if the number is close to a simple decimal up to 3 places
            elif abs(num_float - nearest_decimal) < threshold:
                rounded_expr = rounded_expr.subs(number, nearest_decimal)
            else:
                rounded_expr = rounded_expr.subs(number, printing_decimal)
        return rounded_expr
    
def Check_FD_Stencil_Expansion(stencil,locations,num_terms=4,lim_terms=2,
                               notebook=True,decimals=10,xdenom=0):
    ''' do a taylor expansion of a stencil assuming equispaced nodes 
    stencil: list of values u_j of the stencil, e.g. [-0.5,0,0.5]
    locations: list of locations j of the stencil, e.g. [-1,0,1]'''
    if notebook:
        from IPython.display import display, Math
        from sympy.printing.latex import latex
    u = sp.Symbol('u')
    h = sp.Symbol(r'(\Delta x)')
    if xdenom==0:
        denom = 1
    else:
        denom = h**xdenom
    u_Taylor = TaylorSeries1D(u,num_terms)

    dx = 0
    for j,a in enumerate(stencil):
        if locations[j]==0:
            dx += a*u / denom
        else:
            taylor = u_Taylor(locations[j]*h)
            dx += a*taylor / denom

    dx = sp.expand(sp.simplify(dx))
    dx = round_to_nearest_simple_decimal(dx,decimals)

    if lim_terms < len(dx.as_ordered_terms()):
        res = sum(dx.as_ordered_terms()[-lim_terms:])
    else:
        res = dx

    if notebook:
        out = f'{latex(res)}'
        display(Math(out))
    else:
        print(res)
    
    
    
def Check_Taylor_Series_1D(Mat,x,num_terms=4,lim_terms=2,notebook=True,decimals=10,delta_x=None):
    ''' Check the taylor expansion of the rows of a 1D matrix given x'''
    if notebook:
        from IPython.display import display, Math
        from sympy.printing.latex import latex

    u = sp.Symbol('u')
    h = sp.Symbol(r'(\Delta x)')
    u_Taylor = TaylorSeries1D(u,num_terms)
    Dxs = []
    if delta_x is None:
        h_avg = np.mean(x[1:] - x[:-1])
    else:
        h_avg = delta_x

    for i in range(len(x)):
        dx = 0
        for j in range(len(x)):
            if j==i:
                dx += Mat[i,j]*u * (h_avg/h)
            else:
                hx = (x[j]-x[i]) * (h/h_avg)
                taylor = u_Taylor(hx)
                dx += Mat[i,j]*taylor * (h_avg/h) 
    
        dx = sp.expand(sp.simplify(dx))
        #threshold = 1e-12
        #small_numbers = set([e for e in dx.atoms(sp.Number) if abs(e) < threshold])
        #numbers_near_one = set([e for e in dx.atoms(sp.Number) if abs(e - 1) < threshold])
        #d = {s: 0 for s in small_numbers}
        #d.update({s: 1 for s in numbers_near_one})
        #Dxs.append(dx.subs(d))
        dx = round_to_nearest_simple_decimal(dx,decimals)

        if lim_terms < len(dx.as_ordered_terms()):
            limited_terms = sum(dx.as_ordered_terms()[-lim_terms:])
            Dxs.append(limited_terms)
        else:
            Dxs.append(dx)


    if notebook:
        out = f"\\text{{Using a value of }} \\Delta x = {latex(h_avg)}"
        display(Math(out))
    else:
        print(f"Using a value of \\Delta x = {h_avg}") 
    for i in range(len(x)):
        if notebook:
            out = f'\\text{{Mat[{i}] = }} {latex(Dxs[i])}'
            display(Math(out))
        else:

            print(r'Mat[{0}] = {1}'.format(i,Dxs[i]))


def calcJacobian_complex_step(f, q, h=1.0e-15):
    """Complex step differentiation for a function f at point q."""
    if q.ndim == 1:
        nen = len(q)
        nelem = 1
    else:
        nen,nelem = q.shape
    nn = nelem*nen 
    A = np.zeros((nn,nn)) 
    for i in range(nen):
        if nn>=400:
            printProgressBar(i, nen-1, prefix = 'Complex Step Progress:')
        for j in range(nelem):
            ei = np.zeros((nen,nelem),dtype=np.complex128)
            ei[i,j] = h*1j
            if q.ndim == 1:
                qi = f(np.complex128(q)+ei)
            else:
                qi = f(np.complex128(q)+ei).flatten('F')
            idx = np.where(np.imag(ei.flatten('F'))>h/10)[0][0]
            A[:,idx] = np.imag(qi)/h
    return A

def calcJacobian_finite_diff(f, q, h=1.0e-4):
    """Finite difference differentiation for a function f at point q."""
    nen,nelem = q.shape
    nn = nelem*nen 
    A = np.zeros((nn,nn)) 
    for i in range(nen):
        if nn>=400:
            printProgressBar(i, nen-1, prefix = 'Complex Step Progress:')
        for j in range(nelem):
            ei = np.zeros((nen,nelem))
            ei[i,j] = 1.*h
            q_r = f(q+ei).flatten('F')
            q_l = f(q-ei).flatten('F')
            idx = np.where(ei.flatten('F')>h/10)[0][0]
            A[:,idx] = (q_r - q_l)/(2*h)
    return A

def compare_Jacobian(f, q, h=1.0e-4, hi=1.0e-15, returnA=False, returnbool=False):
    """Compare the Jacobian calculated by complex step and finite difference."""
    A_complex = calcJacobian_complex_step(f, q, hi)
    A_finite = calcJacobian_finite_diff(f, q, h)
    abs_diff = np.abs(A_complex - A_finite)
    rel_diff = abs_diff / (np.maximum(np.abs(A_complex), np.abs(A_finite)) + h)
    maxdiff_abs = np.max(abs_diff)
    maxdiff_rel = np.max(rel_diff)

    reltol = 0.01
    abstol = 10*h
    ok = (maxdiff_abs < abstol) and (maxdiff_rel < reltol)
    
    if ok:
        print("The Jacobians are equal within the tolerance.")
    
    else:
        print("The Jacobians differ.")
        print("Maximum absolute difference:", maxdiff_abs)
        i, j = np.where(rel_diff == maxdiff_rel)
        if i is not [] and j is not []:
            print("Maximum relative difference:", maxdiff_rel, f"({A_complex[i[0], j[0]]:.2e} and {A_finite[i[0], j[0]]:.2e})")
    
    if returnA:
        if returnbool:
            return A_complex, A_finite, ok
        else:
            return A_complex, A_finite
    elif returnbool:
        return ok
    