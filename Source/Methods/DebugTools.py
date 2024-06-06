#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:31:11 2020

@author: bercik
"""

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

    cProfile.run(command, filename)
    
    if verbose:
        # use this line to strip only the file name (ignore path)
        stats = pstats.Stats(filename).sort_stats(sortby)
    else:
        # use this line to include the full directory path
        stats = pstats.Stats(filename).strip_dirs().sort_stats(sortby)
    return stats.print_stats(n_stats or {})

import sympy as sp
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
    
def Check_Taylor_Series_1D(Mat,x,num_terms=4,notebook=True):
    ''' Check the taylor expansion of the rows of a 1D matrix given x'''
    if notebook:
        from IPython.display import display, Math
        from sympy.printing.latex import latex

    u = sp.Symbol('u')
    h = sp.Symbol(r'(\Delta x)')
    u_Taylor = TaylorSeries1D(u,num_terms)
    Dxs = []

    for i in range(len(x)):
        dx = 0
        for j in range(len(x)):
            if j==i:
                dx += Mat[i,j]*u
            else:
                hx = (x[j]-x[i])*h
                taylor = u_Taylor(hx)
                dx += Mat[i,j]*taylor
    
        threshold = 1e-12
        dx = sp.simplify(dx)
        small_numbers = set([e for e in dx.atoms(sp.Number) if abs(e) < threshold])
        d = {s: 0 for s in small_numbers}
        Dxs.append(dx.subs(d))

    for i in range(len(x)):
        if notebook:
            out = f'\\text{{Mat[{i}] = }} {latex(Dxs[i])}'
            display(Math(out))
        else:
            print(r'Mat[{0}] = {1}'.format(i,Dxs[i]))


    