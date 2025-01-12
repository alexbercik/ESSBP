#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:31:11 2020

@author: bercik
"""

import sympy as sp
import numpy as np

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

    def round_to_nearest_simple_decimal(expr, threshold=1e-9):
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
        dx = round_to_nearest_simple_decimal(dx)

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


    