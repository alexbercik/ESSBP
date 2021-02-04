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