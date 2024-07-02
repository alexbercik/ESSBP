#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from sys import path, stdout

n_nested_folder = 1
folder_path = os.path.abspath('')

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

import numpy as np
from Source.Disc.MakeSbpOp import MakeSbpOp
from Source.Disc.MakeMesh import MakeMesh
import matplotlib.pyplot as plt
import contextlib
import scipy as sc

p=2
nn=20 # = 9 for p=2
ref = MakeSbpOp(p=p,sbp_type='csbp',nn=nn)
Href, Dref = np.diag(ref.H), ref.D
refmesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=0,warp_type='default')
x = refmesh.x
function = 'tanh' # 'sigmoid', 'corners'
#refrefmesh.plot()

def test_f1(x):
    return np.sin(np.pi*x)
def test_df1(x):
    return np.pi*np.cos(np.pi*x)
def test_f2(x):
    return np.cos(np.pi*x)
def test_df2(x):
    return -np.pi*np.sin(np.pi*x)
def test_f3(x):
    return np.sin(2*np.pi*x + 0.1)
def test_df3(x):
    return 2*np.pi*np.cos(2*np.pi*x + 0.1)
def test_f4(x):
    return np.cos(2*np.pi*x + 0.1)
def test_df4(x):
    return -2*np.pi*np.sin(2*np.pi*x + 0.1)
def test_f5(x):
    #return x
    return np.sin(0.5*np.pi*x + 0.2)
def test_df5(x):
    #return np.ones(x.shape)
    return 0.5*np.pi*np.cos(0.5*np.pi*x + 0.2)
def test_f6(x):
    #return x**p
    return np.cos(0.5*np.pi*x + 0.2)
def test_df6(x):
    #return p*x**(p-1)
    return -0.5*np.pi*np.sin(0.5*np.pi*x + 0.2)


# double check the above are correct
ter1 = np.max(abs(Dref @ test_f1(x) - test_df1(x)))
ter2 = np.max(abs(Dref @ test_f2(x) - test_df2(x)))
ter3 = np.max(abs(Dref @ test_f3(x) - test_df3(x)))
ter4 = np.max(abs(Dref @ test_f4(x) - test_df4(x)))
ter5 = np.max(abs(Dref @ test_f5(x) - test_df5(x)))
ter6 = np.max(abs(Dref @ test_f6(x) - test_df6(x)))
print('sanity check function derivatives:',ter1,ter2,ter3,ter4,ter5,ter6)

assert(function in ['sigmoid','corners','tanh']),'dude.'

def get_ops(warp_factors):
    if function=='corners':
        warp_factor1,warp_factor2,warp_factor3 = warp_factors[0],warp_factors[1],warp_factors[2]
    with contextlib.redirect_stdout(None):
        if function=='sigmoid':
            mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=warp_factors,warp_type='sigmoid')
        elif function=='tanh':
            mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=warp_factors,warp_type='tanh')
        else:
            mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=warp_factor1,
                            warp_factor2=warp_factor2, warp_factor3=warp_factor3, warp_type='corners')
        mesh.get_jac_metrics(ref, periodic=False,
                                metric_method = 'exact', 
                                bdy_metric_method = 'exact',
                                jac_method='exact',
                                use_optz_metrics = 'False',
                                calc_exact_metrics = False)
        H, D, _ = ref.ref_2_phys(mesh, 'skew_sym')
        H, D = H[:,0], D[:,:,0]
        x = mesh.x

        ter1 = D @ test_f1(x) - test_df1(x)
        ter2 = D @ test_f2(x) - test_df2(x)
        ter3 = D @ test_f3(x) - test_df3(x)
        ter4 = D @ test_f4(x) - test_df4(x)
        ter5 = D @ test_f5(x) - test_df5(x)
        ter6 = D @ test_f6(x) - test_df6(x)

        
        er1 = ter1 @ (H * ter1)
        er2 = ter2 @ (H * ter2)
        er3 = ter3 @ (H * ter3)
        er4 = ter4 @ (H * ter4)
        er5 = ter5 @ (H * ter5)
        er6 = ter6 @ (H * ter6)
        '''
        er1 = np.linalg.norm(ter1)
        er2 = np.linalg.norm(ter2)
        er3 = np.linalg.norm(ter3)
        er4 = np.linalg.norm(ter4)
        er5 = np.linalg.norm(ter5)
        er6 = np.linalg.norm(ter6)
        '''

        er = er1 + er2 + er3 + er4 + er5 + er6
        return er

if function=='sigmoid':
    res = sc.optimize.minimize_scalar(get_ops, bounds=(-1, 100), method='bounded')
    print('optimum warp_factor is', res.x)
    print('the error is ', res.fun)
    print('wheras the reference error is ', get_ops(0))


elif function=='tanh':
    eps = 1e-10
    #res = sc.optimize.minimize_scalar(get_ops, bounds=(eps, 1.0 - eps), method='bounded')
    res = sc.optimize.basinhopping(get_ops, 0.1, stepsize=0.05, minimizer_kwargs={'bounds':[(eps, 1.0 - eps)]})
    print('optimum warp_factor is', res.x)
    print('the error is ', res.fun)
    print('wheras the reference error is ', get_ops(0))
    print('improvement of ', get_ops(0) - res.fun)

else:
    def ineq_f(x):
        with contextlib.redirect_stderr(None):
            fac = x[2]**(1-x[0])
            if fac == np.inf:
                print('WARNING: exponent (1-warpfactor1) too small, capping manually.')
                fac = 1E16
            ineq = np.array([x[0]-1,
                            x[1],
                            fac/(x[0]*(1-2*x[2])+2*x[2]) - x[1],
                            x[2],
                            0.5-x[2]])
        return ineq
        
    def ineq_df(x):
        with contextlib.redirect_stderr(None):
            fac = x[2]**(1-x[0])
        if fac == np.inf or x[2]<0:
            print('WARNING: exponent (1-warpfactor1) too small, setting derivatives manually.')
            db_dx0 = 1E16
            db_dx2 = 1E16
        else:
            db_dx0 = (-fac*np.log(x[2])*(x[0]*(1-2*x[2])+2*x[2]) - fac*(1-2*x[2]))/((x[0]*(1-2*x[2])+2*x[2])**2)
            db_dx2 = ((1-x[0])*x[0]*(x[2]**(-x[0]) - 2*(fac)))/((x[0]*(1-2*x[2])+2*x[2])**2)
        ineq_der = np.array([[1,0,0],
                            [0,1,0],
                            [db_dx0, -1, db_dx2],
                            [0,0,1],
                            [0,0,-1]])
        return ineq_der

    ineq_cons =  {'type': 'ineq', 'fun' : ineq_f, 'jac' : ineq_df}

    #x0 = np.array([4,30,0.2])
    x0s = [np.array([2,5,0.1]),
        np.array([2,2,0.2]),
        np.array([2,1.5,0.35]),
        np.array([3,100,0.05]),
        np.array([3,30,0.1]),
        np.array([3,8,0.2]),
        np.array([3,3,0.35]),
        np.array([4,200,0.1]),
        np.array([4,30,0.2]),
        np.array([4,8,0.35]),
        np.array([5,1500,0.1]),
        np.array([5,100,0.2]),
        np.array([5,20,0.35]),
        np.array([6,15000,0.1]),
        np.array([6,500,0.2]),
        np.array([6,50,0.35])]
    init = np.array([1,0,0.2])
    best_min_x = x0s[0]
    best_min_f = get_ops(best_min_x)
    for x0 in x0s:
        res = sc.optimize.minimize(get_ops, x0, method='SLSQP', constraints=[ineq_cons])
        if res.fun < best_min_f:
            print('Multi-start: found a new minimum {0} @ x={1}'.format(res.fun,res.x))
            best_min_x, best_min_f = res.x, res.fun
        else:
            print('Multi-start: local min of {0} @ x={1}'.format(res.fun,res.x))

    np.set_printoptions(precision=16)
    print('optimum warp_factors are', best_min_x)
    print('the error is ', best_min_f)
    print('wheras the reference error is ', get_ops(init))


with contextlib.redirect_stdout(None):
    if function=='sigmoid':
        mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=res.x,warp_type='sigmoid')
    elif function=='tanh':
        mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=res.x,warp_type='tanh')
    else:
        mesh = MakeMesh(dim=1,xmin=0,xmax=1,nelem=1,x_op=ref.x,warp_factor=best_min_x[0],
                        warp_factor2=best_min_x[1], warp_factor3=best_min_x[2], warp_type='corners')
    mesh.get_jac_metrics(ref, periodic=False,
                            metric_method = 'exact', 
                            bdy_metric_method = 'exact',
                            jac_method='exact',
                            use_optz_metrics = 'False',
                            calc_exact_metrics = False)
    H, D, _ = ref.ref_2_phys(mesh, 'skew_sym')
    H, D = H[:,0], D[:,:,0]

#test = MakeSbpOp(p=p,sbp_type='optz',nn=nn)
#print('double check I coded it in ESSBP correctly:', np.max(abs(D-test.D)))