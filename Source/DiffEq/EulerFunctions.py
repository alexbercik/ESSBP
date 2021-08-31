#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:07:16 2021

@author: bercik
"""
from numba import njit
import numpy as np
import Source.Methods.Functions as fn

''' A collection of numerical 2-point fluxes for the Inviscid fluxes of the 
    Euler and Navier-Stokes equations. All jitted for speed '''

@njit    
def dEdq_1D(q):
    ''' the flux jacobian A in 1D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    rho = q[::3,:]
    u = q[1::3,:]/rho
    k = u**2/2
    g_e_rho = g * q[2::3,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = (g-3) * k
    r22 = (3-g) * u
    r23 = r1*(g-1)
    r31 = (g-1) * (u**3) - u * g_e_rho
    r32 = g_e_rho - 3*(g-1) * k
    r33 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r21,r22,r23,r31,r32,r33)
    return dEdq

@njit    
def dEdq_1D_complex(q):
    ''' the flux jacobian A in 1D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    rho = q[::3,:]
    u = q[1::3,:]/rho
    k = u**2/2
    g_e_rho = g * q[2::3,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = (g-3) * k
    r22 = (3-g) * u
    r23 = r1*(g-1)
    r31 = (g-1) * (u**3) - u * g_e_rho
    r32 = g_e_rho - 3*(g-1) * k
    r33 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r21,r22,r23,r31,r32,r33)
    return dEdq

@njit    
def dExdq_2D(q):
    ''' the x flux jacobian A in 2D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    u2 = u**2
    k = (u2+v**2)/2
    g_e_rho = g * q[3::4,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = g1 * k - u2
    r22 = (3-g) * u
    r23 = -g1 * v
    r24 = r1 * g1
    r31 = -u * v
    r41 = (2 * g1) * k * u - u * g_e_rho
    r42 = g_e_rho - g1*(k + u2)
    r43 = -g1 * u * v
    r44 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r0,r21,r22,r23,r24,r31,v,u,r0,r41,r42,r43,r44)
    return dEdq

@njit    
def dEydq_2D(q):
    ''' the y flux jacobian A in 2D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    v2 = v**2
    k = (u**2+v2)/2
    g_e_rho = g * q[3::4,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = -u * v  
    r31 = g1 * k - v2
    r32 = -g1 * u
    r33 = (3-g) * v
    r34 = r1 * g1
    r41 = (2 * g1) * k * v - v * g_e_rho
    r42 = -g1 * u * v
    r43 = g_e_rho - g1*(k + v2)
    r44 = g * v
    
    dEdq = fn.block_diag(r0,r0,r1,r0,r21,v,u,r0,r31,r32,r33,r34,r41,r42,r43,r44)
    return dEdq

@njit    
def dExdq_2D_complex(q):
    ''' the x flux jacobian A in 2D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    u2 = u**2
    k = (u2+v**2)/2
    g_e_rho = g * q[3::4,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = g1 * k - u2
    r22 = (3-g) * u
    r23 = -g1 * v
    r24 = r1 * g1
    r31 = -u * v
    r32 = v
    r33 = u
    r41 = (2 * g1) * k * u - u * g_e_rho
    r42 = g_e_rho - g1*(k + u2)
    r43 = -g1 * u * v
    r44 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r0,r21,r22,r23,r24,r31,r32,r33,r0,r41,r42,r43,r44)
    return dEdq

@njit    
def dEydq_2D_complex(q):
    ''' the y flux jacobian A in 2D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    v2 = v**2
    k = (u**2+v2)/2
    g_e_rho = g * q[3::4,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = -u * v
    r22 = v
    r23 = u   
    r31 = g1 * k - v2
    r32 = -g1 * u
    r33 = (3-g) * v
    r34 = r1 * g1
    r41 = (2 * g1) * k * v - v * g_e_rho
    r42 = -g1 * u * v
    r43 = g_e_rho - g1*(k + v2)
    r44 = g * v
    
    dEdq = fn.block_diag(r0,r0,r1,r0,r21,r22,r23,r0,r31,r32,r33,r34,r41,r42,r43,r44)
    return dEdq

@njit    
def dExdq_3D(q):
    ''' the x flux jacobian A in 3D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    u2 = u**2
    k = (u2+v**2+w**2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = g1 * k - u2
    r22 = (3-g) * u
    r23 = -g1 * v
    r24 = -g1 * w
    r25 = r1 * g1
    r31 = -u * v
    r41 = -u * w
    r51 = (2 * g1) * k * u - u * g_e_rho
    r52 = g_e_rho - g1*(k + u2)
    r53 = -g1 * u * v
    r54 = -g1 * u * w
    r55 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r0,r0,r21,r22,r23,r24,r25,r31,v,u,r0,r0,r41,w,r0,u,r0,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEydq_3D(q):
    ''' the y flux jacobian A in 3D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    v2 = v**2
    k = (u**2+v2+w**2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = -u * v  
    r31 = g1 * k - v2
    r32 = -g1 * u
    r33 = (3-g) * v
    r34 = -g1 * w
    r35 = r1 * g1
    r41 = - w * v
    r51 = (2 * g1) * k * v - v * g_e_rho
    r52 = -g1 * u * v
    r53 = g_e_rho - g1*(k + v2)
    r54 = -g1 * v * w
    r55 = g * v
    
    dEdq = fn.block_diag(r0,r0,r1,r0,r0,r21,v,u,r0,r0,r31,r32,r33,r34,r35,r41,r0,w,v,r0,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEzdq_3D(q):
    ''' the z flux jacobian A in 3D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    w2 = w**2
    k = (u**2+v**2+w2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = -u * w  
    r31 = -v * w
    r41 = g1 * k - w2
    r42 = -g1 * u
    r43 = -g1 * v
    r44 = (3-g) * w
    r45 = r1 * g1
    r51 = (2 * g1) * k * w - w * g_e_rho
    r52 = -g1 * u * w
    r53 = -g1 * v * w
    r54 = g_e_rho - g1*(k + w2)
    r55 = g * w
    
    dEdq = fn.block_diag(r0,r0,r0,r1,r0,r21,w,r0,u,r0,r31,r0,w,v,r0,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dExdq_3D_complex(q):
    ''' the x flux jacobian A in 3D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    u2 = u**2
    k = (u2+v**2+w**2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = g1 * k - u2
    r22 = (3-g) * u
    r23 = -g1 * v
    r24 = -g1 * w
    r25 = r1 * g1
    r31 = -u * v
    r41 = -u * w
    r51 = (2 * g1) * k * u - u * g_e_rho
    r52 = g_e_rho - g1*(k + u2)
    r53 = -g1 * u * v
    r54 = -g1 * u * w
    r55 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r0,r0,r21,r22,r23,r24,r25,r31,v,u,r0,r0,r41,w,r0,u,r0,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEydq_3D_complex(q):
    ''' the y flux jacobian A in 3D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    v2 = v**2
    k = (u**2+v2+w**2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = -u * v  
    r31 = g1 * k - v2
    r32 = -g1 * u
    r33 = (3-g) * v
    r34 = -g1 * w
    r35 = r1 * g1
    r41 = - w * v
    r51 = (2 * g1) * k * v - v * g_e_rho
    r52 = -g1 * u * v
    r53 = g_e_rho - g1*(k + v2)
    r54 = -g1 * v * w
    r55 = g * v
    
    dEdq = fn.block_diag(r0,r0,r1,r0,r0,r21,v,u,r0,r0,r31,r32,r33,r34,r35,r41,r0,w,v,r0,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEzdq_3D_complex(q):
    ''' the z flux jacobian A in 3D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    w2 = w**2
    k = (u**2+v**2+w2)/2
    g_e_rho = g * q[4::5,:]/rho

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = -u * w  
    r31 = -v * w
    r41 = g1 * k - w2
    r42 = -g1 * u
    r43 = -g1 * v
    r44 = (3-g) * w
    r45 = r1 * g1
    r51 = (2 * g1) * k * w - w * g_e_rho
    r52 = -g1 * u * w
    r53 = -g1 * v * w
    r54 = g_e_rho - g1*(k + w2)
    r55 = g * w
    
    dEdq = fn.block_diag(r0,r0,r0,r1,r0,r21,w,r0,u,r0,r31,r0,w,v,r0,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEdq_eigs_1D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::3,:]
    u = q[1::3,:]/rho
    k = u**2/2
    e = q[2::3,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::3,:] = u
        Lam[1::3,:] = u + a
        Lam[2::3,:] = u - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r21 = u*a
        r22 = u*b + c
        r23 = u*b - c
        r31 = k*a
        r32 = k*b + u*c + p/((2*g1)*b)
        r33 = k*b - u*c + p/((2*g1)*b)     
        Y = fn.block_diag(a,b,b,r21,r22,r23,r31,r32,r33)
        if trans:
            YT = fn.block_diag(a,r21,r31,b,r22,r32,b,r23,r33)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r11 = 1/a - a*k/p
        r12 = a*u/p
        r13 = -a/p
        r21 = k*b - u*c
        r22 = c - u*b
        r31 = k*b + u*c
        r32 = -c - u*b
        Yinv = fn.block_diag(r11,r12,r13,r21,r22,b,r31,r32,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT
        
@njit    
def dExdq_eigs_2D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    k = (u**2+v**2)/2
    e = q[3::4,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::4,:] = u
        Lam[1::4,:] = u
        Lam[2::4,:] = u + a
        Lam[3::4,:] = u - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = u*a
        r23 = u*b + c
        r24 = u*b - c
        r31 = v*a
        r32 = -np.sqrt(2)*c
        r33 = v*b
        r41 = k*a
        r42 = v*r32
        r43 = k*b + u*c + p/((2*g1)*b)
        r44 = k*b - u*c + p/((2*g1)*b)     
        Y = fn.block_diag(a,r0,b,b,r21,r0,r23,r24,r31,r32,r33,r33,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(a,r21,r31,r41,r0,r0,r32,r42,b,r23,r33,r43,b,r24,r33,r44)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r11 = 1/a - a*k/p
        r12 = a*u/p
        r13 = a*v/p
        r14 = -a/p
        r21 = np.sqrt(2)*c*v
        r0 = np.zeros(np.shape(rho))
        r23 = - np.sqrt(2)*c
        r31 = k*b - u*c
        r32 = c - u*b
        r33 = -v*b
        r41 = k*b + u*c
        r42 = -c - u*b
        Yinv = fn.block_diag(r11,r12,r13,r14,r21,r0,r23,r0,r31,r32,r33,b,r41,r42,r33,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT     
        
@njit    
def dEydq_eigs_2D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    k = (u**2+v**2)/2
    e = q[3::4,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::4,:] = v
        Lam[1::4,:] = v
        Lam[2::4,:] = v + a
        Lam[3::4,:] = v - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = np.sqrt(2)*c
        r22 = u*a
        r23 = u*b
        r32 = v*a
        r33 = v*b + c
        r34 = v*b - c
        r41 = u*r21
        r42 = k*a
        r43 = k*b + v*c + p/((2*g1)*b)
        r44 = k*b - v*c + p/((2*g1)*b)     
        Y = fn.block_diag(r0,a,b,b,r21,r22,r23,r23,r0,r32,r33,r34,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(r0,r21,r0,r41,a,r22,r32,r42,b,r23,r33,r43,b,r23,r34,r44)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r12 = np.sqrt(2)*c
        r11 = -u*r12
        r0 = np.zeros(np.shape(rho))
        r21 = 1/a - a*k/p
        r22 = a*u/p
        r23 = a*v/p
        r24 = -a/p
        r31 = k*b - v*c
        r32 = -u*b
        r33 = c - v*b
        r41 = k*b + v*c
        r43 = -c - v*b
        Yinv = fn.block_diag(r11,r12,r0,r0,r21,r22,r23,r24,r31,r32,r33,b,r41,r32,r43,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT  

@njit    
def dExdq_eigs_3D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    k = (u**2+v**2+w**2)/2
    e = q[4::5,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::5,:] = u
        Lam[1::5,:] = u
        Lam[2::5,:] = u
        Lam[3::5,:] = u + a
        Lam[4::5,:] = u - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = u*a
        r24 = u*b + c
        r25 = u*b - c
        r31 = v*a
        r33 = np.sqrt(2)*c
        r34 = v*b
        r41 = w*a
        r44 = w*b
        r51 = k*a
        r52 = -w*r33
        r53 = v*r33
        r54 = k*b + u*c + p/((2*g1)*b)
        r55 = k*b - u*c + p/((2*g1)*b)     
        Y = fn.block_diag(a,r0,r0,b,b,r21,r0,r0,r24,r25,r31,r0,r33,r34,r34,r41,-r33,r0,r44,r44,r51,r52,r53,r54,r55)
        if trans:
            YT = fn.block_diag(a,r21,r31,r41,r51,r0,r0,r0,-r33,r52,r0,r0,r33,r0,r53,b,r24,r34,r44,r54,b,r25,r34,r44,r55)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r11 = 1/a - a*k/p
        r12 = a*u/p
        r13 = a*v/p
        r14 = a*w/p
        r15 = -a/p
        r33 = np.sqrt(2)*c
        r21 = r33*w
        r0 = np.zeros(np.shape(rho))
        r31 = -r33*v
        r0 = np.zeros(np.shape(rho))
        r41 = k*b - u*c
        r42 = c - u*b
        r43 = -v*b
        r44 = -w*b
        r51 = k*b + u*c
        r52 = -c - u*b
        Yinv = fn.block_diag(r11,r12,r13,r14,r15,r21,r0,r0,-r33,r0,r31,r0,r33,r0,r0,r41,r42,r43,r44,b,r51,r52,r43,r44,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT     
        
@njit    
def dEydq_eigs_3D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    k = (u**2+v**2+w**2)/2
    e = q[4::5,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::5,:] = v
        Lam[1::5,:] = v
        Lam[2::5,:] = v
        Lam[3::5,:] = v + a
        Lam[4::5,:] = v - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r22 = u*a
        r23 = np.sqrt(2)*c
        r24 = u*b
        r32 = v*a
        r34 = v*b + c
        r35 = v*b - c
        r42 = w*a
        r44 = w*b
        r51 = -w*r23
        r52 = k*a
        r53 = u*r23
        r54 = k*b + v*c + p/((2*g1)*b)
        r55 = k*b - v*c + p/((2*g1)*b)     
        Y = fn.block_diag(r0,a,r0,b,b,r0,r22,r23,r24,r24,r0,r32,r0,r34,r35,-r23,r42,r0,r44,r44,r51,r52,r53,r54,r55)
        if trans:
            YT = fn.block_diag(r0,r0,r0,-r23,r51,a,r22,r32,r42,r52,r0,r23,r0,r0,r53,b,r24,r34,r44,r54,b,r24,r35,r44,r55)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r32 = np.sqrt(2)*c
        r11 = w*r32
        r0 = np.zeros(np.shape(rho))
        r21 = 1/a - a*k/p
        r22 = a*u/p
        r23 = a*v/p
        r24 = a*w/p
        r25 = -a/p
        r31 = -u*r32
        r41 = k*b - v*c
        r42 = -u*b
        r43 = c - v*b
        r44 = -w*b
        r51 = k*b + v*c
        r53 = -c - v*b
        Yinv = fn.block_diag(r11,r0,r0,-r32,r0,r21,r22,r23,r24,r25,r31,r32,r0,r0,r0,r41,r42,r43,r44,b,r51,r42,r53,r44,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT  
        
@njit    
def dEzdq_eigs_3D(q,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*3,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    k = (u**2+v**2+w**2)/2
    e = q[4::5,:]
    p = g1*(e-rho*k) # pressure
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::5,:] = w
        Lam[1::5,:] = w
        Lam[2::5,:] = w
        Lam[3::5,:] = w + a
        Lam[4::5,:] = w - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p/2)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r22 = np.sqrt(2)*c
        r23 = u*a
        r24 = u*b
        r33 = v*a
        r34 = v*b
        r43 = w*a
        r44 = w*b + c
        r45 = w*b - c
        r51 = -v*r22
        r52 = u*r22
        r53 = k*a
        r54 = k*b + w*c + p/((2*g1)*b)
        r55 = k*b - w*c + p/((2*g1)*b)     
        Y = fn.block_diag(r0,r0,a,b,b,r0,r22,r23,r24,r24,-r22,r0,r33,r34,r34,r0,r0,r43,r44,r45,r51,r52,r53,r54,r55)
        if trans:
            YT = fn.block_diag(r0,r0,-r22,r0,r51,r0,r22,r0,r0,r52,a,r23,r33,r43,r53,b,r24,r34,r44,r54,b,r24,r34,r45,r55)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r22 = np.sqrt(2)*c
        r11 = v*r22
        r0 = np.zeros(np.shape(rho))
        r21 = -u*r22
        r31 = 1/a - a*k/p
        r32 = a*u/p
        r33 = a*v/p
        r34 = a*w/p
        r35 = -a/p
        r41 = k*b - w*c
        r42 = -u*b
        r43 = -v*b
        r44 = c -w*b
        r51 = k*b + w*c
        r54 = -c - w*b
        Yinv = fn.block_diag(r11,r0,-r22,r0,r0,r21,r22,r0,r0,r0,r31,r32,r33,r34,r35,r41,r42,r43,r44,b,r51,r42,r43,r54,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT  

@njit 
def abs_Roe_fix_1D(Lam):
    ''' Take eigenvalue matrix, return absolute value with entropy fix '''
    Lam_abs = np.abs(Lam)
    d = 0.1 * np.maximum(Lam_abs[1::3,:],Lam_abs[2::3,:]) # Like Zingg textbook, but smooth cutoff
    x,y = np.shape(Lam)
    Lam_fix = np.zeros((x,y))
    for xi in range(x):
        dxi = xi//3
        for yi in range(y):
            if Lam_abs[xi,yi] < d[dxi,yi]:
                Lam_fix[xi,yi] = 0.5*(Lam_abs[xi,yi]**2/d[dxi,yi] + d[dxi,yi])
            else:
                Lam_fix[xi,yi] = Lam_abs[xi,yi]
    return Lam_fix
                
@njit     
def abs_Roe_fix_2D(Lam):
    ''' Take eigenvalue matrix, return absolute value with entropy fix '''
    Lam_abs = np.abs(Lam)
    d = 0.1 * np.maximum(Lam_abs[2::4,:],Lam_abs[3::4,:]) # Like Zingg textbook, but smooth cutoff
    x,y = np.shape(Lam)
    Lam_fix = np.zeros((x,y))
    for xi in range(x):
        dxi = xi//4
        for yi in range(y):
            if Lam_abs[xi,yi] < d[dxi,yi]:
                Lam_fix[xi,yi] = 0.5*(Lam_abs[xi,yi]**2/d[dxi,yi] + d[dxi,yi])
            else:
                Lam_fix[xi,yi] = Lam_abs[xi,yi]
    return Lam_fix

@njit     
def abs_Roe_fix_3D(Lam):
    ''' Take eigenvalue matrix, return absolute value with entropy fix '''
    Lam_abs = np.abs(Lam)
    d = 0.1 * np.maximum(Lam_abs[3::5,:],Lam_abs[4::5,:]) # Like Zingg textbook, but smooth cutoff
    x,y = np.shape(Lam)
    Lam_fix = np.zeros((x,y))
    for xi in range(x):
        dxi = xi//5
        for yi in range(y):
            if Lam_abs[xi,yi] < d[dxi,yi]:
                Lam_fix[xi,yi] = 0.5*(Lam_abs[xi,yi]**2/d[dxi,yi] + d[dxi,yi])
            else:
                Lam_fix[xi,yi] = Lam_abs[xi,yi]
    return Lam_fix

@jit(nopython=True)
def log_mean(qL,qR):
    # logarithmic mean. Useful for EC fluxes like Ismail-Roe.
    xi = qL/qR
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    q = (qL+qR)/F
    return q
    