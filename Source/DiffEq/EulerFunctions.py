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
def calcEx_1D(q):
    ''' the flux vector in 1D, hard coded with g=1.4 '''
    # decompose_q
    q_0 = q[0::3]
    q_1 = q[1::3]
    q_2 = q[2::3]
    u = q_1 / q_0

    k = u*q_1                    # Common term: rho * u^2 * S
    ps = 0.4*(q_2 - k/2)  # p * S

    e0 = q_1
    e1 = k + ps
    e2 = u*(q_2 + ps)

    # assemble_vec
    E = np.stack((e0,e1,e2)).reshape(q.shape, order='F')
    return E

@njit    
def dExdq_1D(q):
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
def dExdq_1D_complex(q):
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
def dEndq_2D(q,n):
    ''' the n-direction flux jacobian A in 2D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    uv = u*v
    u2 = u**2
    v2 = v**2
    k = (u2+v2)/2
    g_e_rho = g * q[3::4,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    uvn = nx*u + ny*v

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = nx*(g1 * k - u2) - ny*uv
    r22 = (nx*(3-g)) * u + ny*v
    r23 = nx*(-g1 * v) + ny*u
    r24 = (nx*g1)*r1
    r31 = -nx*uv + ny*(g1 * k - v2)
    r32 = nx*v + ny*(-g1 * u)
    r33 = nx*u + ny*((3-g) * v)
    r34 = (ny*g1)*r1
    r41 = ((2 * g1) * k - g_e_rho) * uvn
    r42 = nx*(g_e_rho - g1*(k + u2)) + ny*(-g1 * uv)
    r43 = nx*(-g1 * uv) + ny*(g_e_rho - g1*(k + v2))
    r44 = g * uvn
    
    dEdq = fn.block_diag(r0,r1*nx,r1*ny,r0,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
    return dEdq

@njit    
def dEndq_2D_complex(q,n):
    ''' the n-direction flux jacobian A in 2D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    uv = u*v
    u2 = u**2
    v2 = v**2
    k = (u2+v2)/2
    g_e_rho = g * q[3::4,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    uvn = nx*u + ny*v

    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = nx*(g1 * k - u2) - ny*uv
    r22 = (nx*(3-g)) * u + ny*v
    r23 = nx*(-g1 * v) + ny*u
    r24 = (nx*g1)*r1
    r31 = -nx*uv + ny*(g1 * k - v2)
    r32 = nx*v + ny*(-g1 * u)
    r33 = nx*u + ny*((3-g) * v)
    r34 = (ny*g1)*r1
    r41 = ((2 * g1) * k - g_e_rho) * uvn
    r42 = nx*(g_e_rho - g1*(k + u2)) + ny*(-g1 * uv)
    r43 = nx*(-g1 * uv) + ny*(g_e_rho - g1*(k + v2))
    r44 = g * uvn
    
    dEdq = fn.block_diag(r0,r1*nx,r1*ny,r0,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
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
def dEndq_3D(q,n):
    ''' the n-direction flux jacobian A in 3D. Note: can NOT handle complex values '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    u2 = u**2
    v2 = v**2
    w2 = w**2
    k = (u2+v2+w2)/2
    g_e_rho = g * q[4::5,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    nz = n[:,2,:]
    uvwn = nx*u + ny*v + nz*w
    
    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = nx*(g1 * k - u2) - ny*(u * v) - nz*(u * w)
    r22 = uvwn + (nx*(2-g))*u
    r23 = -(nx*g1) * v + ny*u
    r24 = -(nx*g1) * w + nz*u
    r25 = (nx*g1)*r1
    r31 = - nx*(u * v) + ny*(g1 * k - v2) - nz*(v * w)
    r32 = -(ny*g1) * u + nx*v
    r33 = uvwn + (ny*(2-g))*v
    r34 = -(ny*g1) * w + nz*v
    r35 = (ny*g1)*r1
    r41 = - nx*(u * w) - ny*(w * v) + nz*(g1 * k - w2)
    r42 = -(nz*g1) * u + nx*w
    r43 = -(nz*g1) * v + ny*w
    r44 = uvwn + (nz*(2-g))*w
    r45 = (nz*g1)*r1
    r51 = ((2 * g1) * k - g_e_rho)*uvwn
    r52 = nx*(g_e_rho - g1*(k + u2)) - ((ny*g1) * u * v) - ((nz*g1) * u * w)
    r53 = -((nx*g1) * u * v) + ny*(g_e_rho - g1*(k + v2)) - ((nz*g1) * v * w)
    r54 = -((nx*g1) * u * w) - ((ny*g1) * v * w) + nz*(g_e_rho - g1*(k + w2))
    r55 = g * uvwn
    
    dEdq = fn.block_diag(r0,nx*r1,ny*r1,nz*r1,r0,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
    return dEdq

@njit    
def dEndq_3D_complex(q,n):
    ''' the n-direction flux jacobian A in 3D. Note: intended for use with complex step '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    u2 = u**2
    v2 = v**2
    w2 = w**2
    k = (u2+v2+w2)/2
    g_e_rho = g * q[4::5,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    nz = n[:,2,:]
    uvwn = nx*u + ny*v + nz*w
    
    # entries of the dEdq (A) matrix
    r1 = np.ones_like(rho,dtype=np.complex128)
    r0 = np.zeros_like(rho,dtype=np.complex128)
    r21 = nx*(g1 * k - u2) - ny*(u * v) - nz*(u * w)
    r22 = uvwn + (nx*(2-g))*u
    r23 = -(nx*g1) * v + ny*u
    r24 = -(nx*g1) * w + nz*u
    r25 = (nx*g1)*r1
    r31 = - nx*(u * v) + ny*(g1 * k - v2) - nz*(v * w)
    r32 = -(ny*g1) * u + nx*v
    r33 = uvwn + (ny*(2-g))*v
    r34 = -(ny*g1) * w + nz*v
    r35 = (ny*g1)*r1
    r41 = - nx*(u * w) - ny*(w * v) + nz*(g1 * k - w2)
    r42 = -(nz*g1) * u + nx*w
    r43 = -(nz*g1) * v + ny*w
    r44 = uvwn + (nz*(2-g))*w
    r45 = (nz*g1)*r1
    r51 = ((2 * g1) * k - g_e_rho)*uvwn
    r52 = nx*(g_e_rho - g1*(k + u2)) - ((ny*g1) * u * v) - ((nz*g1) * u * w)
    r53 = -((nx*g1) * u * v) + ny*(g_e_rho - g1*(k + v2)) - ((nz*g1) * v * w)
    r54 = -((nx*g1) * u * w) - ((ny*g1) * v * w) + nz*(g_e_rho - g1*(k + w2))
    r55 = g * uvwn
    
    dEdq = fn.block_diag(r0,nx*r1,ny*r1,nz*r1,r0,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
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
        c = -np.sqrt(p)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = u*a
        r23 = u*b - c/np.sqrt(2)
        r24 = u*b + c/np.sqrt(2)
        r31 = v*a
        r33 = v*b
        r41 = k*a
        r42 = v*c
        r43 = k*b - u*c/np.sqrt(2) + p/((2*g1)*b)
        r44 = k*b + u*c/np.sqrt(2) + p/((2*g1)*b)
        Y = fn.block_diag(a,r0,b,b,r21,r0,r23,r24,r31,c,r33,r33,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(a,r21,r31,r41,r0,r0,c,r42,b,r23,r33,r43,b,r24,r33,r44)
        else:
            YT = None 
    else:
        Y = None
        YT = None
    if inv:
        ap = -np.sqrt(rho*(g1/g))/p # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = -1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r11 = (g+1)*k*ap - g*e*ap/rho
        r12 = -u*ap
        r13 = -v*ap
        r23 = np.sqrt(2)*c
        r21 = -r23*v
        r31 = k*b + u*c
        r32 = -u*b - c
        r33 = -v*b
        r41 = k*b - u*c
        r42 = -u*b + c
        Yinv = fn.block_diag(r11,r12,r13,ap,r21,r0,r23,r0,r31,r32,r33,b,r41,r42,r33,b)
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
        c = np.sqrt(p)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = u*a
        r23 = u*b
        r31 = v*a
        r33 = v*b + c/np.sqrt(2)
        r34 = v*b - c/np.sqrt(2)
        r41 = k*a
        r42 = u*c
        r43 = k*b + v*c/np.sqrt(2) + p/((2*g1)*b)
        r44 = k*b - v*c/np.sqrt(2) + p/((2*g1)*b)
        Y = fn.block_diag(a,r0,b,b,r21,c,r23,r23,r31,r0,r33,r34,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(a,r21,r31,r41,r0,c,r0,r42,b,r23,r33,r43,b,r23,r34,r44)
        else:
            YT = None 
    else:
        Y = None
        YT = None
    if inv:
        ap = -np.sqrt(rho*(g1/g))/p # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r11 = (g+1)*k*ap - g*e*ap/rho
        r12 = -u*ap
        r13 = -v*ap
        r22 = np.sqrt(2)*c
        r21 = -r22*u
        r31 = k*b - v*c
        r32 = -u*b 
        r33 = -v*b + c
        r41 = k*b + v*c
        r43 = -v*b - c
        Yinv = fn.block_diag(r11,r12,r13,ap,r21,r22,r0,r0,r31,r32,r33,b,r41,r32,r43,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT  

@njit    
def dEndq_eigs_2D(q,n,val=True,vec=True,inv=True,trans=False):
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
    nx = n[:,0,:]
    ny = n[:,1,:]
    uvn = nx*u + ny*v
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::4,:] = uvn
        Lam[1::4,:] = uvn
        Lam[2::4,:] = uvn + a
        Lam[3::4,:] = uvn - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p)   
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r21 = u*a
        r22 = ny*c
        r32 = -nx*c
        r23 = u*b - r32/np.sqrt(2)
        r24 = u*b + r32/np.sqrt(2)
        r31 = v*a
        r33 = v*b + r22/np.sqrt(2)
        r34 = v*b - r22/np.sqrt(2)
        r41 = k*a
        r42 = (ny*u-nx*v)*c
        r43 = k*b + uvn*c/np.sqrt(2) + p/((2*g1)*b)
        r44 = k*b - uvn*c/np.sqrt(2) + p/((2*g1)*b)
        Y = fn.block_diag(a,r0,b,b,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(a,r21,r31,r41,r0,r22,r32,r42,b,r23,r33,r43,b,r24,r34,r44)
        else:
            YT = None        
    else:
        Y = None
        YT = None
    if inv:
        ap = -np.sqrt(rho*(g1/g))/p # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r0 = np.zeros(np.shape(rho))
        r11 = (g+1)*k*ap - g*e*ap/rho
        r12 = -u*ap
        r13 = -v*ap
        r21 = np.sqrt(2)*c*(nx*v-ny*u)
        r22 = np.sqrt(2)*c*ny
        r23 = -np.sqrt(2)*c*nx
        r31 = k*b - uvn*c
        r32 = -u*b + nx*c
        r33 = -v*b + ny*c
        r41 = k*b + uvn*c
        r42 = -u*b - nx*c
        r43 = -v*b - ny*c
        Yinv = fn.block_diag(r11,r12,r13,ap,r21,r22,r23,r0,r31,r32,r33,b,r41,r42,r43,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT 

@njit    
def symmetrizer_1D(q):
    ''' take a q of shape (nen*3,nelem) and builds the symmetrizing matrix P.
    P is the symmetric positive definite matrix that symmetrizes the flux
    jacobian A=dEdq upon multiplication from the right. This is equal to the
    derivative of conservative variables with respect to entropy variables,
    or the Hessian of the entropy potential. '''
    g = 1.4 # hard coded throughout
    rho = q[::3,:]
    rhou = q[1::3,:]
    rhou2 = rhou**2/rho
    e = q[2::3,:]
    p = (g-1)*(e-rhou2/2) # pressure
    
    r22 = rhou2 + p
    r23 = rhou*(p+e)/rho
    r33 = g*e**2/rho - (g-1)*rhou2**2/4/rho
    
    P = fn.block_diag(rho,rhou,e,rhou,r22,r23,e,r23,r33)
    return P

@njit    
def symmetrizer_2D(q):
    ''' take a q of shape (nen*4,nelem) and builds the symmetrizing matrix P.
    P is the symmetric positive definite matrix that symmetrizes the flux
    jacobian A=dEndq upon multiplication from the right. This is equal to the
    derivative of conservative variables with respect to entropy variables,
    or the Hessian of the entropy potential. '''
    g = 1.4 # hard coded throughout
    rho = q[::4,:]
    rhou = q[1::4,:]
    rhov = q[2::4,:]
    rhou2 = rhou**2/rho
    rhov2 = rhov**2/rho
    e = q[3::4,:]
    p = (g-1)*(e-(rhou2 + rhov2)/2) # pressure
    
    r22 = rhou2 + p
    r23 = rhou*rhov/rho
    r24 = rhou*(p+e)/rho
    r33 = rhov2 + p
    r34 = rhov*(p+e)/rho
    r44 = g*e**2/rho - (g-1)*(rhou2+rhov2)**2/4/rho # TODO: can i reduce the error here???
    
    P = fn.block_diag(rho,rhou,rhov,e,rhou,r22,r23,r24,rhov,r23,r33,r34,e,r24,r34,r44)
    return P

@njit    
def symmetrizer_3D(q):
    ''' take a q of shape (nen*5,nelem) and builds the symmetrizing matrix P.
    P is the symmetric positive definite matrix that symmetrizes the flux
    jacobian A=dEndq upon multiplication from the right. This is equal to the
    derivative of conservative variables with respect to entropy variables,
    or the Hessian of the entropy potential. '''
    g = 1.4 # hard coded throughout
    rho = q[::5,:]
    rhou = q[1::5,:]
    rhov = q[2::5,:]
    rhow = q[3::5,:]
    rhou2 = rhou**2/rho
    rhov2 = rhov**2/rho
    rhow2 = rhow**2/rho
    e = q[4::5,:]
    p = (g-1)*(e-(rhou2 + rhov2 + rhow2)/2) # pressure
    
    r22 = rhou2 + p
    r23 = rhou*rhov/rho
    r24 = rhou*rhow/rho
    r25 = rhou*(p+e)/rho
    r33 = rhov2 + p
    r34 = rhov*rhow/rho
    r35 = rhov*(p+e)/rho
    r44 = rhow2 + p
    r45 = rhow*(p+e)/rho
    r55 = g*e**2/rho - (g-1)*(rhou2+rhov2+rhow2)**2/4/rho
    
    P = fn.block_diag(rho,rhou,rhov,rhow,e,rhou,r22,r23,r24,r25,rhov,r23,r33,r34,r35,rhow,r24,r34,r44,r45,e,r25,r35,r45,r55)
    return P

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
        r41 = np.sqrt(2)*c
        r22 = u*a
        r24 = u*b
        r32 = v*a
        r34 = v*b + c
        r35 = v*b - c
        r42 = w*a
        r44 = w*b
        r51 = w*r41
        r52 = k*a
        r53 = -u*r41
        r54 = k*b + v*c + p/((2*g1)*b)
        r55 = k*b - v*c + p/((2*g1)*b)     
        Y = fn.block_diag(r0,a,r0,b,b,r0,r22,-r41,r24,r24,r0,r32,r0,r34,r35,r41,r42,r0,r44,r44,r51,r52,r53,r54,r55)
        if trans:
            YT = fn.block_diag(r0,r0,r0,r41,r51,a,r22,r32,r42,r52,r0,-r41,r0,r0,r53,b,r24,r34,r44,r54,b,r24,r35,r44,r55)
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
        r14 = -np.sqrt(2)*c
        r11 = w*r14
        r0 = np.zeros(np.shape(rho))
        r21 = 1/a - a*k/p
        r22 = a*u/p
        r23 = a*v/p
        r24 = a*w/p
        r25 = -a/p
        r31 = -u*r14
        r41 = k*b - v*c
        r42 = -u*b
        r43 = c - v*b
        r44 = -w*b
        r51 = k*b + v*c
        r53 = -c - v*b
        Yinv = fn.block_diag(r11,r0,r0,-r14,r0,r21,r22,r23,r24,r25,r31,r14,r0,r0,r0,r41,r42,r43,r44,b,r51,r42,r53,r44,b)
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
def dEndq_eigs_3D(q,n,val=True,vec=True,inv=True,trans=False):
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
    nx = n[:,0,:]
    ny = n[:,1,:]
    nz = n[:,2,:]
    uvwn = nx*u + ny*v + nz*w
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed
        Lam[::5,:] = uvwn
        Lam[1::5,:] = uvwn
        Lam[2::5,:] = uvwn
        Lam[3::5,:] = uvwn + a
        Lam[4::5,:] = uvwn - a
    else:
        Lam = None
    if vec:
        a = np.sqrt(rho*(g1/g)) # useful constants
        b = np.sqrt(rho/(2*g))
        c = np.sqrt(p)   
        # entries of the eigenvector (Y) matrix
        r11 = nx*a
        r12 = ny*a
        r13 = nz*a
        r21 = u*r11
        r22 = u*r12 + nz*c
        r23 = u*r13 - ny*c
        r24 = u*b + nx*c/np.sqrt(2)
        r25 = u*b - nx*c/np.sqrt(2)
        r31 = v*r11 - nz*c
        r32 = v*r12
        r33 = v*r13 + nx*c
        r34 = v*b + ny*c/np.sqrt(2)
        r35 = v*b - ny*c/np.sqrt(2)
        r41 = w*r11 + ny*c
        r42 = w*r12 - nx*c
        r43 = w*r13
        r44 = w*b + nz*c/np.sqrt(2)
        r45 = w*b - nz*c/np.sqrt(2)
        r51 = nx*k*a + (ny*w-nz*v)*c
        r52 = ny*k*a + (nz*u-nx*w)*c
        r53 = nz*k*a + (nx*v-ny*u)*c
        r54 = k*b + uvwn*c/np.sqrt(2) + p/((2*g1)*b)
        r55 = k*b - uvwn*c/np.sqrt(2) + p/((2*g1)*b)     
        Y = fn.block_diag(r11,r12,r13,b,b,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
        if trans:
            YT = fn.block_diag(r11,r21,r31,r41,r51,r12,r22,r32,r42,r52,r13,r23,r33,r43,r53,b,r24,r34,r44,r54,b,r25,r35,r45,r55)
        else:
            YT = None
    else:
        Y = None
        YT = None
    if inv:
        a = np.sqrt(rho*(g1/g))
        ap = np.sqrt(rho*(g1/g))/p # useful constants
        b = g1 * (np.sqrt(rho/(2*g)) / p)
        c = 1/np.sqrt(2*p)
        # entries of the eigenvector (Y) matrix
        r11 = nx*(1/a-k*a/p) - (ny*w-nz*v)/np.sqrt(p)
        r12 = nx*u*ap
        r13 = nx*v*ap - np.sqrt(2)*nz*c
        r14 = nx*w*ap + np.sqrt(2)*ny*c
        r15 = -nx*ap
        r21 = ny*(1/a-k*a/p) - (nz*u-nx*w)/np.sqrt(p)
        r22 = ny*u*ap + np.sqrt(2)*nz*c
        r23 = ny*v*ap
        r24 = ny*w*ap - np.sqrt(2)*nx*c
        r25 = -ny*ap
        r31 = nz*(1/a-k*a/p) - (nx*v-ny*u)/np.sqrt(p)
        r32 = nz*u*ap - np.sqrt(2)*ny*c
        r33 = nz*v*ap + np.sqrt(2)*nx*c
        r34 = nz*w*ap
        r35 = -nz*ap
        r41 = k*b - uvwn*c
        r42 = -u*b + nx*c
        r43 = -v*b + ny*c
        r44 = -w*b + nz*c
        r51 = k*b + uvwn*c
        r52 = -u*b - nx*c
        r53 = -v*b - ny*c
        r54 = -w*b - nz*c
        Yinv = fn.block_diag(r11,r12,r13,r14,r15,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,b,r51,r52,r53,r54,b)
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT   

@njit 
def abs_Roe_fix_1D(Lam):
    ''' Take eigenvalue matrix (actually flat), return absolute value with entropy fix '''
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
    ''' Take eigenvalue matrix (actually flat), return absolute value with entropy fix '''
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
    ''' Take eigenvalue matrix (actually flat), return absolute value with entropy fix '''
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

@njit
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

@njit 
def Ismail_Roe(qL,qR):
    '''
    Return the ismail roe flux given two states qL and qR where each is
    of shape (neq=3,), and returns a numerical flux of shape (neq=3,)
    note subroutine defined in ismail-roe appendix B for logarithmic mean
    '''
    g = 1.4 # hard coded!
    
    rhoL, rhoR = qL[0], qR[0]
    uL, uR = qL[1]/rhoL, qR[1]/rhoR
    pL, pR = (g-1)*(qL[2] - (rhoL * uL**2)/2), (g-1)*(qR[2] - (rhoR * uR**2)/2)

    alphaL = np.sqrt(rhoL/pL)
    alphaR = np.sqrt(rhoR/pR)
    betaL = np.sqrt(rhoL*pL)
    betaR = np.sqrt(rhoR*pR)

    xi_alpha = alphaL/alphaR
    zeta_alpha = (1-xi_alpha)/(1+xi_alpha)
    zeta_alpha2 = zeta_alpha**2
    if zeta_alpha2 < 0.01:
        F_alpha = 2*(1. + zeta_alpha2/3. + zeta_alpha2**2/5. + zeta_alpha2**3/7.)
    else:
        F_alpha = - np.log(xi_alpha)/zeta_alpha
    alpha_ln = (alphaL+alphaR)/F_alpha

    xi_beta = betaL/betaR
    zeta_beta = (1-xi_beta)/(1+xi_beta)
    zeta_beta2 = zeta_beta**2
    if zeta_beta2 < 0.01:
        F_beta = 2*(1. + zeta_beta2/3. + zeta_beta2**2/5. + zeta_beta2**3/7.)
    else:
        F_beta = - np.log(xi_beta) / zeta_beta
    beta_ln = (betaL+betaR)/F_beta

    alpha_avg = 0.5*(alphaL+alphaR)
    beta_avg = 0.5*(betaL+betaR)

    rho_avg = alpha_avg * beta_ln
    a_avg2 = (0.5/rho_avg)*((g+1)*beta_ln/alpha_ln + (g-1)*beta_avg/alpha_avg)
    u_avg = 0.5 * (uL*alphaL + uR*alphaR) / alpha_avg
    p_avg = beta_avg / alpha_avg
    H_avg = a_avg2/(g - 1) + 0.5*u_avg**2
    
    rhou_avg = rho_avg*u_avg
    return np.array([rhou_avg, rhou_avg*u_avg + p_avg, rhou_avg*H_avg]) 







if __name__ == "__main__":
    
    nelem = 2
    nen = 3
    
    q1D = np.random.rand(nen*3,nelem) + 10
    q1D[2::3,:] *= 1000 # fix e to ensure pressure is positive
    
    Lam1D, Y1D, Yinv1D, YT1D = dEdq_eigs_1D(q1D,val=True,vec=True,inv=True,trans=True)
    A1D = dEdq_1D(q1D)
    P1D = symmetrizer_1D(q1D)
    AP1D = fn.gm_gm(A1D,P1D)
    
    print('---- Testing 1D functions (all should be zero) ----')
    print('eigenvector inverse: ', np.max(abs(np.linalg.inv(Y1D[:,:,0])-Yinv1D[:,:,0])))
    print('eigenvector transpose: ', np.max(abs(Y1D[:,:,0].T - YT1D[:,:,0])))
    print('eigendecomposition: ', np.max(abs(A1D - fn.gm_gm(Y1D,fn.gdiag_gm(Lam1D,Yinv1D)))))
    print('A @ P symmetrizer: ', np.max(abs(AP1D[:,:,0] - AP1D[:,:,0].T)))
    print('P - eigenvector scaling: ', np.max(abs(P1D - fn.gm_gm(Y1D,YT1D))))
    print('')
    
    q2D = np.random.rand(nen*4,nelem) + 10
    q2D[3::4,:] *= 1000 # fix e to ensure pressure is positive
    
    n2D = np.random.rand(nen,2,nelem)
    norm = np.sqrt(n2D[:,0,:]**2 + n2D[:,1,:]**2)
    n2D[:,0,:] /= norm
    n2D[:,1,:] /= norm
    nx2D = np.zeros((nen,2,nelem))
    nx2D[:,0,:] = 1
    ny2D = np.zeros((nen,2,nelem))
    ny2D[:,1,:] = 1
    
    Lam2D, Y2D, Yinv2D, YT2D = dEndq_eigs_2D(q2D,n2D,val=True,vec=True,inv=True,trans=True)
    Lamx2D, Yx2D, Yinvx2D, YTx2D = dExdq_eigs_2D(q2D,val=True,vec=True,inv=True,trans=True)
    Lamy2D, Yy2D, Yinvy2D, YTy2D = dEydq_eigs_2D(q2D,val=True,vec=True,inv=True,trans=True)
    Lamx22D, Yx22D, Yinvx22D, YTx22D = dEndq_eigs_2D(q2D,nx2D,val=True,vec=True,inv=True,trans=True)
    Lamy22D, Yy22D, Yinvy22D, YTy22D = dEndq_eigs_2D(q2D,ny2D,val=True,vec=True,inv=True,trans=True)
    An2D = dEndq_2D(q2D,n2D)
    Ax2D = dExdq_2D(q2D)
    Ay2D = dEydq_2D(q2D)
    P2D = symmetrizer_2D(q2D)
    AP2D = fn.gm_gm(An2D,P2D)
    from scipy.linalg import block_diag
    xblocks = []
    yblocks = []
    for node in range(nen):
        xblocks.append(np.ones((4,4))*n2D[node,0,0])
        yblocks.append(np.ones((4,4))*n2D[node,1,0])
    Nx = block_diag(*xblocks)
    Ny = block_diag(*yblocks)
    An22D = Ax2D[:,:,0]*Nx + Ay2D[:,:,0]*Ny
    
    print('---- Testing 2D functions (all should be zero) ----')
    print('An = nx*Ax + ny*Ay: ', np.max(abs(An2D[:,:,0]-An22D)))
    print('x/nx eigenvalues: ', np.max(abs(Lamx2D-Lamx22D)))
    print('y/ny eigenvalues: ', np.max(abs(Lamy2D-Lamy22D)))
    print('x/nx eigenvectors: ', np.max(abs(Yx2D-Yx22D)))
    print('y/ny eigenvectors: ', np.max(abs(Yy2D-Yy22D)))
    print('x/nx eigenvector transpose: ', np.max(abs(YTx2D-YTx22D)))
    print('y/ny eigenvector transpose: ', np.max(abs(YTy2D-YTy22D)))
    print('x/nx eigenvector inverse: ', np.max(abs(Yinvx2D-Yinvx22D)))
    print('y/ny eigenvector inverse: ', np.max(abs(Yinvy2D-Yinvy22D)))
    print('x eigenvector inverse: ', np.max(abs(np.linalg.inv(Yx2D[:,:,0])-Yinvx2D[:,:,0])))
    print('y eigenvector inverse: ', np.max(abs(np.linalg.inv(Yy2D[:,:,0])-Yinvy2D[:,:,0])))
    print('n eigenvector inverse: ', np.max(abs(np.linalg.inv(Y2D[:,:,0])-Yinv2D[:,:,0])))
    print('x eigenvector transpose: ', np.max(abs(Yx2D[:,:,0].T - YTx2D[:,:,0])))
    print('y eigenvector transpose: ', np.max(abs(Yy2D[:,:,0].T - YTy2D[:,:,0])))
    print('n eigenvector transpose: ', np.max(abs(Y2D[:,:,0].T - YT2D[:,:,0])))
    print('x eigendecomposition: ', np.max(abs(Ax2D - fn.gm_gm(Yx2D,fn.gdiag_gm(Lamx2D,Yinvx2D)))))
    print('y eigendecomposition: ', np.max(abs(Ay2D - fn.gm_gm(Yy2D,fn.gdiag_gm(Lamy2D,Yinvy2D)))))
    print('nx eigendecomposition: ', np.max(abs(Ax2D - fn.gm_gm(Yx22D,fn.gdiag_gm(Lamx22D,Yinvx22D)))))
    print('ny eigendecomposition: ', np.max(abs(Ay2D - fn.gm_gm(Yy22D,fn.gdiag_gm(Lamy22D,Yinvy22D)))))
    print('n eigendecomposition: ', np.max(abs(An2D - fn.gm_gm(Y2D,fn.gdiag_gm(Lam2D,Yinv2D)))))
    print('An @ P symmetrizer: ', np.max(abs(AP2D[:,:,0] - AP2D[:,:,0].T)))
    print('P - eigenvector scaling: ', np.max(abs(P2D - fn.gm_gm(Y2D,YT2D))))
    print('')
    
    q3D = np.random.rand(nen*5,nelem) + 10
    q3D[4::5,:] *= 1000 # fix e to ensure pressure is positive
    
    n3D = np.random.rand(nen,3,nelem)
    norm = np.sqrt(n3D[:,0,:]**2 + n3D[:,1,:]**2 + n3D[:,2,:]**2)
    n3D[:,0,:] /= norm
    n3D[:,1,:] /= norm
    n3D[:,2,:] /= norm
    nx3D = np.zeros((nen,3,nelem))
    nx3D[:,0,:] = 1
    ny3D = np.zeros((nen,3,nelem))
    ny3D[:,1,:] = 1
    nz3D = np.zeros((nen,3,nelem))
    nz3D[:,2,:] = 1
    
    Lam3D, Y3D, Yinv3D, YT3D = dEndq_eigs_3D(q3D,n3D,val=True,vec=True,inv=True,trans=True)
    Lamx3D, Yx3D, Yinvx3D, YTx3D = dExdq_eigs_3D(q3D,val=True,vec=True,inv=True,trans=True)
    Lamy3D, Yy3D, Yinvy3D, YTy3D = dEydq_eigs_3D(q3D,val=True,vec=True,inv=True,trans=True)
    Lamz3D, Yz3D, Yinvz3D, YTz3D = dEzdq_eigs_3D(q3D,val=True,vec=True,inv=True,trans=True)
    Lamx23D, Yx23D, Yinvx23D, YTx23D = dEndq_eigs_3D(q3D,nx3D,val=True,vec=True,inv=True,trans=True)
    Lamy23D, Yy23D, Yinvy23D, YTy23D = dEndq_eigs_3D(q3D,ny3D,val=True,vec=True,inv=True,trans=True)
    Lamz23D, Yz23D, Yinvz23D, YTz23D = dEndq_eigs_3D(q3D,nz3D,val=True,vec=True,inv=True,trans=True)
    An3D = dEndq_3D(q3D,n3D)
    Ax3D = dExdq_3D(q3D)
    Ay3D = dEydq_3D(q3D)
    Az3D = dEzdq_3D(q3D)
    P3D = symmetrizer_3D(q3D)
    AP3D = fn.gm_gm(An3D,P3D)
    from scipy.linalg import block_diag
    xblocks = []
    yblocks = []
    zblocks = []
    for node in range(nen):
        xblocks.append(np.ones((5,5))*n3D[node,0,0])
        yblocks.append(np.ones((5,5))*n3D[node,1,0])
        zblocks.append(np.ones((5,5))*n3D[node,2,0])
    Nx = block_diag(*xblocks)
    Ny = block_diag(*yblocks)
    Nz = block_diag(*zblocks)
    An23D = Ax3D[:,:,0]*Nx + Ay3D[:,:,0]*Ny + Az3D[:,:,0]*Nz
    
    print('---- Testing 3D functions (all should be zero) ----')
    print('An = nx*Ax + ny*Ay + nz*Az: ', np.max(abs(An3D[:,:,0]-An23D)))
    print('x/nx eigenvalues: ', np.max(abs(Lamx3D-Lamx23D)))
    print('y/ny eigenvalues: ', np.max(abs(Lamy3D-Lamy23D)))
    print('z/nz eigenvalues: ', np.max(abs(Lamz3D-Lamz23D)))
    print('x/nx eigenvectors: ', np.max(abs(Yx3D-Yx23D)))
    print('y/ny eigenvectors: ', np.max(abs(Yy3D-Yy23D)))
    print('z/nz eigenvectors: ', np.max(abs(Yz3D-Yz23D)))
    print('x/nx eigenvector transpose: ', np.max(abs(YTx3D-YTx23D)))
    print('y/ny eigenvector transpose: ', np.max(abs(YTy3D-YTy23D)))
    print('z/nz eigenvector transpose: ', np.max(abs(YTz3D-YTz23D)))
    print('x/nx eigenvector inverse: ', np.max(abs(Yinvx3D-Yinvx23D)))
    print('y/ny eigenvector inverse: ', np.max(abs(Yinvy3D-Yinvy23D)))
    print('z/nz eigenvector inverse: ', np.max(abs(Yinvz3D-Yinvz23D)))
    print('x eigenvector inverse: ', np.max(abs(np.linalg.inv(Yx3D[:,:,0])-Yinvx3D[:,:,0])))
    print('y eigenvector inverse: ', np.max(abs(np.linalg.inv(Yy3D[:,:,0])-Yinvy3D[:,:,0])))
    print('z eigenvector inverse: ', np.max(abs(np.linalg.inv(Yz3D[:,:,0])-Yinvz3D[:,:,0])))
    print('n eigenvector inverse: ', np.max(abs(np.linalg.inv(Y3D[:,:,0])-Yinv3D[:,:,0])))
    print('x eigenvector transpose: ', np.max(abs(Yx3D[:,:,0].T - YTx3D[:,:,0])))
    print('y eigenvector transpose: ', np.max(abs(Yy3D[:,:,0].T - YTy3D[:,:,0])))
    print('z eigenvector transpose: ', np.max(abs(Yz3D[:,:,0].T - YTz3D[:,:,0])))
    print('n eigenvector transpose: ', np.max(abs(Y3D[:,:,0].T - YT3D[:,:,0])))
    print('x eigendecomposition: ', np.max(abs(Ax3D - fn.gm_gm(Yx3D,fn.gdiag_gm(Lamx3D,Yinvx3D)))))
    print('y eigendecomposition: ', np.max(abs(Ay3D - fn.gm_gm(Yy3D,fn.gdiag_gm(Lamy3D,Yinvy3D)))))
    print('z eigendecomposition: ', np.max(abs(Az3D - fn.gm_gm(Yz3D,fn.gdiag_gm(Lamz3D,Yinvz3D)))))
    print('nx eigendecomposition: ', np.max(abs(Ax3D - fn.gm_gm(Yx23D,fn.gdiag_gm(Lamx23D,Yinvx23D)))))
    print('ny eigendecomposition: ', np.max(abs(Ay3D - fn.gm_gm(Yy23D,fn.gdiag_gm(Lamy23D,Yinvy23D)))))
    print('nz eigendecomposition: ', np.max(abs(Az3D - fn.gm_gm(Yz23D,fn.gdiag_gm(Lamz23D,Yinvz23D)))))
    print('n eigendecomposition: ', np.max(abs(An3D - fn.gm_gm(Y3D,fn.gdiag_gm(Lam3D,Yinv3D)))))
    print('An @ P symmetrizer: ', np.max(abs(AP3D[:,:,0] - AP3D[:,:,0].T)))
    print('P - eigenvector scaling: ', np.max(abs(P3D - fn.gm_gm(Y3D,YT3D))))