#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:07:16 2021

@author: bercik
"""
import os
from sys import path

n_nested_folder = 2
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

from numba import njit
import numpy as np
import Source.Methods.Functions as fn

''' A collection of numerical 2-point fluxes for the Inviscid fluxes of the 
    Euler and Navier-Stokes equations. All jitted for speed '''

@njit 
def calcEx_1D(q):
    ''' the flux vector in 1D, hard coded with g=1.4 '''
    # correctly returns S * Ex if quasi1D Euler
    g = 1.4 # hard coded throughout

    # decompose_q
    fac = 1./q[0::3]
    q_1 = q[1::3]
    q_2 = q[2::3]
    u = q_1 * fac
    k = u*q_1   # = rho * u^2 * S if quasi1D Euler
    p = (g-1)*(q_2 - 0.5*k) # = p * S if quasi1D Euler

    # assemble_vec 
    E = np.zeros(q.shape)
    E[::3,:] = q_1
    E[1::3,:] = k + p
    E[2::3,:] = u*(q_2 + p)
    return E

@njit 
def calcEx_2D(q):
    ''' the flux vector in 2D, hard coded with g=1.4 '''
    g = 1.4 # hard coded throughout

    # decompose_q
    fac = 1./q[0::4]
    q_1 = q[1::4]
    q_2 = q[2::4]
    q_3 = q[3::4]
    u = q_1 * fac
    v = q_2 * fac
    p = (g-1)*(q_3 - 0.5*(u*q_1 + v*q_2))  

    # assemble_vec
    E = np.zeros(q.shape)
    E[::4,:] = q_1
    E[1::4,:] = q_1*u + p
    E[2::4,:] = q_1*v
    E[3::4,:] = u*(q_3 + p)
    return E

@njit 
def calcEy_2D(q):
    ''' the flux vector in 2D, hard coded with g=1.4 '''
    g = 1.4 # hard coded throughout

    # decompose_q
    fac = 1./q[0::4]
    q_1 = q[1::4]
    q_2 = q[2::4]
    q_3 = q[3::4]
    u = q_1 * fac
    v = q_2 * fac
    p = (g-1)*(q_3 - 0.5*(u*q_1 + v*q_2))  

    # assemble_vec
    E = np.zeros(q.shape)
    E[::4,:] = q_2
    E[1::4,:] = q_2*u
    E[2::4,:] = q_2*v + p
    E[3::4,:] = v*(q_3 + p)
    return E

@njit 
def calcExEy_2D(q):
    ''' the flux vector in 2D, hard coded with g=1.4 '''
    g = 1.4 # hard coded throughout

    # decompose_q
    fac = 1./q[0::4]
    q_1 = q[1::4]
    q_2 = q[2::4]
    q_3 = q[3::4]
    u = q_1 * fac
    v = q_2 * fac
    p = (g-1)*(q_3 - 0.5*(u*q_1 + v*q_2))  

    #assemble_xvec
    Ex = np.zeros(q.shape)
    Ex[::4,:] = q_1
    Ex[1::4,:] = q_1*u + p
    Ex[2::4,:] = q_1*v
    Ex[3::4,:] = u*(q_3 + p)

    # assemble_yvec
    Ey = np.zeros(q.shape)
    Ey[::4,:] = q_2
    Ey[1::4,:] = q_2*u
    Ey[2::4,:] = q_2*v + p
    Ey[3::4,:] = v*(q_3 + p)
    return Ex, Ey

@njit    
def dExdq_1D(q):
    ''' the flux jacobian A in 1D. Note: can NOT handle complex values '''
    # correctly returns A if quasi1D Euler (no S)
    g = 1.4 # hard coded throughout
    rho = q[::3,:] # = rho * S if quasi1D Euler
    u = q[1::3,:]/rho # = u if quasi1D Euler
    k = u*u/2 # = k if quasi1D Euler
    g_e_rho = g * q[2::3,:]/rho # = g_e_rho if quasi1D Euler

    # entries of the dEdq (A) matrix
    r1 = np.ones(np.shape(rho))
    r0 = np.zeros(np.shape(rho))
    r21 = (g-3) * k
    r22 = (3-g) * u
    r23 = r1*(g-1)
    r31 = ((2 * (g-1)) * k - g_e_rho) * u
    r32 = g_e_rho - (3*(g-1)) * k
    r33 = g * u
    
    dEdq = fn.block_diag(r0,r1,r0,r21,r22,r23,r31,r32,r33)
    return dEdq

@njit    
def dEndq_1D(q,dxidx):
    ''' the flux jacobian A in 1D including metric scaling. Note: can NOT handle complex values
    INPUTS: q : array of shape (nen*neq_node,nelem)
            dxidx : metrics of shape (nen,nelem) corresponding to a single xi '''
    # correctly returns An if quasi1D Euler (no S)
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::3,:]
    u = q[1::3,:]/rho
    k = (u*u)/2
    g_e_rho = g * q[2::3,:]/rho
    un = dxidx*u

    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho))
    r21 = (g-3) * dxidx * k 
    r22 = (3-g) * un
    r23 = g1 * dxidx
    r31 = ((2 * g1) * k - g_e_rho) * un
    r32 = dxidx*(g_e_rho - (3 * g1) * k) 
    r33 = g * un
    
    dEdq = fn.block_diag(r0,dxidx,r0,r21,r22,r23,r31,r32,r33)
    return dEdq

@njit    
def dEndq_1D_complex(q,dxidx):
    ''' the flux jacobian A in 1D including metric scaling. Note: can NOT handle complex values
    INPUTS: q : array of shape (nen*neq_node,nelem)
            dxidx : metrics of shape (nen,nelem) corresponding to a single xi '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::3,:]
    u = q[1::3,:]/rho
    k = (u*u)/2
    g_e_rho = g * q[2::3,:]/rho
    un = dxidx*u

    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho),dtype=np.complex128)
    r21 = (g-3) * dxidx * k 
    r22 = (3-g) * un
    r23 = g1 * dxidx
    r31 = ((2 * g1) * k - g_e_rho) * un
    r32 = dxidx*(g_e_rho - (3 * g1) * k) 
    r33 = g * un
    
    dEdq = fn.block_diag(r0,dxidx,r0,r21,r22,r23,r31,r32,r33)
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
    ''' the n-direction flux jacobian A in 2D. Note: can NOT handle complex values 
    INPUTS: q : array of shape (nen*neq_node,nelem)
            n : metrics of shape (nen,2,nelem) corresponding to a single xi '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    uv = u*v
    u2 = u*u
    v2 = v*v
    k = (u2+v2)/2
    g_e_rho = g * q[3::4,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    uvn = nx*u + ny*v

    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho))
    r21 = nx*(g1 * k - u2) - ny*uv
    r22 = (2-g)*nx*u + uvn
    r23 = -g1*nx*v + ny*u
    r24 = nx*g1
    r31 = -nx*uv + ny*(g1 * k - v2)
    r32 = nx*v - g1*ny*u
    r33 = uvn + (2-g)*ny*v
    r34 = ny*g1
    r41 = ((2*g1) * k - g_e_rho) * uvn
    r42 = nx*(g_e_rho - g1*(k + u2)) - g1*ny*uv
    r43 = ny*(g_e_rho - g1*(k + v2)) - g1*nx*uv
    r44 = g * uvn
    
    dEdq = fn.block_diag(r0,nx,ny,r0,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
    return dEdq

@njit    
def dEndq_2D_complex(q,n):
    ''' the n-direction flux jacobian A in 2D. Note: intended for use with complex step 
    INPUTS: q : array of shape (nen*neq_node,nelem)
            n : metrics of shape (nen,2,nelem) corresponding to a single xi '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    uv = u*v
    u2 = u*u
    v2 = v*v
    k = (u2+v2)/2
    g_e_rho = g * q[3::4,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    uvn = nx*u + ny*v

    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho),dtype=np.complex128)
    r21 = nx*(g1 * k - u2) - ny*uv
    r22 = (2-g)*nx*u + uvn
    r23 = -g1*nx*v + ny*u
    r24 = nx*g1
    r31 = -nx*uv + ny*(g1 * k - v2)
    r32 = nx*v - g1*ny*u
    r33 = uvn + (2-g)*ny*v
    r34 = ny*g1
    r41 = ((2*g1) * k - g_e_rho) * uvn
    r42 = nx*(g_e_rho - g1*(k + u2)) - g1*ny*uv
    r43 = ny*(g_e_rho - g1*(k + v2)) - g1*nx*uv
    r44 = g * uvn
    
    dEdq = fn.block_diag(r0,nx,ny,r0,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
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
    ''' the n-direction flux jacobian A in 3D. Note: can NOT handle complex values 
    INPUTS: q : array of shape (nen*neq_node,nelem)
            n : metrics of shape (nen,3,nelem) corresponding to a single xi '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::5,:]
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    u2 = u*u
    v2 = v*v
    w2 = w*w
    uv = u * v
    uw = u * w
    vw = v * w
    k = (u2+v2+w2)/2
    g_e_rho = g * q[4::5,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    nz = n[:,2,:]
    uvwn = nx*u + ny*v + nz*w
    
    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho))
    r21 = nx*(g1 * k - u2) - ny*uv - nz*uw
    r22 = uvwn + (nx*(2-g))*u
    r23 = -(nx*g1) * v + ny*u
    r24 = -(nx*g1) * w + nz*u
    r25 = nx*g1
    r31 = - nx*uv + ny*(g1 * k - v2) - nz*vw
    r32 = -(ny*g1) * u + nx*v
    r33 = uvwn + (ny*(2-g))*v
    r34 = -(ny*g1) * w + nz*v
    r35 = ny*g1
    r41 = - nx*uw - ny*vw + nz*(g1 * k - w2)
    r42 = -(nz*g1) * u + nx*w
    r43 = -(nz*g1) * v + ny*w
    r44 = uvwn + (nz*(2-g))*w
    r45 = nz*g1
    r51 = ((2 * g1) * k - g_e_rho)*uvwn
    r52 = nx*(g_e_rho - g1*(k + u2)) - (ny * g1 * uv) - (nz * g1 * uw)
    r53 = -(nx*g1 * uv) + ny*(g_e_rho - g1*(k + v2)) - (nz*g1 * vw)
    r54 = -(nx * g1 * uw) - (ny * g1 * vw) + nz*(g_e_rho - g1*(k + w2))
    r55 = g * uvwn
    
    dEdq = fn.block_diag(r0,nx,ny,nz,r0,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
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
    uv = u * v
    uw = u * w
    vw = v * w
    k = (u2+v2+w2)/2
    g_e_rho = g * q[4::5,:]/rho
    nx = n[:,0,:]
    ny = n[:,1,:]
    nz = n[:,2,:]
    uvwn = nx*u + ny*v + nz*w
    
    # entries of the dEdq (A) matrix
    r0 = np.zeros(np.shape(rho),dtype=np.complex128)
    r21 = nx*(g1 * k - u2) - ny*uv - nz*uw
    r22 = uvwn + (nx*(2-g))*u
    r23 = -(nx*g1) * v + ny*u
    r24 = -(nx*g1) * w + nz*u
    r25 = nx*g1
    r31 = - nx*uv + ny*(g1 * k - v2) - nz*vw
    r32 = -(ny*g1) * u + nx*v
    r33 = uvwn + (ny*(2-g))*v
    r34 = -(ny*g1) * w + nz*v
    r35 = ny*g1
    r41 = - nx*uw - ny*vw + nz*(g1 * k - w2)
    r42 = -(nz*g1) * u + nx*w
    r43 = -(nz*g1) * v + ny*w
    r44 = uvwn + (nz*(2-g))*w
    r45 = nz*g1
    r51 = ((2 * g1) * k - g_e_rho)*uvwn
    r52 = nx*(g_e_rho - g1*(k + u2)) - (ny * g1 * uv) - (nz * g1 * uw)
    r53 = -(nx*g1 * uv) + ny*(g_e_rho - g1*(k + v2)) - (nz*g1 * vw)
    r54 = -(nx * g1 * uw) - (ny * g1 * vw) + nz*(g_e_rho - g1*(k + w2))
    r55 = g * uvwn
    
    dEdq = fn.block_diag(r0,nx,ny,nz,r0,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45,r51,r52,r53,r54,r55)
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
    # Eigenvalues are the same for quasi1D Euler, but eigenvectors scaled by S to match P
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::3,:] # = rho * S if quasi1D Euler
    u = q[1::3,:]/rho # = u if quasi1D Euler
    k = 0.5*u*u # = k if quasi1D Euler
    e = q[2::3,:] # = e * S if quasi1D Euler
    p = g1*(e-rho*k) # pressure  = p * S if quasi1D Euler
    
    if val:
        Lam = np.zeros(np.shape(q))
        a = np.sqrt(g*p/rho) # sound speed, = a if quasi1D Euler
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
    ''' take a q of shape (nen*4,nelem) and performs an eigendecomposition,
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
        r11 = np.sqrt(rho*(g1/g))
        r13 = np.sqrt(rho/(2*g))
        r32 = -np.sqrt(p)   
        r0 = np.zeros(np.shape(rho))
        r21 = u*r11
        r23 = u*r13 - r32/np.sqrt(2)
        r24 = u*r13 + r32/np.sqrt(2)
        r31 = v*r11 + r32
        r33 = v*r13
        r41 = k*r11 + v*r32
        r42 = v*r32
        r43 = k*r13 - u*r32/np.sqrt(2) + p/((2*g1)*r13)
        r44 = k*r13 + u*r32/np.sqrt(2) + p/((2*g1)*r13)
        Y = fn.block_diag(r11,r0,r13,r13,r21,r0,r23,r24,r31,r32,r33,r33,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(r11,r21,r31,r41,r0,r0,r32,r42,r13,r23,r33,r43,r13,r24,r33,r44)
        else:
            YT = None 
    else:
        Y = None
        YT = None
    if inv:
        c1 = 1/np.sqrt(p)
        c2 = -c1/np.sqrt(2)
        c3 = np.sqrt((g/g1)/rho)
        r24 = np.sqrt(rho*(g1/g))/p
        r14 = -r24
        r34 = np.sqrt(g1/2) * r24
        r0 = np.zeros(np.shape(rho))
        r11 = c3 + r14*k
        r12 = -u*r14
        r13 = -v*r14
        r21 = v*c1 - c3 + r24*k
        r22 = -r24*u
        r23 = -c1 - r24*v
        r31 = k*r34 + u*c2
        r32 = -u*r34 - c2
        r33 = -v*r34
        r41 = k*r34 - u*c2
        r42 = -u*r34 + c2
        Yinv = fn.block_diag(r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r33,r34)
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
        c1 = np.sqrt(rho/g)
        c2 = 1/np.sqrt(2*g*rho)
        a = np.sqrt(g*p/rho)
        r12 = c1*np.sqrt(g1)
        r13 = c2*rho
        r21 = c1*a
        r22 = r12*u + r21
        r32 = r12*v
        r41 = r21*u
        r42 = r12*k + r41
        r23 = r13*u
        r33 = r13*(v + a)
        r34 = r13*(v - a)
        t2 = a*v
        t1 = (g/g1)*(p/rho) + k
        r43 = r13*(t1 + t2)
        r44 = r13*(t1 - t2)
        Y = fn.block_diag(r0,r12,r13,r13,r21,r22,r23,r23,r0,r32,r33,r34,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(r0,r21,r0,r41,r12,r22,r32,r42,r13,r23,r33,r43,r13,r23,r34,r44)
        else:
            YT = None 
    else:
        Y = None
        YT = None
    if inv:        
        c1 = np.sqrt(g/g1)/np.sqrt(rho)
        r24 = -1/(c1*p) 
        r14 = -r24
        r44 = np.sqrt(g1)/(np.sqrt(2)*c1*p) 
        t1 = -1/np.sqrt(p)
        r13 = r24*v
        r22 = r14*u
        r12 = -t1 - r22
        r23 = -r13
        t2 = r24*k
        r11 = t1*u -c1 - t2
        r21 = c1 + t2
        t1 = -t1/np.sqrt(2)
        r32 = -r44*u
        t2 = r44*v
        r33 =  t1 - t2
        r43 = -t1 - t2
        t2 = t1*v
        t1 = r44*k
        r31 = -t2 + t1
        r41 =  t2 + t1
        Yinv = fn.block_diag(r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r44,r41,r32,r43,r44)
        
    else:
        Yinv = None
    
    return Lam, Y, Yinv, YT  

@njit    
def dEndq_eigs_2D(q,n,val=True,vec=True,inv=True,trans=False):
    ''' take a q of shape (nen*4,nelem) and performs an eigendecomposition,
    returns the eigenvectors, eigenvalues, inverse or transpose. Use the scaling
    from Merriam 1989 / Barth 1999  to coincide with entropy variable identity.
    Note: Barth assumes n is normalized, this does not (worked out in maple doc).
    Y : columns are the n linearly independent eigenvectors of flux jacobian A
    Lam : eigenvalues of the flux jacobian (gdiag shape)
    Yinv : inverse of Y
    YT : Transpose of Y '''
    g = 1.4 # hard coded throughout
    g1 = g-1
    rho = q[::4,:]
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    k = (u*u+v*v)/2
    #e = q[3::4,:]
    #p = g1*(q[3::4,:]-rho*k) # pressure
    nx = n[:,0,:]
    ny = n[:,1,:]
    norm2 = nx*nx + ny*ny
    norm = np.sqrt(norm2)
    a2_norm2g1 = g*(q[3::4,:]/rho-k) # = (g*p)/(rho*g1)
    a = np.sqrt(a2_norm2g1*(g1*norm2)) # sound speed = norm*np.sqrt(g*p/rho)
    uvn = nx*u + ny*v
    uvn2 = ny*u -nx*v
    
    if val:
        Lam = np.zeros(np.shape(q))
        Lam[::4,:] = uvn
        Lam[1::4,:] = uvn
        Lam[2::4,:] = uvn + a
        Lam[3::4,:] = uvn - a
    else:
        Lam = None
    if vec:
        t3 = 1./np.sqrt(nx*nx + 2.*ny*ny)
        t4 = np.sign(nx-ny)
        t2 = np.sqrt(rho/g)
        r11 = t2*(norm*t3*np.sqrt(g1))
        r12 = t2*(ny*t4*t3*np.sqrt(g1))
        r13 = t2/np.sqrt(2)
        t1 = t2*a*t3
        t3 = t4/norm
        t4 = ny/norm2
        t2 = t1*ny
        r21 = r11*u - t2*t4
        r22 = r12*u + t2*t3
        t2 = t1*nx
        r31 = r11*v + t2*t4
        r32 = r12*v - t2*t3
        t2 = t1*uvn2
        r41 = r11*k - t2*t4
        r42 = r12*k + t2*t3
        t1 = a*(nx/norm2)
        r23 = r13*(u + t1)
        r24 = r13*(u - t1)
        t1 = a*(ny/norm2)
        r33 = r13*(v + t1)
        r34 = r13*(v - t1)
        t2 = a*uvn/norm2
        t1 = a2_norm2g1 + k
        r43 = r13*(t1 + t2)
        r44 = r13*(t1 - t2)
        Y = fn.block_diag(r11,r12,r13,r13,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44)
        if trans:
            YT = fn.block_diag(r11,r21,r31,r41,r12,r22,r32,r42,r13,r23,r33,r43,r13,r24,r34,r44)
        else:
            YT = None        
    else:
        Y = None
        YT = None
    if inv:
        t4 = np.sqrt(g/rho)
        r44 = t4/a2_norm2g1/np.sqrt(2)
        t1 =  t4/a/np.sqrt(2)
        t2 = r44*k
        t3 = t1*uvn
        r31 = t2 - t3
        r41 = t2 + t3
        t2 = -r44*u
        t3 = t1*nx
        r32 = t2 + t3
        r42 = t2 - t3
        t2 = -r44*v
        t3 = t1*ny
        r33 = t2 + t3
        r43 = t2 - t3
        t3 = np.sign(nx-ny)
        t4 = t4/np.sqrt(nx*nx + 2.*ny*ny)
        t1 = -t4/a2_norm2g1/np.sqrt(g1)
        r24 = t1*(ny*t3)
        r14 = t1*norm
        t2 = t4/np.sqrt(g1)
        t4 = t4/a
        t1 = t4*uvn2
        r11 = r14*k + t1*ny + t2*norm
        r21 = r24*k - t1*(norm*t3) + t2*(ny*t3)
        t1 = t4*ny
        t3 = norm*t3
        r12 = -r14*u - t1*ny
        r22 = -r24*u + t1*t3
        t1 = t4*nx
        r13 = -r14*v + t1*ny
        r23 = -r24*v - t1*t3
        Yinv = fn.block_diag(r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r44,r41,r42,r43,r44)
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
    # returns S * P if quasi1D Euler
    g = 1.4 # hard coded throughout
    rho = q[::3,:] # = rho * S if quasi1D Euler
    rhou = q[1::3,:] # = rho * u * S if quasi1D Euler
    rhou2 = rhou*rhou/rho # = rho * u^2 * S if quasi1D Euler
    e = q[2::3,:] # = e * S if quasi1D Euler
    p = (g-1)*(e-rhou2/2) # pressure = p * S if quasi1D Euler
    
    r22 = rhou2 + p
    r23 = rhou*(p+e)/rho
    r33 = g*e*e/rho - ((g-1)/4)*rhou2*rhou2/rho
    
    P = fn.block_diag(rho,rhou,e,rhou,r22,r23,e,r23,r33)
    return P

@njit    
def symmetrizer_2D(q):
    ''' take a q of shape (nen*4,nelem) and builds the symmetrizing matrix P.
    P is the symmetric positive definite matrix that symmetrizes the flux
    jacobian A=dEndq upon multiplication from the right. This is equal to the
    derivative of conservative variables with respect to entropy variables,
    or the Hessian of the entropy potential. 
    Although this does symmetrize the  '''
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
    r44 = g*e**2/rho - (g-1)*(rhou2+rhov2)**2/4/rho
    
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
    ''' take a q of shape (nen*5,nelem) and performs an eigendecomposition,
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
    p = g1*(q[4::5,:]-rho*k) # pressure
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
        fac = 1./np.sqrt(2)  
        # entries of the eigenvector (Y) matrix
        r11 = nx*a
        r12 = ny*a
        r13 = nz*a
        r21 = u*r11
        r22 = u*r12 + nz*c
        r23 = u*r13 - ny*c
        r24 = u*b + nx*c*fac
        r25 = u*b - nx*c*fac
        r31 = v*r11 - nz*c
        r32 = v*r12
        r33 = v*r13 + nx*c
        r34 = v*b + ny*c*fac
        r35 = v*b - ny*c*fac
        r41 = w*r11 + ny*c
        r42 = w*r12 - nx*c
        r43 = w*r13
        r44 = w*b + nz*c*fac
        r45 = w*b - nz*c*fac
        r51 = nx*k*a + (ny*w-nz*v)*c
        r52 = ny*k*a + (nz*u-nx*w)*c
        r53 = nz*k*a + (nx*v-ny*u)*c
        r54 = k*b + uvwn*c*fac + p/((2*g1)*b)
        r55 = k*b - uvwn*c*fac + p/((2*g1)*b)     
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
def maxeig_dExdq_1D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rhoS = q[::3,:] 
    u = q[1::3,:]/rhoS 
    e_rho = q[2::3,:]/rhoS 
    p_rho = 0.4*(e_rho-0.5*u*u) # pressure / rho, even if quasi1D Euler
    a = np.sqrt(1.4*p_rho) # sound speed, = a if quasi1D Euler
    lam = np.maximum(np.abs(u+a),np.abs(u-a))
    return lam 

@njit
def maxeig_dExdq_2D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rho = q[::4,:] 
    fac = 1/rho
    u = q[1::4,:] * fac
    v = q[2::4,:] * fac
    e_rho = q[3::4,:] * fac
    p_rho = 0.4*(e_rho-0.5*(u*u+v*v)) # pressure / rho
    a = np.sqrt(1.4*p_rho) # sound speed
    lam = np.maximum(np.abs(u+a),np.abs(u-a))
    return lam 

@njit
def maxeig_dEydq_2D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rho = q[::4,:] 
    fac = 1/rho
    u = q[1::4,:] * fac
    v = q[2::4,:] * fac
    e_rho = q[3::4,:] * fac
    p_rho = 0.4*(e_rho-0.5*(u*u+v*v)) # pressure / rho
    a = np.sqrt(1.4*p_rho) # sound speed
    lam = np.maximum(np.abs(v+a),np.abs(v-a))
    return lam 

@njit
def maxeig_dExdq_3D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rho = q[::5,:] 
    fac = 1/rho
    u = q[1::5,:] * fac
    v = q[2::5,:] * fac
    w = q[3::5,:] * fac
    e_rho = q[4::5,:] * fac
    p_rho = 0.4*(e_rho-0.5*(u*u+v*v+w*w)) # pressure / rho
    a = np.sqrt(1.4*p_rho) # sound speed
    lam = np.maximum(np.abs(u+a),np.abs(u-a))
    return lam 

@njit
def maxeig_dEydq_3D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rho = q[::5,:] 
    fac = 1/rho
    u = q[1::5,:] * fac
    v = q[2::5,:] * fac
    w = q[3::5,:] * fac
    e_rho = q[4::5,:] * fac
    p_rho = 0.4*(e_rho-0.5*(u*u+v*v+w*w)) # pressure / rho
    a = np.sqrt(1.4*p_rho) # sound speed
    lam = np.maximum(np.abs(v+a),np.abs(v-a))
    return lam 

@njit
def maxeig_dEzdq_3D(q):
    ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:]'''
    rho = q[::5,:] 
    fac = 1/rho
    u = q[1::5,:] * fac
    v = q[2::5,:] * fac
    w = q[3::5,:] * fac
    e_rho = q[4::5,:] * fac
    p_rho = 0.4*(e_rho-0.5*(u*u+v*v+w*w)) # pressure / rho
    a = np.sqrt(1.4*p_rho) # sound speed
    lam = np.maximum(np.abs(w+a),np.abs(w-a))
    return lam 

@njit
def entropy_1D(q):
    ''' return the nodal values of the entropy s(q).
     Note: this is not quite the "normal" entropy for quasi1D euler when svec \neq 1, but is a correct entropy '''
    rho = q[::3,:] # actually rho * S
    u = q[1::3,:]/rho
    e = q[2::3,:] # actually e * S
    p = 0.4*(e - 0.5*rho*u*u) # pressure * S
    s = np.log(p/(rho**1.4)) # specific entropy
    S = -rho*s/0.4
    return S

@njit
def entropy_var_1D(q):
    ''' return the nodal values of the entropy variables w(q). '''
    # Note: the same entropy variables for quasi1D euler (no svec dependence)
    g = 1.4 # hard coded throughout

    rho = q[0::3] # = rho * S if quasi1D Euler
    u = q[1::3,:]/rho
    e = q[2::3] # = e * S if quasi1D Euler
    k = rho*u*u # = rho * u^2 * S if quasi1D Euler
    p = (g-1)*(e - 0.5*k) # = p * S if quasi1D Euler
    s = np.log(p/(rho**1.4)) # specific entropy (not quite physical entropy if quasi1D Euler)
    fac = rho/p 

    # assemble_vec 
    w = np.zeros(q.shape)
    w[::3,:] = (g-s)/(g-1) - 0.5*fac*k
    w[1::3,:] = fac*u
    w[2::3,:] = -fac
    return w

@njit
def entropy_2D(q):
    ''' return the nodal values of the entropy s(q).'''
    rho = q[::4,:] 
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    e = q[3::4,:]
    p = 0.4*(e - 0.5*rho*(u*u+v*v)) # pressure
    s = np.log(p/(rho**1.4)) # specific entropy
    S = -rho*s/0.4
    return S

@njit
def entropy_var_2D(q):
    ''' return the nodal values of the entropy variables w(q). '''
    g = 1.4 # hard coded throughout

    rho = q[0::4] 
    u = q[1::4,:]/rho
    v = q[2::4,:]/rho
    e = q[3::4] 
    k = rho*(u*u + v*v)
    p = (g-1)*(e - 0.5*k) 
    s = np.log(p/(rho**1.4)) 
    fac = rho/p 

    # assemble_vec 
    w = np.zeros(q.shape)
    w[::4,:] = (g-s)/(g-1) - 0.5*fac*k
    w[1::4,:] = fac*u
    w[2::4,:] = fac*v
    w[3::4,:] = -fac
    return w

@njit
def entropy_3D(q):
    ''' return the nodal values of the entropy s(q).'''
    rho = q[::5,:] 
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    e = q[4::5,:]
    p = 0.4*(e - 0.5*rho*(u*u+v*v+w*w)) # pressure
    s = np.log(p/(rho**1.4)) # specific entropy
    S = -rho*s/0.4
    return S

@njit
def entropy_var_3D(q):
    ''' return the nodal values of the entropy variables w(q). '''
    g = 1.4 # hard coded throughout

    rho = q[0::4] 
    u = q[1::5,:]/rho
    v = q[2::5,:]/rho
    w = q[3::5,:]/rho
    e = q[3::4] 
    k = rho*(u*u + v*v + w*w)
    p = (g-1)*(e - 0.5*k) 
    s = np.log(p/(rho**1.4)) 
    fac = rho/p 

    # assemble_vec 
    w = np.zeros(q.shape)
    w[::4,:] = (g-s)/(g-1) - 0.5*fac*k
    w[1::4,:] = fac*u
    w[2::4,:] = fac*v
    w[3::4,:] = -fac
    return w

@njit 
def Ismail_Roe_flux_1D(qL,qR):
    '''
    Return the ismail roe flux given two states qL and qR where each is
    of shape (3,), and returns a numerical flux of shape (3,)
    note subroutine defined in ismail-roe appendix B for logarithmic mean
    '''
    g = 1.4 # hard coded!
    
    rhoL = qL[0]
    rhoR = qR[0]
    uL = qL[1]/rhoL
    uR = qR[1]/rhoR
    pL = (g-1)*(qL[2] - (rhoL * uL*uL)/2)
    pR = (g-1)*(qR[2] - (rhoR * uR*uR)/2)

    alphaL = np.sqrt(rhoL/pL) # z1 in paper
    alphaR = np.sqrt(rhoR/pR)
    betaL = np.sqrt(rhoL*pL) # z3 in paper
    betaR = np.sqrt(rhoR*pR)

    # logarithmic mean of alpha, or z1
    xi = alphaL/alphaR
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    alpha_ln = (alphaL+alphaR)/F

    # logarithmic mean of beta, or z3
    xi = betaL/betaR
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    beta_ln = (betaL+betaR)/F

    # arithmetic means of alpha and beta
    alpha_avg = 0.5*(alphaL+alphaR) # z1 in paper
    beta_avg = 0.5*(betaL+betaR) # z3 in paper

    # determine final quantities using averaged values
    rho_avg = alpha_avg * beta_ln
    a_avg2 = (0.5/rho_avg)*((g+1)*beta_ln/alpha_ln + (g-1)*beta_avg/alpha_avg)
    u_avg = 0.5 * (uL*alphaL + uR*alphaR) / alpha_avg
    p_avg = beta_avg / alpha_avg
    H_avg = a_avg2/(g - 1) + 0.5*u_avg**2
    rhou_avg = rho_avg*u_avg

    E = np.zeros(3)
    E[0] = rhou_avg
    E[1] = rhou_avg*u_avg + p_avg
    E[2] = rhou_avg*H_avg
    return E

@njit 
def Ismail_Roe_fluxes_2D(qL,qR):
    '''
    Return the ismail roe fluxes given two states qL and qR where each is
    of shape (4,), and returns a numerical flux of shape (4,)
    note subroutine defined in ismail-roe appendix B for logarithmic mean
    '''
    g = 1.4 # hard coded!
    
    rhoL = qL[0]
    rhoR = qR[0]
    uL = qL[1]/rhoL
    uR = qR[1]/rhoR
    vL = qL[2]/rhoL
    vR = qR[2]/rhoR
    pL = (g-1)*(qL[3] - (rhoL * (uL*uL + vL*vL))/2)
    pR = (g-1)*(qR[3] - (rhoR * (uR*uR + vR*vR))/2)

    alphaL = np.sqrt(rhoL/pL)
    alphaR = np.sqrt(rhoR/pR)
    betaL = np.sqrt(rhoL*pL)
    betaR = np.sqrt(rhoR*pR)

    # logarithmic mean of alpha, or z1
    xi = alphaL/alphaR
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    alpha_ln = (alphaL+alphaR)/F

    # logarithmic mean of beta, or z3
    xi = betaL/betaR
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    beta_ln = (betaL+betaR)/F

    # arithmetic means of alpha and beta
    alpha_avg = 0.5*(alphaL+alphaR) # z1 in paper
    beta_avg = 0.5*(betaL+betaR) # z3 in paper

    # determine final quantities using averaged values
    rho_avg = alpha_avg * beta_ln
    a_avg2 = (0.5/rho_avg)*((g+1)*beta_ln/alpha_ln + (g-1)*beta_avg/alpha_avg)
    u_avg = 0.5 * (uL*alphaL + uR*alphaR) / alpha_avg
    v_avg = 0.5 * (vL*alphaL + vR*alphaR) / alpha_avg
    p_avg = beta_avg / alpha_avg
    H_avg = a_avg2/(g - 1) + 0.5*(u_avg*u_avg + v_avg*v_avg)
    rhou_avg = rho_avg*u_avg
    rhov_avg = rho_avg*v_avg

    Fx = np.zeros(qL.shape)
    Fx[0] = rhou_avg
    Fx[1] = rhou_avg*u_avg + p_avg
    Fx[2] = rhou_avg*v_avg
    Fx[3] = rhou_avg*H_avg

    Fy = np.zeros(qL.shape)
    Fy[0] = rhov_avg
    Fy[1] = rhov_avg*u_avg
    Fy[2] = rhov_avg*v_avg + p_avg
    Fy[3] = rhov_avg*H_avg
    return Fx, Fy

@njit 
def Central_flux_1D(qL,qR):
    '''
    Return the central flux given two states qL and qR where each is
    of shape (3,), and returns a numerical flux of shape (3,)
    '''
    g = 1.4 # hard coded!
    
    # decompose_q
    fac = 1./qL[0]
    q_1L = qL[1]
    q_2L = qL[2]
    uL = q_1L * fac
    kL = uL*q_1L  
    pL = (g-1)*(q_2L - 0.5*kL)  

    fac = 1./qR[0]
    q_1R = qR[1]
    q_2R = qR[2]
    uR = q_1R * fac
    kR = uR*q_1R  
    pR = (g-1)*(q_2R - 0.5*kR)   

    #assemble_vec
    E = np.zeros(3)
    E[0] = 0.5*(q_1L + q_1R)
    E[1] = 0.5*(kL + pL + kR + pR)
    E[2] = 0.5*(uL*(q_2L + pL) + uR*(q_2R + pR))
    return E

@njit 
def Central_fluxes_2D(qL,qR):
    '''
    Return the central flux given two states qL and qR where each is
    of shape (4,), and returns a numerical flux of shape (4,)
    '''
    g = 1.4 # hard coded!
    
    # decompose_q
    fac = 1./qL[0]
    q_1L = qL[1]
    q_2L = qL[2]
    q_3L = qL[3]
    uL = q_1L * fac
    vL = q_2L * fac
    pL = (g-1)*(q_3L - 0.5*(uL*q_1L + vL*q_2L))  

    fac = 1./qR[0]
    q_1R = qR[1]
    q_2R = qR[2]
    q_3R = qR[3]
    uR = q_1R * fac
    vR = q_2R * fac
    pR = (g-1)*(q_3R - 0.5*(uR*q_1R + vR*q_2R))  

    #assemble_xvec
    Ex = np.zeros(4)
    Ex[0] = 0.5*(q_1L + q_1R)
    Ex[1] = 0.5*(q_1L*uL + pL + q_1R*uR + pR)
    Ex[2] = 0.5*(q_1L*vL + q_1R*vR)
    Ex[3] = 0.5*(uL*(q_3L + pL) + uR*(q_3R + pR))

    # assemble_yvec
    Ey = np.zeros(4)
    Ey[0] = 0.5*(q_2L + q_2R)
    Ey[1] = 0.5*(q_2L*uL + q_2R*uR)
    Ey[2] = 0.5*(q_2L*vL + pL + q_2R*vR + pR)
    Ey[3] = 0.5*(vL*(q_3L + pL) + vR*(q_3R + pR))
    return Ex, Ey

@njit 
def Ranocha_flux_1D(qL,qR):
    '''
    Return the Ranocha flux given two states qL and qR where each is
    of shape (1,), and returns a numerical flux of shape (4,)
    '''
    g = 1.4 # hard coded!
    
    # decompose_q
    q_0L = qL[0]
    q_1L = qL[1]
    q_2L = qL[2]
    uL = q_1L / q_0L
    pL = (g-1)*(q_2L - 0.5*uL*q_1L)  

    q_0R = qR[0]
    q_1R = qR[1]
    q_2R = qR[2]
    uR = q_1R / q_0R
    pR = (g-1)*(q_2R - 0.5*uR*q_1R)  

    # logarithmic mean of density
    xi = q_0L/q_0R
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    rho_ln = (q_0L+q_0R)/F

    # logarithmic mean of density / pressure
    xi = q_0L*pR/(q_0R*pL)
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    rhop_ln = (q_0L/pL+q_0R/pR)/F

    # arithmetic means
    u_avg = 0.5*(uL+uR)
    #p_avg = 0.5*(pL+pR)

    # product means
    #u2_pavg = uL*uR
    #pu_pavg = 0.5*(pL*uR + pR*uL)

    #assemble_xvec
    fac = rho_ln*(1./(rhop_ln*(g-1)) + 0.5*uL*uR)
    E = np.zeros(3)
    E[0] = rho_ln*u_avg
    E[1] = E[0]*u_avg + 0.5*(pL+pR)
    E[2] = fac*u_avg + 0.5*(pL*uR + pR*uL)
    return E

@njit 
def Ranocha_fluxes_2D(qL,qR):
    '''
    Return the Ranocha flux given two states qL and qR where each is
    of shape (4,), and returns a numerical flux of shape (4,)
    '''
    g = 1.4 # hard coded!
    
    # decompose_q
    q_0L = qL[0]
    q_1L = qL[1]
    q_2L = qL[2]
    q_3L = qL[3]
    uL = q_1L / q_0L
    vL = q_2L / q_0L
    pL = (g-1)*(q_3L - 0.5*(uL*q_1L + vL*q_2L))  

    q_0R = qR[0]
    q_1R = qR[1]
    q_2R = qR[2]
    q_3R = qR[3]
    uR = q_1R / q_0R
    vR = q_2R / q_0R
    pR = (g-1)*(q_3R - 0.5*(uR*q_1R + vR*q_2R))  

    # logarithmic mean of density
    xi = q_0L/q_0R
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    rho_ln = (q_0L+q_0R)/F

    # logarithmic mean of density / pressure
    xi = q_0L*pR/(q_0R*pL)
    zeta = (1-xi)/(1+xi)
    zeta2 = zeta**2
    if zeta2 < 0.01:
        F = 2*(1. + zeta2/3. + zeta2**2/5. + zeta2**3/7.)
    else:
        F = - np.log(xi)/zeta
    rhop_ln = (q_0L/pL+q_0R/pR)/F

    # arithmetic means
    u_avg = 0.5*(uL+uR)
    v_avg = 0.5*(vL+vR)
    p_avg = 0.5*(pL+pR)

    # product means
    #u2_pavg = uL*uR
    #v2_pavg = vL*vR
    #pu_pavg = 0.5*(pL*uR + pR*uL)
    #pv_pavg = 0.5*(pL*vR + pR*vL)

    #assemble_xvec
    fac = rho_ln*(1./(rhop_ln*(g-1)) + 0.5*(uL*uR + vL*vR))
    Ex = np.zeros(4)
    Ex[0] = rho_ln*u_avg
    Ex[1] = Ex[0]*u_avg + p_avg
    Ex[2] = Ex[0]*v_avg
    Ex[3] = fac*u_avg + 0.5*(pL*uR + pR*uL)

    # assemble_yvec
    Ey = np.zeros(4)
    Ey[0] = rho_ln*v_avg
    Ey[1] = Ex[2]
    Ey[2] = Ey[0]*v_avg + p_avg
    Ey[3] = fac*v_avg + 0.5*(pL*vR + pR*vL)
    return Ex, Ey

@njit
def dEndq_eig_abs_dq_1D(dxidx, q, qg, flux_type):
    '''
    calculates abs(A)@(q-qg) according to the implentation in diablo. Used in SATs.
    INPUTS:
    dxidx = the metric terms in the desired direction (indpendent of J), shape(nen,nelem)
    q = the flow state of the "local" node, shape (nen*3,nelem)
    qg = the flow state of the "ghost" node, shape (nen*3,nelem)
    '''

    gamma = 1.4 # hard coded throughout
    gami = 0.4
    tau = 1.
    sat_Vl = 0.025
    sat_Vn = 0.025

    rhoL = q[::3,:]
    fac = 1.0/rhoL
    uL = q[1::3,:]*fac
    phi = 0.5*(uL*uL)
    eL = q[2::3,:]
    HL = gamma*eL*fac - gami*phi

    dA = np.abs(dxidx)

    rhoR = qg[::3,:] 
    fac = 1.0/rhoR
    uR = qg[1::3,:]*fac
    phi = 0.5*(uR*uR)
    eR = qg[2::3,:]
    HR = gamma*eR*fac - gami*phi

    # Rho average
    sqL = np.sqrt(rhoL)
    sqR = np.sqrt(rhoR)
    fac = 1.0/(sqL + sqR)
    u = (sqL*uL + sqR*uR)*fac
    H = (sqL*HL + sqR*HR)*fac
    phi = 0.5*(u*u)
    a = np.sqrt(gami*(H - phi))
    Un = u*dxidx

    lambda1 = np.abs(Un + dA*a)
    lambda2 = np.abs(Un - dA*a)
    lambda3 = np.abs(Un)
    rhoA = lambda3 + dA*a

    # The structure here follows exactly Swanson & Turkel 1992 to construct |A|*dq
    # BUT this should be multiplied by -tau/2 at the end. Since E_i do not use lambda_i,
    # this is equivalent to multiplying lambda_i by -tau/2, hence we do this below.
    if (flux_type == 1):
        # Roe average flux with Hicken's fix
        lambda1 = -0.5*(np.maximum(lambda1,sat_Vn *rhoA))
        lambda2 = -0.5*(np.maximum(lambda2,sat_Vn *rhoA))
        lambda3 = -0.5*(np.maximum(lambda3,sat_Vl *rhoA))
    elif (flux_type == 2):
        # Roe average flux with entropy fix
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        d = 0.2 * lmax
        ilen,jlen = lambda1.shape
        for j in range(jlen):
            for i in range(ilen):
                dij = d[i,j]
                if (lambda1[i,j] < dij): 
                    lambda1[i,j] = (lambda1[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda2[i,j] < dij): 
                    lambda2[i,j] = (lambda2[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda3[i,j] < dij): 
                    lambda3[i,j] = (lambda3[i,j]**2 + dij*dij)/(2.*dij)
        lambda1 = (-0.5*tau)*lambda1
        lambda2 = (-0.5*tau)*lambda2
        lambda3 = (-0.5*tau)*lambda3
    elif (flux_type == 3):
        # local Lax-Friedrichs flux
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        lambda1 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda2 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda3 = -0.5*(tau*np.maximum(lmax,sat_Vl *rhoA))
    elif (flux_type == 4):
        # Alex Bercik's local Lax-Friedrichs flux (more simple & dissipative)
        lambda1 = np.abs(u) + a #max possible eigenvalue
        rhoA = np.abs(lambda1*dxidx)
        fi = - 0.5*tau*fn.repeat_neq_gv(rhoA,3)*(q - qg)
        return fi
    else:
        return np.zeros(q.shape)

    dq1 = rhoL - rhoR
    dq2 = uL - uR
    dq3 = eL - eR

    # diagonal matrix multiply
    fi = np.zeros(q.shape)
    fi[::3,:] = lambda3*dq1
    fi[1::3,:] = lambda3*dq2
    fi[2::3,:] = lambda3*dq3

    # get E1*dq
    E1dq = np.zeros(q.shape)
    fac = phi*dq1 - u*dq2 + dq3
    E1dq[::3,:] = fac
    E1dq[1::3,:] = fac*u
    E1dq[2::3,:] = fac*H

    # get E2*dq
    E2dq = np.zeros(q.shape)
    fac2 = -Un*dq1 + dxidx*dq2
    E2dq[1::3,:] = fac2*dxidx
    E2dq[2::3,:] = fac2*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 + lambda2) - lambda3,3)
    tmp2 = fn.repeat_neq_gv(gami/(a*a),3)
    tmp3 = fn.repeat_neq_gv(1.0/(dA*dA),3)
    fi = fi + tmp1*(tmp2*E1dq + tmp3*E2dq)

    # get E3*dq
    E1dq[::3,:] = fac2
    E1dq[1::3,:] = fac2*u
    E1dq[2::3,:] = fac2*H

    # get E4*dq
    E2dq[::3,:] = 0.0
    E2dq[1::3,:] = fac*dxidx
    E2dq[2::3,:] = fac*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 - lambda2)/(dA*a),3)
    fi = fi + tmp1*(E1dq + gami*E2dq)
    return fi

@njit
def dEndq_eig_abs_dq_2D(dxidx, q, qg, flux_type):
    '''
    calculates abs(An)@(q-qg) according to the implentation in diablo. Used in SATs.
    INPUTS:
    dxidx = the metric terms in the desired direction (indpendent of J), shape(nen,2,nelem)
            these are the dxi_x, dxi_dy, where xi is a fixed computational direction
    q = the flow state of the "local" node, shape (nen*4,nelem)
    qg = the flow state of the "ghost" node, shape (nen*4,nelem)
    '''

    gamma = 1.4 # hard coded throughout
    gami = 0.4
    tau = 1.
    sat_Vl = 0.025
    sat_Vn = 0.025

    rhoL = q[::4,:]
    fac = 1.0/rhoL
    uL = q[1::4,:]*fac
    vL = q[2::4,:]*fac
    phi = 0.5*(uL*uL + vL*vL)
    eL = q[4::4,:]
    HL = gamma*eL*fac - gami*phi

    nx = dxidx[:,0,:]
    ny = dxidx[:,1,:]
    dA = np.sqrt(nx*nx + ny*ny)

    rhoR = qg[::4,:] 
    fac = 1.0/rhoR
    uR = qg[1::4,:]*fac
    vR = qg[2::4,:]*fac
    phi = 0.5*(uR*uR + vR*vR)
    eR = qg[3::4,:]
    HR = gamma*eR*fac - gami*phi

    # Rho average
    sqL = np.sqrt(rhoL)
    sqR = np.sqrt(rhoR)
    fac = 1.0/(sqL + sqR)
    u = (sqL*uL + sqR*uR)*fac
    v = (sqL*vL + sqR*vR)*fac
    H = (sqL*HL + sqR*HR)*fac
    phi = 0.5*(u*u + v*v)
    a = np.sqrt(gami*(H - phi))
    Un = u*nx + v*ny

    lambda1 = np.abs(Un + dA*a)
    lambda2 = np.abs(Un - dA*a)
    lambda3 = np.abs(Un)
    rhoA = lambda3 + dA*a

    # The structure here follows exactly Swanson & Turkel 1992 to construct |A|*dq
    # BUT this should be multiplied by -tau/2 at the end. Since E_i do not use lambda_i,
    # this is equivalent to multiplying lambda_i by -tau/2, hence we do this below.
    if (flux_type == 1):
        # Roe average flux with Hicken's fix
        lambda1 = -0.5*(np.maximum(lambda1,sat_Vn *rhoA))
        lambda2 = -0.5*(np.maximum(lambda2,sat_Vn *rhoA))
        lambda3 = -0.5*(np.maximum(lambda3,sat_Vl *rhoA))
    elif (flux_type == 2):
        # Roe average flux with entropy fix
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        d = 0.2 * lmax
        ilen,jlen = lambda1.shape
        for j in range(jlen):
            for i in range(ilen):
                dij = d[i,j]
                if (lambda1[i,j] < dij): 
                    lambda1[i,j] = (lambda1[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda2[i,j] < dij): 
                    lambda2[i,j] = (lambda2[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda3[i,j] < dij): 
                    lambda3[i,j] = (lambda3[i,j]**2 + dij*dij)/(2.*dij)
        lambda1 = (-0.5*tau)*lambda1
        lambda2 = (-0.5*tau)*lambda2
        lambda3 = (-0.5*tau)*lambda3
    elif (flux_type == 3):
        # local Lax-Friedrichs flux
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        lambda1 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda2 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda3 = -0.5*(tau*np.maximum(lmax,sat_Vl *rhoA))
    elif (flux_type == 4):
        # Alex Bercik's local Lax-Friedrichs flux (more simple & dissipative)
        lambda1 = np.abs(u) + a #max eigenvalue in x
        lambda2 = np.abs(v) + a #max eigenvalue in y
        rhoA = np.abs(lambda1*nx + lambda2*ny)
        # Note: definition of rhoA is very slightly different to above (L1 norm isntead of L2 on n_i)
        fi = - 0.5*tau*fn.repeat_neq_gv(rhoA,5)*(q - qg)
        return fi
    elif (flux_type == 5):
        # Roe average flux with entropy fix like from Zingg textbook
        # this is directly copied from the function abs_Roe_fix_2D
        ilen,jlen = lambda1.shape
        d = 0.1 * np.maximum(lambda1,lambda2) # Like Zingg textbook, but smooth cutoff
        for j in range(jlen):
            for i in range(ilen):
                dij = d[i,j]

    else:
        return np.zeros(q.shape)

    dq1 = rhoL - rhoR
    dq2 = uL - uR
    dq3 = vL - vR
    dq4 = eL - eR

    # diagonal matrix multiply
    fi = np.zeros(q.shape)
    fi[::4,:] = lambda3*dq1
    fi[1::4,:] = lambda3*dq2
    fi[2::4,:] = lambda3*dq3
    fi[3::4,:] = lambda3*dq4

    # get E1*dq
    E1dq = np.zeros(q.shape)
    fac = phi*dq1 - u*dq2 - v*dq3 + dq4
    E1dq[::4,:] = fac
    E1dq[1::4,:] = fac*u
    E1dq[2::4,:] = fac*v
    E1dq[3::4,:] = fac*H

    # get E2*dq
    E2dq = np.zeros(q.shape)
    fac2 = -Un*dq1 + nx*dq2 + ny*dq3
    E2dq[1::4,:] = fac2*nx
    E2dq[2::4,:] = fac2*ny
    E2dq[3::4,:] = fac2*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 + lambda2) - lambda3,4)
    tmp2 = fn.repeat_neq_gv(gami/(a*a),4)
    tmp3 = fn.repeat_neq_gv(1.0/(dA*dA),4)
    fi = fi + tmp1*(tmp2*E1dq + tmp3*E2dq)

    # get E3*dq
    E1dq[::4,:] = fac2
    E1dq[1::4,:] = fac2*u
    E1dq[2::4,:] = fac2*v
    E1dq[3::4,:] = fac2*H

    # get E4*dq
    E2dq[::4,:] = 0.0
    E2dq[1::4,:] = fac*nx
    E2dq[2::4,:] = fac*ny
    E2dq[3::4,:] = fac*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 - lambda2)/(dA*a),4)
    fi = fi + tmp1*(E1dq + gami*E2dq)
    return fi

@njit
def dEndq_eig_abs_dq_3D(dxidx, q, qg, flux_type):
    '''
    calculates abs(An)@(q-qg) according to the implentation in diablo. Used in SATs.
    INPUTS:
    dxidx = the metric terms in the desired direction (indpendent of J), shape(nen,3,nelem)
            these are the dxi_x, dxi_dy, dxi_dz, where xi is a fixed computational direction
    q = the flow state of the "local" node, shape (nen*5,nelem)
    qg = the flow state of the "ghost" node, shape (nen*5,nelem)
    '''

    gamma = 1.4 # hard coded throughout
    gami = 0.4
    tau = 1.
    sat_Vl = 0.025
    sat_Vn = 0.025

    rhoL = q[::5,:]
    fac = 1.0/rhoL
    uL = q[1::5,:]*fac
    vL = q[2::5,:]*fac
    wL = q[3::5,:]*fac
    phi = 0.5*(uL*uL + vL*vL + wL*wL)
    eL = q[4::5,:]
    HL = gamma*eL*fac - gami*phi

    nx = dxidx[:,0,:]
    ny = dxidx[:,1,:]
    nz = dxidx[:,2,:]
    dA = np.sqrt(nx*nx + ny*ny + nz*nz)

    rhoR = qg[::5,:] 
    fac = 1.0/rhoR
    uR = qg[1::5,:]*fac
    vR = qg[2::5,:]*fac
    wR = qg[3::5,:]*fac
    phi = 0.5*(uR*uR + vR*vR + wR*wR)
    eR = qg[4::5,:]
    HR = gamma*eR*fac - gami*phi

    # Rho average
    sqL = np.sqrt(rhoL)
    sqR = np.sqrt(rhoR)
    fac = 1.0/(sqL + sqR)
    u = (sqL*uL + sqR*uR)*fac
    v = (sqL*vL + sqR*vR)*fac
    w = (sqL*wL + sqR*wR)*fac
    H = (sqL*HL + sqR*HR)*fac
    phi = 0.5*(u*u + v*v + w*w)
    a = np.sqrt(gami*(H - phi))
    Un = u*nx + v*ny + w*nz

    lambda1 = np.abs(Un + dA*a)
    lambda2 = np.abs(Un - dA*a)
    lambda3 = np.abs(Un)
    rhoA = lambda3 + dA*a

    # The structure here follows exactly Swanson & Turkel 1992 to construct |A|*dq
    # BUT this should be multiplied by -tau/2 at the end. Since E_i do not use lambda_i,
    # this is equivalent to multiplying lambda_i by -tau/2, hence we do this below.
    if (flux_type == 1):
        # Roe average flux with Hicken's fix
        lambda1 = -0.5*(np.maximum(lambda1,sat_Vn *rhoA))
        lambda2 = -0.5*(np.maximum(lambda2,sat_Vn *rhoA))
        lambda3 = -0.5*(np.maximum(lambda3,sat_Vl *rhoA))
    elif (flux_type == 2):
        # Roe average flux with entropy fix
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        d = 0.2 * lmax
        ilen,jlen = lambda1.shape
        for j in range(jlen):
            for i in range(ilen):
                dij = d[i,j]
                if (lambda1[i,j] < dij): 
                    lambda1[i,j] = (lambda1[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda2[i,j] < dij): 
                    lambda2[i,j] = (lambda2[i,j]**2 + dij*dij)/(2.*dij)
                if (lambda3[i,j] < dij): 
                    lambda3[i,j] = (lambda3[i,j]**2 + dij*dij)/(2.*dij)
        lambda1 = (-0.5*tau)*lambda1
        lambda2 = (-0.5*tau)*lambda2
        lambda3 = (-0.5*tau)*lambda3
    elif (flux_type == 3):
        # local Lax-Friedrichs flux
        lmax = np.maximum(np.maximum(lambda1,lambda2),lambda3)
        lambda1 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda2 = -0.5*(tau*np.maximum(lmax,sat_Vn *rhoA))
        lambda3 = -0.5*(tau*np.maximum(lmax,sat_Vl *rhoA))
    elif (flux_type == 4):
        # Alex Bercik's local Lax-Friedrichs flux (more simple & dissipative)
        lambda1 = np.abs(u) + a #max eigenvalue in x
        lambda2 = np.abs(v) + a #max eigenvalue in y
        lambda3 = np.abs(w) + a #max eigenvalue in z
        rhoA = np.abs(lambda1*nx + lambda2*ny + lambda3*nz)
        # Note: definition of rhoA is very slightly different to above (L1 norm isntead of L2 on n_i)
        #if (skew_sym):
        fi = - 0.5*tau*fn.repeat_neq_gv(rhoA,5)*(q - qg)
        #else:
        #    fi(:) = sgn*d0_5*(eulerFlux(dxidx,q) + eulerFlux(dxidx,qg)) &
        #        - d0_5*tau*rhoA*(q(:) - qg(:))
        return fi
    else:
        return np.zeros(q.shape)

    dq1 = rhoL - rhoR
    dq2 = uL - uR
    dq3 = vL - vR
    dq4 = wL - wR
    dq5 = eL - eR

    # diagonal matrix multiply
    fi = np.zeros(q.shape)
    fi[::5,:] = lambda3*dq1
    fi[1::5,:] = lambda3*dq2
    fi[2::5,:] = lambda3*dq3
    fi[3::5,:] = lambda3*dq4
    fi[4::5,:] = lambda3*dq5

    # get E1*dq
    E1dq = np.zeros(q.shape)
    fac = phi*dq1 - u*dq2 - v*dq3 - w*dq4 + dq5
    E1dq[::5,:] = fac
    E1dq[1::5,:] = fac*u
    E1dq[2::5,:] = fac*v
    E1dq[3::5,:] = fac*w
    E1dq[4::5,:] = fac*H

    # get E2*dq
    E2dq = np.zeros(q.shape)
    fac2 = -Un*dq1 + nx*dq2 + ny*dq3 + nz*dq4
    E2dq[1::5,:] = fac2*nx
    E2dq[2::5,:] = fac2*ny
    E2dq[3::5,:] = fac2*nz
    E2dq[4::5,:] = fac2*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 + lambda2) - lambda3,5)
    tmp2 = fn.repeat_neq_gv(gami/(a*a),5)
    tmp3 = fn.repeat_neq_gv(1.0/(dA*dA),5)
    fi = fi + tmp1*(tmp2*E1dq + tmp3*E2dq)

    # get E3*dq
    E1dq[::5,:] = fac2
    E1dq[1::5,:] = fac2*u
    E1dq[2::5,:] = fac2*v
    E1dq[3::5,:] = fac2*w
    E1dq[4::5,:] = fac2*H

    # get E4*dq
    E2dq[::5,:] = 0.0
    E2dq[1::5,:] = fac*nx
    E2dq[2::5,:] = fac*ny
    E2dq[3::5,:] = fac*nz
    E2dq[4::5,:] = fac*Un

    # add to fi
    tmp1 = fn.repeat_neq_gv(0.5*(lambda1 - lambda2)/(dA*a),5)
    fi = fi + tmp1*(E1dq + gami*E2dq)
    #if (.not. skew_sym) then
    #    !-- This is already taken care of for Skew Symmetric Fluxes
    #    fi(:) = fi(:) + sgn*d0_5*(eulerFlux(dxidx,q) + eulerFlux(dxidx,qg))
    #end if
    return fi





if __name__ == "__main__":
    
    nelem = 2
    nen = 3
    
    q1D = np.random.rand(nen*3,nelem) + 10
    q1D[2::3,:] *= 1000 # fix e to ensure pressure is positive

    n1D = np.random.rand(nen,nelem)
    
    Lam1D, Y1D, Yinv1D, YT1D = dEdq_eigs_1D(q1D,val=True,vec=True,inv=True,trans=True)
    An1D = dEndq_1D(q1D,n1D)
    A1D = dExdq_1D(q1D)
    P1D = symmetrizer_1D(q1D)
    AP1D = fn.gm_gm(A1D,P1D)
    from scipy.linalg import block_diag
    xblocks = []
    for node in range(nen):
        xblocks.append(np.ones((3,3))*n1D[node,0])
    Nx = block_diag(*xblocks)
    An21D = A1D[:,:,0]*Nx

    q = q1D[:3,0]
    F = calcEx_1D(q1D[:3,:1])[:,0]
    F_IsmailRoe = Ismail_Roe_flux_1D(q,q)
    F_Central = Central_flux_1D(q,q)
    F_Ranocha = Ranocha_flux_1D(q,q)
    
    print('---- Testing 1D functions (all should be zero) ----')
    print('An = nx*Ax: ', np.max(abs(An1D[:,:,0]-An21D)))
    print('eigenvector inverse: ', np.max(abs(np.linalg.inv(Y1D[:,:,0])-Yinv1D[:,:,0])))
    print('eigenvector transpose: ', np.max(abs(Y1D[:,:,0].T - YT1D[:,:,0])))
    print('eigendecomposition: ', np.max(abs(A1D - fn.gm_gm(Y1D,fn.gdiag_gm(Lam1D,Yinv1D)))))
    print('A @ P symmetrizer: ', np.max(abs(AP1D[:,:,0] - AP1D[:,:,0].T)))
    print('P = Y@Y.T symmetrizer: ', np.max(abs(P1D - fn.gm_gm(Y1D,YT1D))))
    print('----')
    print('Consistency of Ismail Roe flux: ', np.max(abs(F-F_IsmailRoe)))
    print('Consistency of Central flux: ', np.max(abs(F-F_Central)))
    print('Consistency of Ranocha flux: ', np.max(abs(F-F_Ranocha)))
    print('')
    
    q2D = np.random.rand(nen*4,nelem) + 10
    q2D[3::4,:] *= 1000 # fix e to ensure pressure is positive
    
    n2D = np.random.rand(nen,2,nelem)
    norm = np.sqrt(n2D[:,0,:]**2 + n2D[:,1,:]**2)
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
    P22D = fn.gm_gm(Y2D,YT2D)
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

    q = q2D[:4,0]
    Fx, Fy = calcExEy_2D(q2D[:4,:1])
    Fx, Fy = Fx[:,0], Fy[:,0]
    Fx_IsmailRoe, Fy_IsmailRoe = Ismail_Roe_fluxes_2D(q,q)
    Fx_Central, Fy_Central = Central_fluxes_2D(q,q)
    Fx_Ranocha, Fy_Ranocha = Ranocha_fluxes_2D(q,q)
    
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
    print('P = Y@Y.T symmetrizer:', np.max(abs(P2D-P22D))) 
    print('An @ P symmetry: ', np.max(abs(AP2D[:,:,0] - AP2D[:,:,0].T)))
    print('----')
    print('Consistency of Ismail Roe flux: ', np.max(abs(np.array([Fx-Fx_IsmailRoe,Fy-Fy_IsmailRoe]))))
    print('Consistency of Central flux: ', np.max(abs(np.array([Fx-Fx_Central,Fy-Fy_Central]))))
    print('Consistency of Ranocha flux: ', np.max(abs(np.array([Fx-Fx_Ranocha,Fy-Fy_Ranocha]))))
    print('')

    print('THERE ARE SOME ERRORS in 3D - MUST REDO SCALING AS FOR 2D')
    """
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
    print('THERE ARE SOME ERRORS: MUST REDO SCALING AS FOR 2D')
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
    """