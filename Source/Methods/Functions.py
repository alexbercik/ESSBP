#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:26:34 2020

@author: bercik
"""

from numba import jit, njit, literal_unroll
import numpy as np
from contextlib import contextmanager
import sys, os
#import scipy.sparse as sp

# The useful functions are defined first, the others are shoved to the bottom

# TODO: This is the function to speed up (is approximately 40% of total code runtime)
@njit
def gm_gv(A,b):
    '''
    Equivalent to np.einsum('ijk,jk->ik',A,b) where A is a 3-tensor of shape
    (nen1,nen2,nelem) and b is a 2-tensor of shape (nen2,nelem). This can be 
    thought of as a global matrix @ global vector.
    Faster than A[:,:,e]@b[:,e] because A[:,:,e] is not a contigous array

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2,nelem)
    b : numpy array of shape (nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nelem)
    '''
    nen1,nen2,nelem = np.shape(A)
    nen2b,nelemb = np.shape(b)
    if nen2!=nen2b:
        raise Exception('array shapes do not match')    
    if nelem!=nelemb:
        raise Exception('element shapes do not match')   
    c = np.zeros((nen1,nelem),dtype=b.dtype)
    for i in range(nen1):
        for j in range(nen2):
            for e in range(nelem):
                c[i,e] += A[i,j,e]*b[j,e]
    return c

@njit
def gm_gm(A,B):
    '''
    Equivalent to np.einsum('ijk,jlk->ilk',A,B) where A is a 3-tensor of shape
    (nen1,nen2,nelem) and B is a 3-tensor of shape (nen2,nen3,nelem). This can be 
    thought of as a global matrix @ global matrix.

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2,nelem)
    B : numpy array of shape (nen2,nen3,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen3,nelem)
    '''
    nen1,nen2,nelem = np.shape(A)
    nen2b,nen3,nelemb = np.shape(B)
    if nen2!=nen2b:
        raise Exception('array shapes do not match')    
    if nelem!=nelemb:
        raise Exception('element shapes do not match')   
    c = np.zeros((nen1,nen3,nelem))
    for i in range(nen1):
        for j in range(nen2):
            for l in range(nen3):
                for e in range(nelem):
                    c[i,l,e] += A[i,j,e]*B[j,l,e]
    return c  

@njit
def gm_lm(A,B):
    '''
    Equivalent to np.einsum('ijk,jl->ilk',A,B) where A is a 3-tensor of shape
    (nen1,nen2,nelem) and B is a 2-tensor of shape (nen2,nen3). This can be 
    thought of as a global matrix @ local matrix.

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2,nelem)
    B : numpy array of shape (nen2,nen3)

    Returns
    -------
    c : numpy array of shape (nen1,nen3,nelem)
    '''
    nen1,nen2,nelem = np.shape(A)
    nen2b,nen3 = np.shape(B)
    if nen2!=nen2b:
        raise Exception('shapes do not match')
    c = np.zeros((nen1,nen3,nelem))
    for i in range(nen1):
        for j in range(nen2):
            for l in range(nen3):
                for e in range(nelem):
                    c[i,l,e] += A[i,j,e]*B[j,l]
    return c

@njit
def gm_lv(A,b):
    '''
    Equivalent to np.einsum('ijk,j->ik',A,b) where A is a 3-tensor of shape
    (nen1,nen2,nelem) and b is a vector of shape (nen2). This can be 
    thought of as a global matrix @ local vector.

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2,nelem)
    b : numpy array of shape (nen2)

    Returns
    -------
    c : numpy array of shape (nen1,nen3,nelem)
    '''
    nen1,nen2,nelem = np.shape(A)
    #nen2b = np.shape(b) # throws an error for some reason
    nen2b = len(b)
    if nen2!=nen2b:
        raise Exception('shapes do not match')
    c = np.zeros((nen1,nelem))
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen1):
                c[i,e] += A[i,j,e] * b[j]
    return c

@njit
def lm_gm(A,B):
    '''
    NOTE: NOT equivalent to A @ B 
    That returns the elemntwise transpose of the desired result.
    Equivalent to np.einsum('ij,jlk->ilk',A,B) where A is a 2-tensor of shape
    (nen1,nen2) and B is a 3-tensor of shape (nen2,nen3,nelem). This can be 
    thought of as a local matrix @ global matrix.

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2)
    B : numpy array of shape (nen2,nen3,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen3,nelem)
    '''
    nen1,nen2 = np.shape(A)
    nen2b,nen3,nelem = np.shape(B)
    if nen2!=nen2b:
        raise Exception('shapes do not match')
    c = np.zeros((nen1,nen3,nelem))
    for i in range(nen1):
        for j in range(nen2):
            for l in range(nen3):
                for e in range(nelem):
                    c[i,l,e] += A[i,j]*B[j,l,e]
    return c

@njit
def gs_lm(A,B):
    '''
    Takes a global scalar of shape either (nelem,) or (1,nelem) and a local
    matrix of shape (nen1,nen2) and returns a global matrix of shape 
    (nen1,nen2,nelem) where each matrix is multiplied by the corresponding scalar

    Parameters
    ----------
    A : numpy array of shape (nelem,) or (1,nelem)
    B : numpy array of shape (nen1,nen2)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    '''
    nelem = A.size
    nen1,nen2 = np.shape(B)
    c = np.zeros((nen1,nen2,nelem))
    if A.ndim == 1:
        for e in range(nelem):
            c[:,:,e] = A[e]*B
    elif A.ndim == 2:
        for e in range(nelem):
            c[:,:,e] = A[0,e]*B
    else: raise Exception('Scalar shape not understood. Should be (nelem,) or (1,nelem)')

    return c

@njit
def gv_lm(A,B):
    '''
    Takes a global vector of shape (nen1,nelem) and a local matrix of shape (nen1,nen2) 
    and returns a global matrix of shape (nen2,nelem)

    Parameters
    ----------
    A : numpy array of shape (nen1,nelem)
    B : numpy array of shape (nen1,nen2)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    '''
    nen1,nelem = np.shape(A)
    nen1b,nen2 = np.shape(B)
    if nen1!=nen1b:
        raise Exception('shapes do not match')
    c = np.zeros((nen2,nelem))
    for e in range(nelem):
            c[:,e] = A[:,e] @ B

    return c

@njit
def gdiag_lm(H,D):
    '''
    Takes a global array of shape (nen1,nelem) that simulates a global diagonal
    matrix of shape (nen1,nen1,nelem), and a local matrix of shape (nen1,nen2) 
    and returns a global matrix of shape (nen1,nen2,nelem), i.e. H @ D

    Parameters
    ----------
    H : numpy array of shape (nen1,nelem)
    D : numpy array of shape (nen1,nen2)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen1,nelem = np.shape(H)
    nen1b,nen2 = np.shape(D)
    if nen1!=nen1b:
        raise Exception('shapes do not match')
    c = np.zeros((nen1,nen2,nelem)) 
    for e in range(nelem):
        c[:,:,e] = (D.T * H[:,e]).T
    return c

@njit
def lm_gdiag(D,H):
    '''
    Takes a a local matrix of shape (nen1,nen2) and a global array of shape 
    (nen2,nelem) that simulates a global diagonal matrix of shape (nen2,nen2,nelem), 
    and returns a global matrix of shape (nen1,nen2,nelem), i.e. D @ H

    Parameters
    ----------
    D : numpy array of shape (nen1,nen2)
    H : numpy array of shape (nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen2b,nelem = np.shape(H)
    nen1,nen2 = np.shape(D)
    if nen2!=nen2b:
        raise Exception('shapes do not match')
    c = np.zeros((nen1,nen2,nelem)) 
    for e in range(nelem):
        c[:,:,e] = D * H[:,e]
    return c

@njit
def gdiag_gm(H,D):
    '''
    Takes a global array of shape (nen1,nelem) that simulates a global diagonal
    matrix of shape (nen1,nen1,nelem), and a global matrix of shape (nen1,nen2,nelem) 
    and returns a global matrix of shape (nen1,nen2,nelem), i.e. H @ D

    Parameters
    ----------
    H : numpy array of shape (nen1,nelem)
    D : numpy array of shape (nen1,nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen1,nelem = np.shape(H)
    nen1b,nen2,nelemb = np.shape(D)
    if nen1!=nen1b:
        raise Exception('array shapes do not match')    
    if nelem!=nelemb:
        raise Exception('element shapes do not match')   
    c = np.zeros((nen1,nen2,nelem))
    for e in range(nelem):
        c[:,:,e] = (D[:,:,e].T * H[:,e]).T
    return c

@njit
def gdiag_gv(H,q):
    '''
    NOTE: Faster to directly use H * q
    Takes a global array of shape (nen,nelem) that simulates a global diagonal
    matrix of shape (nen,nen,nelem), and a global vector of shape (nen,nelem) 
    and returns a global vector of shape (nen,nelem), i.e. H @ q

    Parameters
    ----------
    H : numpy array of shape (nen,nelem)
    D : numpy array of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen,nelem = np.shape(H)
    nenb,nelemb = np.shape(q)
    if nen!=nenb:
        raise Exception('array shapes do not match')    
    if nelem!=nelemb:
        raise Exception('element shapes do not match')   
    c = H * q
    return c

@njit
def gm_gv_colmultiply(A,q):
    '''
    Takes a global matrix of shape (nen1,nen2,nelem) and a global vector of
    shape (nen2,nelem) and returns a global matrix of shape (nen1,nen2,nelem),
    where the columns of the matrix are multiplied by the entries of the 
    vector i.e. res_ij = A_ij q_j

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2,nelem)
    q : numpy array of shape (nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen1,nen2,nelem = np.shape(A)
    nen2b,nelemb = np.shape(q)
    if nen2!=nen2b:
        raise Exception('array shapes do not match')    
    if nelem!=nelemb:
        raise Exception('element shapes do not match')   
    c = np.zeros((nen1,nen2,nelem))
    for e in range(nelem):
        c[:,:,e] = A[:,:,e] * q[:,e]
    return c

@njit
def diag(q):
    '''
    Takes a 2-dim numpy array q of shape (nen,nelem) and returns a 3-dim
    array of shape (nen,nen,nelem) with the (nen) entries of q along the 
    (nen,nen) diagonals. Can be thought of as local diagonal matrices of q

    Parameters
    ----------
    q : numpy array of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen,nen,nelem)
    '''
    i,k = np.shape(q)
    c=np.empty((i,i,k))
    for e in range(k):
        c[:,:,e] = np.diag(q[:,e])
    return c


def check_q_shape(q):
    '''
    Does nothing (q is by default in structured form)
    TODO: Generalize this if we move to more general meshes
    '''
    assert(q.ndim==2),'ERROR: q is the wrong shape.'
    return q

@njit
def block_diag(*entries):
    '''
    Takes neq_node^2 2-dim numpy arrays q of shape (nen,nelem) and returns a 3-dim
    array of shape (nen*neq_node,nen*neq_node,nelem) where each local elem matrix
    is block diagonal with (neq_node,neq_node) 2-dim array blocks with entries
    given from the *entries arrays in order from top left, left to right, then 
    top to bottom (normal reading direction)

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    TODO: Speed up?
    '''
    neq_node = int(np.sqrt(len(entries)))
    nen,nelem = entries[0].shape
    dtype = entries[0].dtype
    blocks = np.zeros((nen,neq_node,neq_node,nelem),dtype=dtype)
    mat = np.zeros((nen*neq_node,nen*neq_node,nelem),dtype=dtype)

    idx = 0
    for entry in literal_unroll(entries): # add literal_unroll for heterogeneous tuple types
        row = idx // neq_node
        col = idx % neq_node
        blocks[:,row,col,:] = entry
        idx += 1
    for i in range(nen):
        a = i*neq_node
        b = (i+1)*neq_node
        mat[a:b,a:b,:] = blocks[i,:,:,:]    
    return mat

@njit
def abs_eig_mat(mat):
    '''
    Given a 3d array in the shape (nen*neq_node,nen*neq_node,nelem), return
    a 3d array in the same shape where the matrices in each element are now
    absoluted through it's eigenvalues. That is, if A is one such matrix, 
    and it has eigenvalues L and right eigenvectors X, then this returns
    X @ abs(L) @ inv(X
    note: inv(X) = X.T if eigenvectors X orthogonal, i.e. if A symmetric real (sym hyperbolic)

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nodes,_,nelem = mat.shape
    dtype=mat.dtype
    mat_abs = np.zeros((nodes,nodes,nelem),dtype=dtype)
    for elem in range(nelem):
        eig_val, eig_vec = np.linalg.eig(mat[:,:,elem])
        mat_abs[:,:,elem] = eig_vec @ np.diag(np.abs(eig_val)).astype(dtype) @ np.linalg.inv(eig_vec)
    return mat_abs

@njit
def spec_rad(mat,neq):
    '''
    Given a 3d block diagonal array in the shape (nen*neq,nen*neq,nelem)
    with blocks of size neq*neq, return a 2d array in the shape (nen,nelem) 
    , i.e. like a global scalar, where the values are the spectral radius of each 

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nen_neq, _, nelem = mat.shape
    nen = int(nen_neq / neq)
    rho = np.zeros((nen,nelem))
    for e in range(nelem):
        for i in range(nen):
            A = mat[i*neq:(i+1)*neq,i*neq:(i+1)*neq,e]
            eigs = np.linalg.eigvals(A)
            rho[i,e] = np.max(np.abs(eigs))
    return rho

@njit
def gm_triblock_flat(blockL,blockM,blockR):
    '''
    Takes 3 global matrices, blockL and blockR arrays of shape (nen,nen,nelem-1) and 
    blockM of shape (nen,nen,nelem) returns a 2d array of shape (nen*nelem,nen*nelem)
    where the nelem (nen,nen) blocks are along the diagonal, blockL blocks are
    to the left of the main diagonal, and blockR blocks are to the right.
    
    actually faster than using slicing! slower function below in commented section

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nen,nenb,nelem = blockM.shape
    nenc,nend,nelemb = blockL.shape
    nene,nenf,nelemc = blockR.shape
    if (nenb!=nen or nenc!=nen or nend!=nen or nene!=nen or nenf!=nen):
        raise Exception('block shapes do not match')    
    if (nelemb!=nelem-1 or nelemc!=nelem-1):
        raise Exception('number of blocks do not match')    
        
    mat = np.zeros((nen*nelem,nen*nelem))
    for e in range(nelem-1):
        for i in range(nen):
            for j in range(nen):
                mat[nen*e+i,nen*e+j] = blockM[i,j,e]
                mat[nen*e+i+nen,nen*e+j] = blockL[i,j,e]
                mat[nen*e+i,nen*e+j+nen] = blockR[i,j,e]
    e = nelem-1
    for i in range(nen):
        for j in range(nen):
            mat[nen*e+i,nen*e+j] = blockM[i,j,e]
            
    return mat
                

@njit
def gm_triblock_flat_periodic(blockL,blockM,blockR):
    '''
    Takes 3 global matrices of shape (nen,nen,nelem) and returns a 2d array 
    of shape (nen*nelem,nen*nelem) where the nelem blockM (nen,nen) blocks are 
    along the diagonal, blockL blocks are to the left of the main diagonal, 
    and blockR blocks are to the right. The first block of blockL is sent to 
    the top right while the last block of blockR is sent to the bottom left.

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nen,nenb,nelem = blockM.shape
    nenc,nend,nelemb = blockL.shape
    nene,nenf,nelemc = blockR.shape
    if (nenb!=nen or nenc!=nen or nend!=nen or nene!=nen or nenf!=nen):
        raise Exception('block shapes do not match')    
    if (nelemb!=nelem or nelemc!=nelem):
        raise Exception('number of blocks do not match')  
    
    mat = np.zeros((nen*nelem,nen*nelem))        
    for e in range(nelem-1):
        for i in range(nen):
            for j in range(nen):
                mat[nen*e+i,nen*e+j] = blockM[i,j,e]
                mat[nen*e+i,nen*(e-1)+j] = blockL[i,j,e]
                mat[nen*e+i,nen*(e+1)+j] = blockR[i,j,e]
    e = nelem-1
    for i in range(nen):
        for j in range(nen):
            mat[nen*e+i,nen*e+j] = blockM[i,j,e]
            mat[nen*e+i,nen*(e-1)+j] = blockL[i,j,e]
            mat[nen*e+i,j] = blockR[i,j,e]
            
    return mat

@njit
def gm_triblock_2D_flat_periodic(blockL,blockM,blockR,blockD,blockU,nelemy):
    '''
    Takes 5 global matrices of shape (nen^2,nen^2,nelemx*nelemy) and returns a 2d array 
    of shape (nen^2*nelemx*nelemy,nen^2*nelemx*nelemy) where the nelemx*nelemy
    blockM (nen^2,nen^2) blocks are along the diagonal, blockL blocks are nelemy 
    blocks to the left of the main diagonal, blockR blocks are nelemy blocks to 
    the right, blockD are immediately to the left, and blockU are immediately to 
    the right. The first block of blockL is sent to -nelemy from the top right,
    the last block of blockR is sent to nelemy from the bottom left. The first
    block of blockD is sent to nelemy from the top left, and the last block of
    blockU is sent to -nelemy from the bottom right.

    Returns
    -------
    c : numpy array of shape (nen^2*nelemx*nelemy,nen^2*nelemx*nelemy)
    '''
    nen,nenb,nelem = blockM.shape
    nenc,nend,nelemb = blockL.shape
    nene,nenf,nelemc = blockR.shape
    neng,nenh,nelemd = blockD.shape
    neni,nenj,neleme = blockU.shape
    if (nenb!=nen or nenc!=nen or nend!=nen or nene!=nen or nenf!=nen or neng!=nen or nenh!=nen or neni!=nen or nenj!=nen):
        raise Exception('block shapes do not match')    
    if (nelemb!=nelem or nelemc!=nelem or nelemd!=nelem or neleme!=nelem):
        raise Exception('number of blocks do not match')  
    
    mat = np.zeros((nen*nelem,nen*nelem))        
    for e in range(nelem-1):
        for i in range(nen):
            for j in range(nen):
                mat[nen*e+i,nen*e+j] = blockM[i,j,e]
                mat[nen*e+i,nen*(e-nelemy)+j] = blockL[i,j,e]
                mat[nen*e+i,nen*(e+nelemy)+j] = blockR[i,j,e]
                mat[nen*e+i,nen*(e-1)+j] = blockD[i,j,e]
                mat[nen*e+i,nen*(e+1)+j] = blockU[i,j,e]
    e = nelem-1
    for i in range(nen):
        for j in range(nen):
            mat[nen*e+i,nen*e+j] = blockM[i,j,e]
            mat[nen*e+i,nen*(e-nelemy)+j] = blockL[i,j,e]
            mat[nen*e+i,nen*nelemy+j] = blockR[i,j,e]
            mat[nen*e+i,nen*(e-1)+j] = blockD[i,j,e]
            mat[nen*e+i,j] = blockU[i,j,e]
            
    return mat

@njit
def lm_gm_had(A,B):
    '''
    Compute the hadamard product between a local matrix (nen1,nen2) and 
    global matrix (nen1,nen2,nelem)

    Returns
    -------
    C : numpy array of shape (nen1,nen2,nelem)
    '''
    nen,nen2,nelem = B.shape
    C = np.zeros((nen,nen2,nelem))
    for e in range(nelem):
        C[:,:,e] = np.multiply(A,B[:,:,e])
            
    return C

@njit
def lm_gm_had_diff(A,B):
    '''
    Compute the hadamard product between a local matrix (nen1,nen2) and 
    global matrix (nen1,nen2,nelem) then sum rows

    Returns
    -------
    C : numpy array of shape (nen1,nen2,nelem)
    '''
    nen,nen2,nelem = B.shape
    C = np.zeros((nen,nen2,nelem))
    for e in range(nelem):
        C[:,:,e] = np.multiply(A,B[:,:,e])
    
    c = np.sum(C,axis=1)
    return c

@njit
def gm_gm_had(A,B):
    '''
    Compute the hadamard product between a local matrix (nen1,nen2) and 
    global matrix (nen1,nen2,nelem)

    Returns
    -------
    C : numpy array of shape (nen1,nen2,nelem)
    '''
    nen,nen2,nelem = B.shape
    C = np.zeros((nen,nen2,nelem))
    for e in range(nelem):
        C[:,:,e] = np.multiply(A[:,:,e],B[:,:,e])
            
    return C

@njit
def gm_gm_had_diff(A,B):
    '''
    Compute the hadamard product between a local matrix (nen1,nen2) and 
    global matrix (nen1,nen2,nelem) then sum rows

    Returns
    -------
    C : numpy array of shape (nen1,nen2,nelem)
    '''
    nen,nen2,nelem = B.shape
    C = np.zeros((nen,nen2,nelem))
    for e in range(nelem):
        C[:,:,e] = np.multiply(A[:,:,e],B[:,:,e])
    
    c = np.sum(C,axis=1)
    return c


def isDiag(M):
    if M.ndim==2:
        i, j = M.shape
        assert i == j 
        test = M.reshape(-1)[:-1].reshape(i-1, j+1)
        return ~np.any(test[:, 1:])
    elif M.ndim==3:
        i, j, e = M.shape
        assert i == j 
        test = M.reshape(-1)[:-e].reshape(i-1, (j+1)*e)
        return ~np.any(test[:, e:])
    else:
        raise Exception('Inputted shape not understood.')
        
@njit
def pad_periodic_1d(q):
    '''
    Take a global vector (nen,nelem) and pad it so that it becomes a global
    vector (nen,nelem+2) where the first element is the last element and the
    last element is the first element. Used in dqdt for periodic bc.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem)
    '''
    nen,nelem = q.shape
    qpad = np.zeros((nen,nelem+2))
    qpad[:,1:-1] = q
    qpad[:,0] = q[:,-1]
    qpad[:,-1] = q[:,0]           
    return qpad

@njit
def pad_1d(q,qL,qR):
    '''
    Take a global vector (nen,nelem) and pad it so that it becomes a global
    vector (nen,nelem+2) where the first element is qL and the last element is 
    qR. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+2)
    '''
    nen,nelem = q.shape
    if qL.shape!=(nen,) or qR.shape!=(nen,):
        raise Exception('shapes do not match') 
    qpad = np.zeros((nen,nelem+2))
    qpad[:,1:-1] = q
    qpad[:,0] = qL
    qpad[:,-1] = qR          
    return qpad

@njit
def pad_1dL(q,qL):
    '''
    Take a global vector (nen,nelem) and pad it so that it becomes a global
    vector (nen,nelem+1) where the first element is qL. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+1)
    '''
    nen,nelem = q.shape
    if qL.shape!=(nen,):
        raise Exception('shapes do not match') 
    qpad = np.zeros((nen,nelem+1))
    qpad[:,1:] = q
    qpad[:,0] = qL         
    return qpad

@njit
def pad_1dR(q,qR):
    '''
    Take a global vector (nen,nelem) and pad it so that it becomes a global
    vector (nen,nelem+1) where the last element is qR. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+1)
    '''
    nen,nelem = q.shape
    if qR.shape!=(nen,):
        raise Exception('shapes do not match') 
    qpad = np.zeros((nen,nelem+1))
    qpad[:,:-1] = q
    qpad[:,-1] = qR          
    return qpad

@njit # TODO: renamed from fix_satL_1D
def shift_left(q):
    '''
    Take a global vector (nen,nelem) and move the first elem to the last elem.
    This is used for example for periodic cases to create qR from q.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nelem = q.shape
    qfix = np.zeros((nen,nelem))
    qfix[:,:-1] = q[:,1:]
    qfix[:,-1] = q[:,0]            
    return qfix

@njit
def shift_right(q):
    '''
    Take a global vector (nen,nelem) and move the first elem to the last elem.
    This is used for example for periodic cases to create qL from q.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nelem = q.shape
    qfix = np.zeros((nen,nelem))
    qfix[:,1:] = q[:,:-1]
    qfix[:,0] = q[:,-1]            
    return qfix

@njit
def fix_dsatL_1D(q):
    '''
    Take a global matrix (nen,nen,nelem) and move the first elem to the last elem.
    Used in dqdt to fix the dsatL vector which has contributions for elements
    -1, 0, 1, ..., -2. Fixes it to contribute to 0, 1, ..., -2, -1.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nen1,nelem = q.shape
    qfix = np.zeros((nen,nen1,nelem))
    qfix[:,:,:-1] = q[:,:,1:]
    qfix[:,:,-1] = q[:,:,0]            
    return qfix

@njit
def reshape_to_meshgrid_2D(q,nen,nelemx,nelemy):
    ''' take a 2D vector q in the shape (nen**2,nelemx*nelemy) and reshape
    to a 2D mesh in the shape (nen*nelem, nen*nelemy) as would be created
    by meshgrid. Can think of this array ordering being the actual bird's 
    eye view of the mesh. '''
    if q.shape != (nen**2,nelemx*nelemy):
        raise Exception('Shape does not match.')  
        
    Q = np.zeros((nen*nelemx, nen*nelemy))
    for ex in range(nelemx):
        for ey in range(nelemy):
            for nx in range(nen):
                for ny in range(nen):
                    Q[ex*nen + nx, ey*nen + ny] = q[nx*nen + ny,ex*nelemy + ey]
    return Q
    
# Don't use nopython @njit in case we need to pass class objects in ec_flux
# June 2023: I put it back in becuase the keyword default of False is being depreciated and 
# it threw a warning. Maybe I can get away with this if it supports class functions?
@njit
def build_F_vol_sca(q, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 1 
    solution vector q, the number of equations per node, and a 2-point 
    flux function. Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    F = np.zeros((nen_neq,nen_neq,nelem))  
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(i,nen_neq):
                f = flux(q[i,e],q[j,e])
                F[i,j,e] = f
                if i != j:
                    F[j,i,e] = f
    return F

@njit
def build_F_vol_sys(neq, q, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 1 
    solution vector q, the number of equations per node, and a 2-point 
    flux function. Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    F = np.zeros((nen_neq,nen_neq,nelem))   
    nen = int(nen_neq / neq)
    for e in range(nelem):
        for i in range(nen):
            for j in range(i,nen):
                idxi = i*neq
                idxi2 = (i+1)*neq
                idxj = j*neq
                idxj2 = (j+1)*neq
                diag = np.diag(flux(q[idxi:idxi2,e],q[idxj:idxj2,e]))
                F[idxi:idxi2,idxj:idxj2,e] = diag
                if i != j:
                    F[idxj:idxj2,idxi:idxi2,e] = diag
    return F

# Don't use nopython @njit in case we need to pass class objects in ec_flux
# June 2023: I put it back in becuase the keyword default of False is being depreciated and 
# it threw a warning. Maybe I can get away with this if it supports class functions?
@njit
def build_F_sca(q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function. for scalar equations, neq=1 '''
    nen_neq, nelem = q1.shape 
    F = np.zeros((nen_neq,nen_neq,nelem))  
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(nen_neq):
                f = flux(q1[i,e],q2[j,e])
                F[i,j,e] = f
    return F

@njit
def build_F_sys(neq, q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function '''
    nen_neq, nelem = q1.shape 
    F = np.zeros((nen_neq,nen_neq,nelem))  
    nen = int(nen_neq / neq)
    for e in range(nelem):
        for i in range(nen):
            for j in range(nen):
                idxi = i*neq
                idxi2 = (i+1)*neq
                idxj = j*neq
                idxj2 = (j+1)*neq
                diag = np.diag(flux(q1[idxi:idxi2,e],q2[idxj:idxj2,e]))
                F[idxi:idxi2,idxj:idxj2,e] = diag
    return F



@njit
def arith_mean(qL,qR):
    ''' arithmetic mean. When used in Hadamard, equivalent to divergence form. '''
    q = (qL+qR)/2
    return q

@njit
def log_mean(qL,qR):
    ''' logarithmic mean. Useful for EC fluxes. '''
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
def prod_mean(q1L,q2L,q1R,q2R):
    '''' product mean. Useful for split-form fluxes. '''
    q = (q1L*q2R+q2L*q1R)/2
    return q

def is_pos_def(A):
    ''' check if a matrix A is symmetric positive definite '''
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

@njit
def repeat_neq_gv(q,neq_node):
    ''' take array of shape (nen,nelem) and return (nen*neq_node,nelem)
        where the value on each node is repeat neq_node times. 
        Note: just as fast as np.repeat(q,neq_node,0) but this 
              is not compatible with jit (axis argument not supported)'''
    nen, nelem = q.shape
    qn = np.zeros((nen*neq_node,nelem)) 
    for e in range(nelem):
        for i in range(nen):
            for i2 in range(i*neq_node,i*neq_node+neq_node):
                qn[i2,e] = q[i,e]
    return qn

@njit
def kron_neq_gm(A,neq_node):
    ''' take array of shape (nen,nen2,nelem) and return (nen*neq_node,nen2*neq_node,nelem)
        the proper kronecker product for the operator acting on a vector (nen2*neq_node,nelem). '''
    nen, nen2, nelem = A.shape
    An = np.zeros((nen*neq_node,nen2*neq_node,nelem)) 
    for e in range(nelem):
        for i in range(nen):
            for n in range(neq_node):
                i2 = i*neq_node + n
                for j in range(nen2):
                    An[i2,j*neq_node+n::neq_node,e] = A[i,j,e]
    return An

@njit
def kron_neq_lm(A,neq_node):
    ''' take array of shape (nen,nen2) and return (nen*neq_node,nen2*neq_node)
        the proper kronecker product for the operator acting on a vector (nen2*neq_node,nelem). '''
    nen, nen2 = A.shape
    An = np.zeros((nen*neq_node,nen2*neq_node)) 
    for i in range(nen):
        for n in range(neq_node):
            i2 = i*neq_node + n
            for j in range(nen2):
                An[i2,j*neq_node+n::neq_node] = A[i,j]
    return An

@njit
def repeat_neq_lv(q,neq_node):
    ''' take array of shape (nen) and return (nen*neq_node)
        where the value on each node is repeat neq_node times. 
        Note: just as fast as np.repeat(q,neq_node,0) but this 
              is not compatible with jit (axis argument not supported)'''
    nen = len(q)
    qn = np.zeros(nen*neq_node) 
    for i in range(nen):
        for j in range(i*neq_node,i*neq_node+neq_node):
            qn[j] = q[i]
    return qn


""" Old functions (no longer useful)

def ldiag_gdiag2(l,g):
    '''
    Takes a a local array of shape (nen) that simulates a local diagonal 
    matrix of shape (nen,nen), and a global array of shape (nen,nelem)
    that simulates a global diagonal matrix of shape (nen,nen,nelem), 
    and returns a global array of shape (nen,nelem), simulating a global
    matrix of shape (nen,nen,nelem), i.e. l @ g

    Parameters
    ----------
    l : numpy array of shape (nen)
    g : numpy array of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen,nelem)
    ''' 
    return l[:,None] * g

@njit
def gv_lvT(A,B):
    '''
    Takes a global vector of shape (nen1,nelem) and a local vector of shape 
    (nen2,) or (1,nen2) and returns the outer product global matrix of shape 
    (nen1,nen2,nelem)

    Parameters
    ----------
    A : numpy array of shape (nen1,nelem)
    B : numpy array of shape (nen2,) or (1,nen2)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    '''
    nen1,nelem = np.shape(A)
    nen2 = B.size
    c = np.zeros((nen1,nen2,nelem))
    if B.ndim == 1:
        for e in range(nelem):
            c[:,:,e] = np.outer(A[:,e],B)
    elif B.ndim == 2:
        b = B[0,:]
        for e in range(nelem):
            c[:,:,e] = np.outer(A[:,e],b)
    else: raise Exception('Local vector shape not understood. Should be (nen,) or (1,nen)')

    return c

@njit
def gm_triblock_flat(blockL,blockM,blockR):
    '''
    Takes 3 3d arrays, blockL and blockR of shape (nen,nen,nelem-1) and 
    blockM of shape (nen,nen,nelem) returns a 2d array of shape (nen*nelem,nen*nelem)
    where the nelem (nen,nen) blocks are along the diagonal, blockL blocks are
    to the left of the main diagonal, and blockR blocks are to the right.

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nen,nenb,nelem = blockM.shape
    nenc,nend,nelemb = blockL.shape
    nene,nenf,nelemc = blockR.shape
    if (nenb!=nen or nenc!=nen or nend!=nen or nene!=nen or nenf!=nen):
        raise Exception('block shapes do not match')    
    if (nelemb!=nelem-1 or nelemc!=nelem-1):
        raise Exception('number of blocks do not match')    
        
    mat = np.zeros((nen*nelem,nen*nelem))
    for e in range(nelem-1):
        nene = nen*e
        nenen = nene+nen
        for j in range(nen):
            nenej = nene+j
            mat[nene:nenen,nenej] = blockM[:,j,e]
            mat[nenen:nenen+nen,nenej] = blockL[:,j,e]
            mat[nene:nenen,nenen+j] = blockR[:,j,e]
    e = nelem-1
    nene = nen*e
    nenen = nene+nen
    for j in range(nen):
        nenej = nene+j
        mat[nene:nenen,nenej] = blockM[:,j,e]
        
    return mat

@njit
def lm_lv(A,b):
    '''
    Equivalent to np.einsum('ij,j->i',A,b) where A is a 2-tensor of shape
    (nen,nen) and b is a 1-tensor of shape (nen). This can be 
    thought of as a local matrix @ local vector.

    Parameters
    ----------
    A : numpy array of shape (nen,nen)
    b : numpy array of shape (nen)
    
    note: in theory this works for general shapes (i,j) and (j)

    Returns
    -------
    c : numpy array of shape (nen)
    '''
    return A@b

@njit
def lm_lm(A,B):
    '''
    Equivalent to np.einsum('ij,jk->ik',A,B) where A is a 2-tensor of shape
    (nen,nen) and b is a 2-tensor of shape (nen,nen). This can be 
    thought of as a local matrix @ local matrix.

    Parameters
    ----------
    A : numpy array of shape (nen,nen)
    B : numpy array of shape (nen,nen)
    
    note: this does not work for general shapes (i,j) and (j,k)

    Returns
    -------
    c : numpy array of shape (nen,nen)
    '''
    return A@B

def dot(A,B):
    '''
    Equivalent to @ where A is a (possibly sparse) 2-array of shape (a,b) and 
    b is either a (possibly sparse) 2-array of shape (b,c) or a numpy array of 
    shape (b).

    Parameters
    ----------
    A : (sparse) array of shape (a,b)
    B : numpy array of shape (b) or (sparse) array of shape (b,c)

    Returns
    -------
    c : numpy array of shape (a) or (sparse) array of shape (a,c)
    '''
    return A.dot(B)

@njit
def diag_1d(q):
    '''
    Takes a 2-dim numpy array q of shape (nen,1) and returns a 2-dim
    array of shape (nen,nen) with the entries of q along the diagonals.

    Parameters
    ----------
    q : numpy array of shape (nen)

    Returns
    -------
    c : numpy array of shape (nen,nen)
    '''
    return np.diag(q[:,0])

def diag_sp(q):
    '''
    Takes a 2-dim numpy array q of shape (n) and returns a 2-dim
    sparse array in csr_matrix format of shape (n,n) with the entries 
    of q along the diagonals.

    Parameters
    ----------
    q : numpy array of shape (n)

    Returns
    -------
    c : sparse dia array of shape (n,n)
    '''
    return sp.diags(q,format="csr")

def diag_sp_FD(q):
    '''
    Takes a 2-dim numpy array q of shape (nen,1) and returns a 2-dim
    sparse array in csr_matrix format of shape (nen,nen) with the entries 
    of q along the diagonals.

    Parameters
    ----------
    q : numpy array of shape (nen)

    Returns
    -------
    c : sparse dia array of shape (nen,nen)
    '''
    return sp.diags(q[:,0],format="csr")

def chk_q_unstr(q):
    '''
    Check shape of q to ensure it is in an unstructured form, i.e. the q
    being passed is the local q for a single element, not the global q.

    Parameters
    ----------
    q : numpy array

    Returns
    -------
    properly shaped q

    '''
    if q.ndim == 1:
        return q
    if q.ndim == 2:
        print('WARNING: Passed q of wrong shape. Taking first element only.')
        return q[:,0]
    else:
        print('ERROR: Unrecognized q shape.')
        
@njit       
def block_diag_1d(*entries):
    '''
    Takes neq_node^2 2-dim numpy arrays q of shape (nen,1) and returns a 2-dim
    block-diagonal array of shape (nen*neq_node,nen*neq_node) where each block
    along the diagonal is an (neq_node,neq_node) 2-dim array with entries
    given from the *entries arrays in order from top left, left to right, then 
    top to bottom (normal reading direction)

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,1)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    neq_node = int(np.sqrt(len(entries)))
    nen = len(entries[0])
    blocks = np.zeros((nen,neq_node,neq_node))
    mat = np.zeros((nen*neq_node,nen*neq_node))

    idx = 0
    for entry in literal_unroll(entries):
        row = idx // neq_node
        col = idx % neq_node
        blocks[:,row,col] = entry[:,0]
        idx += 1
    for i in range(nen):
        a = i*neq_node
        b = (i+1)*neq_node
        mat[a:b,a:b] = blocks[i]    
    return mat
"""