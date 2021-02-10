#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:26:34 2020

@author: bercik
"""

from numba import jit, literal_unroll
import numpy as np
#import scipy.sparse as sp

# The useful functions are defined first, the others are shoved to the bottom

@jit(nopython=True)
def gm_gv(A,b):
    '''
    Equivalent to np.einsum('ijk,jk->ik',A,b) where A is a 3-tensor of shape
    (nen,nen,nelem) and b is a 2-tensor of shape (nen,nelem). This can be 
    thought of as a global matrix @ global vector.

    Parameters
    ----------
    A : numpy array of shape (nen,nen,nelem)
    b : numpy array of shape (nen,nelem)
    
    note: in theory this works for general shapes (i,j,k) and (j,k)

    Returns
    -------
    c : numpy array of shape (nen,nelem)
    '''
    ri,rj,re = np.shape(A)
    c = np.zeros((ri,re))
    for i in range(ri):
        for j in range(rj):
            for e in range(re):
                c[i,e] += A[i,j,e]*b[j,e]
    return c

@jit(nopython=True)
def gm_gm(A,B):
    '''
    Equivalent to np.einsum('ijk,jlk->ilk',A,B) where A is a 3-tensor of shape
    (nen,nen,nelem) and B is a 3-tensor of shape (nen,nen,nelem). This can be 
    thought of as a global matrix @ global matrix.

    Parameters
    ----------
    A : numpy array of shape (nen,nen,nelem)
    B : numpy array of shape (nen,nen,nelem)
    
    note: this does not work for general shapes (i,j,k) and (j,l,k)

    Returns
    -------
    c : numpy array of shape (nen,nen,nelem)
    '''
    ri,rj,re = np.shape(A)
    c = np.zeros((ri,rj,re))
    for i in range(ri):
        for j in range(rj):
            for l in range(rj):
                for e in range(re):
                    c[i,l,e] += A[i,j,e]*B[j,l,e]
    return c

@jit(nopython=True)
def gm_lm(A,B):
    '''
    Equivalent to np.einsum('ijk,jl->ilk',A,B) where A is a 3-tensor of shape
    (nen,nen,nelem) and B is a 2-tensor of shape (nen,nen). This can be 
    thought of as a global matrix @ local matrix.

    Parameters
    ----------
    A : numpy array of shape (nen,nen,nelem)
    B : numpy array of shape (nen,nen)
    
    note: this does not work for general shapes (i,j,k) and (j,l,k)

    Returns
    -------
    c : numpy array of shape (nen,nen,nelem)
    '''
    ri,rj,re = np.shape(A)
    c = np.zeros((ri,rj,re))
    for i in range(ri):
        for j in range(rj):
            for l in range(rj):
                for e in range(re):
                    c[i,l,e] += A[i,j,e]*B[j,l]
    return c

@jit(nopython=True)
def lm_gm(A,B):
    '''
    Equivalent to np.einsum('ij,jlk->ilk',A,B) where A is a 2-tensor of shape
    (nen,nen) and B is a 3-tensor of shape (nen,nen,nelem). This can be 
    thought of as a local matrix @ global matrix.

    Parameters
    ----------
    A : numpy array of shape (nen,nen)
    B : numpy array of shape (nen,nen,nelem)
    
    note: this does not work for general shapes (i,j,k) and (j,l,k)

    Returns
    -------
    c : numpy array of shape (nen,nen,nelem)
    '''
    ri,rj,re = np.shape(B)
    c = np.zeros((ri,rj,re))
    for i in range(ri):
        for j in range(rj):
            for l in range(rj):
                for e in range(re):
                    c[i,l,e] += A[i,j]*B[j,l,e]
    return c

@jit(nopython=True)
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

@jit(nopython=True)
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
    '''
    neq_node = int(np.sqrt(len(entries)))
    nen,nelem = entries[0].shape
    blocks = np.zeros((nen,neq_node,neq_node,nelem))
    mat = np.zeros((nen*neq_node,nen*neq_node,nelem))

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

@jit(nopython=True)
def abs_eig_mat(mat):
    '''
    Given a 3d array in the shape (nen*neq_node,nen*neq_node,nelem), return
    a 3d array in the same shape where the matrices in each element are now
    absoluted through it's eigenvalues. That is, if A is one such matrix, 
    and it has eigenvalues L and right eigenvectors X, then this returns
    X @ abs(L) @ X.T

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nodes,_,nelem = mat.shape
    mat_abs = np.zeros((nodes,nodes,nelem))
    for elem in range(nelem):
        eig_val, eig_vec = np.linalg.eig(mat[:,:,elem])
        mat_abs[:,:,elem] = eig_vec @ np.diag(np.abs(eig_val)) @ np.linalg.inv(eig_vec)
    return mat_abs



""" Old functions (no longer useful)

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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
        
@jit(nopython=True)       
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