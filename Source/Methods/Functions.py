#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:26:34 2020

@author: bercik
"""

from numba import njit, literal_unroll
import numpy as np

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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'

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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'

    c = np.zeros((nen1,nen3,nelem),dtype=B.dtype)
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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 

    c = np.zeros((nen1,nen3,nelem),dtype=B.dtype)
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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 

    c = np.zeros((nen1,nelem),dtype=b.dtype)
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen1):
                c[i,e] += A[i,j,e] * b[j]
    return c

@njit
def lm_gm(A,B):
    '''
    NOTE: NOT equivalent to A @ B 
    That returns the elementwise transpose of the desired result.
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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 

    c = np.zeros((nen1,nen3,nelem),dtype=B.dtype)
    for i in range(nen1):
        for j in range(nen2):
            for l in range(nen3):
                for e in range(nelem):
                    c[i,l,e] += A[i,j]*B[j,l,e]
    return c

#@njit
def lm_gv(A,b):
    '''
    equivalent to A @ b

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2)
    b : numpy array of shape (nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nelem)
    '''
    c = A @ b
    return c

#@njit
def lm_lv(A,b):
    '''
    equivalent to A @ b

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2)
    b : numpy array of shape (nen2)

    Returns
    -------
    c : numpy array of shape (nen1)
    '''
    c = A @ b
    return c

#@njit
def lm_lm(A,B):
    '''
    equivalent to A @ B

    Parameters
    ----------
    A : numpy array of shape (nen1,nen2)
    b : numpy array of shape (nen2,nen2)

    Returns
    -------
    C : numpy array of shape (nen1,nelem)
    '''
    C = A @ B
    return C

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
    c = np.zeros((nen1,nen2,nelem),dtype=B.dtype)
    if A.ndim == 1:
        for e in range(nelem):
            c[:,:,e] = A[e]*B
    elif A.ndim == 2:
        for e in range(nelem):
            c[:,:,e] = A[0,e]*B
    else: raise Exception('Scalar shape not understood. Should be (nelem,) or (1,nelem)')

    return c

@njit
def gdiag_lm(H, D):
    '''
    Takes a global array of shape (nen1, nelem) that simulates a global diagonal
    matrix of shape (nen1, nen1, nelem), and a local matrix of shape (nen1, nen2) 
    and returns a global matrix of shape (nen1, nen2, nelem), i.e. H @ D

    Parameters
    ----------
    H : numpy array of shape (nen1, nelem)
    D : numpy array of shape (nen1, nen2)

    Returns
    -------
    c : numpy array of shape (nen1, nen2, nelem)
    ''' 
    nen1, nelem = H.shape
    nen1b, nen2 = D.shape
    assert nen1==nen1b, f'array shapes do not match, {nen1} != {nen1b}' 
    
    c = np.zeros((nen1, nen2, nelem), dtype=D.dtype)
    
    # Optimized loop ordering for better performance
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen1):
                c[i, j, e] = H[i, e] * D[i, j]
                
    return c

@njit
def lm_gdiag(D, H):
    '''
    Takes a local matrix of shape (nen1, nen2) and a global array of shape 
    (nen2, nelem) that simulates a global diagonal matrix of shape (nen2, nen2, nelem), 
    and returns a global matrix of shape (nen1, nen2, nelem), i.e. D @ H

    Parameters
    ----------
    D : numpy array of shape (nen1, nen2)
    H : numpy array of shape (nen2, nelem)

    Returns
    -------
    c : numpy array of shape (nen1, nen2, nelem)
    ''' 
    nen2, nelem = H.shape
    nen1, nen2b = D.shape
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 
    
    c = np.zeros((nen1, nen2, nelem), dtype=H.dtype)
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen1):
                c[i, j, e] = D[i, j] * H[j, e]
                
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
    assert nen1==nen1b, f'array shapes do not match, {nen1} != {nen1b}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'
    
    c = np.zeros((nen1, nen2, nelem), dtype=D.dtype)
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen1):
                c[i, j, e] = H[i, e] * D[i, j, e]
                
    return c


@njit
def gm_gdiag(D,H):
    '''
    Takes a global matrix of shape (nen1,nen1,nelem) and a global array of shape
    (nen1,nelem) that simulates a global diagonal, and returns a global matrix 
    of shape (nen1,nen2,nelem), i.e. H @ D

    Parameters
    ----------
    H : numpy array of shape (nen1,nelem)
    D : numpy array of shape (nen1,nen2,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)

    NOTE: probably faster just to do D * H, gives the same result
    ''' 
    nen1,nen2,nelem = np.shape(D)
    nen2b,nelemb = np.shape(H)
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'

    c = np.zeros((nen1,nen2,nelem),dtype=H.dtype)
    for e in range(nelem):
        c[:,:,e] = D[:,:,e] * H[:,e]
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
    assert nen==nenb, f'array shapes do not match, {nen} != {nenb}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'

    c = H * q
    return c

@njit
def ldiag_gv(H,q):
    '''
    Takes a local array of shape (nen) that simulates a local diagonal
    matrix of shape (nen,nen), and a global vector of shape (nen,nelem) 
    and returns a global vector of shape (nen,nelem), i.e. H @ q

    Parameters
    ----------
    H : numpy array of shape (nen,nelem)
    D : numpy array of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen1,nen2,nelem)
    ''' 
    nen = len(H)
    nenb,nelemb = np.shape(q)
    assert nen==nenb, f'array shapes do not match, {nen} != {nenb}' 

    c = np.zeros((nen,nelemb),dtype=q.dtype) 
    for e in range(nelemb):
        c[:,e] = H * q[:,e]  
    return c

@njit
def ldiag_lm(H_diag, M):
    '''
    Multiply a diagonal matrix H with diagonal entries H_diag by a dense matrix M.

    Parameters
    ----------
    H_diag : 1D numpy array
        The diagonal elements of the matrix H.
    M : 2D numpy array
        The dense matrix to be multiplied.

    Returns
    -------
    result : 2D numpy array
        The result of H * M where H is a diagonal matrix with elements H_diag.
    '''
    rows, cols = M.shape
    result = np.zeros((rows, cols), dtype=M.dtype)

    # Scale each row of M by the corresponding element in H_diag
    for i in range(rows):
        for j in range(cols):
            result[i, j] = H_diag[i] * M[i, j]
    
    return result
    
@njit
def lm_ldiag(M, H_diag):
    '''
    Multiply a dense matrix M by a diagonal matrix H with diagonal entries H_diag.

    Parameters
    ----------
    M : 2D numpy array
        The dense matrix to be multiplied.
    H_diag : 1D numpy array
        The diagonal elements of the matrix H.

    Returns
    -------
    result : 2D numpy array
        The result of M * H where H is a diagonal matrix with elements H_diag.
    '''
    
    return M * H_diag

@njit
def gbdiag_gbdiag(A,B):
    '''
    Takes two global arrays of shape (nen,neq,neq,nelem) and simulates matrix 
    multiplication between 2 global block-diagonal matrices
    equivalent to np.einsum('nije,njke->nike', A, B)

    Parameters
    ----------
    A : numpy array of shape (nen,neq,neq,nelem)
    B : numpy array of shape (nen,neq,neq,nelem)

    Returns
    -------
    c : numpy array of shape (nen,neq,neq,nelem)
    '''
    nen,neq,neq2,nelem = np.shape(A)
    nenb,neqb,neqb2,nelemb = np.shape(B)
    assert nen==nenb and neq==neqb and neq2==neqb2 and nelem==nelemb, f'array shapes do not match, {nen} != {nenb}, {neq} != {neqb}, {neq2} != {neqb2}, {nelem} != {nelemb}'
    assert neq==neq2, f'array shapes are not block diagonal, {neq} != {neq2}'
    
    c = np.zeros((nen,neq,neq,nelem),dtype=B.dtype)
    for e in range(nelem):
        for n in range(nen):
            for k in range(neq): 
                for i in range(neq):
                    sum_value = 0
                    for j in range(neq):
                        sum_value += A[n, i, j, e] * B[n, j, k, e]
                    c[n, i, k, e] = sum_value
    return c

@njit
def gbdiag_gv(A,b):
    '''
    Takes a global arrays of shape (nen,neq,neq,nelem) and a global
    vector of shape (nen*neq) and simulates matrix multiplication

    Parameters
    ----------
    A : numpy array of shape (nen,neq,neq,nelem)
    b : numpy array of shape (nen*neq,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq,nelem)
    '''
    nen,neq,neq2,nelem = np.shape(A)
    nen_neq,nelemb = np.shape(b)
    assert nen*neq==nen_neq and nelem==nelemb, f'array shapes do not match, {nen*neq} != {nen_neq}, {nelem} != {nelemb}'
    assert neq==neq2, f'array is not block diagonal ({neq} != {neq2})'
    
    c = np.zeros((nen_neq,nelem),dtype=b.dtype)
    for e in range(nelem):
        for n in range(nen):
            for i in range(neq):
                sum_value = 0
                for j in range(neq):
                    sum_value += A[n, i, j, e] * b[n * neq + j, e]
                c[n * neq + i, e] = sum_value
    return c

@njit
def gdiag_gbdiag(A,B):
    '''
    Takes a global diagonal array (nen*neq,nelem) and a 
    global arrays of shape (nen,neq,neq,nelem) and simulates matrix 
    multiplication, returning a global array of shape (nen,neq,neq,nelem)

    Parameters
    ----------
    A : numpy array of shape (nen*neq,nelem)
    B : numpy array of shape (nen,neq,neq,nelem)

    Returns
    -------
    c : numpy array of shape (nen,neq,neq,nelem)
    '''
    nen_neq,nelem = np.shape(A)
    nen,neq,neqb,nelemb = np.shape(B)
    assert nen*neq==nen_neq and nelem==nelemb, f'array shapes do not match, {nen*neq} != {nen_neq}, {nelem} != {nelemb}'
    assert neq==neqb, f'array is not block diagonal ({neq} != {neqb})'
    
    c = np.zeros((nen,neq,neq,nelem),dtype=B.dtype)
    for e in range(nelem):
        for n in range(nen):
            for i in range(neq):
                for j in range(neq):
                    c[n, i, j, e] = A[n * neq + i, e] * B[n, i, j, e]
    return c

@njit
def gbdiag_gdiag(A,B):
    '''
    Takes a global block diagonal array (nen,neq,neq,nelem) and a 
    global diagonal array of shape (nen*neq,nelem) and simulates matrix 
    multiplication, returning a global array of shape (nen,neq,neq,nelem)

    Parameters
    ----------
    A : numpy array of shape (nen,neq,neq,nelem)
    B : numpy array of shape (nen*neq,nelem)

    Returns
    -------
    c : numpy array of shape (nen,neq,neq,nelem)
    '''
    nen_neq,nelem = np.shape(B)
    nen,neq,neqb,nelemb = np.shape(A)
    assert nen*neq==nen_neq and nelem==nelemb, f'array shapes do not match, {nen*neq} != {nen_neq}, {nelem} != {nelemb}'
    assert neq==neqb, f'array is not block diagonal ({neq} != {neqb})'
    
    c = np.zeros((nen,neq,neq,nelem),dtype=B.dtype)
    for e in range(nelem):
        for n in range(nen):
            for i in range(neq):
                for j in range(neq):
                    c[n, i, j, e] = A[n, i, j, e] * B[n * neq + j, e]
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
    assert nen2==nen2b, f'array shapes do not match, {nen2} != {nen2b}' 
    assert nelem==nelemb, f'element shapes do not match, {nelem} != {nelemb}'
    c = np.zeros((nen1,nen2,nelem),dtype=q.dtype)
    for e in range(nelem):
        c[:,:,e] = A[:,:,e] * q[:,e]
    return c

@njit
def gdiag_to_gm(q):
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
    c=np.zeros((i,i,k),dtype=q.dtype)
    for e in range(k):
        c[:,:,e] = np.diag(q[:,e])
    return c

@njit
def gm_to_gdiag(A):
    '''
    Takes a 3-dim numpy array A of shape (nen,nen,nelem) and returns a
    2-dim array of shape (nen,nelem) of the (nen) diagonal entries of A. 
    Can be thought of as the equivalent of np.diag(A) for a 3-dim array

    Parameters
    ----------
    A : numpy array of shape (nen,nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen,nelem)
    '''
    i,j,k = np.shape(A)
    assert i==j, f'input array is not square, {i} != {j}'

    c=np.zeros((i,k),dtype=A.dtype)
    for e in range(k):
        c[:,e] = np.diag(A[:,:,e])
    return c

@njit
def gdiag_to_gbdiag(q):
    '''
    Takes a 2-dim numpy array q of shape (nen,nelem) and returns a 3-dim
    array of shape (nen,1,1,nelem) simulating a global block-diagonal matrix

    Parameters
    ----------
    q : numpy array of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen,nen,nelem)
    '''
    nen,nelem = np.shape(q)
    c = np.reshape(q,(nen,1,1,nelem),dtype=q.dtype)
    return c

def check_q_shape(q):
    '''
    Does nothing (q is by default in structured form)
    TODO: Generalize this if we move to more general meshes
    '''
    assert(q.ndim==2),'ERROR: q is the wrong shape.'
    return q

@njit
def build_gbdiag(*entries):
    '''
    Takes neq_node^2 2-dim numpy arrays q of shape (nen,nelem) and returns an
    array of shape (nen,neq_node,neq_node,nelem) simulating a global block-diagonal
    matrix, where each entry in nen is a (neq_node,neq_node) local elem matrix
    with entries given from the *entries arrays in order from top left, 
    left to right, then top to bottom (normal reading direction)

    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen,neq_node,neq_node,nelem)
    '''
    neq_node = int(np.sqrt(len(entries)))
    nen,nelem = entries[0].shape
    dtype = entries[0].dtype
    mat = np.zeros((nen,neq_node,neq_node,nelem),dtype=dtype)

    idx = 0
    for entry in literal_unroll(entries): # add literal_unroll for heterogeneous tuple types
        row = idx // neq_node
        col = idx % neq_node
        mat[:,row,col,:] = entry
        idx += 1
    return mat

@njit
def abs_eig_mat(mat):
    '''
    Given a 4d array in the shape (nen,neq,neq,nelem), return
    a 4d array in the same shape where the matrices in each nen,nelem are now
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
    nen,neq,neqb,nelem = mat.shape
    assert neq==neqb, f'array shapes are not block diagonal ({neq} != {neqb})'
    mattype=mat.dtype
    mat_abs = np.zeros((nen,neq,neq,nelem),dtype=mattype)
    for e in range(nelem):
        for n in range(nen):
            eig_val, eig_vec = np.linalg.eig(mat[n,:,:,e])
            mat_abs[n,:,:,e] = eig_vec @ np.diag(np.abs(eig_val)).astype(mattype) @ np.linalg.inv(eig_vec)
    return mat_abs

@njit
def inv_gm(mat):
    ''' Return the inverse of a global matrix of shape (nen,nen,nelem) '''
    nodes,_,nelem = mat.shape
    dtype=mat.dtype
    mat_inv = np.zeros((nodes,nodes,nelem),dtype=dtype)
    for elem in range(nelem):
        mat_inv[:,:,elem] = np.linalg.inv(mat[:,:,elem])
    return mat_inv

@njit
def spec_rad(mat,neq):
    '''
    Given a 4d block diagonal array in the shape (nen,neq,neq,nelem)
    with blocks of size neq*neq, return a 2d array in the shape (nen,nelem) 
    , i.e. like a global scalar, where the values are the spectral radius of each 
    
    Parameters
    ----------
    *entries : numpy arrays of shape (nen,nelem)

    Returns
    -------
    c : numpy array of shape (nen*neq_node,nen*neq_node)
    '''
    nen, neq, neq2, nelem = mat.shape
    assert neq==neq2, f'array shapes are not block diagonal ({neq} != {neq2})'
    rho = np.zeros((nen,nelem),dtype=mat.dtype)
    for e in range(nelem):
        for i in range(nen):
            A = mat[i,:,:,e].astype(np.complex128)
            eigs = np.abs(np.linalg.eigvals(A))
            rho[i,e] = np.max(eigs)
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
    assert (nenb==nen and nenc==nen and nend==nen and nene==nen and nenf==nen), f'block shapes do not match {nenb} != {nen}, {nenc} != {nen}, {nend} != {nen}, {nene} != {nen}, {nenf} != {nen}'
    assert (nelemb==nelem-1 and nelemc==nelem-1), f'number of blocks do not match {nelemb} != {nelem-1}, {nelemc} != {nelem-1}'
        
    mat = np.zeros((nen*nelem,nen*nelem),dtype=blockL.dtype)
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
    assert (nenb==nen or nenc==nen or nend==nen or nene==nen or nenf==nen), f'block shapes do not match {nenb} != {nen}, {nenc} != {nen}, {nend} != {nen}, {nene} != {nen}, {nenf} != {nen}'  
    assert (nelemb==nelem or nelemc==nelem), f'number of blocks do not match {nelemb} != {nelem}, {nelemc} != {nelem}'
    
    mat = np.zeros((nen*nelem,nen*nelem),dtype=blockL.dtype)        
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
    assert (nenb==nen and nenc==nen and nend==nen and nene==nen and nenf==nen and neng==nen and nenh==nen and neni==nen and nenj==nen), \
        f'block shapes do not match {nenb} != {nen}, {nenc} != {nen}, {nend} != {nen}, {nene} != {nen}, {nenf} != {nen}, {neng} != {nen}, {nenh} != {nen}, {neni} != {nen}, {nenj} != {nen}'  
    assert (nelemb==nelem and nelemc==nelem and nelemd==nelem and neleme==nelem), f'number of blocks do not match {nelemb} != {nelem}, {nelemc} != {nelem}, {nelemd} != {nelem}, {neleme} != {nelem}'
    
    mat = np.zeros((nen*nelem,nen*nelem),dtype=blockL.dtype)        
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
    C = np.zeros((nen,nen2,nelem),dtype=B.dtype)
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
    C = np.zeros((nen,nen2,nelem),dtype=B.dtype)
    for e in range(nelem):
        C[:,:,e] = np.multiply(A,B[:,:,e])
    
    c = np.sum(C,axis=1)
    return c

@njit
def gm_gm_had(A,B):
    '''
    Compute the hadamard product between a global matrix (nen1,nen2,nelem) and 
    global matrix (nen1,nen2,nelem)

    Returns
    -------
    C : numpy array of shape (nen1,nen2,nelem)
    '''
    nen,nen2,nelem = B.shape
    C = np.zeros((nen,nen2,nelem),dtype=B.dtype)
    for e in range(nelem):
        C[:,:,e] = np.multiply(A[:,:,e],B[:,:,e])
            
    return C

@njit
def gm_gm_had_diff(A,B):
    '''
    Compute the hadamard product between a global matrix (nen1,nen2,nelem) and 
    global matrix (nen1,nen2,nelem) then sum rows

    Returns
    -------
    c : numpy array of shape (nen2,nelem)
    '''
    nen,nen2,nelem = A.shape
    nenb,nen2b,nelemb = B.shape
    assert nen==nenb and nen2==nen2b and nelem==nelemb, f'array shapes do not match, {nen} != {nenb}, {nen2} != {nen2b}, {nelem} != {nelemb}'
    c = np.zeros((nen,nelem),dtype=B.dtype)
    for e in range(nelem):
        for j in range(nen2):
            for i in range(nen):
                c[i,e] += A[i,j,e]*B[i,j,e]
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
    qpad = np.zeros((nen,nelem+2),dtype=q.dtype)
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
    assert qL.shape==(nen,) and qR.shape==(nen,), f'shapes do not match {qL.shape} != {nen}, {qR.shape} != {nen}'
    qpad = np.zeros((nen,nelem+2),dtype=q.dtype)
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
    assert qL.shape==(nen,), f'shapes do not match {qL.shape} != {nen}'
    qpad = np.zeros((nen,nelem+1),dtype=q.dtype)
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
    assert qR.shape==(nen,), f'shapes do not match {qR.shape} != {nen}'
    qpad = np.zeros((nen,nelem+1),dtype=q.dtype)
    qpad[:,:-1] = q
    qpad[:,-1] = qR          
    return qpad

@njit
def pad_ndR(q,qR):
    '''
    Take a global vector (nen,dim,nelem) and pad it so that it becomes a global
    vector (nen,dim,nelem+1) where the last element is qR. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+1)
    '''
    nen,dim,nelem = q.shape
    assert qR.shape==(nen,dim), f'shapes do not match {qR.shape} != {nen,dim}'
    qpad = np.zeros((nen,dim,nelem+1),dtype=q.dtype)
    qpad[:,:,:-1] = q
    qpad[:,:,-1] = qR          
    return qpad

@njit
def pad_gm_1dL(q,qL):
    '''
    Take a global vector (nen,nen2, nelem) and pad it so that it becomes a global
    matrix (nen,nen2,nelem+1) where the first element is qL. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+1)
    '''
    nen,nen2,nelem = q.shape
    assert qL.shape==(nen,nen2), f'shapes do not match {qL.shape} != {nen,nen2}'
    qpad = np.zeros((nen,nen2,nelem+1),dtype=q.dtype)
    qpad[:,:,1:] = q
    qpad[:,:,0] = qL         
    return qpad

@njit
def pad_gm_1dR(q,qR):
    '''
    Take a global vector (nen,nen2,nelem) and pad it so that it becomes a global
    vector (nen,nen2,nelem+1) where the last element is qR. Used in Sat calculations.

    Returns
    -------
    qpad : numpy array of shape (nen,nelem+1)
    '''
    nen,nen2,nelem = q.shape
    assert qR.shape==(nen,nen2), f'shapes do not match {qR.shape} != {nen,nen2}'
    qpad = np.zeros((nen,nen2,nelem+1),dtype=q.dtype)
    qpad[:,:,:-1] = q
    qpad[:,:,-1] = qR          
    return qpad

@njit # renamed from fix_satL_1D
def shift_left(q):
    '''
    Take a global vector (nen,nelem) and move the first elem to the last elem.
    This is used for example for periodic cases to create qR from q.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nelem = q.shape
    qfix = np.zeros((nen,nelem),dtype=q.dtype)
    qfix[:,:-1] = q[:,1:]
    qfix[:,-1] = q[:,0]            
    return qfix

@njit
def shift_right(q):
    '''
    Take a global vector (nen,nelem) and move the last elem to the first elem.
    This is used for example for periodic cases to create qL from q.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nelem = q.shape
    qfix = np.zeros((nen,nelem),dtype=q.dtype)
    qfix[:,1:] = q[:,:-1]
    qfix[:,0] = q[:,-1]            
    return qfix

@njit 
def shift_mat_left(A):
    '''
    Take a global matrix (nen,nen2,nelem) and move the first elem to the last elem.
    This is used for example for periodic cases to create AR from A.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nen2,nelem = A.shape
    Afix = np.zeros((nen,nen2,nelem),dtype=A.dtype)
    Afix[:,:,:-1] = A[:,:,1:]
    Afix[:,:,-1] = A[:,:,0]            
    return Afix

@njit
def shift_mat_right(A):
    '''
    Take a global matrix (nen,nen2,nelem) and move the last elem to the first elem.
    This is used for example for periodic cases to create AL from A.

    Returns
    -------
    qfix : numpy array of shape (nen,nelem)
    '''
    nen,nen2,nelem = A.shape
    Afix = np.zeros((nen,nen2,nelem),dtype=A.dtype)
    Afix[:,:,1:] = A[:,:,:-1]
    Afix[:,:,0] = A[:,:,-1]            
    return Afix

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
    qfix = np.zeros((nen,nen1,nelem),dtype=q.dtype)
    qfix[:,:,:-1] = q[:,:,1:]
    qfix[:,:,-1] = q[:,:,0]            
    return qfix

@njit
def reshape_to_meshgrid_2D(q,nen,nelemx,nelemy):
    ''' take a 2D vector q in the shape (nen**2,nelemx*nelemy) and reshape
    to a 2D mesh in the shape (nen*nelem, nen*nelemy) as would be created
    by meshgrid. Can think of this array ordering being the actual bird's 
    eye view of the mesh. '''
    assert q.shape == (nen**2,nelemx*nelemy), f'array shape does not match {q.shape} != {(nen**2,nelemx*nelemy)}'
        
    Q = np.zeros((nen*nelemx, nen*nelemy),dtype=q.dtype)
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
    F = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype)  
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(i,nen_neq):
                f = flux(q[i,e],q[j,e])
                F[i,j,e] = f
                if i != j:
                    F[j,i,e] = f
    return F

@njit
def build_F_vol_sca_2d(q, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 1 
    solution vector q, the number of equations per node, and a 2-point 
    flux function. Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    Fx = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype)  
    Fy = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype) 
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(i,nen_neq):
                fx, fy = flux(q[i,e],q[j,e])
                Fx[i,j,e] = fx
                Fy[i,j,e] = fy
                if i != j:
                    Fx[j,i,e] = fx
                    Fy[j,i,e] = fy
    return Fx, Fy

@njit
def build_F_vol_sys(neq, q, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 1 
    solution vector q, the number of equations per node, and a 2-point 
    flux function. Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    F = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype)   
    nen = int(nen_neq / neq)
    for e in range(nelem):
        for i in range(nen):
            idxi = i*neq
            idxi2 = (i+1)*neq
            for j in range(i,nen):
                idxj = j*neq
                idxj2 = (j+1)*neq
                diag = np.diag(flux(q[idxi:idxi2,e],q[idxj:idxj2,e]))
                F[idxi:idxi2,idxj:idxj2,e] = diag
                if i != j:
                    F[idxj:idxj2,idxi:idxi2,e] = diag
    return F

@njit
def build_F_vol_sys_2d(neq, q, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 1 
    solution vector q, the number of equations per node, and a 2-point 
    flux function. Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    Fx = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype)   
    Fy = np.zeros((nen_neq,nen_neq,nelem),dtype=q.dtype)   
    nen = int(nen_neq / neq)
    for e in range(nelem):
        for i in range(nen):
            for j in range(i,nen):
                idxi = i*neq
                idxi2 = (i+1)*neq
                idxj = j*neq
                idxj2 = (j+1)*neq
                fx, fy = flux(q[idxi:idxi2,e],q[idxj:idxj2,e])
                Fx[idxi:idxi2,idxj:idxj2,e] = np.diag(fx)
                Fy[idxi:idxi2,idxj:idxj2,e] = np.diag(fy)
                if i != j:
                    Fx[idxj:idxj2,idxi:idxi2,e] = np.diag(fx)
                    Fy[idxj:idxj2,idxi:idxi2,e] = np.diag(fy)
    return Fx, Fy

# Don't use nopython @njit in case we need to pass class objects in ec_flux
# June 2023: I put it back in becuase the keyword default of False is being depreciated and 
# it threw a warning. Maybe I can get away with this if it supports class functions?
@njit
def build_F_sca(q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function. for scalar equations, neq=1 '''
    nen_neq, nelem = q1.shape 
    F = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype)  
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(nen_neq):
                f = flux(q1[i,e],q2[j,e])
                F[i,j,e] = f
    return F

@njit
def build_F_sca_2d(q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function. for scalar equations, neq=1, simultaneously for x and y fluxes '''
    nen_neq, nelem = q1.shape 
    Fx = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype)  
    Fy = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype) 
    for e in range(nelem):
        for i in range(nen_neq):
            for j in range(nen_neq):
                fx, fy = flux(q1[i,e],q2[j,e])
                Fx[i,j,e] = fx
                Fy[i,j,e] = fy
    return Fx, Fy

@njit
def build_F_sys(neq, q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function '''
    nen_neq, nelem = q1.shape 
    F = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype)  
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
def build_F_sys_2d(neq, q1, q2, flux):
    ''' builds a Flux differencing matrix (used for Hadamard form) given 2 
    solution vectors q1, q2, the number of equations per node, and a 2-point 
    flux function, simultaneously for x and y fluxes '''
    nen_neq, nelem = q1.shape 
    Fx = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype)  
    Fy = np.zeros((nen_neq,nen_neq,nelem),dtype=q1.dtype)  
    nen = int(nen_neq / neq)
    for e in range(nelem):
        for i in range(nen):
            for j in range(nen):
                idxi = i*neq
                idxi2 = (i+1)*neq
                idxj = j*neq
                idxj2 = (j+1)*neq
                fx, fy = flux(q1[idxi:idxi2,e],q2[idxj:idxj2,e])
                Fx[idxi:idxi2,idxj:idxj2,e] = np.diag(fx)
                Fy[idxi:idxi2,idxj:idxj2,e] = np.diag(fy)
    return Fx, Fy



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
def kron_lm_ldiag(Dx, diag):
    '''
    Compute the Kronecker product of a dense matrix Dx and a diagonal matrix defined by diag.

    Parameters
    ----------
    Dx : ndarray
        A dense matrix of shape (m, n).
    diag : ndarray
        A 1D array representing the diagonal entries of a p x p diagonal matrix.

    Returns
    -------
    result : ndarray
        The resulting matrix of the Kronecker product with shape (m * p, n * p).
    '''
    m, n = Dx.shape
    p = len(diag)
    
    # Allocate the result matrix
    result = np.zeros((m * p, n * p), dtype=Dx.dtype)
    
    # Populate the result matrix by scaling each block of Dx by diag[k]
    for i in range(m):
        for j in range(n):
            # Scale Dx[i, j] by each entry in diag and place it in the correct block
            result[i * p:(i + 1) * p, j * p:(j + 1) * p] = Dx[i, j] * np.diag(diag)
    
    return result

@njit
def kron_ldiag_lm(diag, Dx):
    '''
    Compute the Kronecker product of a diagonal matrix (represented by diag) and a dense matrix Dx.

    Parameters
    ----------
    diag : ndarray
        A 1D array representing the diagonal entries of a p x p diagonal matrix.
    Dx : ndarray
        A dense matrix of shape (m, n).

    Returns
    -------
    result : ndarray
        The resulting matrix of the Kronecker product with shape (p * m, p * n).
    '''
    p = len(diag)      # Size of the diagonal matrix
    m, n = Dx.shape    # Shape of the dense matrix Dx
    
    # Allocate the result matrix
    result = np.zeros((p * m, p * n), dtype=Dx.dtype)
    
    # Populate the result matrix by scaling each block of Dx by diag[i]
    for i in range(p):
        # Scale the entire Dx matrix by diag[i] and place it in the appropriate block
        result[i * m:(i + 1) * m, i * n:(i + 1) * n] = diag[i] * Dx
    
    return result

@njit
def repeat_neq_gv(q,neq_node):
    ''' take array of shape (nen,nelem) and return (nen*neq_node,nelem)
        where the value on each node is repeat neq_node times. 
        Note: just as fast as np.repeat(q,neq_node,0) but this 
              is not compatible with jit (axis argument not supported)'''
    nen, nelem = q.shape
    qn = np.zeros((nen*neq_node,nelem),dtype=q.dtype) 
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
    An = np.zeros((nen*neq_node,nen2*neq_node,nelem),dtype=A.dtype) 
    for e in range(nelem):
        for i in range(nen):
            for n in range(neq_node):
                i2 = i*neq_node + n
                for j in range(nen2):
                    An[i2,j*neq_node+n::neq_node,e] = A[i,j,e]
    return An

@njit
def unkron_neq_gm(A,neq):
    ''' take array of shape (nen*neq_node,nen2*neq_node,nelem) and return (nen,nen2,nelem)
        undoes the proper kronecker product for the operator acting on a vector (nen2*neq_node,nelem). '''
    nen_neq, nen_neq2, nelem = A.shape
    nen, nen2 = nen_neq // neq, nen_neq2 // neq
    An = np.zeros((nen,nen2,nelem),dtype=A.dtype) 
    for e in range(nelem):
        for i in range(nen):
            i2 = i*neq
            for j in range(nen2):
                j2 = j*neq
                An[i,j,e] = A[i2,j2,e]
    return An

@njit
def kron_neq_lm(A,neq_node):
    ''' take array of shape (nen,nen2) and return (nen*neq_node,nen2*neq_node)
        the proper kronecker product for the operator acting on a vector (nen2*neq_node,nelem). '''
    nen, nen2 = A.shape
    An = np.zeros((nen*neq_node,nen2*neq_node),dtype=A.dtype) 
    for i in range(nen):
        for n in range(neq_node):
            i2 = i*neq_node + n
            for j in range(nen2):
                An[i2,j*neq_node+n::neq_node] = A[i,j]
    return An

@njit
def unkron_neq_lm(A,neq):
    ''' take array of shape (nen*neq_node,nen2*neq_node) and return (nen,nen2)
        undoes the proper kronecker product for the operator acting on a vector (nen2*neq_node). '''
    nen_neq, nen_neq2 = A.shape
    nen, nen2 = nen_neq // neq, nen_neq2 // neq
    An = np.zeros((nen,nen2),dtype=A.dtype) 
    for i in range(nen):
        i2 = i*neq
        for j in range(nen2):
            j2 = j*neq
            An[i,j] = A[i2,j2]
    return An

@njit
def repeat_neq_lv(q,neq_node):
    ''' take array of shape (nen) and return (nen*neq_node)
        where the value on each node is repeat neq_node times. 
        Note: just as fast as np.repeat(q,neq_node,0) but this 
              is not compatible with jit (axis argument not supported)'''
    nen = len(q)
    qn = np.zeros(nen*neq_node,dtype=q.dtype) 
    for i in range(nen):
        for j in range(i*neq_node,i*neq_node+neq_node):
            qn[j] = q[i]
    return qn

@njit
def sparse_block_diag(A):
    ''' given a matrix A of shape (nen,nen2,nelem), return a sparse LHS
    of shape (nen*nelem,nen2*nelem) with the (nen,nen2) blocks of A on the diag'''
    nen,nen2,nelem = A.shape
    #mat = sp.lil_matrix((nen*nelem,nen2*nelem))
    mat = np.zeros((nen*nelem,nen2*nelem),dtype=A.dtype)
    for e in range(nelem):
        mat[nen*e:nen*(e+1),nen2*e:nen2*(e+1)] = A[:,:,e]
    #mat.eliminate_zeros()
    return mat

@njit
def sparse_block_diag_R_1D(A):
    ''' given a matrix A of shape (nen,nen2,nelem), return a sparse LHS
    of shape (nen*nelem,nen2*nelem) with the (nen,nen2) blocks of A on the 
    right-side of the diag (and wraps around). Appropriate for 1D.'''
    nen,nen2,nelem = A.shape
    #mat = sp.lil_matrix((nen*nelem,nen2*nelem))
    mat = np.zeros((nen*nelem,nen2*nelem),dtype=A.dtype)
    for e in range(nelem-1):
        mat[nen*e:nen*(e+1),nen2*(e+1):nen2*(e+2)] = A[:,:,e]
    mat[nen*(nelem-1):,:nen2] = A[:,:,nelem-1]
    #mat.eliminate_zeros()
    return mat

@njit
def sparse_block_diag_L_1D(A):
    ''' given a matrix A of shape (nen,nen2,nelem), return a sparse LHS
    of shape (nen*nelem,nen2*nelem) with the (nen,nen2) blocks of A on the 
    left-side of the diag (and wraps around). Appropriate for 1D.'''
    nen,nen2,nelem = A.shape
    #mat = sp.lil_matrix((nen*nelem,nen2*nelem))
    mat = np.zeros((nen*nelem,nen2*nelem),dtype=A.dtype)
    for e in range(1,nelem):
        mat[nen*e:nen*(e+1),nen2*(e-1):nen2*e] = A[:,:,e]
    mat[:nen,nen2*(nelem-1):nen2*nelem] = A[:,:,0]
    #mat.eliminate_zeros()
    return mat

@njit
def assemble_satx_2d(mat_list,nelemx,nelemy):
    ''' given a list of matrices of shape (nen,nen2,nelem[idx]), 
    put them back in global order (nen,nen2,nelem)
    where each entry is a list of matrices that would be selected
     by e.g. satx.vol_x_mat[idx], where idx is one row in x '''''
    nelemy2 = len(mat_list)
    assert nelemy2 == nelemy, f'nelemy does not match {nelemy2} != {nelemy}'
    nen1,nen2,nelemx2 = mat_list[0].shape
    assert nelemx2 == nelemx, f'nelemx does not match {nelemx2} != {nelemx}'
    
    mat_glob = np.zeros((nen1, nen2, nelemx*nelemy),dtype=mat_list[0].dtype)
    for ey in range(nelemy):
        for ex in range(nelemx):
            idx = ex*nelemy + ey
            mat_glob[:,:,idx] = mat_list[ey][:,:,ex]
    return mat_glob

@njit
def assemble_saty_2d(mat_list,nelemx,nelemy):
    ''' given a list of matrices of shape (nen,nen2,nelem[idx]), 
    put them back in global order (nen,nen2,nelem)
    where each entry is a list of matrices that would be selected
     by e.g. satx.vol_y_mat[idx], where idx is one row in y '''
    nelemx2 = len(mat_list)
    assert nelemx2 == nelemx, f'nelemx does not match {nelemx2} != {nelemx}'
    nen1,nen2,nelemy2 = mat_list[0].shape
    assert nelemy2 != nelemy, f'nelemy does not match {nelemy2} != {nelemy}'
    
    mat_glob = np.zeros((nen1, nen2, nelemx*nelemy),dtype=mat_list[0].dtype)
    for ex in range(nelemx):
        for ey in range(nelemy):
            idx = ex*nelemy + ey
            mat_glob[:,:,idx] = mat_list[ex][:,:,ex]
    return mat_glob

@njit
def VolxVoly_had_Fvol_diff(Volx,Voly,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the Volx and Voly
    matrices and the Fvol matrices, then sum the rows. Made for 2d.

    Parameters
    ----------
    Volx, Voly : global matrices
        The matrices that represent the volume operators in the x and y directions
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx + dEydy on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq
    tol = 1e-13

    # Sanity checks on sizes of arrays - can comment this out later
    nenb, nenb2, nelemb = Volx.shape
    nenc, nenc2, nelemc = Voly.shape
    assert nelemb == nelem and nelemc == nelem and nenb == nen and nenc == nen and nenb2 == nen and nenc2 == nen, \
        f'Number of elements do not match {nelemb} != {nelem} or {nelemc} or {nelem} or {nenb} or {nen} or {nenc} or {nen}'

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    
    # loop for each element
    for e in range(nelem):
        for row in range(nen):
            qidx = row*neq
            for col in range(row+1,nen):
                Volx_val = Volx[row, col, e]
                Voly_val = Voly[row, col, e]

                if abs(Volx_val) > tol:

                    qidxT = col*neq
                    fx, fy = flux(q[qidxT:qidxT+neq, e], q[qidx:qidx+neq, e])

                    # add the result of the hadamard product
                    c[qidx:qidx+neq, e] -= fx * Volx_val

                    # Volx is skew-symmetric with respect to H, so reuse the flux calculation
                    c[qidxT:qidxT+neq, e] -= fx * Volx[col, row, e]

                    # maybe reuse for Voly as well
                    if abs(Voly_val) > tol:

                            # add the result of the hadamard product
                            c[qidx:qidx+neq, e] -= fy * Voly_val

                            # add the result of the hadamard product to the transpose
                            c[qidxT:qidxT+neq, e] -= fy * Voly[col, row, e]
                
                elif abs(Voly_val) > tol:

                    qidxT = col*neq
                    _, fy = flux(q[qidxT:qidxT+neq, e], q[qidx:qidx+neq, e])

                    # add the result of the hadamard product
                    c[qidx:qidx+neq, e] -= fy * Voly_val

                    # Voly is skew-symmetric with respect to H, so reuse the flux calculation
                    c[qidxT:qidxT+neq, e] -= fy * Voly[col, row, e]
    
    return c

@njit
def Sat2d_had_Fsat_diff_periodic(tax,tay,tbx,tby,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the Volx and Voly
    matrices and the Fvol matrices, then sum the rows. Made for 2d.

    Parameters
    ----------
    taTx, taTy, tbx, tby : global matrices
        The matrices that represent the boundary operators in the x and y directions
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx + dEydy on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq
    tol = 1e-13

    # Sanity checks on sizes of arrays - can comment this out later
    nenb, nenb2, nelemb = tax.shape
    nenc, nenc2, nelemc = tay.shape
    nend, nend2, nelemd = tbx.shape
    nene, nene2, neleme = tby.shape
    assert nelemb == nelem and nelemc == nelem and nelemd == nelem and neleme == nelem,\
        f'Number of elements do not match {nelemb} != {nelem} or {nelemc} or {nelem} or {nelemd} or {nelem} or {neleme} or {nelem}'
    assert nenb == nen and nenc == nen and nend == nen and nene == nen,\
        f'Number of nodes do not match {nenb} != {nen} or {nenc} or {nen} or {nend} or {nen} or {nene} or {nen}'
    assert nenb2 == nen and nenc2 == nen and nend2 == nen and nene2 == nen, \
        f'Number of nodes do not match {nenb2} != {nen} or {nenc2} or {nen} or {nend2} or {nen} or {nene2} or {nen}'
    

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    
    # loop for each element
    for e in range(nelem): # will ignore right-most interface by periodic BC (same as leftmost)
        # here think of e as either the looping over elements and considering the left interface,
        # or looping over each interface e and stopping before hitting the rightmost 
        if e == 0:
            # periodic BC: leftmost interface is the same as rightmost
            qL = q[:,-1]
            qR = q[:,0]
            eb = -1
        else:
            qL = q[:,e-1]
            qR = q[:,e]
            eb = e-1

        for row in range(nen):
            qidx = row*neq

            for col in range(nen):
                tax_val = tax[col, row, e]
                tay_val = tay[col, row, e]
                tbx_val = tbx[row, col, e]
                tby_val = tby[row, col, e]
                flux_used = False

                if abs(tax_val) > tol:
                    flux_used = True
                    qidxT = col*neq
                    
                    fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])
                    c[qidxT:qidxT+neq, e] += fx * tax_val

                if abs(tay_val) > tol:
                    if not flux_used:
                        flux_used = True
                        qidxT = col*neq
                        fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])
                    
                    c[qidxT:qidxT+neq, e] += fy * tay_val
                
                if abs(tbx_val) > tol:
                    if not flux_used:
                        flux_used = True
                        qidxT = col*neq
                        fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                    c[qidx:qidx+neq, eb] -= fx * tbx_val

                if abs(tby_val) > tol:
                    if not flux_used:
                        qidxT = col*neq
                        _, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                    c[qidx:qidx+neq, eb] -= fy * tby_val
    
    return c

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