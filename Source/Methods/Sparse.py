#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 27 2024

@author: bercik
"""

import numpy as np
from numba import njit, types
from numba.typed import List

@njit
def dense_to_csr(A):
    '''
    Converts a dense 2D matrix (nen1, nen2) to CSR format.

    Parameters
    ----------
    A : numpy array of shape (nen1, nen2)

    Returns
    -------
        csr_data : tuple (data, indices, indptr)
        data : non-zero values of the matrix
        indices : column indices of the data elements
        indptr : row pointers indicating the start of each row in the data
    '''
    tol = 1e-13
    nen1, nen2 = A.shape
    data = []
    indices = []
    indptr = [0]
    
    for i in range(nen1):
        row_start = len(data)
        for j in range(nen2):
            if abs(A[i, j]) > tol:
                data.append(A[i, j])
                indices.append(j)
        indptr.append(row_start + len(data) - row_start)
    
    return np.array(data), np.array(indices), np.array(indptr)

@njit
def gm_to_sparse(A):
    '''
    Converts a global 3D matrix (nen1, nen2, nelem) into a list of CSR matrices,
    where each CSR matrix corresponds to a slice A[:,:,e].

    Parameters
    ----------
    A : numpy array of shape (nen1, nen2, nelem)

    Returns
    -------
    csr_list : List of CSR matrices in the form (data, indices, indptr)
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    '''
    nen1, nen2, nelem = A.shape
    csr_list = List()

    for e in range(nelem):
        # Extract the slice A[:,:,e] and convert it to CSR
        csr_data = dense_to_csr(A[:, :, e])
        csr_list.append(csr_data)

    return csr_list

@njit
def csr_mv(data, indices, indptr, vec):
    '''
    Perform sparse matrix-vector multiplication in CSR format.

    Parameters
    ----------
    data : 1D array
        Non-zero values of the CSR matrix
    indices : 1D array
        Column indices corresponding to values in `data`
    indptr : 1D array
        Row pointers to start of rows in `data`
    vec : 1D array
        Vector to multiply with the CSR matrix

    Returns
    -------
    result : 1D array
        The result of the matrix-vector multiplication
    '''
    nen1 = len(indptr) - 1
    result = np.zeros(nen1, dtype=vec.dtype)
    
    for i in range(nen1):
        for jj in range(indptr[i], indptr[i+1]):
            result[i] += data[jj] * vec[indices[jj]]
    
    return result

@njit
def sparse_gm_gv(csr_list, b):
    '''
    Perform global matrix-vector multiplication using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    b : numpy array of shape (nen2, nelem)
        The global vector for multiplication

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(csr_list[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(csr_list)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=b.dtype)
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        # Get the CSR data for the current element
        data, indices, indptr = csr_list[e]
        # Perform sparse matrix-vector multiplication for this element
        c[:, e] = csr_mv(data, indices, indptr, b[:, e])
    
    return c

if __name__ == '__main__':
    # Example usage
    A = np.random.rand(10, 11, 3)
    b = np.random.rand(11, 3)
    csr_list = gm_to_sparse(A)
    c = sparse_gm_gv(csr_list, b)

    from Functions import gm_gv
    c2 = gm_gv(A, b)
    print(np.max(abs(c-c2))) # Should be zero