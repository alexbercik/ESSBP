#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 27 2024

@author: bercik
"""

import numpy as np
from numba import njit, types, literal_unroll
from numba.typed import List
tol = 1e-13 # tolerance for zero values

@njit
def lm_to_sp(A):
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
    nen1, nen2 = A.shape
    data = []
    indices = []
    indptr = [0]
    data_len = 0
    
    for i in range(nen1):
        for j in range(nen2):
            if abs(A[i, j]) > tol:
                data.append(A[i, j])
                indices.append(j)
                data_len += 1
        indptr.append(data_len)
    
    return np.array(data), np.array(indices), np.array(indptr)


@njit
def gm_to_sp(A):
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
        csr_data = lm_to_sp(A[:, :, e])
        csr_list.append(csr_data)

    return csr_list

@njit
def sp_to_lm(csr):
    '''
    Converts a sparse CSR format to dense 2D matrix (nen1, nen2).

    Parameters
    ----------
    csr_data : tuple (data, indices, indptr)
        data : non-zero values of the matrix
        indices : column indices of the data elements
        indptr : row pointers indicating the start of each row in the data

    Returns
    -------
        A : numpy array of shape (nen1, nen2)
    '''
    # Get the CSR data for the current element
    data, indices, indptr = csr

    nen1 = len(indptr) - 1
    nen2 = np.max(indices) + 1
    A = np.zeros((nen1, nen2), dtype=data.dtype)
    
    for i in range(nen1):
        for j in range(indptr[i], indptr[i+1]):
            idx = indices[j]
            A[i,idx] = data[j]

    return A

@njit
def sp_to_gm(csr_list):
    '''
    Converts a list of CSR matrices to a global 3D matrix (nen1, nen2, nelem),
    where each CSR matrix corresponds to a slice A[:,:,e].

    Parameters
    ----------
    csr : List of CSR matrices in the form (data, indices, indptr)
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    A : numpy array of shape (nen1, nen2, nelem)

    Returns
    -------
    A : numpy array of shape (nen1, nen2, nelem)
    '''

    nelem = len(csr_list)
    nen1 = len(csr_list[0][2]) - 1
    nen2 = np.max(csr_list[0][1]) + 1
    A = np.zeros((nen1, nen2, nelem), dtype=csr_list[0][0].dtype)
    
    for e in range(nelem):
        A[:, :, e] = sp_to_lm(csr_list[e])
    
    return A

@njit
def lm_lv(csr, vec):
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

    # Get the CSR data for the current element
    data, indices, indptr = csr

    nen1 = len(indptr) - 1
    result = np.zeros(nen1, dtype=vec.dtype)
    
    for i in range(nen1):
        for jj in range(indptr[i], indptr[i+1]):
            result[i] += data[jj] * vec[indices[jj]]
    
    return result

@njit
def gm_gv(csr_list, b):
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
        c[:, e] = lm_lv(csr_list[e], b[:, e])
    
    return c

@njit
def lm_gv(csr, b):
    '''
    Perform local matrix - global vector multiplication using a CSR matrix.

    Parameters
    ----------
    csr : CSR matrix
        a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    b : numpy array of shape (nen2, nelem)
        The global vector for multiplication

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(csr[2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nen2, nelem = np.shape(b)  # Number of elements (same as the third dimension of the original tensor)

    if nen1 != nen2:
        raise ValueError('Dimensions do not match')
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=b.dtype)
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c[:, e] = lm_lv(csr, b[:, e])
    
    return c

@njit
def set_gm_sparsity(gm_list):
    '''
    Set the sparsity pattern of a list of global dense matrices (shape: nen1, nen2, nelem)
    to the intersection of all sparsity patterns.

    Parameters
    ----------
    csr_lists : List of global matrices
        Each element in the list is an ndarray (nen1, nen2, nelem)

    Returns
    -------
    sparsity : tuple (indices, indptr)
        The sparsity pattern of the intersection of all given matrices
    '''
    ngm = len(gm_list)
    nen1, nen2, nelem = gm_list[0].shape

    for gm in gm_list:
        if gm.shape != (nen1, nen2, nelem):
            raise ValueError('Dimensions do not match', gm.shape, (nen1, nen2, nelem))
        
    sp_gm_list = List()

    for e in range(nelem):
        # Initialize CSR format arrays
        indices = []
        indptr = []
        indptr.append(0)
        data_len = 0

        for i in range(nen1):
            for j in range(nen2):
                any_above_tol = False
                for k in range(ngm):
                    if abs(gm_list[k][i, j, e]) >= tol:
                        any_above_tol = True
                        break
                if any_above_tol:
                    data_len += 1
                    indices.append(j)
            indptr.append(data_len)

        sp_gm_list.append((np.array(indices), np.array(indptr)))
    
    return sp_gm_list



@njit
def lm_lm_had_diff(A,B):
    '''
    Compute the Hadamard product between two CSR matrices with potentially 
    different sparsity patterns, then sum rows

    Parameters
    ----------
    A (data1, indices1, indptr1) : CSR representation of the first matrix
    B (data2, indices2, indptr2) : CSR representation of the second matrix

    Returns
    -------
    c : numpy array of shape (nen2,nelem)
    '''
    # Get the CSR data for the current element
    data1, indices1, indptr1 = A
    data2, indices2, indptr2 = B
    nen1 = len(indptr1) - 1

    c = np.zeros(nen1, dtype=data2.dtype)
    
    for row in range(nen1):
        data_srt1 = indptr1[row]
        data_end1 = indptr1[row+1]
        data_srt2 = indptr2[row]
        data_end2 = indptr2[row+1]
        for col_ptr1 in range(data_srt1, data_end1):
            col1 = indices1[col_ptr1]
            for col_ptr2 in range(data_srt2, data_end2):
                col2 = indices2[col_ptr2]
                if col1 == col2:
                    c[row] += data1[col_ptr1] * data2[col_ptr2]
    
    return c

@njit
def lmT_lm_had_diff(AT,B):
    '''
    Compute the Hadamard product between two CSR matrices with potentially 
    different sparsity patterns, then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT (data1, indices1, indptr1) : CSR representation of the first matrix
    B (data2, indices2, indptr2) : CSR representation of the second matrix

    Returns
    -------
    c : numpy array of shape (nen2,nelem)
    '''
    # Get the CSR data for the current element
    data1, indices1, indptr1 = AT
    data2, indices2, indptr2 = B
    nen1 = len(indptr1) - 1

    c = np.zeros(nen1, dtype=data2.dtype)
    
    for row in range(nen1):
        data_srt1 = indptr1[row]
        data_end1 = indptr1[row+1]
        data_srt2 = indptr2[row]
        data_end2 = indptr2[row+1]
        for col_ptr1 in range(data_srt1, data_end1):
            col1 = indices1[col_ptr1]
            for col_ptr2 in range(data_srt2, data_end2):
                col2 = indices2[col_ptr2]
                if col1 == col2:
                    c[col1] += data1[col_ptr1] * data2[col_ptr2]
    
    return c

@njit
def gm_gm_had_diff(A,B):
    '''
    Compute the hadamard product between a sparse global matrix (list of CSR matrices) 
    and another sparse global matrix, then sum rows

    Parameters
    ----------
    A, B : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(A[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(A)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B[0][0].dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lm_lm_had_diff(A[e], B[e])
    
    return c

@njit
def gmT_gm_had_diff(AT,B):
    '''
    Compute the hadamard product between a sparse global matrix (list of CSR matrices) 
    and another sparse global matrix, then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT, B : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(AT[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(AT)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B[0][0].dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lmT_lm_had_diff(AT[e], B[e])
    
    return c

@njit
def lm_gm_had_diff(A,B):
    '''
    Compute the hadamard product between a sparse local matrix (CSR matrix) 
    and another sparse global matrix (list of CSR matrices), then sum rows

    Parameters
    ----------
    A (data1, indices1, indptr1) : CSR representation of the first matrix
    B : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(A[2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(B)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B[0][0].dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lm_lm_had_diff(A, B[e])
    
    return c

@njit
def lmT_gm_had_diff(AT,B):
    '''
    Compute the hadamard product between a sparse local matrix (CSR matricex) 
    and another sparse global matrix (list of CSR matrices), then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT (data1, indices1, indptr1) : CSR representation of the first matrix
    B : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(AT[2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(B)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B[0][0].dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lmT_lm_had_diff(AT, B[e])
    
    return c

@njit 
def build_F_vol_sys(neq, q, flux, sparsity_unkronned, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.
    Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    F_vol = List()
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity_unkronned[e][0]
        indptr = sparsity_unkronned[e][1]
        new_indices = sparsity[e][0]
        new_indptr = sparsity[e][1]
        new_data = np.zeros((len(new_indices)), dtype=q.dtype)       
        colptrs = np.zeros(nen, dtype=np.int32)
        
        for i in range(nen): # loop over rows, NOT kroned with neq
            idxi = i * neq # actual dense initial row index
            idxi2 = (i + 1) * neq # actual dense final row index

            col_start = indptr[i]
            col_end = indptr[i + 1]

            for j in range(i,nen): # loop over columns, NOT kroned with neq

                col_start_T = indptr[j]
                col_end_T = indptr[j + 1]

                add_entry = (j in indices[col_start:col_end])
                add_entry_T = (i in indices[col_start_T:col_end_T]) and (i != j)

                if add_entry or add_entry_T:

                    idxj = j * neq # actual dense initial column index
                    idxj2 = (j + 1) * neq  # actual dense final colum index
                    diag = flux(q[idxi:idxi2, e], q[idxj:idxj2, e])

                    for k in range(neq):
                        new_row = i * neq + k # actual dense row index
                        new_col = j * neq + k  # actual dense column index
                        
                        if add_entry:
                            new_col_start = new_indptr[new_row]
                            new_col_ptr = new_col_start + colptrs[i]
                            new_data[new_col_ptr] = diag[k]

                        if add_entry_T:
                            new_col_start = new_indptr[new_col]
                            new_col_ptr = new_col_start + colptrs[j]
                            new_data[new_col_ptr] = diag[k]

                    if add_entry: colptrs[i] += 1
                    if add_entry_T: colptrs[j] += 1


        F_vol.append((new_data, new_indices, new_indptr))
    return F_vol

@njit 
def build_F_sys(neq, q1, q2, flux, sparsity_unkronned, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen_neq, nelem = q1.shape 
    F = List()
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity_unkronned[e][0]
        indptr = sparsity_unkronned[e][1]
        new_indices = sparsity[e][0]
        new_indptr = sparsity[e][1]
        new_data = np.zeros((len(new_indices)), dtype=q1.dtype)       
        
        for i in range(nen): # loop over rows, NOT kroned with neq
            idxi = i * neq # actual dense initial row index
            idxi2 = (i + 1) * neq # actual dense final row index
            colptr = 0
            col_start = indptr[i]
            col_end = indptr[i + 1]

            for j in range(nen): # loop over columns, NOT kroned with neq
                if (j in indices[col_start:col_end]):

                    idxj = j * neq # actual dense initial column index
                    idxj2 = (j + 1) * neq  # actual dense final colum index
                    diag = flux(q1[idxi:idxi2, e], q2[idxj:idxj2, e])

                    for k in range(neq):
                        new_row = i * neq + k # actual dense row index
                        #new_col = j * neq + k # actual dense column index
                        
                        new_col_start = new_indptr[new_row]
                        new_col_ptr = new_col_start + colptr
                        new_data[new_col_ptr] = diag[k]

                    colptr += 1

        F.append((new_data, new_indices, new_indptr))
    return F
    
@njit 
def build_F_vol_sys_2d(neq, q, flux, xsparsity_unkronned, xsparsity,
                                     ysparsity_unkronned, ysparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.
    Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    Fx_vol = List()
    Fy_vol = List()
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        xindices = xsparsity_unkronned[e][0]
        xindptr = xsparsity_unkronned[e][1]
        xnew_indices = xsparsity[e][0]
        xnew_indptr = xsparsity[e][1]
        yindices = ysparsity_unkronned[e][0]
        yindptr = ysparsity_unkronned[e][1]
        ynew_indices = ysparsity[e][0]
        ynew_indptr = ysparsity[e][1]
        xnew_data = np.zeros((len(xnew_indices)), dtype=q.dtype)      
        ynew_data = np.zeros((len(ynew_indices)), dtype=q.dtype)      
        xcolptrs = np.zeros(nen, dtype=np.int32)
        ycolptrs = np.zeros(nen, dtype=np.int32)
        
        for i in range(nen): # loop over rows, NOT kroned with neq
            idxi = i * neq # actual dense initial row index
            idxi2 = (i + 1) * neq # actual dense final row index

            xcol_start = xindptr[i]
            xcol_end = xindptr[i + 1]
            ycol_start = yindptr[i]
            ycol_end = yindptr[i + 1]

            for j in range(i,nen): # loop over columns, NOT kroned with neq

                xcol_start_T = xindptr[j]
                xcol_end_T = xindptr[j + 1]
                ycol_start_T = yindptr[j]
                ycol_end_T = yindptr[j + 1]

                xadd_entry = (j in xindices[xcol_start:xcol_end])
                xadd_entry_T = (i in xindices[xcol_start_T:xcol_end_T]) and (i != j)
                yadd_entry = (j in yindices[ycol_start:ycol_end])
                yadd_entry_T = (i in yindices[ycol_start_T:ycol_end_T]) and (i != j)

                if xadd_entry or xadd_entry_T or yadd_entry or yadd_entry_T:

                    idxj = j * neq # actual dense initial column index
                    idxj2 = (j + 1) * neq  # actual dense final colum index
                    xdiag, ydiag = flux(q[idxi:idxi2, e], q[idxj:idxj2, e])

                    for k in range(neq):
                        new_row = i * neq + k # actual dense row index
                        new_col = j * neq + k  # actual dense column index
                        
                        if xadd_entry:
                            xnew_col_start = xnew_indptr[new_row]
                            xnew_col_ptr = xnew_col_start + xcolptrs[i]
                            xnew_data[xnew_col_ptr] = xdiag[k]

                        if xadd_entry_T:
                            xnew_col_start = xnew_indptr[new_col]
                            xnew_col_ptr = xnew_col_start + xcolptrs[j]
                            xnew_data[xnew_col_ptr] = xdiag[k]

                        if yadd_entry:
                            ynew_col_start = ynew_indptr[new_row]
                            ynew_col_ptr = ynew_col_start + ycolptrs[i]
                            ynew_data[ynew_col_ptr] = ydiag[k]

                        if yadd_entry_T:
                            ynew_col_start = ynew_indptr[new_col]
                            ynew_col_ptr = ynew_col_start + ycolptrs[j]
                            ynew_data[ynew_col_ptr] = ydiag[k]

                    if xadd_entry: xcolptrs[i] += 1
                    if xadd_entry_T: xcolptrs[j] += 1
                    if yadd_entry: ycolptrs[i] += 1
                    if yadd_entry_T: ycolptrs[j] += 1


        Fx_vol.append((xnew_data, xnew_indices, xnew_indptr))
        Fy_vol.append((ynew_data, ynew_indices, ynew_indptr))
    return Fx_vol, Fy_vol

@njit 
def build_F_sys_2d(neq, q1, q2, flux, xsparsity_unkronned, xsparsity,
                                      ysparsity_unkronned, ysparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen_neq, nelem = q1.shape 
    Fx = List()
    Fy = List()
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        xindices = xsparsity_unkronned[e][0]
        xindptr = xsparsity_unkronned[e][1]
        xnew_indices = xsparsity[e][0]
        xnew_indptr = xsparsity[e][1]
        xnew_data = np.zeros((len(xnew_indices)), dtype=q1.dtype)   
        yindices = ysparsity_unkronned[e][0]
        yindptr = ysparsity_unkronned[e][1]
        ynew_indices = ysparsity[e][0]
        ynew_indptr = ysparsity[e][1]
        ynew_data = np.zeros((len(ynew_indices)), dtype=q1.dtype)    
        
        for i in range(nen): # loop over rows, NOT kroned with neq
            idxi = i * neq # actual dense initial row index
            idxi2 = (i + 1) * neq # actual dense final row index
            xcolptr = 0
            ycolptr = 0
            xcol_start = xindptr[i]
            xcol_end = xindptr[i + 1]
            ycol_start = yindptr[i]
            ycol_end = yindptr[i + 1]

            for j in range(nen): # loop over columns, NOT kroned with neq
                
                xadd_entry = (j in xindices[xcol_start:xcol_end])
                yadd_entry = (j in yindices[ycol_start:ycol_end])

                if xadd_entry or yadd_entry:

                    idxj = j * neq # actual dense initial column index
                    idxj2 = (j + 1) * neq  # actual dense final colum index
                    xdiag, ydiag = flux(q1[idxi:idxi2, e], q2[idxj:idxj2, e])

                    for k in range(neq):
                        new_row = i * neq + k # actual dense row index
                        #new_col = j * neq + k # actual dense column index
                        
                        if xadd_entry:
                            xnew_col_start = xnew_indptr[new_row]
                            xnew_col_ptr = xnew_col_start + xcolptr
                            xnew_data[xnew_col_ptr] = xdiag[k]
                        
                        if yadd_entry:
                            ynew_col_start = ynew_indptr[new_row]
                            ynew_col_ptr = ynew_col_start + ycolptr
                            ynew_data[ynew_col_ptr] = ydiag[k]

                    if xadd_entry: xcolptr += 1
                    if yadd_entry: ycolptr += 1

        Fx.append((xnew_data, xnew_indices, xnew_indptr))
        Fy.append((ynew_data, ynew_indices, ynew_indptr))
    return Fx, Fy


if __name__ == '__main__':
    import Functions as fn

    gm = np.random.rand(10, 11, 3)
    # add some sparsity
    gm[1, 1, 0] = 0.
    gm[1, 5, 0] = 0.
    gm[4, 2, 2] = 0.
    gm_sp = gm_to_sp(gm)
    gm2 = np.random.rand(10, 11, 3)
    gm2[5, 6, 0] = 0.
    gm2[4, 1, 0] = 0.
    gm2[7, 8, 1] = 0.
    gm2_sp = gm_to_sp(gm2)
    gv = np.random.rand(11, 3)
    lm = np.random.rand(10, 11)
    lm_sp = lm_to_sp(lm)
    lm2 = np.random.rand(10, 11)
    lm2_sp = lm_to_sp(lm2)
    gm_list = [gm, gm]

    c = gm_gv(gm_sp, gv)
    c2 = fn.gm_gv(gm, gv)
    print('test gm_gv:', np.max(abs(c-c2)))

    c = lm_lm_had_diff(lm_sp, lm2_sp)
    c2 = np.sum(np.multiply(lm, lm2),axis=1)
    print('test lm_lm_had_diff:', np.max(abs(c-c2)))

    c = gm_gm_had_diff(gm_sp, gm2_sp)
    c2 = fn.gm_gm_had_diff(gm, gm2)
    print('test gm_gm_had_diff:', np.max(abs(c-c2)))

    sp_gm_list = set_gm_sparsity(gm_list)
    diff = 0.
    for e in range(3):
        diff += np.max(abs(sp_gm_list[e][0] - gm_sp[e][1]))
        diff += np.max(abs(sp_gm_list[e][1] - gm_sp[e][2]))
    print('test set_sparsity:', diff)

    print('test sp_to_gm:', np.max(abs(sp_to_gm(gm_sp)) - gm))

    neq = 2
    nelem = 2
    nen = 3
    q = np.random.rand(neq * nen, nelem)
    q2 = np.random.rand(neq * nen, nelem)
    gm = np.random.rand(nen, nen, nelem)
    # add some sparsity
    gm[1, 0, 0] = 0.
    gm[1, 2, 0] = 0.
    gm[2, 2, 1] = 0.
    gm[0, 2, 1] = 0.
    gm_kron = fn.kron_neq_gm(gm, neq)
    gm_kron_sp = gm_to_sp(gm_kron)
    sparsity_unkronned = set_gm_sparsity([gm])
    sparsity_kron = set_gm_sparsity([gm_kron])
    gm2 = np.random.rand(nen, nen, nelem)
    # add some sparsity
    gm2[0, 1, 0] = 0.
    gm2[1, 1, 0] = 0.
    gm2[1, 2, 1] = 0.
    gm2[1, 1, 1] = 0.
    gm2_kron = fn.kron_neq_gm(gm2, neq)
    gm2_kron_sp = gm_to_sp(gm2_kron)
    sparsity_unkronned2 = set_gm_sparsity([gm2])
    sparsity_kron2 = set_gm_sparsity([gm2_kron])

    @njit
    def flux(q1,q2):
        return 0.5*(q1 + q2)
    
    F = build_F_vol_sys(neq, q, flux, sparsity_unkronned, sparsity_kron)
    c = gm_gm_had_diff(gm_kron_sp, F)
    F2 = fn.build_F_vol_sys(neq, q, flux)
    c2 = fn.gm_gm_had_diff(gm_kron, F2)
    print('test build_F_vol_sys:', np.max(abs(c-c2)))

    F = build_F_sys(neq, q, q2, flux, sparsity_unkronned, sparsity_kron)
    c = gm_gm_had_diff(gm_kron_sp, F)
    F2 = fn.build_F_sys(neq, q, q2, flux)
    c2 = fn.gm_gm_had_diff(gm_kron, F2)
    print('test build_F_sys:', np.max(abs(c-c2)))

    @njit
    def flux(q1,q2):
        return 0.5*(q1 + q2), 2*(q1 - 0.5*q2)

    F, F2 = build_F_vol_sys_2d(neq, q, flux, sparsity_unkronned, sparsity_kron,
                                      sparsity_unkronned2, sparsity_kron2)
    c = gm_gm_had_diff(gm_kron_sp, F) + gm_gm_had_diff(gm2_kron_sp, F2)
    F, F2 = fn.build_F_vol_sys_2d(neq, q, flux)
    c2 = fn.gm_gm_had_diff(gm_kron, F) + fn.gm_gm_had_diff(gm2_kron, F2)
    print('test build_F_vol_sys_2d:', np.max(abs(c-c2)))

    F, F2 = build_F_sys_2d(neq, q, q2, flux, sparsity_unkronned, sparsity_kron,
                                      sparsity_unkronned2, sparsity_kron2)
    c = gm_gm_had_diff(gm_kron_sp, F) + gm_gm_had_diff(gm2_kron_sp, F2)
    F, F2 = fn.build_F_sys_2d(neq, q, q2, flux)
    c2 = fn.gm_gm_had_diff(gm_kron, F) + fn.gm_gm_had_diff(gm2_kron, F2)
    print('test build_F_sys_2d:', np.max(abs(c-c2)))









