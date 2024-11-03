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
def sp_to_lm(csr, nen1=0, nen2=0):
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

    if nen1 == 0:
        nen1 = len(indptr) - 1
    else:
        assert nen1 == len(indptr) - 1, f'Number of rows in CSR matrix {len(indptr) - 1} does not match inputted nen1 {nen1}'
    if nen2 == 0:
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
    nen2 = get_gm_max_numcols(csr_list)
    A = np.zeros((nen1, nen2, nelem), dtype=csr_list[0][0].dtype)
    if nen2 == 0: return A
    
    for e in range(nelem):
        A[:, :, e] = sp_to_lm(csr_list[e], nen1, nen2)
    
    return A

@njit
def get_gm_max_numcols(csr_list):
    max_value = 0
    for csr in csr_list:
        # Loop through each element in csr[1] to find the maximum
        for val in csr[1]:
            if val + 1 > max_value:
                max_value = val + 1
    return max_value

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
def lm_lm(csr1, csr2):
    '''
    Perform sparse matrix-matrix multiplication in CSR format.

    Parameters
    ----------
    csr1 : tuple (data1, indices1, indptr1)
        CSR representation of the first matrix (non-zero values, column indices, row pointers)
    csr2 : tuple (data2, indices2, indptr2)
        CSR representation of the second matrix (non-zero values, column indices, row pointers)

    Returns
    -------
    result_csr : tuple (data, indices, indptr)
        CSR representation of the result of the matrix-matrix multiplication.
    '''

    data1, indices1, indptr1 = csr1
    data2, indices2, indptr2 = csr2
    dtype = data1.dtype

    # Check the number of rows and columns in csr1
    nrows_csr1 = len(indptr1) - 1
    ncols_csr1 = max(indices1) + 1  # Number of columns in csr1

    # Check the number of rows and columns in csr2
    nrows_csr2 = len(indptr2) - 1
    ncols_csr2 = max(indices2) + 1  # Number of columns in csr2

    # Ensure matrices are compatible for multiplication
    if ncols_csr1 > nrows_csr2:
        raise ValueError(f"Matrix dimension mismatch: csr1 has >={ncols_csr1} columns, csr2 has {nrows_csr2} rows.")

    # Initialize result arrays
    nrows = nrows_csr1
    ncols = ncols_csr2
    indptr_result = np.zeros(nrows + 1, dtype=np.int64)

    data_result = []
    indices_result = []

    # Working arrays to accumulate results
    row_accumulator = np.zeros(ncols, dtype=dtype)
    marker = -np.ones(ncols, dtype=np.int64)  # Marker array to track columns in row_accumulator

    # Perform the multiplication
    for i in range(nrows):
        # Reset the row accumulator and marker for each row
        row_accumulator.fill(0)
        marker.fill(-1)

        # Iterate over non-zero elements of row i in csr1
        for jj in range(indptr1[i], indptr1[i+1]):
            col1 = indices1[jj]
            val1 = data1[jj]

            # Multiply with corresponding row in csr2
            for kk in range(indptr2[col1], indptr2[col1+1]):
                col2 = indices2[kk]
                val2 = data2[kk]

                # Accumulate the result in row_accumulator
                if marker[col2] != i:
                    marker[col2] = i
                    row_accumulator[col2] = val1 * val2
                else:
                    row_accumulator[col2] += val1 * val2

        # Now collect all non-zero entries from row_accumulator for this row
        current_length = 0
        for j in range(ncols):
            if marker[j] == i and row_accumulator[j] != 0:
                indices_result.append(j)
                data_result.append(row_accumulator[j])
                current_length += 1
        
        # Update row pointer
        indptr_result[i + 1] = indptr_result[i] + current_length

    # Convert lists to arrays for CSR format
    data_result = np.array(data_result, dtype=dtype)
    indices_result = np.array(indices_result, dtype=np.int64)

    return data_result, indices_result, indptr_result

@njit
def lm_gm(csr, csr_list):
    '''
    multiply a local matrix and global matrix

    Parameters
    ----------
    csr : tuple (data, indices, indptr)
        CSR representation of the first matrix (non-zero values, column indices, row pointers)
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    # Initialize result array
    c = List()
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(len(csr_list)):
        c.append(lm_lm(csr, csr_list[e]))
    
    return c

@njit
def lm_to_lmT(csr,nrows=0,ncols=0):
    '''
    Transpose a sparse matrix in CSR format, returning it as a new CSR format matrix.

    Parameters
    ----------
    csr : tuple
        The input CSR matrix in (data, indices, indptr) format.
    nrows : int
        Number of rows in the original matrix.
    ncols : int
        Number of columns in the original matrix.

    Returns
    -------
    csr_transposed : tuple
        Transposed matrix in CSR format (data, indices, indptr).
    '''
    data, indices, indptr = csr
    nnz = len(data)

    # Infer nrows and ncols from csr
    if nrows == 0: 
        nrows = len(indptr) - 1
    else:
        assert (nrows == len(indptr) - 1), f'Number of rows in CSR matrix {len(indptr) - 1} does not match inputted nrows {nrows}'
    if ncols == 0: ncols = max(indices) + 1
    
    # Step 1: Count non-zeros per column (to allocate space)
    col_counts = np.zeros(ncols, dtype=np.int64)
    for idx in indices:
        col_counts[idx] += 1

    # Step 2: Build indptr for the transposed matrix
    trans_indptr = np.zeros(ncols + 1, dtype=np.int64)
    trans_indptr[1:] = np.cumsum(col_counts)

    # Step 3: Initialize arrays for data and indices
    trans_data = np.zeros(nnz, dtype=data.dtype)
    trans_indices = np.zeros(nnz, dtype=np.int64)
    next_position = np.zeros(ncols, dtype=np.int64)

    # Step 4: Populate transposed data and indices
    for row in range(nrows):
        row_start = indptr[row]
        row_end = indptr[row + 1]
        for i in range(row_start, row_end):
            col = indices[i]
            dest = trans_indptr[col] + next_position[col]
            trans_data[dest] = data[i]
            trans_indices[dest] = row
            next_position[col] += 1

    return trans_data, trans_indices, trans_indptr

@njit
def gm_to_gmT(csr_list, nrows=0, ncols=0):
    '''
    Take the tanspose of a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    c = [lm_to_lmT(csr,nrows,ncols) for csr in csr_list]
    return c

@njit
def lm_lmT(csr1, csr2):
    '''
    Perform sparse matrix-matrix multiplication where the second matrix is transposed, in CSR format.

    Parameters
    ----------
    csr1 : tuple (data1, indices1, indptr1)
        CSR representation of the first matrix A (non-zero values, column indices, row pointers).
    csr2 : tuple (data2, indices2, indptr2)
        CSR representation of the second matrix B (non-zero values, column indices, row pointers).

    Returns
    -------
    result_csr : tuple (data, indices, indptr)
        CSR representation of the result of the matrix-matrix multiplication A * B^T.
    '''
    data1, indices1, indptr1 = csr1
    trans_data, trans_indices, trans_indptr = lm_to_lmT(csr2)
    dtype = trans_data.dtype

    # Infer dimensions from csr1 and csr2
    nrows_A = len(indptr1) - 1
    ncols_B = len(trans_indptr) - 1

    indptr_result = np.zeros(nrows_A + 1, dtype=np.int64)
    data_result = []
    indices_result = []

    row_accumulator = np.zeros(nrows_A, dtype=dtype)
    marker = -np.ones(nrows_A, dtype=np.int64)  # Marker array to track columns in row_accumulator
    
    for i in range(nrows_A):
        row_start = indptr_result[i]
        current_length = 0
        
        # Iterate over non-zero elements of row i in csr1 (A)
        for jj in range(indptr1[i], indptr1[i+1]):
            colA = indices1[jj]
            valA = data1[jj]

            # Multiply with corresponding row in transposed csr2 (B^T)
            for kk in range(trans_indptr[colA], trans_indptr[colA+1]):
                rowB = trans_indices[kk]  # Row index in B (treated as column index in B^T)
                valB = trans_data[kk]

                # Accumulate the result in row_accumulator
                if marker[rowB] != i:
                    row_accumulator[rowB] = 0
                    marker[rowB] = i
                row_accumulator[rowB] += valA * valB

        # Now collect all non-zero entries from row_accumulator for this row
        for j in range(ncols_B):
            if marker[j] == i and row_accumulator[j] != 0:
                indices_result.append(j)
                data_result.append(row_accumulator[j])
                current_length += 1
        
        # Update row pointer
        indptr_result[i+1] = row_start + current_length

    data_result = np.array(data_result, dtype=dtype)
    indices_result = np.array(indices_result, dtype=np.int64)

    return data_result, indices_result, indptr_result

@njit
def lm_dgm(csr, mat):
    '''
    Perform sparse matrix-matrix multiplication in CSR format.

    Parameters
    ----------
    data : 1D array
        Non-zero values of the CSR matrix
    indices : 1D array
        Column indices corresponding to values in `data`
    indptr : 1D array
        Row pointers to start of rows in `data`
    mat : 3D array
        Dense global matrix to multiply with the CSR matrix (number of rows should match CSR matrix columns)

    Returns
    -------
    result : 3D array - dense global matrix
        The result of the matrix-matrix multiplication
    '''

    # Get the CSR data for the current element
    data, indices, indptr = csr

    nen1 = len(indptr) - 1
    _, nen2, nelem = mat.shape
    result = np.zeros((nen1, nen2, nelem), dtype=mat.dtype)
    
    for e in range(nelem):
        for i in range(nen1):
            for jj in range(indptr[i], indptr[i+1]):
                col_idx = indices[jj]
                for k in range(nen2):
                    result[i, k, e] += data[jj] * mat[col_idx, k, e]
    
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
    
    # Initialize result array
    data, indices, indptr = csr
    nen1 = len(indptr) - 1  # Number of rows in the sparse matrix 
    _, nelem = np.shape(b)  # Number of elements (same as the third dimension of the original tensor)
    c = np.zeros((nen1, nelem), dtype=b.dtype)
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        for i in range(nen1):
            for jj in range(indptr[i], indptr[i+1]):
                c[i, e] += data[jj] * b[indices[jj], e]
    
    return c

@njit
def lm_gdiag(csr, H):
    '''
    Perform local matrix - global diagonal multiplication by scaling columns using a single CSR matrix and multiple scaling factors.

    Parameters
    ----------
    csr : tuple (data, indices, indptr)
        CSR representation of the input matrix.
    H : numpy array of shape (nen, nelem)
        The global diagonal scaling factors. Each column in H corresponds to a different scaled output.

    Returns
    -------
    List of CSR matrices
        Each element in the list is a CSR matrix (data, indices, indptr) representing the scaled matrix.
    '''
    data, indices, indptr = csr
    #nen1 = len(indptr) - 1  # Number of rows in the CSR matrix
    nen2, nelem = H.shape  # Dimensions of H (columns must match the CSR matrix's columns)

    # Check for dimension compatibility
    max_col = max(indices) + 1
    if nen2 < max_col:
        raise ValueError('Dimensions do not match: number of columns in CSR matrix must be <= the number of rows in H', max_col, nen2)

    # Initialize result list
    csr_result_list = List()

    # Loop over each column in H, corresponding to a different diagonal scaling
    for e in range(nelem):
        # Prepare a new array to store scaled data for the current CSR matrix
        new_data = np.empty_like(data)

        # Scale each non-zero element in `data` by the corresponding H element for this `e`
        for i in range(len(data)):
            col = indices[i]
            # Scale data by the corresponding element in H for this column and element `e`
            new_data[i] = data[i] * H[col, e]

        # Append the scaled CSR matrix to the result list
        csr_result_list.append((new_data, indices, indptr))

    return csr_result_list

@njit
def gdiag_lm(H, csr):
    '''
    Perform global diagonal - local matrix multiplication using a CSR matrix.

    Parameters
    ----------
    csr : CSR matrix
        a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    H : numpy array of shape (nen2, nelem)
        The global diagonal for multiplication

    Returns
    -------
    csr : CSR matrix
        a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    '''
    
    # Initialize result array
    data, indices, indptr = csr
    nen1 = len(indptr) - 1  # Number of rows in the sparse matrix 
    nen2, nelem = np.shape(H)  # Number of elements
    if nen1 != nen2:
        raise ValueError('Dimensions do not match', nen1, nen2)
    csr_list = List()
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        # Copy data for each new CSR matrix to preserve the original matrix
        new_data = np.zeros_like(data)

        # Multiply each row's data by the corresponding H element for this `e`
        for row in range(nen1):
            start = indptr[row]
            end = indptr[row + 1]
            
            # Scale non-zero elements in the row by H[row, e]
            row_scale = H[row, e]
            for i in range(start, end):
                new_data[i] = data[i] * row_scale
        
        # Append the resulting CSR matrix to the list
        csr_list.append((new_data, indices, indptr))
    
    return csr_list

@njit
def gdiag_gm(H, csr_list):
    '''
    Perform global diagonal - global matrix multiplication using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    H : numpy array of shape (nen2, nelem)
        The global diagonal for multiplication.

    Returns
    -------
    List of CSR matrices
        Each element in the list is a CSR matrix (data, indices, indptr) representing the scaled matrix.
    '''
    nen2, nelem = H.shape  # H's dimensions

    # Check if the number of CSR matrices matches the number of elements in H
    if len(csr_list) != nelem:
        raise ValueError('Dimensions do not match: number of CSR matrices must equal columns in H')

    # Initialize result list
    csr_result_list = List()

    # Perform scaling for each CSR matrix in csr_list
    for e in range(nelem):
        data, indices, indptr = csr_list[e]
        nen1 = len(indptr) - 1  # Number of rows in the CSR matrix

        # Check if the matrix dimensions match H
        if nen1 != nen2:
            raise ValueError(f'Dimensions do not match for element {e}: {nen1} rows in CSR matrix vs {nen2} in H')

        # Initialize a new array to hold the scaled data for the current CSR matrix
        new_data = np.empty_like(data)

        # Scale each row's data by the corresponding H element for this matrix `e`
        for row in range(nen1):
            start = indptr[row]
            end = indptr[row + 1]
            
            # Scale non-zero elements in the row by H[row, e]
            row_scale = H[row, e]
            for i in range(start, end):
                new_data[i] = data[i] * row_scale

        # Append the resulting CSR matrix to the list
        csr_result_list.append((new_data, indices, indptr))

    return csr_result_list

@njit
def lm_ldiag(csr, H):
    '''
    Scale each column of a sparse matrix in CSR format by a corresponding entry in a 1D array H.
    Simulates csr @ np.diag(H) where csr is a sparse matrix in CSR format.

    Parameters
    ----------
    csr : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    H : 1D array
        A 1D array where H[j] scales column j of the matrix.

    Returns
    -------
    scaled_csr : tuple
        A tuple (data, indices, indptr) representing the scaled sparse matrix in CSR format.
    '''
    data, indices, indptr = csr
    
    # Scale each entry in data by the corresponding entry in H based on the column index
    scaled_data = np.empty_like(data)
    for i in range(len(data)):
        col_index = indices[i]
        scaled_data[i] = data[i] * H[col_index]
    
    # Return the scaled matrix in CSR format
    return scaled_data, indices, indptr

@njit
def add_lm_lm(csr1, csr2):
    '''
    Add two sparse matrices in CSR format.

    Parameters
    ----------
    csr1 : tuple (data1, indices1, indptr1)
        CSR representation of the first matrix.
    csr2 : tuple (data2, indices2, indptr2)
        CSR representation of the second matrix.

    Returns
    -------
    result_csr : tuple (data, indices, indptr)
        CSR representation of the sum of the two matrices.
    '''
    data1, indices1, indptr1 = csr1
    data2, indices2, indptr2 = csr2

    # Ensure both CSR matrices have the same number of rows
    nrows = len(indptr1) - 1
    if len(indptr2) - 1 != nrows:
        raise ValueError("CSR matrices must have the same number of rows.")
    
    indptr_result = np.zeros(nrows + 1, dtype=np.int64)
    data_result = []
    indices_result = []

    # Iterate over each row
    for i in range(nrows):
        start1, end1 = indptr1[i], indptr1[i + 1]
        start2, end2 = indptr2[i], indptr2[i + 1]
        
        # Initialize pointers for both rows
        idx1 = start1
        idx2 = start2
        
        # Merge the rows by iterating over both in sorted order
        while idx1 < end1 or idx2 < end2:
            if idx1 < end1 and (idx2 >= end2 or indices1[idx1] < indices2[idx2]):
                # Element from csr1 only
                col = indices1[idx1]
                val = data1[idx1]
                idx1 += 1
            elif idx2 < end2 and (idx1 >= end1 or indices2[idx2] < indices1[idx1]):
                # Element from csr2 only
                col = indices2[idx2]
                val = data2[idx2]
                idx2 += 1
            else:
                # Elements from both csr1 and csr2
                col = indices1[idx1]  # indices1[idx1] == indices2[idx2] here
                val = data1[idx1] + data2[idx2]
                idx1 += 1
                idx2 += 1
            
            # Append only non-zero results
            if val != 0:
                indices_result.append(col)
                data_result.append(val)

        # Update the row pointer for the result matrix
        indptr_result[i + 1] = len(data_result)

    # Convert lists to arrays for CSR format
    data_result = np.array(data_result, dtype=data1.dtype)
    indices_result = np.array(indices_result, dtype=np.int64)

    return data_result, indices_result, indptr_result

@njit
def add_gm_gm(csr_list1, csr_list2):
    '''
    Perform global matrix addition using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    nen1 = len(csr_list1[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(csr_list1)  # Number of elements (same as the third dimension of the original tensor)

    nen2 = len(csr_list1[0][2]) - 1  
    nelem2 = len(csr_list1) 

    if nen1 != nen2 or nelem != nelem2:
        raise ValueError('Dimensions do not match', nen1, nen2, nelem, nelem2)
    
    # Initialize result array
    c = List()
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c.append(add_lm_lm(csr_list1[e], csr_list2[e]))
    
    return c

@njit
def set_gm_union_sparsity(gm_list):
    '''
    Set the sparsity pattern of a list of global dense matrices (shape: nen1, nen2, nelem)
    to the union of all sparsity patterns.

    Parameters
    ----------
    gm_list : List of global matrices
        Each element in the list is an ndarray (nen1, nen2, nelem)

    Returns
    -------
    sparsity : tuple (indices, indptr)
        The sparsity pattern of the union of all given matrices
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


#@njit # njit leads to some error that I don't care enough to fix. Only gets called once anyway.
def set_spgm_union_sparsity(csr_list):
    '''
    Set the sparsity pattern of a list of global CSR matrices to the union of all sparsity patterns.

    Parameters
    ----------
    csr_list : List of global matrices
        Each element in the list is a list of tuples (data, indices, indptr),
        where data, indices, and indptr represent the CSR components of the matrix.

    Returns
    -------
    sparsity_list : List of tuples (indices_glob, indptr_glob)
        A list where each entry is the sparsity pattern of the union of all given matrices for that element.
    '''
    ngm = len(csr_list)
    nelem = len(csr_list[0])

    # Validate that all matrices have the same number of elements
    for csr in csr_list:
        if len(csr) != nelem:
            raise ValueError('Number of elements do not match')

    sparsity_list = [] #List()

    for e in range(nelem):
        # Ensure all matrices have the same number of rows
        n_rows = len(csr_list[0][e][2]) - 1
        for csr in csr_list:
            if len(csr[e][2]) - 1 != n_rows:
                raise ValueError('Number of rows do not match', n_rows, len(csr[e][2]) - 1)

        indices_glob = []
        indptr_glob = [0]
        nnz = 0

        # Loop over each row to determine the union of nonzero elements
        for row in range(n_rows):
            row_indices_set = set()
            for k in range(ngm):
                _, indices, indptr_local = csr_list[k][e]
                row_start = indptr_local[row]
                row_end = indptr_local[row + 1]
                for col_ptr in range(row_start, row_end):
                    row_indices_set.add(indices[col_ptr])

            row_indices_sorted = sorted(row_indices_set)
            indices_glob.extend(row_indices_sorted)
            nnz += len(row_indices_sorted)
            indptr_glob.append(nnz)

        sparsity_list.append((np.array(indices_glob, dtype=np.int64), np.array(indptr_glob, dtype=np.int64)))

    return sparsity_list




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
def lm_dlm_had_diff(A,B):
    '''
    Compute the Hadamard product between a CSR matrices and a dense
    local matrix, then sum rows

    Parameters
    ----------
    A (data1, indices1, indptr1) : CSR representation of the first matrix
    B : (nen1, nen2) numpy array

    Returns
    -------
    c : numpy array of shape (nen1,nelem)
    '''
    # Get the CSR data for the current element
    data, indices, indptr = A
    nen1 = len(indptr) - 1

    c = np.zeros(nen1, dtype=B.dtype)
    
    for row in range(nen1):
        data_srt = indptr[row]
        data_end = indptr[row+1]
        for col_ptr in range(data_srt, data_end):
            col = indices[col_ptr]
            c[row] += data[col_ptr] * B[row, col]
    
    return c

@njit
def lmT_dlm_had_diff(AT,B):
    '''
    Compute the Hadamard product between a CSR matrices and a dense
    local matrix, then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT (data1, indices1, indptr1) : CSR representation of the first matrix
    B : (nen1, nen2) numpy array

    Returns
    -------
    c : numpy array of shape (nen2,nelem)
    '''
    # Get the CSR data for the current element
    data, indices, indptr = AT
    nen1 = len(indptr) - 1

    c = np.zeros(nen1, dtype=data.dtype)
    
    for row in range(nen1):
        data_srt = indptr[row]
        data_end = indptr[row+1]
        for col_ptr in range(data_srt, data_end):
            col = indices[col_ptr]
            c[col] += data[col_ptr] * B[row, col]
    
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
def gm_dgm_had_diff(A,B):
    '''
    Compute the hadamard product between a sparse global matrix (list of CSR matrices) 
    and a dense global matrix, then sum rows

    Parameters
    ----------
    A : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    B : numpy array of shape (nen1, nen2, nelem)

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(A[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(A)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B.dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lm_dlm_had_diff(A[e], B[:,:,e])
    
    return c

@njit
def gmT_dgm_had_diff(AT,B):
    '''
    Compute the hadamard product between a sparse global matrix (list of CSR matrices) 
    and a dense global matrix, then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    B : numpy array of shape (nen1, nen2, nelem)

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(AT[0][2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nelem = len(AT)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B.dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lmT_dlm_had_diff(AT[e], B[:,:,e])
    
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
def lm_dgm_had_diff(A,B):
    '''
    Compute the hadamard product between a sparse local matrix (CSR matrix) 
    and a dense global matrix, then sum rows

    Parameters
    ----------
    A (data1, indices1, indptr1) : CSR representation of the first matrix
    B : numpy array of shape (nen1, nen2, nelem)

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(A[2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nen1b, _, nelem = np.shape(B)
    if nen1 != nen1b:
        raise ValueError('Dimensions do not match', nen1, nen1b)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B.dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lm_dlm_had_diff(A, B[:,:,e])
    
    return c

@njit
def lmT_dgm_had_diff(AT,B):
    '''
    Compute the hadamard product between a sparse local matrix (CSR matricex) 
    and a dense global matrix, then sum rows
    this computes sum_j A.T_ji B_ji = sum_j A_ij B_ji = had_diff(A,B.T)

    Parameters
    ----------
    AT (data1, indices1, indptr1) : CSR representation of the first matrix
    B : B : numpy array of shape (nen2, nen3, nelem)

    Returns
    -------
    c : numpy array of shape (nen1, nelem)
        Result of the matrix-vector multiplication
    '''
    nen1 = len(AT[2]) - 1  # Number of rows in the sparse matrix (from indptr)
    nen1b, _, nelem = np.shape(B)
    if nen1 != nen1b:
        raise ValueError('Dimensions do not match', nen1, nen1b)
    
    # Initialize result array
    c = np.zeros((nen1, nelem), dtype=B.dtype)
    
    # Perform sparse hadamard product for each element
    for e in range(nelem):
        c[:, e] = lmT_dlm_had_diff(AT, B[:,:,e])
    
    return c

@njit 
def build_F_vol_sca(q, flux, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.
    Takes advantage of symmetry since q1 = q2 = q '''
    nen, nelem = q.shape 
    F_vol = List()
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity[e][0]
        indptr = sparsity[e][1]
        new_data = np.zeros((len(indices)), dtype=q.dtype)       
        colptrs = np.zeros(nen, dtype=np.int64)
        
        for i in range(nen): # loop over rows

            col_start = indptr[i]
            col_end = indptr[i + 1]

            for j in range(i,nen): # loop over columns

                col_start_T = indptr[j]
                col_end_T = indptr[j + 1]

                add_entry = (j in indices[col_start:col_end])
                add_entry_T = (i in indices[col_start_T:col_end_T]) and (i != j)

                if add_entry or add_entry_T:

                    fij = flux(q[i, e], q[j, e])
                        
                    if add_entry:
                        new_col_ptr = col_start + colptrs[i]
                        new_data[new_col_ptr] = fij

                    if add_entry_T:
                        new_col_ptr = col_start_T + colptrs[j]
                        new_data[new_col_ptr] = fij

                    if add_entry: colptrs[i] += 1
                    if add_entry_T: colptrs[j] += 1


        F_vol.append((new_data, indices, indptr))
    return F_vol

@njit 
def build_F_sca(q1, q2, flux, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen, nelem = q1.shape 
    F = List()
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity[e][0]
        indptr = sparsity[e][1]
        new_data = np.zeros((len(indices)), dtype=q1.dtype)       
        
        for i in range(nen): # loop over rows, NOT kroned with neq
            colptr = 0
            col_start = indptr[i]
            col_end = indptr[i + 1]

            for j in range(nen): # loop over columns, NOT kroned with neq
                if (j in indices[col_start:col_end]):

                    fij = flux(q1[i, e], q2[j, e])
                        
                    new_col_ptr = col_start + colptr
                    new_data[new_col_ptr] = fij

                    colptr += 1

        F.append((new_data, indices, indptr))
    return F

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
        colptrs = np.zeros(nen, dtype=np.int64)
        
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
def build_F_vol_sca_2d(q, flux, xsparsity, ysparsity):
    '''
    Builds sparsified Flux differencing matrices (used for Hadamard form) for a 2D problem given a 
    solution vector q, a 2-point flux function, and sparsity patterns for both x and y directions. 
    Only computes the entries specified by indices and indptr in the given sparsity patterns.
    Takes advantage of symmetry since q1 = q2 = q.

    Parameters
    ----------
    q : ndarray
        Solution vector of shape (nen, nelem).
    flux : function
        A 2-point flux function that takes two arguments and returns flux values.
    xsparsity : List of CSR matrices
        Sparsity pattern for the x-direction, where each entry is a tuple (indices, indptr).
    ysparsity : List of CSR matrices
        Sparsity pattern for the y-direction, where each entry is a tuple (indices, indptr).

    Returns
    -------
    Fx_vol : List of CSR matrices
        A list where each entry is a tuple (data, indices, indptr) representing the flux differencing matrix in the x direction.
    Fy_vol : List of CSR matrices
        A list where each entry is a tuple (data, indices, indptr) representing the flux differencing matrix in the y direction.
    '''
    nen, nelem = q.shape
    Fx_vol = List()
    Fy_vol = List()

    for e in range(nelem):
        # Initialize lists to store CSR data for x and y directions
        xindices = xsparsity[e][0]
        xindptr = xsparsity[e][1]
        yindices = ysparsity[e][0]
        yindptr = ysparsity[e][1]

        xnew_data = np.zeros(len(xindices), dtype=q.dtype)
        ynew_data = np.zeros(len(yindices), dtype=q.dtype)
        xcolptrs = np.zeros(nen, dtype=np.int64)
        ycolptrs = np.zeros(nen, dtype=np.int64)

        for i in range(nen):  # Loop over rows
            xcol_start = xindptr[i]
            xcol_end = xindptr[i + 1]
            ycol_start = yindptr[i]
            ycol_end = yindptr[i + 1]

            for j in range(i, nen):  # Loop over columns
                xcol_start_T = xindptr[j]
                xcol_end_T = xindptr[j + 1]
                ycol_start_T = yindptr[j]
                ycol_end_T = yindptr[j + 1]

                xadd_entry = (j in xindices[xcol_start:xcol_end])
                xadd_entry_T = (i in xindices[xcol_start_T:xcol_end_T]) and (i != j)
                yadd_entry = (j in yindices[ycol_start:ycol_end])
                yadd_entry_T = (i in yindices[ycol_start_T:ycol_end_T]) and (i != j)

                if xadd_entry or xadd_entry_T or yadd_entry or yadd_entry_T:
                    fij_x, fij_y = flux(q[i, e], q[j, e])

                    if xadd_entry:
                        xnew_col_ptr = xcol_start + xcolptrs[i]
                        xnew_data[xnew_col_ptr] = fij_x

                    if xadd_entry_T:
                        xnew_col_ptr = xcol_start_T + xcolptrs[j]
                        xnew_data[xnew_col_ptr] = fij_x

                    if yadd_entry:
                        ynew_col_ptr = ycol_start + ycolptrs[i]
                        ynew_data[ynew_col_ptr] = fij_y

                    if yadd_entry_T:
                        ynew_col_ptr = ycol_start_T + ycolptrs[j]
                        ynew_data[ynew_col_ptr] = fij_y

                    if xadd_entry:
                        xcolptrs[i] += 1
                    if xadd_entry_T:
                        xcolptrs[j] += 1
                    if yadd_entry:
                        ycolptrs[i] += 1
                    if yadd_entry_T:
                        ycolptrs[j] += 1

        Fx_vol.append((xnew_data, xindices, xindptr))
        Fy_vol.append((ynew_data, yindices, yindptr))

    return Fx_vol, Fy_vol
    
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
        xcolptrs = np.zeros(nen, dtype=np.int64)
        ycolptrs = np.zeros(nen, dtype=np.int64)
        
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

@njit
def unkron_neq_gm(csr_list, neq):
    '''
    Take a list of CSR matrices of shape (nen*neq, nen2*neq, nelem) and return a new list of CSR matrices
    of shape (nen, nen2, nelem), effectively undoing the Kronecker product operation for
    the operator acting on a vector (nen2*neq, nelem).

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr), representing the CSR components of the matrix.
    neq : int
        Number of equations per node.

    Returns
    -------
    unkron_csr_list : List of CSR matrices
        A list where each entry is a tuple (data, indices, indptr) representing the un-kronned version of the original matrix.
    '''
    unkron_csr_list = List()

    for e in range(len(csr_list)):
        data, indices, indptr = csr_list[e]
        nen_neq = len(indptr) - 1
        nen = nen_neq // neq

        new_data = []
        new_indices = []
        new_indptr = [0] * (nen + 1)

        for row in range(nen):
            row_start = indptr[row * neq]
            row_end = indptr[row * neq + 1]
            nnz_counter = 0

            for col_idx in range(row_start, row_end):
                col = indices[col_idx]
                if col % neq == 0:
                    new_data.append(data[col_idx])
                    new_indices.append(indices[col_idx] // neq)
                    nnz_counter += 1

            new_indptr[row + 1] = new_indptr[row] + nnz_counter

        unkron_csr_list.append((np.array(new_data, dtype=data.dtype),
                                np.array(new_indices, dtype=np.int64),
                                np.array(new_indptr, dtype=np.int64)))

    return unkron_csr_list

@njit
def unkron_neq_sparsity(csr_list, neq):
    '''
    Take a list of CSR sparsity tuples (indices, indptr) and return a new list of
    CSR sparsity tuples, effectively undoing the Kronecker product operation for
    the operator acting on a vector (nen*neq, nelem).

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (indices, indptr), representing the CSR components of the matrix.
    neq : int
        Number of equations per node.

    Returns
    -------
    unkron_csr_list : List of CSR matrices
        A list where each entry is a tuple (indices, indptr) representing the un-kronned version of the original matrix.
    '''
    unkron_csr_list = List()

    for e in range(len(csr_list)):
        indices, indptr = csr_list[e]
        nen_neq = len(indptr) - 1
        nen = nen_neq // neq

        new_indices = []
        new_indptr = [0] * (nen + 1)

        for row in range(nen):
            row_start = indptr[row * neq]
            row_end = indptr[row * neq + 1]
            nnz_counter = 0

            for col_idx in range(row_start, row_end):
                col = indices[col_idx]
                if col % neq == 0:
                    new_indices.append(indices[col_idx] // neq)
                    nnz_counter += 1

            new_indptr[row + 1] = new_indptr[row] + nnz_counter

        unkron_csr_list.append((np.array(new_indices, dtype=np.int64),
                                np.array(new_indptr, dtype=np.int64)))

    return unkron_csr_list

@njit
def assemble_satx_2d(csr_list,nelemx,nelemy):
    ''' given a list of csr matrices like (nen,nen2,nelem[idx]), 
    put them back in global order (nen,nen2,nelem)
    where each entry is a list of matrices that would be selected
     by e.g. satx.vol_x_mat[idx], where idx is one row in x

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a list of tuples (data, indices, indptr),
        where data, indices, and indptr represent the CSR components of the matrix.
    nelemx : int
        Number of elements in the x direction.
    nelemy : int
        Number of elements in the y direction.

    Returns
    -------
    mat_glob : List of CSR matrices
        A list where each entry is a tuple (data, indices, indptr) representing the global CSR matrix.
    '''
    nelemy2 = len(csr_list)
    if nelemy2 != nelemy:
        raise ValueError('nelemy does not match', nelemy2, nelemy)
    nelemx2 = len(csr_list[0])
    if nelemx2 != nelemx:
        raise ValueError('nelemx does not match', nelemx2, nelemx)

    mat_glob = [(np.zeros(0, dtype=csr_list[0][0][0].dtype), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))] * (nelemx * nelemy)
    for ey in range(nelemy):
        for ex in range(nelemx):
            idx = ex * nelemy + ey
            mat_glob[idx] = csr_list[ey][ex]
    return mat_glob

@njit
def assemble_saty_2d(csr_list,nelemx,nelemy):
    '''
    Given a list of CSR matrices of shape (nen, nen2, nelem[idx]),
    put them back in global order (nen, nen2, nelem)
    where each entry is a list of matrices that would be selected
    by e.g. saty.vol_y_mat[idx], where idx is one row in y.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a list of tuples (data, indices, indptr),
        where data, indices, and indptr represent the CSR components of the matrix.
    nelemx : int
        Number of elements in the x direction.
    nelemy : int
        Number of elements in the y direction.

    Returns
    -------
    mat_glob : List of CSR matrices
        A list where each entry is a tuple (data, indices, indptr) representing the global CSR matrix.
    '''
    nelemx2 = len(csr_list)
    if nelemx2 != nelemx:
        raise ValueError('nelemx does not match', nelemx2, nelemx)
    nelemy2 = len(csr_list[0])
    if nelemy2 != nelemy:
        raise ValueError('nelemy does not match', nelemy2, nelemy)

    mat_glob = [(np.zeros(0, dtype=csr_list[0][0][0].dtype), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))] * (nelemx * nelemy)
    for ex in range(nelemx):
        for ey in range(nelemy):
            idx = ex * nelemy + ey
            mat_glob[idx] = csr_list[ex][ey]
    return mat_glob

@njit
def kron_lm_lm(Dx, Dy):
    '''
    Compute the Kronecker product of two CSR matrices Dx and Dy.

    Parameters
    ----------
    Dx : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    Dy : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the Kronecker product.
    '''
    Dx_data, Dx_indices, Dx_indptr = Dx
    Dy_data, Dy_indices, Dy_indptr = Dy
    
    Dx_rows = len(Dx_indptr) - 1
    Dy_rows = len(Dy_indptr) - 1
    Dx_cols = max(Dx_indices) + 1
    Dy_cols = max(Dy_indices) + 1

    # Initialize lists to store the resulting CSR components
    data = []
    indices = []
    indptr = [0]
    
    for i in range(Dx_rows):
        for j in range(Dy_rows):
            # Get the non-zero row slices for current row in Dx and Dy
            Dx_row_start = Dx_indptr[i]
            Dx_row_end = Dx_indptr[i + 1]
            Dy_row_start = Dy_indptr[j]
            Dy_row_end = Dy_indptr[j + 1]
            
            for dx_idx in range(Dx_row_start, Dx_row_end):
                for dy_idx in range(Dy_row_start, Dy_row_end):
                    # Compute the data value
                    value = Dx_data[dx_idx] * Dy_data[dy_idx]
                    data.append(value)
                    
                    # Compute the column index
                    dx_col = Dx_indices[dx_idx]
                    dy_col = Dy_indices[dy_idx]
                    col_index = dx_col * Dy_cols + dy_col
                    indices.append(col_index)
                    
            # Update indptr for the Kronecker product row
            indptr.append(len(data))
    
    # Convert lists to arrays for CSR format
    return np.array(data), np.array(indices), np.array(indptr)

@njit
def kron_eye_lm(Dy, p, n=0):
    '''
    Compute the Kronecker product of a p x p identity matrix and a CSR matrix Dy.

    Parameters
    ----------
    Dy : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    p : int
        Size of the identity matrix.
    n : int, optional
        Number of columns in Dy. If not provided, it will be inferred as the maximum value in indices + 1.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the Kronecker product.
    '''
    Dy_data, Dy_indices, Dy_indptr = Dy

    # Infer dimensions of Dy if not provided
    m = len(Dy_indptr) - 1 # rows in Dy
    if n == 0: n = max(Dy_indices) + 1 # columns in Dy

    # Prepare arrays for CSR format
    data = np.zeros(len(Dy_data) * p, dtype=Dy_data.dtype)
    indices = np.zeros(len(Dy_indices) * p, dtype=np.int64)
    indptr = np.zeros(m * p + 1, dtype=np.int64)

    # Manually repeat Dy_data and Dy_indices for each block
    for i in range(p):
        start_data = i * len(Dy_data)
        start_indices = i * len(Dy_indices)
        
        for j in range(len(Dy_data)):
            data[start_data + j] = Dy_data[j]
            indices[start_indices + j] = Dy_indices[j] + i * n
        
        # Update indptr for each row block
        for k in range(m + 1):
            indptr[i * m + k] = Dy_indptr[k] + i * len(Dy_data)
    
    return data, indices, indptr

@njit
def kron_lm_eye(Dx, p):
    '''
    Compute the Kronecker product of a CSR matrix Dx and a p x p identity matrix.

    Parameters
    ----------
    Dx : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    p : int
        Size of the identity matrix.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the Kronecker product.
    '''
    Dx_data, Dx_indices, Dx_indptr = Dx
    m = len(Dx_indptr) - 1  # Number of rows in Dx

    # Prepare arrays for the expanded CSR format
    expanded_data = np.zeros(len(Dx_data) * p, dtype=Dx_data.dtype)
    expanded_indices = np.zeros(len(Dx_indices) * p, dtype=np.int64)
    expanded_indptr = np.zeros(m * p + 1, dtype=np.int64)

    data_index = 0  # Position in expanded_data and expanded_indices

    # Iterate over each row in Dx, expanding into p rows in the result
    for i in range(m):
        row_start = Dx_indptr[i]
        row_end = Dx_indptr[i + 1]

        # For each row in the p x p block
        for k in range(p):
            # For each non-zero element in the current row of Dx
            for j in range(row_start, row_end):
                value = Dx_data[j]
                col_index = Dx_indices[j]

                # Place the value along the diagonal of the p x p block
                expanded_data[data_index] = value
                expanded_indices[data_index] = col_index * p + k
                data_index += 1

            # Update indptr for the next row in the expanded matrix
            expanded_indptr[i * p + k + 1] = data_index

    return expanded_data, expanded_indices, expanded_indptr

@njit
def kron_ldiag_lm(diag, Dy, n=0):
    '''
    Compute the Kronecker product of a p x p diagonal matrix and a CSR matrix Dy.

    Parameters
    ----------
    Dy : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    diag : array-like
        A 1D array representing the diagonal entries of the p x p diagonal matrix.
    n : int, optional
        Number of columns in Dy. If not provided, it will be inferred as the maximum value in indices + 1.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the Kronecker product.
    '''
    Dy_data, Dy_indices, Dy_indptr = Dy
    p = len(diag)  # The size of the diagonal matrix is inferred from the length of diag

    # Infer dimensions of Dy if not provided
    m = len(Dy_indptr) - 1  # rows in Dy
    if n == 0:
        n = max(Dy_indices) + 1  # columns in Dy

    # Prepare arrays for CSR format
    data = np.zeros(len(Dy_data) * p, dtype=Dy_data.dtype)
    indices = np.zeros(len(Dy_indices) * p, dtype=np.int64)
    indptr = np.zeros(m * p + 1, dtype=np.int64)

    # Manually repeat Dy_data and Dy_indices for each block, scaling by diag elements
    for i in range(p):
        start_data = i * len(Dy_data)
        start_indices = i * len(Dy_indices)
        
        for j in range(len(Dy_data)):
            # Scale each entry of Dy by the corresponding diagonal element
            data[start_data + j] = Dy_data[j] * diag[i]
            indices[start_indices + j] = Dy_indices[j] + i * n
        
        # Update indptr for each row block
        for k in range(m + 1):
            indptr[i * m + k] = Dy_indptr[k] + i * len(Dy_data)
    
    return data, indices, indptr

@njit
def kron_lm_ldiag(Dx, diag):
    '''
    Compute the Kronecker product of a CSR matrix Dx and a p x p diagonal matrix.

    Parameters
    ----------
    Dx : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    diag : array-like
        A 1D array representing the diagonal entries of the p x p diagonal matrix.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the Kronecker product.
    '''
    Dx_data, Dx_indices, Dx_indptr = Dx
    m = len(Dx_indptr) - 1  # Number of rows in Dx
    p = len(diag)  # The size of the diagonal matrix is inferred from the length of diag

    # Prepare arrays for the expanded CSR format
    expanded_data = np.zeros(len(Dx_data) * p, dtype=Dx_data.dtype)
    expanded_indices = np.zeros(len(Dx_indices) * p, dtype=np.int64)
    expanded_indptr = np.zeros(m * p + 1, dtype=np.int64)

    data_index = 0  # Position in expanded_data and expanded_indices

    # Iterate over each row in Dx, expanding into p rows in the result
    for i in range(m):
        row_start = Dx_indptr[i]
        row_end = Dx_indptr[i + 1]

        # For each row in the p x p block
        for k in range(p):
            # For each non-zero element in the current row of Dx
            for j in range(row_start, row_end):
                value = Dx_data[j] * diag[k]  # Scale by diag[k]
                col_index = Dx_indices[j]

                # Place the value in the appropriate block position
                expanded_data[data_index] = value
                expanded_indices[data_index] = col_index * p + k
                data_index += 1

            # Update indptr for the next row in the expanded matrix
            expanded_indptr[i * p + k + 1] = data_index

    return expanded_data, expanded_indices, expanded_indptr

@njit
def kron_gm_eye(Dx_list, p):
    '''
    Compute the Kronecker product of a list of CSR matrices and a p x p identity matrix.

    Parameters
    ----------
    Dx_list : List of CSR matrices
        Each element is a tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    p : int
        Size of the identity matrix.

    Returns
    -------
    List of CSR matrices
        Each element in the list is a CSR matrix (data, indices, indptr) representing the Kronecker product for each CSR in Dx_list.
    '''
    # Prepare a list to store the result CSR matrices
    kron_result_list = List()

    # Iterate over each CSR matrix in Dx_list
    for Dx in Dx_list:
        Dx_data, Dx_indices, Dx_indptr = Dx
        m = len(Dx_indptr) - 1  # Number of rows in the current Dx

        # Prepare arrays for the expanded CSR format for this Dx
        expanded_data = np.zeros(len(Dx_data) * p, dtype=Dx_data.dtype)
        expanded_indices = np.zeros(len(Dx_indices) * p, dtype=np.int64)
        expanded_indptr = np.zeros(m * p + 1, dtype=np.int64)

        data_index = 0  # Position in expanded_data and expanded_indices

        # Iterate over each row in Dx, expanding into p rows in the result
        for i in range(m):
            row_start = Dx_indptr[i]
            row_end = Dx_indptr[i + 1]

            # For each row in the p x p block
            for k in range(p):
                # For each non-zero element in the current row of Dx
                for j in range(row_start, row_end):
                    value = Dx_data[j]
                    col_index = Dx_indices[j]

                    # Place the value along the diagonal of the p x p block
                    expanded_data[data_index] = value
                    expanded_indices[data_index] = col_index * p + k
                    data_index += 1

                # Update indptr for the next row in the expanded matrix
                expanded_indptr[i * p + k + 1] = data_index

        # Append the resulting expanded CSR matrix for this Dx to the result list
        kron_result_list.append((expanded_data, expanded_indices, expanded_indptr))

    return kron_result_list

@njit
def kron_neq_lm(Dx, neq_node):
    '''
    Compute the modified Kronecker product of a CSR matrix Dx with a neq_node x neq_node structured pattern.

    Parameters
    ----------
    Dx : tuple
        A tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    neq_node : int
        Number of degrees of freedom per node.

    Returns
    -------
    csr_kron : tuple
        A tuple (data, indices, indptr) representing the CSR matrix of the expanded operator.
    '''
    Dx_data, Dx_indices, Dx_indptr = Dx
    m = len(Dx_indptr) - 1  # Number of rows in Dx

    # Prepare arrays for the expanded CSR format
    expanded_data = np.zeros(len(Dx_data) * neq_node, dtype=Dx_data.dtype)
    expanded_indices = np.zeros(len(Dx_indices) * neq_node, dtype=np.int64)
    expanded_indptr = np.zeros(m * neq_node + 1, dtype=np.int64)

    data_index = 0  # Position in expanded_data and expanded_indices

    # Iterate over each row in Dx, expanding into neq_node rows in the result
    for i in range(m):
        row_start = Dx_indptr[i]
        row_end = Dx_indptr[i + 1]

        # For each degree of freedom in the neq_node block
        for n in range(neq_node):
            # For each non-zero element in the current row of Dx
            for j in range(row_start, row_end):
                value = Dx_data[j]
                col_index = Dx_indices[j]

                # Spread the value across the appropriate positions
                expanded_data[data_index] = value
                expanded_indices[data_index] = col_index * neq_node + n
                data_index += 1

            # Update indptr for the next row in the expanded matrix
            expanded_indptr[i * neq_node + n + 1] = data_index

    return expanded_data, expanded_indices, expanded_indptr

@njit
def kron_neq_gm(csr_list, neq_node):
    '''
    Compute the modified Kronecker product of a global CSR matrix Dx with a neq_node x neq_node structured pattern.

    Parameters
    ----------
    Dx : list oftuple
        A list of tuples (data, indices, indptr) representing a global sparse matrix in CSR format.
    neq_node : int
        Number of degrees of freedom per node.

    Returns
    -------
    c : list of tuples
        A list of tuples (data, indices, indptr) representing the CSR matrix of the expanded operator.
    '''
    # Initialize result array
    c = List()
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(len(csr_list)):
        c.append(kron_neq_lm(csr_list[e], neq_node))
    
    return c

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
    lv = np.random.rand(11)
    gv = np.random.rand(11, 3)
    gv2 = np.random.rand(10, 3)
    lm = np.random.rand(10, 11)
    lm_sp = lm_to_sp(lm)
    lm2 = np.random.rand(11, 10)
    lm2_sp = lm_to_sp(lm2)
    lm3 = np.random.rand(10, 11)
    lm3_sp = lm_to_sp(lm3)
    gm_list = [gm, gm]

    c = lm_lv(lm_sp, lv)
    c2 = fn.lm_lv(lm, lv)
    print('test lm_lv:', np.max(abs(c-c2)))

    c = sp_to_lm(lm_lm(lm_sp, lm2_sp))
    c2 = fn.lm_lm(lm, lm2)
    print('test lm_lm:', np.max(abs(c-c2)))

    c = sp_to_lm(lm_lmT(lm_sp, lm3_sp))
    c2 = lm @ lm3.T
    print('test lm_lmT:', np.max(abs(c-c2)))

    c = lm_gv(lm_sp, gv)
    c2 = fn.lm_gv(lm, gv)
    print('test lm_gv:', np.max(abs(c-c2)))

    c = gm_gv(gm_sp, gv)
    c2 = fn.gm_gv(gm, gv)
    print('test gm_gv:', np.max(abs(c-c2)))

    c = sp_to_gm(lm_gdiag(lm_sp, gv))
    c2 = fn.lm_gdiag(lm, gv)
    print('test lm_gdiag:', np.max(abs(c-c2)))

    c = sp_to_gm(gdiag_lm(gv, lm2_sp))
    c2 = fn.gdiag_lm(gv, lm2)
    print('test gdiag_lm:', np.max(abs(c-c2)))

    c = sp_to_gm(gdiag_gm(gv2, gm_sp))
    c2 = fn.gdiag_gm(gv2, gm)
    print('test gdiag_gm:', np.max(abs(c-c2)))

    c = sp_to_lm(lm_ldiag(lm_sp, lv))
    c2 = lm @ np.diag(lv)
    print('test lm_ldiag:', np.max(abs(c-c2)))

    c = sp_to_gm(lm_gm(lm2_sp, gm_sp))
    c2 = fn.lm_gm(lm2, gm)
    print('test lm_gm:', np.max(abs(c-c2)))

    c = lm_dgm(lm2_sp, gm)
    c2 = fn.lm_gm(lm2, gm)
    print('test lm_dgm:', np.max(abs(c-c2)))

    c = sp_to_gm(add_gm_gm(gm_sp, gm2_sp))
    c2 = gm + gm2
    print('test add_gm_gm:', np.max(abs(c-c2)))

    c = sp_to_lm(kron_lm_lm(lm_sp, lm2_sp))
    c2 = np.kron(lm, lm2)
    print('test kron_lm_lm:', np.max(abs(c-c2)))

    eye = np.eye(5)
    c = sp_to_lm(kron_eye_lm(lm2_sp, 5))
    c2 = np.kron(eye, lm2)
    print('test kron_eye_lm:', np.max(abs(c-c2)))

    c = sp_to_lm(kron_lm_eye(lm_sp, 5))
    c2 = np.kron(lm, eye)
    print('test kron_lm_eye:', np.max(abs(c-c2)))

    c = sp_to_lm(kron_neq_lm(lm_sp, 5))
    c2 = fn.kron_neq_lm(lm, 5)
    print('test kron_neq_lm:', np.max(abs(c-c2)))

    lm2 = np.random.rand(10, 11)
    lm2_sp = lm_to_sp(lm2)

    c = lm_lm_had_diff(lm_sp, lm2_sp)
    c2 = np.sum(np.multiply(lm, lm2),axis=1)
    print('test lm_lm_had_diff:', np.max(abs(c-c2)))

    c = gm_gm_had_diff(gm_sp, gm2_sp)
    c2 = fn.gm_gm_had_diff(gm, gm2)
    print('test gm_gm_had_diff:', np.max(abs(c-c2)))

    c = lm_dgm_had_diff(lm_sp, gm)
    c2 = fn.lm_gm_had_diff(lm, gm)
    print('test lm_dgm_had_diff:', np.max(abs(c-c2)))

    sp_gm_list = set_gm_union_sparsity(gm_list)
    diff = 0.
    for e in range(3):
        diff += np.max(abs(sp_gm_list[e][0] - gm_sp[e][1]))
        diff += np.max(abs(sp_gm_list[e][1] - gm_sp[e][2]))
    print('test set_gm_union_sparsity:', diff)

    gm_list = [gm, gm2]
    spgm_list = [gm_sp, gm2_sp]
    sp_gm_list = set_gm_union_sparsity(gm_list)
    sp_gm_list2 = set_spgm_union_sparsity(spgm_list)
    diff = 0.
    for e in range(3):
        diff += np.max(abs(sp_gm_list[e][0] - sp_gm_list2[e][0]))
        diff += np.max(abs(sp_gm_list[e][1] - sp_gm_list2[e][1]))
    print('test set_spgm_union_sparsity:', diff)


    print('test sp_to_gm:', np.max(abs(sp_to_gm(gm_sp)) - gm))

    gm = np.random.rand(10, 10, 3)
    # add some sparsity
    gm[1, 1, 0] = 0.
    gm[1, 5, 0] = 0.
    gm[4, 2, 2] = 0.
    gmT = np.transpose(gm,(1,0,2))
    gm_sp = gm_to_sp(gm)
    gmT_sp = gm_to_sp(gmT)
    gm2 = np.random.rand(10, 10, 3)
    gm2[5, 6, 0] = 0.
    gm2[4, 1, 0] = 0.
    gm2[7, 8, 1] = 0.
    gm2_sp = gm_to_sp(gm2)
    lm = np.random.rand(10, 10)
    lmT = lm.T
    lm_sp = lm_to_sp(lm)
    lmT_sp = lm_to_sp(lmT)
    lm2 = np.random.rand(10, 10)
    lm2_sp = lm_to_sp(lm2)

    c = lmT_lm_had_diff(lmT_sp, lm2_sp)
    c2 = np.sum(np.multiply(lm, lm2.T),axis=1)
    print('test lmT_lm_had_diff:', np.max(abs(c-c2)))

    c = lmT_gm_had_diff(lmT_sp, gm_sp)
    c2 = fn.lm_gm_had_diff(lm,np.transpose(gm,(1,0,2)))
    print('test lmT_gm_had_diff:', np.max(abs(c-c2)))

    c = gmT_gm_had_diff(gmT_sp, gm2_sp)
    c2 = fn.gm_gm_had_diff(gm,np.transpose(gm2,(1,0,2)))
    print('test gmT_gm_had_diff:', np.max(abs(c-c2)))

    c = lmT_dlm_had_diff(lmT_sp, lm2)
    c2 = np.sum(np.multiply(lm, lm2.T),axis=1)
    print('test lmT_dlm_had_diff:', np.max(abs(c-c2)))

    c = lmT_dgm_had_diff(lmT_sp, gm)
    c2 = fn.lm_gm_had_diff(lm,np.transpose(gm,(1,0,2)))
    print('test lmT_dgm_had_diff:', np.max(abs(c-c2)))

    c = gmT_dgm_had_diff(gmT_sp, gm2)
    c2 = fn.gm_gm_had_diff(gm,np.transpose(gm2,(1,0,2)))
    print('test gmT_dgm_had_diff:', np.max(abs(c-c2)))

    
    @njit
    def flux(q1,q2):
        return 0.5*(q1 + q2)
    
    nelem = 2
    nen = 3
    q = np.random.rand(nen, nelem)
    q2 = np.random.rand(nen, nelem)
    gm = np.random.rand(nen, nen, nelem)
    # add some sparsity
    gm[1, 0, 0] = 0.
    gm[1, 2, 0] = 0.
    gm[2, 2, 1] = 0.
    gm[0, 2, 1] = 0.
    sparsity = set_gm_union_sparsity([gm])
    gm_sp = gm_to_sp(gm)
    gm2 = np.random.rand(nen, nen, nelem)
    # add some sparsity
    gm2[2, 0, 0] = 0.
    gm2[1, 1, 0] = 0.
    gm2[0, 2, 1] = 0.
    gm2[1, 2, 1] = 0.
    sparsity2 = set_gm_union_sparsity([gm2])
    gm2_sp = gm_to_sp(gm2)

    F = build_F_vol_sca(q, flux, sparsity)
    c = gm_gm_had_diff(gm_sp, F)
    F2 = fn.build_F_vol_sca(q, flux)
    c2 = fn.gm_gm_had_diff(gm, F2)
    print('test build_F_vol_sca:', np.max(abs(c-c2)))

    F = build_F_sca(q, q2, flux, sparsity)
    c = gm_gm_had_diff(gm_sp, F)
    F2 = fn.build_F_sca(q, q2, flux)
    c2 = fn.gm_gm_had_diff(gm, F2)
    print('test build_F_sca:', np.max(abs(c-c2)))

    @njit
    def flux(q1,q2):
        return 0.5*(q1 + q2), 2*(q1 - 0.5*q2)

    F, F2 = build_F_vol_sca_2d(q, flux, sparsity, sparsity2)
    c = gm_gm_had_diff(gm_sp, F) + gm_gm_had_diff(gm2_sp, F2)
    F, F2 = fn.build_F_vol_sca_2d(q, flux)
    c2 = fn.gm_gm_had_diff(gm, F) + fn.gm_gm_had_diff(gm2, F2)
    print('test build_F_vol_sca_2d:', np.max(abs(c-c2)))

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
    gm_sp = gm_to_sp(gm)
    gm_kron = fn.kron_neq_gm(gm, neq)
    gm_kron_sp = gm_to_sp(gm_kron)
    sparsity_unkronned = set_gm_union_sparsity([gm])
    sparsity_kron = set_gm_union_sparsity([gm_kron])
    gm2 = np.random.rand(nen, nen, nelem)
    # add some sparsity
    gm2[0, 1, 0] = 0.
    gm2[1, 1, 0] = 0.
    gm2[1, 2, 1] = 0.
    gm2[1, 1, 1] = 0.
    gm2_kron = fn.kron_neq_gm(gm2, neq)
    gm2_kron_sp = gm_to_sp(gm2_kron)
    sparsity_unkronned2 = set_gm_union_sparsity([gm2])
    sparsity_kron2 = set_gm_union_sparsity([gm2_kron])

    gm_sp2 = unkron_neq_gm(gm_kron_sp, neq)
    diff = 0.
    for e in range(nelem):
        diff += np.max(abs(gm_sp2[e][0] - gm_sp[e][0]))
        diff += np.max(abs(gm_sp2[e][1] - gm_sp[e][1]))
        diff += np.max(abs(gm_sp2[e][2] - gm_sp[e][2]))
    print('test unkron_neq_gm:', diff)
    

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









