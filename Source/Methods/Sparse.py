#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 27 2024

@author: bercik
"""

import numpy as np
from numba import njit, int64, float64, complex64
from numba.typed import List
from numba.experimental import jitclass
global_tol = 1e-13

# Specification for a CSR matrix with float64 data
lm_float_spec = [
    ('data', float64[:]),
    ('indices', int64[:]),
    ('indptr', int64[:]),
    ('nrows', int64),
    ('ncols', int64),
    ('nnz', int64),
    ('tol', float64)
]

# will only ever use real matrices. No need for multiple typedispatch.
@jitclass(lm_float_spec)
class lmCSR:
    def __init__(self, data, indices, indptr):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nrows = len(indptr) - 1
        self.ncols = np.max(indices) + 1
        self.nnz = len(data)
        self.tol = global_tol # tolerance for zero values


    def dense(self, nrows=0, ncols=0):
        # Converts CSR format to dense 2D matrix (nrows, ncols)
        if nrows == 0:
            nrows = self.nrows
        else:
            assert nrows == len(self.indptr) - 1, f'Number of rows in CSR matrix {len(self.indptr) - 1} does not match given nrows {nrows}'
        if ncols == 0:
            ncols = np.max(self.indices) + 1
        A = np.zeros((nrows, ncols), dtype=self.data.dtype)
        
        for i in range(nrows):
            for j in range(self.indptr[i], self.indptr[i+1]):
                A[i, self.indices[j]] = self.data[j]
        return A
    
    def prune(self):
        # Removes entries in a CSR matrix representation that are below the specified tolerance.
        data = []
        indices = []
        indptr = [0]
        data_len = 0
        
        for i in range(self.nrows):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                value = self.data[j]
                if abs(value) >= self.tol:
                    data.append(value)
                    indices.append(self.indices[j])
                    data_len += 1
            indptr.append(data_len)

        if len(data) == 0: # insert at least one entry to avoid errors
            data = [0.]
            indices = [0]
            indptr = [0] + [1]*self.nrows
        
        self.data = np.array(data, dtype=np.float64)
        self.indices = np.array(indices, dtype=np.int64)
        self.indptr = np.array(indptr, dtype=np.int64)

    def mult_lv(self, vec):
        # Perform sparse matrix-vector multiplication in CSR format
        result = np.zeros(self.nrows, dtype=vec.dtype)
        
        for i in range(self.nrows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                result[i] += self.data[jj] * vec[self.indices[jj]]
        return result
    
    def mult_gv(self, vec):
        # Perform sparse matrix-vector multiplication in CSR format
        _, nelem = np.shape(vec)
        result = np.zeros((self.nrows,nelem), dtype=vec.dtype)
        
        for e in range(nelem):
            for i in range(self.nrows):
                for jj in range(self.indptr[i], self.indptr[i+1]):
                    result[i,e] += self.data[jj] * vec[self.indices[jj],e]
        return result
    
    def mult_ldiag(self, H):
        # Perform local matrix - diagonal multiplication in CSR format
        ncols = len(H)
        assert self.ncols <= ncols, f'Number of columns in CSR matrix {self.ncols} should be <= number of rows in H {ncols}'
        
        # Scale each entry in data by the corresponding entry in H based on the column index
        new_data = np.zeros(self.nnz, dtype=np.float64)
        for i in range(self.nnz):
            new_data[i] = self.data[i] * H[self.indices[i]]
        lm = lmCSR(new_data, self.indices, self.indptr)
        return lm
    
    def mult_gdiag(self, H):
        # Perform local matrix - global diagonal multiplication in CSR format
        ncols, nelem = np.shape(H)
        assert self.ncols <= ncols, f'Number of columns in CSR matrix {self.ncols} should be <= number of rows in H {ncols}'
        gm_list = [] # numba can not pickle jitclass objects
        
        for e in range(nelem):
            new_data = np.zeros(self.nnz, dtype=np.float64)
            # Scale each non-zero element in `data` by the corresponding H element for this `e`
            for i in range(self.nnz):
                new_data[i] = self.data[i] * H[self.indices[i], e]
            lm = lmCSR(new_data, self.indices, self.indptr)
            gm_list.append(lm)
        return gm_list
    
    def premult_ldiag(self, H):
        # perform global diagonal - local matrix multiplication in CSR format
        nrows = len(H)  # Number of elements
        assert (self.nrows == nrows), f"Number of rows in CSR matrix {self.nrows} does not match number of rows in H {nrows}"

        # Scale non-zero elements in each row by corresponding H[row]
        new_data = np.zeros(self.nnz, dtype=np.float64)
        for row in range(nrows):            
            row_scale = H[row]
            for i in range(self.indptr[row], self.indptr[row + 1]):
                new_data[i] = self.data[i] * row_scale
        lm = lmCSR(new_data, self.indices, self.indptr)
        return lm
    
    def premult_gdiag(self, H):
        # perform global diagonal - local matrix multiplication in CSR format
        nrows, nelem = np.shape(H)  # Number of elements
        assert self.nrows == nrows, f'Dimensions do not match {self.nrows}, {nrows}'
        gm_list = [] # numba can not pickle jitclass objects
        
        for e in range(nelem):
            new_data = np.zeros(self.nnz, dtype=np.float64)
            # Multiply each row's data by the corresponding H element for this `e`
            for row in range(nrows):            
                # Scale non-zero elements in the row by H[row, e]
                row_scale = H[row, e]
                for i in range(self.indptr[row], self.indptr[row + 1]):
                    new_data[i] = self.data[i] * row_scale
            lm = lmCSR(new_data, self.indices, self.indptr)
            gm_list.append(lm)
        return gm_list

    def T(self,nrows=0,ncols=0):
        # transpose a CSR matrix
        # nrows (optional) = number of rows in the original CSR matrix
        # ncols (optional) = number of columns in the original CSR matrix
        if nrows == 0: 
            nrows = self.nrows
        else:
            assert (nrows == self.nrows), f'Number of rows in CSR matrix {self.nrows} does not match inputted nrows {nrows}'
        if ncols == 0: ncols = self.ncols
        
        # Count non-zeros per column (to allocate space)
        col_counts = np.zeros(ncols, dtype=np.int64)
        for idx in self.indices:
            col_counts[idx] += 1

        # Build indptr for the transposed matrix
        indptrT = np.zeros(ncols + 1, dtype=np.int64)
        indptrT[1:] = np.cumsum(col_counts)

        # Initialize arrays for data and indices
        dataT = np.zeros(self.nnz, dtype=np.float64)
        indicesT = np.zeros(self.nnz, dtype=np.int64)
        next_position = np.zeros(ncols, dtype=np.int64)

        # Populate transposed data and indices
        for row in range(nrows):
            for i in range(self.indptr[row], self.indptr[row + 1]):
                col = self.indices[i]
                dest = indptrT[col] + next_position[col]
                dataT[dest] = self.data[i]
                indicesT[dest] = row
                next_position[col] += 1

        lm = lmCSR(dataT, indicesT, indptrT)
        return lm
    
    def kron_eye_lm(self, n, ncols=0):
        # Compute the Kronecker product of a n x n identity matrix and a CSR matrix.
        # n = number of rows / columns in the identity matrix
        # ncols (optional) = number of columns in the CSR matrix
        if ncols == 0: ncols = self.ncols

        # Prepare arrays for CSR format
        data = np.zeros(self.nnz * n, dtype=np.float64)
        indices = np.zeros(self.nnz * n, dtype=np.int64)
        indptr = np.zeros(self.nrows * n + 1, dtype=np.int64)

        # Manually repeat data and indices for each block
        for i in range(n):
            for j in range(self.nnz):
                data[i * self.nnz + j] = self.data[j]
                indices[i * self.nnz + j] = self.indices[j] + i * ncols
            
            # Update indptr for each row block
            for k in range(self.nrows + 1):
                indptr[i * self.nrows + k] = self.indptr[k] + i * self.nnz
        
        lm = lmCSR(data, indices, indptr)
        return lm
    
    def kron_lm_eye(self, n):
        # Compute the Kronecker product of a CSR matrix and a n x n identity matrix.
        # n = number of rows / columns in the identity matrix

        # Prepare arrays for the expanded CSR format
        data = np.zeros(self.nnz * n, dtype=np.float64)
        indices = np.zeros(self.nnz * n, dtype=np.int64)
        indptr = np.zeros(self.nrows * n + 1, dtype=np.int64)

        data_index = 0  # Position in data and indices

        # Iterate over each row in Mat, expanding into n rows in the result
        for i in range(self.nrows):
            for k in range(n):
                for j in range(self.indptr[i], self.indptr[i + 1]):
                    # Place the value along the diagonal of the n x n block
                    data[data_index] = self.data[j]
                    indices[data_index] = self.indices[j] * n + k
                    data_index += 1

                # Update indptr for the next row in the expanded matrix
                indptr[i * n + k + 1] = data_index

        lm = lmCSR(data, indices, indptr)
        return lm
    
    def kron_ldiag_lm(self, H, ncols=0):
        # Compute the Kronecker product of a local diagonal matrix and a CSR matrix.
        # ncols (optional) = number of columns in the CSR matrix
        if ncols == 0: ncols = self.ncols
        n = len(H)

        # Prepare arrays for the expanded CSR format
        data = np.zeros(self.nnz * n, dtype=np.float64)
        indices = np.zeros(self.nnz * n, dtype=np.int64)
        indptr = np.zeros(self.nrows * n + 1, dtype=np.int64)

        # Manually repeat data and indices for each block
        for i in range(n):
            for j in range(self.nnz):
                data[i * self.nnz + j] = self.data[j] * H[i]
                indices[i * self.nnz + j] = self.indices[j] + i * n
            
            # Update indptr for each row block
            for k in range(self.nrows + 1):
                indptr[i * self.nrows + k] = self.indptr[k] + i * self.nnz
        
        lm = lmCSR(data, indices, indptr)
        return lm
    
    def kron_lm_ldiag(self, H):
        # Compute the Kronecker product of a CSR matrix and a local diagonal matrix.
        n = len(H)

        # Prepare arrays for the expanded CSR format
        data = np.zeros(self.nnz * n, dtype=np.float64)
        indices = np.zeros(self.nnz * n, dtype=np.int64)
        indptr = np.zeros(self.nrows * n + 1, dtype=np.int64)

        data_index = 0  # Position in data and indices

        # Iterate over each row in Mat, expanding into n rows in the result
        for i in range(self.nrows):
            for k in range(n):
                for j in range(self.indptr[i], self.indptr[i + 1]):
                    # Place the value along the diagonal of the n x n block
                    data[data_index] = self.data[j] * H[k]
                    indices[data_index] = self.indices[j] * n + k
                    data_index += 1

                # Update indptr for the next row in the expanded matrix
                indptr[i * n + k + 1] = data_index

        lm = lmCSR(data, indices, indptr)
        return lm
    
    def copy(self):
        return lmCSR(self.data.copy(), self.indices.copy(), self.indptr.copy())
    


    
    



############ BEGIN FUNCTIONS FOR USE IN THE MAIN CODE ############

@njit
def sp_to_lm(A):
    return A.dense()

@njit
def lm_to_sp(A):
    '''
    Converts a dense 2D matrix (nrows, ncols) to CSR format.

    Parameters
    ----------
    A : numpy array of shape (nrows, ncols)

    Returns
    -------
        csr_data : tuple (data, indices, indptr)
        data : non-zero values of the matrix
        indices : column indices of the data elements
        indptr : row pointers indicating the start of each row in the data
    '''
    nrows, ncols = A.shape
    data = []
    indices = []
    indptr = [0]

    data_len = 0
    for i in range(nrows):
        for j in range(ncols):
            if abs(A[i, j]) > global_tol:
                data.append(A[i, j])
                indices.append(j)
                data_len += 1
        indptr.append(data_len)

    if len(data) == 0: # insert at least one entry to avoid errors
        data = [0.]
        indices = [0]
        indptr = [0] + [1]*nrows

    lm = lmCSR(np.array(data, dtype=np.float64), 
            np.array(indices, dtype=np.int64), 
            np.array(indptr, dtype=np.int64) )
    return lm


@njit
def gm_to_sp(A):
    '''
    Converts a global 3D matrix (nrows, ncols, nelem) into a list of CSR matrices,
    where each CSR matrix corresponds to a slice A[:,:,e].

    Parameters
    ----------
    A : numpy array of shape (nrows, ncols, nelem)

    Returns
    -------
    csr_list : List of CSR matrices in the form (data, indices, indptr)
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    '''
    _, _, nelem = A.shape
    csr_list = [] # numba can not pickle jitclass objects

    for e in range(nelem):
        # Extract the slice A[:,:,e] and convert it to CSR
        csr_data = lm_to_sp(A[:, :, e])
        csr_list.append(csr_data)

    return csr_list

@njit
def sp_to_gm(csr_list):
    '''
    Converts a list of CSR matrices to a global 3D matrix (nrows, ncols, nelem),
    where each CSR matrix corresponds to a slice A[:,:,e].

    Parameters
    ----------
    csr : List of CSR matrices 
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    A : numpy array of shape (nrows, ncols, nelem)

    Returns
    -------
    A : numpy array of shape (nrows, ncols, nelem)
    '''

    nelem = len(csr_list)
    nrows = csr_list[0].nrows
    ncols = get_gm_max_numcols(csr_list)
    A = np.zeros((nrows, ncols, nelem), dtype=np.float64)
    if ncols == 0: return A
    
    for e in range(nelem):
        A[:, :, e] = csr_list[e].dense(nrows, ncols)
    return A

@njit
def prune_lm(csr):
    csr.prune()
    return csr

@njit
def prune_gm(csr_list):
    """
    Removes entries in a list of CSR matrices that are below the specified tolerance.
    
    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    tol : float
        Entries with an absolute value below this tolerance will be removed.
    
    Returns
    -------
    pruned_csr_list : List of CSR matrices
        Each element in the list is a CSR matrix corresponding to a single element of the third dimension.
    """
    pruned_csr_list = [] # numba can not pickle jitclass objects
    
    for e in range(len(csr_list)):
        csr_list[e].prune()
        pruned_csr_list.append(csr_list[e])
    
    return pruned_csr_list

@njit
def get_gm_max_numcols(csr_list):
    max_value = 0
    for csr in csr_list:
        # Loop through each element in csr[1] to find the maximum
        if csr.ncols > max_value:
            max_value = csr.ncols
    return max_value

@njit
def lm_lv(csr, vec):
    return csr.mult_lv(vec)

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
    # Ensure matrices are compatible for multiplication
    assert csr1.ncols <= csr2.nrows, f"Matrix dimension mismatch: csr1 has {csr1.ncols} columns, csr2 has {csr2.nrows} rows."

    # Initialize result arrays
    indptr = np.zeros(csr1.nrows + 1, dtype=np.int64)
    data = []
    indices = []

    # Working arrays to accumulate results
    row_accumulator = np.zeros(csr2.ncols, dtype=np.float64)
    marker = -np.ones(csr2.ncols, dtype=np.int64)  # Marker array to track columns in row_accumulator

    # Perform the multiplication
    for i in range(csr1.nrows): # nrows
        # Reset the row accumulator and marker for each row
        row_accumulator.fill(0)
        marker.fill(-1)

        # Iterate over non-zero elements of row i in csr1
        for jj in range(csr1.indptr[i], csr1.indptr[i+1]):
            col1 = csr1.indices[jj]

            # Multiply with corresponding row in csr2
            for kk in range(csr2.indptr[col1], csr2.indptr[col1+1]):
                col2 = csr2.indices[kk]

                # Accumulate the result in row_accumulator
                if marker[col2] != i:
                    marker[col2] = i
                    row_accumulator[col2] = csr1.data[jj] * csr2.data[kk]
                else:
                    row_accumulator[col2] += csr1.data[jj] * csr2.data[kk]

        # Now collect all non-zero entries from row_accumulator for this row
        current_length = 0
        for j in range(csr2.ncols): # ncols
            if marker[j] == i and row_accumulator[j] != 0:
                indices.append(j)
                data.append(row_accumulator[j])
                current_length += 1
        
        # Update row pointer
        indptr[i + 1] = indptr[i] + current_length

    if len(data) == 0: # insert at least one entry to avoid errors
        data = [0.]
        indices = [0]
        indptr[1:] += 1

    # Convert lists to arrays for CSR format
    lm = lmCSR(np.array(data, dtype=np.float64), 
            np.array(indices, dtype=np.int64), indptr )
    return lm

@njit
def lm_gm(csr, csr_list):
    '''
    multiply a local matrix and global matrix

    Parameters
    ----------
    csr : CSR representation of the first matrix (non-zero values, column indices, row pointers)
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    # Initialize result array
    c = [] # numba can not pickle jitclass objects
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(len(csr_list)):
        c.append(lm_lm(csr, csr_list[e]))
    
    return c

@njit
def gm_to_gmT(csr_list, nrows=0, ncols=0):
    '''
    Take the tanspose of a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    c = [csr.T(nrows,ncols) for csr in csr_list]
    return c

@njit
def lm_to_lmT(csr, nrows=0, ncols=0):
    return csr.T(nrows,ncols)

@njit
def lm_lmT(csr1, csr2):
    return lm_lm(csr1, csr2.T())

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
    nrows, ncols, nelem = mat.shape
    assert (csr.ncols <= nrows), f"Matrix dimension mismatch: csr has {csr.ncols} columns, mat has {nrows} rows."
    result = np.zeros((csr.nrows, ncols, nelem), dtype=mat.dtype)
    
    for e in range(nelem):
        for i in range(csr.nrows):
            for jj in range(csr.indptr[i], csr.indptr[i+1]):
                col_idx = csr.indices[jj]
                for k in range(ncols):
                    result[i, k, e] += csr.data[jj] * mat[col_idx, k, e]
    
    return result

@njit
def gm_gv(csr_list, b):
    '''
    Perform global matrix-vector multiplication using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    b : numpy array of shape (ncols, nelem)
        The global vector for multiplication

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nelem = len(csr_list)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((csr_list[0].nrows, nelem), dtype=b.dtype)
    
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c[:, e] = csr_list[e].mult_lv(b[:, e])
    
    return c

@njit
def lm_gv(csr, b):
    return csr.mult_gv(b)

@njit
def lm_ldiag(csr, H):
    return csr.mult_ldiag(H)

@njit
def lm_gdiag(csr, H):
    return csr.mult_gdiag(H)

@njit
def gdiag_lm(H, csr):
    return csr.premult_gdiag(H)

@njit
def gdiag_gm(H, csr_list):
    '''
    Perform global diagonal - global matrix multiplication using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    H : numpy array of shape (ncols, nelem)
        The global diagonal for multiplication.

    Returns
    -------
    List of CSR matrices
        Each element in the list is a CSR matrix (data, indices, indptr) representing the scaled matrix.
    '''
    _, nelem = H.shape
    # Check if the number of CSR matrices matches the number of elements in H
    assert len(csr_list) == nelem, f"Dimensions do not match: number of CSR matrices must equal columns in H {len(csr_list)} != {nelem}"

    # Initialize result list
    gm_list = [] # numba can not pickle jitclass objects

    # Perform scaling for each CSR matrix in csr_list
    for e in range(nelem):
        lm = csr_list[e].premult_ldiag(H[:, e])
        gm_list.append(lm)
    
    return gm_list

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

    # Ensure both CSR matrices have the same number of rows
    assert csr1.nrows == csr2.nrows, f"CSR matrices must have the same number of rows {csr1.nrows} != {csr2.nrows}"
    
    indptr = np.zeros(csr1.nrows + 1, dtype=np.int64)
    data = []
    indices = []

    # Iterate over each row
    for i in range(csr1.nrows):
        start1, end1 = csr1.indptr[i], csr1.indptr[i + 1]
        start2, end2 = csr2.indptr[i], csr2.indptr[i + 1]
        idx1 = start1 # initialize pointers for both rows
        idx2 = start2
        
        # Merge the rows by iterating over both in sorted order
        while idx1 < end1 or idx2 < end2:
            if idx1 < end1 and (idx2 >= end2 or csr1.indices[idx1] < csr2.indices[idx2]):
                # Element from csr1 only
                col = csr1.indices[idx1]
                val = csr1.data[idx1]
                idx1 += 1
            elif idx2 < end2 and (idx1 >= end1 or csr2.indices[idx2] < csr1.indices[idx1]):
                # Element from csr2 only
                col = csr2.indices[idx2]
                val = csr2.data[idx2]
                idx2 += 1
            else:
                # Elements from both csr1 and csr2
                col = csr1.indices[idx1]  # csr1.indices[idx1] == csr2.indices[idx2] here
                val = csr1.data[idx1] + csr2.data[idx2]
                idx1 += 1
                idx2 += 1
            
            # Append only non-zero results
            if val != 0:
                indices.append(col)
                data.append(val)

        # Update the row pointer for the result matrix
        indptr[i + 1] = len(data)

    # Convert lists to arrays for CSR format
    if len(data) == 0: # insert at least one entry to avoid errors
        data = [0.]
        indices = [0]
        indptr[1:] += 1

    data = np.array(data, dtype=np.float64)
    indices = np.array(indices, dtype=np.int64)
    lm = lmCSR(data, indices, indptr)
    return lm

@njit
def subtract_lm_lm(csr1, csr2):
    lm = lmCSR((-1.)*csr2.data, csr2.indices, csr2.indptr)
    return add_lm_lm(csr1, lm)

@njit
def add_gm_gm(csr_list1, csr_list2):
    '''
    Perform global matrix addition using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    nelem = len(csr_list1)  # Number of elements (same as the third dimension of the original tensor)
    nelem2 = len(csr_list1) 

    assert csr_list1[0].nrows == csr_list2[0].nrows and nelem == nelem2, f"Dimensions do not match {csr_list1[0].nrows} != {csr_list2[0].nrows}, {nelem} != {nelem2}"
    
    # Initialize result array
    c = [] # numba can not pickle jitclass objects
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c.append(add_lm_lm(csr_list1[e], csr_list2[e]))
    return c

@njit
def subtract_gm_gm(csr_list1, csr_list2):
    '''
    Perform global matrix subtraction using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    nelem = len(csr_list1)  # Number of elements (same as the third dimension of the original tensor)
    nelem2 = len(csr_list1) 

    assert csr_list1[0].nrows == csr_list2[0].nrows and nelem == nelem2, f"Dimensions do not match {csr_list1[0].nrows} != {csr_list2[0].nrows}, {nelem} != {nelem2}"
    
    # Initialize result array
    c = [] # numba can not pickle jitclass objects
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c.append(subtract_lm_lm(csr_list1[e], csr_list2[e]))
    return c

@njit
def subtract_gm_lm(csr_list1, csr):
    '''
    Perform global-local matrix subtraction using a list of CSR matrices.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format
    csr : CSR matrix

    '''
    nelem = len(csr_list1)  # Number of elements (same as the third dimension of the original tensor)

    assert csr_list1[0].nrows == csr.nrows, f"Dimensions do not match {csr_list1[0].nrows} != {csr.nrows}"
    
    # Initialize result array
    c = [] # numba can not pickle jitclass objects
    # Perform sparse matrix-vector multiplication for each element
    for e in range(nelem):
        c.append(subtract_lm_lm(csr_list1[e], csr))
    return c

@njit
def scalar_gm(val, csr_list):
    '''
    Multiply a global matrix by a scalar.

    Parameters
    ----------
    csr_list : List of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format

    '''
    gm_list = [] # numba can not pickle jitclass objects

    for e in range(len(csr_list)):
        csr = csr_list[e]
        gm_list.append(lmCSR(val*csr.data, csr.indices, csr.indptr))
    
    return gm_list

@njit
def Vol_had_Fvol_diff(Vol_list,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the sparse Vol
    matrix and the Fvol matrix, then sum the rows. Made for 1d.

    Parameters
    ----------
    Vol_list : List of CSR matrices
        Represents the volume operators for each element
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    
    # loop for each element
    for e in range(nelem):
        Vol = Vol_list[e]
        
        for row in range(nen):
            qidx = row*neq
            
            for volidx in range(Vol.indptr[row], Vol.indptr[row + 1]):
                col = Vol.indices[volidx]
                if col <= row: continue  # Skip lower triangle and diagonal

                qidxT = col*neq
                f = flux(q[qidxT:qidxT+neq, e], q[qidx:qidx+neq, e])

                # add the result of the hadamard product
                c[qidx:qidx+neq, e] -= f * Vol.data[volidx]

                # Vol is skew-symmetric with respect to H, so reuse the flux calculation
                for volidxT in range(Vol.indptr[col], Vol.indptr[col + 1]):
                    if Vol.indices[volidxT] == row:
                        # add the result of the hadamard product to the transpose
                        c[qidxT:qidxT+neq, e] -= f * Vol.data[volidxT]
                        break

    return c

@njit
def Sat1d_had_Fsat_diff_periodic(taT,tb,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the sparse Vol
    matrix and the Fvol matrix, then sum the rows. Made for 1d, periodic bc.

    Parameters
    ----------
    taT, tb : CSR matrices
        Boundary operator for a SAT interface (since 1D, the same in each element)
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication, should be along a single row or column of elements
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    maxcols = max(taT.ncols, tb.ncols)
    
    # loop for each element
    for e in range(nelem): # will ignore right-most interface by periodic BC (same as leftmost)
        # here think of e as either the looping over elements and considering the left interface,
        # or looping over each interface e and stopping before hitting the rightmost 
        qL = q[:,e-1]
        qR = q[:,e]
        eb = e-1

        for row in range(nen):
            qidx = row*neq
            usedcols = np.zeros(maxcols, dtype=np.bool_)
            
            for taTidx in range(taT.indptr[row], taT.indptr[row + 1]):
                col = taT.indices[taTidx]
                usedcols[col] = True
                qidxT = col*neq

                f = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # taT_data: add the result of the hadamard product
                c[qidxT:qidxT+neq, e] += f * taT.data[taTidx]

                # tbx_data: reuse the flux calculation if possible
                for tbidx in range(tb.indptr[row], tb.indptr[row + 1]):
                    if tb.indices[tbidx] == col:
                        c[qidx:qidx+neq, eb] -= f * tb.data[tbidx]
                        break

            for tbidx in range(tb.indptr[row], tb.indptr[row + 1]):
                col = tb.indices[tbidx]
                if usedcols[col]: continue # Skip if this column was already done
                usedcols[col] = True
                qidxT = col*neq

                f = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # tbx_data: add the result of the hadamard product
                c[qidx:qidx+neq, eb] -= f * tb.data[tbidx]
            
    return c

@njit
def VolxVoly_had_Fvol_diff(Volx_list,Voly_list,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the sparse Volx and Voly
    matrices and the Fvol matrices, then sum the rows. Made for 2d.

    Parameters
    ----------
    Volx_list, Voly_list : List of CSR matrices
        These represent the volume operators for each element
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx + dEydy on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq

    # Sanity checks on sizes of arrays - can comment this out later
    #nelemb, nelemc = len(Volx), len(Voly)
    #nenb, nenc = Volx[0].nrows, Voly[0].nrows
    #if nelemb != nelem or nelemc != nelem or nenb != nen or nenc != nen:
    #    raise ValueError('Dimensions do not match', nelemb, nelemc, nenb, nenc, nelem, nen)

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    
    # loop for each element
    for e in range(nelem):
        Volx, Voly = Volx_list[e], Voly_list[e]
        
        for row in range(nen):
            qidx = row*neq
            
            for volxidx in range(Volx.indptr[row], Volx.indptr[row + 1]):
                col = Volx.indices[volxidx]
                if col <= row: continue  # Skip lower triangle and diagonal

                qidxT = col*neq
                fx, fy = flux(q[qidxT:qidxT+neq, e], q[qidx:qidx+neq, e])

                # add the result of the hadamard product
                c[qidx:qidx+neq, e] -= fx * Volx.data[volxidx]

                # Volx is skew-symmetric with respect to H, so reuse the flux calculation
                for volxidxT in range(Volx.indptr[col], Volx.indptr[col + 1]):
                    if Volx.indices[volxidxT] == row:
                        # add the result of the hadamard product to the transpose
                        c[qidxT:qidxT+neq, e] -= fx * Volx.data[volxidxT]
                        break

                # Check if col is in Voly as well to reuse fy
                for volyidx in range(Voly.indptr[row], Voly.indptr[row + 1]):
                    if Voly.indices[volyidx] == col:

                        # add the result of the hadamard product
                        c[qidx:qidx+neq, e] -= fy * Voly.data[volyidx]

                        for volyidxT in range(Voly.indptr[col], Voly.indptr[col + 1]):
                            if Voly.indices[volyidxT] == row:
                                # add the result of the hadamard product to the transpose
                                c[qidxT:qidxT+neq, e] -= fy * Voly.data[volyidxT]
                                break
                        break
                
            # loop over remaining y columns
            colsx = Volx.indices[Volx.indptr[row]:Volx.indptr[row + 1]]
            for volyidx in range(Voly.indptr[row], Voly.indptr[row + 1]):
                col = Voly.indices[volyidx]
                if col <= row or col in colsx: continue  # Skip lower triangle, diagonal, and already done columns
                
                qidxT = col*neq
                _, fy = flux(q[qidxT:qidxT+neq, e], q[qidx:qidx+neq, e])
                
                # add the result of the hadamard product
                c[qidx:qidx+neq, e] -= fy * Voly.data[volyidx]

                # Voly is skew-symmetric with respect to H, so reuse the flux calculation
                for volyidxT in range(Voly.indptr[col], Voly.indptr[col + 1]):
                    if Voly.indices[volyidxT] == row:
                        # add the result of the hadamard product
                        c[qidxT:qidxT+neq, e] -= fy * Voly.data[volyidxT]
                        break

    return c

@njit
def Sat2d_had_Fsat_diff_periodic(taTx_list,taTy_list,tbx_list,tby_list,q,flux,neq):
    '''
    A specialized function to compute the hadamard product between the sparse Volx and Voly
    matrices and the Fvol matrices, then sum the rows. Made for 2d, periodic bc.

    Parameters
    ----------
    taTx, taTy, tbx, tby : List of CSR matrices
        Each element represents a boundary operator for a SAT interface
    q : numpy array of shape (nen_neq, nelem)
        The global vector for multiplication, should be along a single row or column of elements
    flux : function
        Function to compute the flux between two states
    neq : int
        Number of equations in the system

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the volume flux differencing
        note the -ve so that it corresponds to dExdx + dEydy on the Right Hand Side
    '''
    nen_neq, nelem = q.shape
    nen = nen_neq // neq

    # Sanity checks on sizes of arrays - can comment this out later
    #nelemb, nelemc, nelemd, neleme = len(taTx), len(taTy), len(tbx), len(tby)
    #if nelemb != nelem or nelemc != nelem or nelemd != nelem or neleme != nelem:
    #    raise ValueError('Number of elements do not match', nelemb, nelemc, nelemd, neleme, nelem)
    #nenb, nenc, nend, nene = taTx[0].nrows, taTy[0].nrows, tbx[0].nrows, tby[0].nrows
    #if nenb != nen or nenc != nen or nend != nen or nene != nen:
    #    raise ValueError('Number of nodes do not match', nenb, nenc, nend, nene, nen)

    # Initialize result array
    c = np.zeros((nen_neq, nelem), dtype=q.dtype)
    
    # loop for each element
    for e in range(nelem): # will ignore right-most interface by periodic BC (same as leftmost)
        # here think of e as either the looping over elements and considering the left interface,
        # or looping over each interface e and stopping before hitting the rightmost 
        qL = q[:,e-1]
        qR = q[:,e]
        eb = e-1
        taTx, taTy = taTx_list[e], taTy_list[e]
        tbx, tby = tbx_list[eb], tby_list[eb]
        maxcols = max(taTx.ncols, taTy.ncols, tbx.ncols, tby.ncols)

        for row in range(nen):
            qidx = row*neq
            usedcols = np.zeros(maxcols, dtype=np.bool_)
            
            for taTxidx in range(taTx.indptr[row], taTx.indptr[row + 1]):
                col = taTx.indices[taTxidx]
                usedcols[col] = True
                qidxT = col*neq

                fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # taTx_data: add the result of the hadamard product
                c[qidxT:qidxT+neq, e] += fx * taTx.data[taTxidx]

                # taTy_data: reuse the flux calculation if possible
                for taTyidx in range(taTy.indptr[row], taTy.indptr[row + 1]):
                    if taTy.indices[taTyidx] == col:
                        c[qidxT:qidxT+neq, e] += fy * taTy.data[taTyidx]
                        break

                # tbx_data: reuse the flux calculation if possible
                for tbxidx in range(tbx.indptr[row], tbx.indptr[row + 1]):
                    if tbx.indices[tbxidx] == col:
                        c[qidx:qidx+neq, eb] -= fx * tbx.data[tbxidx]
                        break
                
                # tby_data: reuse the flux calculation if possible
                for tbyidx in range(tby.indptr[row], tby.indptr[row + 1]):
                    if tby.indices[tbyidx] == col:
                        c[qidx:qidx+neq, eb] -= fy * tby.data[tbyidx]
                        break

            for taTyidx in range(taTy.indptr[row], taTy.indptr[row + 1]):
                col = taTy.indices[taTyidx]
                if usedcols[col]: continue # Skip if this column was already done
                usedcols[col] = True
                qidxT = col*neq

                fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # taTy_data: add the result of the hadamard product
                c[qidxT:qidxT+neq, e] += fy * taTy.data[taTyidx]

                # tbx_data: reuse the flux calculation if possible
                for tbxidx in range(tbx.indptr[row], tbx.indptr[row + 1]):
                    if tbx.indices[tbxidx] == col:
                        c[qidx:qidx+neq, eb] -= fx * tbx.data[tbxidx]
                        break
                
                # tby_data: reuse the flux calculation if possible
                for tbyidx in range(tby.indptr[row], tby.indptr[row + 1]):
                    if tby.indices[tbyidx] == col:
                        c[qidx:qidx+neq, eb] -= fy * tby.data[tbyidx]
                        break

            for tbxidx in range(tbx.indptr[row], tbx.indptr[row + 1]):
                col = tbx.indices[tbxidx]
                if usedcols[col]: continue # Skip if this column was already done
                usedcols[col] = True
                qidxT = col*neq

                fx, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # tbx_data: add the result of the hadamard product
                c[qidx:qidx+neq, eb] -= fx * tbx.data[tbxidx]

                # tby_data: reuse the flux calculation if possible
                for tbyidx in range(tby.indptr[row], tby.indptr[row + 1]):
                    if tby.indices[tbyidx] == col:
                        c[qidx:qidx+neq, eb] -= fy * tby.data[tbyidx]
                        break

            for tbyidx in range(tby.indptr[row], tby.indptr[row + 1]):
                if usedcols[col]: continue # Skip if this column was already done
                usedcols[col] = True
                qidxT = col*neq

                _, fy = flux(qL[qidx:qidx+neq], qR[qidxT:qidxT+neq])

                # tby_data: add the result of the hadamard product
                c[qidx:qidx+neq, eb] -= fy * tby.data[tbyidx]
            
    return c

@njit
def set_gm_union_sparsity(gm_list):
    '''
    Set the sparsity pattern of a list of global dense matrices (shape: nrows, ncols, nelem)
    to the union of all sparsity patterns.

    Parameters
    ----------
    gm_list : List of global matrices
        Each element in the list is an ndarray (nrows, ncols, nelem)

    Returns
    -------
    sparsity : tuple (indices, indptr)
        The sparsity pattern of the union of all given matrices
    '''
    ngm = len(gm_list)
    nrows, ncols, nelem = gm_list[0].shape

    for gm in gm_list:
        assert gm.shape == (nrows, ncols, nelem), f"Dimensions do not match {gm.shape} != {(nrows, ncols, nelem)}"
        
    sp_gm_list = List()

    for e in range(nelem):
        # Initialize CSR format arrays
        indices = []
        indptr = []
        indptr.append(0)
        data_len = 0

        for i in range(nrows):
            for j in range(ncols):
                any_above_tol = False
                for k in range(ngm):
                    if abs(gm_list[k][i, j, e]) >= global_tol:
                        any_above_tol = True
                        break
                if any_above_tol:
                    data_len += 1
                    indices.append(j)
            indptr.append(data_len)

        sp_gm_list.append((np.array(indices, dtype=np.int64), np.array(indptr, dtype=np.int64)))
    
    return sp_gm_list


#@njit # njit leads to some error that I don't care enough to fix. Only gets called once anyway.
# i think the issue is with set
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
        assert len(csr) == nelem, f"Number of elements do not match {len(csr)} != {nelem}"

    sparsity_list = []

    for e in range(nelem):
        # Ensure all matrices have the same number of rows
        n_rows = csr_list[0][e].nrows
        for csr in csr_list:
            assert csr[e].nrows == n_rows, f"Number of rows do not match {csr[e].nrows} != {n_rows}"

        indices_glob = []
        indptr_glob = [0]
        nnz = 0

        # Loop over each row to determine the union of nonzero elements
        for row in range(n_rows):
            row_indices_set = set()
            for k in range(ngm):
                lm = csr_list[k][e]
                for col_ptr in range(lm.indptr[row], lm.indptr[row + 1]):
                    row_indices_set.add(lm.indices[col_ptr])

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
    c : numpy array of shape (ncols,nelem)
    '''
    assert A.nrows == B.nrows, f'Number of rows in CSR matrix {A.nrows} does not match number of rows in CSR matrix {B.nrows}'
    
    c = np.zeros(A.nrows, dtype=np.float64)
    for row in range(A.nrows):
        for col_ptr1 in range(A.indptr[row], A.indptr[row+1]):
            col = A.indices[col_ptr1]
            for col_ptr2 in range(B.indptr[row], B.indptr[row+1]):
                if col == B.indices[col_ptr2]:
                    c[row] += A.data[col_ptr1] * B.data[col_ptr2]
    
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
    c : numpy array of shape (ncols,nelem)
    '''
    # Get the CSR data for the current element
    assert AT.nrows == B.nrows, f'Number of rows in CSR matrix {AT.nrows} does not match number of rows in CSR matrix {B.nrows}'
    
    c = np.zeros(AT.nrows, dtype=np.float64)
    for row in range(AT.nrows):
        for col_ptr1 in range(AT.indptr[row], AT.indptr[row+1]):
            col = AT.indices[col_ptr1]
            for col_ptr2 in range(B.indptr[row], B.indptr[row+1]):
                if col == B.indices[col_ptr2]:
                    c[col] += AT.data[col_ptr1] * B.data[col_ptr2]
    return c

@njit
def lm_dlm_had_diff(A,B):
    '''
    Compute the Hadamard product between a CSR matrices and a dense
    local matrix, then sum rows

    Parameters
    ----------
    A (data1, indices1, indptr1) : CSR representation of the first matrix
    B : (nrows, ncols) numpy array

    Returns
    -------
    c : numpy array of shape (nrows,nelem)
    '''
    nrows,ncols = B.shape
    assert A.nrows == nrows, f'Number of rows in CSR matrix {A.nrows} does not match number of rows in dense matrix {nrows}'
    assert A.ncols <= ncols, f'Number of columns in CSR matrix {A.ncols} is greater than number of columns in dense matrix {ncols}'

    c = np.zeros(nrows, dtype=B.dtype)
    for row in range(nrows):
        for col_ptr in range(A.indptr[row], A.indptr[row+1]):
            col = A.indices[col_ptr]
            c[row] += A.data[col_ptr] * B[row, col]
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
    B : (nrows, ncols) numpy array

    Returns
    -------
    c : numpy array of shape (ncols,nelem)
    '''
    nrows,ncols = B.shape
    assert AT.nrows == ncols, f'Number of rows in CSR matrix {AT.nrows} does not match number of columns in dense matrix {ncols}'
    assert AT.ncols <= nrows, f'Number of columns in CSR matrix {AT.ncols} is greater than number of row in dense matrix {nrows}'

    c = np.zeros(nrows, dtype=B.dtype)
    for row in range(nrows):
        for col_ptr in range(AT.indptr[row], AT.indptr[row+1]):
            col = AT.indices[col_ptr]
            c[col] += AT.data[col_ptr] * B[row, col]
    
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
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows = A[0].nrows  # Number of rows in the sparse matrix (from indptr)
    nelem = len(A)  # Number of elements (same as the third dimension of the original tensor)
    assert (nelem == len(B)), f'Number of elements in A {nelem} does not match number of elements in B {len(B)}'
    
    # Initialize result array
    c = np.zeros((nrows, nelem), np.float64)
    
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
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows = B[0].nrows  # Number of rows in the sparse matrix (from indptr)
    nelem = len(AT)  # Number of elements (same as the third dimension of the original tensor)
    assert (nelem == len(B)), f'Number of elements in AT {nelem} does not match number of elements in B {len(B)}'
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=np.float64)
    
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
    B : numpy array of shape (nrows, ncols, nelem)

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows = A[0].nrows  # Number of rows in the sparse matrix (from indptr)
    nelem = len(A)  # Number of elements (same as the third dimension of the original tensor)
    nrowsb, _, nelemb = np.shape(B)
    assert (nelem == nelemb), f'Number of elements in A {nelem} does not match number of elements in B {nelemb}'
    assert (nrows == nrowsb), f'Number of rows in A {nrows} does not match number of rows in B {nrowsb}'
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=B.dtype)
    
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
    B : numpy array of shape (nrows, ncols, nelem)

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows, _, nelem = np.shape(B)
    assert (nelem == len(AT)), f'Number of elements in AT {len(AT)} does not match number of elements in B {nelem}'
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=B.dtype)
    
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
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nelem = len(B)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((A.nrows, nelem), dtype=np.float64)
    
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
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows = B[0].nrows 
    nelem = len(B)  # Number of elements (same as the third dimension of the original tensor)
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=np.float64)
    
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
    B : numpy array of shape (nrows, ncols, nelem)

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows, _, nelem = np.shape(B)
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=B.dtype)
    
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
    B : B : numpy array of shape (ncols, nen3, nelem)

    Returns
    -------
    c : numpy array of shape (nrows, nelem)
        Result of the matrix-vector multiplication
    '''
    nrows, _, nelem = np.shape(B)
    
    # Initialize result array
    c = np.zeros((nrows, nelem), dtype=B.dtype)
    
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
    F_vol = [] # numba can not pickle jitclass objects
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity[e][0]
        indptr = sparsity[e][1]
        new_data = np.zeros((len(indices)), dtype=np.float64)       
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


        F_vol.append(lmCSR(new_data, indices, indptr))
    return F_vol

@njit 
def build_F_sca(q1, q2, flux, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen, nelem = q1.shape 
    F = [] # numba can not pickle jitclass objects
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity[e][0]
        indptr = sparsity[e][1]
        new_data = np.zeros((len(indices)), dtype=np.float64)       
        
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

        F.append(lmCSR(new_data, indices, indptr))
    return F

#@njit 
def build_F_vol_sys(neq, q, flux, sparsity_unkronned, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.
    Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    F_vol = [] # numba can not pickle jitclass objects
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity_unkronned[e][0]
        indptr = sparsity_unkronned[e][1]
        new_indices = sparsity[e][0]
        new_indptr = sparsity[e][1]
        new_data = np.zeros((len(new_indices)), dtype=np.float64)       
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


        F_vol.append(lmCSR(new_data, new_indices, new_indptr))
    return F_vol

@njit 
def build_F_sys(neq, q1, q2, flux, sparsity_unkronned, sparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen_neq, nelem = q1.shape 
    F = [] # numba can not pickle jitclass objects
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        indices = sparsity_unkronned[e][0]
        indptr = sparsity_unkronned[e][1]
        new_indices = sparsity[e][0]
        new_indptr = sparsity[e][1]
        new_data = np.zeros((len(new_indices)), dtype=np.float64)       
        
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

        F.append(lmCSR(new_data, new_indices, new_indptr))
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
    Fx_vol = [] # numba can not pickle jitclass objects
    Fy_vol = [] # numba can not pickle jitclass objects

    for e in range(nelem):
        # Initialize lists to store CSR data for x and y directions
        xindices = xsparsity[e][0]
        xindptr = xsparsity[e][1]
        yindices = ysparsity[e][0]
        yindptr = ysparsity[e][1]

        xnew_data = np.zeros(len(xindices), dtype=np.float64)
        ynew_data = np.zeros(len(yindices), dtype=np.float64)
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

        Fx_vol.append(lmCSR(xnew_data, xindices, xindptr))
        Fy_vol.append(lmCSR(ynew_data, yindices, yindptr))

    return Fx_vol, Fy_vol
    
@njit
def build_F_vol_sys_2d(neq, q, flux, xsparsity_unkronned, xsparsity,
                                     ysparsity_unkronned, ysparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.
    Takes advantage of symmetry since q1 = q2 = q '''
    nen_neq, nelem = q.shape 
    Fx_vol = [] # numba can not pickle jitclass objects
    Fy_vol = [] # numba can not pickle jitclass objects
    nen = nen_neq // neq
    
    for e in range(nelem):
        # Extract sparsity patterns for element `e`
        xindices, xindptr = xsparsity_unkronned[e]
        xnew_indices, xnew_indptr = xsparsity[e]
        yindices, yindptr = ysparsity_unkronned[e]
        ynew_indices, ynew_indptr = ysparsity[e]
        
        # Initialize sparse data arrays
        xnew_data = np.zeros(len(xnew_indices), dtype=np.float64)
        ynew_data = np.zeros(len(ynew_indices), dtype=np.float64)
        xcolptrs = np.zeros(nen, dtype=np.int64)
        ycolptrs = np.zeros(nen, dtype=np.int64)

        for i in range(nen):
            idxi = i * neq
            idxi2 = (i + 1) * neq
            xcol_start, xcol_end = xindptr[i], xindptr[i + 1]
            ycol_start, ycol_end = yindptr[i], yindptr[i + 1]

            for j in range(i, nen):
                idxj = j * neq
                idxj2 = (j + 1) * neq
                xcol_start_T, xcol_end_T = xindptr[j], xindptr[j + 1]
                ycol_start_T, ycol_end_T = yindptr[j], yindptr[j + 1]

                xadd_entry = (j in xindices[xcol_start:xcol_end])
                xadd_entry_T = (i in xindices[xcol_start_T:xcol_end_T]) and (i != j)
                yadd_entry = (j in yindices[ycol_start:ycol_end])
                yadd_entry_T = (i in yindices[ycol_start_T:ycol_end_T]) and (i != j)

                # Proceed only if any entry is required
                if xadd_entry or xadd_entry_T or yadd_entry or yadd_entry_T:
                    xdiag, ydiag = flux(q[idxi:idxi2, e], q[idxj:idxj2, e])

                    for k in range(neq):
                        new_row = i * neq + k
                        new_col = j * neq + k

                        if xadd_entry:
                            xnew_col_start = xnew_indptr[new_row]
                            xnew_data[xnew_col_start + xcolptrs[i]] = xdiag[k]
                        if xadd_entry_T:
                            xnew_col_start = xnew_indptr[new_col]
                            xnew_data[xnew_col_start + xcolptrs[j]] = xdiag[k]
                        if yadd_entry:
                            ynew_col_start = ynew_indptr[new_row]
                            ynew_data[ynew_col_start + ycolptrs[i]] = ydiag[k]
                        if yadd_entry_T:
                            ynew_col_start = ynew_indptr[new_col]
                            ynew_data[ynew_col_start + ycolptrs[j]] = ydiag[k]

                    # Update column pointers after each assignment
                    if xadd_entry: xcolptrs[i] += 1
                    if xadd_entry_T: xcolptrs[j] += 1
                    if yadd_entry: ycolptrs[i] += 1
                    if yadd_entry_T: ycolptrs[j] += 1

        # Append sparse matrix data for the element
        Fx_vol.append(lmCSR(xnew_data, xnew_indices, xnew_indptr))
        Fy_vol.append(lmCSR(ynew_data, ynew_indices, ynew_indptr))

    return Fx_vol, Fy_vol

@njit 
def build_F_sys_2d(neq, q1, q2, flux, xsparsity_unkronned, xsparsity,
                                      ysparsity_unkronned, ysparsity):
    ''' Builds a sparsified Flux differencing matrix (used for Hadamard form) given a 
    solution vector q, the number of equations per node, a 2-point flux function, and 
    a sparsity pattern. Only computes the entries specified by indices and indptr.'''
    nen_neq, nelem = q1.shape 
    Fx = [] # numba can not pickle jitclass objects
    Fy = [] # numba can not pickle jitclass objects
    nen = nen_neq // neq
    for e in range(nelem):
        # Initialize lists to store CSR data
        xindices = xsparsity_unkronned[e][0]
        xindptr = xsparsity_unkronned[e][1]
        xnew_indices = xsparsity[e][0]
        xnew_indptr = xsparsity[e][1]
        xnew_data = np.zeros((len(xnew_indices)), dtype=np.float64)   
        yindices = ysparsity_unkronned[e][0]
        yindptr = ysparsity_unkronned[e][1]
        ynew_indices = ysparsity[e][0]
        ynew_indptr = ysparsity[e][1]
        ynew_data = np.zeros((len(ynew_indices)), dtype=np.float64)    
        
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

        Fx.append(lmCSR(xnew_data, xnew_indices, xnew_indptr))
        Fy.append(lmCSR(ynew_data, ynew_indices, ynew_indptr))
    return Fx, Fy

@njit
def unkron_neq_gm(csr_list, neq):
    '''
    Take a list of CSR matrices of shape (nen*neq, ncols*neq, nelem) and return a new list of CSR matrices
    of shape (nen, ncols, nelem), effectively undoing the Kronecker product operation for
    the operator acting on a vector (ncols*neq, nelem).

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
    unkron_csr_list = [] # numba can not pickle jitclass objects

    for e in range(len(csr_list)):
        lm = csr_list[e]
        nen = lm.nrows // neq

        new_data = []
        new_indices = []
        new_indptr = [0] * (nen + 1)

        for row in range(nen):
            nnz_counter = 0

            for col_idx in range(lm.indptr[row * neq], lm.indptr[row * neq + 1]):
                col = lm.indices[col_idx]
                if col % neq == 0:
                    new_data.append(lm.data[col_idx])
                    new_indices.append(lm.indices[col_idx] // neq)
                    nnz_counter += 1

            new_indptr[row + 1] = new_indptr[row] + nnz_counter

        new_lm = lmCSR(np.array(new_data, dtype=np.float64),
                        np.array(new_indices, dtype=np.int64),
                        np.array(new_indptr, dtype=np.int64))

        unkron_csr_list.append(new_lm)

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

        unkron_csr_list.append(List([np.array(new_indices, dtype=np.int64),
                                    np.array(new_indptr, dtype=np.int64)]))

    return unkron_csr_list

@njit
def assemble_satx_2d(csr_list,nelemx,nelemy):
    ''' given a list of csr matrices like (nen,ncols,nelem[idx]), 
    put them back in global order (nen,ncols,nelem)
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
    assert nelemy2 == nelemy, f'nelemy does not match, {nelemy2} != {nelemy}'
    nelemx2 = len(csr_list[0])
    assert nelemx2 == nelemx, f'nelemx does not match, {nelemx2} != {nelemx}'

    mat_glob = [(np.zeros(0, dtype=csr_list[0][0][0].dtype), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))] * (nelemx * nelemy)
    for ey in range(nelemy):
        for ex in range(nelemx):
            idx = ex * nelemy + ey
            mat_glob[idx] = csr_list[ey][ex]
    return mat_glob

@njit
def assemble_saty_2d(csr_list,nelemx,nelemy):
    '''
    Given a list of CSR matrices of shape (nen, ncols, nelem[idx]),
    put them back in global order (nen, ncols, nelem)
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
    assert nelemx2 == nelemx, f'nelemx does not match, {nelemx2} != {nelemx}'
    nelemy2 = len(csr_list[0])
    assert nelemy2 == nelemy, f'nelemy does not match, {nelemy2} != {nelemy}'

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
    Dx, Dy : sparse lmCSR matrices

    Returns
    -------
    csr_kron : the CSR matrix of the Kronecker product, i.e. np.kron(Dx, Dy)
    '''
    # Initialize lists to store the resulting CSR components
    data = []
    indices = []
    indptr = [0]
    
    for i in range(Dx.nrows):
        for j in range(Dy.nrows):
            # Get the non-zero row slices for current row in Dx and Dy
            Dx_row_start = Dx.indptr[i]
            Dx_row_end = Dx.indptr[i + 1]
            Dy_row_start = Dy.indptr[j]
            Dy_row_end = Dy.indptr[j + 1]
            
            for dx_idx in range(Dx_row_start, Dx_row_end):
                for dy_idx in range(Dy_row_start, Dy_row_end):
                    # Compute the data value
                    value = Dx.data[dx_idx] * Dy.data[dy_idx]
                    data.append(value)
                    
                    # Compute the column index
                    dx_col = Dx.indices[dx_idx]
                    dy_col = Dy.indices[dy_idx]
                    col_index = dx_col * Dy.ncols + dy_col
                    indices.append(col_index)
                    
            # Update indptr for the Kronecker product row
            indptr.append(len(data))
    
    # Convert lists to arrays for CSR format
    lm = lmCSR(np.array(data, dtype=np.float64), 
            np.array(indices, dtype=np.int64), 
            np.array(indptr, dtype=np.int64) )
    return lm

@njit
def kron_eye_lm(Dy, n, ncols=0):
    return Dy.kron_eye_lm(n, ncols)

@njit
def kron_lm_eye(Dx, n):
    return Dx.kron_lm_eye(n)

@njit
def kron_ldiag_lm(diag, Dy, ncols=0):
    return Dy.kron_ldiag_lm(diag, ncols)

@njit
def kron_lm_ldiag(Dx, diag):
    return Dx.kron_lm_ldiag(diag)

@njit
def kron_gm_eye(Dx_list, n):
    '''
    Compute the Kronecker product of a list of CSR matrices and a n x n identity matrix.

    Parameters
    ----------
    Dx_list : List of CSR matrices
        Each element is a tuple (data, indices, indptr) representing a sparse matrix in CSR format.
    n : int
        Size of the identity matrix.

    Returns
    -------
    List of CSR matrices
        Each element in the list is a CSR matrix (data, indices, indptr) representing the Kronecker product for each CSR in Dx_list.
    '''
    # Prepare a list to store the result CSR matrices
    kron_result_list = [] # numba can not pickle jitclass objects

    # Iterate over each CSR matrix in Dx_list
    for Dx in Dx_list:
        kron_result_list.append(Dx.kron_lm_eye(n))

    return kron_result_list

@njit
def kron_neq_lm(Dx, neq):
    return Dx.kron_lm_eye(neq)

@njit
def kron_neq_gm(csr_list, neq_node):
    return kron_gm_eye(csr_list, neq_node)

@njit
def is_skewsym(global_matrix):
    '''
    Checks if each CSR matrix in a global matrix (list of CSR matrices) is skew-symmetric.

    Parameters
    ----------
    global_matrix : list of CSR matrices
        Each element in the list is a tuple (data, indices, indptr) representing a sparse matrix in CSR format.

    Returns
    -------
    bool
        True if all CSR matrices are skew-symmetric, False otherwise.
    '''
    for matrix in global_matrix:
        for row in range(matrix.n_rows):
            # Loop over non-zero columns in the current row
            for idx in range(matrix.indptr[row], matrix.indptr[row + 1]):
                col = matrix.indices[idx]
                if col > row:  # Check only upper triangular part
                    value = matrix.data[idx]

                    # Find the corresponding element in the lower triangular part
                    # i.e., in row `col` and column `row`
                    found = False
                    for rev_idx in range(matrix.indptr[col], matrix.indptr[col + 1]):
                        if matrix.indices[rev_idx] == row:
                            if abs(matrix.data[rev_idx] + value) > global_tol:
                                print(col,row,value,matrix.data[rev_idx])
                                return False  # Not skew-symmetric
                            found = True
                            break
                    
                    # If no corresponding element found, not skew-symmetric
                    if not found:
                        return False

    return True

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
        diff += np.max(abs(sp_gm_list[e][0] - gm_sp[e].indices))
        diff += np.max(abs(sp_gm_list[e][1] - gm_sp[e].indptr))
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
        diff += np.max(abs(gm_sp2[e].data - gm_sp[e].data))
        diff += np.max(abs(gm_sp2[e].indices - gm_sp[e].indices))
        diff += np.max(abs(gm_sp2[e].indptr - gm_sp[e].indptr))
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









