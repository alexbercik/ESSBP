#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous SBP (C-SBP) solver implementation
Inherits from PdeSolverSbp and implements global DOF storage with gather/scatter operations

Created on Jan 12 2026

@author: bercik
"""

import numpy as np
from Source.Methods.Functions import gdiag_gv, ldiag_lv
from Source.Solvers.PdeSolverSbp import PdeSolverSbp


class NullSAT:
    """Null SAT object that always returns zeros, used to disable internal SATs in C-SBP"""
    
    def __init__(self, shape):
        """
        Parameters
        ----------
        shape : tuple
            Shape of the local solution array (n, m) where n=nodes per element, m=number of elements
        """
        self.shape = shape
        self.neq_node = None  # Will be set if needed
        
    def calc(self, *args, **kwargs):
        """Return zeros with the appropriate shape"""
        # Extract shape from first argument if it's an array
        if len(args) > 0 and hasattr(args[0], 'shape'):
            return np.zeros_like(args[0])
        # Otherwise use stored shape
        return np.zeros(self.shape)
    
    def calc_dfdq(self, *args, **kwargs):
        """Return zeros for linearized SAT"""
        if len(args) > 0 and hasattr(args[0], 'shape'):
            q = args[0]
            # Return zeros with appropriate shape for dfdq
            return np.zeros((q.size, q.size))
        return np.zeros((self.shape[0] * self.shape[1], self.shape[0] * self.shape[1]))


class PdeSolverCSbp(PdeSolverSbp):
    """
    Continuous SBP (C-SBP) solver with global DOF storage and matrix-free gather/scatter.
    
    This class stores the solution in a global conforming DOF vector and uses gather/scatter
    operations to interface with element-local kernels from the base PdeSolverSbp class.
    """
    
    def init_disc_specific(self):
        """Initialize C-SBP specific discretization"""
        # First call base class initialization
        if self.print_progress:
            print('C-SBP initializing: First calling base SBP class, will overwrite with null SATs later.')
        super().init_disc_specific()
        
        # Assert periodic boundary conditions only (for now)
        if self.dim == 1:
            assert self.periodic, "C-SBP currently only supports periodic boundary conditions"
        else:
            assert all(self.periodic), "C-SBP currently only supports periodic boundary conditions"

        # Assert that we use operators with boundary nodes
        assert self.sbp.bdy_nodes, "C-SBP only supports operators with boundary nodes"

        # Build global-to-local mapping (gid)
        self._build_global_id_mapping()
        
        # Precompute local and global H weights
        self._build_H_weights()
        
        # Replace SAT objects with NullSAT to disable internal coupling
        self._replace_sats_with_null()
        
        # Override shape metadata for global DOFs
        self._update_shape_metadata()
        
        # Rebind function handles to ensure they point to overridden methods
        self._rebind_function_handles()
        
        if self.print_progress:
            print(f'C-SBP initialized: {self.N_global} global DOFs (from {self.qshape[0]*self.qshape[1]} local DOFs)')
    
    
    def _build_global_id_mapping(self):
        """
        Build global-to-local index mapping at node level (without neq_node expansion).
        
        Creates gid array of shape (n, m) where gid[i, k] = global node index
        for local node i in element k. Node indices are 0 to N_nodes_global-1.
        DOF indices are computed on-the-fly in gather/scatter operations.
        
        For periodic boundaries, manually stitches boundary nodes using structured mesh layout.
        """
        tol = 1e-12
        
        if self.dim == 1:
            # For 1D, get coordinates from mesh.x_elem
            coords = self.mesh.x_elem  # shape (nen, nelem)
            
            # Build gid by grouping nodes at same physical location
            coords_rounded = np.round(coords / tol) * tol
            unique_coords, inverse = np.unique(coords_rounded.flatten(), return_inverse=True)
            self.Nn_global = len(unique_coords)
            self.N_global = self.Nn_global * self.neq_node
            
            # Reshape to node-level gid (nen, nelem) - no neq_node expansion
            self.gid = inverse.reshape((self.nen, self.nelem)).astype(int)
            
            # Manually stitch periodic boundaries using structured mesh
            if self.periodic:
                # Leftmost node of leftmost element (gid[0, 0]) should match
                # rightmost node of rightmost element (gid[-1, -1]) for periodic BC
                left_gid = self.gid[0, 0]  # Use first element's left node as reference
                right_gid = self.gid[-1, -1]  # Use last element's right node as reference
                
                # Use the minimum gid as the reference
                ref_gid = min(left_gid, right_gid)
                # Only match the boundary nodes (leftmost of first element, rightmost of last element)
                self.gid[0, 0] = ref_gid
                self.gid[-1, -1] = ref_gid
                
                # Recompute Nn_global after stitching
                self.Nn_global = len(np.unique(self.gid))
                self.N_global = self.Nn_global * self.neq_node
                
                # Remap gid to be contiguous 0 to Nn_global-1
                unique_gids, inverse_gid = np.unique(self.gid, return_inverse=True)
                self.gid = inverse_gid.reshape(self.gid.shape).astype(int)
                self.Nn_global = len(unique_gids)
                self.N_global = self.Nn_global * self.neq_node
            
        elif self.dim == 2:
            # For 2D, use mesh.xy_elem (shape: nen**2, 2, nelem[0]*nelem[1])
            n_elem = self.nelem[0] * self.nelem[1]
            coords = self.mesh.xy_elem  # shape (nen**2, 2, n_elem)
            
            # Round coordinates to avoid floating point issues
            coords_rounded = np.round(coords / tol) * tol
            
            # Flatten and find unique coordinates
            coords_flat = coords_rounded.reshape(-1, 2)
            unique_coords, inverse = np.unique(coords_flat, axis=0, return_inverse=True)
            self.Nn_global = len(unique_coords)
            self.N_global = self.Nn_global * self.neq_node
            
            # Reshape to node-level gid (nen**2, n_elem) - no neq_node expansion
            self.gid = inverse.reshape((self.nen**2, n_elem)).astype(int)
            
            # Manually stitch periodic boundaries using structured mesh
            # Nodes within each element are arranged as a 2D grid (nen x nen)
            # After meshgrid and reshape: nodes are ordered as rows (y varies first, then x)
            # So node index = i_y * nen + i_x, where i_y and i_x are local indices
            
            if self.periodic[0]:
                # Left face: nodes with i_x = 0 (indices 0, nen, 2*nen, ..., (nen-1)*nen)
                # Right face: nodes with i_x = nen-1 (indices nen-1, 2*nen-1, ..., nen**2-1)
                left_face_indices = np.arange(0, self.nen**2, self.nen)  # [0, nen, 2*nen, ...]
                right_face_indices = np.arange(self.nen-1, self.nen**2, self.nen)  # [nen-1, 2*nen-1, ...]
                
                # For each y-position (j), stitch left face of i=0 to right face of i=nelem[0]-1
                for j in range(self.nelem[1]):
                    e_left = self.nelem[1] * 0 + j  # Element at i=0, j=j
                    e_right = self.nelem[1] * (self.nelem[0]-1) + j  # Element at i=nelem[0]-1, j=j
                    
                    # Match each node on left face to corresponding node on right face
                    for node_idx in range(len(left_face_indices)):
                        left_gid = self.gid[left_face_indices[node_idx], e_left]
                        right_gid = self.gid[right_face_indices[node_idx], e_right]
                        # Use the minimum gid as the reference
                        ref_gid = min(left_gid, right_gid)
                        self.gid[left_face_indices[node_idx], e_left] = ref_gid
                        self.gid[right_face_indices[node_idx], e_right] = ref_gid
            
            if self.periodic[1]:
                # Bottom face: nodes with i_y = 0 (indices 0 to nen-1)
                # Top face: nodes with i_y = nen-1 (indices (nen-1)*nen to nen**2-1)
                bottom_face_indices = np.arange(self.nen)  # [0, 1, ..., nen-1]
                top_face_indices = np.arange((self.nen-1)*self.nen, self.nen**2)  # [(nen-1)*nen, ..., nen**2-1]
                
                # For each x-position (i), stitch bottom face of j=0 to top face of j=nelem[1]-1
                for i in range(self.nelem[0]):
                    e_bottom = self.nelem[1] * i + 0  # Element at i=i, j=0
                    e_top = self.nelem[1] * i + (self.nelem[1]-1)  # Element at i=i, j=nelem[1]-1
                    
                    # Match each node on bottom face to corresponding node on top face
                    for node_idx in range(len(bottom_face_indices)):
                        bottom_gid = self.gid[bottom_face_indices[node_idx], e_bottom]
                        top_gid = self.gid[top_face_indices[node_idx], e_top]
                        # Use the minimum gid as the reference
                        ref_gid = min(bottom_gid, top_gid)
                        self.gid[bottom_face_indices[node_idx], e_bottom] = ref_gid
                        self.gid[top_face_indices[node_idx], e_top] = ref_gid
            
            # Recompute Nn_global after stitching
            self.Nn_global = len(np.unique(self.gid))
            self.N_global = self.Nn_global * self.neq_node
            
            # Remap gid to be contiguous 0 to Nn_global-1
            unique_gids, inverse_gid = np.unique(self.gid, return_inverse=True)
            self.gid = inverse_gid.reshape(self.gid.shape).astype(int)
            self.Nn_global = len(unique_gids)
            self.N_global = self.Nn_global * self.neq_node
                    
        elif self.dim == 3:
            # For 3D, use mesh.xyz_elem (shape: nen**3, 3, nelem[0]*nelem[1]*nelem[2])
            n_elem = self.nelem[0] * self.nelem[1] * self.nelem[2]
            coords = self.mesh.xyz_elem  # shape (nen**3, 3, n_elem)
            
            # Round coordinates to avoid floating point issues
            coords_rounded = np.round(coords / tol) * tol
            
            # Flatten and find unique coordinates
            coords_flat = coords_rounded.reshape(-1, 3)
            unique_coords, inverse = np.unique(coords_flat, axis=0, return_inverse=True)
            self.Nn_global = len(unique_coords)
            self.N_global = self.Nn_global * self.neq_node
            
            # Reshape to node-level gid (nen**3, n_elem) - no neq_node expansion
            self.gid = inverse.reshape((self.nen**3, n_elem)).astype(int)
            
            # Manually stitch periodic boundaries using structured mesh
            # Nodes within each element are arranged as a 3D grid (nen x nen x nen)
            # After meshgrid and reshape: nodes are ordered with z varying fastest, then y, then x
            # So node index = i_z * nen**2 + i_y * nen + i_x
            
            if self.periodic[0]:
                # Left face: i_x = 0 (indices 0, nen, 2*nen, ..., (nen-1)*nen, and repeated for each z)
                # Right face: i_x = nen-1 (indices nen-1, 2*nen-1, ..., nen**2-1, and repeated for each z)
                left_face_base = np.arange(0, self.nen**2, self.nen)  # Base indices for z=0
                right_face_base = np.arange(self.nen-1, self.nen**2, self.nen)  # Base indices for z=0
                left_face_indices = []
                right_face_indices = []
                for iz in range(self.nen):
                    left_face_indices.extend(left_face_base + iz * self.nen**2)
                    right_face_indices.extend(right_face_base + iz * self.nen**2)
                left_face_indices = np.array(left_face_indices)
                right_face_indices = np.array(right_face_indices)
                
                # For each y and z position, stitch left face of i=0 to right face of i=nelem[0]-1
                for j in range(self.nelem[1]):
                    for k in range(self.nelem[2]):
                        e_left = self.nelem[1]*self.nelem[2]*0 + self.nelem[2]*j + k
                        e_right = self.nelem[1]*self.nelem[2]*(self.nelem[0]-1) + self.nelem[2]*j + k
                        
                        # Match each node on left face to corresponding node on right face
                        for node_idx in range(len(left_face_indices)):
                            left_gid = self.gid[left_face_indices[node_idx], e_left]
                            right_gid = self.gid[right_face_indices[node_idx], e_right]
                            ref_gid = min(left_gid, right_gid)
                            self.gid[left_face_indices[node_idx], e_left] = ref_gid
                            self.gid[right_face_indices[node_idx], e_right] = ref_gid
            
            if self.periodic[1]:
                # Bottom face: i_y = 0 (indices 0 to nen-1, and repeated for each z)
                # Top face: i_y = nen-1 (indices (nen-1)*nen to nen**2-1, and repeated for each z)
                bottom_face_base = np.arange(self.nen)  # Base indices for z=0
                top_face_base = np.arange((self.nen-1)*self.nen, self.nen**2)  # Base indices for z=0
                bottom_face_indices = []
                top_face_indices = []
                for iz in range(self.nen):
                    bottom_face_indices.extend(bottom_face_base + iz * self.nen**2)
                    top_face_indices.extend(top_face_base + iz * self.nen**2)
                bottom_face_indices = np.array(bottom_face_indices)
                top_face_indices = np.array(top_face_indices)
                
                # For each x and z position, stitch bottom face of j=0 to top face of j=nelem[1]-1
                for i in range(self.nelem[0]):
                    for k in range(self.nelem[2]):
                        e_bottom = self.nelem[1]*self.nelem[2]*i + self.nelem[2]*0 + k
                        e_top = self.nelem[1]*self.nelem[2]*i + self.nelem[2]*(self.nelem[1]-1) + k
                        
                        # Match each node on bottom face to corresponding node on top face
                        for node_idx in range(len(bottom_face_indices)):
                            bottom_gid = self.gid[bottom_face_indices[node_idx], e_bottom]
                            top_gid = self.gid[top_face_indices[node_idx], e_top]
                            ref_gid = min(bottom_gid, top_gid)
                            self.gid[bottom_face_indices[node_idx], e_bottom] = ref_gid
                            self.gid[top_face_indices[node_idx], e_top] = ref_gid
            
            if self.periodic[2]:
                # Front face: i_z = 0 (indices 0 to nen**2-1)
                # Back face: i_z = nen-1 (indices (nen-1)*nen**2 to nen**3-1)
                front_face_indices = np.arange(self.nen**2)  # [0, 1, ..., nen**2-1]
                back_face_indices = np.arange((self.nen-1)*self.nen**2, self.nen**3)  # [(nen-1)*nen**2, ..., nen**3-1]
                
                # For each x and y position, stitch front face of k=0 to back face of k=nelem[2]-1
                for i in range(self.nelem[0]):
                    for j in range(self.nelem[1]):
                        e_front = self.nelem[1]*self.nelem[2]*i + self.nelem[2]*j + 0
                        e_back = self.nelem[1]*self.nelem[2]*i + self.nelem[2]*j + (self.nelem[2]-1)
                        
                        # Match each node on front face to corresponding node on back face
                        for node_idx in range(len(front_face_indices)):
                            front_gid = self.gid[front_face_indices[node_idx], e_front]
                            back_gid = self.gid[back_face_indices[node_idx], e_back]
                            ref_gid = min(front_gid, back_gid)
                            self.gid[front_face_indices[node_idx], e_front] = ref_gid
                            self.gid[back_face_indices[node_idx], e_back] = ref_gid
            
            # Recompute Nn_global after stitching
            self.Nn_global = len(np.unique(self.gid))
            self.N_global = self.Nn_global * self.neq_node
            
            # Remap gid to be contiguous 0 to Nn_global-1
            unique_gids, inverse_gid = np.unique(self.gid, return_inverse=True)
            self.gid = inverse_gid.reshape(self.gid.shape).astype(int)
            self.Nn_global = len(unique_gids)
            self.N_global = self.Nn_global * self.neq_node
        
        # Build DOF-level gid for efficient gather/scatter
        self._build_dof_gid()
    
    def _build_dof_gid(self):
        """
        Build DOF-level gid array from node-level gid.
        
        If neq_node == 1, gid_neq just references gid (same array).
        Otherwise, expands gid to DOF level: gid_neq[i*neq_node + eq, k] = gid[i, k] * neq_node + eq
        """
        if self.neq_node == 1:
            # For single equation, node-level and DOF-level are the same
            self.gid_neq = self.gid
        else:
            # Expand node-level gid to DOF-level using broadcasting
            nen_nodes, nelem = self.gid.shape
            eq_indices = np.arange(self.neq_node)[:, None, None]  # (neq_node, 1, 1)
            node_indices = self.gid[None, :, :]  # (1, nen_nodes, nelem)
            
            # DOF index = node_index * neq_node + eq_index
            dof_gid = (node_indices * self.neq_node + eq_indices)  # (neq_node, nen_nodes, nelem)
            
            # Reshape to (nen_nodes*neq_node, nelem) with correct ordering
            # Transpose to get (nen_nodes, neq_node, nelem), then reshape
            self.gid_neq = dof_gid.transpose(1, 0, 2).reshape(nen_nodes * self.neq_node, nelem).astype(int)
    
    def _build_H_weights(self):
        """
        Build local and global H weights.
        
        H_loc is node-level (shape matches gid and H_phys).
        H_glob is also node-level (shape: N_nodes_global).
        DOF-level expansion is done on-the-fly when needed.
        """
        # H_phys is already computed in base class as (n, m) for element-local nodes
        self.H_loc = self.H_phys  # Reference, no copy - shape (nen, nelem) etc.
        
        # Build global H at node level by scatter-add of local H
        # gid shape: (nen, nelem), H_loc shape: (nen, nelem) - perfect match!
        self.H_glob = np.zeros(self.Nn_global, dtype=self.H_loc.dtype)
        np.add.at(self.H_glob, self.gid.ravel(), self.H_loc.ravel())
        
        # Avoid division by zero (shouldn't happen, but safety check)
        assert np.all(self.H_glob > 1e-15), \
            "C-SBP error: detected global H matrix with zero or negative entries"

        self.H_glob_inv = 1.0 / self.H_glob
        self.volume = np.sum(self.H_glob)
    
    def _replace_sats_with_null(self):
        """Replace SAT objects with NullSAT to disable internal coupling"""
        # NullSAT needs shape matching base class qshape (includes neq_node)
        # Use base class qshape which is (nen*neq_node, nelem) etc.
        local_shape = self.qshape  # Shape from base class before override
        
        if self.dim == 1:
            self.sat = NullSAT(local_shape)
        elif self.dim == 2:
            self.satx = NullSAT(local_shape)
            self.saty = NullSAT(local_shape)
        elif self.dim == 3:
            self.satx = NullSAT(local_shape)
            self.saty = NullSAT(local_shape)
            self.satz = NullSAT(local_shape)
    
    def _update_shape_metadata(self):
        """Update shape metadata to reflect global DOF storage"""
        self.qshape_global = (self.N_global, 1)
        self.qshape_local = self.qshape
        self.qshape = self.qshape_global
        
        # Update self.nn to reflect unique nodes per direction after conformity
        # Scale Nn_global**(1/dim) by the number of elements in each direction
        if self.dim == 1:
            # For 1D, nn is just the total number of unique nodes
            self.nn = self.Nn_global
        elif self.dim == 2:
            # For 2D: nx * ny ≈ Nn_global, with nx/ny proportional to nelem[0]/nelem[1]
            base = self.Nn_global ** (1/2)
            # Scale by element ratios: nx/ny = nelem[0]/nelem[1]
            # So nx = base * sqrt(nelem[0] / nelem[1]), ny = base * sqrt(nelem[1] / nelem[0])
            nx = round(base * np.sqrt(self.nelem[0] / self.nelem[1]))
            ny = round(base * np.sqrt(self.nelem[1] / self.nelem[0]))
            # Ensure product matches Nn_global (adjust ny to match)
            if nx > 0:
                ny = round(self.Nn_global / nx)
            self.nn = (nx, ny)
        elif self.dim == 3:
            # For 3D: nx * ny * nz ≈ Nn_global, with nx:ny:nz = nelem[0]:nelem[1]:nelem[2]
            # If nx = k*nelem[0], ny = k*nelem[1], nz = k*nelem[2], then:
            # k^3 * nelem[0]*nelem[1]*nelem[2] = Nn_global
            # k = (Nn_global / (nelem[0]*nelem[1]*nelem[2]))^(1/3)
            nelem_prod = self.nelem[0] * self.nelem[1] * self.nelem[2]
            k = (self.Nn_global / nelem_prod) ** (1/3)
            nx = round(k * self.nelem[0])
            ny = round(k * self.nelem[1])
            nz = round(k * self.nelem[2])
            # Ensure product matches Nn_global (adjust nz to match)
            if nx > 0 and ny > 0:
                nz = round(self.Nn_global / (nx * ny))
            self.nn = (nx, ny, nz)
        
        # Override mesh coordinates to use global (unique) nodes only
        # Use gid to map from local nodes to global nodes, picking first occurrence
        if self.dim == 1:
            # 1D: x_elem is (nen, nelem), gid is (nen, nelem)
            x_global = np.zeros(self.Nn_global)
            # Track which global nodes we've already assigned
            assigned = np.zeros(self.Nn_global, dtype=bool)
            for e in range(self.nelem):
                for i in range(self.nen):
                    g = self.gid[i, e]
                    if not assigned[g]:
                        x_global[g] = self.mesh.x_elem[i, e]
                        assigned[g] = True
            self.mesh.x = x_global
            self.mesh.nn = self.nn
        elif self.dim == 2:
            # 2D: xy_elem is (nen**2, 2, nelem[0]*nelem[1]), gid is (nen**2, nelem[0]*nelem[1])
            xy_global = np.zeros((self.Nn_global, 2))
            # Track which global nodes we've already assigned
            assigned = np.zeros(self.Nn_global, dtype=bool)
            for e in range(self.nelem[0] * self.nelem[1]):
                for i in range(self.nen**2):
                    g = self.gid[i, e]
                    if not assigned[g]:
                        xy_global[g, :] = self.mesh.xy_elem[i, :, e]
                        assigned[g] = True
            self.mesh.xy = xy_global
        elif self.dim == 3:
            # 3D: xyz_elem is (nen**3, 3, nelem[0]*nelem[1]*nelem[2]), gid is (nen**3, nelem[0]*nelem[1]*nelem[2])
            xyz_global = np.zeros((self.Nn_global, 3))
            # Track which global nodes we've already assigned
            assigned = np.zeros(self.Nn_global, dtype=bool)
            nelem_total = self.nelem[0] * self.nelem[1] * self.nelem[2]
            for e in range(nelem_total):
                for i in range(self.nen**3):
                    g = self.gid[i, e]
                    if not assigned[g]:
                        xyz_global[g, :] = self.mesh.xyz_elem[i, :, e]
                        assigned[g] = True
            self.mesh.xyz = xyz_global
    
    def _rebind_function_handles(self):
        """
        Rebind function handles to ensure they point to overridden methods.
        
        The base class PdeSolverSbp.init_disc_specific() sets self.dqdt to a specific
        method (e.g., self.dqdt_1d_div), which overwrites our overridden dqdt method.
        We need to rebind it to our overridden method.
        """
        # Store reference to base class dqdt before rebinding (base class sets it to dqdt_1d_div etc.)
        self._dqdt_base = self.dqdt
        
        # Rebind dqdt to the overridden method (base class sets it to dqdt_1d_div etc.)
        # Get the overridden method from the class, not the instance
        import types
        self.dqdt = types.MethodType(self.__class__.dqdt, self)
        
        # Rebind other function handles
        self.energy = self.sbp_energy
        self.kinetic_energy = self.sbp_kinetic_energy
        self.conservation = self.sbp_conservation
        self.energy_der = self.sbp_energy_der
        self.conservation_der = self.sbp_conservation_der
        self.entropy = self.sbp_entropy
        self.entropy_der = self.sbp_entropy_der
        self.enstrophy = self.sbp_enstrophy
    
    def gather(self, q_global):
        """
        Gather global DOF vector to local element representation.
        
        Uses precomputed DOF-level gid for efficiency.
        
        Parameters
        ----------
        q_global : ndarray, shape (N_global, 1) or (N_global,)
            Global DOF vector (includes all equations)
        
        Returns
        -------
        q_loc : ndarray, shape (nen*neq_node, nelem)
            Local element representation with all equations
        """
        q_flat = np.asarray(q_global).reshape(-1)  # 1D view
        assert q_flat.size == self.N_global, (
            f"Expected q_global of size {self.N_global}, got {q_flat.size}"
        )
        
        # Use precomputed DOF-level gid
        return q_flat[self.gid_neq]
    
    def scatter_add(self, arr_loc):
        """
        Scatter-add local DOF-level array to global DOF array.
        
        Uses precomputed DOF-level gid for efficiency.
        
        Parameters
        ----------
        arr_loc : ndarray, shape (nen*neq_node, nelem)
            Local element array at DOF level
        
        Returns
        -------
        arr_global : ndarray, shape (N_global,) Note: NOT (N_global,1)
            Global array after scatter-add at DOF level
        """
        # Expected shape: (nen*neq_node, nelem) - DOF-level
        assert arr_loc.shape == self.gid_neq.shape, (
            f"arr_loc shape {arr_loc.shape} must match gid_neq shape {self.gid_neq.shape}"
        )
        
        # Scatter-add using precomputed DOF-level indices
        arr_global = np.zeros((self.N_global,), dtype=arr_loc.dtype)
        np.add.at(arr_global, self.gid_neq.ravel(), arr_loc.ravel())
        
        return arr_global
    
    def assemble_global(self, q_loc):
        """
        Assemble from local to global using H-weighted assembly.
        
        q_global[g] = (1 / H_glob[g]) * sum_{(i,k)->g} H_loc[i,k] * q_loc[i,k]
        
        H_loc is node-level, but q_loc is DOF-level. We broadcast H_loc
        to match q_loc shape, then scatter-add, then multiply by H_glob_inv expanded to DOF-level.
        
        Parameters
        ----------
        q_loc : ndarray, shape (nen*neq_node, nelem)
            Local strong-form RHS at DOF level
        
        Returns
        -------
        q_global : ndarray, shape (N_global, 1)
            Global strong-form RHS at DOF level
        """
        # H_loc has shape (nen, nelem), q_loc has shape (nen*neq_node, nelem)
        # Broadcast H_loc to match: expand along neq_node dimension
        H_q_loc = gdiag_gv(self.H_loc, q_loc, neq_node=self.neq_node)
        
        # Scatter-add H_loc * q_loc (at DOF level now)
        numerator = self.scatter_add(H_q_loc)  # (N_global,)
        
        # Divide by H_glob (at DOF level)
        q_global = ldiag_lv(self.H_glob_inv, numerator, neq_node=self.neq_node)  # (N_global,)
        
        return q_global[:, np.newaxis]

    def _ensure_global(self, q):
        """
        Ensure q is in global form. If local, convert to global using H-weighted assembly.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
        
        Returns
        -------
        q_global : ndarray, shape (N_global, 1)
            Global DOF vector (preserves 2D shape for compatibility with diffeq functions)
        """
        q = np.asarray(q)
        
        # Check if it's local by comparing shape
        if q.shape == self.qshape_local:
            # Local: convert to global using H-weighted assembly
            return self.assemble_global(q) # shape (N_global, 1)
        else:
            # Global: ensure it's (N_global, 1) shape
            assert q.size == self.N_global, (
                f"Expected q of size {self.N_global}, got {q.size}"
            )
            q_reshaped = q.reshape(self.N_global, 1)
            return q_reshaped
    
    def _ensure_local(self, q):
        """
        Ensure q is in local form. If global, convert to local using gather.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
        
        Returns
        -------
        q_local : ndarray, shape (nen*neq_node, nelem)
            Local element representation
        """
        q = np.asarray(q)
        
        # Check if it's local by comparing shape
        if q.shape == self.qshape_local:
            # Already local: return as-is
            return q
        else:
            # Global: convert to local using gather
            assert q.size == self.N_global, (
                f"Expected q of size {self.N_global}, got {q.size}"
            )
            return self.gather(q)
    
    def dqdt(self, q, t=0.0):
        """
        dq/dt wrapper that handles both local and global input.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
        t : float, optional
            Time (for time-dependent problems)
        
        Returns
        -------
        dqdt_global : ndarray, shape (N_global, 1)
            Global time derivative
        """
        # Ensure q is local
        q_loc = self._ensure_local(q)
        
        # Call base class dqdt (which will now have NullSATs)
        # Use stored reference since base class sets dqdt as an attribute, not a method
        dqdt_loc = self._dqdt_base(q_loc, t)
        
        # Assemble to global (returns shape (N_global, 1))
        dqdt_global = self.assemble_global(dqdt_loc)
        
        # Return in shape (N_global, 1) (assemble_global always returns (N_global, 1))
        return dqdt_global
    
    def solve(self, q0=None):
        """
        Override solve() to ensure q0 is in global shape before passing to time marching.
        
        The base class solve() gets q0 from diffeq.set_q0() which returns local shape.
        For C-SBP, we need to convert it to global shape before passing to TimeMarching.
        """
        # Get q0 (will be in local shape from base class)
        if q0 is None:
            q0 = self.diffeq.set_q0()
        
        # Convert q0 to global shape if it's local
        q0_global = self._ensure_global(q0)
        
        # Call base class solve with global q0
        # We need to temporarily override q0 to be global, then call super
        # But actually, we should just replicate the base class logic with our global q0
        if self.dt_to_be_set:
            raise Exception('Time step not set yet. Use set_timestep(dt) before running solve().')
        
        if self.t_initial != 0.0:
            self.set_timestep(self.dt)
        
        from Source.TimeMarch.TimeMarching import TimeMarching
        tm_class = TimeMarching(self.diffeq, self.tm_method, self.keep_all_ts,
                        skip_ts = self.skip_ts,
                        bool_plot_sol = self.bool_plot_sol,
                        bool_calc_cons_obj = self.bool_calc_cons_obj,
                        print_sol_norm = self.print_sol_norm,
                        print_residual = self.print_residual,
                        check_resid_conv = self.check_resid_conv,
                        dqdt=self.dqdt, dfdq=self.dfdq,
                        rtol=self.tm_rtol, atol=self.tm_atol)
        
        tm_class.nframes = self.tm_nframes
        tm_class.print_progress = self.print_progress
        
        # Pass global q0 to time marching
        # q_sol will be returned in global shape (N_global, 1) or (N_global, n_ts)
        self.q_sol = tm_class.solve(q0_global, self.dt, self.n_ts, self.t_initial)
        self.cons_obj = tm_class.cons_obj
        self.t_final = tm_class.t_final
    
    def plot_sol(self, q=None, interpolate=True, **kwargs):
        """
        Override plot_sol() to ensure q is in local shape before calling base method.
        
        The base class plot_sol() expects q in local shape (nen*neq_node, nelem).
        For C-SBP, q_sol is stored in global shape, so we need to convert it to local.
        """
        # Get q if not provided (will be in global shape from q_sol)
        if q is None:
            if self.q_sol is None:
                q = self.diffeq.set_q0()  # This returns local shape
                if 'time' not in kwargs: kwargs['time'] = 0.0
            else:
                if self.q_sol.ndim == 2:
                    q = self.q_sol  # Global shape (N_global, 1)
                elif self.q_sol.ndim == 3:
                    q = self.q_sol[:, :, -1]  # Global shape (N_global, 1)
                if 'time' not in kwargs: kwargs['time'] = self.t_final
        
        # Convert q to local shape for base class plotting
        q_local = self._ensure_local(q)
        
        # Call base class plot_sol with local q
        super().plot_sol(q_local, interpolate=interpolate, **kwargs)

    def calc_error(self, q=None, tf=None, method=None, use_all_t=False,
                   var2plot_name=None):
        """
        Override calc_error() to ensure q is in the same shape (local/global) as the input q.
        
        The base class calc_error() otherwise calculates q_exa in local shape (nen*neq_node, nelem).
        """
        if (q is None) or (q is not None and q.shape == self.qshape_global):
            # will pull from self.q_sol, i.e. global shape, so we need q_exa in global shape
            if tf == None: tf = self.t_final
            if q is None:
                if self.q_sol.ndim == 2: q = self.q_sol
                elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
            
            # Convert q to local format for exact_sol (which expects local format)
            q_local = self._ensure_local(q)
            
            # Call exact_sol with local q, then convert result to global
            q_exa = self._ensure_global(self.diffeq.exact_sol(tf, guess=q_local))

            # Call base class calc_error with global q and q_exa
            return super().calc_error(q=q, tf=tf, method=method, use_all_t=use_all_t,
                   var2plot_name=var2plot_name, q_exa=q_exa)

        elif q is not None and q.shape == self.qshape_local:
            # fine to call base class calc_error with local q, but this may cause errors
            print('WARNING: calc_error called with local q. This may cause errors.')
            return super().calc_error(q=q, tf=tf, method=method, use_all_t=use_all_t,
                   var2plot_name=var2plot_name)
        else:
            raise ValueError(f'ERROR: calc_error called with invalid q shape. Expected {self.qshape_global}, got {q.shape if q is not None else None}')

    def calc_RHS_jac(self, q=None, t=0., exact_dfdq=True, step=1.0e-4, istep=1.0e-15, 
                 finite_diff=False, print_nothing=False, print_error=False):
        """
        Override calc_RHS_jac() to ensure q is global.
        We don't want to let the base class calc_RHS_jac() set q to a local shape.
        """
        if q is None:
            if hasattr(self, 'q_sol'):
                if self.q_sol is not None:
                    if self.q_sol.ndim == 2: q = self.q_sol
                    elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
                else:
                    q = self._ensure_global(self.diffeq.set_q0())
            else:
                q = self._ensure_global(self.diffeq.set_q0())
        else:
            q = self._ensure_global(q)
            assert q.shape == self.qshape_global, f'ERROR: q must be in global shape. Expected {self.qshape_global}, got {q.shape}'

        return super().calc_RHS_jac(q=q, t=t, exact_dfdq=exact_dfdq, step=step, istep=istep, 
                   finite_diff=finite_diff, print_nothing=print_nothing, print_error=print_error)

    
    def sbp_energy(self, q, neq=None):
        """
        Compute global SBP energy using H_glob and q.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
        neq : int, optional
            Number of equations per node (for compatibility with base class)
        
        Returns
        -------
        energy : float
            Global energy
        """
        #TODO: add support for neq != self.neq_node
        # Ensure q is global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        q_flat = q_global.flatten()
        
        # Energy = sum(H_glob * q_global^2)
        # H_glob is node-level, q_flat is DOF-level - use element-wise multiplication
        energy = np.sum(ldiag_lv(self.H_glob, q_flat**2, neq_node=self.neq_node))
        
        return energy
    
    def sbp_conservation(self, q):
        """
        Compute global conservation using H_glob.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
        
        Returns
        -------
        cons : float
            Global conservation (scalar)
        """
        # Ensure q is global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        q_flat = q_global.flatten()
        
        cons = np.sum(ldiag_lv(self.H_glob, q_flat, neq_node=self.neq_node))
        
        return cons
    
    def sbp_conservation_der(self, dqdt):
        """
        Compute derivative of global conservation.
        
        Parameters
        ----------
        dqdt : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global time derivative
        
        Returns
        -------
        cons : float
            Derivative of global conservation
        """
        return self.sbp_conservation(dqdt)
    
    def sbp_energy_der(self, q, dqdt):
        """
        Compute derivative of global energy.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global solution vector at DOF level
        dqdt : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global time derivative at DOF level
        
        Returns
        -------
        energy_der : float
            Derivative of global energy
        """
        # Ensure both are global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        dqdt_global = self._ensure_global(dqdt)
        q_flat = q_global.flatten()
        dqdt_flat = dqdt_global.flatten()
        
        # d/dt(energy) = sum(H_glob * q_global * dqdt_global)
        energy_der = np.sum(ldiag_lv(self.H_glob, q_flat*dqdt_flat, neq_node=self.neq_node))
        
        return energy_der
    
    def sbp_entropy(self, q):
        """
        Compute global entropy using H_glob.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global solution vector at DOF level
        
        Returns
        -------
        entropy : float
            Global entropy
        """
        # Ensure q is global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        
        # diffeq.entropy expects (N_global, 1) shape
        s_flat = np.asarray(self.diffeq.entropy(q_global)).reshape(-1)
        
        # Sum over all DOFs (recall s is a scalar at each node)
        ent = np.sum(self.H_glob * s_flat)
        
        return ent
    
    def sbp_entropy_der(self, q, dqdt):
        """
        Compute derivative of global entropy.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global solution vector at DOF level
        dqdt : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global time derivative at DOF level
        
        Returns
        -------
        entropy_der : float
            Derivative of global entropy
        """
        # Ensure both are global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        dqdt_global = self._ensure_global(dqdt)
        
        # diffeq.entropy_var expects (N_global, 1) shape
        w_flat = np.asarray(self.diffeq.entropy_var(q_global)).reshape(-1)
        dqdt_flat = dqdt_global.flatten()
        
        # d/dt(entropy) = sum(H_glob * w_global * dqdt_global)
        entropy_der = np.sum(ldiag_lv(self.H_glob, w_flat*dqdt_flat, neq_node=self.neq_node))
        
        return entropy_der
    
    def sbp_kinetic_energy(self, q):
        """
        Compute global kinetic energy.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global solution vector at DOF level
        
        Returns
        -------
        kinetic_energy : float
            Global kinetic energy
        """
        # Ensure q is global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        
        # diffeq.kinetic_energy expects (N_global, 1) shape
        k_flat = np.asarray(self.diffeq.kinetic_energy(q_global)).reshape(-1)
        
        # Integrate, then normalize by volume
        energy = np.sum(ldiag_lv(self.H_glob, k_flat, neq_node=self.neq_node)) / self.volume
        
        return energy
    
    def sbp_enstrophy(self, q):
        """
        Compute global enstrophy.
        
        Parameters
        ----------
        q : ndarray
            Either local shape (nen*neq_node, nelem) or global shape (N_global, 1) or (N_global,)
            Global solution vector
        
        Returns
        -------
        enstrophy : float
            Global enstrophy
        """
        # Ensure q is global (returns (N_global, 1))
        q_global = self._ensure_global(q)
        
        # diffeq.enstropy expects (N_global, 1) shape
        s_flat = np.asarray(self.diffeq.enstropy(q_global)).reshape(-1)
        
        # Integrate
        ent = np.sum(ldiag_lv(self.H_glob, s_flat, neq_node=self.neq_node))
        
        return ent
