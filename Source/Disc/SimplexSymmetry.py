
# Add the root folder of ECO to the search path
import os
import sys

test_folder_path, _ = os.path.split(__file__)
root_folder_path, _ = os.path.split(test_folder_path)
sys.path.append(root_folder_path)

# Import the required modules
import numpy as np

from Source.Disc.RefSimplexElem import RefSimplexElem


class SimplexSymmetry:
    '''
        Purpose
        ----------
        This class does the following:
            1) identifies all of the symmetry groups for the nodes
            2) Creates permutation matrices for all the sym lines or planes
            3) Creates perm matrices for directional operators, eg: D_x -> D_y
            4) Indentifies which nodes are on facet 0 (needed for sbp_fam Rdn1)
    '''

    tol = 1e-8

    def __init__(self, xy):
        '''
        Parameters
        ----------
        xy : numpy array
            Each row has the cartesian nodal locations for one node.

        Returns
        -------
        None
        '''

        ''' Add inputs to the class '''

        self.xy = xy
         # TODO add vert as input and convert to reference simplex elem

        ''' Extract required parameters '''

        self.nn, self.dim = self.xy.shape

        if self.dim == 1 and self.nn == 1 and self.xy[0] == 0:
            self.dim = 0

        self.n_facets = self.dim + 1

        ''' Calculate all requried data '''

        # Identify symmetry groups
        self.xy_bary = self.barycentric(self.xy)
        self.sym_g_mat, self.sym_g_id = self.identify_sym_groups()

        # Create permutation for the directional operators
        self.perm_idx_all = self.all_perm()
        self.perm_idx_dir = self.dir_perm()

        # Create permutation matrices
        self.perm_mat_all = self.make_perm_matrix(self.perm_idx_all)
        self.perm_mat_dir = self.make_perm_matrix(self.perm_idx_dir)
        self.perm_mat_r = self.make_perm_mat_r()

        self.node_on_facet0 = self.find_nodes_on_facet0()

    @staticmethod
    def barycentric(xy, vert=None):
        '''
        Parameters
        ----------
        xy : numpy array
            Each row has the cartesian nodal locations for one node.
        vert : numpy array, optional
            Nodal locations for the vertices of the reference simplex element.
            The default is None.

        Returns
        -------
        xy_bary : numpy array
            Barycentric coordinates of the nodes.
        '''

        nn, dim = xy.shape
        nvert = dim + 1

        if vert is None:
            vert = RefSimplexElem.ref_vertices(dim)

        vert2 = np.hstack((vert, np.ones((nvert, 1))))
        xy2 = np.hstack((xy, np.ones((nn, 1))))

        xy_bary = np.linalg.solve(vert2.T, xy2.T).T

        return xy_bary

    def identify_sym_groups(self):
        '''
        Returns
        -------
        sym_g_mat : numpy array of int
            Each row holds the number of all the nodes in a symmetry group.
            The symmetry groups with a lower ID number are listed first.
            The nodes in each symmetry group are organized to be ordered in a
            particular way, as detailed for each element type.
        sym_g_id : numpy array of int
            One entry per symmetry group. Each entry is an integer, which
            indicates the type of symmetry group.
        '''

        xy_bary = self.xy_bary
        sort_bary = np.sort(xy_bary, axis=1)

        # Indicates if the symmetry group of a node has been identified
        node_classified = np.full(self.nn, False)

        list_of_groups = [None] * self.nn
        n_node_in_group = np.zeros(self.nn) # No. of nodes in each symmetry group

        ''' Identify the nodes that are in the same groups '''
        n_groups = 0
        for i in range(self.nn):

            if node_classified[i]:
                continue

            same_group = np.max(np.abs(sort_bary - sort_bary[i, :]), axis=1) < self.tol
            node_classified[same_group] = True

            idx_nodes = np.nonzero(same_group)[0]
            list_of_groups[n_groups] = idx_nodes
            n_node_in_group[n_groups] = len(idx_nodes)
            n_groups += 1

        if np.min(node_classified) == 0:
            raise Exception('At least one node was not classified in a sym group')

        ''' Organize the nodes that are in the same groups '''
        # Each row in sym_g_mat holds all of the nodes in one sym group
        # The symmetry groups with the fewest nodes are considered first

        # Give an id number to each sym group:
        #   0: centroid             1: vertex-centered  2: edged-centered
        #   3: mid-edge-centered    4: face-centered
        sym_g_id = np.full(n_groups, -1, dtype=int)

        cnt = -1
        if self.dim == 0:
            sym_g_mat = np.zeros((1,1), dtype=int)

        elif self.dim == 1:
            # The nodes in each group are given from 0 to 1
            max_n_per_sym = 2
            sym_g_mat = np.full((n_groups, max_n_per_sym), -1, dtype=int)

            for i in range(n_groups):

                cnt += 1
                node_vec = list_of_groups[i]

                if len(node_vec) == 1:
                    sym_g_id[cnt] = 0
                    sym_g_mat[cnt, 0] = node_vec[0]
                elif len(node_vec) == 2:
                    sym_g_id[cnt] = 1
                    if self.xy[node_vec[0]] > self.xy[node_vec[1]]:
                        sym_g_mat[cnt, :] = node_vec
                    else:
                        sym_g_mat[cnt, :] = np.flip(node_vec)

        elif self.dim == 2:
            # The nodes in each group are given counterclockwise
            max_n_per_sym = 6
            sym_g_mat = np.full((n_groups, max_n_per_sym), -1, dtype=int)

            for i in range(n_groups):

                cnt += 1
                node_vec = list_of_groups[i]

                if len(node_vec) == 1:
                    sym_g_id[cnt] = 0
                    sym_g_mat[cnt, 0] = node_vec[0]
                elif len(node_vec) == 3:
                    sym_g_id[cnt] = 1
                    for i in range(3):
                        idx_i = node_vec[i]
                        if np.abs(xy_bary[idx_i, 1] - xy_bary[idx_i, 2]) < self.tol:
                            sym_g_mat[cnt, 0] = idx_i
                        elif np.abs(xy_bary[idx_i, 0] - xy_bary[idx_i, 2]) < self.tol:
                            sym_g_mat[cnt, 1] = idx_i
                        elif np.abs(xy_bary[idx_i, 0] - xy_bary[idx_i, 1]) < self.tol:
                            sym_g_mat[cnt, 2] = idx_i
                        else:
                            raise Exception('Error in classifying S21 group')
                elif len(node_vec) == 6:
                    sym_g_id[cnt] = 2
                    for i in range(6):
                        if xy_bary[idx_i, 1] > xy_bary[idx_i, 2]: # Check sym line 0
                            if xy_bary[idx_i, 0] > xy_bary[idx_i, 1]: # Check sym line 2
                                sym_g_mat[cnt, 0] = idx_i
                            else:
                                if xy_bary[idx_i, 0] > xy_bary[idx_i, 2]: # Check sym line 1
                                    sym_g_mat[cnt, 1] = idx_i
                                else:
                                    sym_g_mat[cnt, 2] = idx_i
                        else:
                            if xy_bary[idx_i, 1] > xy_bary[idx_i, 0]: # Check sym line 2
                                sym_g_mat[cnt, 3] = idx_i
                            else:
                                if xy_bary[idx_i, 2] > xy_bary[idx_i, 0]: # Check sym line 1
                                    sym_g_mat[cnt, 4] = idx_i
                                else:
                                    sym_g_mat[cnt, 5] = idx_i
                else:
                    raise Exception('Error identifying 2D symmetry groups')

        elif self.dim == 3:
            # The nodes in each group are given from 0 to 1
            max_n_per_sym = 24
            sym_g_mat = np.full((n_groups, max_n_per_sym), -1, dtype=int)

            for i in range(n_groups):
                cnt += 1
                node_vec = list_of_groups[i]

                if len(node_vec) == 1:
                    sym_g_id[cnt] = 0
                    sym_g_mat[cnt, 0] = node_vec[0]
                elif len(node_vec) == 4:
                    sym_g_id[cnt] = 1

                    # Determine the value of the one FP for this sym group
                    idx_node = node_vec[0]           # Used the first node
                    bary_test = xy_bary[idx_node, 0] # Does not matter which index is used
                    lam_eq = np.abs(xy_bary[idx_node,:] - bary_test) < self.tol

                    if np.sum(lam_eq) == 1:
                        idx_alpha = np.argmax(lam_eq)
                    elif np.sum(lam_eq) == 3:
                        idx_alpha = np.argmin(lam_eq)

                    alpha = xy_bary[idx_node, idx_alpha]

                    for i in range(4):
                        idx_i = node_vec[i]
                        is_alpha = np.abs(xy_bary[idx_node, :] - alpha) < self.tol
                        idx_alpha = np.argmax(is_alpha)

                        sym_g_mat[cnt, idx_alpha] = idx_i


                else:
                    raise Exception('This sym group has not yet been considered')

        else:
            raise Exception('Only 1, 2 and 3D are currently available')

        return sym_g_mat, sym_g_id

    def all_perm(self):
        '''
        Returns
        -------
        all_perm : numpy array of int
            Each row holds the permutation of the nodes for all of the symmetry
            lines or planes (3D)
        '''

        if self.dim == 0:
            all_perm = np.zeros((1,1), dtype=int)
        elif self.dim == 1:
            all_perm = np.zeros((1, self.nn), dtype=int)
            all_perm[0,:] = self.make_perm_idx(1-self.xy)
        elif self.dim == 2:
            all_perm = np.zeros((3, self.nn), dtype=int)
            sum_xy = np.sum(self.xy, axis=1)

            all_perm[0, :] = self.make_perm_idx(self.xy[:,1], self.xy[:,0])
            all_perm[1, :] = self.make_perm_idx(self.xy[:,0], 1-sum_xy)
            all_perm[2, :] = self.make_perm_idx(1-sum_xy, self.xy[:,1])
        elif self.dim == 3:
            all_perm = np.zeros((6, self.nn), dtype=int)
            sum_xy = np.sum(self.xy, axis=1)

            all_perm[0, :] = self.make_perm_idx(self.xy[:,0], self.xy[:,2], self.xy[:,1])
            all_perm[1, :] = self.make_perm_idx(self.xy[:,2], self.xy[:,1], self.xy[:,0])
            all_perm[2, :] = self.make_perm_idx(self.xy[:,1], self.xy[:,0], self.xy[:,2])

            all_perm[3, :] = self.make_perm_idx(self.xy[:,0], self.xy[:,1], 1-sum_xy)
            all_perm[4, :] = self.make_perm_idx(self.xy[:,0], 1-sum_xy, self.xy[:,2])
            all_perm[5, :] = self.make_perm_idx(1-sum_xy, self.xy[:,1], self.xy[:, 2])
        else:
            raise Exception('Only 1D and 2D available for all_perm')

        return all_perm

    def make_perm_idx(self, x_equal, y_equal=None, z_equal=None):
        '''
        Purpose
        ----------
        Finds the pair of nodes across symmetry lines or planes. The pair of
        nodes must satisfy self.xy[:,0] == x_equal and analogously for 2 and 3D.
        Refer to Appendix B of André Marchildon's thesis on SBP operators for
        more information.

        Parameters
        ----------
        x_equal : numpy array
            x-coordinates that the each node pair must have
        y_equal : numpy array, optional
            y-coordinates that the each node pair must have.
            The default is None.
        z_equal : numpy array, optional
            x-coordinates that the each node pair must have.
            The default is None.

        Returns
        -------
        perm_idx : numpy array of integers
            Each row gives the index of the node pair for one symmetry line
            or plane.
        '''

        perm_idx = -1*np.ones(self.nn, dtype=int)

        for i in range(self.nn):
            # Check if the mapping for node i has already be done
            if perm_idx[i] > -1:
                continue

            xy_i = self.xy[i,:]

            if self.dim == 1:
                check = np.abs(x_equal - xy_i[0]) < self.tol
            if self.dim == 2:
                check0 = np.abs(x_equal - xy_i[0]) < self.tol
                check1 = np.abs(y_equal - xy_i[1]) < self.tol
                check = check0 & check1
            elif self.dim == 3:
                check0 = np.abs(x_equal - xy_i[0]) < self.tol
                check1 = np.abs(y_equal - xy_i[1]) < self.tol
                check2 = np.abs(z_equal - xy_i[2]) < self.tol
                check = check0 & check1 & check2

            if np.sum(check) == 1:
                j = np.argmax(check)
                perm_idx[i] = j
                perm_idx[j] = i
            else:
                raise Exception('Error trying to find symetric node pair')

        return perm_idx

    def dir_perm(self):
        '''
        Purpose
        ----------
        Identifies the symmetry line or plane that is symmetrical about
        two coordintes, eg x=y or x=z.
        This is used to permute directional operators from one cartesian
        direction to another.

        Returns
        -------
        dir_perm : numpy array of int
            Each row has the permutations to permutate to a different direction
            The first row is no permutation, it has the same impact as the
            identity matrix
        '''

        if self.dim == 0:
            dir_perm = np.zeros((1, 1), dtype=int)
        elif self.dim == 1:
            dir_perm = np.zeros((1, self.nn), dtype=int)
            dir_perm[0, :] = np.arange(0, self.nn, dtype=int)
        if self.dim == 2:
            dir_perm = np.zeros((2, self.nn), dtype=int)
            dir_perm[0, :] = np.arange(0, self.nn, dtype=int)
            dir_perm[1, :] = self.perm_idx_all[0, :] # x -> y
        elif self.dim == 3:
            dir_perm = np.zeros((4, self.nn), dtype=int)
            dir_perm[0, :] = np.arange(0, self.nn, dtype=int)
            dir_perm[1, :] = self.perm_idx_all[2, :] # x -> y
            dir_perm[2, :] = self.perm_idx_all[1, :] # x -> z
            dir_perm[3, :] = self.perm_idx_all[0, :] # y -> z

        return dir_perm

    @staticmethod
    def make_perm_matrix(perm_order):
        '''
        Parameters
        ----------
        perm_order : numpy array of int
            Permutation order across a symmetry line or plane.

        Returns
        -------
        perm : numpy array of int
            Creates a permutation matrix for each row in perm_order.
            The i-th perm 2D array is given by perm[i,:,:]
        '''

        n_perm, nn = perm_order.shape

        perm = np.zeros((n_perm, nn, nn), dtype=int)
        idty = np.eye(nn)

        for i in range(0, n_perm):
            perm[i,:,:] = idty[:, perm_order[i, :]]

        return perm

    def make_perm_mat_r(self):
        '''
        Purpose
        ----------
        Returns the permutation matrices that permute the int/ext operators rr
        from facet one to any other facet and it does so with the proper
        orientation.
        For more information on this refer to Appendix B of André Marchildon's
        thesis on SBP operators.

        Returns
        -------
        perm_mat_r : numpy array of int
            The i-th 2D permuation array is given by perm_mat_r[i,:,:].
            This is useful to calculate the operator rr for any facet knowing
            facet 0: R2 = R0 perm_mat_r[2,:,:].
        '''

        perm_mat_r = np.zeros((self.n_facets, self.nn, self.nn), dtype=int)
        perm_mat_r[0,:,:] = np.eye(self.nn, dtype=int)

        perm_mat_all = self.perm_mat_all

        if self.dim == 1:
            perm_mat_r[1,:,:] = perm_mat_all[0,:,:]  # Facet 0 to 1
        elif self.dim == 2:
            perm_mat_r[1,:,:] = perm_mat_all[0,:,:] @ perm_mat_all[2,:,:]   # Facet 0 to 1
            perm_mat_r[2,:,:] = perm_mat_all[0,:,:] @ perm_mat_all[1,:,:]   # Facet 0 to 2
            # perm_mat_r[2,:,:] = perm_mat_all[1,:,:] @ perm_mat_all[0,:,:] # Facet 1 to 2
        elif self.dim == 3:
            perm_mat_r[1,:,:] = perm_mat_all[2,:,:] @ perm_mat_all[5,:,:]   # Facet 0 to 1
            perm_mat_r[2,:,:] = perm_mat_all[2,:,:] @ perm_mat_all[4,:,:]   # Facet 0 to 2
            perm_mat_r[3,:,:] = perm_mat_all[1,:,:] @ perm_mat_all[3,:,:]   # Facet 0 to 3
            # perm_mat_r[4,:,:] = perm_mat_all[4,:,:] @ perm_mat_all[2,:,:] # Facet 1 to 2
            # perm_mat_r[5,:,:] = perm_mat_all[4,:,:] @ perm_mat_all[1,:,:] # Facet 1 to 3
            # perm_mat_r[6,:,:] = perm_mat_all[1,:,:] @ perm_mat_all[0,:,:] # Facet 2 to 3
        elif self.dim != 0:
            raise Exception('Can only accept 0 to 3 dimensions for make_perm_mat_r')

        return perm_mat_r

    def find_nodes_on_facet0(self):
        '''
        Returns
        -------
        node_on_facet0 : nupy array of bool
            For each node it indicates if the node is on facet zero for the
            reference simplex element.
        '''

        if self.dim <= 1:
            node_on_facet0 = self.xy[:,0] < self.tol
        else:
            node_on_facet0 = (1 - np.sum(self.xy, axis=1)) < self.tol

        return node_on_facet0
