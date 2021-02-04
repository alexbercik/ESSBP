import numpy as np


class RefSimplexElem:
    '''
        Purpose
        ----------
        Provides the following information for a reference simplex element:
            1) The cartesian coordinates of the vertices
            2) The normal vector for each facet
            3) A mapping from facet number to vertex number
            4) The size of the reference vertex element
    '''

    def __init__(self, dim):
        '''
        Parameters
        ----------
        dim : int
            The number of dimension for the simplex element.

        Returns
        -------
        None.
        '''

        ''' Add inputs to the class '''
        assert isinstance(dim, int), "dim should be an integer"
        self.dim = dim

        ''' Get info for the reference simplex element'''

        self.vert = self.ref_vertices(self.dim)
        self.normal = self.ref_facet_normal(self.dim)
        self.f2v = self.facet_2_vertices(self.dim)
        self.elem_size = self.ref_elem_size(self.vert)

    @staticmethod
    def ref_vertices(dim):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.

        Returns
        -------
        vert : numpy array
            Each row is the cartesian coordinate for one vertex.
        '''

        if dim == 0:
            vert = np.array([[0]])
        elif dim == 1:
            vert = np.array([[0, 1]]).T
        elif dim == 2:
            vert = np.array([[0, 0],
                            [1, 0],
                            [0, 1]])
        elif dim == 3:
            vert = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        else:
            raise Exception('Vertices are only available for 0 to 3 dimensions')

        return vert

    @staticmethod
    def ref_facet_normal(dim):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.

        Returns
        -------
        normal : numpy array
            Each row is one (not unit) normal vector for one facet.
        '''

        if dim == 1:
            normal = np.array([[-1], [1]])
        elif dim == 2:
            normal = np.array([
                [1, 1],
                [-1, 0],
                [0, -1]])
        elif dim == 3:
            normal = np.array([
                [1, 1, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]])
        else:
            raise Exception('Normals are only available for 1 to 3 dimensions')

        return normal

    @staticmethod
    def facet_2_vertices(dim):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.

        Returns
        -------
        f2v : numpy array
            Column i gives all the vertices that are on facet i.
        '''

        if dim == 1:
            f2v = np.array([[0, 1]], dtype=int)
        elif dim == 2:
            # The facet no. is given by the number of the opposing vertex
            f2v = np.array([[1, 2, 0],
                            [2, 0, 1]], dtype=int)
        elif dim == 3:
            # The facet no. is given by the number of the opposing vertex
            f2v = np.array([[1, 2, 0, 0],
                            [2, 0, 1, 2],
                            [3, 3, 3, 1]], dtype=int)
        else:
            raise Exception('Facet to vertex mapping is only available for 1 to 3 dimensions')

        return f2v

    @staticmethod
    def ref_elem_size(vert):
        '''
        Parameters
        ----------
        vert : numpy array
            Each row are the cartesian coordinates for one vertex.

        Returns
        -------
        size : float
            The length, area of volume for a line, triangle or tetrahedral
            element, respectively.
        '''

        n_vert = vert.shape[0]

        dim = n_vert - 1

        if dim == 0:
            size = 1
        elif dim == 1:
            size = vert[1, 0] - vert[0, 0]
        elif dim == 2:
            v0 = vert[0,:]
            v1 = vert[1,:]
            v2 = vert[2,:]

            size = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v1))
        elif dim == 3:
            v0 = vert[0,:]
            v1 = vert[1,:]
            v2 = vert[2,:]
            v3 = vert[3,:]

            size = (1/6) * np.linalg.norm(np.dot(np.cross(v1-v0, v2-v0), v3-v0))

        else:
            raise Exception('ref_elem_size Only works for 0 to 3 dimensions')

        return size