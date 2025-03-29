#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:19:35 2020

@author: bercik
"""

# Add the root folder of ECO to the search path
import os
import sys

test_folder_path, _ = os.path.split(__file__)
root_folder_path, _ = os.path.split(test_folder_path)
sys.path.append(root_folder_path)

# Import the required modules
import numpy as np

from Source.Disc.SbpQuadRule import SbpQuadRule
from Source.Disc.RefSimplexElem import RefSimplexElem
from Source.Disc.MakeSbpOp import MakeSbpOp

class MakeDgOp:

    tol = 1e-8
    dim = 1 # only currently set up for dim=1

    def __init__(self, p, quad='lgl', basis_type='lagrange'):

        '''
        Parameters
        ----------
        p : int
            Degree of the DG scheme (degree of polynomial basis).
        dim : int
            Dimension of the DG scheme.
        quad : string, optional
            Determines the flux nodes and quadrature. Currently only option is 
            LGL. TO DO: Generalize.
        basis_type : string, optional
            Indicates the type of basis to use for the flux nodes. Currently
            only option is lagrange. TO DO: change this

        Returns
        -------
        None.

        '''

        ''' Add inputs to the class '''

        self.p = p
        self.quad_type = quad
        self.basis_type = basis_type
        
        ############################ FLUX NODES ##############################
        # Define quadrature on flux nodes quad.xq
        if quad == 'lgl': self.quad = SbpQuadRule(1, p, sbp_fam='R0', quad_rule='lgl')
        else: raise Exception("Scheme only currently set up for collocated DGSEM with LGL quadrature")        
        
        
        ########################## SOLUTION NODES ############################
        # Define solution nodes. Currectly collocating solution and flux nodes
        # for DGSEM, but in general self.xy are solution nodes, and self.quad.xy 
        # are flux nodes. These do not have to be equal
        self.xy = self.quad.xq
        self.wBary = self.BaryWeights(self.xy)
        
        
        ############################ FACET NODES #############################
        # Define facet nodes. Once again, collocate for DGSEM
        # TODO: temporarily adding this in
        #ref_elem = RefSimplexElem(self.dim)
        # vert: each row is the cartesian coordinate for one vertex
        self.vert = np.array([[0],[1]]) #ref_elem.vert
        # normal: each row gives an outward (not unit) normal vector for a facet
        self.normal = np.array([[-1],[1]]) #ref_elem.normal
        
        
        ############################# CONSTANTS ##############################
        # nn: number of solution nodes
        self.nn_sol = np.shape(self.xy)[0]
        # number of flux nodes
        self.nn_flux = self.quad.nn
        # number of facet nodes
        self.nn_facet = 1 # temporary hack for dim=1. #TODO: could use quad.wqf?
        # n_facet: number of facets for the reference element
        self.n_facet = self.dim + 1
        
        
        ########################## USEFUL MATRICES ###########################
        # van: vandermonde matrix (in DGSEM, vandermonde is identity matrix)
        self.van = self.VandermondeLagrange1D(self.quad.xq,self.xy,self.wBary)
            # TODO: should I move this along with BaryWeights to BasisFun?
        # van_f: facet vandermonde matrices for each facet (equivalent to rrL and rrR)
        self.vanf = np.zeros((self.n_facet, self.nn_facet, self.nn_sol))
        for i in range(self.n_facet):
            self.vanf[i] = self.VandermondeLagrange1D(self.vert[i],self.xy,self.wBary)
        # weight: inner product matrix W (stores weights for flux nodes)
        self.weight = self.InnerProduct(self.quad.wq)
        # weight_f: facet inner product matrix W (stores weights for facet nodes)
        self.weightf = self.InnerProduct(self.quad.wqf)
        # dd_x_sol: x derivative operator defined on solution nodes
        self.dd_x_sol = self.DerivativeLagrange1D(self.xy,self.wBary)
            # TODO: Make more general for dim>1
        
        # The following are not actually used. Regardless they are good for debugging.
        
        # mass: mass matrix M (acts on solution nodes, not flux nodes)
        self.mass = self.van.T @ self.weight @ self.van  
        # proj: Discrete orthogonal projection operator P (maps from flux nodes
        #       to solution nodes, the opposite of vandermonde matrix. P@V=I)
        self.proj = np.linalg.inv(self.mass) @ self.van.T @ self.weight # define projection matrix     
            # TODO: facet mass matrix? - currently all in BoundaryOperator1D    
            # TODO: facet weight matrix?
        # dd_x: x derivative operator defined on flux nodes
        dd_x = self.van @ self.dd_x_sol @ self.proj
        # ee_x: x directional boundary integration operator
        ee_x = self.BoundaryOperator1D()
            # TODO: Make more general for dim>1
            
        # Permute all of the directional operators (TODO: actually permute)
        self.ee = self.permute_dir_op(ee_x)
        self.dd = self.permute_dir_op(dd_x)
        
        # Test the operators
        MakeSbpOp.sbp_test_cub_rule(self.xy, self.mass, 2 * self.p - 1)
        # TODO: include the line below. Here rr is like van_facet (see BoundaryOperator1D)
        # MakeSbpOp.sbp_test_acc_rr(self.rr, self.xy, self.xqf, self.p)
        MakeSbpOp.sbp_test_derivative(self.dd, self.xy, p)
        


    def BoundaryOperator1D(self):
        '''
         Purpose
        -------
        Calculates the directional boundary integration operator in 1D.
        Follows definitions and notation from Tristan's paper - E matrix.
        Operates on flux nodes. This need only slightly be tweaked for 
        multiple dimensions.
        '''
        n = np.shape(self.quad.xq)[0] # number of flux nodes
        E = np.zeros((n,n)) # initialize boundary operator
        
        # facet weight matrix for reference facet
        W = self.InnerProduct(self.quad.wqf)
            # TODO: Make more general for any set of facet nodes
        
        for i in range(np.shape(self.vert)[0]):
            # facet vandermonde matrix for facet i
            Vi = self.vanf[i]
            # facet mass matrix for facet i
            Mi = Vi.T @ W @ Vi
            # outward normal component (1D) of facet i
            ni = self.normal[i,0]
                # TODO: these are not normalized, could be a problem for dim>1
                #        also beware of choosing proper component
            # sum over facets to get boundary integration
            E = E + ( self.proj.T @ Mi @ self.proj ) * ni
        return E
    
    def permute_dir_op(self, op_dir0):
        '''
         Purpose
        -------
        NOTE: Currently doesn't actually do anything. I only have this here
        temporarily to put the patrices in the same form as the rest of the
        modules in the code (See equivalent function in MakeSbpOp). 
        I should fix this.

        Parameters
        ----------
        op_dir0 : numpy array
            Directional operator for the zero-th (x) direction
        '''

        op_dir_all = np.zeros((self.dim, self.nn_sol, self.nn_sol))

        for d in range(self.dim):
            #perm_mat_dir = self.perm.perm_mat_dir[d,:,:]
            #op_dir_all[d,:,:] = perm_mat_dir @ op_dir0 @ perm_mat_dir
            op_dir_all[d,:,:] = op_dir0

        return op_dir_all   
    
    @staticmethod
    def BaryWeights(xNode):
        '''
        Purpose
        -------
        Calculates the 1D barycentric weights useful for the calculation of 
        lagrange polynomials and lagrange polynomial interpolation. For more
        information, see https://en.wikipedia.org/wiki/Lagrange_polynomial

        Parameters
        ----------
        nodal locations on which the lagrange polynomials are defined
        
        Returns
        ----------
        barycentric weights for lagrange interpolation at given nodal locations
        '''
        if xNode.ndim != 1:
            dim = xNode.shape[1]
            assert dim == 1, 'barycentric weights only currently set up for dim=1'
        
        Np=len(xNode) # number of basis functions
        wB=np.ones(Np) # initiate vector to store barycentric weights
        
        # temporarily set wB_j = l'(x_j) = \prod_{i \neq j} (x_j-x_i)
        for j in range(1,Np):
            for k in range(0,j):
                wB[k] = wB[k]*(xNode[k]-xNode[j])
                wB[j] = wB[j]*(xNode[j]-xNode[k])
        # correctly set wB_j = 1 / l'(x_j)
        wB = 1.0/wB
        return wB

    @staticmethod
    def VandermondeLagrange1D(xFlux,xNode,wBary=None):
        '''
        Purpose
        -------
        Calculates the Vandermonde Matrix in 1D from a Lagrange basis. Using
        Tristan's notation V, this maps from solution nodes {S_p}={xNode} (on
        which the Lagrange basis is defined) to flux nodes {S_\omega} = {xFlux}
        (equivalently, lagrange polynomials defined by barycentric weights
        wBary and nodal locations xNode evaluated at given points xFlux) For
        more information, see https://en.wikipedia.org/wiki/Lagrange_polynomial
        
        The left inverse of the Vandermonde is given by self.proj, P, which maps
        from flux nodes to solution nodes. In DGSEM, both are the idenity matrix

        Parameters
        ----------
        xFlux: locations at which to evaluate lagrange polynomials
        wBary: barycentric weights defining lagrange polynomial basis
        xNode: nodal locations defining lagrange polynomial basis
        
        Returns
        ----------
        Vandermonde matrix where jth column gives the jth lagrange 
        polynomial evaluated at row ith node in xFlux. Acts on a function
        defined by solution nodal values of xNode.
        '''
        if xNode.ndim != 1:
            dim = xNode.shape[1]
            assert dim == 1, ("Lagrange Vandermonde only set up for dim=1.")
            x = xNode[:,0]
        else:
            x = xNode

        if wBary is None:
            # Calculate barycentric weights based on x_old
            wBary = MakeDgOp.BaryWeights(x)
        
        Np = len(x) # number of nodes that define lagrange polynomials
        M = len(xFlux) # points to which we interpolate
        V=np.zeros((M,Np)) # initialize matrix
        for i in range(M):  
            xi = xFlux[i]
            if any(abs(xi - x) < 1e-12):
                V[i, np.where(abs(xi - x) < 1e-12)] = 1.0
            else:
                l=1
                for k in range(Np):
                    l = l*(xi-x[k]) # useful polynomial l(x)
                    V[i,k] = wBary[k]/(xi-x[k]) # intermediate step
                V[i,:]=V[i,:]*l # define l_j(x_i)
        return V
    
    @staticmethod
    def DerivativeLagrange1D(xNode,wBary=None):
        '''
         Purpose
        -------
        Calculates the 1D DG derivative operator defined on the reference
        element assuming a lagrange nodal basis. Uses barycentric form, 
        for more info see https://en.wikipedia.org/wiki/Lagrange_polynomial

        Parameters
        ----------
        wBary: barycentric weights defining lagrange polynomial basis
        xNode: nodal locations defining lagrange polynomial basis
        '''
        if xNode.ndim != 1:
            dim = xNode.shape[1]
            assert dim == 1, ("Lagrange Derivative only set up for dim=1.")
            x = xNode[:,0]
        else:
            x = xNode

        if wBary is None:
            # Calculate barycentric weights based on x_old
            wBary = MakeDgOp.BaryWeights(x)
        
        Np = len(xNode)
        D = np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                if (i!=j):
                    # D_ij = d l_j / d x at x=xNode[i]
                    D[i,j] = wBary[j]/wBary[i]/(xNode[i]-xNode[j])
                    # exploit that rows sum to 0 for case x_i = x_j
                    D[i,i] = D[i,i] - D[i,j]
        return D 
    
    @staticmethod
    def VandermondeLegendre1D(x_in,p,orthonormal=True):
        '''
        Purpose
        -------
        Calculates the Vandermonde Matrix in 1D from a Legendre modal basis 
        to evaluations of the Legendre basis at nodal locations.

        If x_in is a vector of p+1 nodal locations, then the Vandermonde matrix
        maps from modal coefficients of the Legendre basis to solution nodes
        {S_p}={x_in} on which the Lagrange basis is defined.
        i.e., u_modal = V @ u_nodal

        Parameters
        ----------
        x_in: nodal locations defining lagrange polynomial basis
        p : int
            Maximum polynomial degree.

        Returns
        -------
        V : ndarray of shape ((p+1), (p+1))
            The Vandermonde matrix whose rows are the orthonormal 
            Legendre polynomials evaluated at x_in.
        '''
        # Sanity check
        if x_in.ndim != 1:
            dim = x_in.shape[1]
            assert dim == 1, ("Legendre Vandermonde only set up for dim=1.")
            x = x_in[:,0]
        else:
            x = x_in
        
        N = len(x)  # number of nodes

        # Initialize the Vandermonde matrix
        V = np.zeros((p+1, N), dtype=float)

        # Fill each row with the orthonormal Legendre polynomial of degree n
        # Orthonormal Legendre: \hat{P}_n(x) = sqrt((2n+1)/2) * P_n(x)
        # where P_n(x) is the standard Legendre polynomial of degree n.
        for n in range(p+1):
            # Create the standard Legendre polynomial of degree n
            # via the numpy.polynomial.legendre module
            coeffs = np.zeros(n+1)
            coeffs[-1] = 1.0  # x^n term
            Pn = np.polynomial.legendre.Legendre(coeffs,domain=np.array([0.,1.]))

            # Compute orthonormal factor
            if orthonormal:
                factor = np.sqrt(2*n + 1)
            else:
                factor = 1.0

            # Evaluate at each node
            V[:,n] = factor * Pn(x)

        return V
    
    @staticmethod
    def EvaluateLegendre1D(x_in,coeffs,orthonormal=True):
        '''
        Purpose
        -------
        Evaluates the Legendre modal basis at given nodal locations.

        IF x is a vector of p+1 nodal locations, then the Vandermonde matrix
        maps from modal coefficients of the Legendre basis to solution nodes
        {S_p}={x_in} on which the Lagrange basis is defined.
        i.e., u_modal = V @ u_nodal

        Parameters
        ----------
        x_in: nodal locations defining lagrange polynomial basis
        coeffs : coefficients of the Legendre modal basis

        Returns
        -------
        u : 1d array of solution evaluated at x_in locations
        '''
        # Sanity check
        if x_in.ndim != 1:
            dim = x_in.shape[1]
            assert dim == 1, ("Legendre only set up for dim=1.")
            x = x_in[:,0]
        else:
            x = x_in
        
        N = len(x)  # number of nodes
        p = len(coeffs) - 1  # polynomial degree

        # set the corrections for orthonormal
        if orthonormal:
            factor = np.sqrt(2*np.arange(p+1) + 1)
        else:
            factor = 1.0

        # create the Legendre polynomial
        Pn = np.polynomial.legendre.Legendre(factor * coeffs,domain=np.array([0.,1.]))

        # Evaluate at each node
        u = Pn(x)

        return u
    
    @staticmethod
    def DerivativeLegendre1D(p, orthonormal=True):
        '''
        Purpose
        -------
        Constructs the differentiation matrix that maps modal Legendre coefficients
        to the derivative coefficients in the Legendre basis.
        Uses the shifted Legendre polynomials Pn^* on [0,1], where Pn^*(x) = Pn(2x - 1).

        Parameters
        ----------
        p : int
            Maximum polynomial degree.
        orthonormal : bool, optional
            Whether to use orthonormal Legendre polynomials. Default is True.

        Returns
        -------
        D : ndarray of shape ((p+1), (p+1))
            The differentiation matrix that transforms Legendre modal coefficients
            to modal derivatives.
        '''
        D = np.zeros((p+1, p+1), dtype=float)

        # Fill the matrix: columns = n, rows = m
        if orthonormal:
            for n in range(1, p+1):
                for m in range(n):
                    if (n - m) % 2 == 1: 
                        D[m, n] = (4*m + 2) * np.sqrt(2*n+1) / np.sqrt(2*m+1)
        else:
            for n in range(1, p+1):
                for m in range(n):
                    if (n - m) % 2 == 1: 
                        D[m, n] = 4*m + 2 

        return D
    
    @staticmethod
    def Filter1D(p,Nc,s):
        '''
        A python implementation of the 1D filter from Hesthaven & Warburton, pg. 130 eq. 5.16.

        Purpose
        -------
        This function returns a 1D filter matrix that can be used to filter out high frequency
        modes in a DGSEM simulation. The filter matrix assumes u is in a modal basis.

        A good set of starting parameters are Nc=0,s=16
        '''
        eps = np.finfo(float).eps
        filterdiag = np.ones(p+1)
        alpha = -np.log(eps)

        # Initialize the filter function
        for i in range(Nc,p+1):
            filterdiag[i] = np.exp(-alpha*((i-Nc)/(p-Nc))**s)

        # normally, would now do F = V @ np.diag(filterdiag) @ inv(V)
        return filterdiag


    
    @staticmethod
    def InnerProduct(weights=None):
        '''
        Purpose
        -------
        Defines the discrete inner product between two vectors represented by
        nodal values on the flux nodes. In Tristan's notation, this acts as a
        matrix on two vectors with values defined on {S_\omega} (or on vectors
        V@xNode, where V is the vandermonde matrix)
        
        When we left and right multiply with the vandermonde, we get the mass
        matrix, which acts on solution nodes.

        Parameters
        ----------
        weights: weights defining quadrature on flux nodes. If given, uses
              Quadrature-Based Approximation from Tristan's notation.
              If weights=None, uses Collocation-Based Approximation.
            
        Note: When collocating solution nodes with flux nodes, 'Quadrature' 
        returns DGSEM with a diagonal matrix while 'Collocation' returns a 
        standard Nodal DG with a dense matrix.
        
        Returns
        ----------
        An SPD matrix which approximates a discrete L2 inner product,
        acting on the left and right to two vectors of size N_\omega
        '''        
        # TO DO: Ensure this works dor dim>1
        if isinstance(weights, np.ndarray)==False:
            raise Exception("Collocation-based inner product not yet set up")
            # should be a dense matrix where entries are pre-integraded 
            # inner products of secondary lagrange basis functions defined 
            # by the flux nodes. TO DO
        else:
            W = np.diag(weights)
        return W
    
    def weak_op(self, detjac, invjac):
        '''
        Create a physical mass matrix, and volume and surface terms that
        already include integration over flux nodes. These will act on fluxes
        defined by solution and facet nodes, respectively. This is for a weak
        DG formulation. For notation, see Tristan's paper, section 4.
        
        Parameters
        ----------
        detjac : numpy array
            Determinant of the mesh Jacobian.
        inv_jac : numpy array
            Inverse of the mesh Jacobian matrix.

        Returns
        -------
        mass : numpy array
            mass matrix.
        vol : numpy array
            matrix acting on volume flux.
        surf : numpy array
            matrices acting on surface fluxes (one for each facet)
        '''
        
        assert self.dim == 1, 'vol term not fully set up for multiple dimensions'
        ################## Mass Matrix ######################
        mass =  self.van.T @ self.weight @ detjac @ self.van

        ################## volume term ######################
        # this term will act on the (actual equation) flux evaluated at flux nodes.
        # the reason we act on flux nodes is to take advantage of overintegration.
        # therefore we must apply the self.van to solution vector before evaluating 
        # flux. The flux may also depend on x, or physical flux nodes (mesh)
        vol = self.dd_x_sol.T @  self.van.T @ self.weight @ detjac @ np.diag(invjac[:,0,0])
        # TODO: for dim>1 we have to loop for all dimensions in invjac
        # TODO: Does the vandermonde mapping have to occur after the determinant???
        
        ################## surface term #####################
        # this term will act on the numerical flux also evaluated at facet nodes.
        # once again this means we must apply self.van_f before evaluating.
        # initialize surface term          
        surf = np.zeros((self.n_facet, self.nn_sol, self.nn_facet))
        for i in range(self.n_facet):
            # TODO: Need a general expression for the jacobian evaluated at
            # the facet for the case where the mapping is not linear.
            # Here I assume Jac(facet) = vanf @ Jac(x) for flux nodes {x_i}
            # This gives me the (dim x dim) jacobians at each facet node
            invjac_f = np.tensordot(self.vanf[i],invjac,axes=(1,0))
            detjac_f = np.tensordot(self.vanf[i],np.diag(detjac),axes=(1,0)) # det at each node, so diagonal entries
            scale = np.zeros(self.nn_facet)
            for j in range(self.nn_facet): # for each facet node
                scale[j] = np.linalg.norm(detjac_f[j] * invjac_f.T[j] @ self.normal[i])
            scale = np.diag(scale)
            #TODO: The norms are not normalized for higher dimensions
            surf[i] = - self.vanf[i].T @ self.weightf @ scale

        return mass, vol, surf

    def strong_op(self, detjac, invjac):
        '''
        Create a physical mass matrix, derivative and boundary terms that
        already include integration over flux nodes. These will act on fluxes
        interpolated to flux nodes, where the facet terms are then interpolated
        again to facet nodes. This is for a strong DG formulation. 
        For notation, see Tristan's paper, section 4.
        
        Parameters
        ----------
        detjac : numpy array
            Determinant of the mesh Jacobian.
        inv_jac : numpy array
            Inverse of the mesh Jacobian matrix.

        Returns
        -------
        mass : numpy array
            mass matrix.
        dd : numpy array
            derivative operator acting on volume flux.
        surf : numpy array
            matrices acting on surface fluxes (one for each facet)
        '''
        
        assert self.dim == 1, 'not fully set up for multiple dimensions'
        ################## Mass Matrix ######################
        mass =  self.van.T @ self.weight @ detjac @ self.van

        ################## volume term ######################
        # this term will act on the (actual equation) flux evaluated at flux nodes.
        # the reason we act on flux nodes is to take advantage of overintegration.
        # therefore we must apply the self.van to solution vector before evaluating 
        # flux. The flux may also depend on x, or physical flux nodes (mesh)
        dd = self.dd_x_sol @ self.proj @ detjac @ np.diag(invjac[:,0,0])
        # TODO: for dim>1 we have to loop for all dimensions in invjac
        
        ################## surface term #####################
        # this term will act on the numerical flux also evaluated at facet nodes.
        # once again this means we must apply self.van_f before evaluating.
        # initialize surface term          
        surf_num = np.zeros((self.n_facet, self.nn_sol, self.nn_facet))
        # initialize map/scale to send numerical flux from flux to facet nodes
        # and integrate over the facet (multiply in van.T @ weightf to both terms)
        # note in reality there should be one for each dimension        
        surf_flux = np.zeros((self.n_facet, self.nn_sol, self.nn_flux))
        for i in range(self.n_facet):
            # TODO: Need a general expression for the jacobian evaluated at
            # the facet for the case where the mapping is not linear or when
            # using GL quadrature as opposed to LGL.
            # Here I assume Jac(facet) = vanf @ Jac(x) for flux nodes {x_i}
            # which only works if vanf is [0,0,0,1] and Jac includes bdy.
            # This gives me the (dim x dim) jacobian determinants at each facet node
            invjac_f = np.tensordot(self.vanf[i],invjac,axes=(1,0))
            detjac_f = np.tensordot(self.vanf[i],np.diag(detjac),axes=(1,0)) # det at each node, so diagonal entries
            scale = np.zeros(self.nn_facet)
            for j in range(self.nn_facet): # for each facet node
                scale[j] = np.linalg.norm(detjac_f[j] * invjac_f.T[j] @ self.normal[i])
            scale = np.diag(scale)
            #TODO: The norms are not normalized for higher dimensions
            surf_num[i] = self.vanf[i].T @ self.weightf @ scale
            #TODO: in reality I would have one term for each dimension
            surf_flux[i] = self.vanf[i].T @ self.weightf @ self.vanf[i] @ self.proj @ detjac @ np.diag(invjac[:,0,0]) * self.normal[i,0]

        return mass, dd, surf_num, surf_flux

##############################################################################
################################ Not used things #############################
##################### (but still helpfull to keep around) ####################
"""

        # Get the legendre Vandermonde matrix at the element nodes
        # NOTE: this is not the vandermonde composed with our actual DG basis, 
        #       rather it is a vandermonde matrix of an equivalent modal 
        #       legendre basis that is useful in constructing operators
        #       (Refer to Hesthaven & Warburton for details)
        leg_basis = BasisFun(self.xy, self.p, basis_type='legendre')
        self.leg_van = leg_basis.van
        self.leg_van_der = leg_basis.van_der
        
        
    def Derivative1D(self):
        '''
         Purpose
        -------
        Calculates the 1D DG derivative operator defined on the reference
        element. Refer to Hesthaven & Warburton for details (pg. 53).

        Parameters
        ----------
        something
        '''
        
        leg_van_der = self.leg_van_der[0,:,:] # Derivative wrt x
        dd = np.linalg.solve(self.leg_van.T, leg_van_der.T).T 
        # note: above is equivalent to dd = van_der @ inv(van)
        
        return dd
    
    def StrongSurfaceInt1D(self):
        '''
         Purpose
        -------
        Calculates the 1D DG strong form surface integral operator 
        (lifting operator) defined on the reference element. Refer to 
        Hesthaven & Warburton for details (pg. 56).

        Parameters
        ----------
        something
        '''

        Emat = np.zeros((self.n_p,self.n_facet*self.nnf));
        Emat[0,0] = 1.0
        Emat[self.n_p-1,-1] = 1.0
        
        lift = self.leg_van @ (self.leg_van.T @ Emat)
        return lift
"""                