#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:54:23 2020

@author: bercik
"""

import numpy as np
import matplotlib.pyplot as plt
import Source.Methods.Functions as fn
import Source.Methods.Sparse as sp
from sys import stderr
from contextlib import redirect_stderr

class MakeMesh:
    
    def __init__(self, dim, xmin, xmax, 
                 nelem, x_op, warp_factor=0,
                 warp_type='default', print_progress=True):
        '''
        Parameters
        ----------
        dim : int
            The dimension of the problem. For now can only be 1 or 2.
        xmin : float or (float,float)
            Min coordinate of the mesh, either x in 1D or (x,y) in 2D
        xmax : float or (float,float)
             Max coordinate of the mesh, either x in 1D or (x,y) in 2D
        nelem : int or (int,int), optional
            No. of elements in the mesh
        x_op : np array
            Indicates the nodal locations for 1D of one element.
        '''

        ''' Add all inputs to the class '''

        self.dim = dim
        self.xmin = xmin
        self.xmax = xmax
        self.nelem = nelem
        self.x_op = x_op
        if isinstance(warp_factor,float) or isinstance(warp_factor,int):
            self.warp_factor = warp_factor
            self.warp_factor2 = 0
            self.warp_factor3 = 0
        elif isinstance(warp_factor,list) or (isinstance(warp_factor,np.ndarray) and np.ndim(warp_factor)==1):
            self.warp_factor = warp_factor[0]
            if len(warp_factor)>1:
                self.warp_factor2 = warp_factor[1]
            else:
                self.warp_factor2 = 0
            if len(warp_factor)>2:
                self.warp_factor3 = warp_factor[2]
            else:
                self.warp_factor3 = 0
            if len(warp_factor)>3:
                raise Exception('Only set up for 1, 2, or 3 warp factors')
        else:
            raise Exception('warp_factor must be a float / int, or list / 1d array of len <= 3, not', warp_factor)
        self.warp_type = warp_type
        self.print_progress = print_progress

        ''' Additional terms '''
        
        if self.print_progress: print('... Building Mesh')
        
        if self.dim == 1:
            self.dom_len = self.xmax - self.xmin
            self.build_mesh_1d()
            if self.warp_factor!=0:
                self.stretch_mesh_1d()
             
        elif self.dim == 2:
            self.dom_len = (self.xmax[0] - self.xmin[0], self.xmax[1] - self.xmin[1])
            self.build_mesh_2d()
            if self.warp_factor!=0:
                self.warp_mesh_2d()
                
        elif self.dim == 3:
            self.dom_len = (self.xmax[0] - self.xmin[0], self.xmax[1] - self.xmin[1], self.xmax[2] - self.xmin[2])
            self.build_mesh_3d()
            if self.warp_factor!=0:
                self.warp_mesh_3d()
        
        else:
            raise Exception('Only currently set up for 1D, 2D, or 3D')




    def build_mesh_1d(self):

        ''' Extract required info '''

        self.nen = len(self.x_op)    # No. of nodes per elem and no. of dim
        self.nn = self.nen * self.nelem     # Total no. of nodes

        ''' Create mesh '''

        # Get the location of all the nodes in the element
        self.x_elem = np.zeros((self.nen, self.nelem)) # Each slice is for one elem
        self.bdy_x = np.zeros((2,self.nelem))
        
        self.vertices = np.linspace(self.xmin,self.xmax,self.nelem+1)

        elem_length = self.vertices[1] - self.vertices[0] # all equal length
        for i in range(self.nelem):
            #self.x_elem[:, i] = (self.vertices[i+1] - self.vertices[i])*self.x_op + self.vertices[i]
            self.x_elem[:, i] = elem_length*self.x_op + self.vertices[i]
            self.bdy_x[0,i], self.bdy_x[1,i] = self.vertices[i], self.vertices[i+1]

        self.x = self.x_elem.flatten('F')
        
        # create exact jacobian (dx_phys/dx_ref)
        self.jac_exa = np.ones((self.nen, 1, self.nelem))*elem_length
        self.jac_inv_exa = np.ones((self.nen, 1, self.nelem))/elem_length
        self.det_jac_exa = np.ones((self.nen, self.nelem))*elem_length
        #self.det_jac_inv_exa = np.ones((self.nen, self.nelem))/elem_length
        
        self.bdy_jac_exa = np.ones((1,2,self.nelem))*elem_length
        #self.bdy_det_jac_exa = np.ones((2,self.nelem))*elem_length
        #self.bdy_jac_inv_exa = np.ones((1,2,self.nelem))/elem_length
            

    def build_mesh_2d(self):

        ''' Extract required info '''

        self.nen = len(self.x_op)    # No. of nodes per elem and no. of dim
        self.nn = (self.nen * self.nelem[0] , self.nen * self.nelem[1])     # Total no. of nodes

        ''' Create mesh '''

        # Get the location of all the nodes in the element
        self.xy_elem = np.zeros((self.nen**2, 2, self.nelem[0]*self.nelem[1])) # Each slice is for one elem 
        self.xy = np.zeros((self.nn[0]*self.nn[1],2))
        self.grid_lines = np.zeros(((self.nelem[0]+self.nelem[1]+2),2,100))
        self.bdy_xy = np.zeros((self.nen,2,4,self.nelem[0]*self.nelem[1])) # nodes, x or y (physical value), facet (Left, Right, Lower, Upper), element
        
        verticesx = np.linspace(self.xmin[0],self.xmax[0],self.nelem[0]+1)
        verticesy = np.linspace(self.xmin[1],self.xmax[1],self.nelem[1]+1)

        elem_length_x = verticesx[1] - verticesx[0] # all equal length
        elem_length_y = verticesy[1] - verticesy[0] # all equal length
        for i in range(self.nelem[0]):
            #x_elem = (verticesx[i+1] - verticesx[i])*self.x_op + verticesx[i]
            x_elem = elem_length_x*self.x_op + verticesx[i]
            for j in range(self.nelem[1]):
                #y_elem = (verticesy[j+1] - verticesy[j])*self.x_op + verticesy[j]
                y_elem = elem_length_y*self.x_op + verticesy[j]
                
                e = self.nelem[1]*i+j              
                self.xy_elem[:,:,e] = np.array(np.meshgrid(y_elem, x_elem)).reshape(2, -1).T[:,[1,0]]
                self.bdy_xy[:,0,0,e], self.bdy_xy[:,0,1,e] = verticesx[i], verticesx[i+1]
                self.bdy_xy[:,1,0,e] = self.bdy_xy[:,1,1,e] = y_elem
                self.bdy_xy[:,1,2,e], self.bdy_xy[:,1,3,e] = verticesy[j], verticesy[j+1]
                self.bdy_xy[:,0,2,e] = self.bdy_xy[:,0,3,e] = x_elem
        
        self.xy_elem_unwarped = np.copy(self.xy_elem) 
        for i in range(self.nelem[0]*self.nelem[1]):
            a = i*self.nen**2
            b = (i+1)*self.nen**2
            self.xy[a:b,:] = self.xy_elem[:,:,i]
        
        for i in range(self.nelem[0]+1):
            self.grid_lines[i,0,:] = verticesx[i]*np.ones(100)
            self.grid_lines[i,1,:] = np.linspace(self.xmin[1],self.xmax[1],100) 
        for i in range(self.nelem[1]+1): 
            j = self.nelem[0]+1+i           
            self.grid_lines[j,0,:] = np.linspace(self.xmin[0],self.xmax[0],100)
            self.grid_lines[j,1,:] = verticesy[i]*np.ones(100)
            
        # create exact jacobian (dx_phys/dx_ref)
        self.jac_exa = np.zeros((self.nen**2, 2, 2, self.nelem[0]*self.nelem[1])) # 2x2 matrix in 2D
        self.jac_exa[:,0,0,:] = elem_length_x
        self.jac_exa[:,1,1,:] = elem_length_y
        #self.jac_inv_exa = np.zeros((self.nen**2, 2, 2, self.nelem[0]*self.nelem[1]))
        #self.jac_inv_exa[:,0,0,:] = 1/elem_length_x
        #self.jac_inv_exa[:,1,1,:] = 1/elem_length_y
        self.det_jac_exa = np.ones((self.nen**2, self.nelem[0]*self.nelem[1]))*(elem_length_x*elem_length_y)
        #self.det_jac_inv_exa = np.ones((self.nen**2, self.nelem[0]*self.nelem[1]))/(elem_length_x*elem_length_y)
        
        self.bdy_jac_exa = np.zeros((self.nen,2,2,4,self.nelem[0]*self.nelem[1]))
        self.bdy_jac_exa[:,0,0,:,:] = elem_length_x
        self.bdy_jac_exa[:,1,1,:,:] = elem_length_y        
        self.bdy_det_jac_exa = np.ones((self.nen,4,self.nelem[0]*self.nelem[1]))*(elem_length_x*elem_length_y)
        #self.bdy_jac_inv_exa = np.zeros((self.nen,2,2,4,self.nelem[0]*self.nelem[1]))
        #self.bdy_jac_inv_exa[:,0,0,:,:] = 1/elem_length_x
        #self.bdy_jac_inv_exa[:,1,1,:,:] = 1/elem_length_y

    def build_mesh_3d(self):

        ''' Extract required info '''

        self.nen = len(self.x_op)    # No. of nodes per elem and no. of dim
        self.nn = (self.nen * self.nelem[0], self.nen * self.nelem[1], self.nen * self.nelem[2])     # Total no. of nodes

        ''' Create mesh '''

        # Get the location of all the nodes in the element
        self.xyz_elem = np.zeros((self.nen**3, 3, self.nelem[0]*self.nelem[1]*self.nelem[2])) # Each slice is for one elem
        self.xyz = np.zeros((self.nn[0]*self.nn[1]*self.nn[2],3))
        self.grid_planes = np.zeros((self.nelem[0]+self.nelem[1]+self.nelem[2]+3,3,25,25))
        self.bdy_xyz = np.zeros((self.nen**2,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
        
        verticesx = np.linspace(self.xmin[0],self.xmax[0],self.nelem[0]+1)
        verticesy = np.linspace(self.xmin[1],self.xmax[1],self.nelem[1]+1)
        verticesz = np.linspace(self.xmin[2],self.xmax[2],self.nelem[2]+1)

        elem_length_x = verticesx[1] - verticesx[0] # all equal length
        elem_length_y = verticesy[1] - verticesy[0] # all equal length
        elem_length_z = verticesz[1] - verticesz[0] # all equal length
        for i in range(self.nelem[0]):
            #x_elem = (verticesx[i+1] - verticesx[i])*self.x_op + verticesx[i]
            x_elem = elem_length_x*self.x_op + verticesx[i]
            for j in range(self.nelem[1]):
                #y_elem = (verticesy[j+1] - verticesy[j])*self.x_op + verticesy[j]
                y_elem = elem_length_y*self.x_op + verticesy[j]
                for k in range(self.nelem[2]):
                    #z_elem = (verticesz[k+1] - verticesz[k])*self.x_op + verticesz[k]
                    z_elem = elem_length_z*self.x_op + verticesz[k]
                    
                    e = self.nelem[1]*self.nelem[2]*i+self.nelem[2]*j+k             
                    self.xyz_elem[:,:,e] = np.array(np.meshgrid(y_elem, x_elem, z_elem)).reshape(3, -1).T[:,[1,0,2]]
                    self.bdy_xyz[:,0,0,e], self.bdy_xyz[:,0,1,e] = verticesx[i], verticesx[i+1]
                    self.bdy_xyz[:,1:,0,e] = self.bdy_xyz[:,1:,1,e] = np.array(np.meshgrid(z_elem, y_elem)).reshape(2, -1).T[:,[1,0]]
                    self.bdy_xyz[:,1,2,e], self.bdy_xyz[:,1,3,e] = verticesy[j], verticesy[j+1]
                    self.bdy_xyz[:,::2,2,e] = self.bdy_xyz[:,::2,3,e] = np.array(np.meshgrid(z_elem, x_elem)).reshape(2, -1).T[:,[1,0]]
                    self.bdy_xyz[:,2,4,e], self.bdy_xyz[:,2,5,e] = verticesz[k], verticesz[k+1]
                    self.bdy_xyz[:,:2,4,e] = self.bdy_xyz[:,:2,5,e] = np.array(np.meshgrid(y_elem, x_elem)).reshape(2, -1).T[:,[1,0]]

        self.xyz_elem_unwarped = np.copy(self.xyz_elem)            
        for i in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
            a = i*self.nen**3
            b = (i+1)*self.nen**3
            self.xyz[a:b,:] = self.xyz_elem[:,:,i]
        
        for i in range(self.nelem[2]+1): # planes in z
            self.grid_planes[i,0,:,:], self.grid_planes[i,1,:,:] = np.meshgrid(np.linspace(self.xmin[0],self.xmax[0],25),np.linspace(self.xmin[1],self.xmax[1],25))
            self.grid_planes[i,2,:,:] = np.ones((25,25))*verticesz[i]
        for i in range(self.nelem[1]+1): # planes in y
            j = self.nelem[2]+1 + i
            self.grid_planes[j,0,:,:], self.grid_planes[j,2,:,:] = np.meshgrid(np.linspace(self.xmin[0],self.xmax[0],25),np.linspace(self.xmin[2],self.xmax[2],25))
            self.grid_planes[j,1,:,:] = np.ones((25,25))*verticesy[i]
        for i in range(self.nelem[0]+1):  # planes in x
            j = self.nelem[2]+self.nelem[1]+1 + i
            self.grid_planes[j,1,:,:], self.grid_planes[j,2,:,:] = np.meshgrid(np.linspace(self.xmin[1],self.xmax[1],25),np.linspace(self.xmin[2],self.xmax[2],25))
            self.grid_planes[j,0,:,:] = np.ones((25,25))*verticesx[i]
            
        # create exact jacobian (dx_phys/dx_ref)
        self.jac_exa = np.zeros((self.nen**3, 3, 3, self.nelem[0]*self.nelem[1]*self.nelem[2])) # 3x3 matrix in 3D
        self.jac_exa[:,0,0,:] = elem_length_x
        self.jac_exa[:,1,1,:] = elem_length_y
        self.jac_exa[:,2,2,:] = elem_length_z
        #self.jac_inv_exa = np.zeros((self.nen**3, 3, 3, self.nelem[0]*self.nelem[1]*self.nelem[2]))
        #self.jac_inv_exa[:,0,0,:] = 1/elem_length_x
        #self.jac_inv_exa[:,1,1,:] = 1/elem_length_y
        #self.jac_inv_exa[:,2,2,:] = 1/elem_length_z
        self.det_jac_exa = np.ones((self.nen**3, self.nelem[0]*self.nelem[1]*self.nelem[2]))*(elem_length_x*elem_length_y*elem_length_z)
        #self.det_jac_inv_exa = np.ones((self.nen**3, self.nelem[0]*self.nelem[1]*self.nelem[2]))/(elem_length_x*elem_length_y*elem_length_z)

        self.bdy_jac_exa = np.zeros((self.nen**2,3,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
        self.bdy_jac_exa[:,0,0,:,:] = elem_length_x
        self.bdy_jac_exa[:,1,1,:,:] = elem_length_y      
        self.bdy_jac_exa[:,2,2,:,:] = elem_length_z
        self.bdy_det_jac_exa = np.ones((self.nen**2,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))*(elem_length_x*elem_length_y*elem_length_z)
        #self.bdy_jac_inv_exa = np.zeros((self.nen**2,3,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
        #self.bdy_jac_inv_exa[:,0,0,:,:] = 1/elem_length_x
        #self.bdy_jac_inv_exa[:,1,1,:,:] = 1/elem_length_y  
        #self.bdy_jac_inv_exa[:,2,2,:,:] = 1/elem_length_z

    def stretch_mesh_1d(self):
        '''
        Stretches a 1d mesh to test coordinate trasformations.
    
        '''
        assert self.dim == 1 , 'Stretching only set up for 1D'
        if self.warp_factor2 != 0 or self.warp_factor3 != 0:
            print('... Stretching mesh by factors of {0}, {1}, {2}'.format(self.warp_factor,self.warp_factor2,self.warp_factor3))
        else:
            print('... Stretching mesh by a factor of {0}'.format(self.warp_factor))

        
        def stretch_line(x):
            ''' Try to keep the warp_factor <0.26 '''
            #assert self.xmin==0,'Chosen warping function is only set up for domains [0,L]'
            arg = (x-self.xmin)/self.dom_len
            new_x = x + self.warp_factor*self.dom_len*np.exp(1-arg)*np.sin(np.pi*arg)
            return new_x
        
        def stretch_line_der(x):
            ''' the derivative of the function stretch_line wrt x (i.e. dnew_x/dx) '''
            arg = (x-self.xmin)/self.dom_len
            der = 1 + self.warp_factor*(np.pi*np.exp(1-arg)*np.cos(np.pi*arg) - np.exp(1-arg)*np.sin(np.pi*arg))
            return der
        
        def stretch_line_quad(x):
            ''' Stretch a line according to a quadratic: '''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            a = self.warp_factor/(self.xmax-self.xmin)
            new_x = a*x**2 + (1-a*(self.xmin+self.xmax))*x + (a*self.xmin*self.xmax)
            return new_x
        
        def stretch_line_quad_der(x):
            ''' the derivative of the function stretch_line_quad wrt x (i.e. dnew_x/dx) '''
            a = self.warp_factor/(self.xmax-self.xmin)
            der = 2*a*x + (1-a*(self.xmin+self.xmax))
            return der   

        def stretch_sigmoid(x):
            ''' a normalized tunable sigmoid function ''' 
            assert(self.warp_factor>-1),'Invalid warp_factor. Use a value >-1'
            arg = (x-self.xmin)/self.dom_len
            new_x = self.dom_len*((1+self.warp_factor)*(arg-0.5)/(1+self.warp_factor*np.abs(2*arg-1)) + 0.5) + self.xmin
            return new_x
        
        def stretch_sigmoid_der(x):
            ''' derivative of the symmetric regularized incomplete beta function '''
            arg = (x-self.xmin)/self.dom_len
            der = (1+self.warp_factor)/((1+self.warp_factor*np.abs(2*arg-1))**2)
            return der
        
        def stretch_tanh(x):
            ''' a tanh(x) function with constraints f(0)=0, f(0.5)=0.5, f(1) = 1 '''
            assert(self.warp_factor>0 and self.warp_factor<1),'Invalid warp_factor. Use a value >0, <1'
            a = 1./(1.- self.warp_factor*self.warp_factor) - 0.5
            c = - np.arctanh(1/(2.0*a))
            b = -2.0 * c
            d = -a * np.tanh(c)
            arg = (x-self.xmin)/self.dom_len
            new_x = (a * np.tanh(b * arg + c) + d)*self.dom_len + self.xmin
            return new_x
    
        def stretch_tanh_der(x):
            assert(self.warp_factor>0 and self.warp_factor<1),'Invalid warp_factor. Use a value >0, <1'
            a = 1./(1.- self.warp_factor*self.warp_factor) - 0.5
            c = - np.arctanh(1/(2.0*a))
            b = -2.0 * c
            d = -a * np.tanh(c)
            arg = (x-self.xmin)/self.dom_len
            der = a * b / np.cosh(b * arg + c)**2
            return der
        
        def stretch_corners(x):
            ''' stretches the corners but keeps the middle linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            a = self.warp_factor
            b = self.warp_factor2
            xk = self.warp_factor3
            #assert(a>=0),'warp_factor must be >0, or >1 to squish at boundaries'
            if a<0:
                print('WARNING: warp_factor1 below allowed bound 1, capping manually.')
                a = 0
            if a>1E16:
                print('WARNING: warp_factor1 above 1E16, capping manually.')
                a = 1E16
            #assert(xk >= 0 and xk <= 0.5),'warp_factor3 must be between 0 and 0.5'
            if xk>0.5:
                print('WARNING: warp_factor3 above allowed bound 0.5, capping manually.')
                xk = 0.5
            if xk<0:
                print('WARNING: warp_factor2 below allowed bound 0, capping manually.')
                xk = 0.
            #assert(b <= xk**(1-a)/(a*(1-2*xk)+2*xk) and b >= 0),'warp_factor2={0} outside allowed range [0,{1}] \n warp_factors were {2}, {3}, {4}'.format(b, xk**(1-a)/(a*(1-2*xk)+2*xk), a, b, xk)
            with redirect_stderr(None):
                fac = xk**(1-a)
            if fac == np.inf:
                print('WARNING: exponent (1-warpfactor1) too small, capping manually.')
                fac = 1E16
            if b > fac/(a*(1-2*xk)+2*xk):
                print('WARNING: warp_factor2 above allowed bound, capping manually.')
                b = fac/(a*(1-2*xk)+2*xk) - 1E-10
            if b < 0:
                print('WARNING: warp_factor2 below allowed bound, capping manually.')
                b = 0.
            c1 = -a*b*xk**(a-1) + 1 + 2*(a-1)*b*xk**a
            c2 = (a-1)*b*xk**a + 0.5
            f = np.where(x<=xk, b*x**a+c1*x,0)
            f = np.where(((xk<=x) & (x<=1-xk)), c2*(2*x-1)+0.5,f)
            f = np.where(1-xk<=x, 1-b*(1-x)**a-c1*(1-x),f)
            return f
        
        def stretch_corners_der(x):
            ''' stretches the corners but keeps the middle linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            a = self.warp_factor
            b = self.warp_factor2
            xk = self.warp_factor3
            #assert(a>=0),'warp_factor must be >0, or >1 to squish at boundaries'
            if a<0:
                print('WARNING: warp_factor1 below allowed bound 1, capping manually.')
                a = 0
            if a>1E16:
                print('WARNING: warp_factor1 above 1E16, capping manually.')
                a = 1E16
            #assert(xk >= 0 and xk <= 0.5),'warp_factor3 must be between 0 and 0.5'
            if xk>0.5:
                print('WARNING: warp_factor3 above allowed bound 0.5, capping manually.')
                xk = 0.5
            if xk<0:
                print('WARNING: warp_factor2 below allowed bound 0, capping manually.')
                xk = 0.
            #assert(b <= xk**(1-a)/(a*(1-2*xk)+2*xk) and b >= 0),'warp_factor2={0} outside allowed range [0,{1}] \n warp_factors were {2}, {3}, {4}'.format(b, xk**(1-a)/(a*(1-2*xk)+2*xk), a, b, xk)
            with redirect_stderr(None):
                fac = xk**(1-a)
            if fac == np.inf:
                print('WARNING: exponent (1-warpfactor1) too small, capping manually.')
                fac = 1E16
            if b > fac/(a*(1-2*xk)+2*xk):
                print('WARNING: warp_factor2 above allowed bound, capping manually.')
                b = fac/(a*(1-2*xk)+2*xk) - 1E-10
            if b < 0:
                print('WARNING: warp_factor2 below allowed bound, capping manually.')
                b = 0.
            c1 = -a*b*xk**(a-1) + 1 + 2*(a-1)*b*xk**a
            c2 = (a-1)*b*xk**a + 0.5
            df = np.where(x<=xk, a*b*x**(a-1)+c1,0)
            df = np.where(((xk<=x) & (x<=1-xk)), c2*2,df)
            df = np.where(1-xk<=x, a*b*(1-x)**(a-1)+c1,df)
            return df
        
        def corners2(x):
            ''' stretches the corners but keeps the ends and middle linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            x0 = self.warp_factor # how far in to start exponential
            x1 = self.warp_factor2 # for far in to stop exponential
            c = self.warp_factor3 # strength of the exponential

            e0 = np.exp(c*x0)
            e1 = np.exp(c*x1)
            
            k = 1/(2*(e1-e0) + 2*c*(e0*x0-e1*x1) + c*e1)
            a0 = k*c*e0
            a1 = k*c*e1
            d = k*e0*(c*x0-1)

            f = np.where(x<=x0, a0*x,0)
            f = np.where(((x0<=x) & (x<=x1)), k*np.exp(c*x)+d,f)
            f = np.where(((x1<=x) & (x<=(1-x1))), a1*x+0.5*(1-a1),f)
            f = np.where((((1-x1)<=x) & (x<=(1-x0))), -k*np.exp(c*(1-x))-d+1,f)
            f = np.where((1-x0)<=x, a0*x+(1-a0),f)
            return f
        
        def corners2_der(x):
            ''' stretches the corners but keeps the ends and middle linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            x0 = self.warp_factor # how far in to start exponential
            x1 = self.warp_factor2 # for far in to stop exponential
            c = self.warp_factor3 # strength of the exponential

            e0 = np.exp(c*x0)
            e1 = np.exp(c*x1)
            
            k = 1/(2*(e1-e0) + 2*c*(e0*x0-e1*x1) + c*e1)
            a0 = k*c*e0
            a1 = k*c*e1
            d = k*e0*(c*x0-1)

            df = np.where(x<=x0, a0,0)
            df = np.where(((x0<=x) & (x<=x1)), c*k*np.exp(c*x),df)
            df = np.where(((x1<=x) & (x<=(1-x1))), a1,df)
            df = np.where((((1-x1)<=x) & (x<=(1-x0))), c*k*np.exp(c*(1-x)),df)
            df = np.where((1-x0)<=x, a0,df)
            return df
        
        def corners_periodic(x):
            ''' stretches the corners but keeps the ends and middle approximately linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            x0 = self.warp_factor # how far in to have transition
            a = self.warp_factor2 # strength of the transition
            N = max(3,int(self.warp_factor3)) # how many modes to use

            f = np.copy(x)
            for n in range(1,N+1):
                c = 2*(1-x0)/(1-2*x0)*np.sin(2*np.pi*n*x0)/(np.pi*n)
                f += a*c*np.sin(2*np.pi*n*x)/(2*np.pi*n)
            return f
        
        def corners_periodic_der(x):
            ''' stretches the corners but keeps the ends and middle approximately linear'''
            assert(self.xmin==0 and self.xmax==1),'Only set up for interval [0,1]'
            x0 = self.warp_factor # how far in to have transition
            a = self.warp_factor2 # strength of the transition
            N = max(3,int(self.warp_factor3)) # how many modes to use

            df = np.ones_like(x)
            for n in range(1,N+1):
                c = 2*(1-x0)/(1-2*x0)*np.sin(2*np.pi*n*x0)/(np.pi*n)
                df += a*c*np.cos(2*np.pi*n*x)
            return df

        
        # switch between different mappings here
        if self.warp_type == 'default' or self.warp_type == 'papers':
            warp_fun = stretch_line
            warp_der = stretch_line_der 
        elif self.warp_type == 'quad':
            warp_fun = stretch_line_quad
            warp_der = stretch_line_quad_der 
        elif self.warp_type == 'sigmoid':
            warp_fun = stretch_sigmoid
            warp_der = stretch_sigmoid_der 
        elif self.warp_type == 'corners':
            warp_fun = stretch_corners
            warp_der = stretch_corners_der 
        elif self.warp_type == 'corners2':
            warp_fun = corners2
            warp_der = corners2_der 
        elif self.warp_type == 'corners_periodic':
            warp_fun = corners_periodic
            warp_der = corners_periodic_der
        elif self.warp_type == 'tanh':
            warp_fun = stretch_tanh
            warp_der = stretch_tanh_der 
        else:
            print('WARNING: mesh.warp_type '+self.warp_type+' not understood. Reverting to default.')
            warp_fun = stretch_line
            warp_der = stretch_line_der        
        
        x_elem_old = np.copy(self.x_elem)
        bdy_x_old = np.copy(self.bdy_x)
        self.x_elem = warp_fun(self.x_elem)
        self.vertices = warp_fun(self.vertices)
        self.x = self.x_elem.flatten('F')
        self.bdy_x = warp_fun(self.bdy_x)
        
        self.jac_exa[:,0,:] *= warp_der(x_elem_old) # chain rule with original transformation
        assert np.all(self.jac_exa>0),"Not a valid coordinate transformation. Try using lower warp_factors than {0}".format((self.warp_factor,self.warp_factor2,self.warp_factor3))
        #self.jac_inv_exa = 1/self.jac_exa
        self.det_jac_exa = self.jac_exa[:,0,:]
        #self.det_jac_inv_exa = 1/self.det_jac_exa
        
        self.bdy_jac_exa *= warp_der(bdy_x_old)
        #self.bdy_det_jac_exa = self.bdy_jac_exa[0,:,:]
        #self.bdy_jac_inv_exa = 1/self.bdy_jac_exa
        
            
    def warp_mesh_2d(self, xy=None):
        '''
        Warps a rectangular mesh to test curvlinear coordinate trasformations.
    
        '''
        assert self.dim == 2 , 'Warping only set up for 2D'
        print('... Warping mesh by a factor of {0}'.format(self.warp_factor))
        
        def warp_rectangle(x,y):
            ''' Based on function from Tristan's Paper. Warps a rectangular mesh
            Try to keep the warp.factor <0.24 ''' 
            assert self.warp_factor<0.24,'Try a warp_factor < 0.24 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            new_x = x + self.warp_factor*self.dom_len[0]*np.sin(np.pi*argx)*np.sin(np.pi*argy)
            new_y = y + self.warp_factor*self.dom_len[1]*np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy) 
            return new_x , new_y

        def warp_rectangle_der(x,y):
            ''' the derivative of the function warp_rectangle wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
            dxdy = self.warp_factor*self.dom_len[0]*np.pi*np.sin(np.pi*argx)*np.cos(np.pi*argy)/self.dom_len[1]
            dydx = self.warp_factor*self.dom_len[1]*np.pi*np.exp(1-argy)*np.cos(np.pi*argx)*np.sin(np.pi*argy)/self.dom_len[0]
            dydy = 1 + self.warp_factor*(np.pi*np.exp(1-argy)*np.sin(np.pi*argx)*np.cos(np.pi*argy) - np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy))
            return dxdx, dxdy, dydx, dydy
        
        def warp_dcp(x,y):
            ''' function from DCP - pretty much the same as above''' 
            #assert self.warp_factor<0.24,'Try a warp_factor < 0.24 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            x_t = argx + self.warp_factor*np.sin(np.pi*argx)*np.sin(np.pi*argy)
            y_t = argy + self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx-0.75)*np.sin(np.pi*argy) 
            new_x = x_t*self.dom_len[0] + self.xmin[0]
            new_y = y_t*self.dom_len[1] + self.xmin[1]
            return new_x , new_y

        def warp_dcp_der(x,y):
            ''' the derivative of the function warp_dcp wrt x (i.e. dnew_x/dx) 
            Note: I think this is wrong, does not take into acount dargx/dy ? '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
            dxdy = self.warp_factor*np.sin(np.pi*argx)*np.cos(np.pi*argy)*np.pi
            dydx = -self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx - 0.75)*np.sin(np.pi*argy) + self.warp_factor*np.exp(1-argx)*np.cos(np.pi*argx - 0.75)*np.sin(np.pi*argy)*np.pi
            dydy = 1 + self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx - 0.75)*np.cos(np.pi*argy)*np.pi
            return dxdx, dxdy, dydx, dydy

        def warp_metrics_paper(x,y):
            ''' Based on function from Tristan's Paper. Warps a rectangular mesh
            Try to keep the warp.factor <0.24 ''' 
            #assert self.warp_factor<0.24,'Try a warp_factor < 0.24 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            new_x = x + self.warp_factor*self.dom_len[0]*np.sin(np.pi*argx)*np.sin(np.pi*argy)
            new_y = y + 0.5*self.warp_factor*self.dom_len[1]*np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy) 
            return new_x , new_y

        def warp_metrics_paper_der(x,y):
            ''' the derivative of the function warp_metrics_paper wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
            dxdy = self.warp_factor*self.dom_len[0]*np.pi*np.sin(np.pi*argx)*np.cos(np.pi*argy)/self.dom_len[1]
            dydx = 0.5*self.warp_factor*self.dom_len[1]*np.pi*np.exp(1-argy)*np.cos(np.pi*argx)*np.sin(np.pi*argy)/self.dom_len[0]
            dydy = 1 + 0.5*self.warp_factor*(np.pi*np.exp(1-argy)*np.sin(np.pi*argx)*np.cos(np.pi*argy) - np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy))
            return dxdx, dxdy, dydx, dydy
        
        def warp_bdy(x,y):
            ''' my modified function in Diablo ''' 
            #assert self.warp_factor<=1,'Try a warp_factor < 1 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            new_x = x + self.warp_factor*self.dom_len[0]*(0.03*np.sin(3*np.pi*argy) + 0.08*np.sin(np.pi*argx)*np.sin(np.pi*argy)*np.exp(1-argy) - 0.015*np.sin(6*np.pi*argy)*np.sin(8*np.pi*argx))
            new_y = y + self.warp_factor*self.dom_len[1]*(0.04*np.sin(2*np.pi*argx) - 0.03*np.sin(5*np.pi*argy)*np.sin(4*np.pi*argx) - 0.015*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argx))
            return new_x , new_y

        def warp_bdy_der(x,y):
            ''' the derivative of the function warp_bdy wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*(np.pi*0.08*np.cos(np.pi*argx)*np.sin(np.pi*argy)*np.exp(1-argy) - 8*np.pi*0.015*np.sin(6*np.pi*argy)*np.cos(8*np.pi*argx))
            dxdy = self.warp_factor*self.dom_len[0]*(3*np.pi*0.03*np.cos(3*np.pi*argy) + np.pi*0.08*np.sin(np.pi*argx)*np.cos(np.pi*argy)*np.exp(1-argy) - 0.08*np.sin(np.pi*argx)*np.sin(np.pi*argy)*np.exp(1-argy) - 6*np.pi*0.015*np.cos(6*np.pi*argy)*np.sin(8*np.pi*argx))/self.dom_len[1]
            dydx = self.warp_factor*self.dom_len[1]*(2*np.pi*0.04*np.cos(2*np.pi*argx) - 4*np.pi*0.03*np.sin(5*np.pi*argy)*np.cos(4*np.pi*argx) - 8*np.pi*0.015*np.sin(7*np.pi*argy)*np.cos(8*np.pi*argx))/self.dom_len[0]
            dydy = 1 + self.warp_factor*(-5*np.pi*0.03*np.cos(5*np.pi*argy)*np.sin(4*np.pi*argx) - 7*np.pi*0.015*np.cos(7*np.pi*argy)*np.sin(8*np.pi*argx))
            return dxdx, dxdy, dydx, dydy
        
        def warp_quad(x,y):
            ''' Warps according to a quadratic (in each direction. Total order is quartic). '''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            warp = 2*self.warp_factor
            xscale = (x-self.xmin[0])/(self.xmax[0]-self.xmin[0])
            yscale = (y-self.xmin[1])/(self.xmax[1]-self.xmin[1])
            new_x = warp*self.dom_len[0]*(yscale**2-yscale)*(xscale**2-xscale) + x
            new_y = 2*warp*self.dom_len[1]*(xscale**2-xscale)*(yscale**2-yscale) + y
            return new_x , new_y

        def warp_quad_der(x,y):
            ''' the derivative of the function warp_quad wrt x (i.e. dnew_x/dx) '''
            xscale = (x-self.xmin[0])/self.dom_len[0]
            yscale = (y-self.xmin[1])/self.dom_len[1]
            warp = 2*self.warp_factor
            dxdx = warp*(yscale**2-yscale)*(2*xscale-1) + 1
            dxdy = warp*self.dom_len[0]*(2*yscale-1)*(xscale**2-xscale)/self.dom_len[1]
            dydx = 2*warp*self.dom_len[1]*(2*xscale-1)*(yscale**2-yscale)/self.dom_len[0]
            dydy = 2*warp*(xscale**2-xscale)*(2*yscale-1) + 1
            return dxdx, dxdy, dydx, dydy
        
        def warp_1D_quad(x,y):
            ''' Warps according to a quadratic. Note that warp_factor irrelevant '''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            #a = self.warp_factor/(self.xmax[0]-self.xmin[0])
            a = self.warp_factor/(self.xmax[1]-self.xmin[1])
            #new_x = a*x**2 + (1-a*(self.xmin[0]+self.xmax[0]))*x + (a*self.xmin[0]*self.xmax[0])
            new_y = a*y**2 + (1-a*(self.xmin[1]+self.xmax[1]))*y + (a*self.xmin[1]*self.xmax[1])
            #return new_x , y
            return x , new_y

        def warp_1D_quad_der(x,y):
            ''' the derivative of the function warp_rectangle_quad wrt x (i.e. dnew_x/dx) '''
            #a = self.warp_factor/(self.xmax[0]-self.xmin[0])
            a = self.warp_factor/(self.xmax[1]-self.xmin[1])
            #dxdx = 2*a*x + (1-a*(self.xmin[0]+self.xmax[0]))
            dxdx = np.ones(x.shape)
            dxdy = np.zeros(x.shape)
            dydx = np.zeros(y.shape)
            #dydy = np.ones(y.shape)
            dydy = 2*a*y + (1-a*(self.xmin[1]+self.xmax[1]))
            return dxdx, dxdy, dydx, dydy
        
        def warp_cubic(x,y):
            ''' Warps according to a cubic (in each direction. Total order is 6).'''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            warp = self.warp_factor*6.5
            xscale = (x-self.xmin[0])/(self.xmax[0]-self.xmin[0])
            yscale = (y-self.xmin[1])/(self.xmax[1]-self.xmin[1])
            new_x = warp*self.dom_len[0]*(yscale**3-1.7*yscale**2+0.7*yscale)*(xscale**3-xscale) + x
            new_y = 1.5*warp*self.dom_len[1]*(xscale**3-1.2*xscale**2+0.2*xscale)*(yscale**3-yscale**2) + y
            return new_x , new_y

        def warp_cubic_der(x,y):
            ''' the derivative of the function warp_cubic wrt x (i.e. dnew_x/dx) '''
            warp = self.warp_factor*6.5
            xscale = (x-self.xmin[0])/self.dom_len[0]
            yscale = (y-self.xmin[1])/self.dom_len[1]
            dxdx = warp*(yscale**3-1.7*yscale**2+0.7*yscale)*(3*xscale**2-1) + 1
            dxdy = warp*self.dom_len[0]*(3*yscale**2-2*1.7*yscale+0.7)*(xscale**3-xscale)/self.dom_len[1]
            dydx = 1.5*warp*self.dom_len[1]*(3*xscale**2-2*1.2*xscale+0.2)*(yscale**3-yscale**2)/self.dom_len[0]
            dydy = 1.5*warp*(xscale**3-1.2*xscale**2+0.2*xscale)*(3*yscale**2-2*yscale) + 1
            return dxdx, dxdy, dydx, dydy
        
        def warp_smooth(x,y):
            ''' Based on function from Tristan's Paper, but modified to be smooth
            across periodic boundaries. Warps a rectangular mesh. keep warp.factor <0.15 '''
            assert self.warp_factor<0.15,'Try a warp_factor < 0.15 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            new_x = x + self.warp_factor*self.dom_len[0]*np.sin(2*np.pi*argx)*np.sin(2*np.pi*argy)
            new_y = y + self.warp_factor*self.dom_len[1]*np.sin(2*np.pi*argx)*np.sin(2*np.pi*argy) 
            return new_x , new_y

        def warp_smooth_der(x,y):
            ''' the derivative of the function warp_smooth wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*2*np.pi*np.cos(2*np.pi*argx)*np.sin(2*np.pi*argy)
            dxdy = self.warp_factor*self.dom_len[0]*2*np.pi*np.sin(2*np.pi*argx)*np.cos(2*np.pi*argy)/self.dom_len[1]
            dydx = self.warp_factor*self.dom_len[1]*2*np.pi*np.cos(2*np.pi*argx)*np.sin(2*np.pi*argy)/self.dom_len[0]
            dydy = 1 + self.warp_factor*2*np.pi*np.sin(2*np.pi*argx)*np.cos(2*np.pi*argy)
            return dxdx, dxdy, dydx, dydy
        
        def warp_skew(x,y):
            ''' A function to purposely have near-zero jacobians '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            b1 = 0.01
            b2 = 0.02
            c1 = 2.4
            c2 = 2.3
            new_x = x + b1*(np.exp(c1*argy) - 1.)
            new_y = y + b2*(np.exp(c2*argx) - 1.)
            return new_x, new_y
        
        def warp_skew_der(x,y):
            ''' A function to purposely have near-zero jacobians '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            b1 = 0.01
            b2 = 0.02
            c1 = 2.4
            c2 = 2.3
            dxdx = np.ones(np.shape(argx))
            dxdy = b1*np.exp(c1*argy)*c1*argy/self.dom_len[1]
            dydx = b2*np.exp(c2*argx)*c2*argx/self.dom_len[0]
            dydy = np.ones(np.shape(argx))
            return dxdx, dxdy, dydx, dydy
        
        def warp_chan(x,y):
            ''' 2d version of the transformation in chan and wilcox '''
            ''' they used warp = 0.125 '''
            argx = 2*(x-self.xmin[0])/self.dom_len[0] - 1
            argy = 2*(y-self.xmin[1])/self.dom_len[1] - 1
            new_x = x + self.warp_factor*self.dom_len[0]*np.cos(np.pi*argx/2)*np.cos(np.pi*argy/2)
            new_y = y + self.warp_factor*self.dom_len[1]*np.cos(np.pi*argx/2)*np.cos(np.pi*argy/2)
            return new_x,new_y
            
        def warp_chan_der(x,y):
            ''' 2d version of the transformation in chan and wilcox '''
            ''' they used warp = 0.125 '''
            argx = 2*(x-self.xmin[0])/self.dom_len[0] - 1
            argy = 2*(y-self.xmin[1])/self.dom_len[1] - 1
            dxdx = 1.0 - np.pi*self.warp_factor*np.sin(np.pi*argx/2)*np.cos(np.pi*argy/2)
            dxdy = - np.pi*self.warp_factor*self.dom_len[0]*np.cos(np.pi*argx/2)*np.sin(np.pi*argy/2)/self.dom_len[1]
            dydx = - np.pi*self.warp_factor*self.dom_len[1]*np.sin(np.pi*argx/2)*np.cos(np.pi*argy/2)/self.dom_len[0]
            dydy = 1.0 - np.pi*self.warp_factor*np.cos(np.pi*argx/2)*np.sin(np.pi*argy/2)
            return dxdx, dxdy, dydx, dydy
        
        # switch between different mappings here
        if self.warp_type == 'default' or self.warp_type == 'papers':
            warp_fun = warp_rectangle
            warp_der = warp_rectangle_der 
        elif self.warp_type == 'metrics_paper':
            warp_fun = warp_metrics_paper
            warp_der = warp_metrics_paper_der
        elif self.warp_type == 'quad':
            warp_fun = warp_quad
            warp_der = warp_quad_der 
        elif self.warp_type == '1Dquad':
            warp_fun = warp_1D_quad
            warp_der = warp_1D_quad_der 
        elif self.warp_type == 'smooth':
            warp_fun = warp_smooth
            warp_der = warp_smooth_der
        elif self.warp_type == 'cubic':
            warp_fun = warp_cubic
            warp_der = warp_cubic_der
        elif self.warp_type == 'dcp':
            warp_fun = warp_dcp
            warp_der = warp_dcp_der
        elif self.warp_type == 'bdy' or self.warp_type == 'strong':
            warp_fun = warp_bdy
            warp_der = warp_bdy_der
        elif self.warp_type == 'skew':
            warp_fun = warp_skew
            warp_der = warp_skew_der
        elif self.warp_type == 'chan':
            warp_fun = warp_chan
            warp_der = warp_chan_der
        else:
            print('WARNING: mesh.warp_type not understood. Reverting to default.')
            warp_fun = warp_rectangle
            warp_der = warp_rectangle_der 

        if xy is not None:
            xy_new = np.copy(xy)
            xy_new[:,0,:], xy_new[:,1,:] = warp_fun(xy[:,0,:], xy[:,1,:])
            return xy_new
        else:
            
            xy_elem_old = np.copy(self.xy_elem)
            bdy_xy_old = np.copy(self.bdy_xy)
            self.xy_elem[:,0,:], self.xy_elem[:,1,:] = warp_fun(self.xy_elem[:,0,:], self.xy_elem[:,1,:])
            self.bdy_xy[:,0,:,:], self.bdy_xy[:,1,:,:] = warp_fun(self.bdy_xy[:,0,:,:],self.bdy_xy[:,1,:,:])
            self.grid_lines[:,0,:], self.grid_lines[:,1,:] = warp_fun(self.grid_lines[:,0,:], self.grid_lines[:,1,:])
            
            for i in range(self.nelem[0]*self.nelem[1]):
                a = i*self.nen**2
                b = (i+1)*self.nen**2
                self.xy[a:b,:] = self.xy_elem[:,:,i]
            
            dxnewdx, dxnewdy, dynewdx, dynewdy = warp_der(xy_elem_old[:,0,:], xy_elem_old[:,1,:])
            dxdxref = np.copy(self.jac_exa[:,0,0,:])
            dydyref = np.copy(self.jac_exa[:,1,1,:])
            # chain rule, ignoring cross terms that are 0 in original transformation
            self.jac_exa[:,0,0,:] = dxnewdx * dxdxref 
            self.jac_exa[:,0,1,:] = dxnewdy * dydyref
            self.jac_exa[:,1,0,:] = dynewdx * dxdxref
            self.jac_exa[:,1,1,:] = dynewdy * dydyref
            for elem in range(self.nelem[0]*self.nelem[1]):
                self.det_jac_exa[:,elem] = np.linalg.det(self.jac_exa[:,:,:,elem])
            assert np.all(self.det_jac_exa>0),"Not a valid coordinate transformation. Try using a lower warp_factor."
            #for elem in range(self.nelem[0]*self.nelem[1]):
            #    self.jac_inv_exa[:,:,:,elem] = np.linalg.inv(self.jac_exa[:,:,:,elem])
            #    self.det_jac_inv_exa[:,elem] =  np.linalg.det(self.jac_inv_exa[:,:,:,elem])
            #if np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa) > 1e-12):
            #    print('WANRING: Numerical error in calculation of determinant inverse is {0:.2g}'.format(np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa))))
            
            dxnewdx, dxnewdy, dynewdx, dynewdy = warp_der(bdy_xy_old[:,0,:,:], bdy_xy_old[:,1,:,:])
            dxdxref = np.copy(self.bdy_jac_exa[:,0,0,:,:])
            dydyref = np.copy(self.bdy_jac_exa[:,1,1,:,:])
            self.bdy_jac_exa[:,0,0,:,:] = dxnewdx * dxdxref # chain rule, ignoring cross terms that are 0 in original transformation
            self.bdy_jac_exa[:,0,1,:,:] = dxnewdy * dydyref
            self.bdy_jac_exa[:,1,0,:,:] = dynewdx * dxdxref
            self.bdy_jac_exa[:,1,1,:,:] = dynewdy * dydyref
            self.bdy_det_jac_exa = self.bdy_jac_exa[:,0,0,:,:] * self.bdy_jac_exa[:,1,1,:,:] \
                                    - self.bdy_jac_exa[:,0,1,:,:] * self.bdy_jac_exa[:,1,0,:,:]
            assert np.all(self.bdy_det_jac_exa>0),"Not a valid coordinate transformation. Try using a lower warp_factor."
            #for elem in range(self.nelem[0]*self.nelem[1]):
                #for i in range(4):
                    #self.bdy_det_jac_exa[:,i,elem] = np.linalg.det(self.bdy_jac_exa[:,:,:,i,elem])
                    #self.bdy_jac_inv_exa[:,:,:,i,elem] = np.linalg.inv(self.bdy_jac_exa[:,:,:,i,elem])

                
        ''' Uncomment the below section to debug and test the consistency of the warping '''
# =============================================================================
#         max_xy = [0,0]
#         max_der = [0,0,0,0,0,0,0,0]
#         
#         for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
#             diff = abs(self.bdy_xy[:,1,0,row::self.nelem[1]][:,0] - self.bdy_xy[:,1,1,row::self.nelem[1]][:,-1])
#             max_xy[0] = max(max_xy[0],np.max(diff))
#             diff = abs(dxnewdx[:,0,row::self.nelem[1]][:,0] - dxnewdx[:,1,row::self.nelem[1]][:,-1])
#             max_der[0] = max(max_der[0],np.max(diff))
#             diff = abs(dxnewdy[:,0,row::self.nelem[1]][:,0] - dxnewdy[:,1,row::self.nelem[1]][:,-1])
#             max_der[1] = max(max_der[1],np.max(diff))
#             diff = abs(dynewdx[:,0,row::self.nelem[1]][:,0] - dynewdx[:,1,row::self.nelem[1]][:,-1])
#             max_der[2] = max(max_der[2],np.max(diff))
#             diff = abs(dynewdy[:,0,row::self.nelem[1]][:,0] - dynewdy[:,1,row::self.nelem[1]][:,-1])
#             max_der[3] = max(max_der[3],np.max(diff))
#             
#             
#         for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
#             start = col*self.nelem[0]
#             end = start + self.nelem[1]
#             diff = abs(self.bdy_xy[:,0,2,start:end][:,0] - self.bdy_xy[:,0,3,start:end][:,-1])
#             max_xy[1] = max(max_xy[1],np.max(diff))
#             diff = abs(dxnewdx[:,2,start:end][:,0] - dxnewdx[:,3,start:end][:,-1])
#             max_der[4] = max(max_der[4],np.max(diff))
#             diff = abs(dxnewdy[:,2,start:end][:,0] - dxnewdy[:,3,start:end][:,-1])
#             max_der[5] = max(max_der[5],np.max(diff))
#             diff = abs(dynewdx[:,2,start:end][:,0] - dynewdx[:,3,start:end][:,-1])
#             max_der[6] = max(max_der[6],np.max(diff))
#             diff = abs(dynewdy[:,2,start:end][:,0] - dynewdy[:,3,start:end][:,-1])
#             max_der[7] = max(max_der[7],np.max(diff))
# 
#         print('Max diff y values along x (Left/Right) boundary: ', max_xy[0])
#         print('Max diff x values along y (Lower/Upper) boundary: ', max_xy[1])
#         #print('Max diff of dxnew(1)/dx(1) along x bdy (not nec. = 0): ', max_der[0])
#         print('Max diff of dxnew(1)/dx(2) along x bdy: ', max_der[1])
#         #print('Max diff of dxnew(2)/dx(1) along x bdy (not nec. = 0): ', max_der[2])
#         print('Max diff of dxnew(2)/dx(2) along x bdy: ', max_der[3])
#         print('Max diff of dxnew(1)/dx(1) along y bdy: ', max_der[4])
#         #print('Max diff of dxnew(1)/dx(2) along y bdy (not nec. = 0): ', max_der[5])
#         print('Max diff of dxnew(2)/dx(1) along y bdy: ', max_der[6])
#         #print('Max diff of dxnew(2)/dx(2) along y bdy (not nec. = 0): ', max_der[7])
# =============================================================================
        
            
    def warp_mesh_3d(self, xyz=None):
        '''
        Warps a cuboid mesh to test curvlinear coordinate trasformations.
    
        '''
        assert self.dim == 3 , 'Warping only set up for 3D'
        print('... Warping mesh by a factor of {0}'.format(self.warp_factor))
        
        def warp_cuboid(x,y,z):
            ''' Based on function from 2019 DDRF Paper. Warps a cuboid mesh
            Try to keep the warp.factor <0.24 ''' 
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]   
            new_x = x + self.warp_factor*self.dom_len[0]*np.sin(np.pi*argx)*np.sin(np.pi*argy)
            new_y = y + self.warp_factor*self.dom_len[1]*np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy) 
            new_z = z + 0.25*self.warp_factor*self.dom_len[2]*(np.sin(2*np.pi*(new_x-self.xmin[0])/self.dom_len[0])+np.sin(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1]))
            return new_x, new_y, new_z
        
        def warp_cuboid_der(x,y,z):
            ''' the derivative of the function warp_cuboid wrt x (i.e. dnew_x/dx) '''   
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1] 
            new_x, new_y, new_z, = warp_cuboid(x,y,z)
            dxdx = 1 + self.warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
            dxdy = self.warp_factor*self.dom_len[0]*np.pi*np.sin(np.pi*argx)*np.cos(np.pi*argy)/self.dom_len[1]
            dxdz = np.zeros(np.shape(dxdx))
            dydx = self.warp_factor*self.dom_len[1]*np.pi*np.exp(1-argy)*np.cos(np.pi*argx)*np.sin(np.pi*argy)/self.dom_len[0]
            dydy = 1 + self.warp_factor*(np.pi*np.exp(1-argy)*np.sin(np.pi*argx)*np.cos(np.pi*argy) - np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy))
            dydz = np.zeros(np.shape(dxdx))
            dzdx = 0.25*self.warp_factor*self.dom_len[2]*2*np.pi*(np.cos(2*np.pi*(new_x-self.xmin[0])/self.dom_len[0])*dxdx/self.dom_len[0] + np.cos(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1])*dydx/self.dom_len[1])
            dzdy = 0.25*self.warp_factor*self.dom_len[2]*2*np.pi*(np.cos(2*np.pi*(new_x-self.xmin[0])/self.dom_len[0])*dxdy/self.dom_len[0] + np.cos(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1])*dydy/self.dom_len[1])
            dzdz = np.ones(np.shape(dxdx))
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
        
        def warp_cuboid_flipped(x,y,z):
            ''' Based on function from 2019 DDRF Paper, but x changed to y, y to z, z to x. Warps a cuboid mesh
            Try to keep the warp.factor <0.24 ''' 
            argy = (y-self.xmin[1])/self.dom_len[1] 
            argz = (y-self.xmin[2])/self.dom_len[2]
            new_y = y + self.warp_factor*self.dom_len[1]*np.sin(np.pi*argy)*np.sin(np.pi*argz)
            new_z = z + self.warp_factor*self.dom_len[2]*np.exp(1-argz)*np.sin(np.pi*argy)*np.sin(np.pi*argz) 
            new_x = x + 0.25*self.warp_factor*self.dom_len[0]*(np.sin(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1])+np.sin(2*np.pi*(new_z-self.xmin[2])/self.dom_len[2]))
            return new_x , new_y, new_z
        
        def warp_cuboid_flipped_der(x,y,z):
            print('WARNING: I think there is a mistake here?')
            ''' the derivative of the function warp_cuboid wrt x (i.e. dnew_x/dx) '''   
            argy = (y-self.xmin[1])/self.dom_len[1]
            argz = (z-self.xmin[2])/self.dom_len[2] 
            new_x, new_y, new_z, = warp_cuboid(x,y,z)
            dydy = 1 + self.warp_factor*np.pi*np.cos(np.pi*argy)*np.sin(np.pi*argz)
            dydz = self.warp_factor*self.dom_len[1]*np.pi*np.sin(np.pi*argy)*np.cos(np.pi*argz)/self.dom_len[2]
            dydx = np.zeros(np.shape(dydy))
            dzdy = self.warp_factor*self.dom_len[2]*np.pi*np.exp(1-argz)*np.cos(np.pi*argy)*np.sin(np.pi*argz)/self.dom_len[1]
            dzdz = 1 + self.warp_factor*(np.pi*np.exp(1-argz)*np.sin(np.pi*argy)*np.cos(np.pi*argz) - np.exp(1-argz)*np.sin(np.pi*argy)*np.sin(np.pi*argz))
            dzdx = np.zeros(np.shape(dydy))
            dxdy = 0.25*self.warp_factor*self.dom_len[0]*2*np.pi*(np.cos(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1])*dydy/self.dom_len[1] + np.cos(2*np.pi*(new_z-self.xmin[2])/self.dom_len[2])*dzdy/self.dom_len[2])
            dxdz = 0.25*self.warp_factor*self.dom_len[0]*2*np.pi*(np.cos(2*np.pi*(new_y-self.xmin[1])/self.dom_len[1])*dydz/self.dom_len[1] + np.cos(2*np.pi*(new_z-self.xmin[2])/self.dom_len[2])*dzdz/self.dom_len[2])
            dxdx = np.ones(np.shape(dydy))
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz

        def warp_asym(x,y,z):
            ''' inspired by cuboid, but removes upper triangular structure that can lead to weird cancellations. '''
            a,b,c,d,e = 0.12, 0.10, 0.10, 0.08, 0.06
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz
            # handy 1D bumps that vanish on faces (α = 0 or 1)
            Sx = np.sin(np.pi * ax)
            Sy = np.sin(np.pi * ay)
            Sz = np.sin(np.pi * az)
            # Original in-plane mode (keeps side faces fixed) + mild z-coupling gated by Sz1
            x_new = x + self.warp_factor*Lx*(Sx*Sy + a*Sz * Sx*(2*Sy - Sx))   # breaks ∂x'/∂z=0 but vanishes on all faces
            # Different in-plane shape + different z-gated coupling to avoid reusing rows/columns
            y_new = y + self.warp_factor*Ly*(np.exp(1 - ay)*Sx*Sy + b*Sz * Sy*(2*Sx - Sy))  # also vanishes on all faces
            # z' warp must vanish on ALL faces: gate by Sz1 and also by side-face bumps
            # Use integer multiples (2π, 3π, …) to keep zeros on ax, ay ∈ {0,1}
            z_new = z + 0.25*self.warp_factor*Lz*Sz*(c*np.sin(2*np.pi*ax) + d*np.sin(2*np.pi*ay) + e*Sx*Sy)  # cross-term adds variety; still zero on faces
            return x_new, y_new, z_new

        def warp_asym_der(x,y,z):
            ''' the derivative of the function warp_asym wrt x (i.e. dnew_x/dx) '''
            a,b,c,d,e = 0.12, 0.10, 0.10, 0.08, 0.06
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz
            Sx = np.sin(np.pi * ax)
            Sy = np.sin(np.pi * ay)
            Sz = np.sin(np.pi * az)
            Cx = np.cos(np.pi * ax)
            Cy = np.cos(np.pi * ay)
            Cz = np.cos(np.pi * az)
            E  = np.exp(1.0 - ay)
            pxLx = np.pi / Lx
            pxLy = np.pi / Ly
            pxLz = np.pi / Lz
            Lx_over_Ly = Lx / Ly
            Lx_over_Lz = Lx / Lz
            Ly_over_Lx = Ly / Lx
            Ly_over_Lz = Ly / Lz
            dxdx = 1.0 + self.warp_factor * (np.pi * Cx * Sy + 2.0 * a * np.pi * Sz * (Sy - Sx) * Cx)
            dxdy = self.warp_factor * (np.pi * Lx_over_Ly) * Sx * Cy * (1.0 + 2.0 * a * Sz)
            dxdz = self.warp_factor * (np.pi * Lx_over_Lz) * a * Cz * Sx * (2.0*Sy - Sx)
            dydx = self.warp_factor * (np.pi * Ly_over_Lx) * Cx * Sy * (E + 2.0 * b * Sz)
            dydy = 1.0 + self.warp_factor * (E * Sx * (-Sy + np.pi * Cy) + 2.0 * b * np.pi * Sz * (Sx - Sy) * Cy)
            dydz = self.warp_factor * (np.pi * Ly_over_Lz) * b * Cz * Sy * (2.0*Sx - Sy)
            B = c*np.sin(2.0*np.pi*ax) + d*np.sin(2.0*np.pi*ay) + e*Sx*Sy
            dzdx = 0.25 * self.warp_factor * Lz * Sz * (c * (2.0*np.pi / Lx) * np.cos(2.0*np.pi*ax) + e * (np.pi / Lx) * Cx * Sy)
            dzdy = 0.25 * self.warp_factor * Lz * Sz * (d * (2.0*np.pi / Ly) * np.cos(2.0*np.pi*ay) + e * (np.pi / Ly) * Sx * Cy)
            dzdz = 1.0 + 0.25 * self.warp_factor * np.pi * Cz * B
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz

        def warp_minimal(x, y, z, a=1.00, b=0.93, c=0.85):
            """
            Minimal cross-coupled warp that keeps every cube face fixed.
            x', y', z' all depend on (x,y,z) via the same interior bump Sx*Sy*Sz,
            with distinct coefficients (a,b,c) to avoid component-wise cancellations.
            """
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz

            Sx = np.sin(2*np.pi * ax)
            Sy = np.sin(3*np.pi * ay)
            Sz = np.sin(np.pi * az)
            B  = Sx * Sy * Sz       # zero on every face (ax, ay, or az ∈ {0,1})

            x_new = x + self.warp_factor * Lx * (a * B)
            y_new = y + self.warp_factor * Ly * (b * B)
            z_new = z + self.warp_factor * Lz * (c * B)
            return x_new, y_new, z_new

        def warp_minimal_der(x, y, z,  a=1.00, b=0.93, c=0.85):
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz
            Sx = np.sin(2*np.pi * ax);  Cx = 2*np.cos(2*np.pi * ax)
            Sy = np.sin(3*np.pi * ay);  Cy = 3*np.cos(3*np.pi * ay)
            Sz = np.sin(np.pi * az);  Cz = np.cos(np.pi * az)
            # derivatives of B = Sx*Sy*Sz
            Bx = (np.pi / Lx) * Cx * Sy * Sz
            By = (np.pi / Ly) * Sx * Cy * Sz
            Bz = (np.pi / Lz) * Sx * Sy * Cz
            wf = self.warp_factor
            dxdx = 1 + wf * Lx * a * Bx
            dxdy =      wf * Lx * a * By
            dxdz =      wf * Lx * a * Bz
            dydx =      wf * Ly * b * Bx
            dydy = 1 + wf * Ly * b * By
            dydz =      wf * Ly * b * Bz
            dzdx =      wf * Lz * c * Bx
            dzdy =      wf * Lz * c * By
            dzdz = 1 + wf * Lz * c * Bz
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz

        def warp_metric_paper(x,y,z):
            ''' the warping i use in the metrics paper '''
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz
            # Define the bump that localizes the warping to the interior (so edges are not affected)
            alpha = 0.#25 # ~0.25 flattens out the bump
            Sx = np.sin(np.pi * ax) + alpha*np.sin(3*np.pi*ax)
            Sy = np.sin(np.pi * ay) + alpha*np.sin(3*np.pi*ay)
            Sz = np.sin(np.pi * az) + alpha*np.sin(3*np.pi*az)
            Cy = 0.5*np.exp(1-ay)
            Cz = np.cos(az-0.5) #1.0 - (az-0.5)**2
            x_new = x + self.warp_factor * Lx * Sx*Sy*Cz
            y_new = y + self.warp_factor * Ly * Sx*Sy*Cy
            z_new = z + self.warp_factor * Lz * 0.25*Sz*(np.sin(2*np.pi*ax)*np.sin(2*np.pi*ay))
            return x_new, y_new, z_new

        def warp_metric_paper_der(x, y, z):
            Lx, Ly, Lz = self.dom_len
            ax = (x - self.xmin[0]) / Lx
            ay = (y - self.xmin[1]) / Ly
            az = (z - self.xmin[2]) / Lz
            alpha = 0.#25

            # Bumps
            Sx  = np.sin(np.pi*ax) + alpha*np.sin(3*np.pi*ax)
            Sy  = np.sin(np.pi*ay) + alpha*np.sin(3*np.pi*ay)
            Sz  = np.sin(np.pi*az) + alpha*np.sin(3*np.pi*az)

            # First derivatives of bumps
            Sx_x = (np.pi/Lx)*(np.cos(np.pi*ax) + 3*alpha*np.cos(3*np.pi*ax))
            Sy_y = (np.pi/Ly)*(np.cos(np.pi*ay) + 3*alpha*np.cos(3*np.pi*ay))
            Sz_z = (np.pi/Lz)*(np.cos(np.pi*az) + 3*alpha*np.cos(3*np.pi*az))

            # Weights and their needed partials
            Cz     = np.cos(az-0.5) #1.0 - (az - 0.5)**2
            dCz_dz = -np.sin(az-0.5) #-(2.0/Lz)*(az - 0.5)
            Cy     = 0.5*np.exp(1.0 - ay)
            dCy_dy = -(1.0/Ly)*Cy

            # z' additional trigs
            sin2x = np.sin(2.0*np.pi*ax)
            cos2x = np.cos(2.0*np.pi*ax)
            sin2y = np.sin(2.0*np.pi*ay)
            cos2y = np.cos(2.0*np.pi*ay)

            # x' = x + wf*Lx * Sx*Sy*Cz
            dxdx = 1.0 + self.warp_factor*Lx*(Sx_x * Sy * Cz)
            dxdy =        self.warp_factor*Lx*(Sx   * Sy_y * Cz)
            dxdz =        self.warp_factor*Lx*(Sx   * Sy   * dCz_dz)

            # y' = y + wf*Ly * Sx*Sy*Cy
            dydx =        self.warp_factor*Ly*(Sx_x * Sy * Cy)
            dydy = 1.0 +  self.warp_factor*Ly*(Sx   * Sy_y * Cy + Sx*Sy*dCy_dy)
            dydz = 0.0

            # z' = z + wf*Lz * 0.25 * Sz * sin(2π ax) * sin(2π ay)
            dzdx = self.warp_factor*Lz*0.25 * ( Sz * (2.0*np.pi/Lx) * cos2x * sin2y )
            dzdy = self.warp_factor*Lz*0.25 * ( Sz * sin2x * (2.0*np.pi/Ly) * cos2y )
            dzdz = 1.0 + self.warp_factor*Lz*0.25 * ( Sz_z * sin2x * sin2y )
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
        
        def warp_chan(x,y,z):
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]     
            argz = (z-self.xmin[2])/self.dom_len[2]  
            xn = x + self.warp_factor*self.dom_len[0]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            yn = y + self.warp_factor*self.dom_len[1]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            zn = z + self.warp_factor*self.dom_len[2]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            return xn , yn, zn
        
        def warp_chan_der(x,y,z):
            argx = 2*(x-self.xmin[0])/self.dom_len[0] - 1
            argy = 2*(y-self.xmin[1])/self.dom_len[1] - 1     
            argz = 2*(z-self.xmin[2])/self.dom_len[2] - 1 
            dxndx = 1 - self.warp_factor*np.pi*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dxndy = - self.warp_factor*self.dom_len[0]/self.dom_len[1]*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dxndz = - self.warp_factor*self.dom_len[0]/self.dom_len[2]*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
            dyndx = - self.warp_factor*self.dom_len[1]/self.dom_len[0]*np.pi*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dyndy = 1 - self.warp_factor*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dyndz = - self.warp_factor*self.dom_len[1]/self.dom_len[2]*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
            dzndx = - self.warp_factor*np.pi*self.dom_len[2]/self.dom_len[0]*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dzndy = - self.warp_factor*self.dom_len[2]/self.dom_len[1]*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
            dzndz = 1 - self.warp_factor*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
            return dxndx, dxndy, dxndz, dyndx, dyndy, dyndz, dzndx, dzndy, dzndz
        
        def warp_quad(x,y,z):
            ''' Warps according to a quadratic (in each direction. Total order is 4) '''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            warp = self.warp_factor*2
            xscale = (x-self.xmin[0])/(self.xmax[0]-self.xmin[0])
            yscale = (y-self.xmin[1])/(self.xmax[1]-self.xmin[1])
            zscale = (z-self.xmin[2])/(self.xmax[2]-self.xmin[2])
            new_x = warp*self.dom_len[0]*(yscale**2-yscale)*(xscale**2-xscale) + x
            new_y = 2*warp*self.dom_len[1]*(xscale**2-xscale)*(yscale**2-yscale) + y
            new_z = 0.8*warp*self.dom_len[2]*((xscale**2-xscale)+(yscale**2-yscale))*(zscale**2-zscale) + z
            return new_x , new_y , new_z

        def warp_quad_der(x,y,z):
            ''' the derivative of the function warp_quad wrt x (i.e. dnew_x/dx) '''
            warp = self.warp_factor*2
            xscale = (x-self.xmin[0])/self.dom_len[0]
            yscale = (y-self.xmin[1])/self.dom_len[1]
            zscale = (z-self.xmin[2])/self.dom_len[2]
            dxdx = warp*(yscale**2-yscale)*(2*xscale-1) + 1
            dxdy = warp*self.dom_len[0]*(2*yscale-1)*(xscale**2-xscale)/self.dom_len[1]
            dxdz = np.zeros(np.shape(dxdx))
            dydx = 2*warp*self.dom_len[1]*(2*xscale-1)*(yscale**2-yscale)/self.dom_len[0]
            dydy = 2*warp*(xscale**2-xscale)*(2*yscale-1) + 1
            dydz = np.zeros(np.shape(dxdx))
            dzdx = 0.8*warp*self.dom_len[2]*(2*xscale-1)*(zscale**2-zscale)/self.dom_len[0]
            dzdy = 0.8*warp*self.dom_len[2]*(2*yscale-1)*(zscale**2-zscale)/self.dom_len[1]
            dzdz = 0.8*warp*((xscale**2-xscale)+(yscale**2-yscale))*(2*zscale-1) + 1
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
        
        def warp_cubic(x,y,z):
            ''' Warps according to a cubic (in each direction. Total order is X).'''
            assert(self.warp_factor<1 and self.warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
            warp = self.warp_factor*4.6
            xscale = (x-self.xmin[0])/self.dom_len[0]
            yscale = (y-self.xmin[1])/self.dom_len[1]
            zscale = (z-self.xmin[2])/self.dom_len[2]
            a1 = 5
            b1 = -7 
            ax = a1*zscale**3 + b1*zscale**2 - (a1+b1)*zscale + 1.5
            a2 = 5
            b2 = -10 
            ay = a2*zscale**3 + b2*zscale**2 - (a2+b2)*zscale + 1.3
            a3 = 5 
            b3 = -8 
            az = a3*yscale**3 + b3*yscale**2 - (a3+b3)*yscale + 1.5
            new_x = warp*self.dom_len[0]*(yscale**3-ax*yscale**2+(ax-1)*yscale)*(xscale**3-xscale) + x
            new_y = warp*self.dom_len[1]*(xscale**3-ay*xscale**2+(ay-1)*xscale)*(yscale**3-yscale**2) + y
            new_z = warp*self.dom_len[2]*(xscale**3-az*xscale**2+(az-1)*xscale)*(zscale**3-zscale) + z
            return new_x , new_y , new_z

        def warp_cubic_der(x,y,z):
            ''' the derivative of the function warp_cubic wrt x (i.e. dnew_x/dx) '''
            warp = self.warp_factor*4.6
            xscale = (x-self.xmin[0])/self.dom_len[0]
            yscale = (y-self.xmin[1])/self.dom_len[1]
            zscale = (z-self.xmin[2])/self.dom_len[2]
            a1 = 5
            b1 = -7 
            ax = a1*zscale**3 + b1*zscale**2 - (a1+b1)*zscale + 1.5
            daxdz = (3*a1*zscale**2 + 2*b1*zscale - (a1+b1))/self.dom_len[2]
            a2 = 5
            b2 = -10 
            ay = a2*zscale**3 + b2*zscale**2 - (a2+b2)*zscale + 1.3
            daydz = (3*a2*zscale**2 + 2*b2*zscale - (a2+b2))/self.dom_len[2]
            a3 = 5 
            b3 = -8 
            az = a3*yscale**3 + b3*yscale**2 - (a3+b3)*yscale + 1.5
            dazdy = (3*a3*yscale**2 + 2*b3*yscale - (a3+b3))/self.dom_len[1]
            dxdx = warp*(yscale**3-ax*yscale**2+(ax-1)*yscale)*(3*xscale**2-1) + 1
            dxdy = warp*self.dom_len[0]*(3*yscale**2-2*ax*yscale+(ax-1))*(xscale**3-xscale)/self.dom_len[1]
            dxdz = warp*self.dom_len[0]*(-daxdz*yscale**2+daxdz*yscale)*(xscale**3-xscale)
            dydx = warp*self.dom_len[1]*(3*xscale**2-2*ay*xscale+(ay-1))*(yscale**3-yscale**2)/self.dom_len[0]
            dydy = warp*(xscale**3-ay*xscale**2+(ay-1)*xscale)*(3*yscale**2-2*yscale) + 1
            dydz = warp*self.dom_len[1]*(-daydz*xscale**2+daydz*xscale)*(yscale**3-yscale**2)
            dzdx = warp*self.dom_len[2]*(3*xscale**2-2*az*xscale+(az-1))*(zscale**3-zscale)/self.dom_len[0]
            dzdy = warp*self.dom_len[2]*(-dazdy*xscale**2+dazdy*xscale)*(zscale**3-zscale)
            dzdz = warp*(xscale**3-az*xscale**2+(az-1)*xscale)*(3*zscale**2-1) + 1
            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
        
        def warp_strong(x,y,z):
            ''' my modified function in Diablo ''' 
            #assert self.warp_factor<=1,'Try a warp_factor < 1 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            argz = (z-self.xmin[2])/self.dom_len[2]
            
            new_x = x + self.warp_factor*self.dom_len[0]*(0.015*np.sin(4*np.pi*argy)+0.016*np.sin(5*np.pi*argz)+0.013*np.sin(5*np.pi*argx)*np.sin(7*np.pi*argy)*np.exp(1-argx)+0.017*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz)*np.sin(6*np.pi*argx)-0.01*np.sin(9*np.pi*argx)*np.sin(4*np.pi*argy)*np.sin(7*np.pi*argz))
            new_y = y + self.warp_factor*self.dom_len[1]*(0.016*np.sin(5*np.pi*argx)-0.015*np.sin(4*np.pi*argz)-0.012*np.sin(8*np.pi*argx)*np.sin(4*np.pi*argy)*np.exp(1-argy)-0.017*np.sin(3*np.pi*argx)*np.sin(9*np.pi*argx)*np.sin(6*np.pi*argz)*np.sin(7*np.pi*argy)-0.01*np.sin(6*np.pi*argx)*np.sin(6*np.pi*argy)*np.sin(7*np.pi*argz))
            new_z = z + self.warp_factor*self.dom_len[2]*(0.018*np.sin(6*np.pi*argx)+0.02*np.sin(5*np.pi*argy)+0.018*np.sin(7*np.pi*argz)*np.sin(9*np.pi*argz)*np.sin(8*np.pi*argx)*np.sin(7*np.pi*argy)+0.01*np.sin(5*np.pi*argx)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz))
            return new_x , new_y, new_z

        def warp_strong_der(x,y,z):
            ''' the derivative of the function warp_bdy wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            argz = (z-self.xmin[2])/self.dom_len[2]
            
            dxdx = 1 + self.warp_factor*(0.013*np.sin(7*np.pi*argy)*np.exp(1-argx)*(5*np.pi*np.cos(5*np.pi*argx)-np.sin(5*np.pi*argx))+0.017*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz)*6*np.pi*np.cos(6*np.pi*argx)-0.01*9*np.pi*np.cos(9*np.pi*argx)*np.sin(4*np.pi*argy)*np.sin(7*np.pi*argz))
            dydx = self.warp_factor*self.dom_len[1]*np.pi*(0.016*5*np.cos(5*np.pi*argx)-0.012*8*np.cos(8*np.pi*argx)*np.sin(4*np.pi*argy)*np.exp(1-argy)-0.017*np.sin(6*np.pi*argz)*np.sin(7*np.pi*argy)*(3*np.cos(3*np.pi*argx)*np.sin(9*np.pi*argx)+np.sin(3*np.pi*argx)*9*np.cos(9*np.pi*argx))-0.01*6*np.cos(6*np.pi*argx)*np.sin(6*np.pi*argy)*np.sin(7*np.pi*argz))/self.dom_len[0]
            dzdx = self.warp_factor*self.dom_len[2]*np.pi*(0.018*6*np.cos(6*np.pi*argx)+0.018*np.sin(7*np.pi*argz)*np.sin(9*np.pi*argz)*8*np.cos(8*np.pi*argx)*np.sin(7*np.pi*argy)+0.01*5*np.cos(5*np.pi*argx)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz))/self.dom_len[0]
            dxdy = self.warp_factor*self.dom_len[0]*np.pi*(0.015*4*np.cos(4*np.pi*argy)+0.013*np.sin(5*np.pi*argx)*7*np.cos(7*np.pi*argy)*np.exp(1-argx)+0.017*(7*np.cos(7*np.pi*argy)*np.sin(8*np.pi*argy)+np.sin(7*np.pi*argy)*8*np.cos(8*np.pi*argy))*np.sin(6*np.pi*argz)*np.sin(6*np.pi*argx)-0.01*np.sin(9*np.pi*argx)*4*np.cos(4*np.pi*argy)*np.sin(7*np.pi*argz))/self.dom_len[1]
            dydy = 1 + self.warp_factor*(-0.012*np.sin(8*np.pi*argx)*np.exp(1-argy)*(4*np.pi*np.cos(4*np.pi*argy)-np.sin(4*np.pi*argy))-0.017*np.sin(3*np.pi*argx)*np.sin(9*np.pi*argx)*np.sin(6*np.pi*argz)*7*np.pi*np.cos(7*np.pi*argy)-0.01*np.sin(6*np.pi*argx)*6*np.pi*np.cos(6*np.pi*argy)*np.sin(7*np.pi*argz))
            dzdy = self.warp_factor*self.dom_len[2]*np.pi*(0.02*5*np.cos(5*np.pi*argy)+0.018*np.sin(7*np.pi*argz)*np.sin(9*np.pi*argz)*np.sin(8*np.pi*argx)*7*np.cos(7*np.pi*argy)+0.01*np.sin(5*np.pi*argx)*8*np.cos(8*np.pi*argy)*np.sin(6*np.pi*argz))/self.dom_len[1]
            dxdz = self.warp_factor*self.dom_len[0]*np.pi*(0.016*5*np.cos(5*np.pi*argz)+0.017*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argy)*6*np.cos(6*np.pi*argz)*np.sin(6*np.pi*argx)-0.01*np.sin(9*np.pi*argx)*np.sin(4*np.pi*argy)*7*np.cos(7*np.pi*argz))/self.dom_len[2]
            dydz = self.warp_factor*self.dom_len[1]*np.pi*(-0.015*4*np.cos(4*np.pi*argz)-0.017*np.sin(3*np.pi*argx)*np.sin(9*np.pi*argx)*6*np.cos(6*np.pi*argz)*np.sin(7*np.pi*argy)-0.01*np.sin(6*np.pi*argx)*np.sin(6*np.pi*argy)*7*np.cos(7*np.pi*argz))/self.dom_len[2]
            dzdz = 1 + self.warp_factor*np.pi*(0.018*(7*np.cos(7*np.pi*argz)*np.sin(9*np.pi*argz)+np.sin(7*np.pi*argz)*9*np.cos(9*np.pi*argz))*np.sin(8*np.pi*argx)*np.sin(7*np.pi*argy)+0.01*np.sin(5*np.pi*argx)*np.sin(8*np.pi*argy)*6*np.cos(6*np.pi*argz))

            return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
        
        # switch between different mappings here
        if self.warp_type == 'default' or self.warp_type == 'papers':
            warp_fun = warp_cuboid
            warp_der = warp_cuboid_der 
        elif self.warp_type == 'papers_flipped' or self.warp_type == 'papers flipped':
            warp_fun = warp_cuboid_flipped
            warp_der = warp_cuboid_flipped_der 
        elif self.warp_type == 'quad':
            warp_fun = warp_quad
            warp_der = warp_quad_der  
        elif self.warp_type == 'cubic':
            warp_fun = warp_cubic
            warp_der = warp_cubic_der
        elif self.warp_type == 'strong':
            warp_fun = warp_strong
            warp_der = warp_strong_der
        elif self.warp_type == 'chan':
            warp_fun = warp_chan
            warp_der = warp_chan_der
        elif self.warp_type == 'asym':
            warp_fun = warp_asym
            warp_der = warp_asym_der
        elif self.warp_type == 'asym_minimal':
            warp_fun = warp_minimal
            warp_der = warp_minimal_der
        elif self.warp_type == 'metrics_paper':
            warp_fun = warp_metric_paper
            warp_der = warp_metric_paper_der
        else:
            print('WARNING: mesh.warp_type not understood. Reverting to default.')
            warp_fun = warp_cuboid
            warp_der = warp_cuboid_der 

        if xyz is not None:
            xyz_new = np.copy(xyz)
            xyz_new[:,0,:], xyz_new[:,1,:], xyz_new[:,2,:] = warp_fun(xyz[:,0,:], xyz[:,1,:], xyz[:,2,:])
            return xyz_new
        else:
        
            xyz_elem_old = np.copy(self.xyz_elem)
            bdy_xyz_old = np.copy(self.bdy_xyz)
            self.xyz_elem[:,0,:], self.xyz_elem[:,1,:], self.xyz_elem[:,2,:] = warp_fun(self.xyz_elem[:,0,:], self.xyz_elem[:,1,:], self.xyz_elem[:,2,:])
            self.bdy_xyz[:,0,:,:], self.bdy_xyz[:,1,:,:], self.bdy_xyz[:,2,:,:] = warp_fun(self.bdy_xyz[:,0,:,:],self.bdy_xyz[:,1,:,:],self.bdy_xyz[:,2,:,:])
            self.grid_planes[:,0,:,:], self.grid_planes[:,1,:,:], self.grid_planes[:,2,:,:] =  warp_fun(self.grid_planes[:,0,:,:], self.grid_planes[:,1,:,:], self.grid_planes[:,2,:,:])
            
            for i in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
                a = i*self.nen**3
                b = (i+1)*self.nen**3
                self.xyz[a:b,:] = self.xyz_elem[:,:,i]
                
            dxnewdx, dxnewdy, dxnewdz, dynewdx, dynewdy, dynewdz, dznewdx, dznewdy, dznewdz = warp_der(xyz_elem_old[:,0,:], xyz_elem_old[:,1,:], xyz_elem_old[:,2,:])
            dxdxref = np.copy(self.jac_exa[:,0,0,:])
            dydyref = np.copy(self.jac_exa[:,1,1,:])
            dzdzref = np.copy(self.jac_exa[:,2,2,:])
            self.jac_exa[:,0,0,:] = dxnewdx * dxdxref # chain rule, ignoring cross terms that are 0 in original transformation
            self.jac_exa[:,0,1,:] = dxnewdy * dydyref
            self.jac_exa[:,0,2,:] = dxnewdz * dzdzref
            self.jac_exa[:,1,0,:] = dynewdx * dxdxref
            self.jac_exa[:,1,1,:] = dynewdy * dydyref
            self.jac_exa[:,1,2,:] = dynewdz * dzdzref
            self.jac_exa[:,2,0,:] = dznewdx * dxdxref
            self.jac_exa[:,2,1,:] = dznewdy * dydyref
            self.jac_exa[:,2,2,:] = dznewdz * dzdzref
            for elem in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
                self.det_jac_exa[:,elem] = np.linalg.det(self.jac_exa[:,:,:,elem])
            assert np.all(self.det_jac_exa>0),"Not a valid coordinate transformation. Try using a lower warp_factor."
            #for elem in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
                #self.jac_inv_exa[:,:,:,elem] = np.linalg.inv(self.jac_exa[:,:,:,elem])
                #self.det_jac_inv_exa[:,elem] =  np.linalg.det(self.jac_inv_exa[:,:,:,elem])
            #if np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa) > 1e-12):
            #    print('WANRING: Numerical error in calculation of determinant inverse is {0:.2g}'.format(np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa))))

            dxnewdx, dxnewdy, dxnewdz, dynewdx, dynewdy, dynewdz, dznewdx, dznewdy, dznewdz = warp_der(bdy_xyz_old[:,0,:,:], bdy_xyz_old[:,1,:,:], bdy_xyz_old[:,2,:,:])
            dxdxref = np.copy(self.bdy_jac_exa[:,0,0,:,:])
            dydyref = np.copy(self.bdy_jac_exa[:,1,1,:,:])
            dzdzref = np.copy(self.bdy_jac_exa[:,2,2,:,:])
            self.bdy_jac_exa[:,0,0,:,:] = dxnewdx * dxdxref # chain rule, ignoring cross terms that are 0 in original transformation
            self.bdy_jac_exa[:,0,1,:,:] = dxnewdy * dydyref
            self.bdy_jac_exa[:,0,2,:,:] = dxnewdz * dzdzref
            self.bdy_jac_exa[:,1,0,:,:] = dynewdx * dxdxref
            self.bdy_jac_exa[:,1,1,:,:] = dynewdy * dydyref
            self.bdy_jac_exa[:,1,2,:,:] = dynewdz * dzdzref
            self.bdy_jac_exa[:,2,0,:,:] = dznewdx * dxdxref
            self.bdy_jac_exa[:,2,1,:,:] = dznewdy * dydyref
            self.bdy_jac_exa[:,2,2,:,:] = dznewdz * dzdzref
            self.bdy_det_jac_exa = np.zeros((self.nen*self.nen,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
            for elem in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
                for i in range(6):
                    self.bdy_det_jac_exa[:,i,elem] = np.linalg.det(self.bdy_jac_exa[:,:,:,i,elem])
                    #self.bdy_jac_inv_exa[:,:,:,i,elem] = np.linalg.inv(self.bdy_jac_exa[:,:,:,i,elem])
            assert np.all(self.bdy_det_jac_exa>0),"Not a valid coordinate transformation. Try using a lower warp_factor."
                    
            ''' Uncomment the below section to debug and test the consistency of the warping '''
# =============================================================================
        """max_xy = [0,0,0,0,0,0,0,0,0]
        max_der = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            
        skipx = self.nelem[1]*self.nelem[2]
        for row in range(skipx):
            diff = abs(self.bdy_xyz[:,0,0,row::skipx][:,0] - self.bdy_xyz[:,0,1,row::skipx][:,-1])
            max_xy[0] = max(max_xy[0],np.max(diff))
            diff = abs(self.bdy_xyz[:,1,0,row::skipx][:,0] - self.bdy_xyz[:,1,1,row::skipx][:,-1])
            max_xy[1] = max(max_xy[1],np.max(diff))
            diff = abs(self.bdy_xyz[:,2,0,row::skipx][:,0] - self.bdy_xyz[:,2,1,row::skipx][:,-1])
            max_xy[2] = max(max_xy[2],np.max(diff))
            diff = abs(dxnewdx[:,0,row::skipx][:,0] - dxnewdx[:,1,row::skipx][:,-1])
            max_der[0] = max(max_der[0],np.max(diff))
            diff = abs(dxnewdy[:,0,row::skipx][:,0] - dxnewdy[:,1,row::skipx][:,-1])
            max_der[1] = max(max_der[1],np.max(diff))
            diff = abs(dxnewdz[:,0,row::skipx][:,0] - dxnewdz[:,1,row::skipx][:,-1])
            max_der[2] = max(max_der[2],np.max(diff))
            diff = abs(dynewdx[:,0,row::skipx][:,0] - dynewdx[:,1,row::skipx][:,-1])
            max_der[3] = max(max_der[3],np.max(diff))
            diff = abs(dynewdy[:,0,row::skipx][:,0] - dynewdy[:,1,row::skipx][:,-1])
            max_der[4] = max(max_der[4],np.max(diff))
            diff = abs(dynewdz[:,0,row::skipx][:,0] - dynewdz[:,1,row::skipx][:,-1])
            max_der[5] = max(max_der[5],np.max(diff))
            diff = abs(dznewdx[:,0,row::skipx][:,0] - dznewdx[:,1,row::skipx][:,-1])
            max_der[6] = max(max_der[6],np.max(diff))
            diff = abs(dznewdy[:,0,row::skipx][:,0] - dznewdy[:,1,row::skipx][:,-1])
            max_der[7] = max(max_der[7],np.max(diff))
            diff = abs(dznewdz[:,0,row::skipx][:,0] - dznewdz[:,1,row::skipx][:,-1])
            max_der[8] = max(max_der[8],np.max(diff))
                
        for coly in range(self.nelem[0]*self.nelem[2]):
            start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
            end = start + self.nelem[1]*self.nelem[2]
            diff = abs(self.bdy_xyz[:,0,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,0,3,start:end:self.nelem[2]][:,-1])
            max_xy[3] = max(max_xy[3],np.max(diff))
            diff = abs(self.bdy_xyz[:,1,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,1,3,start:end:self.nelem[2]][:,-1])
            max_xy[4] = max(max_xy[4],np.max(diff))
            diff = abs(self.bdy_xyz[:,2,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,2,3,start:end:self.nelem[2]][:,-1])
            max_xy[5] = max(max_xy[5],np.max(diff))
            diff = abs(dxnewdx[:,2,start:end:self.nelem[2]][:,0] - dxnewdx[:,3,start:end:self.nelem[2]][:,-1])
            max_der[9] = max(max_der[9],np.max(diff))
            diff = abs(dxnewdy[:,2,start:end:self.nelem[2]][:,0] - dxnewdy[:,3,start:end:self.nelem[2]][:,-1])
            max_der[10] = max(max_der[10],np.max(diff))
            diff = abs(dxnewdz[:,2,start:end:self.nelem[2]][:,0] - dxnewdz[:,3,start:end:self.nelem[2]][:,-1])
            max_der[11] = max(max_der[11],np.max(diff))
            diff = abs(dynewdx[:,2,start:end:self.nelem[2]][:,0] - dynewdx[:,3,start:end:self.nelem[2]][:,-1])
            max_der[12] = max(max_der[12],np.max(diff))
            diff = abs(dynewdy[:,2,start:end:self.nelem[2]][:,0] - dynewdy[:,3,start:end:self.nelem[2]][:,-1])
            max_der[13] = max(max_der[13],np.max(diff))
            diff = abs(dynewdz[:,2,start:end:self.nelem[2]][:,0] - dynewdz[:,3,start:end:self.nelem[2]][:,-1])
            max_der[14] = max(max_der[14],np.max(diff))
            diff = abs(dznewdx[:,2,start:end:self.nelem[2]][:,0] - dznewdx[:,3,start:end:self.nelem[2]][:,-1])
            max_der[15] = max(max_der[15],np.max(diff))
            diff = abs(dznewdy[:,2,start:end:self.nelem[2]][:,0] - dznewdy[:,3,start:end:self.nelem[2]][:,-1])
            max_der[16] = max(max_der[16],np.max(diff))
            diff = abs(dznewdz[:,2,start:end:self.nelem[2]][:,0] - dznewdz[:,3,start:end:self.nelem[2]][:,-1])
            max_der[17] = max(max_der[17],np.max(diff))

        
        for colz in range(self.nelem[0]*self.nelem[2]):
            start = colz*self.nelem[2]            
            end = start + self.nelem[2]
            diff = abs(self.bdy_xyz[:,0,4,start:end][:,0] - self.bdy_xyz[:,0,5,start:end][:,-1])
            max_xy[6] = max(max_xy[6],np.max(diff))
            diff = abs(self.bdy_xyz[:,1,4,start:end][:,0] - self.bdy_xyz[:,1,5,start:end][:,-1])
            max_xy[7] = max(max_xy[7],np.max(diff))
            diff = abs(self.bdy_xyz[:,2,4,start:end][:,0] - self.bdy_xyz[:,2,5,start:end][:,-1])
            max_xy[8] = max(max_xy[8],np.max(diff))
            diff = abs(dxnewdx[:,4,start:end][:,0] - dxnewdx[:,5,start:end][:,-1])
            max_der[18] = max(max_der[18],np.max(diff))
            diff = abs(dxnewdy[:,4,start:end][:,0] - dxnewdy[:,5,start:end][:,-1])
            max_der[19] = max(max_der[19],np.max(diff))
            diff = abs(dxnewdz[:,4,start:end][:,0] - dxnewdz[:,5,start:end][:,-1])
            max_der[20] = max(max_der[20],np.max(diff))
            diff = abs(dynewdx[:,4,start:end][:,0] - dynewdx[:,5,start:end][:,-1])
            max_der[21] = max(max_der[21],np.max(diff))
            diff = abs(dynewdy[:,4,start:end][:,0] - dynewdy[:,5,start:end][:,-1])
            max_der[22] = max(max_der[22],np.max(diff))
            diff = abs(dynewdz[:,4,start:end][:,0] - dynewdz[:,5,start:end][:,-1])
            max_der[23] = max(max_der[23],np.max(diff))
            diff = abs(dznewdx[:,4,start:end][:,0] - dznewdx[:,5,start:end][:,-1])
            max_der[24] = max(max_der[24],np.max(diff))
            diff = abs(dznewdy[:,4,start:end][:,0] - dznewdy[:,5,start:end][:,-1])
            max_der[25] = max(max_der[25],np.max(diff))
            diff = abs(dznewdz[:,4,start:end][:,0] - dznewdz[:,5,start:end][:,-1])
            max_der[26] = max(max_der[26],np.max(diff))

        #print('Max diff x values along x_ref direction boundary (=domain[0]): ', max_xy[0])
        print('Max diff y values along x_ref direction boundary: ', max_xy[1])
        print('Max diff z values along x_ref direction boundary: ', max_xy[2])
        print('Max diff x values along y_ref direction boundary: ', max_xy[3])
        #print('Max diff y values along y_ref direction boundary: (=domain[1])', max_xy[4])
        print('Max diff z values along y_ref direction boundary: ', max_xy[5])
        print('Max diff x values along z_ref direction boundary: ', max_xy[6])
        print('Max diff y values along z_ref direction boundary: ', max_xy[7])
        #print('Max diff z values along z_ref direction boundary: (=domain[2])', max_xy[8])
        #print('Max diff of dxnew(1)/dxref(1) along x bdy (not nec. = 0): ', max_der[0])
        print('Max diff of dxnew(1)/dxref(2) along x bdy: ', max_der[1])
        print('Max diff of dxnew(1)/dxref(3) along x bdy: ', max_der[2])
        #print('Max diff of dxnew(2)/dxref(1) along x bdy (not nec. = 0): ', max_der[3])
        print('Max diff of dxnew(2)/dxref(2) along x bdy: ', max_der[4])
        print('Max diff of dxnew(2)/dxref(3) along x bdy: ', max_der[5])
        #print('Max diff of dxnew(3)/dxref(1) along x bdy (not nec. = 0): ', max_der[6])
        print('Max diff of dxnew(3)/dxref(2) along x bdy: ', max_der[7])
        print('Max diff of dxnew(3)/dxref(3) along x bdy: ', max_der[8])
        print('Max diff of dxnew(1)/dxref(1) along y bdy: ', max_der[9])
        #print('Max diff of dxnew(1)/dxref(2) along y bdy (not nec. = 0): ', max_der[10])
        print('Max diff of dxnew(1)/dxref(3) along y bdy: ', max_der[11])
        print('Max diff of dxnew(2)/dxref(1) along y bdy: ', max_der[12])
        #print('Max diff of dxnew(2)/dxref(2) along y bdy (not nec. = 0): ', max_der[13])
        print('Max diff of dxnew(2)/dxref(3) along y bdy: ', max_der[14])
        print('Max diff of dxnew(3)/dxref(1) along y bdy: ', max_der[15])
        #print('Max diff of dxnew(3)/dxref(2) along y bdy (not nec. = 0): ', max_der[16])
        print('Max diff of dxnew(3)/dxref(3) along y bdy: ', max_der[17])
        print('Max diff of dxnew(1)/dxref(1) along z bdy: ', max_der[18])
        print('Max diff of dxnew(1)/dxref(2) along z bdy: ', max_der[19])
        #print('Max diff of dxnew(1)/dxref(3) along z bdy (not nec. = 0): ', max_der[20])
        print('Max diff of dxnew(2)/dxref(1) along z bdy: ', max_der[21])
        print('Max diff of dxnew(2)/dxref(2) along z bdy: ', max_der[22])
        #print('Max diff of dxnew(2)/dxref(3) along z bdy (not nec. = 0): ', max_der[23])
        print('Max diff of dxnew(3)/dxref(1) along z bdy: ', max_der[24])
        print('Max diff of dxnew(3)/dxref(2) along z bdy: ', max_der[25])
        #print('Max diff of dxnew(3)/dxref(3) along z bdy (not nec. = 0): ', max_der[26]) """
# =============================================================================

    def plot(self,plt_save_name=None,markersize=4,fontsize=12,dpi=1000,label=True,
             label_all_lines=True, nodes=True, bdy_nodes=False,save_format='png',lw=1):
        if self.dim == 1:
            fig = plt.figure(figsize=(6,1))
            ax = plt.axes(frameon=False) # turn off the frame
            
            ax.hlines(0.35,self.xmin,self.xmax,color='black',lw=1)  # Draw a horizontal line at y=1
            ax.set_xlim(self.xmin-self.dom_len/100,self.xmax+self.dom_len/100)
            ax.set_ylim(0,1)

            ax.axes.get_yaxis().set_visible(False) # turn off y axis 
            ax.plot(self.vertices,0.35*np.ones(self.nelem+1),'|',ms=20,color='black',lw=1)  # Plot a line at each location specified in a
            
            if bdy_nodes:
                ax.plot(self.bdy_x,0.35*np.ones(self.bdy_x.shape),'s',color='red',ms=markersize)
                #ax.plot(self.bdy_x[0],0.35,'s',color='red',ms=markersize)

            if nodes:
                ax.plot(self.x,0.35*np.ones(self.nn),'o',color='blue',ms=markersize)
                #ax.plot(self.x[:-1],0.35*np.ones(self.nn-1),'o',color='blue',ms=markersize)
            
            if label and label_all_lines:
                ax.tick_params(axis='x',length=0,labelsize=fontsize) # hide x ticks
                ax.set_xticks(self.vertices) # label element boundaries
            elif label:
                pass # use default labels
            else:
                ax.set_xticks([])
        
        elif self.dim == 2:
            fig = plt.figure(figsize=(6,6*self.dom_len[1]/self.dom_len[0])) # scale figure properly
            ax = plt.axes(frameon=False) # turn off the frame
            
            xmax = max(self.xmax[0], np.max(self.grid_lines[:,0,:]))
            xmin = min(self.xmin[0], np.min(self.grid_lines[:,0,:]))
            ymax = max(self.xmax[1], np.max(self.grid_lines[:,1,:]))
            ymin = min(self.xmin[1], np.min(self.grid_lines[:,1,:]))
            
            ax.set_xlim(xmin-self.dom_len[0]/100,xmax+self.dom_len[0]/100)
            ax.set_ylim(ymin-self.dom_len[1]/100,ymax+self.dom_len[1]/100)

            for line in self.grid_lines:
                ax.plot(line[0],line[1],color='black',lw=lw)

            if bdy_nodes:
                #ax.scatter(self.bdy_xy[:,0,:,:],self.bdy_xy[:,1,:,:],marker='o',color='r',s=4)
                ax.scatter(self.bdy_xy[:,0,:,:],self.bdy_xy[:,1,:,:],marker='s',color='red',s=markersize)
            
            if nodes:
                #ax.scatter(self.xy[:,0],self.xy[:,1],marker='s',c='b',s=9)
                ax.scatter(self.xy[:,0],self.xy[:,1],marker='o',c='blue',s=markersize)
    
            if label and label_all_lines:
                ax.tick_params(axis='both',length=0,labelsize=fontsize) # hide ticks
                edge_verticesx = np.linspace(self.xmin[0],self.xmax[0],self.nelem[0]+1)
                edge_verticesy = np.linspace(self.xmin[1],self.xmax[1],self.nelem[1]+1)
                ax.set_xticks(edge_verticesx) # label element boundaries
                ax.set_yticks(edge_verticesy)
            elif label:
                pass # use default labels
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                
        elif self.dim == 3:
            fig = plt.figure() # TODO: scale figure properly (not possible atm)
            ax = plt.axes(projection="3d") # turn off the frame
            #ax.set_aspect('equal') # not implemented yet
            
            ax.set_xlim(self.xmin[0]-self.dom_len[0]/100,self.xmax[0]+self.dom_len[0]/100)
            ax.set_ylim(self.xmin[1]-self.dom_len[1]/100,self.xmax[1]+self.dom_len[1]/100)
            ax.set_zlim(self.xmin[2]-self.dom_len[2]/100,self.xmax[2]+self.dom_len[2]/100)

            if nodes:
                ax.scatter3D(self.xyz[:,0], self.xyz[:,1], self.xyz[:,2], c='blue',s=markersize)
                
            if bdy_nodes:
                ax.scatter3D(self.bdy_xyz[:,0,:,:],self.bdy_xyz[:,1,:,:],self.bdy_xyz[:,2,:,:],c='red',s=markersize)
            
            for plane in self.grid_planes:
                ax.plot_surface(plane[0,:,:],plane[1,:,:],plane[2,:,:],color='black',lw=1,alpha=0.3)
            
            if label and label_all_lines:
                ax.xaxis._axinfo['tick']['length']=0 # hide ticks
                ax.yaxis._axinfo['tick']['length']=0
                ax.zaxis._axinfo['tick']['length']=0
                edge_verticesx = np.linspace(self.xmin[0],self.xmax[0],self.nelem[0]+1)
                edge_verticesy = np.linspace(self.xmin[1],self.xmax[1],self.nelem[1]+1)
                edge_verticesz = np.linspace(self.xmin[2],self.xmax[2],self.nelem[2]+1)
                ax.set_xticks(edge_verticesx) # label element boundaries
                ax.set_yticks(edge_verticesy)
                ax.set_zticks(edge_verticesz)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
            elif label:
                pass # use default labels
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
            
            
        if plt_save_name is not None:
            fig.tight_layout()
            fig.savefig(plt_save_name, format=save_format,dpi=dpi)
            

    def get_jac_metrics(self, sbp, periodic, metric_method='exact', bdy_metric_method='exact',
                        jac_method='exact', use_optz_metrics = True, calc_exact_metrics = False,
                        optz_method = 'default', had_metric_alpha = 1, had_metric_beta = 0): 
        '''
        Parameters
        ----------
        sbp : np float array size (nen,nen)
            class for Derivative operator on reference element.
        periodic : bool or tuple of bool
            whether mesh is periodic or not in each direction
        metric_method : str
            options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact'
            which approximation method to use for metric terms.
            The default is 'exact'.
        bdy_metric_method : str
            options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact', 'interpolate'
            which approximation method to use for metric terms.
            The default is 'exact'.
        jac_method : str
            options: 'calculate', 'direct', 'deng', 'match', 'backout', 'exact'
            which approximation method to use for metric jacobian.
            The default is 'exact'.
        use_optz_metrics : bool
            if True, we use optimization procedure from DDRF et al 2019 to 
            construct metrics that preserve free stream.
            The default is True.
        calc_exact_metrics : bool
            if True, calculate the exact metrics alongside interpolated metrics.
            The default is False.
        optz_method : str
            Choose the different optimization methods:
            'default' / 'alex' / 'essbp' : Additional preoptimization for surface integrals, then ddrf.
            'ddrf' / papers : from from DDRF et al. 
            'diablo' : the procedure implemented in Diablo. - though this one is suspicious.
            'generalized' : Uses had_metric_alpha and had_metric_beta for a generalized optimization
            The defalt is 'default'.
        had_metric_alpha / had_metric_beta : float
            Parameters for a generalized Hadamard form (See ESSBP documentation).
            Only get called wuth optz_method = 'generalized'. Default is 1 and 0.

        Sets
        -------
        det_jac : np float array (nen,nelem) or (nen^2,nelem)
            The determinant of the Jacobian for the transformation for the mesh.
            Should technically be a diagonal matrix where each diagonal entry
            corresponds to the jacobian at that node.
        det_jac_inv : np float array (nen,nelem) or (nen^2,nelem)
            The inverse of the above matrix, so just 1/det_jac
        metrics : np float array (nen^2,4,nelem)
           Only returned if dim = 2. Computes the metric invariants. Columns are
           [det_jac*Dx_phys@x_ref, det_jac*Dy_phys@x_ref, det_jac*Dx_phys@y_ref, det_jac*Dy_phys@y_ref]
           = [ det(J) * dxr_dxp , det(J) * dxr_dyp , det(J) * dyr_dxp , det(J) * dyr_dyp ]
           = [ dyp_dyr , - dxp_dyr , - dyp_dxr , dxp_dxr ]
        bdy_metrics : np float array (nen,2,4,nelem)
            In each physical direction (2) we add up for each computational direction (2)
            a contribution to the SAT on each boundary (2). Shape is as follows:
            nen : d-1 dimension number of nodes on the boundary
            2 : first the left (a) side, then the right (b) side
            4 : [ det(J) * dxr_dxp , det(J) * dxr_dyp , det(J) * dyr_dxp , det(J) * dyr_dyp ]
        '''
        if self.print_progress: print('... Computing Grid Metrics')
        
        if metric_method=='exact': calc_exact_metrics = True

        if self.dim == 1:
            
            self.metrics = np.ones((self.nen,1,self.nelem))
            self.bdy_metrics = np.ones((2,self.nelem))
            
            if jac_method=='exact':
                self.det_jac = self.det_jac_exa
            else:
                if jac_method!='calculate' or jac_method!='direct':
                    print("WARNING: Did not understand jac_method. For 1D, try 'exact' or 'calculate'.")
                    print("         Defaulting to 'calculate'.")
                self.det_jac = abs(sbp.D @ self.x_elem) # using lagrange interpolation
                
            
        elif self.dim == 2:
            
            if calc_exact_metrics: 
                self.metrics_exa = np.zeros((self.nen**2,4,self.nelem[0]*self.nelem[1])) 
                self.bdy_metrics_exa = np.zeros((self.nen,4,4,self.nelem[0]*self.nelem[1])) 
                #self.fac_normals_exa = np.zeros((self.nen,4,2,self.nelem[0]*self.nelem[1]))
                self.bdy_jac_factor = np.zeros((self.nen,4,self.nelem[0]*self.nelem[1]))
                # nodes, boundary (left, right, lower, upper), d(xi_i)/d(x_j) (dx_r/dx_p, dx_r/dy_p, dy_r/dx_p, dy_r/dy_p), elem
            
                #self.metrics_exa[:,0,:] = self.det_jac_exa * self.jac_inv_exa[:,0,0,:]
                #self.metrics_exa[:,1,:] = self.det_jac_exa * self.jac_inv_exa[:,0,1,:]
                #self.metrics_exa[:,2,:] = self.det_jac_exa * self.jac_inv_exa[:,1,0,:]
                #self.metrics_exa[:,3,:] = self.det_jac_exa * self.jac_inv_exa[:,1,1,:]
                # I think using the below is more accurate? Avoids taking matrix inverses
                self.metrics_exa[:,0,:] = self.jac_exa[:,1,1,:]
                self.metrics_exa[:,1,:] = - self.jac_exa[:,0,1,:]
                self.metrics_exa[:,2,:] = - self.jac_exa[:,1,0,:]
                self.metrics_exa[:,3,:] = self.jac_exa[:,0,0,:]
                
                for f in range(4): # loop over facets (left, right, lower, upper)
                    #self.bdy_metrics_exa[:,f,0,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,0,f,:]
                    #self.bdy_metrics_exa[:,f,1,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,1,f,:]
                    #self.bdy_metrics_exa[:,f,2,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,0,f,:]
                    #self.bdy_metrics_exa[:,f,3,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,1,f,:]
                    self.bdy_metrics_exa[:,f,0,:] = self.bdy_jac_exa[:,1,1,f,:]
                    self.bdy_metrics_exa[:,f,1,:] = - self.bdy_jac_exa[:,0,1,f,:]
                    self.bdy_metrics_exa[:,f,2,:] = - self.bdy_jac_exa[:,1,0,f,:]
                    self.bdy_metrics_exa[:,f,3,:] = self.bdy_jac_exa[:,0,0,f,:]

                    if f == 0:
                        nxref = -1
                        nyref = 0
                    elif f == 1:
                        nxref = 1
                        nyref = 0
                    elif f == 2:
                        nxref = 0
                        nyref = -1
                    elif f == 3:
                        nxref = 0
                        nyref = 1      
                    x_unnormed = nxref*self.bdy_metrics_exa[:,f,0,:] + nyref*self.bdy_metrics_exa[:,f,2,:]
                    y_unnormed = nxref*self.bdy_metrics_exa[:,f,1,:] + nyref*self.bdy_metrics_exa[:,f,3,:]
                    norm = np.sqrt(x_unnormed**2 + y_unnormed**2)
                    #self.fac_normals_exa[:,f,0,:] = x_unnormed / norm
                    #self.fac_normals_exa[:,f,1,:] = y_unnormed / norm  
                    self.bdy_jac_factor[:,f,:] = norm
            
            if metric_method=='exact':
                self.metrics = np.copy(self.metrics_exa)
                
                if jac_method=='calculate' or jac_method=='direct':
                    Dx = np.kron(sbp.D, np.eye(self.nen)) # shape (nen^2,nen^2)
                    Dy = np.kron(np.eye(self.nen), sbp.D) 
                    dxp_dxr = Dx @ self.xy_elem[:,0,:]
                    dxp_dyr = Dy @ self.xy_elem[:,0,:]
                    dyp_dxr = Dx @ self.xy_elem[:,1,:]
                    dyp_dyr = Dy @ self.xy_elem[:,1,:]
                    # metric jacobian (determinant) is given by 
                    # Dx_ref@x_phys*Dy_ref@y_phys - Dy_ref@x_phys*Dx_ref@y_phys  
                    self.det_jac = dxp_dxr*dyp_dyr - dxp_dyr*dyp_dxr 
            
            elif metric_method.lower()== 'kopriva' or metric_method.lower()== 'kcw' \
                or metric_method.lower()== 'kopriva_extrap' or metric_method.lower()== 'kcw_extrap':
                self.metrics = np.zeros((self.nen**2,4,self.nelem[0]*self.nelem[1]))
                from Source.Disc.MakeSbpOp import MakeSbpOp
                sbp_lgl = MakeSbpOp(sbp.p,'lgl',print_progress=False)
                from Source.Disc.MakeDgOp import MakeDgOp
                Vlgltosbp = MakeDgOp.VandermondeLagrange1D(sbp.x,sbp_lgl.x)
                Vsbptolgl = MakeDgOp.VandermondeLagrange1D(sbp_lgl.x,sbp.x)
                Vlgltosbp = np.kron(Vlgltosbp,Vlgltosbp)
                Vsbptolgl = np.kron(Vsbptolgl,Vsbptolgl)

                eye = np.eye(self.nen)
                Dx = np.kron(sbp_lgl.D, eye)
                Dy = np.kron(eye, sbp_lgl.D)

                if metric_method.lower()== 'kopriva_extrap' or metric_method.lower()== 'kcw_extrap':
                # The following does NOT produce unique surface values, so we'll need to average
                    xy_elem = np.einsum('ij,jme->ime', Vsbptolgl, self.xy_elem)
                    # we need to average now, keeping in mind vertices are shared between more than two elements
                    xy_elem = self.average_facet_nodes(xy_elem,sbp_lgl.nn,periodic)
                    x_elem, y_elem = xy_elem[:,0,:], xy_elem[:,1,:]
                else:
                    x_unwarped = self.xy_elem_unwarped[:,0,:]
                    y_unwarped = self.xy_elem_unwarped[:,1,:]
                    xy_unwarped_lgl = np.zeros_like(self.xy_elem_unwarped)
                    xy_unwarped_lgl[:,0,:] = Vsbptolgl @ x_unwarped
                    xy_unwarped_lgl[:,1,:] = Vsbptolgl @ y_unwarped
                    xy_elem = self.warp_mesh_2d(xy=xy_unwarped_lgl)
                    x_elem, y_elem = xy_elem[:,0,:], xy_elem[:,1,:]
                
                dxp_dxr = Dx @ x_elem
                dxp_dyr = Dy @ x_elem
                dyp_dxr = Dx @ y_elem
                dyp_dyr = Dy @ y_elem    
                self.metrics[:,0,:] = Vlgltosbp @ dyp_dyr
                self.metrics[:,1,:] = - Vlgltosbp @ dxp_dyr
                self.metrics[:,2,:] = - Vlgltosbp @ dyp_dxr
                self.metrics[:,3,:] = Vlgltosbp @ dxp_dxr 
                
            else:
                self.metrics = np.zeros((self.nen**2,4,self.nelem[0]*self.nelem[1])) 
            
                if metric_method!='calculate':
                    print("WARNING: Did not understand metric_method. For 2D, try 'exact' or 'interpolate'.")
                    print("         Defaulting to 'calculate'.")
                # using lagrange interpolation (or directly using FD operators)
                Dx = np.kron(sbp.D, np.eye(self.nen)) # shape (nen^2,nen^2)
                Dy = np.kron(np.eye(self.nen), sbp.D)             
                dxp_dxr = Dx @ self.xy_elem[:,0,:]
                dxp_dyr = Dy @ self.xy_elem[:,0,:]
                dyp_dxr = Dx @ self.xy_elem[:,1,:]
                dyp_dyr = Dy @ self.xy_elem[:,1,:]
                        
                self.metrics[:,0,:] = dyp_dyr
                self.metrics[:,1,:] = - dxp_dyr
                self.metrics[:,2,:] = - dyp_dxr
                self.metrics[:,3,:] = dxp_dxr 
                
                if jac_method=='calculate' or jac_method=='direct':
                    # metric jacobian (determinant) is given by 
                    # Dx_ref@x_phys*Dy_ref@y_phys - Dy_ref@x_phys*Dx_ref@y_phys  
                    self.det_jac = dxp_dxr*dyp_dyr - dxp_dyr*dyp_dxr 
            
            if bdy_metric_method=='exact':
                self.bdy_metrics = np.copy(self.bdy_metrics_exa)
                
            elif bdy_metric_method=='interpolate' or bdy_metric_method=='extrapolate' or bdy_metric_method=='project':
                self.bdy_metrics = np.zeros((self.nen,4,4,self.nelem[0]*self.nelem[1])) 
                #self.bdy_metrics_err = np.zeros((self.nen,4,4,self.nelem[0]*self.nelem[1]))  #TODO: temp
                eye = np.eye(self.nen)
                txbT = np.kron(sbp.tR.reshape((self.nen,1)), eye).T
                txaT = np.kron(sbp.tL.reshape((self.nen,1)), eye).T
                tybT = np.kron(eye, sbp.tR.reshape((self.nen,1))).T
                tyaT = np.kron(eye, sbp.tL.reshape((self.nen,1))).T                
                average = True # for testing things when not averaging surface metrics
                print_diff = True
                maxdiff = 0.
              
                if average:
                    for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
                        for i in [0,1]: # loop over matrix entries
                            Lmetrics = txbT @ self.metrics[:,i,row::self.nelem[1]]
                            Rmetrics = txaT @ self.metrics[:,i,row::self.nelem[1]]
                            if self.nelem[0] != 1:
                                avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                                if print_diff:
                                    maxdiff = max(maxdiff, np.max(abs(Lmetrics[:,:-1] - Rmetrics[:,1:])))
                                self.bdy_metrics[:,0,i,row::self.nelem[1]][:,1:] = avgmetrics
                                self.bdy_metrics[:,1,i,row::self.nelem[1]][:,:-1] = avgmetrics
                                #self.bdy_metrics_err[:,0,i,row::self.nelem[1]][:,1:] = abs(Lmetrics[:,:-1] - Rmetrics[:,1:])
                                #self.bdy_metrics_err[:,1,i,row::self.nelem[1]][:,:-1] = abs(Lmetrics[:,:-1] - Rmetrics[:,1:])
                            if periodic[0]:   
                                avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                                self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] = avgmetrics
                                self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1] = avgmetrics 
                                #self.bdy_metrics_err[:,0,i,row::self.nelem[1]][:,0] = abs(Lmetrics[:,-1] - Rmetrics[:,0])
                                #self.bdy_metrics_err[:,1,i,row::self.nelem[1]][:,-1] = abs(Lmetrics[:,-1] - Rmetrics[:,0])
                                if print_diff:
                                    maxdiff = max(maxdiff, np.max(abs(Lmetrics[:,-1] - Rmetrics[:,0])))
                            else:
                                self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] = Rmetrics[:,0]
                                self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1] = Lmetrics[:,-1]
                        
                    for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                        start = col*self.nelem[0]
                        end = start + self.nelem[1]
                        for i in [2,3]: # loop over matrix entries
                            Lmetrics = tybT @ self.metrics[:,i,start:end]
                            Rmetrics = tyaT @ self.metrics[:,i,start:end]
                            if self.nelem[1] != 1:
                                avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                                if print_diff:
                                    maxdiff = max(maxdiff, np.max(abs(Lmetrics[:,:-1] - Rmetrics[:,1:])))
                                self.bdy_metrics[:,2,i,start:end][:,1:] = avgmetrics
                                self.bdy_metrics[:,3,i,start:end][:,:-1] = avgmetrics
                                #self.bdy_metrics_err[:,2,i,start:end][:,1:] = abs(Lmetrics[:,:-1] - Rmetrics[:,1:])
                                #self.bdy_metrics_err[:,3,i,start:end][:,:-1] = abs(Lmetrics[:,:-1] - Rmetrics[:,1:])
                            if periodic[1]:
                                avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                                self.bdy_metrics[:,2,i,start:end][:,0] = avgmetrics
                                self.bdy_metrics[:,3,i,start:end][:,-1] = avgmetrics 
                                #self.bdy_metrics_err[:,2,i,start:end][:,0] = abs(Lmetrics[:,-1] - Rmetrics[:,0])
                                #self.bdy_metrics_err[:,3,i,start:end][:,-1] = abs(Lmetrics[:,-1] - Rmetrics[:,0])
                                if print_diff:
                                    maxdiff = max(maxdiff, np.max(abs(Lmetrics[:,-1] - Rmetrics[:,0])))
                            else:
                                self.bdy_metrics[:,2,i,start:end][:,0] = Rmetrics[:,0]
                                self.bdy_metrics[:,3,i,start:end][:,-1] = Lmetrics[:,-1]
                                
                    if print_diff:
                        print('Maximum difference of surface metrics when averaging =', maxdiff)
                else:
                    for i in range(4):
                        self.bdy_metrics[:,0,i,:] = txaT @ self.metrics[:,i,:]
                        self.bdy_metrics[:,1,i,:] = txbT @ self.metrics[:,i,:]
                        self.bdy_metrics[:,2,i,:] = tyaT @ self.metrics[:,i,:]
                        self.bdy_metrics[:,3,i,:] = tybT @ self.metrics[:,i,:]
                        
                # set unused components to None to avoid mistakes
                self.ignore_surface_metrics()
                    
            elif bdy_metric_method =='calculate_extrap_x':
                print_diff = True
                maxdiff = 0.
                eye = np.eye(self.nen)
                txbT = np.kron(sbp.tR.reshape((self.nen,1)), eye).T
                txaT = np.kron(sbp.tL.reshape((self.nen,1)), eye).T
                tybT = np.kron(eye, sbp.tR.reshape((self.nen,1))).T
                tyaT = np.kron(eye, sbp.tL.reshape((self.nen,1))).T  
                
                bdy_xy = np.zeros((self.nen,4,2,self.nelem[0]*self.nelem[1]))
                for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
                    for i in [0,1]: # loop over x,y
                        Lxy = txbT @ self.xy_elem[:,i,row::self.nelem[1]]
                        Rxy = txaT @ self.xy_elem[:,i,row::self.nelem[1]]
                        if self.nelem[0] != 1:
                            avgxy = (Lxy[:,:-1] + Rxy[:,1:])/2
                            if print_diff:
                                maxdiff = max(maxdiff, np.max(abs(Lxy[:,:-1] - Rxy[:,1:])))
                            bdy_xy[:,0,i,row::self.nelem[1]][:,1:] = avgxy
                            bdy_xy[:,1,i,row::self.nelem[1]][:,:-1] = avgxy
                        #if periodic[0]:   
                        #    avgxy = (Lxy[:,-1] + Rxy[:,0])/2
                        #    bdy_xy[:,0,i,row::self.nelem[1]][:,0] = avgxy
                        #    bdy_xy[:,1,i,row::self.nelem[1]][:,-1] = avgxy
                        #    if print_diff:
                        #        maxdiff = max(maxdiff, np.max(abs(Lxy[:,-1] - Rxy[:,0])))
                        #else:
                        bdy_xy[:,0,i,row::self.nelem[1]][:,0] = Rxy[:,0]
                        bdy_xy[:,1,i,row::self.nelem[1]][:,-1] = Lxy[:,-1]
                    
                for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                    start = col*self.nelem[0]
                    end = start + self.nelem[1]
                    for i in [0,1]: # loop over x,y
                        Lxy = tybT @ self.xy_elem[:,i,start:end]
                        Rxy = tyaT @ self.xy_elem[:,i,start:end]
                        if self.nelem[1] != 1:
                            avgxy = (Lxy[:,:-1] + Rxy[:,1:])/2
                            if print_diff:
                                maxdiff = max(maxdiff, np.max(abs(Lxy[:,:-1] - Rxy[:,1:])))
                            bdy_xy[:,2,i,start:end][:,1:] = avgxy
                            bdy_xy[:,3,i,start:end][:,:-1] = avgxy
                        #if periodic[1]:
                        #    avgxy = (Lxy[:,-1] + Rxy[:,0])/2
                        #    bdy_xy[:,2,i,start:end][:,0] = avgxy
                        #    bdy_xy[:,3,i,start:end][:,-1] = avgxy
                        #    if print_diff:
                        #        maxdiff = max(maxdiff, np.max(abs(Lxy[:,-1] - Rxy[:,0])))
                        #else:
                        bdy_xy[:,2,i,start:end][:,0] = Rxy[:,0]
                        bdy_xy[:,3,i,start:end][:,-1] = Lxy[:,-1]
                
                if print_diff:
                    print('Maximum difference of surface xy when averaging =', maxdiff)
                            
                self.bdy_metrics = np.zeros((self.nen,4,4,self.nelem[0]*self.nelem[1])) 
                for f in range(4): # loop over facets (left, right, lower, upper)
                    if (f == 0) or (f == 1):
                        self.bdy_metrics[:,f,0,:] = sbp.D @ bdy_xy[:,f,1,:]
                        self.bdy_metrics[:,f,1,:] = - sbp.D @ bdy_xy[:,f,0,:]
                        self.bdy_metrics[:,f,2,:] = None
                        self.bdy_metrics[:,f,3,:] = None
                    elif (f == 2) or (f == 3):
                        self.bdy_metrics[:,f,0,:] = None
                        self.bdy_metrics[:,f,1,:] = None
                        self.bdy_metrics[:,f,2,:] = - sbp.D @ bdy_xy[:,f,1,:]
                        self.bdy_metrics[:,f,3,:] = sbp.D @ bdy_xy[:,f,0,:]
                        
                for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
                    for i in [0,1]: # loop over facets 1,2
                        if periodic[0]:  
                            avg_bdy_met = 0.5*(self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] +
                                               self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1])
                            self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] = avg_bdy_met
                            self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1] = avg_bdy_met
                for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                    start = col*self.nelem[0]
                    end = start + self.nelem[1]
                    for i in [2,3]: # loop over facets 3,4
                        if periodic[1]:
                            avg_bdy_met = 0.5*(self.bdy_metrics[:,2,i,start:end][:,0] +
                                               self.bdy_metrics[:,3,i,start:end][:,-1])
                            self.bdy_metrics[:,2,i,start:end][:,0] = avg_bdy_met
                            self.bdy_metrics[:,3,i,start:end][:,-1] = avg_bdy_met

            else: 
                if bdy_metric_method!='calculate':
                    print("WARNING: Did not understand bdy_metric_method. For 2D, try 'exact', 'calculate', or 'interpolate'.")
                    print("         Defaulting to 'calculate'.")
                self.bdy_metrics = np.zeros((self.nen,4,4,self.nelem[0]*self.nelem[1])) 
                for f in range(4): # loop over facets (left, right, lower, upper)
                    if (f == 0) or (f == 1):
                        self.bdy_metrics[:,f,0,:] = sbp.D @ self.bdy_xy[:,1,f,:]
                        self.bdy_metrics[:,f,1,:] = - sbp.D @ self.bdy_xy[:,0,f,:]
                        self.bdy_metrics[:,f,2,:] = None
                        self.bdy_metrics[:,f,3,:] = None
                    elif (f == 2) or (f == 3):
                        self.bdy_metrics[:,f,0,:] = None
                        self.bdy_metrics[:,f,1,:] = None
                        self.bdy_metrics[:,f,2,:] = - sbp.D @ self.bdy_xy[:,1,f,:]
                        self.bdy_metrics[:,f,3,:] = sbp.D @ self.bdy_xy[:,0,f,:]
        
            
            if use_optz_metrics:  
                from scipy.sparse import lil_matrix
                # overwrite metrics with optimized ones         
                eye = np.eye(self.nen)
                txb = np.kron(sbp.tR.reshape((self.nen,1)), eye)
                txa = np.kron(sbp.tL.reshape((self.nen,1)), eye)
                tyb = np.kron(eye, sbp.tR.reshape((self.nen,1)))
                tya = np.kron(eye, sbp.tL.reshape((self.nen,1)))
                if optz_method == 'essbp' or optz_method == 'default' or optz_method == 'alex' or optz_method == 'generalized':
                    # First optimize surface metrics, then do default optimization
                    #A = np.zeros((self.nelem[0]*self.nelem[1],self.nelem[0]*self.nelem[1]))
                    A = lil_matrix((self.nelem[0]*self.nelem[1],self.nelem[0]*self.nelem[1]), dtype=float)
                    Hperp = np.diag(sbp.H)
                    H2sum = np.sum(Hperp*Hperp)
                    for ix in range(self.nelem[0]):
                        for iy in range(self.nelem[1]):
                            start = ix*self.nelem[1] + iy
                                
                            A[start,start] += 4*H2sum
                            if iy != self.nelem[1]-1:
                                A[start,start+1] += -H2sum
                            elif (iy == self.nelem[1]-1) and periodic[1]:
                                A[start,ix*self.nelem[1]] += -H2sum
                            if iy != 0:
                                A[start,start-1] += -H2sum
                            elif (iy == 0) and periodic[1]:
                                A[start,start+self.nelem[1]-1] += -H2sum
                            if ix != self.nelem[0]-1:
                                A[start,start+self.nelem[1]] += -H2sum
                            elif (ix == self.nelem[0]-1) and periodic[0]:
                                A[start,iy] += -H2sum
                            if ix != 0:
                                A[start,start-self.nelem[1]] += -H2sum
                            elif (ix == 0) and periodic[0]:
                                A[start,(self.nelem[0]-1)*self.nelem[1] + iy] += -H2sum
                    A = A.tocsr(); A.eliminate_zeros();
          
                    for phys_dir in range(2):
                        if phys_dir == 0: # matrix entries for metric terms
                            term = 'x'
                            xm = 0 # l=x, m=x
                            ym = 2 # l=y, m=x
                        else: 
                            term = 'y'
                            xm = 1 # l=x, m=y
                            ym = 3 # l=y, m=y
                            
                        RHS = -np.dot(Hperp, ( self.bdy_metrics[:,1,xm,:] - self.bdy_metrics[:,0,xm,:] \
                                             + self.bdy_metrics[:,3,ym,:] - self.bdy_metrics[:,2,ym,:] ))
              
                        print('Metric Optz: '+term+' surface integral GCL constraints violated by a max of {0:.2g}'.format(np.max(abs(RHS))))
                        if np.max(abs(RHS)) < 2e-16:
                            print('... good enough already. skipping optimization.')
                        else:
                            #if fn.is_pos_def(A):
                            #    print('Check: A is SPD')
                            #else:
                            #    print('Check: A is NOT SPD')
                            if (periodic[0] and periodic[1]):
                                #lam = np.linalg.lstsq(A,RHS,rcond=-1)[0]
                                lam = fn.solve_lin_system(A,RHS,False)
                            else:
                                #lam = np.linalg.solve(A,RHS)
                                lam = fn.solve_lin_system(A,RHS,True)
                            #print('... verify Ax-b=0 solution quality: ', np.max(A@lam - RHS))
                            #print('rank(A) = ', np.linalg.matrix_rank(A))
                            #print('rank([Ab]) = ', np.linalg.matrix_rank(np.c_[A,RHS]))
                            
                                    
                            for ix in range(self.nelem[0]):
                                for iy in range(self.nelem[1]):
                                    elem = ix*self.nelem[1] + iy
    
                                    if iy != self.nelem[1]-1:
                                        self.bdy_metrics[:,3,ym,elem] += Hperp * lam[elem]
                                        self.bdy_metrics[:,2,ym,elem+1] += Hperp * lam[elem]
                                    elif (iy == self.nelem[1]-1) and periodic[1]:
                                        self.bdy_metrics[:,3,ym,elem] += Hperp * lam[elem]
                                        self.bdy_metrics[:,2,ym,ix*self.nelem[1]] += Hperp * lam[elem]
                                    if iy != 0:
                                        self.bdy_metrics[:,2,ym,elem] -= Hperp * lam[elem]
                                        self.bdy_metrics[:,3,ym,elem-1] -= Hperp * lam[elem]
                                    elif (iy == 0) and periodic[1]:
                                        self.bdy_metrics[:,2,ym,elem] -= Hperp * lam[elem]
                                        self.bdy_metrics[:,3,ym,elem+(self.nelem[1]-1)] -= Hperp * lam[elem]
                                    if ix != self.nelem[0]-1:
                                        self.bdy_metrics[:,1,xm,elem] += Hperp * lam[elem]
                                        self.bdy_metrics[:,0,xm,elem+self.nelem[1]] += Hperp * lam[elem]
                                    elif (ix == self.nelem[0]-1) and periodic[0]:
                                        self.bdy_metrics[:,1,xm,elem] += Hperp * lam[elem]
                                        self.bdy_metrics[:,0,xm,iy] += Hperp * lam[elem]
                                    if ix != 0:
                                        self.bdy_metrics[:,0,xm,elem] -= Hperp * lam[elem]
                                        self.bdy_metrics[:,1,xm,elem-self.nelem[1]] -= Hperp * lam[elem]
                                    elif (ix == 0) and periodic[0]:
                                        self.bdy_metrics[:,0,xm,elem] -= Hperp * lam[elem]
                                        self.bdy_metrics[:,1,xm,(self.nelem[0]-1)*self.nelem[1] + iy] -= Hperp * lam[elem]
                                       
                            RHS = -np.dot(Hperp, ( self.bdy_metrics[:,1,xm,:] - self.bdy_metrics[:,0,xm,:] \
                                             + self.bdy_metrics[:,3,ym,:] - self.bdy_metrics[:,2,ym,:] ))
                                
                            print('... largest (single side) correction term to '+term+' surface metrics is {0:.2g}'.format(np.max(abs(lam))*np.max(abs(Hperp))))
                            print('... '+term+' surface integral GCL constraints are now satisfied to {0:.2g}'.format(np.max(abs(RHS))))
                                           
                    
                    # now proceed to the normal optimization procedure
                    if optz_method != 'generalized':
                        optz_method = 'papers'
        
                                    
                if optz_method == 'papers' or optz_method == 'ddrf':
                    QxT = np.kron(sbp.Q, sbp.H).T
                    QyT = np.kron(sbp.H, sbp.Q).T
                    M = np.hstack((QxT,QyT))
                    #Minv = np.linalg.pinv(M, rcond=1e-13)
                    #if np.max(abs(Minv)) > 1e8:
                    #    print('WARNING: There may be an error in Minv of metric optimization. Try a higher rcond.')
                    for phys_dir in range(2):
                        if phys_dir == 0: # matrix entries for metric terms
                            term = 'x'
                            xm = 0 # l=x, m=x
                            ym = 2 # l=y, m=x
                        else: 
                            term = 'y'
                            xm = 1 # l=x, m=y
                            ym = 3 # l=y, m=y
                            
                        c = txb @ sbp.H @ self.bdy_metrics[:,1,xm,:] - txa @ sbp.H @ self.bdy_metrics[:,0,xm,:] \
                          + tyb @ sbp.H @ self.bdy_metrics[:,3,ym,:] - tya @ sbp.H @ self.bdy_metrics[:,2,ym,:]
                        if np.any(abs(np.sum(c,axis=0))>1e-13):
                            print('WARNING: '+term+'surface integral GCL constraint violated by a max of {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                            print('         the c_'+term+' vector in optimization will not add to zero => Optimization will not work!')
                        aex = np.vstack((self.metrics[:,xm,:],self.metrics[:,ym,:]))
                        gcl = M @ aex - c
                        if np.max(abs(gcl)) < 2e-16:
                            a = aex
                        else:
                            #a = aex - Minv @ ( M @ aex - c )
                            #print(np.max(abs(gcl)))
                            cor = np.linalg.lstsq(M, gcl, rcond=1e-13)[0]
                            a = aex - cor
                        print('Metric Optz: modified '+term+' volume metrics by a max amount {0:.2g}'.format(np.max(abs(a - aex))))
                        self.metrics[:,xm,:] = np.copy(a[:self.nen**2,:])
                        self.metrics[:,ym,:] = np.copy(a[self.nen**2:,:])
                     
                elif optz_method == 'diablo': # NOTE: THIS IS WRONG. SEE EMAIL WITH DAVID CRAIG PENNER
                    Dx = np.kron(sbp.D, eye)
                    Dy = np.kron(eye, sbp.D)
                    M = np.hstack((Dx,Dy))
                    Minv = np.linalg.pinv(M, rcond=1e-13)
                    Hinv = np.linalg.inv(np.kron(sbp.H, sbp.H))
                    Ex = txb @ sbp.H @ txb.T - txa @ sbp.H @ txa.T
                    Ey = tyb @ sbp.H @ tyb.T - tya @ sbp.H @ tya.T
                    # first for x dimension
                    # this is the line that is SUS, why use the exact metrics here?
                    c = Hinv @ ( Ex @ self.metrics[:,0,:] + Ey @ self.metrics[:,2,:] \
                      - txb @ sbp.H @ self.bdy_metrics[:,1,0,:] + txa @ sbp.H @ self.bdy_metrics[:,0,0,:] \
                      - tyb @ sbp.H @ self.bdy_metrics[:,3,2,:] + tya @ sbp.H @ self.bdy_metrics[:,2,2,:] )
                    aex = np.vstack((self.metrics[:,0,:],self.metrics[:,2,:]))
                    a = aex - Minv @ ( M @ aex - c )
                    self.metrics[:,0,:] = a[:self.nen**2,:]
                    self.metrics[:,2,:] = a[self.nen**2:,:]
                    # now for y dimension
                    # this is the line that is SUS, why use the exact metrics here?
                    c = Hinv @ ( Ex @ self.metrics[:,1,:] + Ey @ self.metrics[:,3,:] \
                      - txb @ sbp.H @ self.bdy_metrics[:,1,1,:] + txa @ sbp.H @ self.bdy_metrics[:,0,1,:] \
                      - tyb @ sbp.H @ self.bdy_metrics[:,3,3,:] + tya @ sbp.H @ self.bdy_metrics[:,2,3,:] )
                    aex = np.vstack((self.metrics[:,1,:],self.metrics[:,3,:]))
                    a = aex - Minv @ ( M @ aex - c )
                    self.metrics[:,1,:] = a[:self.nen**2,:]
                    self.metrics[:,3,:] = a[self.nen**2:,:]
                    
                elif optz_method == 'generalized': #TODO
                    QxT = np.kron(sbp.Q, sbp.H).T
                    QyT = np.kron(sbp.H, sbp.Q).T
                    M = np.hstack((QxT,QyT))
                    Minv = np.linalg.pinv(M, rcond=1e-13)
                    if np.max(abs(Minv)) > 1e8:
                        print('WARNING: There may be an error in Minv of metric optimization. Try a higher rcond.')
                    for phys_dir in range(2):
                        if phys_dir == 0: # matrix entries for metric terms
                            term = 'x'
                            xm = 0 # l=x, m=x
                            ym = 2 # l=y, m=x
                        else: 
                            term = 'y'
                            xm = 1 # l=x, m=y
                            ym = 3 # l=y, m=y
                            
                        c = txb @ sbp.H @ self.bdy_metrics[:,1,xm,:] - txa @ sbp.H @ self.bdy_metrics[:,0,xm,:] \
                          + tyb @ sbp.H @ self.bdy_metrics[:,3,ym,:] - tya @ sbp.H @ self.bdy_metrics[:,2,ym,:]
                        if np.any(abs(np.sum(c,axis=0))>1e-13):
                            print('WARNING: '+term+'surface integral GCL constraint violated by a max of {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                            print('         the c_'+term+' vector in optimization will not add to zero => Optimization will not work!')
                        aex = np.vstack((self.metrics[:,xm,:],self.metrics[:,ym,:]))
                        a = aex - Minv @ ( M @ aex - c )
                        print('Metric Optz: modified '+term+' volume metrics by a max amount {0:.2g}'.format(np.max(abs(a - aex))))
                        self.metrics[:,xm,:] = np.copy(a[:self.nen**2,:])
                        self.metrics[:,ym,:] = np.copy(a[self.nen**2:,:])

                else:
                    raise Exception("metric optimization method '"+optz_method+"' not understood")
                    
            if jac_method=='exact':
                self.det_jac = self.det_jac_exa
            elif jac_method=='calculate' or jac_method=='direct':
                pass # already done  
            elif jac_method=='deng':
                Dx = np.kron(sbp.D, np.eye(self.nen)) # shape (nen^2,nen^2)
                Dy = np.kron(np.eye(self.nen), sbp.D)             
                self.det_jac = ( Dx @ ( self.xy_elem[:,0,:] * self.metrics[:,0,:] + self.xy_elem[:,1,:] * self.metrics[:,1,:] ) \
                               + Dy @ ( self.xy_elem[:,0,:] * self.metrics[:,2,:] + self.xy_elem[:,1,:] * self.metrics[:,3,:] ))/2
            elif jac_method=='match' or jac_method=='backout':
                self.det_jac = self.metrics[:,0,:] * self.metrics[:,3,:] - self.metrics[:,1,:] * self.metrics[:,2,:]
            else:
                print('ERROR: Did not understant inputted jac_method: ', jac_method)
                print("       Defaulting to 'exact'.")
                self.det_jac = self.det_jac_exa
                    
                
                
        elif self.dim == 3:
            
            if calc_exact_metrics or metric_method.lower()=='exact':
                self.metrics_exa = np.zeros((self.nen**3,9,self.nelem[0]*self.nelem[1]*self.nelem[2])) 
                self.bdy_metrics_exa = np.zeros((self.nen**2,6,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                self.bdy_jac_factor = np.zeros((self.nen**2,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                #self.fac_normals_exa = np.zeros((self.nen**2,6,3,self.nelem[0]*self.nelem[1]*self.nelem[2]))
            
                self.metrics_exa[:,0,:] = self.jac_exa[:,1,1,:] * self.jac_exa[:,2,2,:] - self.jac_exa[:,1,2,:] * self.jac_exa[:,2,1,:]
                self.metrics_exa[:,1,:] = self.jac_exa[:,0,2,:] * self.jac_exa[:,2,1,:] - self.jac_exa[:,0,1,:] * self.jac_exa[:,2,2,:]
                self.metrics_exa[:,2,:] = self.jac_exa[:,0,1,:] * self.jac_exa[:,1,2,:] - self.jac_exa[:,0,2,:] * self.jac_exa[:,1,1,:]
                self.metrics_exa[:,3,:] = self.jac_exa[:,1,2,:] * self.jac_exa[:,2,0,:] - self.jac_exa[:,1,0,:] * self.jac_exa[:,2,2,:]
                self.metrics_exa[:,4,:] = self.jac_exa[:,0,0,:] * self.jac_exa[:,2,2,:] - self.jac_exa[:,0,2,:] * self.jac_exa[:,2,0,:]
                self.metrics_exa[:,5,:] = self.jac_exa[:,0,2,:] * self.jac_exa[:,1,0,:] - self.jac_exa[:,0,0,:] * self.jac_exa[:,1,2,:]
                self.metrics_exa[:,6,:] = self.jac_exa[:,1,0,:] * self.jac_exa[:,2,1,:] - self.jac_exa[:,1,1,:] * self.jac_exa[:,2,0,:]
                self.metrics_exa[:,7,:] = self.jac_exa[:,0,1,:] * self.jac_exa[:,2,0,:] - self.jac_exa[:,0,0,:] * self.jac_exa[:,2,1,:]
                self.metrics_exa[:,8,:] = self.jac_exa[:,0,0,:] * self.jac_exa[:,1,1,:] - self.jac_exa[:,0,1,:] * self.jac_exa[:,1,0,:]
                
                #self.metrics_exa[:,0,:] = self.det_jac_exa * self.jac_inv_exa[:,0,0,:]
                #self.metrics_exa[:,1,:] = self.det_jac_exa * self.jac_inv_exa[:,0,1,:]
                #self.metrics_exa[:,2,:] = self.det_jac_exa * self.jac_inv_exa[:,0,2,:]
                #self.metrics_exa[:,3,:] = self.det_jac_exa * self.jac_inv_exa[:,1,0,:]
                #self.metrics_exa[:,4,:] = self.det_jac_exa * self.jac_inv_exa[:,1,1,:]
                #self.metrics_exa[:,5,:] = self.det_jac_exa * self.jac_inv_exa[:,1,2,:]
                #self.metrics_exa[:,6,:] = self.det_jac_exa * self.jac_inv_exa[:,2,0,:]
                #self.metrics_exa[:,7,:] = self.det_jac_exa * self.jac_inv_exa[:,2,1,:]
                #self.metrics_exa[:,8,:] = self.det_jac_exa * self.jac_inv_exa[:,2,2,:]
                
                for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                    self.bdy_metrics_exa[:,f,0,:] = self.bdy_jac_exa[:,1,1,f,:] * self.bdy_jac_exa[:,2,2,f,:] - self.bdy_jac_exa[:,1,2,f,:] * self.bdy_jac_exa[:,2,1,f,:]
                    self.bdy_metrics_exa[:,f,1,:] = self.bdy_jac_exa[:,0,2,f,:] * self.bdy_jac_exa[:,2,1,f,:] - self.bdy_jac_exa[:,0,1,f,:] * self.bdy_jac_exa[:,2,2,f,:]
                    self.bdy_metrics_exa[:,f,2,:] = self.bdy_jac_exa[:,0,1,f,:] * self.bdy_jac_exa[:,1,2,f,:] - self.bdy_jac_exa[:,0,2,f,:] * self.bdy_jac_exa[:,1,1,f,:]
                    self.bdy_metrics_exa[:,f,3,:] = self.bdy_jac_exa[:,1,2,f,:] * self.bdy_jac_exa[:,2,0,f,:] - self.bdy_jac_exa[:,1,0,f,:] * self.bdy_jac_exa[:,2,2,f,:]
                    self.bdy_metrics_exa[:,f,4,:] = self.bdy_jac_exa[:,0,0,f,:] * self.bdy_jac_exa[:,2,2,f,:] - self.bdy_jac_exa[:,0,2,f,:] * self.bdy_jac_exa[:,2,0,f,:]
                    self.bdy_metrics_exa[:,f,5,:] = self.bdy_jac_exa[:,0,2,f,:] * self.bdy_jac_exa[:,1,0,f,:] - self.bdy_jac_exa[:,0,0,f,:] * self.bdy_jac_exa[:,1,2,f,:]
                    self.bdy_metrics_exa[:,f,6,:] = self.bdy_jac_exa[:,1,0,f,:] * self.bdy_jac_exa[:,2,1,f,:] - self.bdy_jac_exa[:,1,1,f,:] * self.bdy_jac_exa[:,2,0,f,:]
                    self.bdy_metrics_exa[:,f,7,:] = self.bdy_jac_exa[:,0,1,f,:] * self.bdy_jac_exa[:,2,0,f,:] - self.bdy_jac_exa[:,0,0,f,:] * self.bdy_jac_exa[:,2,1,f,:]
                    self.bdy_metrics_exa[:,f,8,:] = self.bdy_jac_exa[:,0,0,f,:] * self.bdy_jac_exa[:,1,1,f,:] - self.bdy_jac_exa[:,0,1,f,:] * self.bdy_jac_exa[:,1,0,f,:]
                
                    #self.bdy_metrics_exa[:,f,0,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,0,f,:]
                    #self.bdy_metrics_exa[:,f,1,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,1,f,:]
                    #self.bdy_metrics_exa[:,f,2,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,2,f,:]
                    #self.bdy_metrics_exa[:,f,3,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,0,f,:]
                    #self.bdy_metrics_exa[:,f,4,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,1,f,:]
                    #self.bdy_metrics_exa[:,f,5,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,2,f,:]
                    #self.bdy_metrics_exa[:,f,6,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,0,f,:]
                    #self.bdy_metrics_exa[:,f,7,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,1,f,:]
                    #self.bdy_metrics_exa[:,f,8,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,2,f,:]

                    if f == 0:
                        nxref = -1
                        nyref = 0
                        nzref = 0
                    elif f == 1:
                        nxref = 1
                        nyref = 0
                        nzref = 0
                    elif f == 2:
                        nxref = 0
                        nyref = -1
                        nzref = 0
                    elif f == 3:
                        nxref = 0
                        nyref = 1
                        nzref = 0
                    elif f == 2:
                        nxref = 0
                        nyref = 0
                        nzref = -1
                    elif f == 3:
                        nxref = 0
                        nyref = 0
                        nzref = 1  
                    x_unnormed = nxref*self.bdy_metrics_exa[:,f,0,:] + nyref*self.bdy_metrics_exa[:,f,3,:] + nzref*self.bdy_metrics_exa[:,f,6,:]
                    y_unnormed = nxref*self.bdy_metrics_exa[:,f,2,:] + nyref*self.bdy_metrics_exa[:,f,4,:] + nzref*self.bdy_metrics_exa[:,f,7,:]
                    z_unnormed = nxref*self.bdy_metrics_exa[:,f,3,:] + nyref*self.bdy_metrics_exa[:,f,5,:] + nzref*self.bdy_metrics_exa[:,f,8,:]
                    norm = np.sqrt(x_unnormed**2 + y_unnormed**2 + y_unnormed**2)
                    #self.fac_normals_exa[:,f,0,:] = x_unnormed / norm
                    #self.fac_normals_exa[:,f,1,:] = y_unnormed / norm
                    #self.fac_normals_exa[:,f,2,:] = z_unnormed / norm
                    self.bdy_jac_factor[:,f,:] = norm
                
            if metric_method.lower()=='exact':
                self.metrics = np.copy(self.metrics_exa)
                
                if jac_method.lower()=='direct' or jac_method.lower()=='calculate':
                    D = sp.lm_to_sp(sbp.D) 
                    Dx = sp.kron_lm_eye(sp.kron_lm_eye(D, self.nen), self.nen)
                    Dy = sp.kron_lm_eye(sp.kron_eye_lm(D, self.nen), self.nen)
                    Dz = sp.kron_eye_lm(sp.kron_eye_lm(D, self.nen), self.nen)
                    dxp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,0,:])
                    dxp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,0,:])
                    dxp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,0,:])
                    dyp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,1,:])
                    dyp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,1,:])
                    dyp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,1,:])
                    dzp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,2,:])
                    dzp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,2,:])
                    dzp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,2,:])
                    # unique (matrix has unique determinant)                            
                    self.det_jac = dxp_dxr*(dyp_dyr*dzp_dzr - dyp_dzr*dzp_dyr) \
                                 - dyp_dxr*(dxp_dyr*dzp_dzr - dxp_dzr*dzp_dyr) \
                                 + dzp_dxr*(dxp_dyr*dyp_dzr - dxp_dzr*dyp_dyr)
            
            elif metric_method.lower()== 'kopriva' or metric_method.lower()== 'kcw' \
                or metric_method.lower()== 'kopriva_extrap' or metric_method.lower()== 'kcw_extrap':
                self.metrics = np.zeros((self.nen**3,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                # do kopriva and yee with lgl operators, then interpolate to lg
                from Source.Disc.MakeSbpOp import MakeSbpOp
                sbp_lgl = MakeSbpOp(sbp.p,'lgl',print_progress=False)
                # the following bases were outputted with the following:
                '''
                basis = []
                for x in ['ones','xi']:
                    for y in ['ones','eta']:
                        for z in ['ones','zeta']:
                            fn = x + '*' + y + '*' + z
                            fn = fn.replace('*ones', '')
                            fn = fn.replace('ones*', '')
                            basis.append(fn)
                print ('= np.vstack((%s)).T' % ', '.join(map(str, basis)))
                '''
                # ones = np.ones(sbp.nn*sbp.nn*sbp.nn)
                # if sbp.p==1:
                #     xi, eta, zeta = np.array(np.meshgrid(sbp.x, sbp.x, sbp.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vsbp = np.vstack((ones, zeta, eta, eta*zeta, xi, xi*zeta, xi*eta, xi*eta*zeta)).T
                #     xi, eta, zeta = np.array(np.meshgrid(sbp_lgl.x, sbp_lgl.x, sbp_lgl.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vlgl = np.vstack((ones, zeta, eta, eta*zeta, xi, xi*zeta, xi*eta, xi*eta*zeta)).T
                # elif sbp.p==2:
                #     xi, eta, zeta = np.array(np.meshgrid(sbp.x, sbp.x, sbp.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vsbp = np.vstack((ones, zeta, zeta**2, eta, eta*zeta, eta*zeta**2, eta**2, eta**2*zeta, eta**2*zeta**2, xi, xi*zeta, xi*zeta**2, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2)).T
                #     xi, eta, zeta = np.array(np.meshgrid(sbp_lgl.x, sbp_lgl.x, sbp_lgl.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vlgl = np.vstack((ones, zeta, zeta**2, eta, eta*zeta, eta*zeta**2, eta**2, eta**2*zeta, eta**2*zeta**2, xi, xi*zeta, xi*zeta**2, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2)).T
                # elif sbp.p==3:
                #     xi, eta, zeta = np.array(np.meshgrid(sbp.x, sbp.x, sbp.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vsbp = np.vstack((ones, zeta, zeta**2, zeta**3, eta, eta*zeta, eta*zeta**2, eta*zeta**3, eta**2, eta**2*zeta, eta**2*zeta**2, eta**2*zeta**3, eta**3, eta**3*zeta, eta**3*zeta**2, eta**3*zeta**3, xi, xi*zeta, xi*zeta**2, xi*zeta**3, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta*zeta**3, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi*eta**2*zeta**3, xi*eta**3, xi*eta**3*zeta, xi*eta**3*zeta**2, xi*eta**3*zeta**3, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*zeta**3, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta*zeta**3, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2, xi**2*eta**2*zeta**3, xi**2*eta**3, xi**2*eta**3*zeta, xi**2*eta**3*zeta**2, xi**2*eta**3*zeta**3, xi**3, xi**3*zeta, xi**3*zeta**2, xi**3*zeta**3, xi**3*eta, xi**3*eta*zeta, xi**3*eta*zeta**2, xi**3*eta*zeta**3, xi**3*eta**2, xi**3*eta**2*zeta, xi**3*eta**2*zeta**2, xi**3*eta**2*zeta**3, xi**3*eta**3, xi**3*eta**3*zeta, xi**3*eta**3*zeta**2, xi**3*eta**3*zeta**3)).T
                #     xi, eta, zeta = np.array(np.meshgrid(sbp_lgl.x, sbp_lgl.x, sbp_lgl.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vlgl = np.vstack((ones, zeta, zeta**2, zeta**3, eta, eta*zeta, eta*zeta**2, eta*zeta**3, eta**2, eta**2*zeta, eta**2*zeta**2, eta**2*zeta**3, eta**3, eta**3*zeta, eta**3*zeta**2, eta**3*zeta**3, xi, xi*zeta, xi*zeta**2, xi*zeta**3, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta*zeta**3, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi*eta**2*zeta**3, xi*eta**3, xi*eta**3*zeta, xi*eta**3*zeta**2, xi*eta**3*zeta**3, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*zeta**3, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta*zeta**3, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2, xi**2*eta**2*zeta**3, xi**2*eta**3, xi**2*eta**3*zeta, xi**2*eta**3*zeta**2, xi**2*eta**3*zeta**3, xi**3, xi**3*zeta, xi**3*zeta**2, xi**3*zeta**3, xi**3*eta, xi**3*eta*zeta, xi**3*eta*zeta**2, xi**3*eta*zeta**3, xi**3*eta**2, xi**3*eta**2*zeta, xi**3*eta**2*zeta**2, xi**3*eta**2*zeta**3, xi**3*eta**3, xi**3*eta**3*zeta, xi**3*eta**3*zeta**2, xi**3*eta**3*zeta**3)).T
                # elif sbp.p==4:
                #     xi, eta, zeta = np.array(np.meshgrid(sbp.x, sbp.x, sbp.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vsbp = np.vstack((ones, zeta, zeta**2, zeta**3, zeta**4, eta, eta*zeta, eta*zeta**2, eta*zeta**3, eta*zeta**4, eta**2, eta**2*zeta, eta**2*zeta**2, eta**2*zeta**3, eta**2*zeta**4, eta**3, eta**3*zeta, eta**3*zeta**2, eta**3*zeta**3, eta**3*zeta**4, eta**4, eta**4*zeta, eta**4*zeta**2, eta**4*zeta**3, eta**4*zeta**4, xi, xi*zeta, xi*zeta**2, xi*zeta**3, xi*zeta**4, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta*zeta**3, xi*eta*zeta**4, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi*eta**2*zeta**3, xi*eta**2*zeta**4, xi*eta**3, xi*eta**3*zeta, xi*eta**3*zeta**2, xi*eta**3*zeta**3, xi*eta**3*zeta**4, xi*eta**4, xi*eta**4*zeta, xi*eta**4*zeta**2, xi*eta**4*zeta**3, xi*eta**4*zeta**4, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*zeta**3, xi**2*zeta**4, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta*zeta**3, xi**2*eta*zeta**4, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2, xi**2*eta**2*zeta**3, xi**2*eta**2*zeta**4, xi**2*eta**3, xi**2*eta**3*zeta, xi**2*eta**3*zeta**2, xi**2*eta**3*zeta**3, xi**2*eta**3*zeta**4, xi**2*eta**4, xi**2*eta**4*zeta, xi**2*eta**4*zeta**2, xi**2*eta**4*zeta**3, xi**2*eta**4*zeta**4, xi**3, xi**3*zeta, xi**3*zeta**2, xi**3*zeta**3, xi**3*zeta**4, xi**3*eta, xi**3*eta*zeta, xi**3*eta*zeta**2, xi**3*eta*zeta**3, xi**3*eta*zeta**4, xi**3*eta**2, xi**3*eta**2*zeta, xi**3*eta**2*zeta**2, xi**3*eta**2*zeta**3, xi**3*eta**2*zeta**4, xi**3*eta**3, xi**3*eta**3*zeta, xi**3*eta**3*zeta**2, xi**3*eta**3*zeta**3, xi**3*eta**3*zeta**4, xi**3*eta**4, xi**3*eta**4*zeta, xi**3*eta**4*zeta**2, xi**3*eta**4*zeta**3, xi**3*eta**4*zeta**4, xi**4, xi**4*zeta, xi**4*zeta**2, xi**4*zeta**3, xi**4*zeta**4, xi**4*eta, xi**4*eta*zeta, xi**4*eta*zeta**2, xi**4*eta*zeta**3, xi**4*eta*zeta**4, xi**4*eta**2, xi**4*eta**2*zeta, xi**4*eta**2*zeta**2, xi**4*eta**2*zeta**3, xi**4*eta**2*zeta**4, xi**4*eta**3, xi**4*eta**3*zeta, xi**4*eta**3*zeta**2, xi**4*eta**3*zeta**3, xi**4*eta**3*zeta**4, xi**4*eta**4, xi**4*eta**4*zeta, xi**4*eta**4*zeta**2, xi**4*eta**4*zeta**3, xi**4*eta**4*zeta**4)).T
                #     xi, eta, zeta = np.array(np.meshgrid(sbp_lgl.x, sbp_lgl.x, sbp_lgl.x)).reshape(3, -1).T[:,[1,0,2]].T
                #     Vlgl = np.vstack((ones, zeta, zeta**2, zeta**3, zeta**4, eta, eta*zeta, eta*zeta**2, eta*zeta**3, eta*zeta**4, eta**2, eta**2*zeta, eta**2*zeta**2, eta**2*zeta**3, eta**2*zeta**4, eta**3, eta**3*zeta, eta**3*zeta**2, eta**3*zeta**3, eta**3*zeta**4, eta**4, eta**4*zeta, eta**4*zeta**2, eta**4*zeta**3, eta**4*zeta**4, xi, xi*zeta, xi*zeta**2, xi*zeta**3, xi*zeta**4, xi*eta, xi*eta*zeta, xi*eta*zeta**2, xi*eta*zeta**3, xi*eta*zeta**4, xi*eta**2, xi*eta**2*zeta, xi*eta**2*zeta**2, xi*eta**2*zeta**3, xi*eta**2*zeta**4, xi*eta**3, xi*eta**3*zeta, xi*eta**3*zeta**2, xi*eta**3*zeta**3, xi*eta**3*zeta**4, xi*eta**4, xi*eta**4*zeta, xi*eta**4*zeta**2, xi*eta**4*zeta**3, xi*eta**4*zeta**4, xi**2, xi**2*zeta, xi**2*zeta**2, xi**2*zeta**3, xi**2*zeta**4, xi**2*eta, xi**2*eta*zeta, xi**2*eta*zeta**2, xi**2*eta*zeta**3, xi**2*eta*zeta**4, xi**2*eta**2, xi**2*eta**2*zeta, xi**2*eta**2*zeta**2, xi**2*eta**2*zeta**3, xi**2*eta**2*zeta**4, xi**2*eta**3, xi**2*eta**3*zeta, xi**2*eta**3*zeta**2, xi**2*eta**3*zeta**3, xi**2*eta**3*zeta**4, xi**2*eta**4, xi**2*eta**4*zeta, xi**2*eta**4*zeta**2, xi**2*eta**4*zeta**3, xi**2*eta**4*zeta**4, xi**3, xi**3*zeta, xi**3*zeta**2, xi**3*zeta**3, xi**3*zeta**4, xi**3*eta, xi**3*eta*zeta, xi**3*eta*zeta**2, xi**3*eta*zeta**3, xi**3*eta*zeta**4, xi**3*eta**2, xi**3*eta**2*zeta, xi**3*eta**2*zeta**2, xi**3*eta**2*zeta**3, xi**3*eta**2*zeta**4, xi**3*eta**3, xi**3*eta**3*zeta, xi**3*eta**3*zeta**2, xi**3*eta**3*zeta**3, xi**3*eta**3*zeta**4, xi**3*eta**4, xi**3*eta**4*zeta, xi**3*eta**4*zeta**2, xi**3*eta**4*zeta**3, xi**3*eta**4*zeta**4, xi**4, xi**4*zeta, xi**4*zeta**2, xi**4*zeta**3, xi**4*zeta**4, xi**4*eta, xi**4*eta*zeta, xi**4*eta*zeta**2, xi**4*eta*zeta**3, xi**4*eta*zeta**4, xi**4*eta**2, xi**4*eta**2*zeta, xi**4*eta**2*zeta**2, xi**4*eta**2*zeta**3, xi**4*eta**2*zeta**4, xi**4*eta**3, xi**4*eta**3*zeta, xi**4*eta**3*zeta**2, xi**4*eta**3*zeta**3, xi**4*eta**3*zeta**4, xi**4*eta**4, xi**4*eta**4*zeta, xi**4*eta**4*zeta**2, xi**4*eta**4*zeta**3, xi**4*eta**4*zeta**4)).T
                # else:
                #     raise Exception('kopriva metrics only set up for p<=4')
                
                # Vlgltosbp = Vsbp @ np.linalg.inv(Vlgl)
                # Vsbptolgl = Vlgl @ np.linalg.inv(Vsbp)
                from Source.Disc.MakeDgOp import MakeDgOp
                Vlgltosbp = MakeDgOp.VandermondeLagrange1D(sbp.x,sbp_lgl.x)
                Vsbptolgl = MakeDgOp.VandermondeLagrange1D(sbp_lgl.x,sbp.x)
                Vlgltosbp = np.kron(np.kron(Vlgltosbp,Vlgltosbp),Vlgltosbp)
                Vsbptolgl = np.kron(np.kron(Vsbptolgl,Vsbptolgl),Vsbptolgl)

                D = sp.lm_to_sp(sbp_lgl.D) 
                Dx = sp.kron_lm_eye(sp.kron_lm_eye(D, self.nen), self.nen)
                Dy = sp.kron_lm_eye(sp.kron_eye_lm(D, self.nen), self.nen)
                Dz = sp.kron_eye_lm(sp.kron_eye_lm(D, self.nen), self.nen) 

                if metric_method.lower()== 'kopriva_extrap' or metric_method.lower()== 'kcw_extrap':
                # The following does NOT produce unique surface values, so we'll need to average
                    xyz_elem = np.einsum('ij,jme->ime', Vsbptolgl, self.xyz_elem)
                    # we need to average now, keeping in mind vertices and edges are shared between more than two elements
                    xyz_elem = self.average_facet_nodes(xyz_elem,sbp_lgl.nn,periodic)
                    x_elem, y_elem, z_elem = xyz_elem[:,0,:], xyz_elem[:,1,:], xyz_elem[:,2,:]
                else:
                    x_unwarped = self.xyz_elem_unwarped[:,0,:]
                    y_unwarped = self.xyz_elem_unwarped[:,1,:]
                    z_unwarped = self.xyz_elem_unwarped[:,2,:]
                    xyz_unwarped_lgl = np.zeros_like(self.xyz_elem_unwarped)
                    xyz_unwarped_lgl[:,0,:] = Vsbptolgl @ x_unwarped
                    xyz_unwarped_lgl[:,1,:] = Vsbptolgl @ y_unwarped
                    xyz_unwarped_lgl[:,2,:] = Vsbptolgl @ z_unwarped
                    xyz_elem = self.warp_mesh_3d(xyz=xyz_unwarped_lgl)
                    x_elem, y_elem, z_elem = xyz_elem[:,0,:], xyz_elem[:,1,:], xyz_elem[:,2,:]
                dxp_dxr = sp.lm_gv(Dx, x_elem)
                dxp_dyr = sp.lm_gv(Dy, x_elem)
                dxp_dzr = sp.lm_gv(Dz, x_elem)
                dyp_dxr = sp.lm_gv(Dx, y_elem)
                dyp_dyr = sp.lm_gv(Dy, y_elem)
                dyp_dzr = sp.lm_gv(Dz, y_elem)
                dzp_dxr = sp.lm_gv(Dx, z_elem)
                dzp_dyr = sp.lm_gv(Dy, z_elem)
                dzp_dzr = sp.lm_gv(Dz, z_elem)
                dXp_dxr = sp.lm_gdiag(Dx, x_elem)
                dXp_dyr = sp.lm_gdiag(Dy, x_elem)
                dXp_dzr = sp.lm_gdiag(Dz, x_elem)
                dYp_dxr = fn.lm_gdiag(Dx, y_elem)
                dYp_dyr = sp.lm_gdiag(Dy, y_elem)
                dYp_dzr = sp.lm_gdiag(Dz, y_elem)
                dZp_dxr = sp.lm_gdiag(Dx, z_elem)
                dZp_dyr = sp.lm_gdiag(Dy, z_elem)
                dZp_dzr = sp.lm_gdiag(Dz, z_elem)

                # self.metrics[:,0,:] = Vlgltosbp @ ( fn.gm_gv(dZp_dzr,dyp_dyr) - fn.gm_gv(dZp_dyr,dyp_dzr) )
                # self.metrics[:,1,:] = Vlgltosbp @ ( fn.gm_gv(dXp_dzr,dzp_dyr) - fn.gm_gv(dXp_dyr,dzp_dzr) )
                # self.metrics[:,2,:] = Vlgltosbp @ ( fn.gm_gv(dYp_dzr,dxp_dyr) - fn.gm_gv(dYp_dyr,dxp_dzr) )
                # self.metrics[:,3,:] = Vlgltosbp @ ( fn.gm_gv(dZp_dxr,dyp_dzr) - fn.gm_gv(dZp_dzr,dyp_dxr) )
                # self.metrics[:,4,:] = Vlgltosbp @ ( fn.gm_gv(dXp_dxr,dzp_dzr) - fn.gm_gv(dXp_dzr,dzp_dxr) )
                # self.metrics[:,5,:] = Vlgltosbp @ ( fn.gm_gv(dYp_dxr,dxp_dzr) - fn.gm_gv(dYp_dzr,dxp_dxr) )
                # self.metrics[:,6,:] = Vlgltosbp @ ( fn.gm_gv(dZp_dyr,dyp_dxr) - fn.gm_gv(dZp_dxr,dyp_dyr) )
                # self.metrics[:,7,:] = Vlgltosbp @ ( fn.gm_gv(dXp_dyr,dzp_dxr) - fn.gm_gv(dXp_dxr,dzp_dyr) )
                # self.metrics[:,8,:] = Vlgltosbp @ ( fn.gm_gv(dYp_dyr,dxp_dxr) - fn.gm_gv(dYp_dxr,dxp_dyr) )
                self.metrics[:,0,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dYp_dyr,dzp_dzr) - sp.gm_gv(dZp_dyr,dyp_dzr) + sp.gm_gv(dZp_dzr,dyp_dyr) - sp.gm_gv(dYp_dzr,dzp_dyr)) )
                self.metrics[:,1,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dZp_dyr,dxp_dzr) - sp.gm_gv(dXp_dyr,dzp_dzr) + sp.gm_gv(dXp_dzr,dzp_dyr) - sp.gm_gv(dZp_dzr,dxp_dyr)) )
                self.metrics[:,2,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dXp_dyr,dyp_dzr) - sp.gm_gv(dYp_dyr,dxp_dzr) + sp.gm_gv(dYp_dzr,dxp_dyr) - sp.gm_gv(dXp_dzr,dyp_dyr)) )
                self.metrics[:,3,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dYp_dzr,dzp_dxr) - sp.gm_gv(dZp_dzr,dyp_dxr) + sp.gm_gv(dZp_dxr,dyp_dzr) - sp.gm_gv(dYp_dxr,dzp_dzr)) )
                self.metrics[:,4,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dZp_dzr,dxp_dxr) - sp.gm_gv(dXp_dzr,dzp_dxr) + sp.gm_gv(dXp_dxr,dzp_dzr) - sp.gm_gv(dZp_dxr,dxp_dzr)) )
                self.metrics[:,5,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dXp_dzr,dyp_dxr) - sp.gm_gv(dYp_dzr,dxp_dxr) + sp.gm_gv(dYp_dxr,dxp_dzr) - sp.gm_gv(dXp_dxr,dyp_dzr)) )
                self.metrics[:,6,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dYp_dxr,dzp_dyr) - sp.gm_gv(dZp_dxr,dyp_dyr) + sp.gm_gv(dZp_dyr,dyp_dxr) - sp.gm_gv(dYp_dyr,dzp_dxr)) )
                self.metrics[:,7,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dZp_dxr,dxp_dyr) - sp.gm_gv(dXp_dxr,dzp_dyr) + sp.gm_gv(dXp_dyr,dzp_dxr) - sp.gm_gv(dZp_dyr,dxp_dxr)) )
                self.metrics[:,8,:] = Vlgltosbp @ ( 0.5*(sp.gm_gv(dXp_dxr,dyp_dyr) - sp.gm_gv(dYp_dxr,dxp_dyr) + sp.gm_gv(dYp_dyr,dxp_dxr) - sp.gm_gv(dXp_dyr,dyp_dxr)) ) 

                
            else:
                self.metrics = np.zeros((self.nen**3,9,self.nelem[0]*self.nelem[1]*self.nelem[2])) 
                if not (metric_method.lower()=='thomaslombard' or metric_method.lower()=='vinokuryee' or metric_method.lower()=='direct'):
                    print("WARNING: Did not understand metric_method. For 3D, try 'exact', 'VinokurYee', or 'ThomasLombard', or 'direct'.")
                    print("         Defaulting to 'VinokurYee'.")
                    metric_method = 'VinokurYee'

                D = sp.lm_to_sp(sbp.D) 
                Dx = sp.kron_lm_eye(sp.kron_lm_eye(D, self.nen), self.nen)
                Dy = sp.kron_lm_eye(sp.kron_eye_lm(D, self.nen), self.nen)
                Dz = sp.kron_eye_lm(sp.kron_eye_lm(D, self.nen), self.nen)
                dxp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,0,:])
                dxp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,0,:])
                dxp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,0,:])
                dyp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,1,:])
                dyp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,1,:])
                dyp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,1,:])
                dzp_dxr = sp.lm_gv(Dx, self.xyz_elem[:,2,:])
                dzp_dyr = sp.lm_gv(Dy, self.xyz_elem[:,2,:])
                dzp_dzr = sp.lm_gv(Dz, self.xyz_elem[:,2,:])
                dXp_dxr = sp.lm_gdiag(Dx, self.xyz_elem[:,0,:])
                dXp_dyr = sp.lm_gdiag(Dy, self.xyz_elem[:,0,:])
                dXp_dzr = sp.lm_gdiag(Dz, self.xyz_elem[:,0,:])
                dYp_dxr = sp.lm_gdiag(Dx, self.xyz_elem[:,1,:])
                dYp_dyr = sp.lm_gdiag(Dy, self.xyz_elem[:,1,:])
                dYp_dzr = sp.lm_gdiag(Dz, self.xyz_elem[:,1,:])
                dZp_dxr = sp.lm_gdiag(Dx, self.xyz_elem[:,2,:])
                dZp_dyr = sp.lm_gdiag(Dy, self.xyz_elem[:,2,:])
                dZp_dzr = sp.lm_gdiag(Dz, self.xyz_elem[:,2,:])
                
                if jac_method.lower()=='direct' or jac_method.lower()=='calculate':
                    # unique (matrix has unique determinant)                            
                    self.det_jac = dxp_dxr*(dyp_dyr*dzp_dzr - dyp_dzr*dzp_dyr) \
                                 - dyp_dxr*(dxp_dyr*dzp_dzr - dxp_dzr*dzp_dyr) \
                                 + dzp_dxr*(dxp_dyr*dyp_dzr - dxp_dzr*dyp_dyr)
                
                if metric_method.lower() == 'thomaslombard':
                    self.metrics[:,0,:] = sp.gm_gv(dZp_dzr,dyp_dyr) - sp.gm_gv(dZp_dyr,dyp_dzr)
                    self.metrics[:,1,:] = sp.gm_gv(dXp_dzr,dzp_dyr) - sp.gm_gv(dXp_dyr,dzp_dzr)
                    self.metrics[:,2,:] = sp.gm_gv(dYp_dzr,dxp_dyr) - sp.gm_gv(dYp_dyr,dxp_dzr)
                    self.metrics[:,3,:] = sp.gm_gv(dZp_dxr,dyp_dzr) - sp.gm_gv(dZp_dzr,dyp_dxr)
                    self.metrics[:,4,:] = sp.gm_gv(dXp_dxr,dzp_dzr) - sp.gm_gv(dXp_dzr,dzp_dxr)
                    self.metrics[:,5,:] = sp.gm_gv(dYp_dxr,dxp_dzr) - sp.gm_gv(dYp_dzr,dxp_dxr)
                    self.metrics[:,6,:] = sp.gm_gv(dZp_dyr,dyp_dxr) - sp.gm_gv(dZp_dxr,dyp_dyr)
                    self.metrics[:,7,:] = sp.gm_gv(dXp_dyr,dzp_dxr) - sp.gm_gv(dXp_dxr,dzp_dyr)
                    self.metrics[:,8,:] = sp.gm_gv(dYp_dyr,dxp_dxr) - sp.gm_gv(dYp_dxr,dxp_dyr) 
                                   
                elif metric_method.lower() == 'vinokuryee':                   
                    self.metrics[:,0,:] = 0.5*(sp.gm_gv(dYp_dyr,dzp_dzr) - sp.gm_gv(dZp_dyr,dyp_dzr) + sp.gm_gv(dZp_dzr,dyp_dyr) - sp.gm_gv(dYp_dzr,dzp_dyr))
                    self.metrics[:,1,:] = 0.5*(sp.gm_gv(dZp_dyr,dxp_dzr) - sp.gm_gv(dXp_dyr,dzp_dzr) + sp.gm_gv(dXp_dzr,dzp_dyr) - sp.gm_gv(dZp_dzr,dxp_dyr))
                    self.metrics[:,2,:] = 0.5*(sp.gm_gv(dXp_dyr,dyp_dzr) - sp.gm_gv(dYp_dyr,dxp_dzr) + sp.gm_gv(dYp_dzr,dxp_dyr) - sp.gm_gv(dXp_dzr,dyp_dyr))
                    self.metrics[:,3,:] = 0.5*(sp.gm_gv(dYp_dzr,dzp_dxr) - sp.gm_gv(dZp_dzr,dyp_dxr) + sp.gm_gv(dZp_dxr,dyp_dzr) - sp.gm_gv(dYp_dxr,dzp_dzr))
                    self.metrics[:,4,:] = 0.5*(sp.gm_gv(dZp_dzr,dxp_dxr) - sp.gm_gv(dXp_dzr,dzp_dxr) + sp.gm_gv(dXp_dxr,dzp_dzr) - sp.gm_gv(dZp_dxr,dxp_dzr))
                    self.metrics[:,5,:] = 0.5*(sp.gm_gv(dXp_dzr,dyp_dxr) - sp.gm_gv(dYp_dzr,dxp_dxr) + sp.gm_gv(dYp_dxr,dxp_dzr) - sp.gm_gv(dXp_dxr,dyp_dzr))
                    self.metrics[:,6,:] = 0.5*(sp.gm_gv(dYp_dxr,dzp_dyr) - sp.gm_gv(dZp_dxr,dyp_dyr) + sp.gm_gv(dZp_dyr,dyp_dxr) - sp.gm_gv(dYp_dyr,dzp_dxr))
                    self.metrics[:,7,:] = 0.5*(sp.gm_gv(dZp_dxr,dxp_dyr) - sp.gm_gv(dXp_dxr,dzp_dyr) + sp.gm_gv(dXp_dyr,dzp_dxr) - sp.gm_gv(dZp_dyr,dxp_dxr))
                    self.metrics[:,8,:] = 0.5*(sp.gm_gv(dXp_dxr,dyp_dyr) - sp.gm_gv(dYp_dxr,dxp_dyr) + sp.gm_gv(dYp_dyr,dxp_dxr) - sp.gm_gv(dXp_dyr,dyp_dxr)) 
                    
                elif metric_method.lower() == 'direct':  
                    self.metrics[:,0,:] = dzp_dzr*dyp_dyr - dzp_dyr*dyp_dzr
                    self.metrics[:,1,:] = dxp_dzr*dzp_dyr - dxp_dyr*dzp_dzr
                    self.metrics[:,2,:] = dyp_dzr*dxp_dyr - dyp_dyr*dxp_dzr
                    self.metrics[:,3,:] = dzp_dxr*dyp_dzr - dzp_dzr*dyp_dxr
                    self.metrics[:,4,:] = dxp_dxr*dzp_dzr - dxp_dzr*dzp_dxr
                    self.metrics[:,5,:] = dyp_dxr*dxp_dzr - dyp_dzr*dxp_dxr
                    self.metrics[:,6,:] = dzp_dyr*dyp_dxr - dzp_dxr*dyp_dyr
                    self.metrics[:,7,:] = dxp_dyr*dzp_dxr - dxp_dxr*dzp_dyr
                    self.metrics[:,8,:] = dyp_dyr*dxp_dxr - dyp_dxr*dxp_dyr 
                        
                    
            if bdy_metric_method.lower()=='exact':
                self.bdy_metrics = np.copy(self.bdy_metrics_exa)
                
            elif bdy_metric_method.lower()=='interpolate' or bdy_metric_method.lower()=='extrapolate' or bdy_metric_method.lower()=='project':
                self.bdy_metrics = np.zeros((self.nen**2,6,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                tR = sp.lm_to_sp(sbp.tR.reshape((self.nen,1)))
                tL = sp.lm_to_sp(sbp.tL.reshape((self.nen,1)))
                txbT = sp.kron_lm_eye(sp.kron_lm_eye(tR, self.nen), self.nen).T()
                txaT = sp.kron_lm_eye(sp.kron_lm_eye(tL, self.nen), self.nen).T()
                tybT = sp.kron_lm_eye(sp.kron_eye_lm(tR, self.nen), self.nen).T()
                tyaT = sp.kron_lm_eye(sp.kron_eye_lm(tL, self.nen), self.nen).T()
                tzbT = sp.kron_eye_lm(sp.kron_eye_lm(tR, self.nen), self.nen).T()
                tzaT = sp.kron_eye_lm(sp.kron_eye_lm(tL, self.nen), self.nen).T()
                skipx = self.nelem[1]*self.nelem[2]
                skipz = self.nelem[0]*self.nelem[1]
                average = True # for testing things when not averaging surface metrics
                print_diff = True
                maxdiff = 0.
                
                if average:
                
                    for rowx in range(skipx):
                        for i in range(3): # loop over matrix entries
                            Lmetrics = sp.lm_gv(txbT, self.metrics[:,i,rowx::skipx])
                            Rmetrics = sp.lm_gv(txaT, self.metrics[:,i,rowx::skipx])
                            if self.nelem[0] != 1:
                                avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,:-1])))
                                self.bdy_metrics[:,0,i,rowx::skipx][:,1:] = avgmetrics
                                self.bdy_metrics[:,1,i,rowx::skipx][:,:-1] = avgmetrics
                            if periodic[0]:   
                                avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,-1])))
                                self.bdy_metrics[:,0,i,rowx::skipx][:,0] = avgmetrics
                                self.bdy_metrics[:,1,i,rowx::skipx][:,-1] = avgmetrics     
                            else:
                                self.bdy_metrics[:,0,i,rowx::skipx][:,0] = Rmetrics[:,0]
                                self.bdy_metrics[:,1,i,rowx::skipx][:,-1] = Lmetrics[:,-1]
                        
                    for coly in range(self.nelem[0]*self.nelem[2]):
                        start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                        end = start + skipx
                        for i in range(3,6): # loop over matrix entries
                            Lmetrics = sp.lm_gv(tybT, self.metrics[:,i,start:end:self.nelem[2]])
                            Rmetrics = sp.lm_gv(tyaT, self.metrics[:,i,start:end:self.nelem[2]])
                            if self.nelem[1] != 1:
                                avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,:-1])))
                                self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,1:] = avgmetrics
                                self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,:-1] = avgmetrics
                            if periodic[1]:
                                avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,-1])))
                                self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,0] = avgmetrics
                                self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,-1] = avgmetrics     
                            else:
                                self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,0] = Rmetrics[:,0]
                                self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,-1] = Lmetrics[:,-1] 
                    
                    for colz in range(skipz):
                        start = colz*self.nelem[2]
                        end = start + self.nelem[2]
                        for i in range(6,9): # loop over matrix entries
                            Lmetrics = sp.lm_gv(tzbT, self.metrics[:,i,start:end])
                            Rmetrics = sp.lm_gv(tzaT, self.metrics[:,i,start:end])
                            if self.nelem[2] != 1:
                                avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,:-1])))
                                self.bdy_metrics[:,4,i,start:end][:,1:] = avgmetrics
                                self.bdy_metrics[:,5,i,start:end][:,:-1] = avgmetrics
                            if periodic[2]:
                                avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                                maxdiff = max(maxdiff, np.max(abs(avgmetrics-Lmetrics[:,-1])))
                                self.bdy_metrics[:,4,i,start:end][:,0] = avgmetrics
                                self.bdy_metrics[:,5,i,start:end][:,-1] = avgmetrics     
                            else:
                                self.bdy_metrics[:,4,i,start:end][:,0] = Rmetrics[:,0]
                                self.bdy_metrics[:,5,i,start:end][:,-1] = Lmetrics[:,-1]
                    
                    if print_diff:
                        print('The boundary metric extrapolations modified by a max of {0:.2g} in averaging.'.format(maxdiff))
                    
                else:
                    for i in range(3):
                        self.bdy_metrics[:,0,i,:] = sp.lm_gv(txaT, self.metrics[:,i,:])
                        self.bdy_metrics[:,1,i,:] = sp.lm_gv(txbT, self.metrics[:,i,:])
                    for i in range(3,6):
                        self.bdy_metrics[:,2,i,:] = sp.lm_gv(tyaT, self.metrics[:,i,:])
                        self.bdy_metrics[:,3,i,:] = sp.lm_gv(tybT, self.metrics[:,i,:])
                    for i in range(6,9):
                        self.bdy_metrics[:,4,i,:] = sp.lm_gv(tzaT, self.metrics[:,i,:])
                        self.bdy_metrics[:,5,i,:] = sp.lm_gv(tzbT, self.metrics[:,i,:])
                        
                # set unused components to None to avoid mistakes
                self.ignore_surface_metrics()
                            
            else: 
                if not (bdy_metric_method.lower()=='thomaslombard' or bdy_metric_method.lower()=='vinokuryee' or bdy_metric_method.lower()=='thomaslombard_extrap'):
                    print("WARNING: Did not understand bdy_metric_method. For 3D, try 'exact', 'VinokurYee', 'ThomasLombard', or 'interpolate'.")
                    print("         Defaulting to 'VinokurYee'.")
                    self.bdy_metric_method = 'VinokurYee'
                self.bdy_metrics = np.zeros((self.nen**2,6,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                D = sp.lm_to_sp(sbp.D)
                D1 = sp.kron_lm_eye(D, self.nen)
                D2 = sp.kron_eye_lm(D, self.nen)
                if bdy_metric_method.lower() == 'thomaslombard':
                    for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                        if (f == 0) or (f == 1):
                            self.bdy_metrics[:,f,0,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,1,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,2,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,3:,:] = None
                        elif (f == 2) or (f == 3):
                            self.bdy_metrics[:,f,:3,:] = None
                            self.bdy_metrics[:,f,3,:] = sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,4,:] = sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,5,:] = sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,6:,:] = None
                        elif (f == 4) or (f == 5):
                            self.bdy_metrics[:,f,:6,:] = None
                            self.bdy_metrics[:,f,6,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,7,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,8,:] = sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:]))
                elif bdy_metric_method.lower() == 'thomaslombard_extrap':
                    bdy_xyz = self.extrapolate_xyz(sbp)
                    self.temp = bdy_xyz
                    # the following is for debugging
                    #for f in range(6):
                    #    print('----- facet', f, '-----')
                    #    print('max difference in x extrapolation:', np.max(abs(bdy_xyz[:,0,f,:]-self.bdy_xyz[:,0,f,:])))
                    #    print('max difference in y extrapolation:', np.max(abs(bdy_xyz[:,1,f,:]-self.bdy_xyz[:,1,f,:])))
                    #    print('max difference in z extrapolation:', np.max(abs(bdy_xyz[:,2,f,:]-self.bdy_xyz[:,2,f,:])))
                    print('Warning: this method is only meant for testing. Domain boundaries are not averaged!')
                    for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                        if (f == 0) or (f == 1):
                            self.bdy_metrics[:,f,0,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,2,f,:]),sp.lm_gv(D1, bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,2,f,:]),sp.lm_gv(D2, bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,1,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,0,f,:]),sp.lm_gv(D1, bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,0,f,:]),sp.lm_gv(D2, bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,2,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,1,f,:]),sp.lm_gv(D1, bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,1,f,:]),sp.lm_gv(D2, bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,3:,:] = None
                        elif (f == 2) or (f == 3):
                            self.bdy_metrics[:,f,:3,:] = None
                            self.bdy_metrics[:,f,3,:] = sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,2,f,:]),sp.lm_gv(D2, bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,2,f,:]),sp.lm_gv(D1, bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,4,:] = sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,0,f,:]),sp.lm_gv(D2, bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,0,f,:]),sp.lm_gv(D1, bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,5,:] = sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,1,f,:]),sp.lm_gv(D2, bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,1,f,:]),sp.lm_gv(D1, bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,6:,:] = None
                        elif (f == 4) or (f == 5):
                            self.bdy_metrics[:,f,:6,:] = None
                            self.bdy_metrics[:,f,6,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,2,f,:]),sp.lm_gv(D1, bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,2,f,:]),sp.lm_gv(D2, bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,7,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,0,f,:]),sp.lm_gv(D1, bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,0,f,:]),sp.lm_gv(D2, bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,8,:] = sp.gm_gv(sp.lm_gdiag(D2, bdy_xyz[:,1,f,:]),sp.lm_gv(D1, bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, bdy_xyz[:,1,f,:]),sp.lm_gv(D2, bdy_xyz[:,0,f,:]))
                elif bdy_metric_method.lower() == 'vinokuryee':
                    for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                        if (f == 0) or (f == 1):
                            self.bdy_metrics[:,f,0,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,1,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,2,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])))
                            self.bdy_metrics[:,f,3:,:] = None
                        elif (f == 2) or (f == 3):
                            self.bdy_metrics[:,f,:3,:] = None
                            self.bdy_metrics[:,f,3,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,4,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,5,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])))
                            self.bdy_metrics[:,f,6:,:] = None
                        elif (f == 4) or (f == 5):
                            self.bdy_metrics[:,f,:6,:] = None
                            self.bdy_metrics[:,f,6,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,7,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,2,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,2,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,8,:] = 0.5*(sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,1,f,:])) - sp.gm_gv(sp.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D2, self.bdy_xyz[:,0,f,:])) \
                                                            +sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,0,f,:])) - sp.gm_gv(sp.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),sp.lm_gv(D1, self.bdy_xyz[:,1,f,:])))
                    self.ignore_surface_metrics()
           
            
            if use_optz_metrics:
                from scipy.sparse import lil_matrix
                assert (bdy_metric_method.lower() != 'vinokuryee' and bdy_metric_method.lower() != 'thomaslombard'),'Must use extrapolated or exact boundary metrics for optimization.'
                # overwrite metrics with optimized ones 
                eye = np.eye(self.nen)
                txb = np.kron(np.kron(sbp.tR.reshape((self.nen,1)), eye), eye)
                txa = np.kron(np.kron(sbp.tL.reshape((self.nen,1)), eye), eye)
                tyb = np.kron(np.kron(eye, sbp.tR.reshape((self.nen,1))), eye)
                tya = np.kron(np.kron(eye, sbp.tL.reshape((self.nen,1))), eye)
                tzb = np.kron(np.kron(eye, eye), sbp.tR.reshape((self.nen,1)))
                tza = np.kron(np.kron(eye, eye), sbp.tL.reshape((self.nen,1)))
                if optz_method == 'essbp' or optz_method == 'default' or optz_method == 'alex' or optz_method == 'generalized':
                    # First optimize surface metrics, then do default optimization
                    
                    #A = np.zeros((self.nelem[0]*self.nelem[1]*self.nelem[2],self.nelem[0]*self.nelem[1]*self.nelem[2]))
                    A = lil_matrix((self.nelem[0]*self.nelem[1]*self.nelem[2],self.nelem[0]*self.nelem[1]*self.nelem[2]), dtype=float)
                    Hperp = np.diag(np.kron(sbp.H,sbp.H))
                    H2sum = np.sum(Hperp*Hperp)
                    for ix in range(self.nelem[0]):
                        for iy in range(self.nelem[1]):
                            for iz in range(self.nelem[2]):
                                start = (ix*self.nelem[1] + iy)*self.nelem[2] + iz
                    
                                A[start,start] += 6*H2sum
                                if iz != self.nelem[2]-1:
                                    A[start,start+1] += -H2sum
                                elif (iz == self.nelem[2]-1) and periodic[2]:
                                    A[start,(ix*self.nelem[1] + iy)*self.nelem[2]] += -H2sum
                                if iz != 0:
                                    A[start,start-1] += -H2sum
                                elif (iz == 0) and periodic[2]:
                                    A[start,start+self.nelem[2]-1] += -H2sum
                                if iy != self.nelem[1]-1:
                                    A[start,start+self.nelem[2]] += -H2sum
                                elif (iy == self.nelem[1]-1) and periodic[1]:
                                    A[start,ix*self.nelem[1]*self.nelem[2] + iz] += -H2sum
                                if iy != 0:
                                    A[start,start-self.nelem[2]] += -H2sum
                                elif (iy == 0) and periodic[1]:
                                    A[start,start+(self.nelem[1]-1)*self.nelem[2]] += -H2sum
                                if ix != self.nelem[0]-1:
                                    A[start,start+self.nelem[1]*self.nelem[2]] += -H2sum
                                elif (ix == self.nelem[0]-1) and periodic[0]:
                                    A[start,iy*self.nelem[2] + iz] += -H2sum
                                if ix != 0:
                                    A[start,start-self.nelem[1]*self.nelem[2]] += -H2sum
                                elif (ix == 0) and periodic[0]:
                                    A[start,((self.nelem[0]-1)*self.nelem[1] + iy)*self.nelem[2] + iz] += -H2sum
                    A = A.tocsr(); A.eliminate_zeros();
                    
                    for phys_dir in range(3):
                        if phys_dir == 0: # matrix entries for metric terms
                            term = 'x'
                            xm = 0 # l=x, m=x
                            ym = 3 # l=y, m=x
                            zm = 6 # l=z, m=x
                        elif phys_dir == 1: 
                            term = 'y'
                            xm = 1 # l=x, m=y
                            ym = 4 # l=y, m=y
                            zm = 7 # l=z, m=x
                        else: 
                            term = 'z'
                            xm = 2 # l=x, m=z
                            ym = 5 # l=y, m=z
                            zm = 8 # l=z, m=z
                            
                        RHS = -np.dot(Hperp, ( self.bdy_metrics[:,1,xm,:] - self.bdy_metrics[:,0,xm,:] \
                                             + self.bdy_metrics[:,3,ym,:] - self.bdy_metrics[:,2,ym,:] \
                                             + self.bdy_metrics[:,5,zm,:] - self.bdy_metrics[:,4,zm,:] ))
                        
                        print('Metric Optz: '+term+' surface integral GCL constraints violated by a max of {0:.2g}'.format(np.max(abs(RHS))))
                        if np.max(abs(RHS)) < 2e-16:
                            print('... good enough already. skipping optimization.')
                        else:
                            if (periodic[0] and periodic[1] and periodic[2]):
                                #if fn.is_pos_def(A):
                                #    print('Check: A is SPD')
                                #else:
                                #    print('Check: A is NOT SPD')
                                #    print('size of A is', A.shape)
                                #    print('rank of A is', int(np.linalg.matrix_rank(A)))
                                #print(np.linalg.eigvals(A))
                                lam = fn.solve_lin_system(A,RHS,False)
                                #lam = np.linalg.lstsq(A,RHS,rcond=-1)[0]
                                #lam2 = np.linalg.solve(A,RHS)
                                #print('max difference in solve is ', np.max(abs(lam-lam2)))
                            else:
                                lam = fn.solve_lin_system(A,RHS,True)
                                #lam = np.linalg.solve(A,RHS)
                            #print('... verify Ax-b=0 solution quality: ', np.max(A@lam - RHS))
                            #print('rank(A) = ', np.linalg.matrix_rank(A))
                            #print('rank([Ab]) = ', np.linalg.matrix_rank(np.c_[A,RHS]))
                            
                            for ix in range(self.nelem[0]):
                                for iy in range(self.nelem[1]):
                                    for iz in range(self.nelem[2]):
                                        elem = (ix*self.nelem[1] + iy)*self.nelem[2] + iz
    
                                        if iz != self.nelem[2]-1:
                                            self.bdy_metrics[:,5,zm,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,4,zm,elem+1] += Hperp * lam[elem]
                                        elif (iz == self.nelem[2]-1) and periodic[2]:
                                            self.bdy_metrics[:,5,zm,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,4,zm,(ix*self.nelem[1] + iy)*self.nelem[2]] += Hperp * lam[elem]
                                        if iz != 0:
                                            self.bdy_metrics[:,4,zm,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,5,zm,elem-1] -= Hperp * lam[elem]
                                        elif (iz == 0) and periodic[2]:
                                            self.bdy_metrics[:,4,zm,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,5,zm,elem+self.nelem[2]-1] -= Hperp * lam[elem]
                                        if iy != self.nelem[1]-1:
                                            self.bdy_metrics[:,3,ym,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,2,ym,elem+self.nelem[2]] += Hperp * lam[elem]
                                        elif (iy == self.nelem[1]-1) and periodic[1]:
                                            self.bdy_metrics[:,3,ym,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,2,ym,ix*self.nelem[1]*self.nelem[2] + iz] += Hperp * lam[elem]
                                        if iy != 0:
                                            self.bdy_metrics[:,2,ym,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,3,ym,elem-self.nelem[2]] -= Hperp * lam[elem]
                                        elif (iy == 0) and periodic[1]:
                                            self.bdy_metrics[:,2,ym,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,3,ym,elem+(self.nelem[1]-1)*self.nelem[2]] -= Hperp * lam[elem]
                                        if ix != self.nelem[0]-1:
                                            self.bdy_metrics[:,1,xm,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,0,xm,elem+self.nelem[1]*self.nelem[2]] += Hperp * lam[elem]
                                        elif (ix == self.nelem[0]-1) and periodic[0]:
                                            self.bdy_metrics[:,1,xm,elem] += Hperp * lam[elem]
                                            self.bdy_metrics[:,0,xm,iy*self.nelem[2] + iz] += Hperp * lam[elem]
                                        if ix != 0:
                                            self.bdy_metrics[:,0,xm,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,1,xm,elem-self.nelem[1]*self.nelem[2]] -= Hperp * lam[elem]
                                        elif (ix == 0) and periodic[0]:
                                            self.bdy_metrics[:,0,xm,elem] -= Hperp * lam[elem]
                                            self.bdy_metrics[:,1,xm,((self.nelem[0]-1)*self.nelem[1] + iy)*self.nelem[2] + iz] -= Hperp * lam[elem]
                                       
                            RHS = -np.dot(Hperp, ( self.bdy_metrics[:,1,xm,:] - self.bdy_metrics[:,0,xm,:] \
                                             + self.bdy_metrics[:,3,ym,:] - self.bdy_metrics[:,2,ym,:] \
                                             + self.bdy_metrics[:,5,zm,:] - self.bdy_metrics[:,4,zm,:] ))
                                
                            print('... largest (single side) correction term to '+term+' surface metrics is {0:.2g}'.format(np.max(abs(lam))*np.max(abs(Hperp))))
                            print('... '+term+' surface integral GCL constraints are now satisfied to {0:.2g}'.format(np.max(abs(RHS))))
                                         
                    
                    # now proceed to the normal optimization procedure
                    if optz_method != 'generalized':
                        optz_method = 'papers'
                    
                if optz_method == 'papers' or optz_method == 'ddrf':
                    Hperp = np.kron(sbp.H,sbp.H)
                    QxT = np.kron(np.kron(sbp.Q, sbp.H), sbp.H).T
                    QyT = np.kron(np.kron(sbp.H, sbp.Q), sbp.H).T
                    QzT = np.kron(np.kron(sbp.H, sbp.H), sbp.Q).T
                    M = np.hstack((QxT,QyT,QzT))
                    #print('size of M is', M.shape)
                    #print('rank of M is', int(np.linalg.matrix_rank(M)))
                    #Minv = np.linalg.pinv(M, rcond=1e-13)
                    #if np.max(abs(Minv)) > 1e8:
                    #    print('WARNING: There may be an error in Minv of metric optimization. Try a higher rcond.')
                    # first for x dimension
                    c = txb @ Hperp @ self.bdy_metrics[:,1,0,:] - txa @ Hperp @ self.bdy_metrics[:,0,0,:] \
                      + tyb @ Hperp @ self.bdy_metrics[:,3,3,:] - tya @ Hperp @ self.bdy_metrics[:,2,3,:] \
                      + tzb @ Hperp @ self.bdy_metrics[:,5,6,:] - tza @ Hperp @ self.bdy_metrics[:,4,6,:]
                    if np.any(abs(np.sum(c,axis=0))>1e-12):
                        print('WARNING: c_x vector in optimized metric computation does not add to zero.')
                        print('         max value (element) of sum is {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                        print('         Surface integrals in x do not hold discretely.')
                    aex = np.vstack((self.metrics[:,0,:],self.metrics[:,3,:],self.metrics[:,6,:]))
                    #a = aex - Minv @ ( M @ aex - c )
                    a = aex - np.linalg.lstsq(M, M @ aex - c, rcond=1e-13)[0]
                    print('... metric optimization modified x-metrics by a maximum of {0:.2g}'.format(np.max(abs(a - aex))))
                    #print('TEMP: testing free stream - max is {0:.2g}'.format(np.max(abs(M @ a - c ))))
                    self.metrics[:,0,:] = a[:self.nen**3,:]
                    self.metrics[:,3,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,6,:] = a[2*self.nen**3:,:]
                    # now for y dimension
                    c = txb @ Hperp @ self.bdy_metrics[:,1,1,:] - txa @ Hperp @ self.bdy_metrics[:,0,1,:] \
                      + tyb @ Hperp @ self.bdy_metrics[:,3,4,:] - tya @ Hperp @ self.bdy_metrics[:,2,4,:] \
                      + tzb @ Hperp @ self.bdy_metrics[:,5,7,:] - tza @ Hperp @ self.bdy_metrics[:,4,7,:]
                    if np.any(abs(np.sum(c,axis=0))>1e-12):
                        print('WARNING: c_y vector in optimized metric computation does not add to zero.')
                        print('         max value (element) of sum is {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                        print('         Surface integrals in y do not hold discretely.')
                    aex = np.vstack((self.metrics[:,1,:],self.metrics[:,4,:],self.metrics[:,7,:]))
                    #a = aex - Minv @ ( M @ aex - c )
                    a = aex - np.linalg.lstsq(M, M @ aex - c, rcond=1e-13)[0]
                    print('... metric optimization modified y-metrics by a maximum of {0:.2g}'.format(np.max(abs(a - aex))))
                    #print('TEMP: testing free stream - max is {0:.2g}'.format(np.max(abs(M @ a - c ))))
                    self.metrics[:,1,:] = a[:self.nen**3,:]
                    self.metrics[:,4,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,7,:] = a[2*self.nen**3:,:]  
                    # now for z dimension
                    c = txb @ Hperp @ self.bdy_metrics[:,1,2,:] - txa @ Hperp @ self.bdy_metrics[:,0,2,:] \
                      + tyb @ Hperp @ self.bdy_metrics[:,3,5,:] - tya @ Hperp @ self.bdy_metrics[:,2,5,:] \
                      + tzb @ Hperp @ self.bdy_metrics[:,5,8,:] - tza @ Hperp @ self.bdy_metrics[:,4,8,:]
                    if np.any(abs(np.sum(c,axis=0))>1e-12):
                        print('WARNING: c_z vector in optimized metric computation does not add to zero.')
                        print('         max value (element) of sum is {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                        print('         Surface integrals in y do not hold discretely.')
                    aex = np.vstack((self.metrics[:,2,:],self.metrics[:,5,:],self.metrics[:,8,:]))
                    #a = aex - Minv @ ( M @ aex - c )
                    a = aex - np.linalg.lstsq(M, M @ aex - c, rcond=1e-13)[0]
                    print('... metric optimization modified z-metrics by a maximum of {0:.2g}'.format(np.max(abs(a - aex))))
                    #print('TEMP: testing free stream - max is {0:.2g}'.format(np.max(abs(M @ a - c ))))
                    self.metrics[:,2,:] = a[:self.nen**3,:]
                    self.metrics[:,5,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,8,:] = a[2*self.nen**3:,:]

                elif optz_method == 'diablo': # TODO: I THINK THIS IS WRONG
                    Hperp = np.kron(sbp.H,sbp.H)
                    Dx = np.kron(np.kron(sbp.D, eye), eye)
                    Dy = np.kron(np.kron(eye, sbp.D), eye)
                    Dz = np.kron(np.kron(eye, eye), sbp.D)
                    M = np.hstack((Dx,Dy,Dz))
                    Minv = np.linalg.pinv(M, rcond=1e-13)
                    Hinv = np.linalg.inv(np.kron(np.kron(sbp.H, sbp.H),sbp.H))
                    Ex = txb @ Hperp @ txb.T - txa @ Hperp @ txa.T
                    Ey = tyb @ Hperp @ tyb.T - tya @ Hperp @ tya.T
                    Ez = tzb @ Hperp @ tzb.T - tza @ Hperp @ tza.T
                    # first for x dimension
                    # this is the line that is SUS, why use the exact metrics here?
                    c = Hinv @ ( Ex @ self.metrics[:,0,:] + Ey @ self.metrics[:,3,:] + Ez @ self.metrics[:,6,:] \
                      - txb @ Hperp @ self.bdy_metrics[:,1,0,:] + txa @ Hperp @ self.bdy_metrics[:,0,0,:] \
                      - tyb @ Hperp @ self.bdy_metrics[:,1,3,:] + tya @ Hperp @ self.bdy_metrics[:,0,3,:] \
                      - tzb @ Hperp @ self.bdy_metrics[:,1,6,:] + tza @ Hperp @ self.bdy_metrics[:,0,6,:] )
                    aex = np.vstack((self.metrics[:,0,:],self.metrics[:,3,:],self.metrics[:,6,:]))
                    a = aex - Minv @ ( M @ aex - c )
                    self.metrics[:,0,:] = a[:self.nen**3,:]
                    self.metrics[:,3,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,6,:] = a[2*self.nen**3:,:]
                    # now for y dimension
                    # this is the line that is SUS, why use the exact metrics here?
                    c = Hinv @ ( Ex @ self.metrics[:,1,:] + Ey @ self.metrics[:,4,:] + Ez @ self.metrics[:,7,:] \
                      - txb @ Hperp @ self.bdy_metrics[:,3,1,:] + txa @ Hperp @ self.bdy_metrics[:,2,1,:] \
                      - tyb @ Hperp @ self.bdy_metrics[:,3,4,:] + tya @ Hperp @ self.bdy_metrics[:,2,4,:] \
                      - tzb @ Hperp @ self.bdy_metrics[:,3,7,:] + tza @ Hperp @ self.bdy_metrics[:,2,7,:] )
                    aex = np.vstack((self.metrics[:,1,:],self.metrics[:,4,:],self.metrics[:,7,:]))
                    a = aex - Minv @ ( M @ aex - c )
                    self.metrics[:,1,:] = a[:self.nen**3,:]
                    self.metrics[:,4,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,7,:] = a[2*self.nen**3:,:]
                    # now for z dimension
                    # this is the line that is SUS, why use the exact metrics here?
                    c = Hinv @ ( Ex @ self.metrics[:,2,:] + Ey @ self.metrics[:,5,:] + Ez @ self.metrics[:,8,:] \
                      - txb @ Hperp @ self.bdy_metrics[:,5,2,:] + txa @ Hperp @ self.bdy_metrics[:,4,2,:] \
                      - tyb @ Hperp @ self.bdy_metrics[:,5,5,:] + tya @ Hperp @ self.bdy_metrics[:,4,5,:] \
                      - tzb @ Hperp @ self.bdy_metrics[:,5,8,:] + tza @ Hperp @ self.bdy_metrics[:,4,8,:] )
                    aex = np.vstack((self.metrics[:,2,:],self.metrics[:,5,:],self.metrics[:,8,:]))
                    a = aex - Minv @ ( M @ aex - c )
                    self.metrics[:,2,:] = a[:self.nen**3,:]
                    self.metrics[:,5,:] = a[self.nen**3:2*self.nen**3,:]
                    self.metrics[:,8,:] = a[2*self.nen**3:,:]
                else:
                    print('WARNING: Optimization procedure not understood. Skipping Optimization.')
                    

            if jac_method=='exact':
                self.det_jac = self.det_jac_exa
            elif jac_method=='calculate' or jac_method=='direct':
                pass # already done  
            elif jac_method=='deng':
                eye = np.eye(self.nen)
                Dx = np.kron(np.kron(sbp.D, eye), eye)
                Dy = np.kron(np.kron(eye, sbp.D), eye)
                Dz = np.kron(np.kron(eye, eye), sbp.D)            
                self.det_jac = ( Dx @ ( self.xyz_elem[:,0,:] * self.metrics[:,0,:] + self.xyz_elem[:,1,:] * self.metrics[:,1,:] + self.xyz_elem[:,2,:] * self.metrics[:,2,:] ) \
                               + Dy @ ( self.xyz_elem[:,0,:] * self.metrics[:,3,:] + self.xyz_elem[:,1,:] * self.metrics[:,4,:] + self.xyz_elem[:,2,:] * self.metrics[:,5,:] ) \
                               + Dz @ ( self.xyz_elem[:,0,:] * self.metrics[:,6,:] + self.xyz_elem[:,1,:] * self.metrics[:,7,:] + self.xyz_elem[:,2,:] * self.metrics[:,8,:] ))/3
            elif jac_method=='match' or jac_method=='backout':
                self.det_jac = np.sqrt( self.metrics[:,8,:] * (self.metrics[:,0,:] * self.metrics[:,4,:] - self.metrics[:,1,:] * self.metrics[:,3,:]) \
                                       -self.metrics[:,7,:] * (self.metrics[:,0,:] * self.metrics[:,5,:] - self.metrics[:,2,:] * self.metrics[:,3,:]) \
                                       +self.metrics[:,6,:] * (self.metrics[:,1,:] * self.metrics[:,5,:] - self.metrics[:,2,:] * self.metrics[:,4,:]))
                self.det_jac = np.nan_to_num(self.det_jac, copy=False)
            else:
                print('ERROR: Did not understant inputted jac_method: ', jac_method)
                print("       Defaulting to 'exact'.")
                self.det_jac = self.det_jac_exa



        self.det_jac_inv = 1/self.det_jac
        if np.any(abs(self.det_jac - self.det_jac_exa) > 1e-12):
            print('WARNING: The Metric Jacobian is not exact. The max difference is {0:.2g}'.format(np.max(abs(self.det_jac - self.det_jac_exa))))
            if metric_method != 'exact':
                print("         Consider using exact Metric Jacobian and Invariants (set 'metric_method':'exact' in settings).")
        if np.any(self.det_jac < 0):
            print('WARNING: There are negative jacobians at {0} nodes! The min value is {1:.2g}'.format(len(np.argwhere(self.det_jac<0)), np.min(self.det_jac)))
                
        # compute unit normals on facets
        if self.dim == 2:
            self.fac_normals = np.zeros((self.nen,4,2,self.nelem[0]*self.nelem[1]))
            for f in range(4):
                if f == 0:
                    nxref = -1
                    nyref = 0
                elif f == 1:
                    nxref = 1
                    nyref = 0
                elif f == 2:
                    nxref = 0
                    nyref = -1
                elif f == 3:
                    nxref = 0
                    nyref = 1                
                x_unnormed = nxref*self.bdy_metrics[:,f,0,:] + nyref*self.bdy_metrics[:,f,2,:]
                y_unnormed = nxref*self.bdy_metrics[:,f,1,:] + nyref*self.bdy_metrics[:,f,3,:]
                norm = np.sqrt(x_unnormed**2 + y_unnormed**2)
                self.fac_normals[:,f,0,:] = x_unnormed / norm
                self.fac_normals[:,f,1,:] = y_unnormed / norm   
                
        elif self.dim == 3:
            self.fac_normals = np.zeros((self.nen**2,6,3,self.nelem[0]*self.nelem[1]*self.nelem[2]))
            for f in range(6):
                if f == 0:
                    nxref = -1
                    nyref = 0
                    nzref = 0
                elif f == 1:
                    nxref = 1
                    nyref = 0
                    nzref = 0
                elif f == 2:
                    nxref = 0
                    nyref = -1
                    nzref = 0
                elif f == 3:
                    nxref = 0
                    nyref = 1
                    nzref = 0
                elif f == 2:
                    nxref = 0
                    nyref = 0
                    nzref = -1
                elif f == 3:
                    nxref = 0
                    nyref = 0
                    nzref = 1  
                x_unnormed = nxref*self.bdy_metrics[:,f,0,:] + nyref*self.bdy_metrics[:,f,3,:] + nzref*self.bdy_metrics[:,f,6,:]
                y_unnormed = nxref*self.bdy_metrics[:,f,2,:] + nyref*self.bdy_metrics[:,f,4,:] + nzref*self.bdy_metrics[:,f,7,:]
                z_unnormed = nxref*self.bdy_metrics[:,f,3,:] + nyref*self.bdy_metrics[:,f,5,:] + nzref*self.bdy_metrics[:,f,8,:]
                norm = np.sqrt(x_unnormed**2 + y_unnormed**2 + y_unnormed**2)
                self.fac_normals[:,f,0,:] = x_unnormed / norm
                self.fac_normals[:,f,1,:] = y_unnormed / norm
                self.fac_normals[:,f,2,:] = z_unnormed / norm
                
    def ignore_surface_metrics(self):
        ''' set the unused components to None '''
        if self.dim == 1:
            print('nothing to do in 1D')
            
        elif self.dim == 2:
            for f in range(4): # loop over facets (left, right, lower, upper)
                if (f == 0) or (f == 1):
                    self.bdy_metrics[:,f,2,:] = None
                    self.bdy_metrics[:,f,3,:] = None
                elif (f == 2) or (f == 3):
                    self.bdy_metrics[:,f,0,:] = None
                    self.bdy_metrics[:,f,1,:] = None
            
        else:
            for f in range(6):
                if (f == 0) or (f == 1):
                    self.bdy_metrics[:,f,3,:] = None
                    self.bdy_metrics[:,f,4,:] = None
                    self.bdy_metrics[:,f,5,:] = None
                    self.bdy_metrics[:,f,6,:] = None
                    self.bdy_metrics[:,f,7,:] = None
                    self.bdy_metrics[:,f,8,:] = None
                elif (f == 2) or (f == 3):
                    self.bdy_metrics[:,f,0,:] = None
                    self.bdy_metrics[:,f,1,:] = None
                    self.bdy_metrics[:,f,2,:] = None
                    self.bdy_metrics[:,f,6,:] = None
                    self.bdy_metrics[:,f,7,:] = None
                    self.bdy_metrics[:,f,8,:] = None
                elif (f == 4) or (f == 5):
                    self.bdy_metrics[:,f,0,:] = None
                    self.bdy_metrics[:,f,1,:] = None
                    self.bdy_metrics[:,f,2,:] = None
                    self.bdy_metrics[:,f,3,:] = None
                    self.bdy_metrics[:,f,4,:] = None
                    self.bdy_metrics[:,f,5,:] = None

    def extrapolate_xyz(self,sbp,nen=None,average=True,print_diff=True):
        ''' extrapolate the boundary xyz values '''
        if nen is None: nen = self.nen
        if self.dim == 1:
            raise Exception('Extrapolation of xyz values is not supported yet for 1D.')
        elif self.dim == 2:
            raise Exception('Extrapolation of xyz values is not supported yet for 2D.')
        elif self.dim == 3:
            bdy_xyz = np.zeros((nen**2,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
            eye = np.eye(nen)
            txbT = np.kron(np.kron(sbp.tR.reshape((nen,1)), eye), eye).T
            txaT = np.kron(np.kron(sbp.tL.reshape((nen,1)), eye), eye).T
            tybT = np.kron(np.kron(eye, sbp.tR.reshape((nen,1))), eye).T
            tyaT = np.kron(np.kron(eye, sbp.tL.reshape((nen,1))), eye).T
            tzbT = np.kron(np.kron(eye, eye), sbp.tR.reshape((nen,1))).T
            tzaT = np.kron(np.kron(eye, eye), sbp.tL.reshape((nen,1))).T
            skipx = self.nelem[1]*self.nelem[2]
            skipz = self.nelem[0]*self.nelem[1]
            
            if average:
                maxdiff = 0.
            
                for rowx in range(skipx):
                    for i in range(3): # loop over matrix entries
                        bdy_xyzR = txaT @ self.xyz_elem[:,i,rowx::skipx]
                        bdy_xyzL = txbT @ self.xyz_elem[:,i,rowx::skipx]
                        if self.nelem[0] != 1:
                            avgbdy_xyz = (bdy_xyzL[:,:-1] + bdy_xyzR[:,1:])/2
                            if print_diff: maxdiff = max(maxdiff, np.max(abs(avgbdy_xyz-bdy_xyzL[:,:-1])))
                            bdy_xyz[:,i,0,rowx::skipx][:,1:] = avgbdy_xyz
                            bdy_xyz[:,i,1,rowx::skipx][:,:-1] = avgbdy_xyz 
                        bdy_xyz[:,i,0,rowx::skipx][:,0] = bdy_xyzR[:,0]
                        bdy_xyz[:,i,1,rowx::skipx][:,-1] = bdy_xyzL[:,-1]
                
                for coly in range(self.nelem[0]*self.nelem[2]):
                    start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                    end = start + skipx
                    for i in range(3): # loop over matrix entries
                        bdy_xyzL = tybT @ self.xyz_elem[:,i,start:end:self.nelem[2]]
                        bdy_xyzR = tyaT @ self.xyz_elem[:,i,start:end:self.nelem[2]]
                        if self.nelem[1] != 1:
                            avgbdy_xyz = (bdy_xyzL[:,:-1] + bdy_xyzR[:,1:])/2
                            if print_diff: maxdiff = max(maxdiff, np.max(abs(avgbdy_xyz-bdy_xyzL[:,:-1])))
                            bdy_xyz[:,i,2,start:end:self.nelem[2]][:,1:] = avgbdy_xyz
                            bdy_xyz[:,i,3,start:end:self.nelem[2]][:,:-1] = avgbdy_xyz    
                        bdy_xyz[:,i,2,start:end:self.nelem[2]][:,0] = bdy_xyzR[:,0]
                        bdy_xyz[:,i,3,start:end:self.nelem[2]][:,-1] = bdy_xyzL[:,-1] 
                
                for colz in range(skipz):
                    start = colz*self.nelem[2]
                    end = start + self.nelem[2]
                    for i in range(3): # loop over matrix entries
                        bdy_xyzL = tzbT @ self.xyz_elem[:,i,start:end]
                        bdy_xyzR = tzaT @ self.xyz_elem[:,i,start:end]
                        if self.nelem[2] != 1:
                            avgbdy_xyz = (bdy_xyzL[:,:-1] + bdy_xyzR[:,1:])/2
                            if print_diff: maxdiff = max(maxdiff, np.max(abs(avgbdy_xyz-bdy_xyzL[:,:-1])))
                            bdy_xyz[:,i,4,start:end][:,1:] = avgbdy_xyz
                            bdy_xyz[:,i,5,start:end][:,:-1] = avgbdy_xyz
                        bdy_xyz[:,i,4,start:end][:,0] = bdy_xyzR[:,0]
                        bdy_xyz[:,i,5,start:end][:,-1] = bdy_xyzL[:,-1]
                
                if print_diff:
                    print('The boundary xyz extrapolations modified by a max of {0:.2g} in averaging.'.format(maxdiff))
                
            else:
                for i in range(3):
                    bdy_xyz[:,i,0,:] = txaT @ self.xyz_elem[:,i,:]
                    bdy_xyz[:,i,1,:] = txbT @ self.xyz_elem[:,i,:]
                    bdy_xyz[:,i,2,:] = tyaT @ self.xyz_elem[:,i,:]
                    bdy_xyz[:,i,3,:] = tybT @ self.xyz_elem[:,i,:]
                    bdy_xyz[:,i,4,:] = tzaT @ self.xyz_elem[:,i,:]
                    bdy_xyz[:,i,5,:] = tzbT @ self.xyz_elem[:,i,:]
            
            return bdy_xyz
                
    def check_surface_metrics(self):
        ''' check that the surface metrics on either side of an interface are equal '''
        if self.dim == 1:
            print('nothing to do in 1D')
            
        elif self.dim == 2:
            maxint = [0,0,0,0]
            maxbdy = [0,0,0,0]
            
            for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
                for i in range(2): # loop over matrix entries
                    diff = abs(self.bdy_metrics[:,0,i,row::self.nelem[1]][:,1:] - self.bdy_metrics[:,1,i,row::self.nelem[1]][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] - self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1])
                    maxbdy[i] = max(maxbdy[i],np.max(diff))
                
            for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                for i in range(2,4): # loop over matrix entries
                    diff = abs(self.bdy_metrics[:,2,i,start:end][:,1:] - self.bdy_metrics[:,3,i,start:end][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,2,i,start:end][:,0] - self.bdy_metrics[:,3,i,start:end][:,-1]) 
                    maxbdy[i] = max(maxbdy[i],np.max(diff))

            print('Max diff of surface metrics entry dxi(1)/dx(1) on interior: ', maxint[0])
            print('Max diff of surface metrics entry dxi(1)/dx(2) on interior: ', maxint[1])
            print('Max diff of surface metrics entry dxi(2)/dx(1) on interior: ', maxint[2])
            print('Max diff of surface metrics entry dxi(2)/dx(2) on interior: ', maxint[3])
            print('Max diff of surface metrics entry dxi(1)/dx(1) on boundary: ', maxbdy[0])
            print('Max diff of surface metrics entry dxi(1)/dx(2) on boundary: ', maxbdy[1])
            print('Max diff of surface metrics entry dxi(2)/dx(1) on boundary: ', maxbdy[2])
            print('Max diff of surface metrics entry dxi(2)/dx(2) on boundary: ', maxbdy[3])
            
        elif self.dim == 3:
            maxint = [0,0,0,0,0,0,0,0,0]
            maxbdy = [0,0,0,0,0,0,0,0,0]
            
            skipx = self.nelem[1]*self.nelem[2]
            for row in range(skipx):
                for i in range(3): # loop over matrix entries
                    diff = abs(self.bdy_metrics[:,0,i,row::skipx][:,1:] - self.bdy_metrics[:,1,i,row::skipx][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,0,i,row::skipx][:,0] - self.bdy_metrics[:,1,i,row::skipx][:,-1])
                    maxbdy[i] = max(maxbdy[i],np.max(diff))
                    
            for coly in range(self.nelem[0]*self.nelem[2]):
                start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                end = start + self.nelem[1]*self.nelem[2]
                for i in range(3,6):
                    diff = abs(self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,1:] - self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,0] - self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,-1])
                    maxbdy[i] = max(maxbdy[i],np.max(diff))
            
            for colz in range(self.nelem[0]*self.nelem[1]):
                start = colz*self.nelem[2]
                end = start + self.nelem[2]
                for i in range(6,9):
                    diff = abs(self.bdy_metrics[:,4,i,start:end][:,1:] - self.bdy_metrics[:,5,i,start:end][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,4,i,start:end][:,0] - self.bdy_metrics[:,5,i,start:end][:,-1])
                    maxbdy[i] = max(maxbdy[i],np.max(diff))

            print('Max diff of surface metrics entry dxi(1)/dx(1) on interior: ', maxint[0])
            print('Max diff of surface metrics entry dxi(1)/dx(2) on interior: ', maxint[1])
            print('Max diff of surface metrics entry dxi(1)/dx(3) on interior: ', maxint[2])
            print('Max diff of surface metrics entry dxi(2)/dx(1) on interior: ', maxint[3])
            print('Max diff of surface metrics entry dxi(2)/dx(2) on interior: ', maxint[4])
            print('Max diff of surface metrics entry dxi(2)/dx(3) on interior: ', maxint[5])
            print('Max diff of surface metrics entry dxi(3)/dx(1) on interior: ', maxint[6])
            print('Max diff of surface metrics entry dxi(3)/dx(2) on interior: ', maxint[7])
            print('Max diff of surface metrics entry dxi(3)/dx(3) on interior: ', maxint[8])
            print('Max diff of surface metrics entry dxi(1)/dx(1) on boundary: ', maxbdy[0])
            print('Max diff of surface metrics entry dxi(1)/dx(2) on boundary: ', maxbdy[1])
            print('Max diff of surface metrics entry dxi(1)/dx(3) on boundary: ', maxbdy[2])
            print('Max diff of surface metrics entry dxi(2)/dx(1) on boundary: ', maxbdy[3])
            print('Max diff of surface metrics entry dxi(2)/dx(2) on boundary: ', maxbdy[4])
            print('Max diff of surface metrics entry dxi(2)/dx(3) on boundary: ', maxbdy[5])
            print('Max diff of surface metrics entry dxi(3)/dx(1) on boundary: ', maxbdy[6])
            print('Max diff of surface metrics entry dxi(3)/dx(2) on boundary: ', maxbdy[7])
            print('Max diff of surface metrics entry dxi(3)/dx(3) on boundary: ', maxbdy[8])
            
    def check_inv_metrics(self,sbp):
        ''' tests the consistency of the inverse metrics, dxphys_dxref from exact values to SBP computed values '''
        eye = np.eye(self.nen)
        if self.dim == 2:
            Dx = np.kron(sbp.D, eye)
            Dy = np.kron(eye, sbp.D)  
            jac = np.copy(self.jac_exa)
            jac[:,0,0,:] = Dx @ self.xy_elem[:,0,:]
            jac[:,0,1,:] = Dy @ self.xy_elem[:,0,:]
            jac[:,1,0,:] = Dx @ self.xy_elem[:,1,:]
            jac[:,1,1,:] = Dy @ self.xy_elem[:,1,:]
        else:  
            Dx = np.kron(np.kron(sbp.D, eye), eye)
            Dy = np.kron(np.kron(eye, sbp.D), eye)
            Dz = np.kron(np.kron(eye, eye), sbp.D)   
            jac = np.copy(self.jac_exa)
            jac[:,0,0,:] = Dx @ self.xyz_elem[:,0,:]
            jac[:,0,1,:] = Dy @ self.xyz_elem[:,0,:]
            jac[:,0,2,:] = Dz @ self.xyz_elem[:,0,:]
            jac[:,1,0,:] = Dx @ self.xyz_elem[:,1,:]
            jac[:,1,1,:] = Dy @ self.xyz_elem[:,1,:]
            jac[:,1,2,:] = Dz @ self.xyz_elem[:,1,:]
            jac[:,2,0,:] = Dx @ self.xyz_elem[:,2,:]
            jac[:,2,1,:] = Dy @ self.xyz_elem[:,2,:]
            jac[:,2,2,:] = Dz @ self.xyz_elem[:,2,:]
        return np.mean(np.mean(abs(jac-self.jac_exa),axis=0),axis=2)
        

    def average_facet_nodes(self,xyz_elem,nen,periodic):
        ''' loops over all the elements and averages the vertices, edges, and facets 
        ASSUMES that the xyz_elem has values located on all boundary nodes (i.e. vertices and edges)'''
        xyz_elem_new = np.copy(xyz_elem)
        if self.dim == 1:
            raise Exception('Averaging of xyz facet values is not supported yet for 1D.')
        elif self.dim == 2:
            # the convention is starting at bottom left, y is looped over first, then x. (and rows first then cols)
            blv = 0 # bottom left vertex index
            tlv = nen-1 # top left vertex index
            brv = nen*nen-nen # bottom right vertex index
            trv = nen*nen-1 # top right vertex index
            def elem(row, col):
                return col*self.nelem[1] + row

            # loop over rows, look at bottom left vertex and average over all surrounding elements.
            for col in range(self.nelem[0]):
                for row in range(self.nelem[1]):
                    # average the bottom left vertex
                    e = elem(row, col)
                    vtx_val = xyz_elem[blv,:,e]
                    fix = 0.
                    vtcs = [[blv,e,fix]]
                    # element to the left (bottom right vertex)
                    if col == 0:
                        if periodic[0]:
                            e2 = elem(row,self.nelem[0]-1)
                            fix = np.array([self.dom_len[0],0.])
                            vtx_val += (xyz_elem[brv,:,e2] - fix)
                            vtcs.append([brv,e2,fix])
                    else:
                        e2 = elem(row,col-1)
                        vtx_val += xyz_elem[brv,:,e2]
                        vtcs.append([brv,e2,0.])
                    # element below (top left vertex)
                    if row == 0:
                        if periodic[1]:
                            e2 = elem(self.nelem[1]-1,col)
                            fix = np.array([0.,self.dom_len[1]])
                            vtx_val += (xyz_elem[tlv,:,e2] - fix)
                            vtcs.append([tlv,e2,fix])
                    else:
                        e2 = elem(row-1,col)
                        vtx_val += xyz_elem[tlv,:,e2]
                        vtcs.append([tlv,e2,0.])
                    # element to the bottom-left diagonal (top right vertex)
                    if row == 0 and col == 0:
                        if periodic[0] and periodic[1]:
                            fix = np.array([self.dom_len[0],self.dom_len[1]])
                            vtx_val += (xyz_elem[trv,:,-1] - fix)
                            vtcs.append([trv,-1,fix])
                    elif row == 0:
                        if periodic[1]:
                            e2 = elem(self.nelem[1]-1,col-1)
                            fix = np.array([0.,self.dom_len[1]])
                            vtx_val += (xyz_elem[trv,:,e2] - fix)
                            vtcs.append([trv,e2,fix])
                    elif col == 0:
                        if periodic[0]:
                            e2 = elem(row-1,self.nelem[0]-1)
                            fix = np.array([self.dom_len[0],0.])
                            vtx_val += (xyz_elem[trv,:,e2] - fix)
                            vtcs.append([trv,e2,fix])
                    else:
                        e2 = elem(row-1,col-1)
                        vtx_val += xyz_elem[trv,:,e2]
                        vtcs.append([trv,e2,0.])
                    # average the vertices
                    vtx_val /= len(vtcs)
                    for vtx in vtcs:
                        #print('nidx:',vtx[0],'eidx:',vtx[1],'fix:',vtx[2],':',xyz_elem[vtx[0],:,vtx[1]]-(vtx_val+vtx[2]))
                        xyz_elem_new[vtx[0],:,vtx[1]] = vtx_val + vtx[2]

                    # now we average the remaining (non-vertex) facet nodes on the left edge
                    if col == 0:
                        if periodic[0]:
                            e2 = elem(row,self.nelem[0]-1)
                            fix = np.array([self.dom_len[0],0.])
                            vtx_val = 0.5*(xyz_elem[blv+1:tlv,:,e] + (xyz_elem[brv+1:trv,:,e2] - fix))
                            xyz_elem_new[blv+1:tlv,:,e] = vtx_val
                            xyz_elem_new[brv+1:trv,:,e2] = vtx_val + fix
                    else:
                        e2 = elem(row,col-1)
                        vtx_val = 0.5*(xyz_elem[blv+1:tlv,:,e] + xyz_elem[brv+1:trv,:,e2])
                        xyz_elem_new[blv+1:tlv,:,e] = vtx_val
                        xyz_elem_new[brv+1:trv,:,e2] = vtx_val

                    # now we average the remaining (non-vertex) facet nodes on the bottom edge
                    if row == 0:
                        if periodic[1]:
                            e2 = elem(self.nelem[1]-1,col)
                            fix = np.array([0.,self.dom_len[1]])
                            vtx_val = 0.5*(xyz_elem[blv+nen:brv:nen,:,e] + (xyz_elem[tlv+nen:trv:nen,:,e2] - fix))
                            xyz_elem_new[blv+nen:brv:nen,:,e] = vtx_val
                            xyz_elem_new[tlv+nen:trv:nen,:,e2] = vtx_val + fix
                    else:
                        e2 = elem(row-1,col)
                        vtx_val = 0.5*(xyz_elem[blv+nen:brv:nen,:,e] + xyz_elem[tlv+nen:trv:nen,:,e2])
                        xyz_elem_new[blv+nen:brv:nen,:,e] = vtx_val
                        xyz_elem_new[tlv+nen:trv:nen,:,e2] = vtx_val

        elif self.dim == 3:
            # indexing helpers: node (i,j,k) with k fastest; element (row, col, sli) with sli fastest
            def node(i, j, k):
                return i*nen*nen + j*nen + k
            def elem(row, col, sli):
                return col*self.nelem[1]*self.nelem[2] + row*self.nelem[2] + sli

            Lx, Ly, Lz = self.dom_len

            for col in range(self.nelem[0]):
                for row in range(self.nelem[1]):
                    for sli in range(self.nelem[2]):
                        e = elem(row, col, sli)

                        # 1) Average the corner at (i,j,k)=(0,0,0) over up to 8 neighbors
                        vlist = []  # entries: [local_node_idx, elem_index, fix_vector]
                        # self
                        vlist.append([node(0,0,0), e, 0.])

                        # enumerate neighbor offsets in {0,-1}^3 excluding (0,0,0)
                        for dx in (0, -1):
                            for dy in (0, -1):
                                for dz in (0, -1):
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nx, ny, nz = col, row, sli
                                    fx = fy = fz = 0.0
                                    # handle x neighbor
                                    if dx == -1:
                                        if col == 0:
                                            if not periodic[0]:
                                                continue
                                            nx = self.nelem[0]-1
                                            fx = Lx
                                        else:
                                            nx = col-1
                                    # handle y neighbor
                                    if dy == -1:
                                        if row == 0:
                                            if not periodic[1]:
                                                continue
                                            ny = self.nelem[1]-1
                                            fy = Ly
                                        else:
                                            ny = row-1
                                    # handle z neighbor
                                    if dz == -1:
                                        if sli == 0:
                                            if not periodic[2]:
                                                continue
                                            nz = self.nelem[2]-1
                                            fz = Lz
                                        else:
                                            nz = sli-1

                                    e2 = elem(ny, nx, nz)
                                    i2 = node(nen-1 if dx == -1 else 0,
                                              nen-1 if dy == -1 else 0,
                                              nen-1 if dz == -1 else 0)
                                    vlist.append([i2, e2, np.array([fx, fy, fz])])

                        vtx_val = np.copy(xyz_elem[node(0,0,0),:,e])
                        for (i2, e2, fix) in vlist[1:]:
                            vtx_val += (xyz_elem[i2,:,e2] - fix)
                        vtx_val /= len(vlist)
                        for (i2, e2, fix) in vlist:
                            xyz_elem_new[i2,:,e2] = vtx_val + (fix if isinstance(fix, np.ndarray) else 0.)

                        # 2) Average remaining non-vertex nodes on three canonical edges
                        # Edge along +x from (0,0,0): i=1..nen-2, j=0, k=0. Share with (-y), (-z), (-y,-z)
                        if self.nelem[1] > 0 and self.nelem[2] > 0:
                            ii = np.arange(1, nen-1)
                            idx_self = ii*nen*nen + 0*nen + 0
                            # neighbors in -y, -z, -y-z
                            have_y = (row > 0) or periodic[1]
                            have_z = (sli > 0) or periodic[2]

                            if have_y or have_z:
                                vals = xyz_elem[idx_self,:,e]
                                cnt = 1

                                if have_y:
                                    if row == 0:
                                        e_y = elem(self.nelem[1]-1, col, sli)
                                        fy = Ly
                                    else:
                                        e_y = elem(row-1, col, sli)
                                        fy = 0.0
                                    idx_y = ii*nen*nen + (nen-1)*nen + 0
                                    vals = vals + (xyz_elem[idx_y,:,e_y] - np.array([0.0, fy, 0.0]))
                                    cnt += 1

                                if have_z:
                                    if sli == 0:
                                        e_z = elem(row, col, self.nelem[2]-1)
                                        fz = Lz
                                    else:
                                        e_z = elem(row, col, sli-1)
                                        fz = 0.0
                                    idx_z = ii*nen*nen + 0*nen + (nen-1)
                                    vals = vals + (xyz_elem[idx_z,:,e_z] - np.array([0.0, 0.0, fz]))
                                    cnt += 1

                                if have_y and have_z:
                                    if row == 0:
                                        ry = self.nelem[1]-1; fy = Ly
                                    else:
                                        ry = row-1; fy = 0.0
                                    if sli == 0:
                                        rz = self.nelem[2]-1; fz = Lz
                                    else:
                                        rz = sli-1; fz = 0.0
                                    e_yz = elem(ry, col, rz)
                                    idx_yz = ii*nen*nen + (nen-1)*nen + (nen-1)
                                    vals = vals + (xyz_elem[idx_yz,:,e_yz] - np.array([0.0, fy, fz]))
                                    cnt += 1

                                vals /= cnt
                                # write back to all contributing elements
                                xyz_elem_new[idx_self,:,e] = vals
                                if have_y:
                                    xyz_elem_new[idx_y,:,e_y] = vals + np.array([0.0, fy, 0.0])
                                if have_z:
                                    xyz_elem_new[idx_z,:,e_z] = vals + np.array([0.0, 0.0, fz])
                                if have_y and have_z:
                                    xyz_elem_new[idx_yz,:,e_yz] = vals + np.array([0.0, fy, fz])

                        # Edge along +y from (0,0,0): j=1..nen-2, i=0, k=0. Share with (-x), (-z), (-x,-z)
                        if self.nelem[0] > 0 and self.nelem[2] > 0:
                            jj = np.arange(1, nen-1)
                            idx_self = 0*nen*nen + jj*nen + 0
                            have_x = (col > 0) or periodic[0]
                            have_z = (sli > 0) or periodic[2]

                            if have_x or have_z:
                                vals = xyz_elem[idx_self,:,e]
                                cnt = 1

                                if have_x:
                                    if col == 0:
                                        e_x = elem(row, self.nelem[0]-1, sli)
                                        fx = Lx
                                    else:
                                        e_x = elem(row, col-1, sli)
                                        fx = 0.0
                                    idx_x = (nen-1)*nen*nen + jj*nen + 0
                                    vals = vals + (xyz_elem[idx_x,:,e_x] - np.array([fx, 0.0, 0.0]))
                                    cnt += 1

                                if have_z:
                                    if sli == 0:
                                        e_z = elem(row, col, self.nelem[2]-1)
                                        fz = Lz
                                    else:
                                        e_z = elem(row, col, sli-1)
                                        fz = 0.0
                                    idx_z = 0*nen*nen + jj*nen + (nen-1)
                                    vals = vals + (xyz_elem[idx_z,:,e_z] - np.array([0.0, 0.0, fz]))
                                    cnt += 1

                                if have_x and have_z:
                                    if col == 0:
                                        cx = self.nelem[0]-1; fx = Lx
                                    else:
                                        cx = col-1; fx = 0.0
                                    if sli == 0:
                                        cz = self.nelem[2]-1; fz = Lz
                                    else:
                                        cz = sli-1; fz = 0.0
                                    e_xz = elem(row, cx, cz)
                                    idx_xz = (nen-1)*nen*nen + jj*nen + (nen-1)
                                    vals = vals + (xyz_elem[idx_xz,:,e_xz] - np.array([fx, 0.0, fz]))
                                    cnt += 1

                                vals /= cnt
                                xyz_elem_new[idx_self,:,e] = vals
                                if have_x:
                                    xyz_elem_new[idx_x,:,e_x] = vals + np.array([fx, 0.0, 0.0])
                                if have_z:
                                    xyz_elem_new[idx_z,:,e_z] = vals + np.array([0.0, 0.0, fz])
                                if have_x and have_z:
                                    xyz_elem_new[idx_xz,:,e_xz] = vals + np.array([fx, 0.0, fz])

                        # Edge along +z from (0,0,0): k=1..nen-2, i=0, j=0. Share with (-x), (-y), (-x,-y)
                        if self.nelem[0] > 0 and self.nelem[1] > 0:
                            kk = np.arange(1, nen-1)
                            idx_self = 0*nen*nen + 0*nen + kk
                            have_x = (col > 0) or periodic[0]
                            have_y = (row > 0) or periodic[1]

                            if have_x or have_y:
                                vals = xyz_elem[idx_self,:,e]
                                cnt = 1

                                if have_x:
                                    if col == 0:
                                        e_x = elem(row, self.nelem[0]-1, sli)
                                        fx = Lx
                                    else:
                                        e_x = elem(row, col-1, sli)
                                        fx = 0.0
                                    idx_x = (nen-1)*nen*nen + 0*nen + kk
                                    vals = vals + (xyz_elem[idx_x,:,e_x] - np.array([fx, 0.0, 0.0]))
                                    cnt += 1

                                if have_y:
                                    if row == 0:
                                        e_y = elem(self.nelem[1]-1, col, sli)
                                        fy = Ly
                                    else:
                                        e_y = elem(row-1, col, sli)
                                        fy = 0.0
                                    idx_y = 0*nen*nen + (nen-1)*nen + kk
                                    vals = vals + (xyz_elem[idx_y,:,e_y] - np.array([0.0, fy, 0.0]))
                                    cnt += 1

                                if have_x and have_y:
                                    if col == 0:
                                        cx = self.nelem[0]-1; fx = Lx
                                    else:
                                        cx = col-1; fx = 0.0
                                    if row == 0:
                                        cy = self.nelem[1]-1; fy = Ly
                                    else:
                                        cy = row-1; fy = 0.0
                                    e_xy = elem(cy, cx, sli)
                                    idx_xy = (nen-1)*nen*nen + (nen-1)*nen + kk
                                    vals = vals + (xyz_elem[idx_xy,:,e_xy] - np.array([fx, fy, 0.0]))
                                    cnt += 1

                                vals /= cnt
                                xyz_elem_new[idx_self,:,e] = vals
                                if have_x:
                                    xyz_elem_new[idx_x,:,e_x] = vals + np.array([fx, 0.0, 0.0])
                                if have_y:
                                    xyz_elem_new[idx_y,:,e_y] = vals + np.array([0.0, fy, 0.0])
                                if have_x and have_y:
                                    xyz_elem_new[idx_xy,:,e_xy] = vals + np.array([fx, fy, 0.0])

                        # 3) Average remaining (non-edge, non-vertex) nodes on three canonical faces
                        # Left face i=0 with j=1..nen-2, k=1..nen-2 shared with (-x)
                        if (col > 0) or periodic[0]:
                            if col == 0:
                                e_x = elem(row, self.nelem[0]-1, sli)
                                fx = Lx
                            else:
                                e_x = elem(row, col-1, sli)
                                fx = 0.0
                            for j in range(1, nen-1):
                                ids_self = j*nen + np.arange(1, nen-1)
                                ids_x = (nen-1)*nen*nen + j*nen + np.arange(1, nen-1)
                                vals = 0.5*(xyz_elem[ids_self,:,e] + (xyz_elem[ids_x,:,e_x] - np.array([fx, 0.0, 0.0])))
                                xyz_elem_new[ids_self,:,e] = vals
                                xyz_elem_new[ids_x,:,e_x] = vals + np.array([fx, 0.0, 0.0])

                        # Bottom face j=0 with i=1..nen-2, k=1..nen-2 shared with (-y)
                        if (row > 0) or periodic[1]:
                            if row == 0:
                                e_y = elem(self.nelem[1]-1, col, sli)
                                fy = Ly
                            else:
                                e_y = elem(row-1, col, sli)
                                fy = 0.0
                            for i in range(1, nen-1):
                                ids_self = i*nen*nen + np.arange(1, nen-1)
                                ids_y = i*nen*nen + (nen-1)*nen + np.arange(1, nen-1)
                                vals = 0.5*(xyz_elem[ids_self,:,e] + (xyz_elem[ids_y,:,e_y] - np.array([0.0, fy, 0.0])))
                                xyz_elem_new[ids_self,:,e] = vals
                                xyz_elem_new[ids_y,:,e_y] = vals + np.array([0.0, fy, 0.0])

                        # Back face k=0 with i=1..nen-2, j=1..nen-2 shared with (-z)
                        if (sli > 0) or periodic[2]:
                            if sli == 0:
                                e_z = elem(row, col, self.nelem[2]-1)
                                fz = Lz
                            else:
                                e_z = elem(row, col, sli-1)
                                fz = 0.0
                            for i in range(1, nen-1):
                                ids_self = i*nen*nen + np.arange(1, nen-1)*nen + 0
                                ids_z = i*nen*nen + np.arange(1, nen-1)*nen + (nen-1)
                                vals = 0.5*(xyz_elem[ids_self,:,e] + (xyz_elem[ids_z,:,e_z] - np.array([0.0, 0.0, fz])))
                                xyz_elem_new[ids_self,:,e] = vals
                                xyz_elem_new[ids_z,:,e_z] = vals + np.array([0.0, 0.0, fz])

        return xyz_elem_new