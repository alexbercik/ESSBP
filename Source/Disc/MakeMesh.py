#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:54:23 2020

@author: andremarchildon
"""

import numpy as np
import matplotlib.pyplot as plt
import Source.Methods.Functions as fn


class MakeMesh:
    
    def __init__(self, dim, xmin, xmax, 
                 nelem, x_op, warp_factor=0, warp_type='default'):
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
        self.warp_factor = warp_factor
        self.warp_type = warp_type

        ''' Additional terms '''
        
        print('... Building Mesh')
        
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
        self.det_jac_inv_exa = np.ones((self.nen, self.nelem))/elem_length
        
        self.bdy_jac_exa = np.ones((1,2,self.nelem))*elem_length
        self.bdy_det_jac_exa = np.ones((2,self.nelem))*elem_length
        self.bdy_jac_inv_exa = np.ones((1,2,self.nelem))/elem_length
            

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
        self.jac_inv_exa = np.zeros((self.nen**2, 2, 2, self.nelem[0]*self.nelem[1]))
        self.jac_inv_exa[:,0,0,:] = 1/elem_length_x
        self.jac_inv_exa[:,1,1,:] = 1/elem_length_y
        self.det_jac_exa = np.ones((self.nen**2, self.nelem[0]*self.nelem[1]))*(elem_length_x*elem_length_y)
        self.det_jac_inv_exa = np.ones((self.nen**2, self.nelem[0]*self.nelem[1]))/(elem_length_x*elem_length_y)
        
        self.bdy_jac_exa = np.zeros((self.nen,2,2,4,self.nelem[0]*self.nelem[1]))
        self.bdy_jac_exa[:,0,0,:,:] = elem_length_x
        self.bdy_jac_exa[:,1,1,:,:] = elem_length_y        
        self.bdy_det_jac_exa = np.ones((self.nen,4,self.nelem[0]*self.nelem[1]))*(elem_length_x*elem_length_y)
        self.bdy_jac_inv_exa = np.zeros((self.nen,2,2,4,self.nelem[0]*self.nelem[1]))
        self.bdy_jac_inv_exa[:,0,0,:,:] = 1/elem_length_x
        self.bdy_jac_inv_exa[:,1,1,:,:] = 1/elem_length_y

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
        self.jac_inv_exa = np.zeros((self.nen**3, 3, 3, self.nelem[0]*self.nelem[1]*self.nelem[2]))
        self.jac_inv_exa[:,0,0,:] = 1/elem_length_x
        self.jac_inv_exa[:,1,1,:] = 1/elem_length_y
        self.jac_inv_exa[:,2,2,:] = 1/elem_length_z
        self.det_jac_exa = np.ones((self.nen**3, self.nelem[0]*self.nelem[1]*self.nelem[2]))*(elem_length_x*elem_length_y*elem_length_z)
        self.det_jac_inv_exa = np.ones((self.nen**3, self.nelem[0]*self.nelem[1]*self.nelem[2]))/(elem_length_x*elem_length_y*elem_length_z)

        self.bdy_jac_exa = np.zeros((self.nen**2,3,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
        self.bdy_jac_exa[:,0,0,:,:] = elem_length_x
        self.bdy_jac_exa[:,1,1,:,:] = elem_length_y      
        self.bdy_jac_exa[:,2,2,:,:] = elem_length_z
        self.bdy_det_jac_exa = np.ones((self.nen**2,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))*(elem_length_x*elem_length_y*elem_length_z)
        self.bdy_jac_inv_exa = np.zeros((self.nen**2,3,3,6,self.nelem[0]*self.nelem[1]*self.nelem[2]))
        self.bdy_jac_inv_exa[:,0,0,:,:] = 1/elem_length_x
        self.bdy_jac_inv_exa[:,1,1,:,:] = 1/elem_length_y  
        self.bdy_jac_inv_exa[:,2,2,:,:] = 1/elem_length_z

    def stretch_mesh_1d(self):
        '''
        Stretches a 1d mesh to test coordinate trasformations.
    
        '''
        assert self.dim == 1 , 'Stretching only set up for 2D'
        print('... Stretching mesh by a factor of {0}'.format(self.warp_factor))

        
        def stretch_line(x):
            ''' Try to keep the warp_factor <0.26 '''
            assert self.xmin==0,'Chosen warping function is only set up for domains [0,L]'
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
        
        # switch between different mappings here
        if self.warp_type == 'default' or self.warp_type == 'papers':
            warp_fun = stretch_line
            warp_der = stretch_line_der 
        elif self.warp_type == 'quad':
            warp_fun = stretch_line_quad
            warp_der = stretch_line_quad_der 
        else:
            print('WARNING: mesh.warp_type not understood. Reverting to default.')
            warp_fun = stretch_line
            warp_der = stretch_line_der        
        
        x_elem_old = np.copy(self.x_elem)
        bdy_x_old = np.copy(self.bdy_x)
        self.x_elem = warp_fun(self.x_elem)
        self.vertices = warp_fun(self.vertices)
        self.x = self.x_elem.flatten('F')
        self.bdy_x = warp_fun(self.bdy_x)
        
        self.jac_exa[:,0,:] *= warp_der(x_elem_old) # chain rule with original transformation
        assert np.all(self.jac_exa>0),"Not a valid coordinate transformation. Try using a lower warp_factor."
        self.jac_inv_exa = 1/self.jac_exa
        self.det_jac_exa = self.jac_exa[:,0,:]
        self.det_jac_inv_exa = 1/self.det_jac_exa
        
        self.bdy_jac_exa *= warp_der(bdy_x_old)
        self.bdy_det_jac_exa = self.bdy_jac_exa[0,:,:]
        self.bdy_jac_inv_exa = 1/self.bdy_jac_exa
        
            
    def warp_mesh_2d(self):
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
            ''' function from DCP ''' 
            #assert self.warp_factor<0.24,'Try a warp_factor < 0.24 for this mesh transformation'
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            x_t = argx + self.warp_factor*np.sin(np.pi*argx)*np.sin(np.pi*argy)
            y_t = argy + self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx-0.75)*np.sin(np.pi*argy) 
            new_x = x_t*self.dom_len[0] + self.xmin[0]
            new_y = y_t*self.dom_len[1] + self.xmin[1]
            return new_x , new_y

        def warp_dcp_der(x,y):
            ''' the derivative of the function warp_dcp wrt x (i.e. dnew_x/dx) '''
            argx = (x-self.xmin[0])/self.dom_len[0]
            argy = (y-self.xmin[1])/self.dom_len[1]
            dxdx = 1 + self.warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
            dxdy = self.warp_factor*np.sin(np.pi*argx)*np.cos(np.pi*argy)*np.pi
            dydx = -self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx - 0.75)*np.sin(np.pi*argy) + self.warp_factor*np.exp(1-argx)*np.cos(np.pi*argx - 0.75)*np.sin(np.pi*argy)*np.pi
            dydy = 1 + self.warp_factor*np.exp(1-argx)*np.sin(np.pi*argx - 0.75)*np.cos(np.pi*argy)*np.pi
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
        
        # switch between different mappings here
        if self.warp_type == 'default' or self.warp_type == 'papers':
            warp_fun = warp_rectangle
            warp_der = warp_rectangle_der 
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
        else:
            print('WARNING: mesh.warp_type not understood. Reverting to default.')
            warp_fun = warp_rectangle
            warp_der = warp_rectangle_der 
            
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
        for elem in range(self.nelem[0]*self.nelem[1]):
            self.jac_inv_exa[:,:,:,elem] = np.linalg.inv(self.jac_exa[:,:,:,elem])
            self.det_jac_inv_exa[:,elem] =  np.linalg.det(self.jac_inv_exa[:,:,:,elem])
        if np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa) > 1e-12):
            print('WANRING: Numerical error in calculation of determinant inverse is {0:.2g}'.format(np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa))))
        
        dxnewdx, dxnewdy, dynewdx, dynewdy = warp_der(bdy_xy_old[:,0,:,:], bdy_xy_old[:,1,:,:])
        dxdxref = np.copy(self.bdy_jac_exa[:,0,0,:,:])
        dydyref = np.copy(self.bdy_jac_exa[:,1,1,:,:])
        self.bdy_jac_exa[:,0,0,:,:] = dxnewdx * dxdxref # chain rule, ignoring cross terms that are 0 in original transformation
        self.bdy_jac_exa[:,0,1,:,:] = dxnewdy * dydyref
        self.bdy_jac_exa[:,1,0,:,:] = dynewdx * dxdxref
        self.bdy_jac_exa[:,1,1,:,:] = dynewdy * dydyref
        for elem in range(self.nelem[0]*self.nelem[1]):
            for i in range(4):
                self.bdy_det_jac_exa[:,i,elem] = np.linalg.det(self.bdy_jac_exa[:,:,:,i,elem])
                self.bdy_jac_inv_exa[:,:,:,i,elem] = np.linalg.inv(self.bdy_jac_exa[:,:,:,i,elem])
                
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
        
            
    def warp_mesh_3d(self):
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
            return new_x , new_y, new_z
        
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
            dxdx = 1 + self.warp_factor*(0.013*np.sin(7*np.pi*argy)*(5*np.pi*np.cos(5*np.pi*argx)*np.exp(1-argx)-np.sin(5*np.pi*argx)*np.exp(1-argx))+0.017*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz)*6*np.pi*np.cos(6*np.pi*argx)-0.01*9*np.pi*np.cos(9*np.pi*argx)*np.sin(4*np.pi*argy)*np.sin(7*np.pi*argz))
            dxdy = self.warp_factor*self.dom_len[0]*(0.015*4*np.pi*np.cos(4*np.pi*argy)+0.013*np.sin(5*np.pi*argx)*7*np.pi*np.cos(7*np.pi*argy)*np.exp(1-argx)+0.017*(7*np.pi*np.cos(7*np.pi*argy)*np.sin(8*np.pi*argy)+8*np.pi*np.sin(7*np.pi*argy)*np.cos(8*np.pi*argy))*np.sin(6*np.pi*argz)*np.sin(6*np.pi*argx)-0.01*np.sin(9*np.pi*argx)*4*np.pi*np.cos(4*np.pi*argy)*np.sin(7*np.pi*argz))/self.dom_len[1]
            dxdz = self.warp_factor*self.dom_len[0]*(5*np.pi*0.016*np.cos(5*np.pi*argz)+0.017*np.sin(7*np.pi*argy)*np.sin(8*np.pi*argy)*6*np.pi*np.cos(6*np.pi*argz)*np.sin(6*np.pi*argx)-0.01*np.sin(9*np.pi*argx)*np.sin(4*np.pi*argy)*7*np.pi*np.cos(7*np.pi*argz))/self.dom_len[2]
            dydx = self.warp_factor*self.dom_len[1]*(0.016*5*np.pi*np.cos(5*np.pi*argx)-0.012*8*np.pi*np.cos(8*np.pi*argx)*np.sin(4*np.pi*argy)*np.exp(1-argy)-0.017*np.pi*(3*np.cos(3*np.pi*argx)*np.sin(9*np.pi*argx)+9*np.sin(3*np.pi*argx)*np.cos(9*np.pi*argx))*np.sin(6*np.pi*argz)*np.sin(7*np.pi*argy)-0.01*6*np.pi*np.cos(6*np.pi*argx)*np.sin(6*np.pi*argy)*np.sin(7*np.pi*argz))/self.dom_len[0]
            dydy = 1 + self.warp_factor*(-0.012*np.sin(8*np.pi*argx)*(4*np.pi*np.cos(4*np.pi*argy)*np.exp(1-argy)-np.sin(4*np.pi*argy)*np.exp(1-argy))-0.017*np.sin(3*np.pi*argx)*np.sin(9*np.pi*argx)*np.sin(6*np.pi*argz)*7*np.pi*np.cos(7*np.pi*argy)-0.01*np.sin(6*np.pi*argx)*6*np.pi*np.cos(6*np.pi*argy)*np.sin(7*np.pi*argz))
            dydz = self.warp_factor*self.dom_len[1]*(-0.015*4*np.pi*np.sin(4*np.pi*argz)-0.017*np.sin(3*np.pi*argx)*np.sin(9*np.pi*argx)*6*np.pi*np.sin(6*np.pi*argz)*np.sin(7*np.pi*argy)-0.01*np.sin(6*np.pi*argx)*np.sin(6*np.pi*argy)*7*np.pi*np.cos(7*np.pi*argz))/self.dom_len[2]
            dzdx = self.warp_factor*self.dom_len[2]*(0.018*6*np.pi*np.sin(6*np.pi*argx)+0.018*np.sin(7*np.pi*argz)*np.sin(9*np.pi*argz)*8*np.pi*np.cos(8*np.pi*argx)*np.sin(7*np.pi*argy)+0.01*5*np.pi*np.cos(5*np.pi*argx)*np.sin(8*np.pi*argy)*np.sin(6*np.pi*argz))/self.dom_len[0]
            dzdy = self.warp_factor*self.dom_len[2]*(0.02*5*np.pi*np.sin(5*np.pi*argy)+0.018*np.sin(7*np.pi*argz)*np.sin(9*np.pi*argz)*np.sin(8*np.pi*argx)*7*np.pi*np.cos(7*np.pi*argy)+0.01*np.sin(5*np.pi*argx)*8*np.pi*np.cos(8*np.pi*argy)*np.sin(6*np.pi*argz))/self.dom_len[1]
            dzdz = 1 + self.warp_factor*(0.018*np.pi*(7*np.cos(7*np.pi*argz)*np.sin(9*np.pi*argz)+9*np.sin(7*np.pi*argz)*np.cos(9*np.pi*argz))*np.sin(8*np.pi*argx)*np.sin(7*np.pi*argy)+0.01*np.sin(5*np.pi*argx)*np.sin(8*np.pi*argy)*6*np.pi*np.cos(6*np.pi*argz))
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
        else:
            print('WARNING: mesh.warp_type not understood. Reverting to default.')
            warp_fun = warp_cuboid
            warp_der = warp_cuboid_der 
        
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
        for elem in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
            self.jac_inv_exa[:,:,:,elem] = np.linalg.inv(self.jac_exa[:,:,:,elem])
            self.det_jac_inv_exa[:,elem] =  np.linalg.det(self.jac_inv_exa[:,:,:,elem])
        if np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa) > 1e-12):
            print('WANRING: Numerical error in calculation of determinant inverse is {0:.2g}'.format(np.max(abs(self.det_jac_inv_exa - 1/self.det_jac_exa))))

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
        for elem in range(self.nelem[0]*self.nelem[1]*self.nelem[2]):
            for i in range(6):
                self.bdy_det_jac_exa[:,i,elem] = np.linalg.det(self.bdy_jac_exa[:,:,:,i,elem])
                self.bdy_jac_inv_exa[:,:,:,i,elem] = np.linalg.inv(self.bdy_jac_exa[:,:,:,i,elem])
                
        ''' Uncomment the below section to debug and test the consistency of the warping '''
# =============================================================================
#         max_xy = [0,0,0,0,0,0,0,0,0]
#         max_der = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#             
#         skipx = self.nelem[1]*self.nelem[2]
#         for row in range(skipx):
#             diff = abs(self.bdy_xyz[:,0,0,row::skipx][:,0] - self.bdy_xyz[:,0,1,row::skipx][:,-1])
#             max_xy[0] = max(max_xy[0],np.max(diff))
#             diff = abs(self.bdy_xyz[:,1,0,row::skipx][:,0] - self.bdy_xyz[:,1,1,row::skipx][:,-1])
#             max_xy[1] = max(max_xy[1],np.max(diff))
#             diff = abs(self.bdy_xyz[:,2,0,row::skipx][:,0] - self.bdy_xyz[:,2,1,row::skipx][:,-1])
#             max_xy[2] = max(max_xy[2],np.max(diff))
#             diff = abs(dxnewdx[:,0,row::skipx][:,0] - dxnewdx[:,1,row::skipx][:,-1])
#             max_der[0] = max(max_der[0],np.max(diff))
#             diff = abs(dxnewdy[:,0,row::skipx][:,0] - dxnewdy[:,1,row::skipx][:,-1])
#             max_der[1] = max(max_der[1],np.max(diff))
#             diff = abs(dxnewdz[:,0,row::skipx][:,0] - dxnewdz[:,1,row::skipx][:,-1])
#             max_der[2] = max(max_der[2],np.max(diff))
#             diff = abs(dynewdx[:,0,row::skipx][:,0] - dynewdx[:,1,row::skipx][:,-1])
#             max_der[3] = max(max_der[3],np.max(diff))
#             diff = abs(dynewdy[:,0,row::skipx][:,0] - dynewdy[:,1,row::skipx][:,-1])
#             max_der[4] = max(max_der[4],np.max(diff))
#             diff = abs(dynewdz[:,0,row::skipx][:,0] - dynewdz[:,1,row::skipx][:,-1])
#             max_der[5] = max(max_der[5],np.max(diff))
#             diff = abs(dznewdx[:,0,row::skipx][:,0] - dznewdx[:,1,row::skipx][:,-1])
#             max_der[6] = max(max_der[6],np.max(diff))
#             diff = abs(dznewdy[:,0,row::skipx][:,0] - dznewdy[:,1,row::skipx][:,-1])
#             max_der[7] = max(max_der[7],np.max(diff))
#             diff = abs(dznewdz[:,0,row::skipx][:,0] - dznewdz[:,1,row::skipx][:,-1])
#             max_der[8] = max(max_der[8],np.max(diff))
#                 
#         for coly in range(self.nelem[0]*self.nelem[2]):
#             start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
#             end = start + self.nelem[1]*self.nelem[2]
#             diff = abs(self.bdy_xyz[:,0,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,0,3,start:end:self.nelem[2]][:,-1])
#             max_xy[3] = max(max_xy[3],np.max(diff))
#             diff = abs(self.bdy_xyz[:,1,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,1,3,start:end:self.nelem[2]][:,-1])
#             max_xy[4] = max(max_xy[4],np.max(diff))
#             diff = abs(self.bdy_xyz[:,2,2,start:end:self.nelem[2]][:,0] - self.bdy_xyz[:,2,3,start:end:self.nelem[2]][:,-1])
#             max_xy[5] = max(max_xy[5],np.max(diff))
#             diff = abs(dxnewdx[:,2,start:end:self.nelem[2]][:,0] - dxnewdx[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[9] = max(max_der[9],np.max(diff))
#             diff = abs(dxnewdy[:,2,start:end:self.nelem[2]][:,0] - dxnewdy[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[10] = max(max_der[10],np.max(diff))
#             diff = abs(dxnewdz[:,2,start:end:self.nelem[2]][:,0] - dxnewdz[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[11] = max(max_der[11],np.max(diff))
#             diff = abs(dynewdx[:,2,start:end:self.nelem[2]][:,0] - dynewdx[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[12] = max(max_der[12],np.max(diff))
#             diff = abs(dynewdy[:,2,start:end:self.nelem[2]][:,0] - dynewdy[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[13] = max(max_der[13],np.max(diff))
#             diff = abs(dynewdz[:,2,start:end:self.nelem[2]][:,0] - dynewdz[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[14] = max(max_der[14],np.max(diff))
#             diff = abs(dznewdx[:,2,start:end:self.nelem[2]][:,0] - dznewdx[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[15] = max(max_der[15],np.max(diff))
#             diff = abs(dznewdy[:,2,start:end:self.nelem[2]][:,0] - dznewdy[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[16] = max(max_der[16],np.max(diff))
#             diff = abs(dznewdz[:,2,start:end:self.nelem[2]][:,0] - dznewdz[:,3,start:end:self.nelem[2]][:,-1])
#             max_der[17] = max(max_der[17],np.max(diff))
# 
#         
#         for colz in range(self.nelem[0]*self.nelem[2]):
#             start = colz*self.nelem[2]
#             end = start + self.nelem[2]
#             diff = abs(self.bdy_xyz[:,0,4,start:end][:,0] - self.bdy_xyz[:,0,5,start:end][:,-1])
#             max_xy[6] = max(max_xy[6],np.max(diff))
#             diff = abs(self.bdy_xyz[:,1,4,start:end][:,0] - self.bdy_xyz[:,1,5,start:end][:,-1])
#             max_xy[7] = max(max_xy[7],np.max(diff))
#             diff = abs(self.bdy_xyz[:,2,4,start:end][:,0] - self.bdy_xyz[:,2,5,start:end][:,-1])
#             max_xy[8] = max(max_xy[8],np.max(diff))
#             diff = abs(dxnewdx[:,4,start:end][:,0] - dxnewdx[:,5,start:end][:,-1])
#             max_der[18] = max(max_der[18],np.max(diff))
#             diff = abs(dxnewdy[:,4,start:end][:,0] - dxnewdy[:,5,start:end][:,-1])
#             max_der[19] = max(max_der[19],np.max(diff))
#             diff = abs(dxnewdz[:,4,start:end][:,0] - dxnewdz[:,5,start:end][:,-1])
#             max_der[20] = max(max_der[20],np.max(diff))
#             diff = abs(dynewdx[:,4,start:end][:,0] - dynewdx[:,5,start:end][:,-1])
#             max_der[21] = max(max_der[21],np.max(diff))
#             diff = abs(dynewdy[:,4,start:end][:,0] - dynewdy[:,5,start:end][:,-1])
#             max_der[22] = max(max_der[22],np.max(diff))
#             diff = abs(dynewdz[:,4,start:end][:,0] - dynewdz[:,5,start:end][:,-1])
#             max_der[23] = max(max_der[23],np.max(diff))
#             diff = abs(dznewdx[:,4,start:end][:,0] - dznewdx[:,5,start:end][:,-1])
#             max_der[24] = max(max_der[24],np.max(diff))
#             diff = abs(dznewdy[:,4,start:end][:,0] - dznewdy[:,5,start:end][:,-1])
#             max_der[25] = max(max_der[25],np.max(diff))
#             diff = abs(dznewdz[:,4,start:end][:,0] - dznewdz[:,5,start:end][:,-1])
#             max_der[26] = max(max_der[26],np.max(diff))
# 
#         #print('Max diff x values along x_ref direction boundary (=domain[0]): ', max_xy[0])
#         print('Max diff y values along x_ref direction boundary: ', max_xy[1])
#         print('Max diff z values along x_ref direction boundary: ', max_xy[2])
#         print('Max diff x values along y_ref direction boundary: ', max_xy[3])
#         #print('Max diff y values along y_ref direction boundary: (=domain[1])', max_xy[4])
#         print('Max diff z values along y_ref direction boundary: ', max_xy[5])
#         print('Max diff x values along z_ref direction boundary: ', max_xy[6])
#         print('Max diff y values along z_ref direction boundary: ', max_xy[7])
#         #print('Max diff z values along z_ref direction boundary: (=domain[2])', max_xy[8])
#         #print('Max diff of dxnew(1)/dxref(1) along x bdy (not nec. = 0): ', max_der[0])
#         print('Max diff of dxnew(1)/dxref(2) along x bdy: ', max_der[1])
#         print('Max diff of dxnew(1)/dxref(3) along x bdy: ', max_der[2])
#         #print('Max diff of dxnew(2)/dxref(1) along x bdy (not nec. = 0): ', max_der[3])
#         print('Max diff of dxnew(2)/dxref(2) along x bdy: ', max_der[4])
#         print('Max diff of dxnew(2)/dxref(3) along x bdy: ', max_der[5])
#         #print('Max diff of dxnew(3)/dxref(1) along x bdy (not nec. = 0): ', max_der[6])
#         print('Max diff of dxnew(3)/dxref(2) along x bdy: ', max_der[7])
#         print('Max diff of dxnew(3)/dxref(3) along x bdy: ', max_der[8])
#         print('Max diff of dxnew(1)/dxref(1) along y bdy: ', max_der[9])
#         #print('Max diff of dxnew(1)/dxref(2) along y bdy (not nec. = 0): ', max_der[10])
#         print('Max diff of dxnew(1)/dxref(3) along y bdy: ', max_der[11])
#         print('Max diff of dxnew(2)/dxref(1) along y bdy: ', max_der[12])
#         #print('Max diff of dxnew(2)/dxref(2) along y bdy (not nec. = 0): ', max_der[13])
#         print('Max diff of dxnew(2)/dxref(3) along y bdy: ', max_der[14])
#         print('Max diff of dxnew(3)/dxref(1) along y bdy: ', max_der[15])
#         #print('Max diff of dxnew(3)/dxref(2) along y bdy (not nec. = 0): ', max_der[16])
#         print('Max diff of dxnew(3)/dxref(3) along y bdy: ', max_der[17])
#         print('Max diff of dxnew(1)/dxref(1) along z bdy: ', max_der[18])
#         print('Max diff of dxnew(1)/dxref(2) along z bdy: ', max_der[19])
#         #print('Max diff of dxnew(1)/dxref(3) along z bdy (not nec. = 0): ', max_der[20])
#         print('Max diff of dxnew(2)/dxref(1) along z bdy: ', max_der[21])
#         print('Max diff of dxnew(2)/dxref(2) along z bdy: ', max_der[22])
#         #print('Max diff of dxnew(2)/dxref(3) along z bdy (not nec. = 0): ', max_der[23])
#         print('Max diff of dxnew(3)/dxref(1) along z bdy: ', max_der[24])
#         print('Max diff of dxnew(3)/dxref(2) along z bdy: ', max_der[25])
#         #print('Max diff of dxnew(3)/dxref(3) along z bdy (not nec. = 0): ', max_der[26])
# =============================================================================

    def plot(self,plt_save_name=None,markersize=4,fontsize=12,dpi=1000,label=True,
             label_all_lines=True, nodes=True, bdy_nodes=False):
        if self.dim == 1:
            fig = plt.figure(figsize=(6,1))
            ax = plt.axes(frameon=False) # turn off the frame
            
            ax.hlines(0.35,self.xmin,self.xmax,color='black',lw=1)  # Draw a horizontal line at y=1
            ax.set_xlim(self.xmin-self.dom_len/100,self.xmax+self.dom_len/100)
            ax.set_ylim(0,1)
            
            if nodes:
                ax.plot(self.x,0.35*np.ones(self.nn),'o',color='blue',ms=markersize)
                
            if bdy_nodes:
                ax.plot(self.bdy_x,0.35*np.ones(self.bdy_x.shape),'s',color='red',ms=markersize)
            
            ax.axes.get_yaxis().set_visible(False) # turn off y axis 
            ax.plot(self.vertices,0.35*np.ones(self.nelem+1),'|',ms=20,color='black',lw=1)  # Plot a line at each location specified in a
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
            
            if nodes:
                ax.scatter(self.xy[:,0],self.xy[:,1],marker='o',c='blue',s=markersize)
                
            if bdy_nodes:
                ax.scatter(self.bdy_xy[:,0,:,:],self.bdy_xy[:,1,:,:],marker='s',color='red',s=markersize)
    
            for line in self.grid_lines:
                ax.plot(line[0],line[1],color='black',lw=1)
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
            fig.savefig(plt_save_name+'.png', format='png',dpi=dpi)
            

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
        print('... Computing Grid Metrics')
        
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
                eye = np.eye(self.nen)
                txbT = np.kron(sbp.tR.reshape((self.nen,1)), eye).T
                txaT = np.kron(sbp.tL.reshape((self.nen,1)), eye).T
                tybT = np.kron(eye, sbp.tR.reshape((self.nen,1))).T
                tyaT = np.kron(eye, sbp.tL.reshape((self.nen,1))).T
                
                for row in range(self.nelem[1]): # starts at bottom left to bottom right, then next row up
                    for i in range(4): # loop over matrix entries
                        Lmetrics = txbT @ self.metrics[:,i,row::self.nelem[1]]
                        Rmetrics = txaT @ self.metrics[:,i,row::self.nelem[1]]
                        avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                        self.bdy_metrics[:,0,i,row::self.nelem[1]][:,1:] = avgmetrics
                        self.bdy_metrics[:,1,i,row::self.nelem[1]][:,:-1] = avgmetrics
                        if periodic[0]:   
                            avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                            self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] = avgmetrics
                            self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1] = avgmetrics     
                        else:
                            self.bdy_metrics[:,0,i,row::self.nelem[1]][:,0] = Rmetrics[:,0]
                            self.bdy_metrics[:,1,i,row::self.nelem[1]][:,-1] = Lmetrics[:,-1]
                    
                for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                    start = col*self.nelem[0]
                    end = start + self.nelem[1]
                    for i in range(4): # loop over matrix entries
                        Lmetrics = tybT @ self.metrics[:,i,start:end]
                        Rmetrics = tyaT @ self.metrics[:,i,start:end]
                        avgmetrics = (Lmetrics[:,:-1] + Rmetrics[:,1:])/2
                        self.bdy_metrics[:,2,i,start:end][:,1:] = avgmetrics
                        self.bdy_metrics[:,3,i,start:end][:,:-1] = avgmetrics
                        if periodic[1]:
                            avgmetrics = (Lmetrics[:,-1] + Rmetrics[:,0])/2
                            self.bdy_metrics[:,2,i,start:end][:,0] = avgmetrics
                            self.bdy_metrics[:,3,i,start:end][:,-1] = avgmetrics     
                        else:
                            self.bdy_metrics[:,2,i,start:end][:,0] = Rmetrics[:,0]
                            self.bdy_metrics[:,3,i,start:end][:,-1] = Lmetrics[:,-1]
                          
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
                
                assert (bdy_metric_method != 'calculate'),'Must use extrapolated or exact boundary metrics for optimization.'
                # overwrite metrics with optimized ones         
                eye = np.eye(self.nen)
                txb = np.kron(sbp.tR.reshape((self.nen,1)), eye)
                txa = np.kron(sbp.tL.reshape((self.nen,1)), eye)
                tyb = np.kron(eye, sbp.tR.reshape((self.nen,1)))
                tya = np.kron(eye, sbp.tL.reshape((self.nen,1)))
                if optz_method == 'essbp' or optz_method == 'default' or optz_method == 'alex' or optz_method == 'generalized':
                    # First optimize surface metrics, then do default optimization
                    A = np.zeros((self.nelem[0]*self.nelem[1],self.nelem[0]*self.nelem[1]))
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
                            if (periodic[0] and periodic[1]):
                                lam = np.linalg.lstsq(A,RHS,rcond=-1)[0]
                            else:
                                lam = np.linalg.solve(A,RHS)
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
            
                self.metrics_exa[:,0,:] = self.det_jac_exa * self.jac_inv_exa[:,0,0,:]
                self.metrics_exa[:,1,:] = self.det_jac_exa * self.jac_inv_exa[:,0,1,:]
                self.metrics_exa[:,2,:] = self.det_jac_exa * self.jac_inv_exa[:,0,2,:]
                self.metrics_exa[:,3,:] = self.det_jac_exa * self.jac_inv_exa[:,1,0,:]
                self.metrics_exa[:,4,:] = self.det_jac_exa * self.jac_inv_exa[:,1,1,:]
                self.metrics_exa[:,5,:] = self.det_jac_exa * self.jac_inv_exa[:,1,2,:]
                self.metrics_exa[:,6,:] = self.det_jac_exa * self.jac_inv_exa[:,2,0,:]
                self.metrics_exa[:,7,:] = self.det_jac_exa * self.jac_inv_exa[:,2,1,:]
                self.metrics_exa[:,8,:] = self.det_jac_exa * self.jac_inv_exa[:,2,2,:]
                
                for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                    self.bdy_metrics_exa[:,f,0,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,0,f,:]
                    self.bdy_metrics_exa[:,f,1,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,1,f,:]
                    self.bdy_metrics_exa[:,f,2,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,0,2,f,:]
                    self.bdy_metrics_exa[:,f,3,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,0,f,:]
                    self.bdy_metrics_exa[:,f,4,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,1,f,:]
                    self.bdy_metrics_exa[:,f,5,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,1,2,f,:]
                    self.bdy_metrics_exa[:,f,6,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,0,f,:]
                    self.bdy_metrics_exa[:,f,7,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,1,f,:]
                    self.bdy_metrics_exa[:,f,8,:] = self.bdy_det_jac_exa[:,f,:] * self.bdy_jac_inv_exa[:,2,2,f,:]
                
            if metric_method.lower()=='exact':
                self.metrics = np.copy(self.metrics_exa)
                
                if jac_method.lower()=='direct' or jac_method.lower()=='calculate':
                    eye = np.eye(self.nen)
                    Dx = np.kron(np.kron(sbp.D, eye), eye)
                    Dy = np.kron(np.kron(eye, sbp.D), eye)
                    Dz = np.kron(np.kron(eye, eye), sbp.D)   
                    dxp_dxr = Dx @ self.xyz_elem[:,0,:]
                    dxp_dyr = Dy @ self.xyz_elem[:,0,:]
                    dxp_dzr = Dz @ self.xyz_elem[:,0,:]
                    dyp_dxr = Dx @ self.xyz_elem[:,1,:]
                    dyp_dyr = Dy @ self.xyz_elem[:,1,:]
                    dyp_dzr = Dz @ self.xyz_elem[:,1,:]
                    dzp_dxr = Dx @ self.xyz_elem[:,2,:]
                    dzp_dyr = Dy @ self.xyz_elem[:,2,:]
                    dzp_dzr = Dz @ self.xyz_elem[:,2,:]
                    # unique (matrix has unique determinant)                            
                    self.det_jac = dxp_dxr*(dyp_dyr*dzp_dzr - dyp_dzr*dzp_dyr) \
                                 - dyp_dxr*(dxp_dyr*dzp_dzr - dxp_dzr*dzp_dyr) \
                                 + dzp_dxr*(dxp_dyr*dyp_dzr - dxp_dzr*dyp_dyr)
                
            else:
                self.metrics = np.zeros((self.nen**3,9,self.nelem[0]*self.nelem[1]*self.nelem[2])) 
                if not (metric_method.lower()=='thomaslombard' or metric_method.lower()=='vinokuryee'):
                    print("WARNING: Did not understand metric_method. For 3D, try 'exact', 'VinokurYee', or 'ThomasLombard'.")
                    print("         Defaulting to 'VinokurYee'.")
                    metric_method = 'VinokurYee'
                eye = np.eye(self.nen)
                Dx = np.kron(np.kron(sbp.D, eye), eye)
                Dy = np.kron(np.kron(eye, sbp.D), eye)
                Dz = np.kron(np.kron(eye, eye), sbp.D)   
                dxp_dxr = Dx @ self.xyz_elem[:,0,:]
                dxp_dyr = Dy @ self.xyz_elem[:,0,:]
                dxp_dzr = Dz @ self.xyz_elem[:,0,:]
                dyp_dxr = Dx @ self.xyz_elem[:,1,:]
                dyp_dyr = Dy @ self.xyz_elem[:,1,:]
                dyp_dzr = Dz @ self.xyz_elem[:,1,:]
                dzp_dxr = Dx @ self.xyz_elem[:,2,:]
                dzp_dyr = Dy @ self.xyz_elem[:,2,:]
                dzp_dzr = Dz @ self.xyz_elem[:,2,:]
                dXp_dxr = fn.lm_gdiag(Dx, self.xyz_elem[:,0,:])
                dXp_dyr = fn.lm_gdiag(Dy, self.xyz_elem[:,0,:])
                dXp_dzr = fn.lm_gdiag(Dz, self.xyz_elem[:,0,:])
                dYp_dxr = fn.lm_gdiag(Dx, self.xyz_elem[:,1,:])
                dYp_dyr = fn.lm_gdiag(Dy, self.xyz_elem[:,1,:])
                dYp_dzr = fn.lm_gdiag(Dz, self.xyz_elem[:,1,:])
                dZp_dxr = fn.lm_gdiag(Dx, self.xyz_elem[:,2,:])
                dZp_dyr = fn.lm_gdiag(Dy, self.xyz_elem[:,2,:])
                dZp_dzr = fn.lm_gdiag(Dz, self.xyz_elem[:,2,:])
                
                if jac_method.lower()=='direct' or jac_method.lower()=='calculate':
                    # unique (matrix has unique determinant)                            
                    self.det_jac = dxp_dxr*(dyp_dyr*dzp_dzr - dyp_dzr*dzp_dyr) \
                                 - dyp_dxr*(dxp_dyr*dzp_dzr - dxp_dzr*dzp_dyr) \
                                 + dzp_dxr*(dxp_dyr*dyp_dzr - dxp_dzr*dyp_dyr)
                
                if metric_method.lower() == 'thomaslombard':
                    self.metrics[:,0,:] = fn.gm_gv(dZp_dzr,dyp_dyr) - fn.gm_gv(dZp_dyr,dyp_dzr)
                    self.metrics[:,1,:] = fn.gm_gv(dXp_dzr,dzp_dyr) - fn.gm_gv(dXp_dyr,dzp_dzr)
                    self.metrics[:,2,:] = fn.gm_gv(dYp_dzr,dxp_dyr) - fn.gm_gv(dYp_dyr,dxp_dzr)
                    self.metrics[:,3,:] = fn.gm_gv(dZp_dxr,dyp_dzr) - fn.gm_gv(dZp_dzr,dyp_dxr)
                    self.metrics[:,4,:] = fn.gm_gv(dXp_dxr,dzp_dzr) - fn.gm_gv(dXp_dzr,dzp_dxr)
                    self.metrics[:,5,:] = fn.gm_gv(dYp_dxr,dxp_dzr) - fn.gm_gv(dYp_dzr,dxp_dxr)
                    self.metrics[:,6,:] = fn.gm_gv(dZp_dyr,dyp_dxr) - fn.gm_gv(dZp_dxr,dyp_dyr)
                    self.metrics[:,7,:] = fn.gm_gv(dXp_dyr,dzp_dxr) - fn.gm_gv(dXp_dxr,dzp_dyr)
                    self.metrics[:,8,:] = fn.gm_gv(dYp_dyr,dxp_dxr) - fn.gm_gv(dYp_dxr,dxp_dyr) 
                                   
                elif metric_method.lower() == 'vinokuryee':                   
                    self.metrics[:,0,:] = 0.5*(fn.gm_gv(dYp_dyr,dzp_dzr) - fn.gm_gv(dZp_dyr,dyp_dzr) + fn.gm_gv(dZp_dzr,dyp_dyr) - fn.gm_gv(dYp_dzr,dzp_dyr))
                    self.metrics[:,1,:] = 0.5*(fn.gm_gv(dZp_dyr,dxp_dzr) - fn.gm_gv(dXp_dyr,dzp_dzr) + fn.gm_gv(dXp_dzr,dzp_dyr) - fn.gm_gv(dZp_dzr,dxp_dyr))
                    self.metrics[:,2,:] = 0.5*(fn.gm_gv(dXp_dyr,dyp_dzr) - fn.gm_gv(dYp_dyr,dxp_dzr) + fn.gm_gv(dYp_dzr,dxp_dyr) - fn.gm_gv(dXp_dzr,dyp_dyr))
                    self.metrics[:,3,:] = 0.5*(fn.gm_gv(dYp_dzr,dzp_dxr) - fn.gm_gv(dZp_dzr,dyp_dxr) + fn.gm_gv(dZp_dxr,dyp_dzr) - fn.gm_gv(dYp_dxr,dzp_dzr))
                    self.metrics[:,4,:] = 0.5*(fn.gm_gv(dZp_dzr,dxp_dxr) - fn.gm_gv(dXp_dzr,dzp_dxr) + fn.gm_gv(dXp_dxr,dzp_dzr) - fn.gm_gv(dZp_dxr,dxp_dzr))
                    self.metrics[:,5,:] = 0.5*(fn.gm_gv(dXp_dzr,dyp_dxr) - fn.gm_gv(dYp_dzr,dxp_dxr) + fn.gm_gv(dYp_dxr,dxp_dzr) - fn.gm_gv(dXp_dxr,dyp_dzr))
                    self.metrics[:,6,:] = 0.5*(fn.gm_gv(dYp_dxr,dzp_dyr) - fn.gm_gv(dZp_dxr,dyp_dyr) + fn.gm_gv(dZp_dyr,dyp_dxr) - fn.gm_gv(dYp_dyr,dzp_dxr))
                    self.metrics[:,7,:] = 0.5*(fn.gm_gv(dZp_dxr,dxp_dyr) - fn.gm_gv(dXp_dxr,dzp_dyr) + fn.gm_gv(dXp_dyr,dzp_dxr) - fn.gm_gv(dZp_dyr,dxp_dxr))
                    self.metrics[:,8,:] = 0.5*(fn.gm_gv(dXp_dxr,dyp_dyr) - fn.gm_gv(dYp_dxr,dxp_dyr) + fn.gm_gv(dYp_dyr,dxp_dxr) - fn.gm_gv(dXp_dyr,dyp_dxr)) 
                    
                        
                    
            if bdy_metric_method.lower()=='exact':
                self.bdy_metrics = np.copy(self.bdy_metrics_exa)
                
            elif bdy_metric_method.lower()=='interpolate' or bdy_metric_method.lower()=='extrapolate' or bdy_metric_method.lower()=='project':
                self.bdy_metrics = np.zeros((self.nen**2,6,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                eye = np.eye(self.nen)
                D1 = np.kron(sbp.D,eye)
                D2 = np.kron(eye,sbp.D)
                txbT = np.kron(np.kron(sbp.tR.reshape((self.nen,1)), eye), eye).T
                txaT = np.kron(np.kron(sbp.tL.reshape((self.nen,1)), eye), eye).T
                tybT = np.kron(np.kron(eye, sbp.tR.reshape((self.nen,1))), eye).T
                tyaT = np.kron(np.kron(eye, sbp.tL.reshape((self.nen,1))), eye).T
                tzbT = np.kron(np.kron(eye, eye), sbp.tR.reshape((self.nen,1))).T
                tzaT = np.kron(np.kron(eye, eye), sbp.tL.reshape((self.nen,1))).T
                skipx = self.nelem[1]*self.nelem[2]
                skipz = self.nelem[0]*self.nelem[1]
                maxdiff = 0
                
                for rowx in range(skipx):
                    for i in range(9): # loop over matrix entries
                        Lmetrics = txbT @ self.metrics[:,i,rowx::skipx]
                        Rmetrics = txaT @ self.metrics[:,i,rowx::skipx]
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
                    for i in range(9): # loop over matrix entries
                        Lmetrics = tybT @ self.metrics[:,i,start:end:self.nelem[2]]
                        Rmetrics = tyaT @ self.metrics[:,i,start:end:self.nelem[2]]
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
                    for i in range(9): # loop over matrix entries
                        Lmetrics = tzbT @ self.metrics[:,i,start:end]
                        Rmetrics = tzaT @ self.metrics[:,i,start:end]
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
                
                print('TEMP: The boundary metric extrapolations modified by a max of {0:.2g} in averaging.'.format(maxdiff))
                            
            else: 
                if not (bdy_metric_method.lower()=='thomaslombard' or bdy_metric_method.lower()=='vinokurYee'):
                    print("WARNING: Did not understand bdy_metric_method. For 3D, try 'exact', 'VinokurYee', 'ThomasLombard', or 'interpolate'.")
                    print("         Defaulting to 'VinokurYee'.")
                self.bdy_metrics = np.zeros((self.nen**2,6,9,self.nelem[0]*self.nelem[1]*self.nelem[2]))
                eye = np.eye(self.nen)
                D1 = np.kron(sbp.D,eye)
                D2 = np.kron(eye,sbp.D)
                if bdy_metric_method.lower() == 'thomaslombard':
                    for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                        if (f == 0) or (f == 1):
                            self.bdy_metrics[:,f,0,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,1,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,2,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,3:,:] = None
                        elif (f == 2) or (f == 3):
                            self.bdy_metrics[:,f,:3,:] = None
                            self.bdy_metrics[:,f,3,:] = fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,4,:] = fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,5,:] = fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:]))
                            self.bdy_metrics[:,f,6:,:] = None
                        elif (f == 4) or (f == 5):
                            self.bdy_metrics[:,f,:6,:] = None
                            self.bdy_metrics[:,f,6,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:]))
                            self.bdy_metrics[:,f,7,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:]))
                            self.bdy_metrics[:,f,8,:] = fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:]))
                elif bdy_metric_method.lower() == 'vinokuryee':
                    for f in range(6): # loop over facets (xleft, xright, yleft, yright, zxleft, zright)
                        if (f == 0) or (f == 1):
                            self.bdy_metrics[:,f,0,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,1,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,2,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])))
                            self.bdy_metrics[:,f,3:,:] = None
                        elif (f == 2) or (f == 3):
                            self.bdy_metrics[:,f,:3,:] = None
                            self.bdy_metrics[:,f,3,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,4,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,5,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])))
                            self.bdy_metrics[:,f,6:,:] = None
                        elif (f == 4) or (f == 5):
                            self.bdy_metrics[:,f,:6,:] = None
                            self.bdy_metrics[:,f,6,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])))
                            self.bdy_metrics[:,f,7,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,2,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,2,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,2,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,2,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])))
                            self.bdy_metrics[:,f,8,:] = 0.5*(fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,0,f,:]),(D2 @ self.bdy_xyz[:,1,f,:])) - fn.gm_gv(fn.lm_gdiag(D1, self.bdy_xyz[:,1,f,:]),(D2 @ self.bdy_xyz[:,0,f,:])) \
                                                            +fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,1,f,:]),(D1 @ self.bdy_xyz[:,0,f,:])) - fn.gm_gv(fn.lm_gdiag(D2, self.bdy_xyz[:,0,f,:]),(D1 @ self.bdy_xyz[:,1,f,:])))
           
            
            if use_optz_metrics:
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
                    
                    A = np.zeros((self.nelem[0]*self.nelem[1]*self.nelem[2],self.nelem[0]*self.nelem[1]*self.nelem[2]))
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
                                lam = np.linalg.lstsq(A,RHS,rcond=-1)[0]
                            else:
                                lam = np.linalg.solve(A,RHS)
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
                    Minv = np.linalg.pinv(M, rcond=1e-13)
                    if np.max(abs(Minv)) > 1e8:
                        print('WARNING: There may be an error in Minv of metric optimization. Try a higher rcond.')
                    # first for x dimension
                    c = txb @ Hperp @ self.bdy_metrics[:,1,0,:] - txa @ Hperp @ self.bdy_metrics[:,0,0,:] \
                      + tyb @ Hperp @ self.bdy_metrics[:,3,3,:] - tya @ Hperp @ self.bdy_metrics[:,2,3,:] \
                      + tzb @ Hperp @ self.bdy_metrics[:,5,6,:] - tza @ Hperp @ self.bdy_metrics[:,4,6,:]
                    if np.any(abs(np.sum(c,axis=0))>1e-12):
                        print('WARNING: c_x vector in optimized metric computation does not add to zero.')
                        print('         max value (element) of sum is {0:.2g}'.format(np.max(abs(np.sum(c,axis=0)))))
                        print('         Surface integrals in x do not hold discretely.')
                    aex = np.vstack((self.metrics[:,0,:],self.metrics[:,3,:],self.metrics[:,6,:]))
                    a = aex - Minv @ ( M @ aex - c )
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
                    a = aex - Minv @ ( M @ aex - c )
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
                    a = aex - Minv @ ( M @ aex - c )
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
                    maxbdy[i] = max(maxint[i],np.max(diff))
                
            for col in range(self.nelem[0]): # starts at bottom left to top left, then next column to right
                start = col*self.nelem[0]
                end = start + self.nelem[1]
                for i in range(2,4): # loop over matrix entries
                    diff = abs(self.bdy_metrics[:,2,i,start:end][:,1:] - self.bdy_metrics[:,3,i,start:end][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,2,i,start:end][:,0] - self.bdy_metrics[:,3,i,start:end][:,-1]) 
                    maxbdy[i] = max(maxint[i],np.max(diff))

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
                    maxbdy[i] = max(maxint[i],np.max(diff))
                    
            for coly in range(self.nelem[0]*self.nelem[2]):
                start = coly + (coly//self.nelem[2])*(self.nelem[1]-1)*self.nelem[2]
                end = start + self.nelem[1]*self.nelem[2]
                for i in range(3,6):
                    diff = abs(self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,1:] - self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,2,i,start:end:self.nelem[2]][:,0] - self.bdy_metrics[:,3,i,start:end:self.nelem[2]][:,-1])
                    maxbdy[i] = max(maxint[i],np.max(diff))
            
            for colz in range(self.nelem[0]*self.nelem[2]):
                start = colz*self.nelem[2]
                end = start + self.nelem[2]
                for i in range(6,9):
                    diff = abs(self.bdy_metrics[:,4,i,start:end][:,1:] - self.bdy_metrics[:,5,i,start:end][:,:-1])
                    maxint[i] = max(maxint[i],np.max(diff))
                    diff = abs(self.bdy_metrics[:,4,i,start:end][:,0] - self.bdy_metrics[:,5,i,start:end][:,-1])
                    maxbdy[i] = max(maxint[i],np.max(diff))

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
            
        


            