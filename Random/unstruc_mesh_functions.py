import numpy as np
import gmsh
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from scipy.special import legendre
from get_hicken_op import getOps3D, getOps2D
from julia import Main


##############################################################
# grid warping functions
##############################################################

def warp_default_2d(x,y,warp_factor,order,xmin=np.array([0.,0.]),xmax=np.array([1.,1.])):
    ''' Based on function from Tristan's Paper. Warps a rectangular mesh
            Try to keep the warp.factor <0.24 '''
    assert warp_factor<0.24,'Try a warp_factor < 0.24 for this mesh transformation'
    dom_len = xmax - xmin
    argx = (x-xmin[0])/dom_len[0]
    argy = (y-xmin[1])/dom_len[1]

    if order==0:
        new_x = x + warp_factor*dom_len[0]*np.sin(np.pi*argx)*np.sin(np.pi*argy)
        new_y = y + warp_factor*dom_len[1]*np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy) 
        return new_x, new_y
    elif order==1:
        dxdx = 1 + warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
        dxdy = warp_factor*dom_len[0]*np.pi*np.sin(np.pi*argx)*np.cos(np.pi*argy)/dom_len[1]
        dydx = warp_factor*dom_len[1]*np.pi*np.exp(1-argy)*np.cos(np.pi*argx)*np.sin(np.pi*argy)/dom_len[0]
        dydy = 1 + warp_factor*(np.pi*np.exp(1-argy)*np.sin(np.pi*argx)*np.cos(np.pi*argy) - np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy))
        return dxdx, dxdy, dydx, dydy
    else:
        raise Exception('order > 1 invalid') 
    
def warp_smooth_2d(x,y,warp_factor,order,xmin=np.array([0.,0.]),xmax=np.array([1.,1.])):
    ''' function from dcp '''
    assert warp_factor<0.15,'Try a warp_factor < 0.15 for this mesh transformation'
    dom_len = xmax - xmin
    argx = (x-xmin[0])/dom_len[0]
    argy = (y-xmin[1])/dom_len[1]

    if order==0:
        new_x = x + warp_factor*dom_len[0]*np.sin(2*np.pi*argx)*np.sin(2*np.pi*argy)
        new_y = y + warp_factor*dom_len[1]*np.sin(2*np.pi*argx)*np.sin(2*np.pi*argy) 
        return new_x, new_y
    elif order==1:
        dxdx = 1 + warp_factor*2*np.pi*np.cos(2*np.pi*argx)*np.sin(2*np.pi*argy)
        dxdy = warp_factor*dom_len[0]*2*np.pi*np.sin(2*np.pi*argx)*np.cos(2*np.pi*argy)/dom_len[1]
        dydx = warp_factor*dom_len[1]*2*np.pi*np.cos(2*np.pi*argx)*np.sin(2*np.pi*argy)/dom_len[0]
        dydy = 1 + warp_factor*2*np.pi*np.sin(2*np.pi*argx)*np.cos(2*np.pi*argy)
        return dxdx, dxdy, dydx, dydy
    else:
        raise Exception('order > 1 invalid') 
        
def warp_chan_2d(x,y,warp_factor,order,xmin=np.array([0.,0.]),xmax=np.array([1.,1.])):
    ''' 2d version of the transformation in chan and wilcox '''
    ''' they used warp = 0.125 '''
    dom_len = xmax - xmin
    argx = 2.*(x-xmin[0])/dom_len[0] - 1.
    argy = 2.*(y-xmin[1])/dom_len[1] - 1.

    if order == 0:
        new_x = x + warp_factor*dom_len[0]*np.cos(np.pi*argx/2)*np.cos(np.pi*argy/2)
        new_y = y + warp_factor*dom_len[1]*np.cos(np.pi*argx/2)*np.cos(np.pi*argy/2)
        return new_x,new_y
    elif order == 1:
        dxdx = 1.0 - np.pi*warp_factor*np.sin(np.pi*argx/2)*np.cos(np.pi*argy/2)
        dxdy = - np.pi*warp_factor*dom_len[0]*np.cos(np.pi*argx/2)*np.sin(np.pi*argy/2)/dom_len[1]
        dydx = - np.pi*warp_factor*dom_len[1]*np.sin(np.pi*argx/2)*np.cos(np.pi*argy/2)/dom_len[0]
        dydy = 1.0 - np.pi*warp_factor*np.cos(np.pi*argx/2)*np.sin(np.pi*argy/2)
        return dxdx, dxdy, dydx, dydy
    else:
        raise Exception('order > 1 invalid') 
        
def warp_cubic_2d(x,y,warp,order,xmin=np.array([0.,0.]),xmax=np.array([1.,1.])):
    ''' 2d version of the transformation in chan and wilcox '''
    ''' they used warp = 0.125 '''
    dom_len = xmax - xmin
    xscale = (x-xmin[0])/dom_len[0]
    yscale = (y-xmin[1])/dom_len[1]

    if order == 0:
        new_x = warp*(xmax[0]-xmin[0])*(yscale**3-1.7*yscale**2+0.7*yscale)*(xscale**3-xscale) + x
        new_y = 1.5*warp*(xmax[1]-xmin[1])*(xscale**3-1.2*xscale**2+0.2*xscale)*(yscale**3-yscale**2) + y
        return new_x, new_y
    elif order == 1:
        dxdx1 = warp*(yscale**3-1.7*yscale**2+0.7*yscale)*(3*xscale**2-1) + 1
        dxdx2 = warp*(xmax[0]-xmin[0])*(3*yscale**2-2*1.7*yscale+0.7)*(xscale**3-xscale)/(xmax[1]-xmin[1])
        dydx1 = 1.5*warp*(xmax[1]-xmin[1])*(3*xscale**2-2*1.2*xscale+0.2)*(yscale**3-yscale**2)/(xmax[0]-xmin[0])
        dydx2 = 1.5*warp*(xscale**3-1.2*xscale**2+0.2*xscale)*(3*yscale**2-2*yscale) + 1
        return dxdx1, dxdx2, dydx1, dydx2
    else:
        raise Exception('order > 1 invalid') 
    
def warp_default_3d(x,y,z,warp_factor,order,xmin=np.array([0.,0.,0.]),xmax=np.array([1.,1.,1.])):
    ''' Based on function from 2019 DDRF Paper. Warps a cuboid mesh
    Try to keep the warp.factor <0.24 ''' 
    dom_len = xmax - xmin
    argx = (x-xmin[0])/dom_len[0]
    argy = (y-xmin[1])/dom_len[1] 
    #argz = (z-xmin[1])/dom_len[2] 

    new_x = x + warp_factor*dom_len[0]*np.sin(np.pi*argx)*np.sin(np.pi*argy)
    new_y = y + warp_factor*dom_len[1]*np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy) 
    if order == 0:   
        new_z = z + 0.25*warp_factor*dom_len[2]*(np.sin(2*np.pi*(new_x-xmin[0])/dom_len[0])+np.sin(2*np.pi*(new_y-xmin[1])/dom_len[1])) 
        return new_x , new_y, new_z
    elif order == 1:
        dxdx = 1 + warp_factor*np.pi*np.cos(np.pi*argx)*np.sin(np.pi*argy)
        dxdy = warp_factor*dom_len[0]*np.pi*np.sin(np.pi*argx)*np.cos(np.pi*argy)/dom_len[1]
        dxdz = np.zeros(np.shape(dxdx))
        dydx = warp_factor*dom_len[1]*np.pi*np.exp(1-argy)*np.cos(np.pi*argx)*np.sin(np.pi*argy)/dom_len[0]
        dydy = 1 + warp_factor*(np.pi*np.exp(1-argy)*np.sin(np.pi*argx)*np.cos(np.pi*argy) - np.exp(1-argy)*np.sin(np.pi*argx)*np.sin(np.pi*argy))
        dydz = np.zeros(np.shape(dxdx))
        dzdx = 0.25*warp_factor*dom_len[2]*2*np.pi*(np.cos(2*np.pi*(new_x-xmin[0])/dom_len[0])*dxdx/dom_len[0] + np.cos(2*np.pi*(new_y-xmin[1])/dom_len[1])*dydx/dom_len[1])
        dzdy = 0.25*warp_factor*dom_len[2]*2*np.pi*(np.cos(2*np.pi*(new_x-xmin[0])/dom_len[0])*dxdy/dom_len[0] + np.cos(2*np.pi*(new_y-xmin[1])/dom_len[1])*dydy/dom_len[1])
        dzdz = np.ones(np.shape(dxdx))
        return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
    else:
        raise Exception('order > 1 invalid') 
    
def warp_chan_3d(x,y,z,warp_factor,order,xmin=np.array([0.,0.,0.]),xmax=np.array([1.,1.,1.])):
    ''' paper uses warp_factor = 0.125 '''
    dom_len = xmax - xmin
    argx = 2*(x-xmin[0])/dom_len[0] - 1
    argy = 2*(y-xmin[1])/dom_len[1] - 1
    argz = 2*(z-xmin[1])/dom_len[2] - 1

    if order == 0:  
        xn = x + warp_factor*dom_len[0]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        yn = y + warp_factor*dom_len[1]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        zn = z + warp_factor*dom_len[2]*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        return xn , yn, zn
    elif order == 1:
        dxndx = 1 - warp_factor*np.pi*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dxndy = - warp_factor*dom_len[0]/dom_len[1]*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dxndz = - warp_factor*dom_len[0]/dom_len[2]*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
        dyndx = - warp_factor*dom_len[1]/dom_len[0]*np.pi*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dyndy = 1 - warp_factor*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dyndz = - warp_factor*dom_len[1]/dom_len[2]*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
        dzndx = - warp_factor*np.pi*dom_len[2]/dom_len[0]*np.sin(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dzndy = - warp_factor*dom_len[2]/dom_len[1]*np.pi*np.cos(np.pi*argx/2.)*np.sin(np.pi*argy/2.)*np.cos(np.pi*argz/2.)
        dzndz = 1 - warp_factor*np.pi*np.cos(np.pi*argx/2.)*np.cos(np.pi*argy/2.)*np.sin(np.pi*argz/2.)
        return dxndx, dxndy, dxndz, dyndx, dyndy, dyndz, dzndx, dzndy, dzndz
    else:
        raise Exception('order > 1 invalid') 

def warp_cubic_3d(x,y,z,warp_factor,order,xmin=np.array([0.,0.,0.]),xmax=np.array([1.,1.,1.])):
    ''' Warps according to a cubic (in each direction. Total order is X).'''
    assert(warp_factor<1 and warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
    dom_len = xmax - xmin
    xscale = (x-xmin[0])/dom_len[0]
    yscale = (y-xmin[1])/dom_len[1] 
    zscale = (z-xmin[1])/dom_len[2]

    warp = warp_factor*4.6
    a1 = 5
    b1 = -7 
    ax = a1*zscale**3 + b1*zscale**2 - (a1+b1)*zscale + 1.5
    a2 = 5
    b2 = -10 
    ay = a2*zscale**3 + b2*zscale**2 - (a2+b2)*zscale + 1.3
    a3 = 5 
    b3 = -8 
    az = a3*yscale**3 + b3*yscale**2 - (a3+b3)*yscale + 1.5
    if order == 0: 
        new_x = warp*dom_len[0]*(yscale**3-ax*yscale**2+(ax-1)*yscale)*(xscale**3-xscale) + x
        new_y = warp*dom_len[1]*(xscale**3-ay*xscale**2+(ay-1)*xscale)*(yscale**3-yscale**2) + y
        new_z = warp*dom_len[2]*(xscale**3-az*xscale**2+(az-1)*xscale)*(zscale**3-zscale) + z
        return new_x , new_y , new_z
    elif order == 1:
        daxdz = (3*a1*zscale**2 + 2*b1*zscale - (a1+b1))/dom_len[2]
        daydz = (3*a2*zscale**2 + 2*b2*zscale - (a2+b2))/dom_len[2]
        dazdy = (3*a3*yscale**2 + 2*b3*yscale - (a3+b3))/dom_len[1]
        dxdx = warp*(yscale**3-ax*yscale**2+(ax-1)*yscale)*(3*xscale**2-1) + 1
        dxdy = warp*dom_len[0]*(3*yscale**2-2*ax*yscale+(ax-1))*(xscale**3-xscale)/dom_len[1]
        dxdz = warp*dom_len[0]*(-daxdz*yscale**2+daxdz*yscale)*(xscale**3-xscale)
        dydx = warp*dom_len[1]*(3*xscale**2-2*ay*xscale+(ay-1))*(yscale**3-yscale**2)/dom_len[0]
        dydy = warp*(xscale**3-ay*xscale**2+(ay-1)*xscale)*(3*yscale**2-2*yscale) + 1
        dydz = warp*dom_len[1]*(-daydz*xscale**2+daydz*xscale)*(yscale**3-yscale**2)
        dzdx = warp*dom_len[2]*(3*xscale**2-2*az*xscale+(az-1))*(zscale**3-zscale)/dom_len[0]
        dzdy = warp*dom_len[2]*(-dazdy*xscale**2+dazdy*xscale)*(zscale**3-zscale)
        dzdz = warp*(xscale**3-az*xscale**2+(az-1)*xscale)*(3*zscale**2-1) + 1
        return dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz
    else:
        raise Exception('order > 1 invalid') 
    
def warp_quad_2d(x,y,warp_factor,order,xmin=np.array([0.,0.,0.]),xmax=np.array([1.,1.,1.])):
    ''' Warps according to a quadratic (in each direction. Total order is quartic).'''
    assert(warp_factor<1 and warp_factor>-1),'Invalid warp_factor. Use a value in range (-1,1)'
    dom_len = xmax - xmin
    xscale = (x-xmin[0])/dom_len[0]
    yscale = (y-xmin[1])/dom_len[1] 
    warp = 2*warp_factor
    if order == 0: 
        new_x = warp*dom_len[0]*(yscale**2-yscale)*(xscale**2-xscale) + x
        new_y = 2*warp*dom_len[1]*(xscale**2-xscale)*(yscale**2-yscale) + y
        return new_x, new_y
    elif order == 1:
        dxdx = warp*(yscale**2-yscale)*(2*xscale-1) + 1
        dxdy = warp*dom_len[0]*(2*yscale-1)*(xscale**2-xscale)/dom_len[1]
        dydx = 2*warp*dom_len[1]*(2*xscale-1)*(yscale**2-yscale)/dom_len[0]
        dydy = 2*warp*(xscale**2-xscale)*(2*yscale-1) + 1
        return dxdx, dxdy, dydx, dydy
    else:
        raise Exception('order > 1 invalid') 


def calc_x(xaff,warp_factor,warp_type,dim,xmin,xmax):
    warp_types = ['quadratic','cubic','ddrf','smooth','chan','none']
    assert (dim==2 or dim==3),f"Invalid dimension {dim}"

    xphys = np.zeros(xaff.shape)

    if warp_type == 'quadratic':
        if dim==2:
            xphys[0], xphys[1] = warp_quad_2d(xaff[0],xaff[1],warp_factor,0,xmin,xmax)
        else:
            raise ValueError(f"quadratic warping not implemented in 3d.")
    elif warp_type == 'cubic':
        if dim==2:
            xphys[0], xphys[1] = warp_cubic_2d(xaff[0],xaff[1],warp_factor,0,xmin,xmax)
        else:
            xphys[0], xphys[1], xphys[2] = warp_cubic_3d(xaff[0],xaff[1],xaff[2],warp_factor,0,xmin,xmax)
    elif warp_type == 'ddrf':
        if dim==2:
            xphys[0], xphys[1] = warp_default_2d(xaff[0],xaff[1],warp_factor,0,xmin,xmax)
        else:
            xphys[0], xphys[1], xphys[2] = warp_default_3d(xaff[0],xaff[1],xaff[2],warp_factor,0,xmin,xmax)
    elif warp_type == 'smooth':
        if dim==2:
            xphys[0], xphys[1] = warp_smooth_2d(xaff[0],xaff[1],warp_factor,0,xmin,xmax)
        else:
            raise ValueError(f"smooth warping not implemented in 3d.")
    elif warp_type == 'chan':
        if dim==2:
            xphys[0], xphys[1] = warp_chan_2d(xaff[0],xaff[1],warp_factor,0,xmin,xmax)
        else:
            xphys[0], xphys[1], xphys[2] = warp_chan_3d(xaff[0],xaff[1],xaff[2],warp_factor,0,xmin,xmax)
    elif warp_type == 'none':
        xphys = np.copy(xaff)
    else:
        raise ValueError(f"Invalid met_method option. Try one of: {', '.join(warp_types)}")
    
    return xphys

def calc_dxdxi(xaff,dxaffdxi,warp_factor,warp_type,dim,xmin,xmax):
    warp_types = ['quadratic','cubic','ddrf','smooth','chan','none']
    assert (dim==2 or dim==3),f"Invalid dimension {dim}"

    dxdxaff = np.zeros((dim,*xaff.shape))

    if warp_type == 'quadratic':
        if dim==2:
            dxdxaff[0,0], dxdxaff[0,1], \
            dxdxaff[1,0], dxdxaff[1,1] = warp_quad_2d(xaff[0],xaff[1],warp_factor,1,xmin,xmax)
        else:
            raise ValueError(f"quadratic warping not implemented in 3d.")
    elif warp_type == 'cubic':
        if dim==2:
            dxdxaff[0,0], dxdxaff[0,1], \
            dxdxaff[1,0], dxdxaff[1,1] = warp_cubic_2d(xaff[0],xaff[1],warp_factor,1,xmin,xmax)
        else:
            dxdxaff[0,0], dxdxaff[0,1], dxdxaff[0,2], \
            dxdxaff[1,0], dxdxaff[1,1], dxdxaff[1,2], \
            dxdxaff[2,0], dxdxaff[2,1], dxdxaff[2,2] = warp_cubic_3d(xaff[0],xaff[1],xaff[2],warp_factor,1,xmin,xmax)
    elif warp_type == 'ddrf':
        if dim==2:
            dxdxaff[0,0], dxdxaff[0,1], \
            dxdxaff[1,0], dxdxaff[1,1] = warp_default_2d(xaff[0],xaff[1],warp_factor,1,xmin,xmax)
        else:
            dxdxaff[0,0], dxdxaff[0,1], dxdxaff[0,2], \
            dxdxaff[1,0], dxdxaff[1,1], dxdxaff[1,2], \
            dxdxaff[2,0], dxdxaff[2,1], dxdxaff[2,2] = warp_default_3d(xaff[0],xaff[1],xaff[2],warp_factor,1,xmin,xmax)
    elif warp_type == 'smooth':
        if dim==2:
            dxdxaff[0,0], dxdxaff[0,1], \
            dxdxaff[1,0], dxdxaff[1,1] = warp_smooth_2d(xaff[0],xaff[1],warp_factor,1,xmin,xmax)
        else:
            raise ValueError(f"smooth warping not implemented in 3d.")
    elif warp_type == 'chan':
        if dim==2:
            dxdxaff[0,0], dxdxaff[0,1], \
            dxdxaff[1,0], dxdxaff[1,1] = warp_chan_2d(xaff[0],xaff[1],warp_factor,1,xmin,xmax)
        else:
            dxdxaff[0,0], dxdxaff[0,1], dxdxaff[0,2], \
            dxdxaff[1,0], dxdxaff[1,1], dxdxaff[1,2], \
            dxdxaff[2,0], dxdxaff[2,1], dxdxaff[2,2] = warp_chan_3d(xaff[0],xaff[1],xaff[2],warp_factor,1,xmin,xmax)
    elif warp_type == 'none':
        if dim==2:
            dxdxaff[0,0] = 1.
            dxdxaff[1,1] = 1.
        else:
            dxdxaff[0,0] = 1.
            dxdxaff[1,1] = 1.
            dxdxaff[2,2] = 1.
    else:
        raise ValueError(f"Invalid met_method option. Try one of: {', '.join(warp_types)}")
    
    dxdxi = np.einsum('ij...m,jkm->ik...m',dxdxaff,dxaffdxi)
    
    return dxdxi



##############################################################
# useful operator functions
##############################################################


def get_vandermonde_monomial_2d(xi,eta,p):
    if p==2:
        V = np.vstack((np.ones_like(xi),xi,eta,xi**2,xi*eta,eta**2)).T
    elif p==3:
        V = np.vstack((np.ones_like(xi),xi,eta,xi**2,xi*eta,eta**2,xi**3,xi**2*eta,xi*eta**2,eta**3)).T
    elif p==4:
        V = np.vstack((np.ones_like(xi),xi,eta,xi**2,xi*eta,eta**2,xi**3,xi**2*eta,xi*eta**2,eta**3,xi**4,xi**3*eta,xi**2*eta**2,xi*eta**3,eta**4)).T
    else:
        raise ValueError('p>4 not coded up for vandermonde matrix')
    return V

def get_vandermondeder_monomial_2d(xi,eta,p):
    zeros = np.zeros_like(xi)
    if p==2:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,2*xi,eta,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,xi,2.*eta)).T
    elif p==3:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,2*xi,eta,zeros,3*xi**2,2*xi*eta,eta**2,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,xi,2.*eta,zeros,xi**2,2*xi*eta,3*eta**2)).T
    elif p==4:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,2*xi,eta,zeros,3*xi**2,2*xi*eta,eta**2,zeros,4*xi**3,3*xi**2*eta,2*xi*eta**2,eta**3,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,xi,2.*eta,zeros,xi**2,2*xi*eta,3*eta**2,zeros,xi**3,xi**2*2*eta,xi*3*eta**2,4*eta**3)).T
    else:
        raise ValueError('p>4 not coded up for vandermonde matrix')
    return xider, etader

def get_vandermonde_monomial_3d(xi,eta,zeta,p):
    if p==2:
        V = np.vstack((np.ones_like(xi),xi,eta,zeta,xi**2,xi*eta,xi*zeta,eta**2,eta*zeta,zeta**2)).T
    elif p==3:
        V = np.vstack((np.ones_like(xi),xi,eta,zeta,xi**2,xi*eta,xi*zeta,eta**2,eta*zeta,zeta**2,xi**3,xi**2*eta,xi**2*zeta,xi*eta**2,xi*eta*zeta,xi*zeta**2,eta**3,eta**2*zeta,eta*zeta**2,zeta**3)).T
    elif p==4:
        V = np.vstack((np.ones_like(xi),xi,eta,zeta,xi**2,xi*eta,xi*zeta,eta**2,eta*zeta,zeta**2,xi**3,xi**2*eta,xi**2*zeta,xi*eta**2,xi*eta*zeta,xi*zeta**2,eta**3,eta**2*zeta,eta*zeta**2,zeta**3,xi**4,xi**3*eta,xi**3*zeta,xi**2*eta**2,xi**2*eta*zeta,xi**2*zeta**2,xi*eta**3,xi*eta**2*zeta,xi*eta*zeta**2,xi*zeta**3,eta**4,eta**3*zeta,eta**2*zeta**2,eta*zeta**3,zeta**4)).T
    else:
        raise ValueError('p>4 not coded up for vandermonde matrix')
    return V

def get_vandermondeder_monomial_3d(xi,eta,zeta,p):
    zeros = np.zeros_like(xi)
    if p==2:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,zeros,2.*xi,eta,zeta,zeros,zeros,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,2.*eta,zeta,zeros)).T
        zetader = np.vstack((zeros,zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,eta,2.*zeta)).T
    elif p==3:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,zeros,2.*xi,eta,zeta,zeros,zeros,zeros,3.*xi**2,2*xi*eta,2*xi*zeta,eta**2,eta*zeta,zeta**2,zeros,zeros,zeros,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,2.*eta,zeta,zeros,zeros,xi**2,zeros,xi*2.*eta,xi*zeta,zeros,3.*eta**2,2.*eta*zeta,zeta**2,zeros)).T
        zetader = np.vstack((zeros,zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,eta,2.*zeta,zeros,zeros,xi**2,zeros,xi*eta,2.*xi*zeta,zeros,eta**2,2.*eta*zeta,3.*zeta**2)).T
    elif p==4:
        xider = np.vstack((zeros,np.ones_like(xi),zeros,zeros,2.*xi,eta,zeta,zeros,zeros,zeros,3.*xi**2,2*xi*eta,2*xi*zeta,eta**2,eta*zeta,zeta**2,zeros,zeros,zeros,zeros,4*xi**3,3*xi**2*eta,3*xi**2*zeta,2*xi*eta**2,2*xi*eta*zeta,2*xi*zeta**2,eta**3,eta**2*zeta,eta*zeta**2,zeta**3,zeros,zeros,zeros,zeros,zeros)).T
        etader = np.vstack((zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,2.*eta,zeta,zeros,zeros,xi**2,zeros,xi*2.*eta,xi*zeta,zeros,3.*eta**2,2.*eta*zeta,zeta**2,zeros,zeros,xi**3,zeros,xi**2*2*eta,xi**2*zeta,zeros,xi*3*eta**2,xi*2*eta*zeta,xi*zeta**2,zeros,4*eta**3,3*eta**2*zeta,2*eta*zeta**2,zeta**3,zeros)).T
        zetader = np.vstack((zeros,zeros,zeros,np.ones_like(xi),zeros,zeros,xi,zeros,eta,2.*zeta,zeros,zeros,xi**2,zeros,xi*eta,2.*xi*zeta,zeros,eta**2,2.*eta*zeta,3.*zeta**2,zeros,zeros,xi**3,zeros,xi**2*eta,xi**2*2*zeta,zeros,xi*eta**2,xi*eta*2*zeta,3*xi*zeta**2,zeros,eta**3,eta**2*2*zeta,eta*3*zeta**2,4*zeta**3)).T
    else:
        raise ValueError('p>4 not coded up for vandermonde matrix')
    return xider, etader, zetader

def legendre_derivative(n, x):
    """
    Compute the derivative of the nth Legendre polynomial at x.
    """
    if n == 0:
        return np.zeros_like(x)
    else:
        Pn = legendre(n)
        Pn_1 = legendre(n-1)
        return n * (Pn_1(x) - x * Pn(x)) / (1 - x**2)

def get_vandermonde_2d(x, y, p):
    """
    Generate a 2D Vandermonde matrix using Legendre polynomials of degree p for given nodal values x and y.

    Parameters:
    x, y : arrays of nodal values
    p : degree of Legendre polynomials

    Returns:
    V : 2D Vandermonde matrix
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Number of nodal points
    n_points = len(x)
    
    # Number of terms in the polynomial basis
    n_terms = (p + 1) * (p + 2) // 2
    
    # Initialize the Vandermonde matrix
    V = np.zeros((n_points, n_terms))
    
    # Fill the Vandermonde matrix
    term = 0
    for i in range(p + 1):
        for j in range(p - i + 1):
            # Evaluate the Legendre polynomial at each nodal point
            L_i = legendre(i)(x)
            L_j = legendre(j)(y)
            
            # Compute the product and store it in the matrix
            V[:, term] = L_i * L_j
            term += 1
    
    return V

def get_vandermondeder_2d(x, y, p):
    """
    Generate the derivative of the 2D Vandermonde matrix using Legendre polynomials of degree p for given nodal values x and y.

    Parameters:
    x, y : arrays of nodal values
    p : degree of Legendre polynomials

    Returns:
    Vx, Vy : Derivative Vandermonde matrices with respect to x and y
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Number of nodal points
    n_points = len(x)
    
    # Number of terms in the polynomial basis
    n_terms = (p + 1) * (p + 2) // 2
    
    # Initialize the derivative Vandermonde matrices
    Vx = np.zeros((n_points, n_terms))
    Vy = np.zeros((n_points, n_terms))
    
    # Fill the derivative Vandermonde matrices
    term = 0
    for i in range(p + 1):
        for j in range(p - i + 1):
            # Evaluate the derivatives of the Legendre polynomial at each nodal point
            dL_i_dx = legendre_derivative(i, x)
            dL_j_dy = legendre_derivative(j, y)
            
            # Evaluate the Legendre polynomials at each nodal point
            L_i = legendre(i)(x)
            L_j = legendre(j)(y)
            
            # Compute the products for the derivative matrices
            Vx[:, term] = dL_i_dx * L_j
            Vy[:, term] = L_i * dL_j_dy
            
            term += 1
    
    return Vx, Vy

def get_vandermonde_3d(x, y, z, p):
    """
    Generate a Vandermonde matrix using Legendre polynomials of degree p for given nodal values x, y, and z.

    Parameters:
    x, y, z : arrays of nodal values
    p : degree of Legendre polynomials

    Returns:
    V : Vandermonde matrix
    """
    # Ensure x, y, and z are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Number of nodal points
    n_points = len(x)
    
    # Number of terms in the polynomial basis
    n_terms = (p + 1) * (p + 2) * (p + 3) // 6
    
    # Initialize the Vandermonde matrix
    V = np.zeros((n_points, n_terms))
    
    # Fill the Vandermonde matrix
    term = 0
    for i in range(p + 1):
        for j in range(p - i + 1):
            for k in range(p - i - j + 1):
                # Evaluate the Legendre polynomial at each nodal point
                L_i = legendre(i)(x)
                L_j = legendre(j)(y)
                L_k = legendre(k)(z)
                
                # Compute the product and store it in the matrix
                V[:, term] = L_i * L_j * L_k
                term += 1
    
    return V

def get_vandermondeder_3d(x, y, z, p):
    """
    Generate the derivative of the Vandermonde matrix using Legendre polynomials of degree p for given nodal values x, y, and z.

    Parameters:
    x, y, z : arrays of nodal values
    p : degree of Legendre polynomials

    Returns:
    Vx, Vy, Vz : Derivative Vandermonde matrices with respect to x, y, and z
    """
    # Ensure x, y, and z are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Number of nodal points
    n_points = len(x)
    
    # Number of terms in the polynomial basis
    n_terms = (p + 1) * (p + 2) * (p + 3) // 6
    
    # Initialize the derivative Vandermonde matrices
    Vx = np.zeros((n_points, n_terms))
    Vy = np.zeros((n_points, n_terms))
    Vz = np.zeros((n_points, n_terms))
    
    # Fill the derivative Vandermonde matrices
    term = 0
    for i in range(p + 1):
        for j in range(p - i + 1):
            for k in range(p - i - j + 1):
                # Evaluate the derivatives of the Legendre polynomial at each nodal point
                dL_i_dx = legendre_derivative(i, x)
                dL_j_dy = legendre_derivative(j, y)
                dL_k_dz = legendre_derivative(k, z)
                
                # Evaluate the Legendre polynomials at each nodal point
                L_i = legendre(i)(x)
                L_j = legendre(j)(y)
                L_k = legendre(k)(z)
                
                # Compute the products for the derivative matrices
                Vx[:, term] = dL_i_dx * L_j * L_k
                Vy[:, term] = L_i * dL_j_dy * L_k
                Vz[:, term] = L_i * L_j * dL_k_dz
                
                term += 1
    
    return Vx, Vy, Vz

StartUpDG_julia_code = """

using Pkg

# Check if virtual environment has already been set up & compiled. If not, activate.
if !haskey(ENV, "JULIA_STARTUPDG_ENV_READY")
    Pkg.activate("//Users/alex/julia_environments/StartUpDG")

    # Precompile packages
    Pkg.precompile()
    ENV["JULIA_ENV_READY"] = "true"
end

using StartUpDG: basis, nodes, vandermonde, Tet, Tri

function get_nodes_2d(N)
    r, s = nodes(Tri(), N)
    return r, s
end

function get_basis_2d(r, s, N)
    V, Vr, Vs = basis(Tri(), N, r, s)
    return V, Vr, Vs
end

function get_vandermonde_2d(r, s, N)
    V = vandermonde(Tri(), N, r, s)
    return V
end

function get_nodes_3d(N)
    r, s, t = nodes(Tet(), N)
    return r, s, t
end

function get_basis_3d(r, s, t, N)
    V, Vr, Vs, Vt = basis(Tet(), N, r, s, t)
    return V, Vr, Vs, Vt
end

function get_vandermonde_3d(r, s, t, N)
    V = vandermonde(Tet(), N, r, s, t)
    return V
end

"""

Main.eval(StartUpDG_julia_code)

# will construct interpolation nodes based on LGL nodes on the edges
# note, this returns nodes for the right triangle [-1,1], same as hicken
StartUpDG_nodes_2d = Main.get_nodes_2d
StartUpDG_nodes_3d = Main.get_nodes_3d

# will return a Vandermonde matrix and it derivaties of Jacobi polynomials 
# (orthogonal on right triangle) evaluated at the given x,y,z
StartUpDG_basis_2d = Main.get_basis_2d
StartUpDG_basis_3d = Main.get_basis_3d

# returns only the Vandermonde matrix, calls the same as above
StartUpDG_vandermonde_2d = Main.get_vandermonde_2d
StartUpDG_vandermonde_3d = Main.get_vandermonde_3d


def tetraCoord(A,B,C,D):
    v1 = B-A ; v2 = C-A ; v3 = D-A
    mat = np.array((v1,v2,v3)).T
    # mat is 3x3 here
    M1 = np.linalg.inv(mat)
    return(M1)

def pointInsideTetrahedron(v1,v2,v3,v4,p):
    # Find the transform matrix from orthogonal to tetrahedron system
    M1=tetraCoord(v1,v2,v3,v4)
    # apply the transform to P (v1 is the origin)
    newp = M1.dot(p-v1)
    # perform test
    return (np.all(newp>=0) and np.all(newp <=1) and np.sum(newp)<=1)

def generate_rand_nodes(n,p):
    nodes = np.zeros((3,n))
    v1 = np.array([0.,0.,0.])
    v2 = np.array([1.,0.,0.])
    v3 = np.array([0.,1.,0.])
    v4 = np.array([0.,0.,1.])
    for i in range(1,n):
        done = False
        while not done:
            xis = np.random.rand(3)
            if pointInsideTetrahedron(v1,v2,v3,v4,xis):
                distanceok = True
                for j in range(i):
                    if np.linalg.norm(xis - nodes[:,j]) < 0.5/(p**3):
                        distanceok = False
                if distanceok:
                    done = True
        nodes[:,i] = xis
    return nodes

##############################################################
# test Operators
##############################################################

def test_operator_2d(Dx,Dy,x,y,tol=1e-10,returndegree=False):
    degx = 0
    degy = 0
    for p in range(1,40):
        #V = get_vandermonde_2d(x,y,p)
        #Vx, Vy = get_vandermondeder_2d(x,y,p)
        V, Vx, Vy = StartUpDG_basis_2d(x,y,p)
        er1 = np.max(abs(Dx@V-Vx))
        er2 = np.max(abs(Dy@V-Vy))
        if er1<tol: degx+=1
        if er2<tol: degy+=1
        if er1>tol and er2>tol:
            break
    if returndegree:
        return degx,degy
    else:
        print('Test: Dx, Dy are degree {0}, {1}.'.format(degx,degy))

def test_operator_3d(Dx,Dy,Dz,x,y,z,tol=1e-10,returndegree=False):
    degx = 0
    degy = 0
    degz = 0
    for p in range(1,40):
        #V = get_vandermonde_3d(x,y,z,p)
        #Vx, Vy, Vz = get_vandermondeder_3d(x,y,z,p)
        V, Vx, Vy, Vz = StartUpDG_basis_3d(x,y,z,p)
        er1 = np.max(abs(Dx@V-Vx))
        er2 = np.max(abs(Dy@V-Vy))
        er3 = np.max(abs(Dz@V-Vz))
        if er1<tol: degx+=1
        if er2<tol: degy+=1
        if er3<tol: degz+=1
        if er1>tol and er2>tol and er3>tol:
            break
    if returndegree:
        return degx,degy,degz
    else:
        print('Test: Dx, Dy, Dz are degree {0}, {1}, {2}.'.format(degx,degy,degz))

def test_Exi_2d(x,Ex,tol=1e-10,returndegree=False):
    ''' tests matrix Ex'''
    p=0
    er=0
    while er<tol:
        p+=1
        # reference tri is [-1,1]
        er = abs(np.sum(Ex @ x**p) + 2*(-1)**p - (1 / (p + 1)) * (1 - (-1)**(p + 1)) )
    if returndegree:
        return int(p-1)
    else:
        print('Test: Exi is degree {0}.'.format(p-1))

def test_Exi_3d(x,Ex,tol=1e-10,returndegree=False):
    ''' tests matrix Ex'''
    p=0
    er=0
    while er<tol:
        p+=1
        # reference tet is [-1,1]
        er = abs(np.sum(Ex @ x**p) + 2*(-1)**p - (2*(-1)**p*p+3*(-1)**p+1)/((p+1)*(p+2)) )
    if returndegree:
        return int(p-1)
    else:
        print('Test: Exi is degree {0}.'.format(p-1))

def test_quad_3d(H,x,tol=1e-10,returndegree=False):
    ''' tests volume quadrature H'''
    p=0
    deg=-1
    er=0
    while er<tol:
        #  reference tri is [-1,1]
        if p%2==0: # even
            integral = (2*p + 4)/(4*((p/2+1)**2)-1)
        else: # odd
            integral = -2/(p+2)
        er = abs(H @ (x**p) - integral )
        if er < tol: 
            p+=1
            deg+=1
    if returndegree:
        return int(deg)
    else:
        print('Test: Volume Quadrature H is degree {0}.'.format(deg))

def test_quad_2d(B,x,tol=1e-10,returndegree=False,facet=True):
    ''' tests surface quadrature B in 3d, or volume quadrature H in 2d'''
    p=0
    deg=-1
    er=0
    while er<tol:
        #  reference tri is [-1,1]
        if p%2==0: # even
            integral = 2/(p+1)
        else: # odd
            integral = -2/(p+2)
        er = abs(B @ (x**p) - integral)
        if er < tol: 
            p+=1
            deg+=1
    if returndegree:
        return int(deg)
    else:
        if facet:
            print('Test: Facet Quadrature B is degree {0}.'.format(deg))
        else:
            print('Test: Volume Quadrature H is degree {0}.'.format(deg))


def test_quad_1d(B,x,tol=1e-10,returndegree=False):
    ''' tests surface quadrature B in 2d'''
    p=0
    deg=-1
    er=0
    while er<tol:
        #  reference line is [-1,1]
        integral = (1 / (p + 1)) * (1 - (-1)**(p + 1))
        er = abs(B @ (x**p) - integral)
        if er < tol: 
            p+=1
            deg+=1
    if returndegree:
        return int(deg)
    else:
        print('Test: Facet Quadrature B is degree {0}.'.format(deg))

def test_extrap_2d(R,x,y,xf,yf,tol=1e-10,returndegree=False):
    ''' tests extrapolation R'''
    p=0
    deg=-1
    er=0
    while er<tol:
        #  reference tri is [-1,1]
        er1 = R@(x**p) - xf**p
        er2 = R@(y**p) - yf**p
        er = np.max(abs(np.array([er1,er2])))
        if er < tol: 
            p+=1
            deg+=1
        if p==20: 
            deg = np.inf
            break
    if returndegree:
        return int(deg)
    else:
        print('Test: Extrapolation R is degree {0}.'.format(deg))

def test_extrap_3d(R,x,y,z,xf,yf,zf,tol=1e-10,returndegree=False):
    ''' tests extrapolation R'''
    p=0
    deg=-1
    er=0
    while er<tol:
        #  reference tri is [-1,1]
        er1 = R@(x**p) - xf**p
        er2 = R@(y**p) - yf**p
        er3 = R@(z**p) - zf**p
        er = np.max(abs(np.array([er1,er2,er3])))
        if er < tol: 
            p+=1
            deg+=1
        if p==20: 
            deg = np.inf
            break
    if returndegree:
        return int(deg)
    else:
        print('Test: Extrapolation R is degree {0}.'.format(deg))



##############################################################
# grid functions
##############################################################

def load_mesh(name,xref,xfref,dim):
    # Initialize Gmsh
    gmsh.initialize()

    # Open the mesh file
    gmsh.open(name)

    # Extract mesh nodes and elements
    _, vertices_coords, _  = gmsh.model.mesh.getNodes(dim=-1, tag=-1, includeBoundary = False, returnParametricCoord = False)

    # Reshape coordinates array
    vertices_coords = vertices_coords.reshape(-1, 3)[:,:dim]

    #elementType=2, 3-node triangle
    if dim==2:
        elements = gmsh.model.mesh.getElementsByType(2) 
    elif dim==3:
        elements = gmsh.model.mesh.getElementsByType(4) 
    else:
        ValueError(f"Invalid dim={dim}")
    element_tags, element_vertices_tags = elements

    # Finalize Gmsh
    gmsh.finalize()

    # get the affine mapping for each element
    num_elem = len(element_tags)
    num_fac = dim + 1
    affine_map = np.zeros((dim,dim+1,num_elem))
    x_affine = np.zeros((*xref.shape,len(element_tags)))
    xf_affine = np.zeros((*xfref.shape,len(element_tags)))

    for elem_idx in range(num_elem): 
        # Get the vertex indices of the i-th element
        elem_vertex_indices = element_vertices_tags[elem_idx * num_fac:(elem_idx + 1) * num_fac] - 1  # Convert to zero-based indexing
        #elem_tag = element_tags[elem_idx]
        
        # Get the physical (mapped) coordinates of the element vertices
        p_phys = vertices_coords[elem_vertex_indices, :dim].T

        if dim == 2:
            # The reference vertex coordinates are (-1,-1), (1,-1), (-1,1).
            # The below is a row of xi, a row of eta, and a row of ones (for +b part of affine trans.)
            p_ref = np.array([[-1.,1.,-1.],[-1.,-1.,1.],[1.,1.,1.]])
        else:
            # The reference vertex coordinates are sbp.vtx = (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
            # The below is a row of xi, a row of eta, a row of zeta, and a row of ones (for +b part of affine trans.)
            p_ref = np.array([[-1.,1.,-1.,-1,],[-1.,-1.,1.,-1.],[-1.,-1.,-1.,1.],[1.,1.,1.,1.]])

        affine_map[:,:,elem_idx] = np.dot(p_phys, np.linalg.inv(p_ref))
        
        # get physical (affine) nodes and store
        x_affine[:,:,elem_idx] = affine_map[:,:dim,elem_idx] @ xref + affine_map[:,dim,elem_idx].reshape((dim,1))

        for f in range(num_fac):
            xf_affine[:,:,f,elem_idx] = affine_map[:,:dim,elem_idx] @ xfref[:,:,f] + affine_map[:,dim,elem_idx].reshape((dim,1))

    return element_vertices_tags, x_affine, xf_affine, affine_map

def plot_mesh(name,x,xf,dim,transformation,savefile=None):
    assert(dim==2),"Plotting only set up right now for dim=2"

    # Initialize Gmsh
    gmsh.initialize()

    # Open the mesh file
    gmsh.open(name)

    # Extract mesh nodes and elements
    _, vertices_coords, _  = gmsh.model.mesh.getNodes(dim=-1, tag=-1, includeBoundary = False, returnParametricCoord = False)

    # Reshape coordinates array
    vertices_coords = vertices_coords.reshape(-1, 3)[:,:dim]

    #elementType=2, 3-node triangle
    if dim==2:
        elements = gmsh.model.mesh.getElementsByType(2) 
    elif dim==3:
        elements = gmsh.model.mesh.getElementsByType(4) 
    else:
        ValueError(f"Invalid dim={dim}")
    element_tags, element_vertices_tags = elements

    # Finalize Gmsh
    gmsh.finalize()

    def interpolate_edge(p1, p2, num_points=10):
        t = np.linspace(0, 1, num_points)
        return (1 - t)[:, None] * p1 + t[:, None] * p2

    # Plot using matplotlib
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(frameon=False)

    for elem_idx in range(len(element_tags)):
        # Get the vertex indices of the i-th element
        elem_vertex_idcs = element_vertices_tags[elem_idx * 3:(elem_idx + 1) * 3] - 1  # Convert to zero-based indexing
        
        # Get the coordinates of the element vertices
        triangle = vertices_coords[elem_vertex_idcs, :2]

        # Interpolate points along each edge of the triangle
        edge1 = interpolate_edge(triangle[0], triangle[1])
        edge2 = interpolate_edge(triangle[1], triangle[2])
        edge3 = interpolate_edge(triangle[2], triangle[0])

        # Combine the edges to form a smoother polygon
        smooth_triangle = np.vstack([edge1, edge2, edge3])

        # Plot the warped mesh elements
        warped_triangle = transformation(smooth_triangle.T).T
        warped_polygon = plt.Polygon(warped_triangle, edgecolor='k', facecolor='none',lw=1)
        ax.add_patch(warped_polygon)

        for f in [0,1,2]:
            ax.scatter(xf[0, :, f, elem_idx], xf[1, :, f, elem_idx], s=16, c='r', marker='s', label='Facet nodes')

        # Calculate the centroid of the triangle and place the tag ontop
        #centroid = np.mean(warped_triangle, axis=0)
        #ax.text(centroid[0], centroid[1], str(elem_idx)+"("+str(elem_tag)+")", color='blue', ha='center', va='center')
        #ax.text(centroid[0], centroid[1], str(elem_idx), color='blue', ha='center', va='center')

    for elem_idx in range(len(element_tags)):
        # do this in a separate loop to make sure it plots on top of all the facet nodes
        ax.scatter(x[0, :, elem_idx], x[1, :, elem_idx], s=16, c='b', marker='o',  label='Volume nodes')

    
    ax.tick_params(axis='both',length=0,labelsize=12,pad=-11) # hide ticks
    edge_verticesx = np.linspace(0,1,5)
    edge_verticesy = np.linspace(0,1,5)
    ax.set_xticks(edge_verticesx) # label element boundaries
    ax.set_yticks(edge_verticesy)
    
    #ax.set_aspect('equal')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title(name)

    if savefile is not None:
        fig.tight_layout()
        fig.savefig(savefile,format='png',dpi=600)
    else:
        plt.show()
    
    

##############################################################
# metrics functions
##############################################################

def calc_met_exa(dxdxi,dim,returnjac=True):
    ''' given the exact inverse metrics, calculate the exact metrics '''
    metrics = np.zeros_like(dxdxi)

    if dim==2:
        metrics[0,0] = dxdxi[1,1]
        metrics[0,1] = - dxdxi[0,1]
        metrics[1,0] = - dxdxi[1,0]
        metrics[1,1] = dxdxi[0,0]
        if returnjac:
            jac = dxdxi[0,0]*dxdxi[1,1] - dxdxi[0,1]*dxdxi[1,0]
    else:
        metrics[0,0] = dxdxi[2,2]*dxdxi[1,1] - dxdxi[2,1]*dxdxi[1,2]
        metrics[0,1] = dxdxi[0,2]*dxdxi[2,1] - dxdxi[0,1]*dxdxi[2,2]
        metrics[0,2] = dxdxi[1,2]*dxdxi[0,1] - dxdxi[1,1]*dxdxi[0,2]
        metrics[1,0] = dxdxi[2,0]*dxdxi[1,2] - dxdxi[2,2]*dxdxi[1,0]
        metrics[1,1] = dxdxi[0,0]*dxdxi[2,2] - dxdxi[0,2]*dxdxi[2,0]
        metrics[1,2] = dxdxi[1,0]*dxdxi[0,2] - dxdxi[1,2]*dxdxi[0,0]
        metrics[2,0] = dxdxi[2,1]*dxdxi[1,0] - dxdxi[2,0]*dxdxi[1,1]
        metrics[2,1] = dxdxi[0,1]*dxdxi[2,0] - dxdxi[0,0]*dxdxi[2,1]
        metrics[2,2] = dxdxi[1,1]*dxdxi[0,0] - dxdxi[1,0]*dxdxi[0,1]
        if returnjac: 
            jac = dxdxi[0,0]*(dxdxi[1,1]*dxdxi[2,2] - dxdxi[1,2]*dxdxi[2,1]) \
                - dxdxi[1,0]*(dxdxi[0,1]*dxdxi[2,2] - dxdxi[0,2]*dxdxi[2,1]) \
                + dxdxi[2,0]*(dxdxi[0,1]*dxdxi[1,2] - dxdxi[0,2]*dxdxi[1,1])
    if returnjac:
        return metrics, jac
    else:
        return metrics
    
def calc_normal_exa(metricsf,Nxi,dim,returnfac=True):
    ''' compute the outward normal using Nanson's formula '''

    Nxi_normed = Nxi / np.linalg.norm(Nxi,axis=0)
    normal = np.einsum('lmnfe,lf->mnfe', metricsf, Nxi_normed)
    fac = np.linalg.norm(normal,axis=0)
    normal = normal / fac

    if returnfac:
        return normal, fac
    else:
        return normal


def calc_direct(D,x,dim):
    ''' calculate the metrics using the direct approach '''

    dxdxi = np.einsum('lij,mj...->mli...', D, x, optimize='optimal')
    metrics = calc_met_exa(dxdxi,dim,returnjac=False)
    
    return metrics

def calc_direct_project(D,H,x,xi,p,dim,use_pinv):
    ''' apply a projection at the end. Shouldn't work though. '''
    
    metrics = calc_direct(D,x,dim)

    if dim == 2:
        V = StartUpDG_vandermonde_2d(xi[0],xi[1],p)
    elif dim == 3:
        #V = get_vandermonde_3d(xi[0],xi[1],xi[2],p)
        V = StartUpDG_vandermonde_3d(xi[0],xi[1],xi[2],p)
    else:
        raise ValueError(f"Invalid dim = {dim}")
    
    if use_pinv:
        P = V @ np.linalg.pinv(V) # interpolation matrix
    else:
        # quadrature-based projection
        M = (V.T * H) @ V # modal mass matrix
        P = V @ np.linalg.inv(M) @ V.T * H

    for l in range(dim):
        for m in range(dim):
            metrics[l,m] = P @ metrics[l,m]

    return metrics

def calc_thomaslombard(D,x):
    ''' calculate the metrics using the Thomas and Lombard approach '''
    # x: (dim, node, element)
    # D: (dim, node, node)
    # dx_dxi: (dim, dim, node, element)
    # metrics: (dim, dim, node, element)

    metrics = np.zeros((3,*x.shape))
    dx_dxi = np.einsum('lij,mje->mlie', D, x, optimize='optimal')

    for l in range(3):
        for m in range(3):
            n = (m+1)%3
            k = (m+2)%3
            s = (l+1)%3
            t = (l+2)%3
            metrics[l,m] = np.einsum('ij,je,je->ie',D[t],x[k],dx_dxi[n,s],optimize='optimal') \
                         - np.einsum('ij,je,je->ie',D[s],x[k],dx_dxi[n,t],optimize='optimal') 

    return metrics

def calc_vinokuryee(D,x):
    ''' calculate the metrics using the Vinokur and Yee approach '''
    # x: (dim, node, element)
    # D: (dim, node, node)
    # dx_dxi: (dim, dim, node, element)
    # metrics: (dim, dim, node, element)

    metrics = np.zeros((3,*x.shape))
    dx_dxi = np.einsum('lij,mje->mlie', D, x, optimize='optimal')

    for l in range(3):
        for m in range(3):
            n = (m+1)%3
            k = (m+2)%3
            s = (l+1)%3
            t = (l+2)%3
            metrics[l,m] = 0.5*(np.einsum('ij,je,je->ie',D[t],x[k],dx_dxi[n,s],optimize='optimal') \
                              - np.einsum('ij,je,je->ie',D[s],x[k],dx_dxi[n,t],optimize='optimal') \
                              - np.einsum('ij,je,je->ie',D[t],x[n],dx_dxi[k,s],optimize='optimal') \
                              + np.einsum('ij,je,je->ie',D[s],x[n],dx_dxi[k,t],optimize='optimal') )

    return metrics

def calc_kopriva(D,H,x,xi,p,symmetric,dim,use_pinv):
    ''' calculate the metrics using the Kopriva approach 
        Vinokur and Yee variant: symmetric = True
        Thomas and Lombdard variant: symmetric = False '''
    # x: (dim, node, element)
    # D: (dim, node, node)
    # dx_dxi: (dim, dim, node, element)
    # metrics: (dim, dim, node, element)

    if dim == 2:
        V = StartUpDG_vandermonde_2d(xi[0],xi[1],p)
    elif dim == 3:
        #V = get_vandermonde_3d(xi[0],xi[1],xi[2],p)
        V = StartUpDG_vandermonde_3d(xi[0],xi[1],xi[2],p)
    else:
        raise ValueError(f"Invalid dim = {dim}")
    
    if use_pinv:
        P = V @ np.linalg.pinv(V) # interpolation matrix
    else:
        # quadrature-based projection
        M = (V.T * H) @ V # modal mass matrix
        P = V @ np.linalg.inv(M) @ V.T * H

    metrics = np.zeros((dim,*x.shape))

    if dim == 2:
        metrics[0,0] = D[1] @ P @ x[1]
        metrics[0,1] = - D[1] @ P @ x[0]
        metrics[1,0] = - D[0] @ P @ x[1]
        metrics[1,1] = D[0] @ P @ x[0]
    
    else:
        dx_dxi = np.einsum('lij,mje->mlie', D, x, optimize='optimal')
        for l in range(3):
            for m in range(3):
                n = (m+1)%3
                k = (m+2)%3
                s = (l+1)%3
                t = (l+2)%3
                if symmetric:
                    metrics[l,m] = 0.5*(np.einsum('ij,jk,ke,ke->ie',D[t],P,x[k],dx_dxi[n,s],optimize='optimal') \
                                      - np.einsum('ij,jk,ke,ke->ie',D[s],P,x[k],dx_dxi[n,t],optimize='optimal') \
                                      - np.einsum('ij,jk,ke,ke->ie',D[t],P,x[n],dx_dxi[k,s],optimize='optimal') \
                                      + np.einsum('ij,jk,ke,ke->ie',D[s],P,x[n],dx_dxi[k,t],optimize='optimal') )
                else:
                    metrics[l,m] = np.einsum('ij,jk,ke,ke->ie',D[t],P,x[k],dx_dxi[n,s],optimize='optimal') \
                                 - np.einsum('ij,jk,ke,ke->ie',D[s],P,x[k],dx_dxi[n,t],optimize='optimal') 

    return metrics

def calc_chan(D,H,x,xi,p,symmetric,dim,op_type,fac_op_type,use_pinv,skipp,init_x,transf,affine_map):
    ''' calculate the metrics using the Chan and Wilcox approach 
        Vinokur and Yee variant: symmetric = True
        Thomas and Lombdard variant: symmetric = False '''
    # x: (dim, node, element)
    # D: (dim, node, node)
    # dx_dxi: (dim, dim, node, element)
    # metrics: (dim, dim, node, element)

    if dim == 2:
        Vp_at_xi = StartUpDG_vandermonde_2d(xi[0], xi[1], p) # p basis functions evaluated at xi nodes
        Vp1_at_xi = StartUpDG_vandermonde_2d(xi[0], xi[1], p+1) # p+1 basis functions evaluated at xi nodes
        nmp1 = int((p+2)*(p+3)/2) # number of modes (basis functions) for p+1
        nmp = int((p+1)*(p+2)/2) # number of modes (basis functions) for p
    elif dim == 3:
        Vp_at_xi = StartUpDG_vandermonde_3d(xi[0], xi[1], xi[2], p) # p basis functions evaluated at xi nodes
        Vp1_at_xi = StartUpDG_vandermonde_3d(xi[0], xi[1], xi[2], p+1) # p+1 basis functions evaluated at xi nodes
        nmp1 = int((p+2)*(p+3)*(p+4)/6) # number of modes (basis functions) for p+1
        nmp = int((p+1)*(p+2)*(p+3)/6) # number of modes (basis functions) for p
    else:
        raise ValueError(f"Invalid dimension {dim}")
    nxi = xi.shape[1]

    if op_type.lower() == 'modal' or  op_type.lower() == 'modal_rand':
        nxip1 = nmp1
        
        if dim == 2:
            if op_type.lower() == 'modal_rand':
                print("WARNING: randomly spaced nodes, i.e. chan_up_type = Modal_Rand, is not set up for dim=2. Defaulting to Modal.")
            xip1 = np.array(StartUpDG_nodes_2d(p+1)) # p+1 LGL interpolation nodes
            temp = StartUpDG_basis_2d(xip1[0], xip1[1], p+1) # p+1 basis evaluated at interpolation nodes
        else:
            if op_type.lower() == 'modal':
                xip1 = np.array(StartUpDG_nodes_3d(p+1)) # p+1 LGL interpolation nodes
            else:
                xip1 = np.array(generate_rand_nodes(nxip1,p+1)) # randomly placed nodes to use for interpolation
            temp = StartUpDG_basis_3d(xip1[0], xip1[1], xip1[2], p+1) # p+1 basis evaluated at interpolation nodes
        Vp1, Vp1der = temp[0], np.array(temp[1:])
        Vp1inv = np.linalg.inv(Vp1)
        Ixip1_to_xi = Vp1_at_xi @ Vp1inv # interpolation from p+1 interpolation nodes (xip1) to (p+1 basis, then) xi nodes.
        Dp1 = np.zeros((dim,nxip1,nmp1))
        for i in range(dim):
            Dp1[i] = Vp1der[i] @ Vp1inv # take modal derivatives to nodal derivatives
        Pp1 = np.eye(nxip1) # projection to a degree p+1 polynomial space, trivial for this modal form

        # now calculate x 
        if init_x == 'xp1' or init_x == 'interp_p1': 
            # start using the most possible accuract xp1, that is the transformation applied to p+1 interpolation points
            if skipp:
                print("WARNING: indicated skipp=False, but also init_x='interp_p1'. Since the transformation is applied to degree")
                print("         p+1 interpolation points, by default we are degree p+1 and skip p interpolation. Ignoring skipp=False.")
            if not use_pinv:
                print("WARNING: indicated use_pinv=False, but also using init_x='interp_p1', so can not use an SBP-H projection.")
                print("         There is no interpolation from p nodes to p+1 nodes anyway. Ignoring use_pinv=False.")
            xap1 = np.einsum('mle,li->mie',affine_map[:,:dim,:], xip1, optimize='optimal') + affine_map[:,dim,np.newaxis,:] # affine p+1 interpolation nodes
            xp1 = np.array(transf(xap1)) # physical p interpolation nodes

        elif init_x == 'interp_p': 
            # start from some degree p interpolation points (with nodes on the boundary), 
            # i.e. assuming the transformation was done with these degree p interpolation nodes
            if skipp:
                print("WARNING: indicated skipp=True, but also init_x='interp_p'. Since the transformation is applied to degree")
                print("         p interpolation points, by default we are degree p, so can not skip p interpolation. Ignoring skipp=True.")
            if not use_pinv:
                print("WARNING: indicated use_pinv=False, but also using init_x='interp_p', so can not use an SBP-H projection.")
                print("         The interpolation from p nodes to p+1 nodes is exact anyway (for p basis). Ignoring use_pinv=False.")
            if dim == 2:
                xip_interp = np.array(StartUpDG_nodes_2d(p)) # p LGL interpolation nodes
                Vp_interp = StartUpDG_vandermonde_2d(xip_interp[0], xip_interp[1], p) # p basis functions evaluated at p interpolation nodes
                Vp_at_p1 = StartUpDG_vandermonde_2d(xip1[0], xip1[1], p) 
            else:
                xip_interp = np.array(StartUpDG_nodes_3d(p)) # p LGL interpolation nodes
                Vp_interp = StartUpDG_vandermonde_3d(xip_interp[0], xip_interp[1], xip_interp[2], p) # p basis functions evaluated at p interpolation nodes
                Vp_at_p1 = StartUpDG_vandermonde_3d(xip1[0], xip1[1], xip1[2], p) 
            xap_interp = np.einsum('mle,li->mie',affine_map[:,:dim,:], xip_interp, optimize='optimal') + affine_map[:,dim,np.newaxis,:] # affine p interpolation nodes
            xp_interp = np.array(transf(xap_interp))
            Ip_interp_to_p1 = Vp_at_p1 @ np.linalg.inv(Vp_interp) # interpolation from p interpolation nodes to (p basis, then evaluation at) p+1 nodes
            xp1 = np.einsum('ij,dje->die',Ip_interp_to_p1,xp_interp,optimize='optimal')

        elif init_x == 'x':
            # start from the physical values corresponding to the xi (sbp) nodes
            # note: if operator does not have sufficient boundary nodes, may lose uniqueness of surface metrics
            if skipp: # DON'T go to a p basis before going up to p+1, just jump straight there (e.g. with projection)
                if (nxi == nmp):
                    print(f"WARNING: indicated skipp=True, but # sbp nodes = {nxi} = # p basis functions , so by default by default we are degree p")
                    print(f"         and so can not skip p interpolation. Also note # p+1 basis functions = {nmp1}.")
                if use_pinv:
                    Ixi_to_xip1 = Vp1 @ np.linalg.pinv(Vp1_at_xi) # approximately interpolate from xi nodes to (p+1 basis, then evaluate at) p+1 interpolation nodes
                else:
                    # quadrature-based projection
                    Mp1 = (Vp1_at_xi.T * H) @ Vp1_at_xi # modal mass matrix for p+1
                    Ixi_to_xip1 = Vp1 @ np.linalg.inv(Mp1) @ Vp1_at_xi.T * H # Interpolate from xi nodes via projection (using 2p H quadrature) to (p+1 basis, then evaluate at) p+1 interpolation nodes
            else: # go to p modal basis first, then go up to p+1
                if (nxi == nmp):
                    print(f"WARNING: since # p+1 sbp nodes = {nxip1} = # p+1 basis functions , use_pinv is irrelevant for Interpolation from xi to p+1.")
                if dim == 2:
                    Vp_at_p1 = StartUpDG_vandermonde_2d(xip1[0], xip1[1], p) #  p basis evaluated at p+1 interpolation nodes
                else:
                    Vp_at_p1 = StartUpDG_vandermonde_3d(xip1[0], xip1[1], xip1[2], p) #  p basis evaluated at p+1 interpolation nodes
                if use_pinv:
                    Ixi_to_xip1 = Vp_at_p1 @ np.linalg.pinv(Vp_at_xi) # Interpolation (not really a projection) from xi nodes to (p basis, to) p+1 interpolation nodes
                else:
                    # quadrature-based projection
                    M = (Vp_at_xi.T * H) @ Vp_at_xi # modal mass matrix for p (takes modes to xi nodes where quadrature is performed)
                    Ixi_to_xip1 = Vp_at_p1 @ np.linalg.inv(M) @ Vp_at_xi.T * H # # Interpolate from xi nodes via projection (using 2p H quadrature) to (p basis, then evaluate at) p+1 interpolation nodes 
            xp1 = np.einsum('ij,dje->die',Ixi_to_xip1,x,optimize='optimal')

        else:
            raise ValueError(f"Invalid chan_init_x option. Try one of: 'x', 'xp1', 'interp_p', 'interp_p1'.")

    elif op_type.lower() in ['diage','omega','gamma']:
        if dim == 2:
            Dx, Dy, _, _, Hp1, _, _, xip1, _, _, _, _ = getOps2D(p+1, 2*p+2, op_type, fac_op_type)
            nxip1 = xip1.shape[1]
            Dp1 = np.zeros((2,nxip1,nxip1))
            Dp1[0,:,:], Dp1[1,:,:], = Dx, Dy
            Vp1_at_xip1 = StartUpDG_vandermonde_2d(xip1[0], xip1[1], p+1) #  p+1 basis evaluated at p+1 xi nodes
        else:
            Dx, Dy, Dz, _, _, _, Hp1, _, _, _, xip1, _, _, _, _ = getOps3D(p+1, 2*p+2, op_type, fac_op_type)
            nxip1 = xip1.shape[1]
            Dp1 = np.zeros((3,nxip1,nxip1))
            Dp1[0,:,:], Dp1[1,:,:], Dp1[2,:,:] = Dx, Dy, Dz
            Vp1_at_xip1 = StartUpDG_vandermonde_3d(xip1[0], xip1[1], xip1[2], p+1) #  p+1 basis evaluated at p+1 xi nodes

        # note: must interpolate / project back to a p+1 basis to then evaluate at p xi nodes, so can't use skipp
        if (nxip1 == nmp1):
            print(f"WARNING: since # p+1 sbp nodes = {nxip1} = # p+1 basis functions, use_pinv is irrelevant for Interpolation from p+1 to p, or Projection from p+1 to p+1.")
        if use_pinv:
            Ixip1_to_xi = Vp1_at_xi @ np.linalg.pinv(Vp1_at_xip1) # approximate interpolation from xi p+1 nodes to (p+1 basis, then evaluation at) xi nodes
            Pp1 = Vp1_at_xip1 @ np.linalg.pinv(Vp1_at_xip1) # projection from xi p+1 nodes to a degree p+1 polynomial space and back to xi p+1 nodes
        else:
            # quadrature-based projection
            Mp1 = (Vp1_at_xip1.T * Hp1) @ Vp1_at_xip1 # modal mass matrix for p+1
            Ixip1_to_xi = Vp1_at_xi @ np.linalg.inv(Mp1) @ Vp1_at_xip1.T * Hp1 # Interpolate from xi p+1 nodes via projection (using 2p+2 H quadrature) to (p+1 basis, then evaluate at) xi p interpolation nodes
            Pp1 = Vp1_at_xip1 @ np.linalg.inv(Mp1) @ Vp1_at_xip1.T * Hp1 # projection from xi p+1 nodes to a degree p+1 polynomial space and back to xi p+1 nodes

        # now calculate x
        if init_x == 'xp1': 
            # start using the most possible accuract xp1, that is the transformation applied to p+1 xi nodes
            if skipp:
                print("WARNING: indicated skipp=False, but also init_x='xp1'. Since the transformation is applied to degree")
                print("         p+1 interpolation points, by default we are degree p+1 and skip p interpolation. Ignoring skipp=False.")
            xap1 = np.einsum('mle,li->mie',affine_map[:,:dim,:], xip1, optimize='optimal') + affine_map[:,dim,np.newaxis,:] # affine p+1 interpolation nodes
            xp1 = np.array(transf(xap1)) # physical p interpolation nodes

        elif init_x == 'interp_p1':
            # start using the most possible accuract xp1, that is the transformation applied to LGL p+1 interpolation nodes
            if skipp:
                print("WARNING: indicated skipp=False, but also init_x='xp1'. Since the transformation is applied to degree")
                print("         p+1 interpolation points, by default we are degree p+1 and skip p interpolation. Ignoring skipp=False.")
            xip1_interp = np.zeros((dim,nmp1))
            if dim == 2:
                xip1_interp = np.array(StartUpDG_nodes_2d(p+1))
                Vp1_interp = StartUpDG_vandermonde_2d(xip1_interp[0], xip1_interp[1], p+1) # p+1 basis functions evaluated at p+1 interpolation nodes
            else:
                xip1_interp = np.array(StartUpDG_nodes_3d(p+1)) # p+1 LGL interpolation nodes
                Vp1_interp = StartUpDG_vandermonde_3d(xip1_interp[0], xip1_interp[1], xip1_interp[2], p+1) # p+1 basis functions evaluated at p+1 interpolation nodes
            xap1_interp = np.einsum('mle,li->mie',affine_map[:,:dim,:], xip1_interp, optimize='optimal') + affine_map[:,dim,np.newaxis,:] # affine p+1 interpolation nodes 
            xp1_interp = np.array(transf(xap1_interp)) # physical p+1 interpolation nodes 
            Ip1_interp_to_xip1 = Vp1_at_xip1 @ np.linalg.inv(Vp1_interp) # interpolation from p+1 interpolation nodes to (p+1 basis, then evaluation at) xi p+1 nodes
            xp1 = np.einsum('ij,dje->die',Ip1_interp_to_xip1,xp1_interp,optimize='optimal')

        elif init_x == 'interp_p': 
            # start from some degree p interpolation points (with nodes on the boundary), 
            # i.e. assuming the transformation was done with these degree p interpolation nodes
            if skipp:
                print("WARNING: indicated skipp=True, but also init_x='interp_p'. Since the transformation is applied to degree")
                print("         p interpolation points, by default we are degree p, so can not skip p interpolation. Ignoring skipp=True.")
            if not use_pinv:
                print("WARNING: indicated use_pinv=False, but also using init_x='interp_p', so can not use an SBP-H projection.")
                print("         The interpolation from p nodes to p+1 nodes is exact anyway (for p basis). Ignoring use_pinv=False.")
            if dim == 2:
                xip_interp = np.array(StartUpDG_nodes_2d(p)) # p LGL interpolation nodes
                Vp_interp = StartUpDG_vandermonde_2d(xip_interp[0], xip_interp[1], p) # p basis functions evaluated at p interpolation nodes
                Vp_at_p1 = StartUpDG_vandermonde_2d(xip1[0], xip1[1], p) 
            else:
                xip_interp = np.array(StartUpDG_nodes_3d(p)) # p LGL interpolation nodes
                Vp_interp = StartUpDG_vandermonde_3d(xip_interp[0], xip_interp[1], xip_interp[2], p) # p basis functions evaluated at p interpolation nodes
                Vp_at_p1 = StartUpDG_vandermonde_3d(xip1[0], xip1[1], xip1[2], p) 
            xap_interp = np.einsum('mle,li->mie',affine_map[:,:dim,:], xip_interp, optimize='optimal') + affine_map[:,dim,np.newaxis,:] # affine p interpolation nodes
            xp_interp = np.array(transf(xap_interp)) # physical p+1 interpolation nodes 
            Ip_interp_to_p1 = Vp_at_p1 @ np.linalg.inv(Vp_interp) # interpolation from p interpolation nodes to (p basis, then evaluation at) p+1 nodes
            xp1 = np.einsum('ij,dje->die',Ip_interp_to_p1,xp_interp,optimize='optimal')

        elif init_x == 'x':
            # start from the physical values corresponding to the xi (sbp) nodes
            # note: if operator does not have sufficient boundary nodes, may lose uniqueness of surface metrics
            if skipp: # DON'T go to a p basis before going up to p+1, just jump straight there (e.g. with projection)
                if (nxi == nmp):
                    print(f"WARNING: indicated skipp=True, but # sbp nodes = {nxi} = # p basis functions , so by default by default we are degree p")
                    print(f"         and so can not skip p interpolation. Also note # p+1 basis functions = {nmp1}.")
                if use_pinv:
                    if (nxi < nmp1):
                        print(f"WARNING: indicated skipp=True, but # sbp nodes = {nxi} < {nmp1} = # p + 1 basis functions,")
                        print("         so interpolation from p to p+1 will introduce sigificant errors.")
                    Ixi_to_xip1 = Vp1_at_xip1 @ np.linalg.pinv(Vp1_at_xi) # approximately interpolate from xi nodes to (p+1 basis, then evaluate at) xi p+1 nodes
                else:
                    # quadrature-based projection
                    print(f"WARNING: since skipp=True and using quadrature projection, H is not accurate enough for projection.")
                    print("          There will be significant errors.")
                    Mp1 = (Vp1_at_xi.T * H) @ Vp1_at_xi # modal mass matrix for p+1 (but using xi SBP)
                    Ixi_to_xip1 = Vp1_at_xip1 @ np.linalg.inv(Mp1) @ Vp1_at_xi.T * H # Interpolate from xi nodes via projection (using 2p H quadrature) to (p+1 basis, then evaluate at) xi p+1 nodes
            else: # go to p modal basis first, then go up to p+1
                if (nxi == nmp):
                    print(f"WARNING: since # p sbp nodes = {nxi} = # p basis functions, use_pinv is irrelevant for Interpolation from p to p+1.")
                if dim == 2:
                    Vp_at_xip1 = StartUpDG_vandermonde_2d(xip1[0], xip1[1], p) #  p basis evaluated at xi p+1 nodes
                else:
                    Vp_at_xip1 = StartUpDG_vandermonde_3d(xip1[0], xip1[1], xip1[2], p) #  p basis evaluated at xi p+1 nodes
                if use_pinv:
                    Ixi_to_xip1 = Vp_at_xip1 @ np.linalg.pinv(Vp_at_xi) # Interpolation (not really a projection) from xi nodes to (p basis, to) xi p+1 nodes
                else:
                    # quadrature-based projection
                    M = (Vp_at_xi.T * H) @ Vp_at_xi # modal mass matrix for p (takes modes to xi nodes where quadrature is performed)
                    Ixi_to_xip1 = Vp_at_xip1 @ np.linalg.inv(M) @ Vp_at_xi.T * H # # Interpolate from xi nodes via projection (using 2p H quadrature) to (p basis, then evaluate at) xi p+1 nodes 
            xp1 = np.einsum('ij,dje->die',Ixi_to_xip1,x,optimize='optimal')

        else:
            raise ValueError(f"Invalid chan_init_x option. Try one of: 'x', 'xp1', 'interp_p', 'interp_p1'.")
    else:
        raise ValueError(f"Invalid chan_op_type option. Try one of: 'Modal', 'Modal_Rand', 'DiagE', 'Omega', 'Gamma'.")

    metrics = np.zeros((dim,*x.shape))

    if dim == 2:
        metrics[0,0] = Ixip1_to_xi @ Dp1[1] @ Pp1 @ xp1[1]
        metrics[0,1] = - Ixip1_to_xi @ Dp1[1] @ Pp1 @ xp1[0]
        metrics[1,0] = - Ixip1_to_xi @ Dp1[0] @ Pp1 @ xp1[1]
        metrics[1,1] = Ixip1_to_xi @ Dp1[0] @ Pp1 @ xp1[0]

    else:
        dx_dxi = np.einsum('lij,mje->mlie', Dp1, xp1, optimize='optimal')
        for l in range(3):
            for m in range(3):
                n = (m+1)%3
                k = (m+2)%3
                s = (l+1)%3
                t = (l+2)%3
                if symmetric: # vinokur and yee variant
                    metrics[l,m] = 0.5*(np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[t],Pp1,xp1[k],dx_dxi[n,s],optimize='optimal') \
                                    - np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[s],Pp1,xp1[k],dx_dxi[n,t],optimize='optimal') \
                                    - np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[t],Pp1,xp1[n],dx_dxi[k,s],optimize='optimal') \
                                    + np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[s],Pp1,xp1[n],dx_dxi[k,t],optimize='optimal') )
                else:
                    metrics[l,m] = np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[t],Pp1,xp1[k],dx_dxi[n,s],optimize='optimal') \
                                - np.einsum('mi,ij,jk,ke,ke->me',Ixip1_to_xi,Dp1[s],Pp1,xp1[k],dx_dxi[n,t],optimize='optimal') 

    return metrics



##############################################################
# other helpful functions
##############################################################

def calc_vol_invariants(D,H,R,B,N,met,metf):
    ''' get the volume invariants '''
    # D: (dim, node, node)
    # met: (dim, dim, node, (facet), element)

    vol = np.einsum('lij,lmje->mie',D,met,optimize='greedy')
    extrap = np.einsum('ijf,lmje->lmife',R,met,optimize='optimal') - metf
    surf = np.einsum('i,jif,j,lf,lmjfe->mie',1/H,R,B,N,extrap,optimize='greedy')
    inv = vol - surf
    return inv, vol, extrap


def calc_convergence(h,err,print_conv=True,title=None,hname=None,n_points=None):
    assert h.shape==err.shape,"The two inputted arrays are not the same shape!"
    assert len(h)>1,"Not enough grids to perform convergence."

    logx = np.log(h)
    logy = np.log(err)

    # Calculate the convergence between every two sets of points
    conv = np.zeros_like(h)
    conv[1:] = (logy[1:] - logy[:-1]) / (logx[1:] - logx[:-1])

    # only use the first n_points for the fit
    if (n_points == None) or n_points > len(h):
        n_points = len(h)

    # Calculate the least squares solution
    logx_plus = np.vstack([logx[:n_points], np.ones(n_points)]).T
    avg_conv, _ = np.linalg.lstsq(logx_plus, logy[:n_points], rcond=None)[0]

    if print_conv:
        print('')
        if hname is None:
            hname = 'h'
        print(f'{title} convergence Lstsq fit = {avg_conv:.2f}')
        data = np.array([h,err,conv]).T
        print(tabulate((data), headers=[hname, 'Error','Convergence'], tablefmt='orgtbl'))

    return conv, avg_conv
    
def plot_conv(h,err,title,xlabel='h',ylabel=None):
    plt.figure()
    m,c = np.polyfit(np.log10(h), np.log10(err), 1)
    yfit = 10**(m*np.log10(h) + c)
    plt.loglog(h,yfit,color='k')
    plt.loglog(h,err,'bo')
    plt.title(title + f' (slope = {m:.2})',fontsize=16)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=14)
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=14)
    plt.show()

