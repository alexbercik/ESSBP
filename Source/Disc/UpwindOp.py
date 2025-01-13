import numpy as np
from julia.api import Julia
from julia import Main
from socket import gethostname
from Source.Disc.MakeSbpOp import MakeSbpOp
import os

hostname = gethostname()
if 'nia' in hostname and 'scinet' in hostname:
     jl = Julia(runtime="/scinet/niagara/software/2022a/opt/base/julia/1.10.4/bin/julia")

# Include the Julia file
folder_path, _ = os.path.split(__file__)
#Main.eval(f'include("{folder_path}/MattssonUpwind.jl")')
Main.include(folder_path + "/MattssonUpwind.jl")

# Access the function from the module
getOps = Main.GetUpwindOperators.getOps

def UpwindOp(p,nn):
    print("NOTE: p no longer defines the order of the operator, but rather the order of the interior stencil.")
    if (p==2 or p==3)  and nn<5:
        raise ValueError('nn set too small ({0}). Needs a minimum of 5.'.format(nn))
    elif (p==4 or p==5) and nn<9:
        raise ValueError('nn set too small ({0}). Needs a minimum of 9.'.format(nn))
    elif (p==6 or p==7) and nn<13:
        raise ValueError('nn set too small ({0}). Needs a minimum of 13.'.format(nn))
    elif (p==8 or p==9) and nn<17:
        raise ValueError('nn set too small ({0}). Needs a minimum of 17.'.format(nn))

    if p%2 == 1: # odd
        bacc = int((p-1)/2)
        bnum = int((p+1)/2)
        order = bacc + 1.5
    else:
        bacc = int(p/2)
        bnum = int(p/2)
        order = bacc + 1.5
    print(f"      the boundary accuracy is {bacc} on the first {bnum} nodes, and should expect order {order} convergence.")
    D, Du, Dm, Q, diss, H, x = getOps(p,nn)
    E = np.zeros((nn,nn))
    E[0,0], E[-1,-1] = -1 , 1
    S = Q - E/2
    tL, tR = np.zeros(nn), np.zeros(nn)
    tL[0] , tR[-1] = 1 , 1
    x = np.array(x)

    return D, Du, Dm, Q, H, E, S, tL, tR, x, diss

def UpwindLGL(p, sigma):
    # copied mostly from the repository:
    # https://github.com/trixi-framework/paper-2024-generalized-upwind-sbp/blob/main/code/code.jl

    # Initialize variables
    nodes, weights, U = None, None, None

    # Define nodes, weights, and Vandermonde matrix
    if p == 2:
        # Gauss-Lobatto nodes
        nodes = np.array([-1.0, 0.0, 1.0])
        # Gauss-Lobatto weights
        weights = np.array([1/3, 4/3, 1/3])
        # Vandermonde matrix
        U = (1 / np.sqrt(6)) * np.array([
            [np.sqrt(2), -np.sqrt(3), 1],
            [np.sqrt(2), 0, -2],
            [np.sqrt(2), np.sqrt(3), 1]
        ])
    elif p == 3:
        # Gauss-Lobatto nodes
        nodes = np.array([-1, -1/np.sqrt(5), 1/np.sqrt(5), 1])
        # Gauss-Lobatto weights
        weights = np.array([1/6, 5/6, 5/6, 1/6])
        # Vandermonde matrix
        U = np.array([
            [0.5, -0.6454972243679028, 0.5, -0.2886751345948129],
            [0.5, -0.28867513459481287, -0.5, 0.6454972243679028],
            [0.5, 0.28867513459481287, -0.5, -0.6454972243679028],
            [0.5, 0.6454972243679028, 0.5, 0.2886751345948129]
        ])
    elif p == 4:
        # Gauss-Lobatto nodes
        nodes = np.array([-1, -np.sqrt(3/7), 0, np.sqrt(3/7), 1])
        # Gauss-Lobatto weights
        weights = np.array([1/10, 49/90, 32/45, 49/90, 1/10])
        # Vandermonde matrix
        U = np.array([
            [0.447213595499958, -0.5916079783099616, 0.5, -0.38729833462074176, 0.223606797749979],
            [0.447213595499958, -0.3872983346207417, -0.16666666666666669, 0.5916079783099616, -0.521749194749951],
            [0.447213595499958, 0.0, -0.6666666666666667, 0.0, 0.596284793999944],
            [0.447213595499958, 0.3872983346207417, -0.16666666666666669, -0.5916079783099616, -0.521749194749951],
            [0.447213595499958, 0.5916079783099616, 0.5, 0.38729833462074176, 0.223606797749979]
        ])
    elif p == 5:
        # Gauss-Lobatto nodes
        nodes = np.array([
            -1, -np.sqrt((7+2*np.sqrt(7))/21), -np.sqrt((7-2*np.sqrt(7))/21),
            np.sqrt((7-2*np.sqrt(7))/21), np.sqrt((7+2*np.sqrt(7))/21), 1
        ])
        # Gauss-Lobatto weights
        weights = np.array([
            1/15, (14-np.sqrt(7))/30, (14+np.sqrt(7))/30,
            (14+np.sqrt(7))/30, (14-np.sqrt(7))/30, 1/15
        ])
        # Vandermonde matrix
        U = np.array([
            [0.408248290463863, -0.547722557505166, 0.483045891539648, -0.408248290463863, 0.316227766016838, -0.182574185835055],
            [0.408248290463863, -0.41903805865559, 0.032338332982759, 0.367654222400928, -0.576443896275457, 0.435014342463468],
            [0.408248290463863, -0.156227735687856, -0.515384224522407, 0.445155822251155, 0.260216130258619, -0.526715472069829],
            [0.408248290463863, 0.156227735687856, -0.515384224522407, -0.445155822251155, 0.260216130258619, 0.526715472069829],
            [0.408248290463863, 0.41903805865559, 0.032338332982759, -0.367654222400928, -0.576443896275457, -0.435014342463468],
            [0.408248290463863, 0.547722557505166, 0.483045891539648, 0.408248290463863, 0.316227766016838, 0.182574185835055]
        ])

    # Dissipation matrix
    sigma_vector = np.zeros(p+1)
    sigma_vector[-1] = - sigma
    S = U @ np.diag(sigma_vector) @ U.T

    # Inverse norm matrix
    P_inv = np.diag(1 / weights)

    # Central and upwind SBP operators
    sbp = MakeSbpOp(p=p, sbp_type='lgl', nn=0, basis_type='legendre', print_progress=False)
    # need to 'fix' because my convention is to use [0,1] instead of [-1,1]
    assert(np.max(abs(0.5*weights-sbp.quad.wq))<1e-10), f'Quadrature weights do not match {0.5*weights} != {sbp.quad.wq}'
    assert(np.max(abs(0.5*(nodes+1)-sbp.quad.xq[:,0]))<1e-10), f'Quadrature nodes do not match {0.5*(nodes+1)} != {sbp.quad.xq[:,0]}'

    Dc = 0.5 * sbp.D # convert to [-1,1]
    Dm = sbp.D - P_inv @ S
    Du = sbp.D + P_inv @ S

    # convert back to [0,1]
    Dm = 2 * Dm
    Du = 2 * Du
    diss = - 0.5 * (Du - Dm)

    return sbp.D, Du, Dm, sbp.Q, sbp.H, sbp.E, sbp.S, sbp.tL, sbp.tR, sbp.x, diss