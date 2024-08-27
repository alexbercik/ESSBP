# Add the root folder of ECO to the search path
import os
import sys

test_folder_path, _ = os.path.split(__file__)
root_folder_path, _ = os.path.split(test_folder_path)
sys.path.append(root_folder_path)

# Import the required modules
import numpy as np
import sys

##### Functions to build operators
def tridiag(nn, a, b, c, bc='SAT', k1=-1, k2=0, k3=1): 
    """ Builds a tri-diagonal matrix 
    INPUT: (a,b,c) arrays on three diagonal bands,
            bc is a string that determines what we do to with BC corner values
           (k1,k2,k3) indicate which band to appy them to
    OUTPUT: 2D array (matrix) """ 
    a = np.ones(nn)*a
    b = np.ones(nn)*b
    c = np.ones(nn)*c
    if bc == 'periodic':
        A = (np.diag(a[:-1], k1) + np.diag(b, k2) + np.diag(c[1:], k3) + 
            np.diag(a[-1:], len(c)-1) + np.diag(c[:1], -(len(c)-1)))
            #np.diag(array, which diagonal - 0 is main, 1 is one above, etc)
    else:
        A = np.diag(a[:-1], k1) + np.diag(b, k2) + np.diag(c[1:], k3)
    return A
    
def pentadiag(nn, a, b, c, d, e, bc='SAT', k1=-2, k2=-1, k3=0, k4=1, k5=2): 
    """ Builds a penta-diagonal matrix 
    INPUT:  nn number of nodes (makes matrix size nn x nn) 
            (a,b,c,d,e) arrays on three diagonal bands,
            bc is a string that determines what we do to with BC corner values
           (k1,k2,k3,k4,k5) indicate which band to appy them to
    OUTPUT: 2D array (matrix) """ 
    a = np.ones(nn)*a
    b = np.ones(nn)*b
    c = np.ones(nn)*c
    d = np.ones(nn)*d
    e = np.ones(nn)*e
    if bc == 'periodic':
        A = (np.diag(a[:-2], k1) + np.diag(b[:-1], k2) + np.diag(c, k3) +
            np.diag(d[1:], k4) + np.diag(e[2:], k5) + np.diag(a[-2:], len(c)-2) +
            np.diag(b[-1:], len(c)-1) + np.diag(d[:1], -(len(c)-1)) +
            np.diag(e[:2], -(len(c)-2)))
            #np.diag(array, which diagonal - 0 is main, 1 is one above, etc)
    else:
        A = (np.diag(a[:-2], k1) + np.diag(b[:-1], k2) + np.diag(c, k3) +
            np.diag(d[1:], k4) + np.diag(e[2:], k5))
    return A

def heptadiag(nn, a, b, c, d, e, f, g, bc='SAT', k1=-3, k2=-2, k3=-1, k4=0, 
              k5=1, k6=2, k7=3): 
    """ Builds a hepta-diagonal matrix 
    INPUT: nn number of nodes (makes matrix size nn x nn) 
            (a,b,c,d,e,f,g) arrays on three diagonal bands,
            bc is a string that determines what we do to with BC corner values
            (k1,k2,k3,k4,k5,k6,k7) indicate which band to appy them to
    OUTPUT: 2D array (matrix) """ 
    a = np.ones(nn)*a
    b = np.ones(nn)*b
    c = np.ones(nn)*c
    d = np.ones(nn)*d
    e = np.ones(nn)*e
    f = np.ones(nn)*f
    g = np.ones(nn)*g
    if bc == 'periodic':
        A = (np.diag(a[:-3], k1) + np.diag(b[:-2], k2) + np.diag(c[:-1], k3) +
            np.diag(d, k4) + np.diag(e[1:], k5) + np.diag(f[2:], k6) + 
            np.diag(g[3:], k7) + np.diag(a[-3:], len(d)-3) +
            np.diag(b[-2:], len(d)-2) + np.diag(c[-1:], len(d)-1) +
            np.diag(e[:1], -(len(d)-1)) + np.diag(f[:2], -(len(d)-2)) + 
            np.diag(g[:3], -(len(d)-3)))
            #np.diag(array, which diagonal - 0 is main, 1 is one above, etc)
    else:
        A = (np.diag(a[:-3], k1) + np.diag(b[:-2], k2) + np.diag(c[:-1], k3) +
            np.diag(d, k4) + np.diag(e[1:], k5) + np.diag(f[2:], k6) + 
            np.diag(g[3:], k7))
    return A

def nonadiag(nn, a, b, c, d, e, f, g, h, i, bc='SAT', k1=-4, k2=-3, k3=-2, 
             k4=-1, k5=0, k6=1, k7=2, k8=3, k9=4): 
    """ Builds a nona-diagonal matrix 
    INPUT: nn number of nodes (makes matrix size nn x nn) 
            (a,b,c,d,e,f,g) arrays on three diagonal bands,
            bc is a string that determines what we do to with BC corner values
           (k1,k2,k3,k4,k5,k6,k7) indicate which band to appy them to
    OUTPUT: 2D array (matrix) """ 
    a = np.ones(nn)*a
    b = np.ones(nn)*b
    c = np.ones(nn)*c
    d = np.ones(nn)*d
    e = np.ones(nn)*e
    f = np.ones(nn)*f
    g = np.ones(nn)*g
    h = np.ones(nn)*h
    i = np.ones(nn)*i
    if bc == 'periodic':
        A = (np.diag(a[:-4], k1) + np.diag(b[:-3], k2) + np.diag(c[:-2], k3) +
            np.diag(d[:-1], k4) + np.diag(e, k5) + np.diag(f[1:], k6) + 
            np.diag(g[2:], k7) + np.diag(h[3:], k8) + np.diag(i[4:], k9) +
            np.diag(a[-4:], len(e)-4) + np.diag(b[-3:], len(e)-3) + 
            np.diag(c[-2:], len(e)-2) + np.diag(d[-1:], len(e)-1) + 
            np.diag(f[:1], -(len(e)-1)) + np.diag(g[:2], -(len(e)-2)) + 
            np.diag(h[:3], -(len(e)-3)) + np.diag(i[:4], -(len(e)-4)))
            #np.diag(array, which diagonal - 0 is main, 1 is one above, etc)
    else:
        A = (np.diag(a[:-4], k1) + np.diag(b[:-3], k2) + np.diag(c[:-2], k3) +
            np.diag(d[:-1], k4) + np.diag(e, k5) + np.diag(f[1:], k6) + 
            np.diag(g[2:], k7) + np.diag(h[3:], k8) + np.diag(i[4:], k9))
    return A

        
def CSbpOp(p,nn):
    """ Builds a 1D CSBP first derivative operator on reference element [0,1] 
    for p=1,2,3,4
    INPUT: derivative order, grid size
    OUTPUT: 2D arrays (matrix) H and D """ 
    
    dx = 1/(nn-1)
    E = np.zeros((nn,nn))
    E[0,0] = -1
    E[-1,-1] = 1
    
    if p==1:
        if nn < 3:
            print ( '' )
            print ( 'CSBP_SET - Fatal error!' )
            print ( '  Illegal value of nn: %d' % ( nn ) )
            print ( '  nn must be at least 3 when p=1' )
            sys.exit ( 'CSBP_SET - Fatal error!' )
      
        H = dx*np.diag(np.ones(nn))
        H[0,0] = dx*0.5
        H[-1,-1] = dx*0.5
        
        Q = tridiag(nn, -1/2, 0, 1/2)
        Q[0,0] = -1/2
        Q[-1,-1] = 1/2

    elif p==2:
        if nn < 9:
            print ( '' )
            print ( 'CSBP_SET - Fatal error!' )
            print ( '  Illegal value of nn: %d' % ( nn ) )
            print ( '  nn must be at least 9 when p=2' )
            sys.exit ( 'CSBP_SET - Fatal error!' )
                
        H = dx*np.diag(np.ones(nn))
        H[0,0] = dx*17/48
        H[1,1] = dx*59/48
        H[2,2] = dx*43/48
        H[3,3] = dx*49/48
                    
        Q = pentadiag(nn, 1/12, -2/3, 0, 2/3, -1/12)
        Q[0,0] = -1/2
        Q[0,1] = 59/96
        Q[0,2] = -1/12
        Q[0,3] = -1/32;
        Q[1,0] = - Q[0,1]
        Q[1,1] = 0
        Q[1,2] = 59/96
        Q[1,3] = 0
        Q[2,0] = -Q[0,2]
        Q[2,1] = -Q[1,2]
        Q[2,3] = 59/96
        Q[3,0] = -Q[0,3]
        Q[3,1] = -Q[1,3]
        Q[3,2] = -Q[2,3]
        
        #bottom portion of the matrices
        for i in range(4):
            for j in range(4):
                Q[-1-i,-1-j] = -Q[i,j]
                if i == j:
                    H[-1-i,-1-i] = H[i,j]
  
    elif p==3:
        if nn < 13:
                print ( '' )
                print ( 'CSBP_SET - Fatal error!' )
                print ( '  Illegal value of nn: %d' % ( nn ) )
                print ( '  nn must be at least 13 when p=3' )
                sys.exit ( 'CSBP_SET - Fatal error!' )
                
        H = dx*np.diag(np.ones(nn))
        H[0,0] = dx*13649/43200
        H[1,1] = dx*12013/8640
        H[2,2] = dx*2711/4320
        H[3,3] = dx*5359/4320
        H[4,4] = dx*7877/8640
        H[5,5] = dx*43801/43200

        Q = heptadiag(nn,-1/60,3/20,-3/4,0,3/4,-3/20,1/60)
        q = 5591070156686698065364559/7931626489314500743872000
        Q[0,0] = -1/2
        Q[0,1] = -0.953E3 / 0.16200E5 + q
        Q[0,2] = 0.715489E6 / 0.259200E6 - (4 * q)
        Q[0,3] = -0.62639E5 / 0.14400E5 + (6* q)
        Q[0,4] = 0.147127E6 / 0.51840E5 - (4* q)
        Q[0,5] = -0.89387E5 / 0.129600E6 + q
        Q[1,0] = - Q[0,1]
        Q[1,2] = -0.57139E5 / 0.8640E4 + (10* q)
        Q[1,3] = 0.745733E6 / 0.51840E5 - (20 * q)
        Q[1,4] = -0.18343E5 / 0.1728E4 + (15 * q)
        Q[1,5] = 0.240569E6 / 0.86400E5 - (4 * q)
        Q[2,0] = -Q[0,2]
        Q[2,1] = -Q[1,2]
        Q[2,3] = -0.176839E6 / 0.12960E5 + (20 * q)
        Q[2,4] = 0.242111E6 / 0.17280E5 - (20 * q)
        Q[2,5] = -0.182261E6 / 0.43200E5 + (6 * q)
        Q[3,0] = -Q[0,3]
        Q[3,1] = -Q[1,3]
        Q[3,2] = -Q[2,3]
        Q[3,4] = -0.165041E6 / 0.25920E5 + (10 * q)
        Q[3,5] = 0.710473E6 / 0.259200E6 - (4 * q)
        Q[3,6] = 1/60
        Q[4,0] = -Q[0,4]
        Q[4,1] = -Q[1,4]
        Q[4,2] = -Q[2,4]
        Q[4,3] = -Q[3,4]
        Q[4,5] = q
        Q[4,6] = -3/20
        Q[4,7] = 1/60
        Q[5,0] = -Q[0,5]
        Q[5,1] = -Q[1,5]
        Q[5,2] = -Q[2,5]
        Q[5,3] = -Q[3,5]
        Q[5,4] = -Q[4,5]
        Q[5,6] = 3/4
        Q[5,7] = -3/20
        Q[5,8] = 1/60
        
        #bottom portion of the matrices
        for i in range(6):
            for j in range(6):
                Q[-1-i,-1-j] = -Q[i,j]
                if i == j:
                    H[-1-i,-1-i] = H[i,j]
        
    elif p==4:
        if nn < 17:
            print ( '' )
            print ( 'CSBP_SET - Fatal error!' )
            print ( '  Illegal value of nn: %d' % ( nn ) )
            print ( '  nn must be at least 17 when p=4' )
            sys.exit ( 'CSBP_SET - Fatal error!' )
                
        H = dx*np.diag(np.ones(nn))
        H[0,0] = dx*0.1498139E7 / 0.5080320E7
        H[1,1] = dx*0.1107307E7 / 0.725760E6
        H[2,2] = dx*0.20761E5 / 0.80640E5
        H[3,3] = dx*0.1304999E7 / 0.725760E6
        H[4,4] = dx*0.299527E6 / 0.725760E6
        H[5,5] = dx*0.103097E6 / 0.80640E5
        H[6,6] = dx*0.670091E6 / 0.725760E6
        H[7,7] = dx*0.5127739E7 / 0.5080320E7

        Q = nonadiag(nn,1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280)
        q16 = 0.08314829949122060462305047907908720666335
        q17 = -0.9521334029619388274601963790830716099197E-2
        q47 = -0.3510216710115618609017136924794334791187E-1
        Q[0,0] = -1/2
        Q[0,1] =  0.59065123E8/0.91445760E8 + q16/0.3E1 + 0.2E1/0.3E1*q17
        Q[0,2] = 0.771343E6 / 0.10160640E8 - 0.8E1 / 0.5E1 * q16 - 0.3E1 *q17
        Q[0,3] = -0.8276887E7 / 0.20321280E8 + (3 * q16) + (5 * q17)
        Q[0,4] = 0.17658817E8 / 0.91445760E8 - 0.8E1 / 0.3E1 * q16 - 0.10E2/ 0.3E1 * q17
        Q[0,5] = q16
        Q[0,6] = q17
        Q[0,7] = -0.1394311E7 / 0.182891520E9 - q16 / 0.15E2 - q17 / 0.3E1
        Q[1,0] = - Q[0,1]
        Q[1,2] = q47 / 0.45E2 + 0.14E2 / 0.3E1 * q16 + 0.77E2 / 0.9E1 * q17- 0.14866699E8 /0.130636800E9
        Q[1,3] = 0.18734719E8 / 0.13063680E8 - 0.35E2 / 0.3E1 * q16 - 0.175E3 / 0.9E1 * q17 -q47 / 0.9E1
        Q[1,4] = -0.2642179E7 / 0.3265920E7 + 0.35E2 / 0.3E1 * q16 + 0.140E3 / 0.9E1 * q17 +0.2E1 / 0.9E1 * q47
        Q[1,5] = 0.1736509E7 / 0.13063680E8 - 0.14E2 / 0.3E1 * q16 - 0.14E2 / 0.9E1 * q17 -0.2E1 / 0.9E1 * q47
        Q[1,6] = -0.13219E5 / 0.1244160E7 - 0.35E2 / 0.9E1 * q17 + q47 / 0.9E1
        Q[1,7] = 0.1407281E7 / 0.11430720E9 + q16 / 0.3E1 + 0.13E2 / 0.9E1 * q17 - q47 /0.45E2
        Q[2,0] = -Q[0,2]
        Q[2,1] = -Q[1,2]
        Q[2,3] = -0.3056891E7 / 0.4354560E7 + (14E0 * q16) + 0.7E2 / 0.3E1 * q17 + q47 /0.3E1
        Q[2,4] = 0.765701E6 / 0.653184E6 - 0.56E2 / 0.3E1 * q16 - 0.245E3 / 0.9E1 * q17 -0.8E1 / 0.9E1 * q47
        Q[2,5] = -0.238939E6 / 0.414720E6 + 0.42E2 / 0.5E1 * q16 + 0.7E1 * q17 + q47
        Q[2,6] = 0.754291E6 / 0.21772800E8 + 0.14E2 / 0.3E1 * q17 - 0.8E1 / 0.15E2 * q47
        Q[2,7] = 0.762499E6 / 0.22861440E8 - 0.2E1 / 0.3E1 * q16 - 0.20E2 / 0.9E1 * q17 + q47/ 0.9E1
        Q[3,0] = -Q[0,3]
        Q[3,1] = -Q[1,3]
        Q[3,2] = -Q[2,3]
        Q[3,4] = -0.10064459E8 / 0.26127360E8 + 0.35E2 / 0.3E1 * q16 + 0.175E3 / 0.9E1 * q17+ 0.10E2 / 0.9E1 * q47
        Q[3,5] = 0.62249E5 / 0.77760E5 - (7E0 * q16) - 0.35E2 / 0.3E1 * q17 - 0.5E1 / 0.3E1* q47
        Q[3,6] = q47
        Q[3,7] = -0.8276887E7 / 0.91445760E8 + 0.2E1 / 0.3E1 * q16 + 0.10E2 / 0.9E1 * q17 -0.2E1 / 0.9E1 * q47
        Q[4,0] = -Q[0,4]
        Q[4,1] = -Q[1,4]
        Q[4,2] = -Q[2,4]
        Q[4,3] = -Q[3,4]
        Q[4,5] = 0.792095E6 / 0.2612736E7 + 0.7E1 / 0.3E1 * q16 + 0.70E2 / 0.9E1 * q17 +0.10E2 / 0.9E1 * q47
        Q[4,6] = -0.42403E5 / 0.207360E6 - 0.35E2 / 0.9E1 * q17 - 0.8E1 / 0.9E1 * q47
        Q[4,7] = 0.13906657E8 / 0.182891520E9 - q16 / 0.3E1 + 0.5E1 / 0.9E1 * q17 + 0.2E1 /0.9E1 * q47
        Q[5,0] = -Q[0,5]
        Q[5,1] = -Q[1,5]
        Q[5,2] = -Q[2,5]
        Q[5,3] = -Q[3,5]
        Q[5,4] = -Q[4,5]
        Q[5,6] = 0.1360207E7 / 0.1741824E7 + 0.7E1 / 0.3E1 * q17 + q47 / 0.3E1
        Q[5,7] = -0.289189E6 / 0.1866240E7 + q16 / 0.15E2 - 0.7E1 / 0.9E1 * q17 - q47 / 0.9E1
        Q[6,0] = -Q[0,6]
        Q[6,1] = -Q[1,6]
        Q[6,2] = -Q[2,6]
        Q[6,3] = -Q[3,6]
        Q[6,4] = -Q[4,6]
        Q[6,5] = -Q[5,6]
        Q[6,7] = 0.16676111E8 / 0.21772800E8 + 0.2E1 / 0.9E1 * q17 + q47 / 0.45E2
        Q[7,0] = -Q[0,7]
        Q[7,1] = -Q[1,7]
        Q[7,2] = -Q[2,7]
        Q[7,3] = -Q[3,7]
        Q[7,4] = -Q[4,7]
        Q[7,5] = -Q[5,7]
        Q[7,6] = -Q[6,7]
        
        #bottom portion of the matrices
        for i in range(8):
            for j in range(8):
                Q[-1-i,-1-j] = -Q[i,j]
                if i == j:
                    H[-1-i,-1-i] = H[i,j]
        
    else:
        print("ERROR: You have not coded this order p yet")
        sys.exit()
     
    D = np.linalg.inv(H)@Q
    S = Q - E/2
    
    return H, D, Q, S, dx
        
        
        
        
        
        
        
        