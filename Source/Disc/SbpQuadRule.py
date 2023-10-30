# Add the root folder of ECO to the search path
import os
import sys

test_folder_path, _ = os.path.split(__file__)
root_folder_path, _ = os.path.split(test_folder_path)
sys.path.append(root_folder_path)

# Import the required modules
import numpy as np
from sys import platform

#if platform == "linux" or platform == "linux2": # True if on SciNet
#    print('Need to install quadpy on SciNet before using this function')
#else:
#    import quadpy as qp

from Source.Disc.BasisFun import BasisFun
from Source.Disc.Quadratures.LGL import LGL_set
from Source.Disc.Quadratures.LG import LG_set
from Source.Disc.Quadratures.NC import NC_set

class SbpQuadRule:
    
    dim = 1 # We only use Tensor-product in this code, although functionality exists for simplices

    def __init__(self, p, sbp_fam='Rd',nn=0,quad_rule=None):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        p : int
            Degree of the SBP operator.
        sbp_fam : string, optional
            The type of SBP family for the operator.
            The families are Rd (Omega), Rdn1 (Gamma) or R0 (diagE).
            The default is 'Rd'.
        nn : int, optional
            The number of nodes to use
            ** Note: So far only affects anything when dim=1
            The default is 0, in which case nn is set automatically
        quad_rule : string, optional
            The specific quadrature rule to use within each family
            ** Note: Only actually affects anything when dim=1
            Options are None (uses default 'lg' or 'lgl' depending on sbp_fam),
                        'lg' (lobatto-gauss), 'lgl' (legendre-gauss-lobatto),
                        'nc' (closed-form newton cotes).
            The default is None.

        Returns
        -------
        self.dim : No. of dimensions.
        self.p : Degree of the SBP operator.
        self.pquad : Degree of the volume quadrature rule.
        self.xq : Cartesian coordinates of the volume quadrature nodes.
        self.wq : Volume quadrature weights.
        self.nn : Number of volume quadrature nodes.
        self.pfquad : Degree of the facet quadrature rule.
        self.xqf : Cartesian coordinates of the facet quadrature nodes.
        self.wqf : Facet quadrature weights.
        self.nnf : Number of facet quadrature nodes.
        self.sbp_fam : The type of SBP family for the operator.
        self.quad_rule : The specific quadrature used.
        self.nmin : Minimum number of nodes needed (cardinality of basis)
        '''

        ''' Add inputs to the class '''
        self.p = p
        self.sbp_fam = sbp_fam
        self.nn = nn
        self.quad_rule = quad_rule

        ''' Extract required parameters '''
        self.nmin = BasisFun.cardinality(self.dim, self.p) # Min no. of nodes

        ''' Get all required data for the quad rules '''

        # Calculate min quad degree for elem and facet quad rules
        pquad_min = 2*self.p-1
        pfquad_min = 2*self.p

        # Rd and Rdn1 classes use LG or LG like quad rules for the facets
        if sbp_fam == 'Rd':
            if self.quad_rule == None: self.quad_rule = 'lg'
            if self.dim == 1:
                if self.quad_rule.lower() == 'lg':
                    self.xq, self.wq, self.pquad = self.quad_rule_1D(pquad_min, 'lg', self.nmin, self.nn)
                    self.xqf, self.wqf, self.pfquad = self.quad_rule_0D()
                else: raise Exception('Invalid choice for quad_rule in Rd family')
            elif self.dim == 2:
                self.xq, self.wq,self.pquad = self.quad_rule_2d_Rd(self.p)
                self.xqf, self.wqf, self.pfquad = self.quad_rule_1D(pfquad_min, 'lg')
            elif self.dim == 3:
                self.xq, self.wq,self.pquad = self.quad_rule_3d_Rd(self.p)
                self.xqf, self.wqf, self.pfquad = self.quad_rule_2D_lg(pfquad_min)
            else:
                raise Exception('Only 1 and 2D operators available for Rd')

        elif sbp_fam == 'Rdn1':
            if self.quad_rule == None: self.quad_rule = 'lgl'
            if self.dim == 1:
                print('** Rdn1 and Rd0 operators are identical in 1D **')
                if self.quad_rule.lower() == 'lgl':
                    self.xq, self.wq, self.pquad = self.quad_rule_1D(pquad_min, 'lgl', self.nmin, self.nn)
                    self.xqf, self.wqf, self.pfquad = self.quad_rule_0D()
                else: raise Exception('Invalid choice for quad_rule in Rdn1 family')
            elif self.dim == 2:
                self.xq, self.wq, self.pquad = self.quad_rule_2d_Rdn1(self.p)
                self.xqf, self.wqf, self.pfquad = self.quad_rule_1D(pfquad_min, 'lg')
            elif self.dim == 3:
                self.xq, self.wq, self.pquad = self.quad_rule_3d_Rdn1(self.p)
                self.xqf, self.wqf, self.pfquad = self.quad_rule_2D_lg(pfquad_min)
            else:
                raise Exception('Only 1, 2 and 3D operators available for Rd')

        elif sbp_fam == 'R0':
            if self.quad_rule == None: self.quad_rule = 'lgl'
            if self.dim == 1:
                if self.quad_rule.lower() == 'lgl':
                    self.xq, self.wq, self.pquad = self.quad_rule_1D(pquad_min, 'lgl', self.nmin, self.nn)
                    self.xqf, self.wqf, self.pfquad = self.quad_rule_0D()
                elif self.quad_rule.lower() == 'nc':
                    self.xq, self.wq, self.pquad = self.quad_rule_1D(pquad_min, 'nc', self.nmin, self.nn)
                    self.xqf, self.wqf, self.pfquad = self.quad_rule_0D()
                else: raise Exception('Invalid choice for quad_rule in R0 family')
            elif self.dim == 2:
                self.xq, self.wq, self.pquad, self.xqf, self.wqf, self.pquadf = self.quad_rule_2d_R0(self.p)
            elif self.dim == 3:
                self.xq, self.wq, self.pquad, self.xqf, self.wqf, self.pquadf = self.quad_rule_3d_R0(self.p)
            else:
                raise Exception('Only 1, 2 and 3D operators available for R0')

        else:
            raise Exception('Unknown SBP family group')

        # Common data
        self.nn = self.wq.size
        self.nnf = self.wqf.size

    @staticmethod
    def quad_lg(dim, pquad_req):
        '''
        Parameters
        ----------
        dim : int
            No. of dimensions.
        pquad_req : int
            Quad rule must be of this degree or greater.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Actual degree of the quad rule.
        '''

        if dim == 1:
            xq, wq, pquad = SbpQuadRule.quad_rule_1D(pquad_req, 'lg')
        elif dim == 2:
            xq, wq, pquad = SbpQuadRule.quad_rule_2D_lg(pquad_req)
        elif dim == 3:
            xq, wq, pquad = SbpQuadRule.quad_rule_3D_lg(pquad_req)
        else:
            raise Exception('Only available for 1 and 2D')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_0D():
        '''
        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        xq = np.zeros((1,1))
        wq = np.ones(1)
        pquad = np.inf

        return xq, wq, pquad

    @staticmethod
    def quad_rule_1D(pquad, quad_name, nmin=0, nni=0):
        '''
        Parameters
        ----------
        pquad : int
            Min required degree for the quadrature rule.
        quad_name : string
            Indicates the type of 1D quad rule, either LG or LGL.
        nmin : int, optional
            Min number of nodes in the quad rule. The default is 0.
        nni : int, optional
            number of nodes in the quad rule. The default is None.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : numpy array
            Degree of the quadrature rule.
        '''
        # NOTE: Slightly more accurate to use set values than quadpy (commented out)

        if quad_name.lower() == 'lg':
            n_calc_p = int(np.ceil(0.5*(pquad+1)))
            nn = np.max((nmin, n_calc_p, nni))
            #qp_class = qp.c1.gauss_legendre(nn)
            xq , wq, pquad = LG_set(nn)
        elif quad_name.lower() == 'lgl':
            n_calc_p = int(np.ceil(0.5*(pquad+3)))
            nn = np.max((nmin, n_calc_p, nni))
            #qp_class = qp.c1.gauss_lobatto(nn)
            xq , wq, pquad = LGL_set(nn)
        elif quad_name.lower() == 'nc':
            n_calc_p = int(2*np.floor(pquad/2)+1)
            nn = np.max((nmin, n_calc_p, nni))
            #qp_class = qp.c1.newton_cotes_closed(nn-1)
            # note that quadpy numbers newton cotes differently, so use nn-1
            xq , wq, pquad = NC_set(nn)
        else:
            raise Exception('Requested 1D quad is not available')
            
        if nni > max(n_calc_p, nmin):
            print('WARNING: Using more nodes than required for degree order.')

        # Convert from domain [-1,1] to [0,1]
        #xq = 0.5*(qp_class.points[:, None] + 1) # Convert from 1D to 2D array
        #wq = 0.5 * qp_class.weights
        #pquad = qp_class.degree
        xq = 0.5*(xq[:, None] + 1) # Convert from 1D to 2D array
        wq = 0.5 * wq

        if np.any(wq<0):
            xq, wq, pquad = None, None, None
            raise Exception('Quadrature has negative weights - not SPD!')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_2D_lg(pquad_req):
        '''
        Parameters
        ----------
        pquad_req : int
            Requsetd degree for the quadrature rule.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Actual degree of the quad rule
        '''

        if pquad_req <= 2:
            pquad = 2
            dat = np.array([[0.166666666666667, 0.166666666666667, 0.166666666666667],
                            [0.666666666666667, 0.166666666666667, 0.166666666666667],
                            [0.166666666666667, 0.666666666666667, 0.166666666666667]])
        if pquad_req <= 4:
            pquad = 4
            dat = np.array([[0.091576213509771, 0.091576213509771, 0.054975871827661],
                            [0.816847572980459, 0.091576213509771, 0.054975871827661],
                            [0.091576213509771, 0.816847572980459, 0.054975871827661],
                            [0.445948490915965, 0.445948490915965, 0.111690794839006],
                            [0.108103018168070, 0.445948490915965, 0.111690794839006],
                            [0.445948490915965, 0.108103018168070, 0.111690794839006]])


        elif pquad_req <= 6:
            pquad = 6
            dat = np.array([[0.249286745170910, 0.249286745170910, 0.058393137863190],
                            [0.501426509658179, 0.249286745170910, 0.058393137863190],
                            [0.249286745170910, 0.501426509658179, 0.058393137863190],
                            [0.063089014491502, 0.063089014491502, 0.025422453185103],
                            [0.873821971016995, 0.063089014491502, 0.025422453185103],
                            [0.063089014491502, 0.873821971016995, 0.025422453185103],
                            [0.310352451033785, 0.053145049844817, 0.041425537809187],
                            [0.053145049844817, 0.310352451033785, 0.041425537809187],
                            [0.636502499121399, 0.053145049844817, 0.041425537809187],
                            [0.053145049844817, 0.636502499121399, 0.041425537809187],
                            [0.636502499121399, 0.310352451033785, 0.041425537809187],
                            [0.310352451033785, 0.636502499121399, 0.041425537809187]])

        elif pquad_req <= 8:
            pquad = 8
            dat = np.array([[0.333333333333333, 0.333333333333333, 0.072157803838894],
                            [0.459292588292723, 0.459292588292723, 0.047545817133642],
                            [0.081414823414554, 0.459292588292723, 0.047545817133642],
                            [0.459292588292723, 0.081414823414554, 0.047545817133642],
                            [0.170569307751760, 0.170569307751760, 0.051608685267359],
                            [0.658861384496479, 0.170569307751760, 0.051608685267359],
                            [0.170569307751760, 0.658861384496479, 0.051608685267359],
                            [0.050547228317031, 0.050547228317031, 0.016229248811599],
                            [0.898905543365938, 0.050547228317031, 0.016229248811599],
                            [0.050547228317031, 0.898905543365938, 0.016229248811599],
                            [0.263112829634638, 0.728492392955404, 0.013615157087217],
                            [0.728492392955404, 0.263112829634638, 0.013615157087217],
                            [0.008394777409957, 0.728492392955404, 0.013615157087217],
                            [0.728492392955404, 0.008394777409957, 0.013615157087217],
                            [0.008394777409957, 0.263112829634638, 0.013615157087217],
                            [0.263112829634638, 0.008394777409957, 0.013615157087217]])

        elif pquad_req <= 10:
            pquad = 10
            dat = np.array([[0.333333333333333, 0.333333333333333, 0.041761699902598],
                            [0.497865432954475, 0.497865432954475, 0.003614925296028],
                            [0.004269134091051, 0.497865432954475, 0.003614925296028],
                            [0.497865432954475, 0.004269134091051, 0.003614925296028],
                            [0.428012449729056, 0.428012449729056, 0.037246088960490],
                            [0.143975100541888, 0.428012449729056, 0.037246088960490],
                            [0.428012449729056, 0.143975100541888, 0.037246088960490],
                            [0.184756412743225, 0.184756412743225, 0.039323236701554],
                            [0.630487174513551, 0.184756412743225, 0.039323236701554],
                            [0.184756412743225, 0.630487174513551, 0.039323236701554],
                            [0.020481218571678, 0.020481218571678, 0.003464161543554],
                            [0.959037562856645, 0.020481218571678, 0.003464161543554],
                            [0.020481218571678, 0.959037562856645, 0.003464161543554],
                            [0.136573576256034, 0.828423433846694, 0.014759160167390],
                            [0.828423433846694, 0.136573576256034, 0.014759160167390],
                            [0.035002989897272, 0.828423433846694, 0.014759160167390],
                            [0.828423433846694, 0.035002989897272, 0.014759160167390],
                            [0.035002989897272, 0.136573576256034, 0.014759160167390],
                            [0.136573576256034, 0.035002989897272, 0.014759160167390],
                            [0.332743600588638, 0.629707329152918, 0.019789683598030],
                            [0.629707329152918, 0.332743600588638, 0.019789683598030],
                            [0.037549070258443, 0.629707329152918, 0.019789683598030],
                            [0.629707329152918, 0.037549070258443, 0.019789683598030],
                            [0.037549070258443, 0.332743600588638, 0.019789683598030],
                            [0.332743600588638, 0.037549070258443, 0.019789683598030]])
        else:
            raise Exception('The highest degree quad rule for the triangle is 10')

        xq = dat[:, :2]
        wq = dat[:,2]

        return xq, wq, pquad

    @staticmethod
    def quad_rule_2d_Rd(p):
        '''
        (uses 2D LG quadrature)
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        if p == 1:
            pquad = 2
            xq = np.array([[0.166666666666667, 0.166666666666667],
                           [0.666666666666667, 0.166666666666667],
                           [0.166666666666667, 0.666666666666667]])
            wq = np.array([0.166666666666667, 0.166666666666667, 0.166666666666667])
        elif p == 2:
            pquad = 4
            xq = np.array([[0.091576213509771, 0.091576213509771],
                           [0.816847572980459, 0.091576213509771],
                           [0.091576213509771, 0.816847572980459],
                           [0.445948490915965, 0.108103018168070],
                           [0.445948490915965, 0.445948490915965],
                           [0.108103018168070, 0.445948490915965]])

            wq = np.array([0.054975871827661, 0.054975871827661, 0.054975871827661,
                           0.111690794839006, 0.111690794839006, 0.111690794839006])


        elif p == 3:
            pquad = 5
            xq = np.array([[0.069311653138313, 0.069311653138313],
                           [0.861376693723373, 0.069311653138313],
                           [0.069311653138313, 0.861376693723373],
                           [0.311322129281642, 0.071079720277502],
                           [0.617598150440857, 0.071079720277502],
                           [0.617598150440857, 0.311322129281642],
                           [0.311322129281642, 0.617598150440857],
                           [0.071079720277502, 0.617598150440857],
                           [0.071079720277502, 0.311322129281642],
                           [0.333333333333333, 0.333333333333333]])

            wq = np.array([0.028876181685753, 0.028876181685753, 0.028876181685753, 0.052311201740830, 0.052311201740830, 0.052311201740830, 0.052311201740830, 0.052311201740830, 0.052311201740830, 0.099504244497763])
        elif p == 4:
            pquad = 7
            xq = np.array([[0.042165614409432, 0.042165614409432],
                           [0.915668771181136, 0.042165614409432],
                           [0.042165614409432, 0.915668771181136],
                           [0.211562058638092, 0.047981341371465],
                           [0.474294689117510, 0.051410621764979],
                           [0.740456599990443, 0.047981341371465],
                           [0.740456599990443, 0.211562058638092],
                           [0.474294689117510, 0.474294689117510],
                           [0.211562058638092, 0.740456599990443],
                           [0.047981341371465, 0.740456599990443],
                           [0.051410621764979, 0.474294689117510],
                           [0.047981341371465, 0.211562058638092],
                           [0.242085973759479, 0.242085973759479],
                           [0.515828052481043, 0.242085973759479],
                           [0.242085973759479, 0.515828052481043]])

            wq = np.array([0.011346539476309, 0.011346539476309, 0.011346539476309, 0.027639396736562, 0.036457103737727, 0.027639396736562, 0.027639396736562, 0.036457103737727, 0.027639396736562, 0.027639396736562, 0.036457103737727, 0.027639396736562, 0.063584229979506, 0.063584229979506, 0.063584229979506])

        else:
            raise Exception('The requested degree for the 2D Rd quad is not available')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_2d_Rdn1(p):
        '''
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        if p == 1:
            pquad = 2
            xq = np.array([[0.3333333333333333, 0, 1, 0],
                           [0.3333333333333333, 0, 0, 1]]).T

            wq = np.array([0.375, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666])
        elif p == 2:
            pquad = 5
            xq = np.array([[0.2073451756635909, 0.5853096486728182, 0.2073451756635909, 0.874741552012739, 0, 0.1252584479872611, 0.1252584479872611, 0, 0.874741552012739, 0.5, 0, 0.5],
           [0.2073451756635909, 0.2073451756635909, 0.5853096486728182, 0.1252584479872611, 0.874741552012739, 0, 0.874741552012739, 0.1252584479872611, 0, 0.5, 0.5, 0]]).T

            wq = np.array([0.1103885289202054, 0.1103885289202054, 0.1103885289202054, 0.01403693266390529, 0.01403693266390529, 0.01403693266390529, 0.01403693266390529, 0.01403693266390529, 0.01403693266390529, 0.02820427241865071, 0.02820427241865071, 0.02820427241865071])

        else:
            raise Exception('The requested degree for the 2D Rdn1 quad is not available')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_2d_R0(p):
        '''
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the element quadrature nodes.
        wq : numpy array
            Element quatrature weights.
        pquad : int
            Degree of the element quadrature rule.
        xqf : numpy array
            Cartesian coordinates of the facet quadrature nodes.
        wqf : numpy array
            Facet quatrature weights.
        pquadf : int
            Degree of the facet quadrature rule.
        '''

        if p == 1:
            # Element quad rule
            pquad = 3
            xq = np.array([[0.3333333333333333, 0, 1, 0, 0.5, 0, 0.5],
           [0.3333333333333333, 0, 0, 1, 0.5, 0.5, 0]]).T

            wq = np.array([0.225, 0.025, 0.025, 0.025, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667])

            # Facet quad rule
            pquadf = 3
            xqf = np.array([[0, 0.5, 1]]).T
            wqf = np.array([0.1666666666666667, 0.6666666666666666, 0.1666666666666667])

        elif p == 2:
            # Element quad rule
            pquad = 4
            xq = np.array([[0.2128543571118084, 0.5742912857763832, 0.2128543571118084, 0, 1, 0, 0.7236067977499789, 0, 0.276393202250021, 0.276393202250021, 0, 0.7236067977499789],
                           [0.2128543571118084, 0.2128543571118084, 0.5742912857763832, 0, 0, 1, 0.276393202250021, 0.7236067977499789, 0, 0.7236067977499789, 0.276393202250021, 0]]).T

            wq = np.array([[0.1067579396609884, 0.1067579396609884, 0.1067579396609884, 0.006261126504899713, 0.006261126504899713, 0.006261126504899713, 0.02682380025038929, 0.02682380025038929, 0.02682380025038929, 0.02682380025038929, 0.02682380025038929, 0.02682380025038929]])

            # Facet quad rule
            pquadf = 5
            xqf = np.array([[0, 0.2763932022500211, 0.7236067977499789, 1]]).T
            wqf = np.array([0.08333333333333333, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333])

        else:
            raise Exception('The requested degree for the 2D R0 quad is not available')

        return xq, wq, pquad, xqf, wqf, pquadf


    @staticmethod
    def quad_rule_3D_lg(pquad_req):
        '''
        Parameters
        ----------
        pquad_req : int
            Requsetd degree for the quadrature rule.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Actual degree of the quad rule
        '''

        if pquad_req <= 2:
            pquad = 2
            dat = np.array([[0.138196601125011, 0.138196601125011, 0.585410196624969, 0.041666666666667],
                           [0.138196601125011, 0.585410196624969, 0.138196601125011, 0.041666666666667],
                           [0.585410196624969, 0.138196601125011, 0.138196601125011, 0.041666666666667],
                           [0.138196601125011, 0.138196601125011, 0.138196601125011, 0.041666666666667]])
        elif pquad_req <= 4:
            pquad = 4
            dat = np.array([[0.092735250310891, 0.092735250310891, 0.721794249067326, 0.012248840519394],
                            [0.092735250310891, 0.721794249067326, 0.092735250310891, 0.012248840519394],
                            [0.721794249067326, 0.092735250310891, 0.092735250310891, 0.012248840519394],
                            [0.092735250310891, 0.092735250310891, 0.092735250310891, 0.012248840519394],
                            [0.310885919263301, 0.310885919263301, 0.067342242210098, 0.018781320953003],
                            [0.310885919263301, 0.067342242210098, 0.310885919263301, 0.018781320953003],
                            [0.067342242210098, 0.310885919263301, 0.310885919263301, 0.018781320953003],
                            [0.310885919263301, 0.310885919263301, 0.310885919263301, 0.018781320953003],
                            [0.045503704125650, 0.454496295874350, 0.454496295874350, 0.007091003462847],
                            [0.454496295874350, 0.045503704125650, 0.454496295874350, 0.007091003462847],
                            [0.454496295874350, 0.454496295874350, 0.045503704125650, 0.007091003462847],
                            [0.045503704125650, 0.454496295874350, 0.045503704125650, 0.007091003462847],
                            [0.045503704125650, 0.045503704125650, 0.454496295874350, 0.007091003462847],
                            [0.454496295874350, 0.045503704125650, 0.045503704125650, 0.007091003462847]])
        elif pquad_req <= 6:
            pquad = 6
            dat = np.array([[0.214602871259152, 0.214602871259152, 0.356191386222544, 0.006653791709695],
                            [0.214602871259152, 0.356191386222544, 0.214602871259152, 0.006653791709695],
                            [0.356191386222544, 0.214602871259152, 0.214602871259152, 0.006653791709695],
                            [0.214602871259152, 0.214602871259152, 0.214602871259152, 0.006653791709695],
                            [0.040673958534611, 0.040673958534611, 0.877978124396166, 0.001679535175887],
                            [0.040673958534611, 0.877978124396166, 0.040673958534611, 0.001679535175887],
                            [0.877978124396166, 0.040673958534611, 0.040673958534611, 0.001679535175887],
                            [0.040673958534611, 0.040673958534611, 0.040673958534611, 0.001679535175887],
                            [0.322337890142275, 0.322337890142275, 0.032986329573174, 0.009226196923942],
                            [0.322337890142275, 0.032986329573174, 0.322337890142275, 0.009226196923942],
                            [0.032986329573174, 0.322337890142275, 0.322337890142275, 0.009226196923942],
                            [0.322337890142275, 0.322337890142275, 0.322337890142275, 0.009226196923942],
                            [0.063661001875018, 0.603005664791649, 0.269672331458316, 0.008035714285714],
                            [0.063661001875018, 0.269672331458316, 0.603005664791649, 0.008035714285714],
                            [0.603005664791649, 0.063661001875018, 0.269672331458316, 0.008035714285714],
                            [0.603005664791649, 0.269672331458316, 0.063661001875018, 0.008035714285714],
                            [0.269672331458316, 0.063661001875018, 0.603005664791649, 0.008035714285714],
                            [0.269672331458316, 0.603005664791649, 0.063661001875018, 0.008035714285714],
                            [0.063661001875018, 0.063661001875018, 0.269672331458316, 0.008035714285714],
                            [0.063661001875018, 0.269672331458316, 0.063661001875018, 0.008035714285714],
                            [0.269672331458316, 0.063661001875018, 0.063661001875018, 0.008035714285714],
                            [0.063661001875018, 0.063661001875018, 0.603005664791649, 0.008035714285714],
                            [0.063661001875018, 0.603005664791649, 0.063661001875018, 0.008035714285714],
                            [0.603005664791649, 0.063661001875018, 0.063661001875018, 0.008035714285714]])

        else:
            raise Exception('The highest degree quad rule for the tetrahedral is 6')

        xq = dat[:, :-1]
        wq = dat[:,-1]

        return xq, wq, pquad

    @staticmethod
    def quad_rule_3d_Rd(p):
        '''
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        if p == 1:
            pquad = 2

            xq = np.array([[0.1381966011250105, 0.5854101966249684, 0.1381966011250105, 0.1381966011250105],
                           [0.1381966011250105, 0.1381966011250105, 0.5854101966249684, 0.1381966011250105],
                           [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249684]]).T

            wq = np.array([0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666])
        elif p == 2:
            xq = np.array([[0.0732572627069854, 0.7802282118790438, 0.0732572627069854, 0.0732572627069854, 0.09370628418119539, 0.4062937158188046, 0.4062937158188046, 0.09370628418119539, 0.09370628418119539, 0.4062937158188046],
           [0.0732572627069854, 0.0732572627069854, 0.7802282118790438, 0.0732572627069854, 0.4062937158188046, 0.09370628418119539, 0.4062937158188046, 0.09370628418119539, 0.4062937158188046, 0.09370628418119539],
           [0.0732572627069854, 0.0732572627069854, 0.0732572627069854, 0.7802282118790438, 0.4062937158188046, 0.4062937158188046, 0.09370628418119539, 0.4062937158188046, 0.09370628418119539, 0.09370628418119539]]).T

            wq = np.array([0.007861271487071622, 0.007861271487071622, 0.007861271487071622, 0.007861271487071622, 0.02253693011973003, 0.02253693011973003, 0.02253693011973003, 0.02253693011973003, 0.02253693011973003, 0.02253693011973003])
        else:
            raise Exception('The requested degree for the DD Rd quad is not available')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_3d_Rdn1(p):
        '''
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        if p == 1:
            pquad = 2

            xq = np.array([[0.25, 0, 1, 0, 0],
                           [0.25, 0, 0, 1, 0],
                           [0.25, 0, 0, 0, 1]]).T
            wq = np.array([0.1333333333333333, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333])
        elif p == 2:
            pquad = 4
            xq = np.array([[0.1505717835195518, 0.5482846494413446, 0.1505717835195518, 0.1505717835195518, 0.1134031783538211, 0, 0.4432984108230895, 0.4432984108230895, 0.4432984108230895, 0.4432984108230895, 0, 0.1134031783538211, 0, 0.1134031783538211, 0.4432984108230895, 0.4432984108230895, 0, 1, 0, 0],
                           [0.1505717835195518, 0.1505717835195518, 0.5482846494413446, 0.1505717835195518, 0.4432984108230895, 0.4432984108230895, 0.1134031783538211, 0, 0.4432984108230895, 0.4432984108230895, 0.1134031783538211, 0, 0.4432984108230895, 0.4432984108230895, 0, 0.1134031783538211, 0, 0, 1, 0],
                           [0.1505717835195518, 0.1505717835195518, 0.1505717835195518, 0.5482846494413446, 0.4432984108230895, 0.4432984108230895, 0.4432984108230895, 0.4432984108230895, 0.1134031783538211, 0, 0.4432984108230895, 0.4432984108230895, 0.1134031783538211, 0, 0.1134031783538211, 0, 0, 0, 0, 1]]).T

            wq = np.array([0.02650163091863721, 0.02650163091863721, 0.02650163091863721, 0.02650163091863721, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.004638686694416115, 0.001248975664781117, 0.001248975664781117, 0.001248975664781117, 0.001248975664781117])

        else:
            raise Exception('The requested degree for the DD Rd quad is not available')

        return xq, wq, pquad

    @staticmethod
    def quad_rule_3d_R0(p):
        '''
        Parameters
        ----------
        p : int
            Degree of the SBP operator.

        Returns
        -------
        xq : numpy array
            Cartesian coordinates of the quadrature nodes.
        wq : numpy array
            Quatrature weights.
        pquad : int
            Degree of the quadrature rule.
        '''

        if p == 1:
            pquad = 2

            xq = np.array([[0.25, 0, 0.5, 0.5, 0, 0, 0.5],
                           [0.25, 0.5, 0, 0.5, 0, 0.5, 0],
                           [0.25, 0.5, 0.5, 0, 0.5, 0, 0]]).T

            wq = np.array([0.06666666666666667, 0.01666666666666667, 0.01666666666666667, 0.01666666666666667, 0.01666666666666667, 0.01666666666666667, 0.01666666666666667])

            # Facet quad rule
            pquadf = 2
            xqf = np.array([[0.5, 0, 0.5],
                            [0.5, 0.5, 0]]).T
            wqf = np.array([0.1666666666666667, 0.1666666666666667, 0.1666666666666667])


        elif p == 2:
            pquad = 4
            xq = np.array([[0.1614317819100586, 0.5157046542698243, 0.1614317819100586, 0.1614317819100586, 0.3333333333333333, 0, 0.3333333333333333, 0.3333333333333333, 0, 0.7777777777777778, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.7777777777777778, 0, 0.7777777777777778, 0, 0.1111111111111111, 0.1111111111111111, 0, 0.5, 0.5, 0, 0, 0.5],
                           [0.1614317819100586, 0.1614317819100586, 0.5157046542698243, 0.1614317819100586, 0.3333333333333333, 0.3333333333333333, 0, 0.3333333333333333, 0.1111111111111111, 0.1111111111111111, 0, 0.7777777777777778, 0.1111111111111111, 0.1111111111111111, 0, 0.7777777777777778, 0.1111111111111111, 0.1111111111111111, 0.7777777777777778, 0, 0.5, 0, 0.5, 0, 0.5, 0],
                           [0.1614317819100586, 0.1614317819100586, 0.1614317819100586, 0.5157046542698243, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0, 0.7777777777777778, 0.1111111111111111, 0.1111111111111111, 0, 0.7777777777777778, 0, 0.7777777777777778, 0.5, 0.5, 0, 0.5, 0, 0]]).T

            wq = np.array([0.02286342553084723, 0.02286342553084723, 0.02286342553084723, 0.02286342553084723, 0.008415068491623122, 0.008415068491623122, 0.008415068491623122, 0.008415068491623122, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002055914429168193, 0.002813619571127827, 0.002813619571127827, 0.002813619571127827, 0.002813619571127827, 0.002813619571127827, 0.002813619571127827])

            # Facet quad rule
            pquadf = 4
            xqf = np.array([[0.3333333333333333, 0.1111111111111111, 0.7777777777777778, 0.1111111111111111, 0.5, 0, 0.5],
                            [0.3333333333333333, 0.1111111111111111, 0.1111111111111111, 0.7777777777777778, 0.5, 0.5, 0]]).T

            wqf = np.array([0.16875, 0.07232142857142858, 0.07232142857142858, 0.07232142857142858, 0.0380952380952381, 0.0380952380952381, 0.0380952380952381])

        else:
            raise Exception('The requested degree for the DD Rd quad is not available')

        return xq, wq, pquad, xqf, wqf, pquadf
