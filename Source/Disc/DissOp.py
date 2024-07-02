import numpy as np

def make_dcp_diss_op(sbp_type, s, nen):
    ''' make the relevant operators according to DCP implementation in diablo '''
    if sbp_type.lower() == 'csbp':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)

        if s==1:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            # Row 1
            Ds[0, 0] = -1.0
            Ds[0, 1] = 1.0
            # Interior rows
            for i in range(1, nen-1):
                Ds[i, i-1] = -0.5
                Ds[i, i+1] = 0.5
            # Row nen
            Ds[nen-1, nen-2] = -1.0
            Ds[nen-1, nen-1] = 1.0
            
            # correct boundary values
            B[0] = 0.
            B[-1] = 0.

        if s==2:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")

            # Row 1
            Ds[0, 0] = 1.0
            Ds[0, 1] = -2.0
            Ds[0, 2] = 1.0
            # Interior rows
            for i in range(1, nen-1):
                Ds[i, i-1] = 1.0
                Ds[i, i] = -2.0
                Ds[i, i+1] = 1.0
            # Row nen
            Ds[nen-1, nen-3] = 1.0
            Ds[nen-1, nen-2] = -2.0
            Ds[nen-1, nen-1] = 1.0
            
            # correct boundary values
            B[0] = 0.
            B[-1] = 0.
        
        elif s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            
            # First half node
            Ds[0, 0] = -1.0
            Ds[0, 1] = 3.0
            Ds[0, 2] = -3.0
            Ds[0, 3] = 1.0

            # Interior half-nodes
            for i in range(1, nen-2):
                Ds[i, i-1] = -1.0
                Ds[i, i] = 3.0
                Ds[i, i+1] = -3.0
                Ds[i, i+2] = 1.0

            # Last half node
            Ds[nen-2, nen-4] = -1.0
            Ds[nen-2, nen-3] = 3.0
            Ds[nen-2, nen-2] = -3.0
            Ds[nen-2, nen-1] = 1.0

            # Last node; nothing is added to this node
            # The last row of Ds remains zero

            # correct boundary values
            B[0] = 0.
            #B[1] = 1.
            B[-1] = 0.
            B[-2] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")

            # First node
            Ds[0, 0] = 1.0
            Ds[0, 1] = -4.0
            Ds[0, 2] = 6.0
            Ds[0, 3] = -4.0
            Ds[0, 4] = 1.0

            # Second node
            Ds[1, 0] = 1.0
            Ds[1, 1] = -4.0
            Ds[1, 2] = 6.0
            Ds[1, 3] = -4.0
            Ds[1, 4] = 1.0

            # Interior nodes
            for i in range(2, nen-2):
                Ds[i, i-2] = 1.0
                Ds[i, i-1] = -4.0
                Ds[i, i] = 6.0
                Ds[i, i+1] = -4.0
                Ds[i, i+2] = 1.0

            # Second last node
            Ds[nen-2, nen-5] = 1.0
            Ds[nen-2, nen-4] = -4.0
            Ds[nen-2, nen-3] = 6.0
            Ds[nen-2, nen-2] = -4.0
            Ds[nen-2, nen-1] = 1.0

            # Last node
            Ds[nen-1, nen-5] = 1.0
            Ds[nen-1, nen-4] = -4.0
            Ds[nen-1, nen-3] = 6.0
            Ds[nen-1, nen-2] = -4.0
            Ds[nen-1, nen-1] = 1.0

            # correct boundary values
            B[0] = 0.
            B[1] = 0.
            B[-1] = 0.
            B[-2] = 0.
        
        elif s==5:
            if nen < 17:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")

            # First half-node
            Ds[0, 0] = -1.0
            Ds[0, 1] = 5.0
            Ds[0, 2] = -10.0
            Ds[0, 3] = 10.0
            Ds[0, 4] = -5.0
            Ds[0, 5] = 1.0

            # Second half-node
            Ds[1, 0] = -1.0
            Ds[1, 1] = 5.0
            Ds[1, 2] = -10.0
            Ds[1, 3] = 10.0
            Ds[1, 4] = -5.0
            Ds[1, 5] = 1.0

            # Interior half-nodes
            for i in range(2, nen-3):
                Ds[i, i-2] = -1.0
                Ds[i, i-1] = 5.0
                Ds[i, i] = -10.0
                Ds[i, i+1] = 10.0
                Ds[i, i+2] = -5.0
                Ds[i, i+3] = 1.0

            # Second last half-node
            Ds[nen-3, nen-6] = -1.0
            Ds[nen-3, nen-5] = 5.0
            Ds[nen-3, nen-4] = -10.0
            Ds[nen-3, nen-3] = 10.0
            Ds[nen-3, nen-2] = -5.0
            Ds[nen-3, nen-1] = 1.0

            # Last half-node
            Ds[nen-2, nen-6] = -1.0
            Ds[nen-2, nen-5] = 5.0
            Ds[nen-2, nen-4] = -10.0
            Ds[nen-2, nen-3] = 10.0
            Ds[nen-2, nen-2] = -5.0
            Ds[nen-2, nen-1] = 1.0

            # Last node; nothing is added to this node
            # The last row of Ds remains zero

            # correct boundary values
            B[0] = 0.
            B[1] = 0.
            B[-1] = 0.
            B[-2] = 0.
            B[-3] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=2,3,4,5.')
        
    elif sbp_type.lower() == 'lgl':
                # Initialize the matrix as a dense NumPy array
                Ds = np.zeros((nen, nen))
                B = np.ones(nen)

                if s==1:
                    if nen != 2:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = -1.0
                    Ds[0, 1] = 1.0

                    # Row 2
                    Ds[1, 0] = -1.0
                    Ds[1, 1] = 1.0

                elif s==2:
                    if nen != 3:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = 1.0
                    Ds[0, 1] = -2.0
                    Ds[0, 2] = 1.0

                    # Row 2
                    Ds[1, 0] = 1.0
                    Ds[1, 1] = -2.0
                    Ds[1, 2] = 1.0

                    # Row 3
                    Ds[2, 0] = 1.0
                    Ds[2, 1] = -2.0
                    Ds[2, 2] = 1.0

                elif s==3:
                    if nen != 4:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")
                    
                    # Row 1
                    Ds[0, 0] = -1.1111111111111111
                    Ds[0, 1] = 2.484519974999766
                    Ds[0, 2] = -2.484519974999766
                    Ds[0, 3] = 1.1111111111111111

                    # Row 2
                    Ds[1, 0] = -1.1111111111111111
                    Ds[1, 1] = 2.484519974999766
                    Ds[1, 2] = -2.484519974999766
                    Ds[1, 3] = 1.1111111111111111

                    # Row 3
                    Ds[2, 0] = -1.1111111111111111
                    Ds[2, 1] = 2.484519974999766
                    Ds[2, 2] = -2.484519974999766
                    Ds[2, 3] = 1.1111111111111111

                    # Row 4
                    Ds[3, 0] = -1.1111111111111111
                    Ds[3, 1] = 2.484519974999766
                    Ds[3, 2] = -2.484519974999766
                    Ds[3, 3] = 1.1111111111111111

                elif s==4:
                    if nen != 5:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = 1.3125
                    Ds[0, 1] = -3.0625
                    Ds[0, 2] = 3.5
                    Ds[0, 3] = -3.0625
                    Ds[0, 4] = 1.3125

                    # Row 2
                    Ds[1, 0] = 1.3125
                    Ds[1, 1] = -3.0625
                    Ds[1, 2] = 3.5
                    Ds[1, 3] = -3.0625
                    Ds[1, 4] = 1.3125

                    # Row 3
                    Ds[2, 0] = 1.3125
                    Ds[2, 1] = -3.0625
                    Ds[2, 2] = 3.5
                    Ds[2, 3] = -3.0625
                    Ds[2, 4] = 1.3125

                    # Row 4
                    Ds[3, 0] = 1.3125
                    Ds[3, 1] = -3.0625
                    Ds[3, 2] = 3.5
                    Ds[3, 3] = -3.0625
                    Ds[3, 4] = 1.3125

                    # Row 5
                    Ds[4, 0] = 1.3125
                    Ds[4, 1] = -3.0625
                    Ds[4, 2] = 3.5
                    Ds[4, 3] = -3.0625
                    Ds[4, 4] = 1.3125

                else:
                    raise Exception('Invalid choice of s. Only coded up s=1,2,3,4.')
                
    elif sbp_type.lower() == 'lg':
                # Initialize the matrix as a dense NumPy array
                Ds = np.zeros((nen, nen))
                B = np.ones(nen)

                if s==1:
                    if nen != 2:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = -1.732050807568877
                    Ds[0, 1] = 1.732050807568877

                    # Row 2
                    Ds[1, 0] = -1.732050807568877
                    Ds[1, 1] = 1.732050807568877

                elif s==2:
                    if nen != 3:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = 6.666666666666667
                    Ds[0, 1] = -13.333333333333334
                    Ds[0, 2] = 6.666666666666667

                    # Row 2
                    Ds[1, 0] = 6.666666666666667
                    Ds[1, 1] = -13.333333333333334
                    Ds[1, 2] = 6.666666666666667

                    # Row 3
                    Ds[2, 0] = 6.666666666666667
                    Ds[2, 1] = -13.333333333333334
                    Ds[2, 2] = 6.666666666666667

                elif s==3:
                    if nen != 4:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = -5.56540505102921
                    Ds[0, 1] = 14.096587055666296
                    Ds[0, 2] = -14.096587055666296
                    Ds[0, 3] = 5.56540505102921

                    # Row 2
                    Ds[1, 0] = -5.56540505102921
                    Ds[1, 1] = 14.096587055666296
                    Ds[1, 2] = -14.096587055666296
                    Ds[1, 3] = 5.56540505102921

                    # Row 3
                    Ds[2, 0] = -5.56540505102921
                    Ds[2, 1] = 14.096587055666296
                    Ds[2, 2] = -14.096587055666296
                    Ds[2, 3] = 5.56540505102921

                    # Row 4
                    Ds[3, 0] = -5.56540505102921
                    Ds[3, 1] = 14.096587055666296
                    Ds[3, 2] = -14.096587055666296
                    Ds[3, 3] = 5.56540505102921

                elif s==4:
                    if nen != 5:
                        raise ValueError(f"Invalid number of nodes. nen = {nen}")

                    # Row 1
                    Ds[0, 0] = 27.50958167164676
                    Ds[0, 1] = -77.90958167164676
                    Ds[0, 2] = 100.8
                    Ds[0, 3] = -77.90958167164676
                    Ds[0, 4] = 27.50958167164676

                    # Row 2
                    Ds[1, 0] = 27.50958167164676
                    Ds[1, 1] = -77.90958167164676
                    Ds[1, 2] = 100.8
                    Ds[1, 3] = -77.90958167164676
                    Ds[1, 4] = 27.50958167164676

                    # Row 3
                    Ds[2, 0] = 27.50958167164676
                    Ds[2, 1] = -77.90958167164676
                    Ds[2, 2] = 100.8
                    Ds[2, 3] = -77.90958167164676
                    Ds[2, 4] = 27.50958167164676

                    # Row 4
                    Ds[3, 0] = 27.50958167164676
                    Ds[3, 1] = -77.90958167164676
                    Ds[3, 2] = 100.8
                    Ds[3, 3] = -77.90958167164676
                    Ds[3, 4] = 27.50958167164676

                    # Row 5
                    Ds[4, 0] = 27.50958167164676
                    Ds[4, 1] = -77.90958167164676
                    Ds[4, 2] = 100.8
                    Ds[4, 3] = -77.90958167164676
                    Ds[4, 4] = 27.50958167164676

                else:
                    raise Exception('Invalid choice of s. Only coded up s=1,2,3,4.')
    return Ds,B

class BaselineDiss:
    def __init__(self, s, n):
        self.s = s
        self.n = n

    def updateD1(self):
        n = self.n
        if self.s in {1, 2, 3, 4}:
            D = np.zeros((n, n))

            # row 1
            D[0, 0] = -1.0
            D[0, 1] = 1.0

            # interior rows
            for i in range(1, n - 1):
                D[i, i - 1] = -0.5
                D[i, i + 1] = 0.5

            # row n
            D[n - 1, n - 2] = -1.0
            D[n - 1, n - 1] = 1.0

            self.D1 = D
        else:
            raise ValueError('unsupported dissipation degree')

    def updateD2(self):
        n = self.n
        if self.s == 1:
            D = np.zeros((n, n))

            # row 1
            D[0, 0] = 1.0
            D[0, 1] = -2.0
            D[0, 2] = 1.0

            # interior rows
            for i in range(1, n - 1):
                D[i, i - 1] = 1.0
                D[i, i] = -2.0
                D[i, i + 1] = 1.0

            # row n
            D[n - 1, n - 3] = 1.0
            D[n - 1, n - 2] = -2.0
            D[n - 1, n - 1] = 1.0

            self.D2 = D
        else:
            raise ValueError('unsupported dissipation degree')

    def updateD3(self):
        n = self.n
        if self.s == 2:
            D = np.zeros((n, n))

            # row 1, 2
            for i in [0,1]:
                D[i, 0] = -0.5
                D[i, 1] = 1.0
                D[i, 3] = -1.0
                D[i, 4] = 0.5

            # interior rows
            for i in range(2, n - 2):
                D[i, i - 2] = -0.5
                D[i, i - 1] = 1.0
                D[i, i + 1] = -1.0
                D[i, i + 2] = 0.5

            # row n-1, n
            for i in [2,1]:
                D[n - i, n - 5] = -0.5
                D[n - i, n - 4] = 1.0
                D[n - i, n - 2] = -1.0
                D[n - i, n - 1] = 0.5

            self.D3 = D

        else:
            raise ValueError('unsupported dissipation degree')

    def updateD4(self):
        n = self.n
        if self.s == 2:
            D = np.zeros((n, n))

            # row 1, 2
            for i in [0,1]:
                D[i, 0] = 1.0
                D[i, 1] = -4.0
                D[i, 2] = 6.0
                D[i, 3] = -4.0
                D[i, 4] = 1.0

            # interior rows
            for i in range(2, n - 2):
                D[i, i - 2] = 1.0
                D[i, i - 1] = -4.0
                D[i, i] = 6.0
                D[i, i + 1] = -4.0
                D[i, i + 2] = 1.0

            # row n-1, n
            for i in [2,1]:
                D[n - i, n - 5] = 1.0
                D[n - i, n - 4] = -4.0
                D[n - i, n - 3] = 6.0
                D[n - i, n - 2] = -4.0
                D[n - i, n - 1] = 1.0

            self.D4 = D

        elif self.s == 3:
            D = np.zeros((n, n))

            # row 1, 2, 3
            for i in [0,1,2]:
                D[i, 0] = 1.0/3.0
                D[i, 1] = -1.0
                D[i, 2] = 1.0
                D[i, 3] = -2.0/3.0
                D[i, 4] = 1.0
                D[i, 5] = -1.0
                D[i, 6] = 1.0/3.0

            # interior rows
            for i in range(3, n - 3):
                D[i, i - 3] = 1.0/3.0
                D[i, i - 2] = -1.0
                D[i, i - 1] = 1.0
                D[i, i ] = -2.0/3.0
                D[i, i + 1] = 1.0
                D[i, i + 2] = -1.0
                D[i, i + 3] = 1.0/3.0

            # row n-2, n-1, n
            for i in [3,2,1]:
                D[n - i, n - 7] = 1.0/3.0
                D[n - i, n - 6] = -1.0
                D[n - i, n - 5] = 1.0
                D[n - i, n - 4] = -2.0/3.0
                D[n - i, n - 3] = 1.0
                D[n - i, n - 2] = -1.0
                D[n - i, n - 1] = 1.0/3.0

            self.D4 = D
        else:
            raise ValueError('unsupported dissipation degree')
        

    def updateD5(self):
        n = self.n
        if self.s == 3:
            D = np.zeros((n, n))

            # row 1, 2
            for i in [0,1,2]:
                D[i, 0] = -0.5
                D[i, 1] = 2.0
                D[i, 2] = -2.5
                D[i, 4] = 2.5
                D[i, 5] = -2.0
                D[i, 6] = 0.5

            # interior rows
            for i in range(3, n - 3):
                D[i, i - 3] = -0.5
                D[i, i - 2] = 2.0
                D[i, i - 1] = -2.5
                D[i, i + 1] = 2.5
                D[i, i + 2] = -2.0
                D[i, i + 3] = 0.5

            # row n-1, n
            for i in [3,2,1]:
                D[n - i, n - 7] = -0.5
                D[n - i, n - 6] = 2.0
                D[n - i, n - 5] = -2.5
                D[n - i, n - 3] = 2.5
                D[n - i, n - 2] = -2.0
                D[n - i, n - 1] = 0.5

            self.D5 = D

        elif self.s == 4:
            D = np.zeros((n, n))

            # row 1, 2, 3, 4
            for i in [0,1,2,3]:
                D[i, 0] = -0.25
                D[i, 1] = 1.0
                D[i, 2] = -1.5
                D[i, 3] = 1.0
                D[i, 5] = -1.0
                D[i, 6] = 1.5
                D[i, 7] = -1.0
                D[i, 8] = 0.25

            # interior rows
            for i in range(4, n - 4):
                D[i, i - 4] = -0.25
                D[i, i - 3] = 1.0
                D[i, i - 2] = -1.5
                D[i, i - 1] = 1.0
                D[i, i + 1] = -1.0
                D[i, i + 2] = 1.5
                D[i, i + 3] = -1.0
                D[i, i + 4] = 0.25

            # row n-3, n-2, n-1, n
            for i in [4, 3,2,1]:
                D[n - i, n - 9] = -0.25
                D[n - i, n - 8] = 1.0
                D[n - i, n - 7] = -1.5
                D[n - i, n - 6] = 1.0
                D[n - i, n - 4] = -1.0
                D[n - i, n - 3] = 1.5
                D[n - i, n - 2] = -1.0
                D[n - i, n - 1] = 0.25

            self.D5 = D
        else:
            raise ValueError('unsupported dissipation degree')
        
    def updateD6(self):
        n = self.n
        if self.s == 3:
            D = np.zeros((n, n))

            # row 1, 2
            for i in [0,1,2]:
                D[i, 0] = 1.0
                D[i, 1] = -6.0
                D[i, 2] = 15.0
                D[i, 3] = -20.0
                D[i, 4] = 15.0
                D[i, 5] = -6.0
                D[i, 6] = 1.0

            # interior rows
            for i in range(3, n - 3):
                D[i, i - 3] = 1.0
                D[i, i - 2] = -6.0
                D[i, i - 1] = 15.0
                D[i, i ] = -20.0
                D[i, i + 1] = 15.0
                D[i, i + 2] = -6.0
                D[i, i + 3] = 1.0

            # row n-1, n
            for i in [3,2,1]:
                D[n - i, n - 7] = 1.0
                D[n - i, n - 6] = -6.0
                D[n - i, n - 5] = 15.0
                D[n - i, n - 4] = -20.0
                D[n - i, n - 3] = 15.0
                D[n - i, n - 2] = -6.0
                D[n - i, n - 1] = 1.0

            self.D6 = D

        elif self.s == 4:
            D = np.zeros((n, n))

            # row 1, 2, 3, 4
            for i in [0,1,2,3]:
                D[i, 0] = 0.3
                D[i, 1] = -1.4
                D[i, 2] = 2.4
                D[i, 3] = -1.8
                D[i, 4] = 1.0
                D[i, 5] = -1.8
                D[i, 6] = 2.4
                D[i, 7] = -1.4
                D[i, 8] = 0.3

            # interior rows
            for i in range(4, n - 4):
                D[i, i - 4] = 0.3
                D[i, i - 3] = -1.4
                D[i, i - 2] = 2.4
                D[i, i - 1] = -1.8
                D[i, i ] = 1.0
                D[i, i + 1] = -1.8
                D[i, i + 2] = 2.4
                D[i, i + 3] = -1.4
                D[i, i + 4] = 0.3

            # row n-3, n-2, n-1, n
            for i in [4, 3,2,1]:
                D[n - i, n - 9] = 0.3
                D[n - i, n - 8] = -1.4
                D[n - i, n - 7] = 2.4
                D[n - i, n - 6] = -1.8
                D[n - i, n - 5] = 1.0
                D[n - i, n - 4] = -1.8
                D[n - i, n - 3] = 2.4
                D[n - i, n - 2] = -1.4
                D[n - i, n - 1] = 0.3

            self.D6 = D
        else:
            raise ValueError('unsupported dissipation degree')
        
    def updateD7(self):
        n = self.n
        if self.s == 4:
            D = np.zeros((n, n))

            # row 1, 2, 3, 4
            for i in [0,1,2,3]:
                D[i, 0] = -0.5
                D[i, 1] = 3.0
                D[i, 2] = -7.0
                D[i, 3] = 7.0
                D[i, 5] = -7.0
                D[i, 6] = 7.0
                D[i, 7] = -3.0
                D[i, 8] = 0.5

            # interior rows
            for i in range(4, n - 4):
                D[i, i - 4] = -0.5
                D[i, i - 3] = 3.0
                D[i, i - 2] = -7.0
                D[i, i - 1] = 7.0
                D[i, i + 1] = -7.0
                D[i, i + 2] = 7.0
                D[i, i + 3] = -3.0
                D[i, i + 4] = 0.5

            # row n-3, n-2, n-1, n
            for i in [4, 3,2,1]:
                D[n - i, n - 9] = -0.5
                D[n - i, n - 8] = 3.0
                D[n - i, n - 7] = -7.0
                D[n - i, n - 6] = 7.0
                D[n - i, n - 4] = -7.0
                D[n - i, n - 3] = 7.0
                D[n - i, n - 2] = -3.0
                D[n - i, n - 1] = 0.5

            self.D7 = D
        else:
            raise ValueError('unsupported dissipation degree')
        
    def updateD8(self):
        n = self.n
        if self.s == 4:
            D = np.zeros((n, n))

            # row 1, 2, 3, 4
            for i in [0,1,2,3]:
                D[i, 0] = 1.0
                D[i, 1] = -8.0
                D[i, 2] = 28.0
                D[i, 3] = -56.0
                D[i, 4] = 70.0
                D[i, 5] = -56.0
                D[i, 6] = 28.0
                D[i, 7] = -8.0
                D[i, 8] = 1.0

            # interior rows
            for i in range(4, n - 4):
                D[i, i - 4] = 1.0
                D[i, i - 3] = -8.0
                D[i, i - 2] = 28.0
                D[i, i - 1] = -56.0
                D[i, i    ] = 70.0
                D[i, i + 1] = -56.0
                D[i, i + 2] = 28.0
                D[i, i + 3] = -8.0
                D[i, i + 4] = 1.0

            # row n-3, n-2, n-1, n
            for i in [4, 3,2,1]:
                D[n - i, n - 9] = 1.0
                D[n - i, n - 8] = -8.0
                D[n - i, n - 7] = 28.0
                D[n - i, n - 6] = -56.0
                D[n - i, n - 5] = 70.0
                D[n - i, n - 4] = -56.0
                D[n - i, n - 3] = 28.0
                D[n - i, n - 2] = -8.0
                D[n - i, n - 1] = 1.0

            self.D8 = D
        else:
            raise ValueError('unsupported dissipation degree')
        

    def updateB1(self):
        n = self.n
        if self.s == 1:
            self.B1 = np.diag([0] + [1] * (n - 2) + [0])
        elif self.s == 2:
            self.B1 = np.diag([0, 0] + [1] * (n - 4) + [0, 0])
        elif self.s == 3:
            self.B1 = np.diag([0, 0, 0] + [1] * (n - 6) + [0, 0, 0])
        elif self.s == 4:
            self.B1 = np.diag([0, 0, 0, 0] + [1] * (n - 8) + [0, 0, 0, 0])
        else:
            raise ValueError('unsupported degree')

    def updateB2(self):
        n = self.n
        if self.s == 1:
            self.B2 = np.diag([0] + [1] * (n - 2) + [0])
        else:
            raise ValueError('unsupported degree')

    def updateB3(self):
        n = self.n
        if self.s == 2:
            self.B3 = np.diag([0, 0] + [1] * (n - 4) + [0, 0])
        else:
            raise ValueError('unsupported dissipation degree')

    def updateB4(self):
        n = self.n
        if self.s == 2:
            self.B4 = np.diag([0, 0] + [1] * (n - 4) + [0, 0])
        elif self.s == 3:
            self.B4 = np.diag([0, 0, 0] + [1] * (n - 6) + [0, 0, 0])
        else:
            raise ValueError('unsupported dissipation degree')

    def updateB5(self):
        n = self.n
        if self.s == 3:
            self.B5 = np.diag([0, 0, 0] + [1] * (n - 6) + [0, 0, 0])
        elif self.s == 4:
            self.B5 = np.diag([0, 0, 0, 0] + [1] * (n - 8) + [0, 0, 0, 0])
        else:
            raise ValueError('unsupported dissipation degree')

    def updateB6(self):
        n = self.n
        if self.s == 3:
            self.B6 = np.diag([0, 0, 0] + [1] * (n - 6) + [0, 0, 0])
        elif self.s == 4:
            self.B6 = np.diag([0, 0, 0, 0] + [1] * (n - 8) + [0, 0, 0, 0])
        else:
            raise ValueError('unsupported dissipation degree')

    def updateB7(self):
        n = self.n
        if self.s == 4:
            self.B7 = np.diag([0, 0, 0, 0] + [1] * (n - 8) + [0, 0, 0, 0])
        else:
            raise ValueError('unsupported dissipation degree')

    def updateB8(self):
        n = self.n
        if self.s == 4:
            self.B8 = np.diag([0, 0, 0, 0] + [1] * (n - 8) + [0, 0, 0, 0])
        else:
            raise ValueError('unsupported dissipation degree')