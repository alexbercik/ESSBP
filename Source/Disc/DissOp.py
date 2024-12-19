import numpy as np

def make_dcp_diss_op(sbp_type, s, nen, boundary_fix=True):
    ''' make the relevant operators according to DCP implementation in diablo '''
    if sbp_type.lower() == 'csbp':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)

        if s==1:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 7:
                print('WARNING: Not enough nodes for volume dissipation interior to be 3rd order. Try nen>=5')

            print('WARNING: I am not sure that s=1 CSBP dissipation operator is correct')

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
            
            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[-1] = 0.

        elif s==2:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 9:
                print('WARNING: Not enough nodes for volume dissipation interior to be 3rd order. Try nen>=5')

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
            
            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[-1] = 0.
        
        elif s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 11:
                print('WARNING: Not enough nodes for volume dissipation interior to be 5th order. Try nen>=11')
            
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

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[-1] = 0.
                B[-2] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 13:
                print('WARNING: Not enough nodes for volume dissipation interior to be 7th order. Try nen>=15')

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

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
        
        elif s==5:
            if nen < 17:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 15:
                print('WARNING: Not enough nodes for volume dissipation interior to be 9th order. Try nen>=19')

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

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
                B[-3] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=1,2,3,4,5.')
        
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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[-1] = 0.

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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[-1] = 0.

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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[1] = 0.
                        B[-1] = 0.
                        B[-2] = 0.

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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[-1] = 0.

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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[-1] = 0.

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

                    if boundary_fix:
                        # correct boundary values
                        B[0] = 0.
                        B[1] = 0.
                        B[-1] = 0.
                        B[-2] = 0.

                else:
                    raise Exception('Invalid choice of s. Only coded up s=1,2,3,4.')
                
    elif sbp_type.lower() == 'hgtl':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)
        
        if s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 11:
                print('WARNING: Not enough nodes for volume dissipation interior to be 5th order. Try nen>=11')
            
            Ds[0, 0] = -1.083333333333333333
            Ds[0, 1] = 2.906084656084656085
            Ds[0, 2] = -2.785714285714285714
            Ds[0, 3] = 0.9629629629629629630

            Ds[1, 0] = -1.083333333333333333
            Ds[1, 1] = 2.906084656084656085
            Ds[1, 2] = -2.785714285714285714
            Ds[1, 3] = 0.9629629629629629630

            Ds[2, 0] = -1.083333333333333333
            Ds[2, 1] = 2.906084656084656085
            Ds[2, 2] = -2.785714285714285714
            Ds[2, 3] = 0.9629629629629629630

            Ds[3, 1] = -0.8718253968253968254
            Ds[3, 2] = 2.785714285714285714
            Ds[3, 3] = -2.888888888888888889
            Ds[3, 4] = 0.975

            # Interior
            for i in range(4, nen-3):
                Ds[i, i-2] = -1.0
                Ds[i, i-1] = 3.0
                Ds[i, i] = -3.0
                Ds[i, i+1] = 1.0

            # copy the bottom as mirror image
            for i in range(3):
                for j in range(5):
                    Ds[-1-i,-1-j] = -Ds[i+1,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
            else:
                B[0] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 15:
                print('WARNING: Not enough nodes for volume dissipation interior to be 7th order. Try nen>=15')

            Ds[0, 0] = 1.312940310246155391
            Ds[0, 1] = -3.641478570211597986
            Ds[0, 2] = 4.878552226940346410
            Ds[0, 3] = -3.462909223317039800
            Ds[0, 4] = 0.9128952563421359841

            Ds[1, 0] = 1.312940310246155391
            Ds[1, 1] = -3.641478570211597986
            Ds[1, 2] = 4.878552226940346410
            Ds[1, 3] = -3.462909223317039800
            Ds[1, 4] = 0.9128952563421359841

            Ds[2, 0] = 1.312940310246155391
            Ds[2, 1] = -3.641478570211597986
            Ds[2, 2] = 4.878552226940346410
            Ds[2, 3] = -3.462909223317039800
            Ds[2, 4] = 0.9128952563421359841

            Ds[3, 1] = 0.6698613273129268267
            Ds[3, 2] = -3.147602598437015770
            Ds[3, 3] = 5.194363834975559699
            Ds[3, 4] = -3.651581025368543936
            Ds[3, 5] = 0.9349584615170731804
            
            Ds[4, 2] = 0.9226430743728613724
            Ds[4, 3] = -3.849266357259159405
            Ds[4, 4] = 5.884778804122421640
            Ds[4, 5] = -3.948460667189903404
            Ds[4, 6] = 0.9903051459537797974

            # Interior
            for i in range(5, nen-5):
                Ds[i, i-2] = 1.0
                Ds[i, i-1] = -4.0
                Ds[i, i] = 6.0
                Ds[i, i+1] = -4.0
                Ds[i, i+2] = 1.0

            # copy the bottom as mirror image
            for i in range(5):
                for j in range(7):
                    Ds[-1-i,-1-j] = -Ds[i,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
        
        elif s==5:
            if nen < 17:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 19:
                print('WARNING: Not enough nodes for volume dissipation interior to be 9th order. Try nen>=19')

            Ds[0, 0] = -1.700418754879667790
            Ds[0, 1] = 4.626321026195094882
            Ds[0, 2] = -6.989915497912342108
            Ds[0, 3] = 7.246281859840947029
            Ds[0, 4] = -4.043021768926141434
            Ds[0, 5] = 0.8607531356821094209

            Ds[1, 0] = -1.700418754879667790
            Ds[1, 1] = 4.626321026195094882
            Ds[1, 2] = -6.989915497912342108
            Ds[1, 3] = 7.246281859840947029
            Ds[1, 4] = -4.043021768926141434
            Ds[1, 5] = 0.8607531356821094209

            Ds[2, 0] = -1.700418754879667790
            Ds[2, 1] = 4.626321026195094882
            Ds[2, 2] = -6.989915497912342108
            Ds[2, 3] = 7.246281859840947029
            Ds[2, 4] = -4.043021768926141434
            Ds[2, 5] = 0.8607531356821094209

            Ds[3, 0] = -1.700418754879667790
            Ds[3, 1] = 4.626321026195094882
            Ds[3, 2] = -6.989915497912342108
            Ds[3, 3] = 7.246281859840947029
            Ds[3, 4] = -4.043021768926141434
            Ds[3, 5] = 0.8607531356821094209
            
            Ds[4, 1] = -0.5608598214817355063
            Ds[4, 2] = 3.051946729778099758
            Ds[4, 3] = -7.163110471615289294
            Ds[4, 4] = 8.086043537852282868
            Ds[4, 5] = -4.303765678410547104
            Ds[4, 6] = 0.8897457038771892783

            Ds[5, 2] = -0.6926335215591183273
            Ds[5, 3] = 4.161549707321771927
            Ds[5, 4] = -9.032797275246172236
            Ds[5, 5] = 9.363390428410692813
            Ds[5, 6] = -4.761255925353745274
            Ds[5, 7] = 0.9617465864265710958

            Ds[6, 3] = -0.9614420585367909516
            Ds[6, 4] = 4.914893322792259047
            Ds[6, 5] = -9.914162790826908281
            Ds[6, 6] = 9.942610990136475789
            Ds[6, 7] = -4.978448200390168788
            Ds[6, 8] = 0.9965487368251331853

            # Interior
            for i in range(7, nen-6):
                Ds[i, i-3] = -1.0
                Ds[i, i-2] = 5.0
                Ds[i, i-1] = -10.0
                Ds[i, i] = 10.0
                Ds[i, i+1] = -5.0
                Ds[i, i+2] = 1.0

            # copy the bottom as mirror image
            for i in range(6):
                for j in range(9):
                    Ds[-1-i,-1-j] = -Ds[i+1,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[2] = 0.
                B[-1] = 0.
                B[-2] = 0.
            else:
                B[0] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=3,4,5.')
        
    elif sbp_type.lower() == 'hgt':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)
        
        if s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 11:
                print('WARNING: Not enough nodes for volume dissipation interior to be 5th order. Try nen>=11')
            
            Ds[0, 0] = -1.543001281358810627
            Ds[0, 1] = 3.881106874464771864
            Ds[0, 2] = -3.426488456865127582
            Ds[0, 3] = 1.088382863759166345

            Ds[1, 0] = -1.543001281358810627
            Ds[1, 1] = 3.881106874464771864
            Ds[1, 2] = -3.426488456865127582
            Ds[1, 3] = 1.088382863759166345

            Ds[2, 0] = -1.543001281358810627
            Ds[2, 1] = 3.881106874464771864
            Ds[2, 2] = -3.426488456865127582
            Ds[2, 3] = 1.088382863759166345

            Ds[3, 1] = -1.025603573198747686
            Ds[3, 2] = 3.041730641368677321
            Ds[3, 3] = -3.020721202505921602
            Ds[3, 4] = 1.004594134335991967

            # Interior
            for i in range(4, nen-3):
                Ds[i, i-2] = -1.0
                Ds[i, i-1] = 3.0
                Ds[i, i] = -3.0
                Ds[i, i+1] = 1.0

            # copy the bottom as mirror image
            for i in range(3):
                for j in range(5):
                    Ds[-1-i,-1-j] = -Ds[i+1,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
            else:
                B[0] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 15:
                print('WARNING: Not enough nodes for volume dissipation interior to be 7th order. Try nen>=15')

            Ds[0, 0] = 1.636854763512432634
            Ds[0, 1] = -5.131236817853932557
            Ds[0, 2] = 6.743247576209597163
            Ds[0, 3] = -4.305660493997691977
            Ds[0, 4] = 1.056794972129594737

            Ds[1, 0] = 1.636854763512432634
            Ds[1, 1] = -5.131236817853932557
            Ds[1, 2] = 6.743247576209597163
            Ds[1, 3] = -4.305660493997691977
            Ds[1, 4] = 1.056794972129594737

            Ds[2, 0] = 1.636854763512432634
            Ds[2, 1] = -5.131236817853932557
            Ds[2, 2] = 6.743247576209597163
            Ds[2, 3] = -4.305660493997691977
            Ds[2, 4] = 1.056794972129594737

            Ds[3, 1] = 1.004900473714089334
            Ds[3, 2] = -3.996705852776887225
            Ds[3, 3] = 5.989057349530251599
            Ds[3, 4] = -3.996741446591572833
            Ds[3, 5] = 0.9994894761241191248
            
            Ds[4, 2] = 0.9949762527329816572
            Ds[4, 3] = -3.990345464565561622
            Ds[4, 4] = 5.992750349423123498
            Ds[4, 5] = -3.996776634839660769
            Ds[4, 6] = 0.9993954972491172364

            # Interior
            for i in range(5, nen-5):
                Ds[i, i-2] = 1.0
                Ds[i, i-1] = -4.0
                Ds[i, i] = 6.0
                Ds[i, i+1] = -4.0
                Ds[i, i+2] = 1.0

            # copy the bottom as mirror image
            for i in range(5):
                for j in range(7):
                    Ds[-1-i,-1-j] = -Ds[i,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
        
        elif s==5:
            if nen < 17:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 19:
                print('WARNING: Not enough nodes for volume dissipation interior to be 9th order. Try nen>=19')

            Ds[0, 0] = -1.780052991452513866
            Ds[0, 1] = 6.168946836619043318
            Ds[0, 2] = -10.40188763841277184
            Ds[0, 3] = 10.05767732447850492
            Ds[0, 4] = -5.057685910130604545
            Ds[0, 5] = 1.013002378898342004

            Ds[1, 0] = -1.780052991452513866
            Ds[1, 1] = 6.168946836619043318
            Ds[1, 2] = -10.40188763841277184
            Ds[1, 3] = 10.05767732447850492
            Ds[1, 4] = -5.057685910130604545
            Ds[1, 5] = 1.013002378898342004

            Ds[2, 0] = -1.780052991452513866
            Ds[2, 1] = 6.168946836619043318
            Ds[2, 2] = -10.40188763841277184
            Ds[2, 3] = 10.05767732447850492
            Ds[2, 4] = -5.057685910130604545
            Ds[2, 5] = 1.013002378898342004

            Ds[3, 0] = -1.780052991452513866
            Ds[3, 1] = 6.168946836619043318
            Ds[3, 2] = -10.40188763841277184
            Ds[3, 3] = 10.05767732447850492
            Ds[3, 4] = -5.057685910130604545
            Ds[3, 5] = 1.013002378898342004
            
            Ds[4, 1] = -0.9086944720658002088
            Ds[4, 2] = 4.559126109253252268
            Ds[4, 3] = -9.381687179238169386
            Ds[4, 4] = 9.617796063844260546
            Ds[4, 5] = -4.865693299431110507
            Ds[4, 6] = 0.9791527776375672871

            Ds[5, 2] = -0.9266839908472963824
            Ds[5, 3] = 4.818313164080509632
            Ds[5, 4] = -9.804269093472928734
            Ds[5, 5] = 9.872893119394457865
            Ds[5, 6] = -4.952716307547752514
            Ds[5, 7] = 0.9924631083930101331

            Ds[6, 3] = -0.9942633830800247710
            Ds[6, 4] = 4.987423433012735702
            Ds[6, 5] = -9.987407596091607243
            Ds[6, 6] = 9.991601538833897745
            Ds[6, 7] = -4.996849915666187968
            Ds[6, 8] = 0.9994959229911865349

            # Interior
            for i in range(7, nen-6):
                Ds[i, i-3] = -1.0
                Ds[i, i-2] = 5.0
                Ds[i, i-1] = -10.0
                Ds[i, i] = 10.0
                Ds[i, i+1] = -5.0
                Ds[i, i+2] = 1.0

            # copy the bottom as mirror image
            for i in range(6):
                for j in range(9):
                    Ds[-1-i,-1-j] = -Ds[i+1,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[2] = 0.
                B[-1] = 0.
                B[-2] = 0.
            else:
                B[0] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=3,4,5.')
        
    elif sbp_type.lower() == 'mattsson':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)
        
        if s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 11:
                print('WARNING: Not enough nodes for volume dissipation interior to be 5th order. Try nen>=11')
            
            Ds[0, 0] = -1.727746398798953985
            Ds[0, 1] = 3.702197671856910570
            Ds[0, 2] = -2.987030659701329605
            Ds[0, 3] = 1.012579386643373020

            Ds[1, 0] = -1.727746398798953985
            Ds[1, 1] = 3.702197671856910570
            Ds[1, 2] = -2.987030659701329605
            Ds[1, 3] = 1.012579386643373020

            Ds[2, 0] = -1.727746398798953985
            Ds[2, 1] = 3.702197671856910570
            Ds[2, 2] = -2.987030659701329605
            Ds[2, 3] = 1.012579386643373020

            Ds[3, 1] = -0.8173849542405728450
            Ds[3, 2] = 2.691630521667999803
            Ds[3, 3] = -2.837461614650824770
            Ds[3, 4] = 0.9632160472233978121

            # Interior
            for i in range(4, nen-3):
                Ds[i, i-2] = -1.0
                Ds[i, i-1] = 3.0
                Ds[i, i] = -3.0
                Ds[i, i+1] = 1.0

            # copy the bottom as mirror image
            for i in range(3):
                for j in range(5):
                    Ds[-1-i,-1-j] = -Ds[i+1,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
            else:
                B[0] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 15:
                print('WARNING: Not enough nodes for volume dissipation interior to be 7th order. Try nen>=15')

            Ds[0, 0] = 5.730211159355025436
            Ds[0, 1] = -12.52199438470801005
            Ds[0, 2] = 11.41940257258177371
            Ds[0, 3] = -5.944279710708495014
            Ds[0, 4] = 1.316660363479705916

            Ds[1, 0] = 5.730211159355025436
            Ds[1, 1] = -12.52199438470801005
            Ds[1, 2] = 11.41940257258177371
            Ds[1, 3] = -5.944279710708495014
            Ds[1, 4] = 1.316660363479705916

            Ds[2, 0] = 5.730211159355025436
            Ds[2, 1] = -12.52199438470801005
            Ds[2, 2] = 11.41940257258177371
            Ds[2, 3] = -5.944279710708495014
            Ds[2, 4] = 1.316660363479705916

            Ds[3, 1] = 1.444151388124975588
            Ds[3, 2] = -4.929248582142859209
            Ds[3, 3] = 6.728613732200695153
            Ds[3, 4] = -4.297441697303572535
            Ds[3, 5] = 1.0539251591207615 #TODO: update this value to match the paper
            
            Ds[4, 2] = 1.046607535776803374
            Ds[4, 3] = -4.088738042770657411
            Ds[4, 4] = 6.065823402094199897
            Ds[4, 5] = -4.029148254415713867
            Ds[4, 6] = 1.005455359315368007

            # Interior
            for i in range(5, nen-5):
                Ds[i, i-2] = 1.0
                Ds[i, i-1] = -4.0
                Ds[i, i] = 6.0
                Ds[i, i+1] = -4.0
                Ds[i, i+2] = 1.0

            # copy the bottom as mirror image
            for i in range(5):
                for j in range(7):
                    Ds[-1-i,-1-j] = -Ds[i,j]

            if boundary_fix:
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=4,5.')

    return Ds,B

def make_dcp_diss_op2(sbp_type, s, nen, boundary_fix=True):
    ''' make the relevant operators according to DCP implementation in the paper 
        NOTE: is exactly the same as above if only B is used, but if H is used,
         then the dissipation operators will be mirror images of each other '''
    if sbp_type.lower() == 'csbp':
        # Initialize the matrix as a dense NumPy array
        Ds = np.zeros((nen, nen))
        B = np.ones(nen)

        if s==1:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 3:
                print('WARNING: Not enough nodes for volume dissipation interior to be 3rd order. Try nen>=5')

            # Row 1
            Ds[0, 0] = -1.0
            Ds[0, 1] = 1.0
            # Interior rows
            for i in range(1, nen):
                Ds[i, i-1] = -1
                Ds[i, i] = 1
            
            if boundary_fix:
                # correct boundary values
                B[0] = 0.

        elif s==2:
            if nen < 3:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 5:
                print('WARNING: Not enough nodes for volume dissipation interior to be 3rd order. Try nen>=5')

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
            
            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[-1] = 0.
        
        elif s==3:
            if nen < 9:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 7:
                print('WARNING: Not enough nodes for volume dissipation interior to be 5th order. Try nen>=11')
            
            Ds[0, 0] = -1.0
            Ds[0, 1] = 3.0
            Ds[0, 2] = -3.0
            Ds[0, 3] = 1.0

            Ds[1, 0] = -1.0
            Ds[1, 1] = 3.0
            Ds[1, 2] = -3.0
            Ds[1, 3] = 1.0

            # Interior half-nodes
            for i in range(2, nen-1):
                Ds[i, i-2] = -1.0
                Ds[i, i-1] = 3.0
                Ds[i, i] = -3.0
                Ds[i, i+1] = 1.0

            Ds[nen-1, nen-4] = -1.0
            Ds[nen-1, nen-3] = 3.0
            Ds[nen-1, nen-2] = -3.0
            Ds[nen-1, nen-1] = 1.0

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
            else:
                B[0] = 0.

        elif s==4:
            if nen < 13:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 9:
                print('WARNING: Not enough nodes for volume dissipation interior to be 7th order. Try nen>=15')

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

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[-1] = 0.
                B[-2] = 0.
        
        elif s==5:
            if nen < 17:
                raise ValueError(f"Invalid number of nodes. nen = {nen}")
            if nen < 11:
                print('WARNING: Not enough nodes for volume dissipation interior to be 9th order. Try nen>=19')

            Ds[0, 0] = -1.0
            Ds[0, 1] = 5.0
            Ds[0, 2] = -10.0
            Ds[0, 3] = 10.0
            Ds[0, 4] = -5.0
            Ds[0, 5] = 1.0

            Ds[1, 0] = -1.0
            Ds[1, 1] = 5.0
            Ds[1, 2] = -10.0
            Ds[1, 3] = 10.0
            Ds[1, 4] = -5.0
            Ds[1, 5] = 1.0

            Ds[2, 0] = -1.0
            Ds[2, 1] = 5.0
            Ds[2, 2] = -10.0
            Ds[2, 3] = 10.0
            Ds[2, 4] = -5.0
            Ds[2, 5] = 1.0

            # Interior half-nodes
            for i in range(3, nen-2):
                Ds[i, i-3] = -1.0
                Ds[i, i-2] = 5.0
                Ds[i, i-1] = -10.0
                Ds[i, i] = 10.0
                Ds[i, i+1] = -5.0
                Ds[i, i+2] = 1.0

            Ds[nen-2, nen-6] = -1.0
            Ds[nen-2, nen-5] = 5.0
            Ds[nen-2, nen-4] = -10.0
            Ds[nen-2, nen-3] = 10.0
            Ds[nen-2, nen-2] = -5.0
            Ds[nen-2, nen-1] = 1.0

            Ds[nen-1, nen-6] = -1.0
            Ds[nen-1, nen-5] = 5.0
            Ds[nen-1, nen-4] = -10.0
            Ds[nen-1, nen-3] = 10.0
            Ds[nen-1, nen-2] = -5.0
            Ds[nen-1, nen-1] = 1.0

            if boundary_fix:
                # correct boundary values
                B[0] = 0.
                B[1] = 0.
                B[2] = 0.
                B[-1] = 0.
                B[-2] = 0.
            else:
                B[0] = 0.

        else:
            raise Exception('Invalid choice of s. Only coded up s=1,2,3,4,5.')
        
    else:
        raise Exception('Invalid choice of sbp_type. Only coded up CSBP.')
    
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