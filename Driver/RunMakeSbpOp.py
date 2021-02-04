
# Add the root folder of ECO to the search path
import os
from sys import path

n_nested_folder = 1
folder_path, _ = os.path.split(__file__)

for i in range(n_nested_folder):
    folder_path, _ = os.path.split(folder_path)

path.append(folder_path)

# Import the required modules
from Source.Disc.MakeSbpOp import MakeSbpOp

import numpy as np
np.printoptions(precision=2)

# Set options
dim = 1 # 1D or 2D (will add 3D another time)
p = 4
pcub = 2*p-1
sbp_fam = 'Rdn1' # Rd, Rdn1, R0
basis_type = 'monomial' # Optional arg, does not impact the const. of sbp op

# Construct the SBP operators
sbp = MakeSbpOp(p, dim, sbp_fam, basis_type=basis_type)

xy = sbp.xy
d1 = sbp.dd[0,:,:]

# print(f'xy = \n {xy}')
# print('sbp.hh = \n', sbp.hh)
# print('sbp.dd = \n', d1)

# d2 = d1 @ d1
# van_der2 = d2 @ sbp.van
# print(f'van_der2 = \n {van_der2}')

# d3 = d2 @ d1
# van_der3 = d3 @ sbp.van
# print(f'van_der3 = \n {van_der3}')


# sbp.__dict__.keys() # prints attributes of class isntance sbp



