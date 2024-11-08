import numpy as np
from julia import Main

julia_code = """
module UpwindOperators

# Check if virtual environment has already been set up & compiled. If not, activate.
if !haskey(ENV, "JULIA_UPWIND_ENV_READY")
    using Pkg
    Pkg.activate(joinpath(ENV["HOME"], "julia_environments", "upwindOP"))

    # Precompile packages
    Pkg.add("SummationByPartsOperators")
    Pkg.precompile()
    ENV["JULIA_UPWIND_READY"] = "true"
end

using SummationByPartsOperators
using LinearAlgebra

function getOps(p,n)
    Dup = upwind_operators(Mattsson2017, derivative_order=1, accuracy_order=p,
                            xmin=0.0, xmax=1.0, N=n)
    H = mass_matrix(Dup)
    D = 0.5 * (Matrix(Dup.plus) + Matrix(Dup.minus))
    diss = - 0.5 * (Matrix(Dup.plus) - Matrix(Dup.minus))
    Q = 0.5 * H * (Matrix(Dup.plus) + Matrix(Dup.minus))
    x = SummationByPartsOperators.grid(Dup)

    return Matrix(D), Matrix(Dup.plus), Matrix(Dup.minus), Matrix(Q), Matrix(diss), Matrix(H), x 
end

end # module UpwindOperators
"""

# Evaluate the Julia code
Main.eval(julia_code)

# Import the function from the module
Main.eval("using .UpwindOperators: getOps")

# Access the function
getOps = Main.getOps

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


