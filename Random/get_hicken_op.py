import numpy as np
import julia
from julia import Main

julia_code = """

using Pkg

# Check if virtual environment has already been set up & compiled. If not, activate.
if !haskey(ENV, "JULIA_HICKEN_ENV_READY")
    Pkg.activate("//Users/alex/julia_environments/hicken")

    # this is to ensure we use a local version of the code
    Pkg.develop(path="//Users/alex/Desktop/UTIAS/Other_Peoples_Things/SummationByParts.jl")

    # Precompile packages
    Pkg.precompile()
    ENV["JULIA_ENV_READY"] = "true"
end

using SummationByParts, LinearAlgebra
using SummationByParts.Cubature, SummationByParts.SymCubatures

function create_Rmat(perm,face,Rraw,nvolnodes)
    # Create a zero matrix of size `length of column` x `total elements in a`
    R = zeros(Float64, (size(Rraw,1),nvolnodes))
    # Fill in the matrix such that each row corresponds to an index in perm_mat[:, face]
    for row in 1:size(Rraw, 1)
        for col in 1:size(Rraw, 2)
            R[row, perm[col, face]] = Rraw[row, col]
        end
    end
    return R
end

function getOps2D(p,q,op_type,fac_nodes)

    if q != 2 * p && q != 2 * p - 1
        println("WARNING (getOps2D): q must be equal to 2*p or 2*p-1. Changing q from $q to $(2 * p).")
        q = 2 * p
    end

    if op_type == "Omega"
        sbp = SummationByParts.getTriSBPOmega(degree=p)
    elseif op_type == "Gamma"
        sbp = SummationByParts.getTriSBPGamma(degree=p)
    elseif op_type == "DiagE"
        if fac_nodes == "lgl"
            sbp = SummationByParts.getTriSBPDiagE(degree=p,vertices=true,quad_degree=q)
        elseif fac_nodes == "lg"
            sbp = SummationByParts.getTriSBPDiagE(degree=p,vertices=false,quad_degree=q)
        else
            println("Unknown fac_nodes: $fac_nodes . Try 'lgl' or 'lg'.")
            return nothing
        end

    else
        println("Unknown op_type: $op_type . Try 'DiagE', 'Gamma', or 'Omega'.")
        return nothing
    end

    Qx = sbp.Q[:,:,1];
    Qy = sbp.Q[:,:,2];
    H = diagm(sbp.w);
    Ex = sbp.E[:,:,1];
    Ey = sbp.E[:,:,2];
    Dx = inv(H)*Qx;
    Dy = inv(H)*Qy;
    #Sx = Qx - 0.5.*Ex;
    #Sy = Qy - 0.5.*Ey;

    if norm(Qx+Qx' - Ex) > 1e-14 || norm(Qy+Qy' - Ey) > 1e-14
        println("Something went wrong creating SBP operators.")
        return nothing
    end
    
    if op_type == "DiagE"
        if fac_nodes == "lgl"
            sbpface, tmp = Cubature.quadrature(2*p, Float64, internal=false)
        else
            sbpface, tmp = Cubature.quadrature(2*p, Float64, internal=true)
        end
    else
        sbpface, tmp = Cubature.quadrature(2*p, Float64, internal=true)
    end

    xy = SymCubatures.calcnodes(sbp.cub, sbp.vtx);
    B = SymCubatures.calcweights(sbpface)
    N = [ 0. 1. -1.;
         -1. 1. 0.] # do this manually based on xv

    facevtx = SymCubatures.getfacevertexindices(sbp.cub)
    # sbp.vtx = [-1. -1.; 1. -1.; -1. 1.]
    xf = zeros(Float64, (2,sbpface.numnodes,3))
    for f in 1:3
        # f=1 bottom face y=-1, f=2 diagonal face, f=3 left face x=-1
        xf[:,:,f] = SymCubatures.calcnodes(sbpface, sbp.vtx[facevtx[:,f],:]) 
    end

    Rraw,perm=SummationByParts.buildfacereconstruction(sbpface, sbp.cub, sbp.vtx, p);

    R1 = create_Rmat(perm,1,Rraw,sbp.numnodes)
    R2 = create_Rmat(perm,2,Rraw,sbp.numnodes)
    R3 = create_Rmat(perm,3,Rraw,sbp.numnodes)
    R = zeros(Float64, (size(R1,1),size(R1,2),3))
    R[:,:,1] = R1
    R[:,:,2] = R2
    R[:,:,3] = R3

    # quick test
    t1 = maximum(abs.(R1*xy[1,:] - xf[1,:,1] ))
    t2 = maximum(abs.(R1*xy[2,:] - xf[2,:,1] ))
    t3 = maximum(abs.(R2*xy[1,:] - xf[1,:,2] ))
    t4 = maximum(abs.(R2*xy[2,:] - xf[2,:,2] ))
    t5 = maximum(abs.(R3*xy[1,:] - xf[1,:,3] ))
    t6 = maximum(abs.(R3*xy[2,:] - xf[2,:,3] ))
    if t1 > 1e-14 || t2 > 1e-14 || t3 > 1e-14 || t4 > 1e-14 || t5 > 1e-14 || t6 > 1e-14
        println("Something went wrong creating SBP operators. Extrapolation fails.")
        return nothing
    end

    E1x = ((R1' .* B') .* N[1,1]) * R1
    E1y = ((R1' .* B') .* N[2,1]) * R1
    E2x = ((R2' .* B') .* N[1,2]) * R2
    E2y = ((R2' .* B') .* N[2,2]) * R2
    E3x = ((R3' .* B') .* N[1,3]) * R3
    E3y = ((R3' .* B') .* N[2,3]) * R3
    Ex = E1x + E2x + E3x
    Ey = E1y + E2y + E3y

    t1 = maximum(abs.(Ex-sbp.E[:,:,1]))
    t2 = maximum(abs.(Ey-sbp.E[:,:,2]))
    if t1 > 1e-15 || t2 > 1e-15
        println("Something went wrong creating SBP operators. E decomposition fails.")
        return nothing
    end

    return Dx, Dy, Qx, Qy, sbp.w, Ex, Ey, xy, xf, B, N, R
end


function getOps3D(p,q,op_type,fac_op_type)

    if q != 2 * p && q != 2 * p - 1
        println("WARNING (getOps2D): q must be equal to 2*p or 2*p-1. Changing q from $q to $(2 * p).")
        q = 2 * p
    end
    
    if op_type == "Omega"
        faceoper =:Omega
        sbp = SummationByParts.getTetSBPOmega(degree=p)
        sbpface_cub, sbpface_vtx = SummationByParts.getTriCubatureOmega(2*p)
    elseif op_type == "Gamma"
        faceoper =:Omega
        sbp = SummationByParts.getTetSBPGamma(degree=p)
        sbpface_cub, sbpface_vtx = SummationByParts.getTriCubatureOmega(2*p)
    elseif op_type == "DiagE"
        if fac_op_type == "DiagE"
            faceoper =:DiagE
        elseif fac_op_type == "Omega"
            faceoper =:Omega
        elseif fac_op_type == "Gamma"
            faceoper =:Gamma
        else
            println("Unknown fac_op_type: $fac_op_type . Try 'DiagE' or 'Omega'.")
            return nothing
        end
        sbp = SummationByParts.getTetSBPDiagE(degree=p, faceopertype=faceoper, cubdegree=q)
        sbpface_cub, sbpface_vtx = SummationByParts.getTriCubatureForTetFaceDiagE(2*p, faceopertype=faceoper)
    
    else
        println("Unknown op_type: $op_type . Try 'DiagE', 'Gamma', or 'Omega'.")
        return nothing
    end
    
    Qx = sbp.Q[:,:,1];
    Qy = sbp.Q[:,:,2];
    Qz = sbp.Q[:,:,3];
    H = diagm(sbp.w);
    Ex = sbp.E[:,:,1];
    Ey = sbp.E[:,:,2];
    Ez = sbp.E[:,:,3];
    Dx = inv(H)*Qx;
    Dy = inv(H)*Qy;
    Dz = inv(H)*Qz;
    #Sx = Qx - 0.5.*Ex;
    #Sy = Qy - 0.5.*Ey;
    #Sz = Qz - 0.5.*Ez;
    
    if norm(Qx+Qx' - Ex) > 1e-14 || norm(Qy+Qy' - Ey) > 1e-14 || norm(Qz+Qz' - Ez) > 1e-14
        println("Something went wrong creating SBP operators.")
        exit()
    end
    
    
    xyz = SymCubatures.calcnodes(sbp.cub, sbp.vtx);
    B = SymCubatures.calcweights(sbpface_cub)
    N = [ 0. 0. 1. -1.;
         0. -1. 1. 0.;
         -1. 0. 1. 0.] # do this manually based on sbp.vtx
    
    facevtx = SymCubatures.getfacevertexindices(sbp.cub)
    # sbp.vtx = [-1. -1. -1.; 1. -1. -1.; -1. 1. -1.; -1. -1. 1.]
    xyf = zeros(Float64, (3,sbpface_cub.numnodes,4))
    for f in 1:4
        # f=1 bottom face z=-1, f=2 front face y=-1, f=3 diagonal face, f=4 left face x=-1
        xyf[:,:,f] = SymCubatures.calcnodes(sbpface_cub, sbp.vtx[facevtx[:,f],:]) 
    end
             
    Rraw,perm=SummationByParts.buildfacereconstruction(sbpface_cub, sbp.cub, sbp.vtx, p, faceopertype=faceoper);
    
    R1 = create_Rmat(perm,1,Rraw,sbp.numnodes)
    R2 = create_Rmat(perm,2,Rraw,sbp.numnodes)
    R3 = create_Rmat(perm,3,Rraw,sbp.numnodes)
    R4 = create_Rmat(perm,4,Rraw,sbp.numnodes)
    R = zeros(Float64, (size(R1,1),size(R1,2),4))
    R[:,:,1] = R1
    R[:,:,2] = R2
    R[:,:,3] = R3
    R[:,:,4] = R4
    
    # quick test
    t1 = maximum(abs.(R1*xyz[1,:] - xyf[1,:,1] ))
    t2 = maximum(abs.(R1*xyz[2,:] - xyf[2,:,1] ))
    t3 = maximum(abs.(R1*xyz[3,:] - xyf[3,:,1] ))
    t4 = maximum(abs.(R2*xyz[1,:] - xyf[1,:,2] ))
    t5 = maximum(abs.(R2*xyz[2,:] - xyf[2,:,2] ))
    t6 = maximum(abs.(R2*xyz[3,:] - xyf[3,:,2] ))
    if t1 > 1e-14 || t2 > 1e-14 || t3 > 1e-14 || t4 > 1e-14 || t5 > 1e-14 || t6 > 1e-14
        println("Something went wrong creating SBP operators. Extrapolation fails.")
        return nothing
    end
    t1 = maximum(abs.(R3*xyz[1,:] - xyf[1,:,3] ))
    t2 = maximum(abs.(R3*xyz[2,:] - xyf[2,:,3] ))
    t3 = maximum(abs.(R3*xyz[3,:] - xyf[3,:,3] ))
    t4 = maximum(abs.(R4*xyz[1,:] - xyf[1,:,4] ))
    t5 = maximum(abs.(R4*xyz[2,:] - xyf[2,:,4] ))
    t6 = maximum(abs.(R4*xyz[3,:] - xyf[3,:,4] ))
    if t1 > 1e-14 || t2 > 1e-14 || t3 > 1e-14 || t4 > 1e-14 || t5 > 1e-14 || t6 > 1e-14
        println("Something went wrong creating SBP operators. Extrapolation fails.")
        return nothing
    end
    
    E1x = ((R1' .* B') .* N[1,1]) * R1
    E1y = ((R1' .* B') .* N[2,1]) * R1
    E1z = ((R1' .* B') .* N[3,1]) * R1
    E2x = ((R2' .* B') .* N[1,2]) * R2
    E2y = ((R2' .* B') .* N[2,2]) * R2
    E2z = ((R2' .* B') .* N[3,2]) * R2
    E3x = ((R3' .* B') .* N[1,3]) * R3
    E3y = ((R3' .* B') .* N[2,3]) * R3
    E3z = ((R3' .* B') .* N[3,3]) * R3
    E4x = ((R4' .* B') .* N[1,4]) * R4
    E4y = ((R4' .* B') .* N[2,4]) * R4
    E4z = ((R4' .* B') .* N[3,4]) * R4
    Ex = E1x + E2x + E3x + E4x
    Ey = E1y + E2y + E3y + E4y
    Ez = E1z + E2z + E3z + E4z
    
    t1 = maximum(abs.(Ex-sbp.E[:,:,1]))
    t2 = maximum(abs.(Ey-sbp.E[:,:,2]))
    t3 = maximum(abs.(Ez-sbp.E[:,:,3]))
    if t1 > 1e-15 || t2 > 1e-15 || t3 > 1e-15
        println("Something went wrong creating SBP operators. E decomposition fails.")
        return nothing
    end

    return Dx, Dy, Dz, Qx, Qy, Qz, sbp.w, Ex, Ey, Ez, xyz, xyf, B, N, R
end
"""

# Evaluate the Julia code
Main.eval(julia_code)
getOps2D = Main.getOps2D
getOps3D = Main.getOps3D

# Call the function
if __name__ == '__main__':
    p = 2
    q = 4
    op_type = "DiagE"
    fac_op_type = "DiagE" #3D
    fac_nodes = "lg" # 2D

    Dx, Dy, Qx, Qy, H, Ex, Ey, xy, xf, B, N, R = getOps2D(p, q, op_type, fac_nodes)
    print('----- 2D -----')
    print('Test Q + Q.T = E :', np.max(abs(Qx + Qx.T - Ex)), np.max(abs(Qy + Qy.T - Ey)))
    print('Test R @ x = xf :', np.max(abs(R[:,:,0]@(xy[0,:]*xy[1,:]) - xf[0,:,0]*xf[1,:,0])), np.max(abs(R[:,:,1]@(xy[0,:]*xy[1,:]) - xf[0,:,1]*xf[1,:,1])), np.max(abs(R[:,:,2]@(xy[0,:]*xy[1,:]) - xf[0,:,2]*xf[1,:,2])))
    print('Test E = R.T@B@N@R :',   np.max(abs( R[:,:,0].T@np.diag(B)*N[0,0]@R[:,:,0] +\
                                                R[:,:,1].T@np.diag(B)*N[0,1]@R[:,:,1] +\
                                                R[:,:,2].T@np.diag(B)*N[0,2]@R[:,:,2] - Ex)), \
                                    np.max(abs( R[:,:,0].T@np.diag(B)*N[1,0]@R[:,:,0] +\
                                                R[:,:,1].T@np.diag(B)*N[1,1]@R[:,:,1] +\
                                                R[:,:,2].T@np.diag(B)*N[1,2]@R[:,:,2] - Ey)) )
    
    Dx, Dy, Dz, Qx, Qy, Qz, H, Ex, Ey, Ez, xyz, xyf, B, N, R = getOps3D(p, q, op_type, fac_op_type)
    print('----- 3D -----')
    print('Test Q + Q.T = E :', np.max(abs(Qx + Qx.T - Ex)), np.max(abs(Qy + Qy.T - Ey)), np.max(abs(Qz + Qz.T - Ez)))
    print('Test R @ x = xf :', np.max(abs(R[:,:,0]@(xyz[1,:]*xyz[2,:]) - xyf[1,:,0]*xyf[2,:,0])), np.max(abs(R[:,:,1]@(xyz[0,:]*xyz[1,:]) - xyf[0,:,1]*xyf[1,:,1])), np.max(abs(R[:,:,2]@(xyz[0,:]*xyz[1,:]) - xyf[0,:,2]*xyf[1,:,2])), np.max(abs(R[:,:,3]@(xyz[0,:]*xyz[1,:]) - xyf[0,:,3]*xyf[1,:,3])))
    print('Test E = R.T@B@N@R :',   np.max(abs( R[:,:,0].T@np.diag(B)*N[0,0]@R[:,:,0] +\
                                                R[:,:,1].T@np.diag(B)*N[0,1]@R[:,:,1] +\
                                                R[:,:,2].T@np.diag(B)*N[0,2]@R[:,:,2] +\
                                                R[:,:,3].T@np.diag(B)*N[0,3]@R[:,:,3] - Ex)), \
                                    np.max(abs( R[:,:,0].T@np.diag(B)*N[1,0]@R[:,:,0] +\
                                                R[:,:,1].T@np.diag(B)*N[1,1]@R[:,:,1] +\
                                                R[:,:,2].T@np.diag(B)*N[1,2]@R[:,:,2] +\
                                                R[:,:,3].T@np.diag(B)*N[1,3]@R[:,:,3] - Ey)), \
                                    np.max(abs( R[:,:,0].T@np.diag(B)*N[2,0]@R[:,:,0] +\
                                                R[:,:,1].T@np.diag(B)*N[2,1]@R[:,:,1] +\
                                                R[:,:,2].T@np.diag(B)*N[2,2]@R[:,:,2] +\
                                                R[:,:,3].T@np.diag(B)*N[2,3]@R[:,:,3] - Ez)) )