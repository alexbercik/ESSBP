module GetUpwindOperators

using Pkg
Pkg.activate(joinpath(ENV["HOME"], "julia_environments", "upwindOP"))

# Check if virtual environment has already been set up & compiled. If not, activate.
# (set this in the environment, or when calling python, i.e. JULIA_UPWIND_ENV_READY=true)
if !haskey(ENV, "JULIA_UPWIND_ENV_READY")

    # Ensure necessary packages are added and precompiled
    Pkg.add("SummationByPartsOperators")
    Pkg.precompile()

    # Set environment variable to indicate setup is complete
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

end # module GetUpwindOperators