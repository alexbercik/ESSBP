# ESSBP
Entropy-Stable Summation by Parts. See the driver files for examples on how to run.

### Install Instructions

In general i reccomend working in a virtual environemnt as follows
    
    python3 -m venv ~/.virtualenvs/ESSBP
    source ~/.virtualenvs/ESSBP/bin/activate

Make sure latex and julia are installed, too (Julia is only necessary for getting upwind operators)
The following modules are needed
    
    pip install --upgrade pip
    pip install numpy==2.0.0 matplotlib==3.9.2 scipy==1.14.0 numba==0.60.0 tabulate==0.9.0 julia==0.6.2 sympy==1.13.3 tex==1.8 latex==0.7.0

#### Installing Julia
Make sure you run the following in the command line (in your virtual environment) to make a julia environment called upwindOP
    
    julia
    using Pkg
    Pkg.activate(joinpath(ENV["HOME"], "julia_environments", "upwindOP"))
    Pkg.add("PyCall")
    Pkg.build("PyCall")
    exit()

    python
    import julia
    julia.install()
    exit()

This should work, but if running on a cluster where intel and gcc are incompatible, julia might struggle to find the libstdc++.so.6 library. I got around this problem by loading python with:

    LD_PRELOAD=/scinet/niagara/software/2022a/opt/base/gcc/11.3.0/lib64/libstdc++.so.6 python

It may also help to include the options

    MPLCONFIGDIR=$SCRATCH/matplotlib JULIA_DEPOT_PATH=$SCRATCH/julia

Finally to avoid compiling the julia environment every time the code is run, use

    JULIA_UPWIND_ENV_READY=true JULIA_PKG_OFFLINE=true

