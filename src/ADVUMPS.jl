module ADVUMPS

using Zygote
using OMEinsum

export vumps, vumps_env
export Z,magnetisation, energy
export hamiltonian, model_tensor, mag_tensor
export Ising, TFIsing, Heisenberg
export init_ipeps, optimiseipeps

include("hamiltonianmodels.jl")
include("cuda_patch.jl")
include("exampletensors.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("fixedpoint.jl")
include("exampleobs.jl")
include("autodiff.jl")
include("ipeps.jl")
include("variationalipeps.jl")

end
