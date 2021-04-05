module ADVUMPS

using Zygote
using OMEinsum

export vumps, vumps_env
export Z,magnetisation,energy
export hamiltonian,model_tensor,mag_tensor
export Ising

include("hamiltonianmodels.jl")
include("vumps.jl")
include("exampletensors.jl")
include("fixedpoint.jl")
include("autodiff.jl")

end
