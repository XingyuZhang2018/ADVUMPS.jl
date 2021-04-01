module ADVUMPS

using Zygote
using OMEinsum

export vumps
export Z,magnetisation,energy
export hamiltonian,model_tensor,mag_tensor
export Ising

include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("fixedpoint.jl")
include("vumps.jl")
include("autodiff.jl")

end
