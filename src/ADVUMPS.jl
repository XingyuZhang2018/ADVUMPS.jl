module ADVUMPS

using Zygote
using OMEinsum

export vumps, vumps_env
export Z,magnetisation, energy
export hamiltonian, model_tensor, mag_tensor
export Ising, TFIsing, Heisenberg
export optimiseipeps

include("hamiltonianmodels.jl")
include("vumps.jl")
include("exampletensors.jl")
include("fixedpoint.jl")
include("autodiff.jl")
include("ipeps.jl")
include("variationalipeps.jl")

end
