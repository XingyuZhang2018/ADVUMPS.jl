module ADVUMPS

using Zygote
using OMEinsum

export vumps
export Z,magnetisation,energy
export hamiltonian
export Ising

include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("vumps.jl")
include("autodiff.jl")

end
