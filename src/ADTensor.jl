module ADTensor

using Zygote
using OMEinsum

export num_grad
export Ising, HamiltonianModel
export model_tensor

include("autodiff.jl")
include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("vumps.jl")


end
