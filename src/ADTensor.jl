module ADTensor

using Zygote
using OMEinsum

export eig,lefteig,righteig,num_grad
export Ising, HamiltonianModel
export model_tensor

include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("vumps.jl")
include("autodiff.jl")

end
