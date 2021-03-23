abstract type HamiltonianModel end


@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

struct Ising <: HamiltonianModel end