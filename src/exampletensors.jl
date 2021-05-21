# tensor for classical 2-d model
const isingβc = log(1+sqrt(2))/2

"""
    tensorfromclassical(h::Matrix)

given a classical 2-body hamiltonian `h`, return the corresponding tensor
for use in e.g. `trg` for a two-dimensional square-lattice.

# Example
```julia
julia> model_tensor(Ising(),β) ≈ tensorfromclassical([β -β; -β β])

true
```
"""
function tensorfromclassical(ham::Matrix)
    wboltzmann = exp.(ham)
    q = sqrt(wboltzmann)
    ein"ij,ik,il,im -> jklm"(q,q,q,q)
end

"""
    model_tensor(::Ising,β)

return the isingtensor at inverse temperature `β` for a two-dimensional
square lattice tensor-network.
"""
function model_tensor(::Ising, β::Real)
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
end

"""
    mag_tensor(::Ising,β)

return the operator for the magnetisation at inverse temperature `β`
at a site in the two-dimensional ising model on a square lattice in tensor-network form.
"""
function mag_tensor(::Ising, β)
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
end

"""
    energy_tensor(::Ising,β)

return the operator for the energy at inverse temperature `β`
at a site in the two-dimensional ising model on a square lattice in tensor-network form.
"""
function energy_tensor(::Ising, β)
    ham = [-1 1;1 -1]
    wboltzmann = exp.(-β .* ham)
    wenergy = ham .* wboltzmann
    qb = sqrt(wboltzmann)
    ein"as,si,bi,ci,di -> abcd"(qb^(-1),wenergy,qb,qb,qb)
end