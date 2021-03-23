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
    magnetisation(model<:HamiltonianModel, β, χ)

return the magnetisation of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `χ` as calculated with
ctmrg. Requires that `mag_tensor` and `model_tensor` are defined for `model`.
"""
function magnetisation(model::MT, β, χ) where {MT <: HamiltonianModel}
    A = rand(ComplexF64,χ,2,χ)
    M = model_tensor(model, β)
    Mag = mag_tensor(model, β)
    λ, AL, C, AR, FL, FR = vumps(A, M;verbose = false, tol = 1e-10, maxit = 100)

    AC = ein"asc,cb -> asb"(AL,C)
    mag = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,Mag,FR,conj(AC))[]
    λ = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,M,FR,conj(AC))[]

    return abs(mag/λ)
end

"""
    magofβ(::Ising,β)

return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(::Ising, β) = β > isingβc ? (1-sinh(2*β)^-4)^(1/8) : 0.

"""
    magofdβ(::Ising,β)

return the analytical result for the derivative of magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofdβ(::Ising, β) = β > isingβc ? (coth(2*β)*csch(2*β)^4)/(1-csch(2*β)^4)^(7/8) : 0.
