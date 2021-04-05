using JLD2
using FileIO

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

"""
    vumps_env(model<:HamiltonianModel, β, D)

return the vumps environment of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Save `env` in file `./data/model_β_D.jld2`. Requires that `model_tensor` are defined for `model`.
"""
function vumps_env(model::MT, β, D) where {MT <: HamiltonianModel}
    M = model_tensor(model, β)
    mkpath("./data/")
    chkp_file = "./data/$(model)_β$(β)_D$(D).jld2"
    if isfile(chkp_file)                               # if backward go to this way will make mistake!!!
        rt = SquareVUMPSRuntime(M, chkp_file, D)   
    else
        rt = SquareVUMPSRuntime(M, Val(:random), D)
    end
    env = vumps(rt; tol=1e-10, maxit=100)
    # save(chkp_file, "env", env)
    return env
end


"""
    Z(env::SquareVUMPSRuntime)

return the partition function of the `env`.
"""
function Z(env::SquareVUMPSRuntime)
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    AC = ein"asc,cb -> asb"(AL,C)
    z = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,M,FR,conj(AC))[]
    λ = ein"αcβ,βη,ηcγ,αγ ->"(FL,C,FR,conj(C))[]
    return z/λ
end

"""
    magnetisation(env::SquareVUMPSRuntime, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env::SquareVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Mag = mag_tensor(model, β)
    AC = ein"asc,cb -> asb"(AL,C)
    mag = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,Mag,FR,conj(AC))[]
    λ = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,M,FR,conj(AC))[]

    return abs(mag/λ)
end

"""
    energy(env::SquareVUMPSRuntime, model::MT, β)

return the energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env::SquareVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ene = energy_tensor(model, β)
    AC = ein"asc,cb -> asb"(AL,C)
    energy = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,Ene,FR,conj(AC))[]
    λ = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL,AC,M,FR,conj(AC))[]

    return energy/λ*2 # factor 2 for counting horizontal and vertical links
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
