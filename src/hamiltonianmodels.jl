using ITensors
abstract type HamiltonianModel end

const σx = Float64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = Float64[1 0; 0 -1]
const id2 = Float64[1 0; 0 1]

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

struct Ising <: HamiltonianModel end

struct diaglocal{T<:Vector} <: HamiltonianModel 
    diag::T
end

"""
    diaglocal(diag::Vector)

return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function hamiltonian(model::diaglocal)
    diag = model.diag
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

@doc raw"
    TFIsing(hx::Real)

return a struct representing the transverse field ising model with magnetisation `hx`.
"
struct TFIsing{T<:Real} <: HamiltonianModel
    hx::T
end

"""
    hamiltonian(model::TFIsing)

return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    hx = model.hx
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

@doc raw"
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`..
"
struct Heisenberg{T<:Real} <: HamiltonianModel
    Jz::T
    Jx::T
    Jy::T
end
Heisenberg() = Heisenberg(1.0,1.0,1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    h = model.Jz * ein"ij,kl -> ijkl"(σz, σz) -
        model.Jx * ein"ij,kl -> ijkl"(σx, σx) -
        model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    h ./ 2
end


@doc raw"
    spinless_fermions(γ::Real,λ::Real)

return a struct representing the free fermions model with paring strength `γ` and chemical potential `λ`.
"
struct SpinlessFermions{T0<:Real,T1<:Real,T2<:Real} <: HamiltonianModel
    t::T0
	γ::T1
    λ::T2
end

"""
    hamiltonian(model::spinless_fermions)
"""
function hamiltonian(model::SpinlessFermions)
    h = [0 0 0 -model.γ;0 -model.λ -model.t 0;0 -model.t -model.λ 0;-model.γ 0 0 -2model.λ]
    return h
end

@doc raw"
    spinful_fermions(γ::Real,λ::Real)

return a struct representing the free fermions model with paring strength `Delta`.
"
struct SpinfulFermions{T0<:Real,T1<:Real,T2<:Real} <: HamiltonianModel
    t::T0
	Δ::T1
	μ::T2
end

"""
	hamiltonian(model::SpinfulFermions)
"""
function hamiltonian(model::SpinfulFermions)
	t = model.t
	Δ = model.Δ
	μ = model.μ
		ampo = AutoMPO()
		sites = siteinds("Electron",2)
		ampo += -t ,"Cdagup",1,"Cup",2
        ampo += -t ,"Cdagup",2,"Cup",1
        ampo += -t ,"Cdagdn",1,"Cdn",2
        ampo += -t ,"Cdagdn",2,"Cdn",1
        
		ampo += Δ,"Cdagup",1,"Cdagdn",2
        ampo += conj(Δ),"Cdn",2,"Cup",1
        ampo += Δ,"Cdagup",2,"Cdagdn",1
        ampo += conj(Δ),"Cdn",1,"Cup",2

		ampo += 0.5*μ,"Cdagup",1,"Cup",1
		ampo += 0.5*μ,"Cdagup",2,"Cup",2
		ampo += 0.5*μ,"Cdagdn",1,"Cdn",1
		ampo += 0.5*μ,"Cdagdn",2,"Cdn",2


		H = MPO(ampo,sites)

		H1 = Array(H[1],inds(H[1])...)
		H2 = Array(H[2],inds(H[2])...)
		h = reshape(ein"aij,apq->ipjq"(H1,H2),16,16)

    return h
end
