using ChainRulesCore
using LinearAlgebra
using KrylovKit

@Zygote.nograd StopFunction
@Zygote.nograd leftorth
@Zygote.nograd rightorth
@Zygote.nograd _initializect_square
@Zygote.nograd printstyled
@Zygote.nograd save
@Zygote.nograd load
@Zygote.nograd error

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NO_FIELDS, NO_FIELDS, NO_FIELDS, permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NO_FIELDS, Δ .* A ./ (n + eps(0f0)), NO_FIELDS
    end
    return n, back
end

"""
    ChainRulesCore.rrule(::typeof(leftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

```
          ┌──  AL ──┐ 
          │    │    │ 
dM   =   FL ──   ── ξl
          │    │    │ 
          └──  AL ──┘ 

          ┌──     ──┐         ┌──  AL ──┐ 
          │    │    │         │    │    │ 
dAL  =   FL ── M ── ξl   +   FL ── M ── ξl
          │    │    │         │    │    │ 
          └──  AL ──┘         └──     ──┘ 
```
"""
function ChainRulesCore.rrule(::typeof(leftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λ, FL = leftenv(AL, M, FL; kwargs...)
    function back((dλ, dFL))
        ξl, info = linsolve(FR -> ein"ηpβ,βaα,csap,γsα -> ηcγ"(AL, FR, M, conj(AL)), permutedims(dFL, (3, 2, 1)), -λ, 1)
        err = ein"abc,cba ->"(FL, ξl)[]
        abs(err) > 1e-5 && throw("FL and ξl aren't orthometric. err = $(err)")
        dAL = -ein"γcη,csap,γsα,βaα -> ηpβ"(FL, M, conj(AL), ξl) - ein"γcη,csap,ηpβ,βaα -> γsα"(FL, M, AL, ξl)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL, AL, conj(AL), ξl)
        # @show info ein"abc,abc ->"(FL,ξl)[] ein"γpη,γpη -> "(FL,dFL)[]
        return NO_FIELDS, dAL, dM, NO_FIELDS...
    end
    return (λ, FL), back
end

"""
    ChainRulesCore.rrule(::typeof(rightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}

```
          ┌──  AR ──┐ 
          │    │    │ 
dM   =   ξr ──   ── FR
          │    │    │ 
          └──  AR ──┘ 

          ┌──     ──┐         ┌──  AR ──┐ 
          │    │    │         │    │    │ 
dAR  =   ξr ── M ── FR   +   ξr ── M ── FR
          │    │    │         │    │    │ 
          └──  AR ──┘         └──     ──┘
```
"""
function ChainRulesCore.rrule(::typeof(rightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λ, FR = rightenv(AR, M, FR; kwargs...)
    function back((dλ, dFR))
        ξr, info = linsolve(FL -> ein"ηpβ,γcη,csap,γsα -> αaβ"(AR, FL, M, conj(AR)), permutedims(dFR, (3, 2, 1)),  -λ, 1)
        err = ein"abc,cba ->"(ξr, FR)[]
        abs(err) > 1e-5 && throw("FR and ξr aren't orthometric. err = $(err)")
        dAR = -ein"γcη,csap,γsα,βaα -> ηpβ"(ξr, M, conj(AR), FR) - ein"γcη,csap,ηpβ,βaα -> γsα"(ξr, M, AR, FR)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(ξr, AR, conj(AR), FR)
        return NO_FIELDS, dAR, dM, NO_FIELDS...
    end
    return (λ, FR), back
end

"""
    ChainRulesCore.rrule(::typeof(ACenv),AC::AbstractArray{T}, FL::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; 

```
          ┌──  AC ──┐ 
          │    │    │ 
dM   =   FL ──   ── FR
          │    │    │ 
          └──  ξ  ──┘ 

          ┌──  AC ──┐ 
          │    │    │ 
dFL  =      ── M ── FR
          │    │    │ 
          └──  ξ  ──┘ 

          ┌──  AC ──┐ 
          │    │    │ 
dFR  =   FL ── M ── 
          │    │    │ 
          └──  ξ  ──┘       
```
"""
function ChainRulesCore.rrule(::typeof(ACenv),AC::AbstractArray{T}, FL::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; 
    kwargs...) where {T}
    λ, AC = ACenv(AC, FL, M, FR; kwargs...)
    function back((dλ, dAC))
        ξ, info = linsolve(AC -> ein"αaγ,αsβ,asbp,ηbβ -> γpη"(FL, AC, M, FR), dAC, -λ, 1)
        err = ein"abc,abc ->"(AC, ξ)[]
        abs(err) > 1e-5 && throw("AC and ξ aren't orthometric. err = $(err)")
        # @show info ein"abc,abc ->"(AC,ξ)[] ein"γpη,γpη -> "(AC,dAC)[]
        dFL = -ein"ηpβ,βaα,csap,γsα -> γcη"(AC, FR, M, ξ)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL, AC, ξ, FR)
        dFR = -ein"ηpβ,γcη,csap,γsα -> βaα"(AC, FL, M, ξ)
        return NO_FIELDS, NO_FIELDS, dFL, dM, dFR
    end
    return (λ, AC), back
end

"""
    ChainRulesCore.rrule(::typeof(ACenv),AC::AbstractArray{T}, FL::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; 

```
          ┌──  C  ──┐ 
          │         │ 
dFL  =      ─────── FR
          │         │ 
          └──  ξ  ──┘ 

          ┌──  C  ──┐ 
          │         │ 
dFR  =   FL ─────── 
          │         │ 
          └──  ξ  ──┘       
```
"""
function ChainRulesCore.rrule(::typeof(Cenv), C::AbstractArray{T}, FL::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λ, C = Cenv(C, FL, FR; kwargs...)
    function back((dλ, dC))
        ξ, info = linsolve(C -> ein"αaγ,αβ,ηaβ -> γη"(FL, C, FR), dC, -λ, 1)
        err = ein"ab,ab ->"(C, ξ)[]
        abs(err) > 1e-5 && throw("C and ξ aren't orthometric. err = $(err)")
        # @show info ein"ab,ab ->"(C,ξ)[] ein"γp,γp -> "(C,dC)[]
        dFL = -ein"ηβ,βaα,γα -> γaη"(C, FR, ξ)
        dFR = -ein"ηβ,γcη,γα -> βcα"(C, FL, ξ)
        return NO_FIELDS, NO_FIELDS, dFL, dFR
    end
    return (λ, C), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        Rt, info = linsolve(x -> R * x, Matrix(I, size(R)), 0, 1)
        dA = (dQ + Q * Symmetric(M, :L)) * (Rt)'
        return NO_FIELDS, dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        Lt, info = linsolve(x -> L * x, Matrix(I, size(L)), 0, 1)
        dA = (Lt)' * (dQ + Symmetric(M, :L) * Q)
        return NO_FIELDS, dA
    end
    return (L, Q), back
end

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])

return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
num_grad(f, K::Real; δ::Real=1e-5) = (f(K + δ / 2) - f(K - δ / 2)) / δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    map(CartesianIndices(a)) do i
        foo = x -> (ac = copy(a); ac[i] = x; f(ac))
        num_grad(foo, a[i], δ=δ)
    end
end