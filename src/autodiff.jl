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
        return  NoTangent(),  NoTangent(),  NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return  NoTangent(), Δ .* A ./ (n + eps(0f0)),  NoTangent()
    end
    return n, back
end

"""
    ChainRulesCore.rrule(::typeof(leftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

```
           ┌──  AL ──┐ 
           │    │    │ 
dM   =  - FL ──   ── ξl
           │    │    │ 
           └──  AL ──┘ 

           ┌──     ──┐         ┌──  AL ──┐ 
           │    │    │         │    │    │ 
dAL  =  - FL ── M ── ξl   -   FL ── M ── ξl
           │    │    │         │    │    │ 
           └──  AL ──┘         └──     ──┘ 
```
"""
function ChainRulesCore.rrule(::typeof(leftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = leftenv(AL, M, FL)
    # @show λl
    function back((dλ, dFL))
        ξl, info = linsolve(FR -> ein"ηpβ,βaα,csap,γsα -> ηcγ"(AL, FR, M, conj(AL)), permutedims(dFL, (3, 2, 1)), -λl, 1)
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAL = -ein"γcη,csap,γsα,βaα -> ηpβ"(FL, M, conj(AL), ξl) - ein"γcη,csap,ηpβ,βaα -> γsα"(FL, M, AL, ξl)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL, AL, conj(AL), ξl)
        return  NoTangent(), dAL, dM,  NoTangent()...
    end
    return (λl, FL), back
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
    λr, FR = rightenv(AR, M, FR)
    # @show λr
    function back((dλ, dFR))
        ξr, info = linsolve(FL -> ein"ηpβ,γcη,csap,γsα -> αaβ"(AR, FL, M, conj(AR)), permutedims(dFR, (3, 2, 1)), -λr, 1)
        # errR = ein"abc,cba ->"(ξr, FR)[]
        # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. err = $(errR)")
        dAR = -ein"γcη,csap,γsα,βaα -> ηpβ"(ξr, M, conj(AR), FR) - ein"γcη,csap,ηpβ,βaα -> γsα"(ξr, M, AR, FR)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(ξr, AR, conj(AR), FR)
        return  NoTangent(), dAR, dM,  NoTangent()...
    end
    return (λr, FR), back
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
    λAC, AC = ACenv(AC, FL, M, FR)
    # @show λAC
    function back((dλ, dAC))
        ξ, info = linsolve(AC -> ein"αaγ,αsβ,asbp,ηbβ -> γpη"(FL, AC, M, FR), dAC, -λAC, 1)
        # errAC = ein"abc,abc ->"(AC, ξ)[]
        # abs(errAC) > 1e-1 && throw("AC and ξ aren't orthometric. err = $(errAC)")
        dFL = -ein"ηpβ,βaα,csap,γsα -> γcη"(AC, FR, M, ξ)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL, AC, ξ, FR)
        dFR = -ein"ηpβ,γcη,csap,γsα -> βaα"(AC, FL, M, ξ)
        return  NoTangent(),  NoTangent(), dFL, dM, dFR
    end
    return (λAC, AC), back
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
    λC, C = Cenv(C, FL, FR)
    # @show λC
    function back((dλ, dC))
        ξ, info = linsolve(C -> ein"αaγ,αβ,ηaβ -> γη"(FL, C, FR), dC, -λC, 1)
        # errC = ein"ab,ab ->"(C, ξ)[]
        # abs(errC) > 1e-1 && throw("C and ξ aren't orthometric. err = $(errC)")
        # @show info ein"ab,ab ->"(C,ξ)[] ein"γp,γp -> "(C,dC)[]
        dFL = -ein"ηβ,βaα,γα -> γaη"(C, FR, ξ)
        dFR = -ein"ηβ,γcη,γα -> βcα"(C, FL, ξ)
        return  NoTangent(),  NoTangent(), dFL, dFR
    end
    return (λC, C), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        dA = (UpperTriangular(R + I * 1e-6) \ (dQ + Q * Symmetric(M, :L))' )'
        return  NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        dA = LowerTriangular(L + I * 1e-6)' \ (dQ + Symmetric(M, :L) * Q)
        return  NoTangent(), dA
    end
    return (L, Q), back
end

"""
    ChainRulesCore.rrule(::typeof(bigleftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

``` 
       ┌── AL──┐     ┌── AL──┐ 
       │   │   │     │   │   │ 
       │──   ──│     │── M ──│ 
dM = - FL  │   ξl  - FL  │   ξl
       │── M ──│     │──   ──│ 
       │   │   │     │   │   │ 
       ┕── AL──┘     ┕── AL──┘ 

        ┌──   ──┐     ┌── AL──┐ 
        │   │   │     │   │   │ 
        │── M ──│     │── M ──│ 
dAL = - FL  │   ξl  - FL  │   ξl
        │── M ──│     │── M ──│ 
        │   │   │     │   │   │ 
        ┕── AL──┘     ┕──   ──┘ 

```
"""
function ChainRulesCore.rrule(::typeof(bigleftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL4::AbstractArray{T}; kwargs...) where {T}
    λl, FL4 = bigleftenv(AL, M, FL4; kwargs...)
    # @show λl
    function back((dλl, dFL4))
        ξl, info = linsolve(FR4 -> ein"fghi,def,ckge,bjhk,aji -> dcba"(FR4,AL,M,M,conj(AL)), dFL4, -λl, 1)
        # errL = ein"abc,cba ->"(FL4, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAL = -ein"dcba,ckge,bjhk,aji,fghi -> def"(FL4, M, M, AL, ξl) -ein"dcba,def,ckge,bjhk,fghi -> aji"(FL4, AL, M, M, ξl)
        dM = -ein"dcba,def,bjhk,aji,fghi -> ckge"(FL4, AL, M, AL, ξl) -ein"dcba,def,ckge,aji,fghi -> bjhk"(FL4, AL, M, AL, ξl)
        return  NoTangent(), dAL, dM,  NoTangent()...
    end
    return (λl, FL4), back
end

"""
    ChainRulesCore.rrule(::typeof(rightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}

```
       ┌── AR──┐     ┌── AR──┐ 
       │   │   │     │   │   │ 
       │──   ──│     │── M ──│ 
dM = - ξr  │   FR  - ξr  │   FR
       │── M ──│     │──   ──│ 
       │   │   │     │   │   │ 
       ┕── AR──┘     ┕── AR──┘ 

        ┌──   ──┐     ┌── AR──┐ 
        │   │   │     │   │   │ 
        │── M ──│     │── M ──│ 
dAR = - ξr  │   FR  - ξr  │   FR
        │── M ──│     │── M ──│ 
        │   │   │     │   │   │ 
        ┕── AR──┘     ┕──   ──┘ 
```
"""
function ChainRulesCore.rrule(::typeof(bigrightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR4::AbstractArray{T}; kwargs...) where {T}
    λr, FR4 = bigrightenv(AR, M, FR4; kwargs...)
    # @show λr
    function back((dλ, dFR4))
        ξr, info = linsolve(FL4 -> ein"dcba,def,ckge,bjhk,aji -> fghi"(FL4,AR,M,M,conj(AR)), dFR4, -λr, 1)
        # errR = ein"abc,cba ->"(ξr, FR4)[]
        # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. err = $(errR)")
        dAR = -ein"dcba,ckge,bjhk,aji,fghi -> def"(ξr, M, M, AR,FR4) -ein"dcba,def,ckge,bjhk,fghi -> aji"(ξr, AR, M, M, FR4)
        dM = -ein"dcba,def,bjhk,aji,fghi -> ckge"(ξr, AR, M, AR, FR4) -ein"dcba,def,ckge,aji,fghi -> bjhk"(ξr, AR, M, AR, FR4)
        return  NoTangent(), dAR, dM,  NoTangent()...
    end
    return (λr, FR4), back
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