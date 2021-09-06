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

# function ChainRulesCore.rrule(::typeof(Base.getindex),arr::CuArray)   
#     function back(dy)     
#         return NO_FIELDS,OMEinsum.asarray(dy, arr)
#     end           
#     return getindex(arr), back
# end 

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
    ChainRulesCore.rrule(::typeof(leftenv), ALu::AbstractArray{T}, ALd::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

```
           ┌──  ALu  ──┐ 
           │     │     │ 
dM    = - FL  ──   ──  ξl
           │     │     │ 
           └──  ALd  ──┘ 

           ┌──       ──┐   
           │     │     │   
dALu  = -  FL ── M ──  ξl  
           │     │     │   
           └──  ALd  ──┘   

           ┌──  ALu  ──┐            a ────┬──── c 
           │     │     │            │     b     │ 
dALd  = -  FL ── M ──  ξl           ├─ d ─┼─ e ─┤ 
           │     │     │            │     g     │ 
           └──       ──┘            f ────┴──── h 
```
"""

function ChainRulesCore.rrule(::typeof(leftenv), ALu::AbstractArray{T}, ALd::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = leftenv(ALu, ALd, M, FL)
    # @show λl
    function back((dλ, dFL))
        # @show ein"abc,abc ->"(conj(FL), dFL)[]
        dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
        ξl, info = linsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(ALu, FR, M, conj(ALd)), conj(dFL), -λl, 1; maxiter = 1)
        @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dALu = -ein"((adf,fgh),dgeb),ceh -> abc"(FL, conj(ALd), M, ξl) 
        dALd = -ein"((adf,abc),dgeb),ceh -> fgh"(FL, ALu, M, ξl)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, ALu, conj(ALd), ξl)
        return NO_FIELDS, conj(dALu), dALd, conj(dM), NO_FIELDS...
    end
    return (λl, FL), back
end

function ChainRulesCore.rrule(::typeof(obs_leftenv), ALu::AbstractArray{T}, ALd::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = obs_leftenv(ALu, ALd, M, FL)
    # @show λl
    function back((dλ, dFL))
        # @show ein"abc,abc ->"(conj(FL), dFL)[]
        dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
        ξl, info = linsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(ALu, FR, M, ALd), conj(dFL), -λl, 1; maxiter = 1)
        @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dALu = -ein"((adf,fgh),dgeb),ceh -> abc"(FL, ALd, M, ξl) 
        dALd = -ein"((adf,abc),dgeb),ceh -> fgh"(FL, ALu, M, ξl)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, ALu, ALd, ξl)
        return NO_FIELDS, conj(dALu), conj(dALd), conj(dM), NO_FIELDS...
    end
    return (λl, FL), back
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

          ┌──  AC ──┐          a ────┬──── c 
          │    │    │          │     b     │ 
dFR  =   FL ── M ──            ├─ d ─┼─ e ─┤  
          │    │    │          │     g     │ 
          └──  ξ  ──┘          f ────┴──── h       
```
"""
function ChainRulesCore.rrule(::typeof(ACenv),AC::AbstractArray{T}, FL::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; 
    kwargs...) where {T}
    λAC, AC = ACenv(AC, FL, M, FR)
    # @show λAC
    function back((dλ, dAC))
        # @show ein"abc,abc ->"(conj(AC), dAC)[]
        dAC -= Array(ein"abc,abc ->"(conj(AC), dAC))[] * AC
        ξ, info = linsolve(AC -> ein"((adf,fgh),dgeb),ceh -> abc"(FL, AC, M, FR), conj(dAC), -λAC, 1; maxiter = 1)
        @assert info.converged==1
        # errAC = ein"abc,abc ->"(AC, ξ)[]
        # abs(errAC) > 1e-1 && throw("AC and ξ aren't orthometric. err = $(errAC)")
        dFL = -ein"((abc,ceh),dgeb),fgh -> adf"(AC, FR, M, ξ)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, AC, ξ, FR)
        dFR = -ein"((abc,adf),dgeb),fgh -> ceh"(AC, FL, M, ξ)
        return NO_FIELDS, NO_FIELDS, conj(dFL), conj(dM), conj(dFR)
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

          ┌──  C  ──┐            a ─── b
          │         │            │     │
dFR  =   FL ───────              ├─ c ─┤
          │         │            │     │
          └──  ξ  ──┘            d ─── e
```
"""
function ChainRulesCore.rrule(::typeof(Cenv), C::AbstractArray{T}, FL::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λC, C = Cenv(C, FL, FR)
    # @show λC
    function back((dλ, dC))
        # @show Array(ein"ab,ab ->"(conj(C), dC))[]
        dC -= Array(ein"ab,ab ->"(conj(C), dC))[] * C
        ξ, info = linsolve(C -> ein"(acd,de),bce -> ab"(FL, C, FR), conj(dC), -λC, 1; maxiter = 1)
        @assert info.converged==1
        # errC = ein"ab,ab ->"(C, ξ)[]
        # abs(errC) > 1e-1 && throw("C and ξ aren't orthometric. err = $(errC)")
        # @show info ein"ab,ab ->"(C,ξ)[] ein"γp,γp -> "(C,dC)[]
        dFL = -ein"(ab,bce),de -> acd"(C, FR, ξ)
        dFR = -ein"(ab,acd),de -> bce"(C, FL, ξ)
        return NO_FIELDS, NO_FIELDS, conj(dFL), conj(dFR)
    end
    return (λC, C), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = Array(R * dR' - dQ' * Q)
        dA = (UpperTriangular(R + I * 1e-6) \ (dQ + Q * _arraytype(Q)(Hermitian(M, :L)))' )'
        # @show ein"ab,ab -> "(A, conj(dA))[]
        return NO_FIELDS, _arraytype(Q)(dA)
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = Array(L' * dL - dQ * Q')
        dA = LowerTriangular(L + I * 1e-6)' \ (dQ + _arraytype(Q)(Hermitian(M, :L)) * Q)
        # @show ein"ab,ab -> "(A, conj(dA))
        return NO_FIELDS, _arraytype(Q)(dA)
    end
    return (L, Q), back
end

function ChainRulesCore.rrule(::typeof(mysvd), A::AbstractArray{T,2}) where {T}
    U,S,V = mysvd(A)
    function back((dU, dS, dV))
        # S .+= 1e-12
        m, n = size(A)
        k = min(m,n)
        Fp, Fm = zeros(k,k), zeros(k,k)
        for j = 1:k, i = 1:k
            if j != i
                Fp[i,j] = (S[j] - S[i])/((S[j] - S[i])^2 .+ 1e-12) + 1/(S[j] + S[i])
                Fm[i,j] = (S[j] - S[i])/((S[j] - S[i])^2 .+ 1e-12) - 1/(S[j] + S[i])
            end
        end
        dA = 1/2 * U * (Fp .* (U' * dU - dU' * U) + Fm .* (V' * dV - dV' * V)) * V' + 
                 (I - U * U') * dU * Diagonal(S.^-1) * V' + U * Diagonal(S.^-1) * dV' * (I - V * V')
        return NO_FIELDS, dA
    end
    return (U,S,V), back
end

"""
    ChainRulesCore.rrule(::typeof(bigleftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

``` 
       ┌── ALu ──┐     ┌── ALu ──┐ 
       │    │    │     │    │    │ 
       │ ──   ── │     │ ── M ── │ 
dM = - FL   │    ξl  - FL   │    ξl
       │ ── M ── │     │ ──   ── │ 
       │    │    │     │    │    │ 
       ┕── ALd ──┘     ┕── ALd ──┘ 

        ┌──     ──┐     ┌── ALu ──┐        a ────┬──── c
        │    │    │     │    │    │        │     b     │
        │ ── M ── │     │ ── M ── │        ├─ d ─┼─ e ─┤
dAL = - FL   │    ξl  - FL   │    ξl       │     f     │
        │ ── M ── │     │ ── M ── │        ├─ g ─┼─ h ─┤
        │    │    │     │    │    │        │     j     │
        ┕── ALd ──┘     ┕──     ──┘        i ────┴──── k

```
"""
function ChainRulesCore.rrule(::typeof(bigleftenv), ALu::AbstractArray{T}, ALd::AbstractArray{T}, M::AbstractArray{T}, FL4::AbstractArray{T}; kwargs...) where {T}
    λl, FL4 = bigleftenv(ALu, ALd, M, FL4; kwargs...)
    # @show λl
    function back((dλl, dFL4))
        dFL4 -= Array(ein"abcd,abcd ->"(conj(FL4), dFL4))[] * FL4
        ξl, info = linsolve(FR4 -> ein"(((cehk,abc),dfeb),gjhf),ijk -> adgi"(FR4,ALu,M,M,ALd), conj(dFL4), -λl, 1; maxiter = 1)
        @assert info.converged==1
        # errL = ein"abc,cba ->"(FL4, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dALu = -ein"(((adgi,ijk),gjhf),dfeb),cehk -> abc"(FL4, ALd, M, M, ξl)
        dALd = -ein"(((adgi,abc),dfeb),gjhf),cehk -> ijk"(FL4, ALu, M, M, ξl)
        dM = -ein"(adgi,abc),(gjhf,(ijk,cehk)) -> dfeb"(FL4, ALu, M, ALd, ξl) - ein"((adgi,abc),dfeb),(ijk,cehk)-> gjhf"(FL4, ALu, M, ALd, ξl)
        return NO_FIELDS, conj(dALu), conj(dALd), conj(dM), NO_FIELDS...
    end
    return (λl, FL4), back
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
function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end


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
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
    # map(CartesianIndices(a)) do i
    #     foo = x -> (ac = copy(a); ac[i] = x; f(ac))
    #     num_grad(foo, a[i], δ=δ)
    # end
end