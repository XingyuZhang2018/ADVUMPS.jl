using ChainRulesCore
using LinearAlgebra
using KrylovKit

@Zygote.nograd StopFunction
# @Zygote.nograd leftorth
# @Zygote.nograd rightorth
@Zygote.nograd _initializect_square
@Zygote.nograd printstyled

# patch since it's currently broken otherwise
# @Zygote.adjoint function Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
#     Base.typed_hvcat(T,rows, xs...), ȳ -> (nothing, nothing, permutedims(ȳ)...)
# end

function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T,rows, xs...)
    function back(ȳ)
        return NO_FIELDS, NO_FIELDS, NO_FIELDS, permutedims(ȳ)...
    end
    return y, back
end

# # improves performance compared to default implementation, also avoids errors
# # with some complex arrays
# function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
#     n = norm(A)
#     function back(Δ)
#         return NO_FIELDS, Δ .* A ./ (n + eps(0f0)), NO_FIELDS
#     end
#     return n, back
# end

function ChainRulesCore.rrule(::typeof(leftenv),AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λ, FL = leftenv(AL, M, FL; kwargs...)
    function back((dλ, dFL))
        ξl = rand(T,size(FL))
        ξl = ein"ηpβ,βaα,csap,γsα -> ηcγ"(AL,ξl,M,conj(AL)) - λ .* ξl
        ξl,info = linsolve(FR -> ein"ηpβ,βaα,csap,γsα -> ηcγ"(AL,FR,M,conj(AL)), dFL, ξl, -λ, 1; kwargs...)
        dAL = -ein"γcη,csap,γsα,βaα -> ηpβ"(FL,M,conj(AL),ξl) - ein"γcη,csap,ηpβ,βaα -> γsα"(FL,M,AL,ξl)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL,AL,conj(AL),ξl)
        # @show info ein"abc,abc ->"(FL,ξl)[] ein"γpη,γpη -> "(FL,dFL)[]
        return NO_FIELDS, dAL, dM, NO_FIELDS...
    end
    return (λ, FL), back
end

function ChainRulesCore.rrule(::typeof(rightenv),AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λ, FR = rightenv(AR, M, FR; kwargs...)
    function back((dλ, dFR))
        ξr = rand(T,size(FR))
        ξr = ein"ηpβ,γcη,csap,γsα -> αaβ"(AR,ξr,M,conj(AR)) - λ .* ξr
        ξr,info = linsolve(FL -> ein"ηpβ,γcη,csap,γsα -> αaβ"(AR,FL,M,conj(AR)), dFR, ξr, -λ, 1; kwargs...)
        dAR = -ein"γcη,csap,γsα,βaα -> ηpβ"(ξr,M,conj(AR),FR) - ein"γcη,csap,ηpβ,βaα -> γsα"(ξr,M,AR,FR)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(ξr,AR,conj(AR),FR)
        return NO_FIELDS, dAR, dM, NO_FIELDS...
    end
    return (λ, FR), back
end

function ChainRulesCore.rrule(::typeof(ACenv),AC::AbstractArray{T}, FL::AbstractArray{T}, M::AbstractArray{T}, 
                                                                    FR::AbstractArray{T}; kwargs...) where {T}
    λ, AC = ACenv(AC, FL, M, FR; kwargs...)
    function back((dλ, dAC))
        ξ = rand(T,size(AC))
        ξ = ein"αaγ,αsβ,asbp,ηbβ -> γpη"(FL,ξ,M,FR) - λ .* ξ
        # b = dAC - AC .* ein"γpη,γpη -> "(AC,dAC)[]
        ξ,info = linsolve(AC -> ein"αaγ,αsβ,asbp,ηbβ -> γpη"(FL,AC,M,FR), dAC, ξ, -λ, 1; kwargs...)
        # @show info ein"abc,abc ->"(AC,ξ)[] ein"γpη,γpη -> "(AC,dAC)[]
        dFL = -ein"ηpβ,βaα,csap,γsα -> ηcγ"(AC,FR,M,ξ)
        dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(FL,AC,ξ,FR)
        dFR = -ein"ηpβ,γcη,csap,γsα -> αaβ"(AC,FL,M,ξ)
        return NO_FIELDS, NO_FIELDS, dFL, dM, dFR
    end
    return (λ, AC), back
end

function ChainRulesCore.rrule(::typeof(Cenv), C::AbstractArray{T}, FL::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λ, C = Cenv(C, FL, FR; kwargs...)
    function back((dλ, dC))
        ξ = rand(T,size(C))
        ξ = ein"αaγ,αβ,ηaβ -> γη"(FL,ξ,FR) - λ .* ξ
        # b = dC - C .* ein"γpη,γpη -> "(C,dC)[]
        ξ,info = linsolve(C -> ein"αaγ,αβ,ηaβ -> γη"(FL,C,FR), dC, ξ, -λ, 1; kwargs...)
        # @show info ein"ab,ab ->"(C,ξ)[] ein"γp,γp -> "(C,dC)[]
        dFL = -ein"ηβ,βaα,γα -> ηaγ"(C,FR,ξ)
        dFR = -ein"ηβ,γcη,γα -> αcβ"(C,FL,ξ)
        return NO_FIELDS, NO_FIELDS, dFL, dFR
    end
    return (λ, C), back
end

#adjoint for QR factorization
#https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos),A::AbstractArray{T,2}) where {T}
    Q,R = qrpos(A)
    function back((dQ,dR))
        M = R * dR' - dQ'*Q
        dA = (dQ + Q * Symmetric(M, :L)) * (R^(-1))'
        return NO_FIELDS, dA
    end
    return (Q,R), back
end

function ChainRulesCore.rrule(::typeof(lqpos),A::AbstractArray{T,2}) where {T}
    L,Q = lqpos(A)
    function back((dL,dQ))
        M = L' * dL - dQ*Q'
        dA = (L^(-1))' * (dQ + Symmetric(M, :L) * Q)
        return NO_FIELDS, dA
    end
    return (L,Q), back
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
num_grad(f, K::Real; δ::Real = 1e-5) = (f(K+δ/2) - f(K-δ/2))/δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real = 1e-5)
    map(CartesianIndices(a)) do i
        foo = x -> (ac = copy(a); ac[i] = x; f(ac))
        num_grad(foo, a[i], δ = δ)
    end
end