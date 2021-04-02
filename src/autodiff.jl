using ChainRulesCore
using LinearAlgebra
using KrylovKit

@Zygote.nograd StopFunction
@Zygote.nograd leftorth
@Zygote.nograd rightorth
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

#################### to do: fix Gauge so that can only calculate one side ####################
function lefteig(A::AbstractArray{T,2}, fl::Function, fr::Function, xl::AbstractArray{T}; kwargs...) where {T}
    λs, ls, _ = eigsolve(fl, xl, 1, :LM; ishermitian = false, kwargs...)
    λ,l = λs[1], reshape(ls[1],:,1)
    return real(λ),real(l)
end

#adjoint for nonsymmetric eigsolve
#https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.245139 eq.(12)
function ChainRulesCore.rrule(::typeof(lefteig),A::AbstractArray{T,2}, fl::Function, fr::Function, 
                                                                            xl::AbstractArray{T}; kwargs...) where {T}
    s = size(A,1)
    λs, ls, _ = eigsolve(fl, xl, 1, :LM; ishermitian = false, kwargs...)
    λ,l = real(λs[1]), real(reshape(ls[1],:,1))
    function back((dλ,dl))
        # ξl = (A - I(s).*λ) \ ((I(s) - r*l')*dl)
        ξl = (A - I(s).*λ) * rand(T,s)
        function f(ξl)
            ξl = reshape(ξl,size(xl))
            reshape(fr(ξl),:,1)
        end
        ξl,info = linsolve(ξl->f(ξl), dl, ξl, -λ, 1; kwargs...)
        # @show info ξl'*l
        dA = - l*ξl'
        return NO_FIELDS, dA, NO_FIELDS, NO_FIELDS, NO_FIELDS
    end
    return (λ,l), back
end

function righteig(A::AbstractArray{T,2}, fl::Function, fr::Function, xr::AbstractArray{T}; kwargs...) where {T}
    λs, rs, _ = eigsolve(fr, xr, 1, :LM; ishermitian = false, kwargs...)
    λ,r = λs[1], reshape(rs[1],:,1)
    return real(λ),real(r)
end

#adjoint for nonsymmetric eigsolve
#https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.245139 eq.(12)
function ChainRulesCore.rrule(::typeof(righteig),A::AbstractArray{T,2}, fl::Function, fr::Function, 
                                xr::AbstractArray{T}; kwargs...) where {T}
    s = size(A,1)
    λs, rs, _ = eigsolve(fr, xr, 1, :LM; ishermitian = false, kwargs...)
    λ,r = real(λs[1]),  real(reshape(rs[1],:,1))
    function back((dλ,dr))
        # ξr = (A' - I(s)*λ) \ ((I(s) - l*r')*dr)
        ξr = (A' - I(s)*λ) * rand(T,s)
        function f(ξr)
            ξr = reshape(ξr,size(xr))
            reshape(fl(ξr),:,1)
        end
        ξr,info = linsolve(ξr->f(ξr), dr, ξr, -λ, 1; kwargs...)
        # @show info r'*ξr
        dA = (- ξr) *r'
        return NO_FIELDS, dA, NO_FIELDS, NO_FIELDS, NO_FIELDS
    end
    return (λ,r), back
end

function eig(A::AbstractArray{T,2}, f::Function, x₀::AbstractArray{T}; kwargs...) where {T}
    λs, vs, _ = eigsolve(f, x₀, 1, :LM; ishermitian = false, kwargs...)
    λ,v = λs[1], reshape(vs[1],:,1)
    return real(λ),real(v)
end

#adjoint for symmetric eigsolve
#https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.245139 eq.(13)
function ChainRulesCore.rrule(::typeof(eig),A::AbstractArray{T,2}, f::Function, x₀::AbstractArray{T}; kwargs...) where {T}
    s = size(A,1)
    λ,v = eig(A, f, x₀; kwargs...)
    # @show v'*v
    # @show norm(v'.*λ - v'*A)
    # @show norm(λ.*v-A*v)
    function back((dλ,dv))
        # ξ = (A - I(s).*λ) \ ((I(s) - v*v')*dv)
        ξ = rand(T,s)
        function ff(ξ)
            ξ = reshape(ξ,size(x₀))
            reshape(f(ξ),:,1)
        end
        b = (I(s) - v*v')*dv
        ξ,_ = linsolve(ξ->ff(ξ), b, ξ, -λ, 1; kwargs...)
        ξ -= (v'*ξ) .* v
        # println("test = ",norm((A - I(s).*λ) * ξ - ((I(s) - v*v')*dv)))
        # println("orth = ",v'*ξ)
        dA = (dλ.*v - ξ)*v'
        # dA = - ξ*v'
        return NO_FIELDS, dA, NO_FIELDS, NO_FIELDS
    end
    return (λ,v), back
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