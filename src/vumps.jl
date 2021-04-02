using LinearAlgebra
using KrylovKit

export AbstractLattice, SquareLattice
abstract type AbstractLattice end
struct SquareLattice <: AbstractLattice end

export VUMPSRuntime, SquareVUMPSRuntime

# NOTE: should be renamed to more explicit names
"""
    VUMPSRuntime{LT}

a struct to hold the tensors during the `vumps` algorithm, containing
- `d × d × d × d` `M` tensor
- `D × d × D` `AL` tensor
- `D × D`     `C` tensor
- `D × d × D` `AR` tensor
- `D × d × D` `FL` tensor
- `D × d × D` `FR` tensor
and `LT` is a AbstractLattice to define the lattice type.
"""
struct VUMPSRuntime{LT,T,N,AT<:AbstractArray{T,N},ET,CT}
    M::AT
    AL::ET
    C::CT
    AR::ET
    FL::ET
    FR::ET
    function VUMPSRuntime{LT}(M::AT, AL::AbstractArray{T}, C::AbstractArray{T}, AR::AbstractArray{T},
        FL::AbstractArray{T}, FR::AbstractArray{T}) where {LT<:AbstractLattice,T,N,AT<:AbstractArray{T,N}}
        new{LT,T,N,AT,typeof(AL),typeof(C)}(M,AL,C,AR,FL,FR)
    end
end

const SquareVUMPSRuntime{T,AT} = VUMPSRuntime{SquareLattice,T,4,AT}
SquareVUMPSRuntime(M::AT,AL,C,AR,FL,FR) where {T,AT<:AbstractArray{T, 4}} = VUMPSRuntime{SquareLattice}(M,AL,C,AR,FL,FR)

getD(rt::VUMPSRuntime) = size(rt.AL, 1)
getd(rt::VUMPSRuntime) = size(rt.M, 1)

@doc raw"
    SquareVUMPSRuntime(M::AbstractArray{T,4}, env::Val, χ::Int)

create a `SquareVUMPSRuntime` with M-tensor `M`. The AL,C,AR,FL,FR
tensors are initialized according to `env`. If `env = Val(:random)`,
the A is initialized as a random D×d×D tensor,and AL,C,AR are the corresponding 
canonical form. FL,FR is the left and right environment:
```
┌── AL─       ┌──        ─ AR──┐         ──┐    
│   │         │            │   │           │      
FL─ M ─  = λL FL─        ─ M ──FR   = λR ──FR   
│   │         │            │   │           │      
┕── AL─       ┕──        ─ AR──┘         ──┘  
```

# example

```jldoctest; setup = :(using ADVUMPS)
julia> rt = SquareVUMPSRuntime(randn(2,2,2,2), Val(:random), 4);

julia> size(rt.AL) == (4,2,4)
true

julia> size(rt.C) == (4,4)
true
```
"
function SquareVUMPSRuntime(M::AbstractArray{T,4}, env::Val, D::Int) where T
    return SquareVUMPSRuntime(M, _initializect_square(M, env, D)...)
end

function _initializect_square(M::AbstractArray{T,4}, env::Val{:random}, D::Int) where T
    d = size(M,1)
    A = rand(T,D,d,D)
    AL, = leftorth(A)
    C, AR = rightorth(AL)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)
    AL,C,AR,FL,FR
end

function vumps(rt::VUMPSRuntime; tol::Real, maxit::Integer)
    # initialize
    olderror = Inf

    stopfun = StopFunction(olderror, -1, tol, maxit)
    rt, err = fixedpoint(res->vumpstep(res...), (rt, olderror, tol), stopfun)
    # @show err
    return rt
end

function vumpstep(rt::VUMPSRuntime,err,tol)
    M,AL,C,AR,FL,FR= rt.M,rt.AL,rt.C,rt.AR,rt.FL,rt.FR
    _, AL, C, AR, = ACCtoALAR(AL, C, AR, M, FL, FR; tol = tol/10)
    _, FL = leftenv(AL, M, FL; tol = tol/10)
    _, FR = rightenv(AR, M, FR; tol = tol/10)
    err = error(AL,C,FL,M,FR)
    return SquareVUMPSRuntime(M, AL, C, AR, FL, FR), err, tol
end

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

"""
    leftorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
"""
function leftorth(A, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, maxiter = 100, kwargs...)
    λ2s, ρs, info = eigsolve(C'*C, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do ρ
        ρE = ein"cd,dsb,csa -> ab"(ρ, A, conj(A))
        return ρE
    end
    ρ = ρs[1] + ρs[1]'
    ρ ./= tr(ρ)
    # C = cholesky!(ρ).U
    # If ρ is not exactly positive definite, cholesky will fail
    F = svd!(ρ)
    C = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
    _, C = qrpos!(C)

    D, d, = size(A)
    Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    rmul!(R, 1/λ)
    numiter = 1
    while norm(C-R) > tol && numiter < maxiter
        # C = R
        λs, Cs, info = eigsolve(R, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do X
            Y = ein"cd,dsb,csa -> ab"(X,A,conj(AL))
            return Y
        end
        _, C = qrpos!(Cs[1])
        # The previous lines can speed up the process when C is still very far from the correct
        # gauge transform, it finds an improved value of C by finding the fixed point of a
        # 'mixed' transfer matrix composed of `A` and `AL`, even though `AL` is also still not
        # entirely correct. Therefore, we restrict the number of iterations to be 1 and don't
        # check for convergence
        Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        rmul!(R, 1/λ)
        numiter += 1
    end
    C = R
    return real(AL), real(C), λ
end

"""
    rightorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a gauge transform C, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ C AR^s = A^s C``, where an initial guess for `C` can be
provided.
"""
function rightorth(A, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, kwargs...)
    AL, C, λ = leftorth(permutedims(A,(3,2,1)), permutedims(C,(2,1)); tol = tol, kwargs...)
    return permutedims(C,(2,1)), permutedims(AL,(3,2,1)), λ
end

"""
    leftenv(A, M, FL; kwargs)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - conj(AL) contracted along the physical dimension.
```
┌── AL─       ┌──         
│   │         │             
FL─ M ─  = λL FL─         
│   │         │             
┕── AL─       ┕──        
```
"""
function leftenv(AL, M, FL = rand(eltype(AL), size(AL,1), size(M,1), size(AL,1)); kwargs...)
    # λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian = false, kwargs...) do FL
    #     FL = ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))
    # end
    AA = ein"asf,bpes,cpd -> abcfed"(AL,M,conj(AL))
    AA = reshape(AA, size(M,1)*size(AL,1)^2, :)
    λL, FL = lefteig(AA, FL -> ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL)),
        FR -> ein"αpγ,γcη,ascp,βsη -> αaβ"(AL,FR,M,conj(AL)), FL; kwargs...)
    FL = reshape(FL, size(AL,1), size(M,1), size(AL,1))
    return λL,FL
end
"""
    rightenv(A, M, FR; kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the right fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
 ─ AR──┐         ──┐   
   │   │           │   
 ─ M ──FR   = λR ──FR  
   │   │           │   
 ─ AR──┘         ──┘  
```
"""
function rightenv(AR, M, FR = randn(eltype(AR), size(AR,1), size(M,3), size(AR,1)); kwargs...)
    # λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian = false, kwargs...) do FR
    #     FR = ein"αpγ,γcη,ascp,βsη -> αaβ"(AR,FR,M,conj(AR))
    # end
    AA = ein"asf,bpes,cpd -> abcfed"(AR,M,conj(AR))
    AA = reshape(AA, size(M,3)*size(AR,1)^2, :)
    λR, FR = righteig(AA, FL -> ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AR,M,conj(AR)),
        FR -> ein"αpγ,γcη,ascp,βsη -> αaβ"(AR,FR,M,conj(AR)), FR; kwargs...)
    # @show norm(λR.*FR-AA*FR)
    FR = reshape(FR, size(AR,1), size(M,3), size(AR,1))
    return λR,FR
end


function ACenv(AC, FL, M, FR;kwargs...)
    D,d,_ = size(FL)
    AA = ein"αaγ,asbp,ηbβ -> αsβγpη"(FL,M,FR)
    AA = reshape(AA,d*D^2,:)
    μ1, AC = eig(AA,AC -> ein"αaγ,γpη,asbp,ηbβ -> αsβ"(FL,AC,M,FR), AC; kwargs...)
    AC = reshape(AC, D, d, D)
    return μ1, AC
end

function Cenv(C, FL, FR;kwargs...)
    D,d,_ = size(FL)
    AA = ein"αaγ,ηaβ -> αβγη"(FL,FR)
    AA = reshape(AA,D^2,:)
    μ0, C = eig(AA,C -> ein"αaγ,γη,ηaβ -> αβ"(FL,C,FR), C; kwargs...)
    C = reshape(C, D, D)
    return μ0, C
end

function ACCtoALAR(AL, C, AR, M, FL, FR; kwargs...)
    D, d, = size(AL)
    AC = ein"asc,cb -> asb"(AL,C)
    μ1, AC = ACenv(AC, FL, M, FR; kwargs...)
    μ0, C = Cenv(C, FL, FR;kwargs...)
    λ = real(μ1/μ0)

    QAC, RAC = qrpos(reshape(AC,(D*d, D)))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', (D, d, D))
    errL = norm(RAC-RC)

    LAC, QAC = lqpos(reshape(AC,(D, d*D)))
    LC, QC = lqpos(C)
    AR = reshape(QC'*QAC, (D, d, D))
    errR = norm(LAC-LC)

    return λ, AL, C, AR, errL, errR
end

function error(AL,C,FL,M,FR)
    AC = ein"asc,cb -> asb"(AL,C)
    MAC = ein"αaγ,γpη,asbp,ηbβ -> αsβ"(FL,AC,M,FR)
    MAC -= ein"asd,cpd,cpb -> asb"(AL,conj(AL),MAC)
    err = norm(MAC)
end