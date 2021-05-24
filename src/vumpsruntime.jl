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
function SquareVUMPSRuntime(M::AbstractArray{T,4}, env, D::Int; verbose = false) where T
    return SquareVUMPSRuntime(M, _initializect_square(M, env, D; verbose = verbose)...)
end

function _initializect_square(M::AbstractArray{T,4}, env::Val{:random}, D::Int; verbose = false) where T
    d = size(M,1)
    A = _arraytype(M)(rand(T,D,d,D))
    AL, = leftorth(A)
    C, AR = rightorth(AL)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)
    verbose && print("random initial vumps environment-> ")
    AL,C,AR,FL,FR
end

function _initializect_square(M::AbstractArray{T,4}, chkp_file::String, D::Int; verbose = false) where T
    env = load(chkp_file)["env"]
    atype = _arraytype(M)
    verbose && print("vumps environment load from $(chkp_file) -> ")   
    atype(env.AL),atype(env.C),atype(env.AR),atype(env.FL),atype(env.FR)
end

function vumps(rt::VUMPSRuntime; tol::Real, maxiter::Int, verbose = false)
    # initialize
    olderror = Inf

    stopfun = StopFunction(olderror, -1, tol, maxiter)
    rt, err = fixedpoint(res->vumpstep(res...), (rt, olderror), stopfun)
    verbose && println("vumps done@step: $(stopfun.counter), error=$(err)")
    return rt
end

function vumpstep(rt::VUMPSRuntime,err)
    # global backratio = 1.0
    # Zygote.@ignore print(round(-log(10,backratio)),' ')
    M,AL,C,AR,FL,FR= rt.M,rt.AL,rt.C,rt.AR,rt.FL,rt.FR
    AC = Zygote.@ignore ein"asc,cb -> asb"(AL,C)
    _, AC = ACenv(AC, FL, M, FR)
    _, C = Cenv(C, FL, FR) 
    AL, AR, _, _ = ACCtoALAR(AC, C)
    _, FL = leftenv(AL, M, FL)
    _, FR = rightenv(AR, M, FR)

    ##### avoid gradient explosion for too many iterations #####
    # M = backratio .* M + Zygote.@ignore (1-backratio) .* M
    # AL = backratio .* AL + Zygote.@ignore (1-backratio) .* AL
    # C = backratio .* C +  Zygote.@ignore (1-backratio) .* C
    # AR = backratio .* AR + Zygote.@ignore (1-backratio) .* AR
    # FL = backratio .* FL + Zygote.@ignore (1-backratio) .* FL
    # FR = backratio .* FR + Zygote.@ignore (1-backratio) .* FR

    err = error(AL,C,FL,M,FR)
    return SquareVUMPSRuntime(M, AL, C, AR, FL, FR), err
end