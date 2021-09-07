using JLD2
using FileIO

"""
tensor order graph: from left to right, top to bottom.
```
a ────┬──── c    a─────b
│     b     │    │     │                     
├─ d ─┼─ e ─┤    ├──c──┤                  
│     g     │    │     │  
f ────┴──── h    d─────e    
```
"""

"""
    Z(env::SquareVUMPSRuntime)

return the partition function of the `env`.
"""
function Z(env::SquareVUMPSRuntime)
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    AC = ein"abc,cd -> abd"(AL,C)
    z = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,AC,M,FR,conj(AC))
    λ = ein"(acd,ab),(bce,de) -> "(FL,C,FR,conj(C))
    return real(Array(z)[]/Array(λ)[])
end

"""
    magnetisation(env::SquareVUMPSRuntime, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env::SquareVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Mag = _arraytype(M)(mag_tensor(model, β))
    AC = ein"abc,cd -> abd"(AL,C)
    mag = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,AC,Mag,FR,conj(AC))
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,AC,M,FR,conj(AC))
    return abs(real(Array(mag)[]/Array(λ)[]))
end

"""
    energy(env::SquareVUMPSRuntime, model::MT, β)

return the energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env::SquareVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ene = _arraytype(M)(energy_tensor(model, β))
    AC = ein"abc,cd -> abd"(AL,C)
    energy = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,AC,Ene,FR,conj(AC))
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,AC,M,FR,conj(AC))
    return real(Array(energy)[]/Array(λ)[]*2) # factor 2 for counting horizontal and vertical links
end

"""
    Z(env)

return the up and down partition function of the `env`.
"""
function Z(env)
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, = env
    ACu = ein"abc,cd -> abd"(ALu,Cu)
    ACd = ein"abc,cd -> abd"(ALd,Cd)
    z = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,ACu,M,FR,conj(ACd))
    λ = ein"(acd,ab),(bce,de) -> "(FL,Cu,FR,conj(Cd))
    λFL_n, FL_n = norm_FL(ALu, ALd)
    λFR_n, FR_n = norm_FR(ARu, ARd)
    # @show λFL_n,λFR_n
    overlap = ein"((ae,adb),bc),((edf,fg),cg) ->"(FL_n,ALu,Cu,conj(ALd),conj(Cd),FR_n)[]/ein"((ac,ab),bd),cd ->"(FL_n,Cu,FR_n,conj(Cd))[]
    return real(Array(z)[]/Array(λ)[]/overlap)
end

"""
    magnetisation(env, model::MT, β)

return the up and down magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env, model::MT, β) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR,  = env
    Mag = _arraytype(M)(mag_tensor(model, β))
    ACu = ein"abc,cd -> abd"(ALu,Cu)
    ACd = ein"abc,cd -> abd"(ALd,Cd)
    mag = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,ACu,Mag,FR,ACd)
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,ACu,M,FR,ACd)
    return abs(real(Array(mag)[]/Array(λ)[]))
end

"""
    energy(env, model::MT, β)

return the up and down energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env, model::MT, β) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR,  = env
    Ene = _arraytype(M)(energy_tensor(model, β))
    ACu = ein"abc,cd -> abd"(ALu,Cu)
    ACd = ein"abc,cd -> abd"(ALd,Cd)
    energy = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,ACu,Ene,FR,ACd)
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,ACu,M,FR,ACd)
    return real(Array(energy)[]/Array(λ)[]*2) # factor 2 for counting horizontal and vertical links
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

"""
    eneofβ(::Ising,β)

return some the numerical integrations of analytical result for the energy at inverse temperature
`β` for the 2d classical ising model.
"""
function eneofβ(::Ising, β)
    if β == 0.0
        return 0
    elseif β == 0.2
        return -0.42822885693016843
    elseif β == 0.4
        return -1.1060792706185651
    elseif β == 0.6
        return -1.909085845408498
    elseif β == 0.8
        return -1.9848514445364174
    elseif β == 1.0
        return -1.997159425837225
    end
end

"""
    Zofβ(::Ising,β)

return some the numerical integrations of analytical result for the partition function at inverse temperature
`β` for the 2d classical ising model.
"""
function Zofβ(::Ising,β)
    if β == 0.0
        return 2.0
    elseif β == 0.2
        return 2.08450374046259
    elseif β == 0.4
        return 2.4093664345022363
    elseif β == 0.6
        return 3.3539286863974582
    elseif β == 0.8
        return 4.96201030069517
    elseif β == 1.0
        return 7.3916307004743125
    end
end