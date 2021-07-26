using JLD2
using FileIO

"""
    vumps_env(model<:HamiltonianModel, β, D)

return the vumps environment of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Save `env` in file `./data/model_β_D.jld2`. Requires that `model_tensor` are defined for `model`.
"""
function vumps_env(model::MT, β, D; tol=1e-10, maxiter=20, verbose = false, savefile = false, atype = Array) where {MT <: HamiltonianModel}
    M = atype(model_tensor(model, β))
    mkpath("./data/$(model)_$(atype)")
    chkp_file = "./data/$(model)_β$(β)_$(atype)/chi$(D).jld2"
    if isfile(chkp_file)                               
        rt = SquareVUMPSRuntime(M, chkp_file, D; verbose = verbose)   
    else
        rt = SquareVUMPSRuntime(M, Val(:random), D; verbose = verbose)
    end
    env = vumps(rt; tol=tol, maxiter=maxiter, verbose = verbose)
    savefile && save(chkp_file, "env", env) # if forward steps is too small, backward go to this way will make mistake!!!
    return env
end

"""
    Z(env::SquareVUMPSRuntime)

return the partition function of the `env`.
"""
function Z(env::SquareVUMPSRuntime)
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    AC = ein"asc,cb -> asb"(AL,C)
    z = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,AC,M,FR,conj(AC))
    λ = ein"(acβ,βη),(ηcγ,aγ) -> "(FL,C,FR,conj(C))
    return Array(z)[]/Array(λ)[]
end

"""
    magnetisation(env::SquareVUMPSRuntime, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env::SquareVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Mag = _arraytype(M)(mag_tensor(model, β))
    AC = ein"asc,cb -> asb"(AL,C)
    mag = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,AC,Mag,FR,conj(AC))
    λ = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,AC,M,FR,conj(AC))
    return Array(mag)[]/Array(λ)[]
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
    AC = ein"asc,cb -> asb"(AL,C)
    energy = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,AC,Ene,FR,conj(AC))
    λ = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,AC,M,FR,conj(AC))
    return Array(energy)[]/Array(λ)[]*2 # factor 2 for counting horizontal and vertical links
end

"""
    Z(env)

return the up and down partition function of the `env`.
"""
function Z(env)
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR = env
    ACu = ein"asc,cb -> asb"(ALu,Cu)
    ACd = ein"asc,cb -> asb"(ALd,Cd)
    z = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,ACu,M,FR,ACd)
    λ = ein"(acβ,βη),(ηcγ,aγ) -> "(FL,Cu,FR,Cd)
    λFL_n, FL_n = norm_FL(ALu, ALd)
    λFR_n, FR_n = norm_FR(ARu, ARd)
    # @show λFL_n,λFR_n
    overlap = ein"((ae,adb),bc),((edf,fg),cg) ->"(FL_n,ALu,Cu,ALd,Cd,FR_n)[]/ein"((ac,ab),bd),cd ->"(FL_n,Cu,FR_n,Cd)[]
    return Array(z)[]/Array(λ)[]/overlap
end

"""
    magnetisation(env, model::MT, β)

return the up and down magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env, model::MT, β) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    Mag = _arraytype(M)(mag_tensor(model, β))
    ACu = ein"asc,cb -> asb"(ALu,Cu)
    ACd = ein"asc,cb -> asb"(ALd,Cd)
    mag = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,ACu,Mag,FR,ACd)
    λ = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,ACu,M,FR,ACd)
    return Array(mag)[]/Array(λ)[]
end

"""
    energy(env, model::MT, β)

return the up and down energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env, model::MT, β) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    Ene = _arraytype(M)(energy_tensor(model, β))
    ACu = ein"asc,cb -> asb"(ALu,Cu)
    ACd = ein"asc,cb -> asb"(ALd,Cd)
    energy = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,ACu,Ene,FR,ACd)
    λ = ein"(((acβ,βsη),cpds),ηdγ),apγ -> "(FL,ACu,M,FR,ACd)
    return Array(energy)[]/Array(λ)[]*2 # factor 2 for counting horizontal and vertical links
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