using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
"""
    energy(h, ipeps; χ, tol, maxiter)

return the energy of the `ipeps` 2-site hamiltonian `h` and calculated via a
ctmrg with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, model::HamiltonianModel, ipeps::IPEPS, oc; χ::Int, tol::Real, maxiter::Int, verbose = false)
    ipeps = indexperm_symmetrize(ipeps)  # NOTE: this is not good
    D = getd(ipeps)^2
    s = gets(ipeps)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(ipeps.bulk, conj(ipeps.bulk))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)
    folder = "./data/$(model)/"
    mkpath(folder)
    chkp_file = folder*"vumps_env_D$(D)_chi$(χ).jld2"
    if isfile(chkp_file)
        rt = SquareVUMPSRuntime(a, chkp_file, χ; verbose = verbose)
    else
        rt = SquareVUMPSRuntime(a, Val(:random), χ; verbose = verbose)
    end
    env = vumps(rt; tol=tol, maxiter=maxiter, verbose = verbose)
    Zygote.@ignore begin
        M, AL, C, AR, FL, FR = env.M, Array{Float64,3}(env.AL), Array{Float64,2}(env.C), Array{Float64,3}(env.AR), Array{Float64,3}(env.FL), Array{Float64,3}(env.FR)
        envsave = SquareVUMPSRuntime(M, AL, C, AR, FL, FR)
        save(chkp_file, "env", envsave)
    end
    e = expectationvalue(h, ap, env, oc)
    return e
end

function optcont(D::Int, χ::Int)
    sd = Dict('n' => D^2, 'f' => χ, 'd' => D^2, 'e' => χ, 'o' => D^2, 'h' => χ, 'j' => χ, 'i' => D^2, 'k' => D^2, 'r' => 2, 's' => 2, 'q' => 2, 'a' => χ, 'c' => χ, 'p' => 2, 'm' => χ, 'g' => D^2, 'l' => χ, 'b' => D^2)
    optimize_greedy(ein"abc,cde,bnodpq,anm,ef,ml,hij,fgh,okigrs,lkj -> pqrs", sd; method=MinSpaceDiff())
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareCTMRGRuntime` `env`.
"""
function expectationvalue(h, ap, env::SquareVUMPSRuntime, oc)
    AL,C,AR,FL,FR = env.AL,env.C,env.AR,env.FL,env.FR
    ap /= norm(ap)

    lr = oc(FL,AL,ap,conj(AL),C,conj(C),FR,AR,ap,conj(AR))
    e = Array(ein"pqrs, pqrs -> "(lr,h))[]
    n = Array(ein"pprr -> "(lr))[]

    # AC = ein"asc,cb -> asb"(AL,C)
    # _, FL4 = bigleftenv(AL, M)
    # _, FR4 = bigrightenv(AR, M)
    # e2 = ein"dcba,def,aji,fghi,ckgepq,bjhkrs,pqrs -> "(FL4,AC,conj(AC),FR4,ap,ap,h)[]
    # n2 = ein"dcba,def,aji,fghi,ckgepq,bjhkrs -> pqrs"(FL4,AC,conj(AC),FR4,ap,ap)
    # n2 = ein"pprr -> "(n2)[]
    # @show e/n e2/n2 (e/n+e2/n2)/2

    return e/n
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true, atype = Array)
    folder = "./data/$(model)/"
    mkpath(folder)
    key = (model, D, χ, tol, maxiter)
    chkp_file = folder*"$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
        bulk = rand(D,D,D,D,2)
        verbose && println("random initial iPEPS")
    end
    ipeps = SquareIPEPS(bulk)
    ipeps = indexperm_symmetrize(ipeps)
    return ipeps, key
end

"""
    optimiseipeps(ipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(ipeps::IPEPS{LT}, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20), atype = Array) where LT
    model, D, χ, tol, maxiter = key
    h = atype(hamiltonian(model))
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, model, IPEPS{LT}(atype(x)), oc; χ=χ, tol=tol, maxiter=maxiter, verbose=verbose))
    ff(x) = real(energy(h, model, IPEPS{LT}(atype(x)), oc; χ=χ, tol=tol, maxiter=maxiter, verbose=verbose))
    g(x) = @timeit to "backward" Zygote.gradient(ff,atype(x))[1]
    res = optimize(f, g, 
        ipeps.bulk, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    println(to)
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    model, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open("./data/$(model)/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save("./data/$(model)/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end