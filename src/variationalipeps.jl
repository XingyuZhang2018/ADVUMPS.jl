using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
"""
    energy(h, ipeps; χ, tol, maxiter)

return the energy of the `ipeps` 2-site hamiltonian `h` and calculated via a
ctmrg with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, ipeps::IPEPS, oc, key; verbose = false)
    folder, model, atype, _, χ, tol, maxiter = key
    # ipeps = indexperm_symmetrize(ipeps)  # NOTE: this is not good
    D = getd(ipeps)^2
    s = gets(ipeps)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(ipeps.bulk, conj(ipeps.bulk))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    env = obs_env(model, a; atype = atype, folder = folder, χ = χ, tol = tol, maxiter = maxiter, verbose = verbose, savefile = true, updown = true)
    e = expectationvalue(h, ap, env, oc)
    return e
end

"""
    oc1, oc2 = optcont(D::Int, χ::Int)

optimise the follow two einsum contractions for the given `D` and `χ` which are used to calculate the energy of the 2-site hamiltonian:

```
                                          a ────┬──c─ d
a ────┬──c──d──┬──── f                    │     b     │
│     b        e     │                    ├─ e ─┼─ f ─┤
├─ g ─┼─   h  ─┼─ i ─┤                    │     g     │
│     k        n     │                    ├─ h ─┼─ i ─┤
j ────┴──l──m──┴──── o                    │     k     │
                                          j ────┴──l─ m 
```
where the central two block are six order tensor have extra bond `pq` and `rs`
"""
function optcont(D::Int, χ::Int)
    sd = Dict('a' => χ, 'b' => D^2,'c' => χ, 'd' => χ, 'e' => D^2, 'f' => χ, 'g' => D^2, 'h' => D^2, 'i' => D^2, 'j' => χ, 'k' => D^2, 'l' => χ, 'm' => χ, 'n' => D^2, 'o' => χ, 'p' => 2, 'q' => 2, 'r' => 2, 's' => 2)
    oc1 = optimize_greedy(ein"agj,abc,gkhbpq,jkl,cd,lm,fio,def,hniers,mno -> pqrs", sd; method=MinSpaceDiff())
    sd = Dict('a' => χ, 'b' => D^2, 'c' => χ, 'd' => χ, 'e' => D^2, 'f' => D^2, 'g' => D^2, 'h' => D^2, 'i' => D^2, 'j' => χ, 'k' => D^2, 'l' => χ, 'm' => χ, 'p' => 2, 'q' => 2, 'r' => 2, 's' => 2)
    oc2 = optimize_greedy(ein"aehj,abc,cd,egfbpq,hkigrs,jkl,lm,dfim -> pqrs", sd; method=MinSpaceDiff())
    oc1, oc2
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareCTMRGRuntime` `env`.
"""
function expectationvalue(h, ap, env, oc)
    M, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FL, FR = env
    oc1, oc2 = oc
    ap /= norm(ap)
    etol = 0
    
    lr = oc1(FLo,ALu,ap,ALd,Cu,Cd,FRo,ARu,ap,ARd)
    e = ein"pqrs, pqrs -> "(lr,h)
    n = ein"pprr -> "(lr)
    println("── = $(Array(e)[]/Array(n)[])") 
    etol += Array(e)[]/Array(n)[]

    _, BgFL = bigleftenv(ALu, ALd, M)
    _, BgFR = bigrightenv(ARu, ARd, M)
    # BgFL = ein"cde, abc -> abde"(FLo[1,1],FL[2,1])
    # BgFR = ein"abc, cde -> adbe"(FRo[1,2],FR[2,2])
    lr2 = oc2(BgFL,ALu,Cu,ap,ap,ALd,Cd,BgFR)
    e2 = ein"pqrs, pqrs -> "(lr2,h)
    n2 = ein"pprr -> "(lr2)
    println("| = $(Array(e2)[]/Array(n2)[])") 
    etol += Array(e2)[]/Array(n2)[]

    return etol
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; folder = "./data/", atype = Array, D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true)
    key = (folder, model, atype, D, χ, tol, maxiter)
    folder = folder*"$(model)_$(atype)/"
    mkpath(folder)
    chkp_file = folder*"$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
        bulk = rand(ComplexF64,D,D,D,D,2)
        verbose && println("random initial iPEPS $chkp_file")
    end
    ipeps = SquareIPEPS(bulk)
    # ipeps = indexperm_symmetrize(ipeps)
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
function optimiseipeps(ipeps::IPEPS{LT}, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20)) where LT
    _, model, atype, D, χ, _, _ = key
    h = atype(hamiltonian(model))
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, IPEPS{LT}(atype(x)), oc, key; verbose=verbose))
    ff(x) = real(energy(h, IPEPS{LT}(atype(x)), oc, key; verbose=verbose))
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

    folder, model, atype, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open(folder*"$(model)_$(atype)/$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"$(model)_$(atype)/$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end