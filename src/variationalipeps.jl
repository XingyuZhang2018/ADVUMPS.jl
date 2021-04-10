using Optim, LineSearches
using LinearAlgebra: I, norm

"""
    diaglocalhamiltonian(diag::Vector)

return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function diaglocalhamiltonian(diag::Vector)
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

"""
    energy(h, ipeps; χ, tol, maxiter)

return the energy of the `ipeps` 2-site hamiltonian `h` and calculated via a
ctmrg with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h::AbstractArray{T,4}, ipeps::IPEPS; χ::Int, tol::Real, maxiter::Int) where T
    ipeps = indexperm_symmetrize(ipeps)  # NOTE: this is not good
    D = getd(ipeps)^2
    s = gets(ipeps)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(ipeps.bulk, conj(ipeps.bulk))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    rt = SquareVUMPSRuntime(a, Val(:random), χ)
    env = vumps(rt; tol=tol, maxiter=maxiter)
    e = expectationvalue(h, ap, env)
    return e
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareCTMRGRuntime` `env`.
"""
function expectationvalue(h, ap, env::SquareVUMPSRuntime)
    M,AL,C,AR,FL,FR = env.M,env.AL,env.C,env.AR,env.FL,env.FR
    ap /= norm(ap)
    # l = ein"abc,cde,anm,ef,ml,bnodpq -> folpq"(FL,AL,conj(AL),C,conj(C),ap)
    # e = ein"folpq,fgh,lkj,hij,okigrs,pqrs -> "(l,AR,conj(AR),FR,ap,h)[]
    # n = ein"folpq,fgh,lkj,hij,okigrs -> pqrs"(l,AR,conj(AR),FR,ap)
    e = ein"abc,cde,anm,ef,ml,fgh,lkj,hij,bnodpq,okigrs,pqrs -> "(FL,AL,conj(AL),C,conj(C),AR,conj(AR),FR,ap,ap,h)[]
    n = ein"abc,cde,anm,ef,ml,fgh,lkj,hij,bnodpq,okigrs -> pqrs"(FL,AL,conj(AL),C,conj(C),AR,conj(AR),FR,ap,ap)
    n = ein"pprr -> "(n)[]
    return e/n
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)
    folder = "./data/"
    mkpath(folder)
    key = (model, D, χ, tol, maxiter)
    chkp_file = folder*"$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["ipeps"]
        println("load iPEPS from $chkp_file")
    else
        bulk = rand(D,D,D,D,2)
        println("random initial iPEPS")
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
function optimiseipeps(ipeps::IPEPS{LT}, h, key;f_tol = 1e-6, optimmethod = LBFGS(m = 20)) where LT
    model, D, χ, tol, maxiter = key
    let energy = x -> real(energy(h, IPEPS{LT}(x); χ=χ, tol=tol, maxiter=maxiter))
        res = optimize(x -> energy(x),
            (G, x) -> (G .= Zygote.gradient(energy,x)[1]), 
            ipeps.bulk, optimmethod,
            Optim.Options(f_tol=f_tol, extended_trace=true,
            callback=os->writelog(os, key)),
            )
    end
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))s   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    model, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open("./data/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save("./data/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end