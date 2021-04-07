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
    energy(h, ipeps; χ, tol, maxit)

return the energy of the `ipeps` 2-site hamiltonian `h` and calculated via a
ctmrg with parameters `χ`, `tol` and `maxit`.
"""
function energy(h::AbstractArray{T,4}, ipeps::IPEPS; χ::Int, tol::Real, maxit::Int) where T
    ipeps = indexperm_symmetrize(ipeps)  # NOTE: this is not good
    D = getd(ipeps)^2
    s = gets(ipeps)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(ipeps.bulk, conj(ipeps.bulk))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    # mkpath("./data/")
    # chkp_file = "./data/D$(D)_χ$(χ).jld2"
    # if isfile(chkp_file)                               
    #     rt = SquareVUMPSRuntime(M, chkp_file, D)   
    # else
    #     rt = SquareVUMPSRuntime(M, Val(:random), D)
    # end
    rt = SquareVUMPSRuntime(a, Val(:random), χ)
    env = vumps(rt; tol=tol, maxit=maxit)
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
    optimiseipeps(ipeps, h; χ, tol, maxit, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using ctmrg with parameters `χ`, `tol` and `maxit`.
"""
function optimiseipeps(ipeps::IPEPS{LT}, h; χ::Int, tol::Real, maxit::Int,
        optimargs = (),
        optimmethod = LBFGS(m = 20)) where LT
    bulk = ipeps.bulk
    let energy = x -> real(energy(h, IPEPS{LT}(x); χ=χ, tol=tol, maxit=maxit))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], bulk, optimmethod, inplace = false, optimargs...)
    end
end
