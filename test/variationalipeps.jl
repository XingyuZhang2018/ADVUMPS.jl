using ADVUMPS
using ADVUMPS: energy, num_grad, diaglocal, optcont
using CUDA
using LinearAlgebra: svd, norm
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "non-interacting with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    χ = 20
    D = 2
    model = diaglocal([1,-1.0])
    h = atype(hamiltonian(model))
    # as = (atype(rand(3,3,3,3,2)) for _ in 1:10)
    # @test all(a -> -1 < energy(h, model, SquareIPEPS(a); χ=5, tol=1e-10, maxiter=20)/2 < 1, as)

    a = zeros(D,D,D,D,2) .+ 1e-12 * rand(D,D,D,D,2)
    a[1,1,1,1,2] = randn()
    # oc = optcont(D, χ)
    @test energy(h, model, SquareIPEPS(atype(a)); χ=χ, tol=1e-10, maxiter=20)/2 ≈ -1

    # a = zeros(2,2,2,2,2) .+ 1e-12 * rand(2,2,2,2,2)
    # a[1,1,1,1,1] = randn()
    # @test energy(h, model, SquareIPEPS(atype(a)); χ=4, tol=1e-10, maxiter=20)/2 ≈ 1

    # a = zeros(2,2,2,2,2) .+ 1e-12 * rand(2,2,2,2,2)
    # a[1,1,1,1,2] = a[1,1,1,1,1] = randn()
    # @test abs(energy(h,model,SquareIPEPS(atype(a)); χ=4, tol=1e-10, maxiter=20)) < 1e-9

    # grad = let energy = x -> real(energy(h, model, SquareIPEPS(atype(x)); χ=4, tol=1e-10, maxiter=20))
    #     res = optimize(energy,
    #         Δ -> Zygote.gradient(energy,Δ)[1], a, LBFGS(m=20), inplace = false)
    # end
    # @test grad != Nothing

    # Random.seed!(100)
    # hdiag = [0.3,-0.43]
    # model = diaglocal(hdiag)
    # ipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6, atype = atype)
    # e = minimum(res)/2
    # @test isapprox(e, minimum(hdiag), atol=1e-3)
end

@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(0)
    # D,χ = 2,10
    # model = TFIsing(1.0)
    # h = atype(hamiltonian(model))
    # ipeps, key = init_ipeps(model; D=D, χ=χ, tol=1e-10, maxiter=10)
    # oc = optcont(D, χ)
    # # energy(h, ipeps, oc, key; verbose = true)
    # gradzygote = first(Zygote.gradient(ipeps.bulk) do x
    #     energy(h,SquareIPEPS(atype(x)), oc, key; verbose = true)
    # end)
    # gradnum = num_grad(ipeps.bulk, δ=1e-3) do x
    #     energy(h,SquareIPEPS(atype(x)), oc, key; verbose = true)
    # end
    # @test isapprox(gradzygote, gradnum, atol=1e-5)

    Random.seed!(0)
    D,χ = 2,20
    model = Heisenberg(1.0,1.0,1.0)
    h = atype(hamiltonian(model))
    ipeps, key = init_ipeps(model; D=D, χ=χ, tol=1e-20, maxiter=10)
    oc = optcont(D, χ)
    energy(h, ipeps, oc, key; verbose = true)
    # gradzygote = first(Zygote.gradient(ipeps.bulk) do x
    #     energy(h,SquareIPEPS(atype(x)), oc, key; verbose = true)
    # end)
    # gradnum = num_grad(ipeps.bulk, δ=1e-3) do x
    #     energy(h,SquareIPEPS(atype(x)), oc, key; verbose = true)
    # end
    # @test isapprox(gradzygote, gradnum, atol=1e-5)
end

@testset "TFIsing with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(3)
    model = TFIsing(1.0)
    ipeps, key = init_ipeps(model;atype = atype, D=2, χ=20, tol=1e-10, maxiter=10)
    res = optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = false)
    e = minimum(res)
    @test isapprox(e, -2.12566, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(0.5)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6, atype = atype)
    # e = minimum(res)
    # @test isapprox(e, -2.0312, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(2.0)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6, atype = atype)
    # e = minimum(res)
    # @test isapprox(e, -2.5113, atol = 1e-2)
end

@testset "heisenberg with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(100)
    model = Heisenberg(1.0,1.0,1.0)
    ipeps, key = init_ipeps(model; atype = atype, D=2, χ=20, tol=1e-10, maxiter=10)
    res = optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = true)
    e = minimum(res)
    @test isapprox(e, -0.66023, atol = 1e-4)

    # Random.seed!(100)
    # model = Heisenberg(1.0,2.0,2.0)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6, atype = atype)
    # e = minimum(res)
    # @test isapprox(e, -1.190, atol = 1e-3)

    # Random.seed!(100)
    # model = Heisenberg(2.0,0.5,0.5)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6, atype = atype)
    # e = minimum(res)
    # @test isapprox(e, -1.0208, atol = 1e-3)
end
 