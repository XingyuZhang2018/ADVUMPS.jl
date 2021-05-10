using Test 
using ADVUMPS
using ADVUMPS: energy, num_grad, diaglocal
using OMEinsum, Zygote, Random
using Optim, LineSearches
using LinearAlgebra: svd, norm

@testset "non-interacting" begin
    Random.seed!(100)
    model = diaglocal([1,-1.0])
    h = hamiltonian(model)
    as = (rand(3,3,3,3,2) for _ in 1:10)
    @test all(a -> -1 < energy(h, model, SquareIPEPS(a); χ=5, tol=1e-10, maxiter=10)/2 < 1, as)

    h = hamiltonian(diaglocal([1,-1.0]))
    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,2] = randn()
    @test energy(h, model, SquareIPEPS(a); χ=4, tol=1e-10, maxiter=10)/2 ≈ -1

    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,1] = randn()
    @test energy(h, model, SquareIPEPS(a); χ=4, tol=1e-10, maxiter=10)/2 ≈ 1

    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,2] = a[1,1,1,1,1] = randn()
    @test abs(energy(h,model,SquareIPEPS(a); χ=4, tol=1e-10, maxiter=10)) < 1e-9

    grad = let energy = x -> real(energy(h, model, SquareIPEPS(x); χ=4, tol=1e-10, maxiter=10))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], a, LBFGS(m=20), inplace = false)
    end
    @test grad != Nothing

    hdiag = [0.3,-0.43]
    model = diaglocal(hdiag)
    ipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    res = optimiseipeps(ipeps, key; f_tol = 1e-6)
    e = minimum(res)/2
    @test isapprox(e, minimum(hdiag), atol=1e-3)
end

@testset "gradient" begin
    Random.seed!(0)
    model = TFIsing(1.0)
    h = hamiltonian(model)
    ipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    gradzygote = first(Zygote.gradient(ipeps) do x
        energy(h,model,x; χ=4, tol=1e-10, maxiter=20)
    end).bulk
    gradnum = num_grad(ipeps.bulk, δ=1e-3) do x
        energy(h,model,SquareIPEPS(x); χ=4, tol=1e-10, maxiter=20)
    end
    @test isapprox(gradzygote, gradnum, atol=1e-3)

    Random.seed!(3)
    model = Heisenberg()
    h = hamiltonian(model)
    ipeps, key = init_ipeps( model; D=2, χ=4, tol=1e-10, maxiter=20)
    gradzygote = first(Zygote.gradient(ipeps) do x
        energy(h,model,x; χ=4, tol=1e-10, maxiter=20)
    end).bulk
    gradnum = num_grad(ipeps.bulk, δ=1e-3) do x
        energy(h,model, SquareIPEPS(x); χ=4, tol=1e-10, maxiter=20)
    end
    @test isapprox(gradzygote , gradnum, atol=1e-3)
end

@testset "TFIsing" begin
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(3)
    model = TFIsing(1.0)
    ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    res = optimiseipeps(ipeps, key; f_tol = 1e-6, verbose = true)
    e = minimum(res)
    @test isapprox(e, -2.12566, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(0.5)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -2.0312, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(2.0)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -2.5113, atol = 1e-2)
end

@testset "heisenberg" begin
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(100)
    model = Heisenberg(1.0,1.0,1.0)
    ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    res = optimiseipeps(ipeps, key; f_tol = 1e-6, verbose = true)
    e = minimum(res)
    @test isapprox(e, -0.66023, atol = 1e-4)

    # Random.seed!(100)
    # model = Heisenberg(1.0,2.0,2.0)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -1.190, atol = 1e-3)

    # Random.seed!(100)
    # model = Heisenberg(2.0,0.5,0.5)
    # ipeps, key = init_ipeps(model; D=2, χ=20, tol=1e-10, maxiter=20)
    # res = optimiseipeps(ipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -1.0208, atol = 1e-3)
end
