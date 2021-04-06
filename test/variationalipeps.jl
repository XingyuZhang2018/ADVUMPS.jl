using Test
using ADVUMPS
using ADVUMPS: energy, num_grad, diaglocalhamiltonian, indexperm_symmetrize
using OMEinsum, Zygote, Random
using Optim, LineSearches
using LinearAlgebra: svd, norm

@testset "non-interacting" begin
    h = diaglocalhamiltonian([1,-1.0])
    as = (rand(3,3,3,3,2) for _ in 1:10)
    @test all(a -> -1 < energy(h,SquareIPEPS(a); χ=5, tol=0.0, maxit=10)/2 < 1, as)

    h = diaglocalhamiltonian([1,-1.0])
    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,2] = randn()
    @test energy(h,SquareIPEPS(a); χ=4, tol=0, maxit=10)/2 ≈ -1

    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,1] = randn()
    @test energy(h,SquareIPEPS(a); χ=4, tol=0, maxit=10)/2 ≈ 1

    a = zeros(2,2,2,2,2) .+ 1e-12 * randn(2,2,2,2,2)
    a[1,1,1,1,2] = a[1,1,1,1,1] = randn()
    @test abs(energy(h,SquareIPEPS(a); χ=4, tol=0, maxit=10)) < 1e-9

    grad = let energy = x -> real(energy(h, SquareIPEPS(x); χ=8, tol=1e-10, maxit=1))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], a, LBFGS(m=20), inplace = false)
    end
    @test grad != Nothing

    hdiag = [0.3,0.1,-0.43]
    h = diaglocalhamiltonian(hdiag)
    a = SquareIPEPS(randn(2,2,2,2,3))
    res = optimiseipeps(a, h; χ=4, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
    e = minimum(res)/2
    @test isapprox(e, minimum(hdiag), atol=1e-3)
end

@testset "TFIsing" begin
    Random.seed!(3)
    h = zeros(2,2,2,2)
    h[1,1,2,2] = h[2,2,1,1] = 1
    h[2,2,2,2] = h[1,1,1,1] = -1
    ipeps = SquareIPEPS(randn(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
    e = minimum(res)
    @test isapprox(e,-1, atol=1e-3)

    h = zeros(2,2,2,2)
    h[1,1,2,2] = h[2,2,1,1] = 1
    h[2,2,2,2] = h[1,1,1,1] = -1
    randu, s,  = svd(randn(2,2))
    h = ein"(((abcd,ai),bj),ck),dl -> ijkl"(h,randu,randu',randu,randu')
    ipeps = SquareIPEPS(randn(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=3, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
    e = minimum(res)
    @test isapprox(e,-1, atol=1e-3)

    # # comparison with results from https://github.com/wangleiphy/tensorgrad
    h = hamiltonian(TFIsing(1.0))
    ipeps = SquareIPEPS(rand(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-3, show_trace=false),))
    e = minimum(res)
    @test isapprox(e, -2.12566, atol = 1e-2)

    h = hamiltonian(TFIsing(0.5))
    ipeps = SquareIPEPS(rand(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-3, show_trace=true),))
    e = minimum(res)
    @test isapprox(e, -2.0312, atol = 1e-2)

    h = hamiltonian(TFIsing(2.0))
    ipeps = SquareIPEPS(rand(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-1, show_trace=true),))
    e = minimum(res)
    @test isapprox(e, -2.5113, atol = 1e-2)
end

@testset "heisenberg" begin
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(3)
    # h = hamiltonian(Heisenberg())
    # ipeps = SquareIPEPS(randn(2,2,2,2,2))
    # a = indexperm_symmetrize(ipeps)
    # res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=20,
    #     optimargs = (Optim.Options(f_tol=1e-6, show_trace=true),))
    # e = minimum(res)
    # @test isapprox(e, -0.66023, atol = 1e-4)

    # h = hamiltonian(Heisenberg(2.0, 2.0, 1.0))
    # ipeps = SquareIPEPS(randn(2,2,2,2,2))
    # a = indexperm_symmetrize(ipeps)
    # res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=100,
    #     optimargs = (Optim.Options(f_tol = 1e-6, show_trace = true),))
    # e = minimum(res)
    # @test isapprox(e, -1.190, atol = 1e-2)

    h = hamiltonian(Heisenberg(0.5, 0.5, 2.0))
    ipeps = SquareIPEPS(randn(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    res = optimiseipeps(a, h; χ=2, tol=1e-10, maxit=20,
        optimargs = (Optim.Options(f_tol = 1e-6, show_trace = true),))
    e = minimum(res)
    @test isapprox(e, -1.0208, atol = 1e-3)
end

@testset "gradient" begin
    Random.seed!(0)
    h = hamiltonian(Heisenberg())
    ipeps = SquareIPEPS(randn(2,2,2,2,2))
    a = indexperm_symmetrize(ipeps)
    gradzygote = first(Zygote.gradient(a) do x
        energy(h,x; χ=2, tol=1e-10, maxit=100)
    end).bulk
    gradnum = num_grad(a.bulk, δ=1e-3) do x
        energy(h, SquareIPEPS(x); χ=2, tol=1e-10, maxit=100)
    end

    @test isapprox(norm(gradzygote-gradnum), 0, atol=1e-3)
end
