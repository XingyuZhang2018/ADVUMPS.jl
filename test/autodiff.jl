using ADVUMPS
using ADVUMPS: num_grad
using ADVUMPS: qrpos,lqpos,mysvd,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,bigleftenv,bigrightenv,obs_leftenv,obs_rightenv
using ADVUMPS: energy,magofdβ,obs_env
using ChainRulesCore
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using OMEinsum: get_size_dict, optimize_greedy, MinSpaceOut, MinSpaceDiff
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    a = atype(randn(dtype, 2,2))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo = x -> sum(atype(Float64[x 2x; 3x 4x]))
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1)
end

@testset "Zygote.@ignore" begin
    function foo1(x)
        return x^2
    end
    function foo2(x)
        return x^2 + Zygote.@ignore x^3
    end
    @test foo1(1) != foo2(1)
    @test Zygote.gradient(foo1,1)[1] ≈ Zygote.gradient(foo2,1)[1]
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test norm(Zygote.gradient(foo, M)[1] - num_grad(foo, M)) ≈ 0 atol = 1e-8
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test norm(Zygote.gradient(foo, M)[1] - num_grad(foo, M)) ≈ 0 atol = 1e-8
end

@testset "svd with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    M = atype(rand(dtype, 10,10))
    function foo(x)
        A = M .*x
        U, S, V = mysvd(A)
        return norm(U) + norm(V)
    end
    @test isapprox(Zygote.gradient(foo, 1.0+1.0im)[1], num_grad(foo, 1.0+1.0im), atol = 1e-5)
end

@testset "linsolve with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D,d = 3,2                                        # a───┬───c 
    ┬ = atype(rand(dtype, D,d,D))                    # │   b   │ 
    工 = ein"abc,dbe -> acde"(┬,conj(┬))             # d───┴───e 
    λcs, cs, info = eigsolve(c -> ein"ad,acde -> ce"(c,工), atype(rand(dtype, D,D)), 1, :LM)
    λc, c = λcs[1], cs[1]
    λↄs, ↄs, info = eigsolve(ↄ -> ein"acde,ce -> ad"(工,ↄ), atype(rand(dtype, D,D)), 1, :LM)
    λↄ, ↄ = λↄs[1], ↄs[1]

    dc = atype(rand(ComplexF64, D,D))
    dc -= Array(ein"ab,ab -> "(conj(c), dc))[] * c
    @test Array(ein"ab,ab -> "(conj(c), dc))[] ≈ 0 atol = 1e-9
    ξc, info = linsolve(ↄ -> ein"acde,ce -> ad"(工,ↄ), conj(dc), -λc, 1)
    @test Array(ein"ab,ab -> "(c, ξc))[] ≈ 0 atol = 1e-9

    dↄ = atype(rand(ComplexF64, D,D))
    dↄ -= Array(ein"ab,ab -> "(conj(ↄ),dↄ))[] * ↄ
    @test Array(ein"ab,ab -> "(conj(ↄ),dↄ))[] ≈ 0 atol = 1e-9
    ξↄ, info = linsolve(c -> ein"ad,acde -> ce"(c,工), conj(dↄ), -λↄ, 1)
    @test Array(ein"ab,ab -> "(ↄ,ξↄ))[] ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with  $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D = 5
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)
        F = ein"ab,ab -> "(D,D)
        return norm(Array(E)[]/Array(F)[])
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3
    A = atype(rand(dtype,D,d,D))
    
    ALu, = leftorth(A)
    ALd, = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)

    S = atype(rand(ComplexF64,D,d,D,D,d,D))
    M = atype(rand(ComplexF64,d,d,d,d))
    function foo1(M)
        _,FL = leftenv(ALu, ALd, M)
        A = ein"(abc,abcdef),def -> "(FL,S,conj(FL))
        B = ein"abc,abc -> "(FL,conj(FL))
        return norm(Array(A)[]/Array(B)[])
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-8

    function foo2(M)
        _,FR = rightenv(ARu, ARd, M)
        A = ein"(abc,abcdef),def -> "(FR,S,FR)
        B = ein"abc,abc -> "(FR,FR)
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-8
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3

    β = rand()
    A = atype(rand(dtype,D,d,D))
    M = atype(rand(ComplexF64,d,d,d,d))
    
    AL,C = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    _, AR = rightorth(A)
    λR,FR = rightenv(AR, AR, M)
    AC = ein"abc,cd -> abd"(AL,C)
    S = atype(rand(ComplexF64,D,d,D,D,d,D))
    
    function foo1(M)
        _, AC = ACenv(AC, FL, M, FR)
        A = ein"(abc,abcdef),def -> "(AC,S,conj(AC))
        B = ein"abc,abc -> "(AC,conj(AC))
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-8

    S = atype(rand(ComplexF64,D,D,D,D))
    function foo2(M)
        _,FL = leftenv(AL, AL, M)
        _,FR = rightenv(AR, AR, M)
        _, C = Cenv(C, FL, FR)
        A = ein"(ab,abcd),cd -> "(C,S,C)
        B = ein"ab,ab -> "(C,C)
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-8
end

@testset "ACCtoALAR with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D, d = 3, 2

    A = atype(rand(dtype, D, d, D))
    S1 = atype(rand(ComplexF64, D, d, D, D, d, D))
    S2 = atype(rand(ComplexF64, D, D, D, D))

    ALo,Co = leftorth(A)
    _, ARo = rightorth(ALo)
    M = atype(rand(ComplexF64,d,d,d,d))
    _, FLo = leftenv(ALo, ALo, M)
    _, FRo = rightenv(ARo, ARo, M)

    ACo = ein"abc,cd -> abd"(ALo,Co)
    _, Co = Cenv(Co, FLo, FRo)
    function foo1(M)
        _, AC = ACenv(ACo, FLo, M, FRo)
        AL, AR, _, _ = ACCtoALAR(AC, Co)
        s = 0
        A = ein"(abc,abcdef),def -> "(AL, S1, AL)
        B = ein"abc,abc -> "(AL, AL)
        s += norm(Array(A)[]/Array(B)[])
        A = ein"(abc,abcdef),def -> "(AR, S1, AR)
        B = ein"abc,abc -> "(AR, AR)
        s += norm(Array(A)[]/Array(B)[])
        A = ein"(abc,abcdef),def -> "(AC, S1, AC)
        B = ein"abc,abc -> "(AC, AC)
        s += norm(Array(A)[]/Array(B)[])
        return s
    end
    @test isapprox(Zygote.gradient(foo1, M)[1], num_grad(foo1, M), atol=1e-3)
end

@testset "few steps vumps with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D, d = 3, 2

    A = atype(rand(dtype, D, d, D))
    S1 = atype(rand(ComplexF64, D, d, D, D, d, D))
    S2 = atype(rand(ComplexF64, D, D, D, D))

    ALo,Co = leftorth(A)
    _, ARo = rightorth(ALo)
    M = atype(rand(ComplexF64,d,d,d,d))
    _, FLo = leftenv(ALo, ALo, M)
    _, FRo = rightenv(ARo, ARo, M)

    ACo = ein"abc,cd -> abd"(ALo,Co)
    _, Co = Cenv(Co, FLo, FRo)
    function foo1(M)
        _, AC = ACenv(ACo, FLo, M, FRo)
        AL, AR, _, _ = ACCtoALAR(AC, Co)
        _, FL = leftenv(ALo, AL, M, FLo)
        _, FR = rightenv(ARo, AR, M, FRo)
        _, AC = ACenv(AC, FL, M, FR)
        _, C = Cenv(Co, FL, FR)
        AL, AR, _, _ = ACCtoALAR(AC, C)
        s = 0
        A = ein"(abc,abcdef),def -> "(AL, S1, AL)
        B = ein"abc,abc -> "(AL, AL)
        s += norm(Array(A)[]/Array(B)[])
        A = ein"(abc,abcdef),def -> "(AR, S1, AR)
        B = ein"abc,abc -> "(AR, AR)
        s += norm(Array(A)[]/Array(B)[])
        A = ein"(abc,abcdef),def -> "(AC, S1, AC)
        B = ein"abc,abc -> "(AC, AC)
        s += norm(Array(A)[]/Array(B)[])
        A = ein"(ab,abcd),cd -> "(C,S2,C)
        B = ein"ab,ab -> "(C,C)
        s += norm(Array(A)[]/Array(B)[])
        return s
    end
    @test isapprox(Zygote.gradient(foo1, M)[1], num_grad(foo1, M), atol=1e-3)
end

@testset "vumps with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    χ = 10
    model = Ising()
    
    function foo1(β) 
        M = atype(model_tensor(model, β))
        env = vumps_env(model, M; χ=χ, tol=1e-15, maxiter=10, verbose = true, savefile = false, atype = atype)
        magnetisation(env,Ising(),β)
    end
    for β = 0.2:0.2:0.8
        @show β
        @test Zygote.gradient(foo1, β)[1] ≈ magofdβ(model,β) atol = 1e-8
    end

    function foo2(β)
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, χ = χ, tol = 1e-20, maxiter = 10, verbose = true, savefile = false)
        magnetisation(env,Ising(),β)
    end
    for β = 0.1:0.1:0.4
        @show β
        @test Zygote.gradient(foo2, β)[1] ≈ magofdβ(model,β) atol = 1e-8
    end
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3

    A = atype(rand(dtype,D,d,D))

    ALu, = leftorth(A)
    ALd, = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    S = atype(rand(ComplexF64,D,d,d,D,D,d,d,D))
    M = atype(rand(ComplexF64,d,d,d,d))
    function foo1(M)
        _,FL4 = bigleftenv(ALu, ALd, M)
        A = ein"(abcd,abcdefgh),efgh -> "(FL4,S,FL4)
        B = ein"abcd,abcd -> "(FL4,FL4)
        return norm(Array(A)[]/Array(B)[])
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _,FR4 = bigrightenv(ARu, ARd, M)
        A = ein"(abcd,abcdefgh),efgh -> "(FR4,S,FR4)
        B = ein"abcd,abcd -> "(FR4,FR4)
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "obs_leftenv and obs_rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3
    A = atype(rand(dtype,D,d,D))
    
    ALu, = leftorth(A)
    ALd, = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)

    S = atype(rand(ComplexF64,D,d,D,D,d,D))
    M = atype(rand(ComplexF64,d,d,d,d))
    function foo1(M)
        _,FL = obs_leftenv(ALu, ALd, M)
        A = ein"(abc,abcdef),def -> "(FL,S,conj(FL))
        B = ein"abc,abc -> "(FL,conj(FL))
        return norm(Array(A)[]/Array(B)[])
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-8

    function foo2(M)
        _,FR = obs_rightenv(ARu, ARd, M)
        A = ein"(abc,abcdef),def -> "(FR,S,FR)
        B = ein"abc,abc -> "(FR,FR)
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-8
end