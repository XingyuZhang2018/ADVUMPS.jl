using ADVUMPS
using ADVUMPS: num_grad, safetr
using ADVUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,bigleftenv,bigrightenv
using ADVUMPS: energy,magofdβ
using ChainRulesCore
using ChainRulesTestUtils
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote

CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    a = atype(randn(2,2))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(atype(Float64[x 2x; 3x 4x]))
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "Zygote.@ignore" begin
    function foo2(x)
        return x^2
    end
    function foo3(x)
        return x^2 + Zygote.@ignore x^3
    end
    @test foo2(1) != foo3(1)
    @test Zygote.gradient(foo2,1)[1] ≈ Zygote.gradient(foo3,1)[1]
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo5(x)
        A = M .* x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1], num_grad(foo5, 1), atol = 1e-5)
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo6(x)
        A = M .*x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
end

@testset "linsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    D,d = 10,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]

    dL = atype(rand(D,D))
    dL -= tr(ein"ab,ad -> bd"(L,dL)) * L
    @test tr(ein"ab,ad -> bd"(L,dL)) ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> ein"abcd,cd -> ab"(工,R), dL, -λL, 1)
    @test tr(ein"ab,ad -> bd"(ξL,L)) ≈ 0 atol = 1e-9

    dR = atype(rand(D,D))
    dR -= tr(ein"ab,ad -> bd"(R,dR)) * R
    @test tr(ein"ab,ad -> bd"(R,dR)) ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> ein"ab,abcd -> cd"(L,工), dR, -λR, 1)
    @test tr(ein"ab,ad -> bd"(ξR,R)) ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    D = 10
    A = atype(rand(D,D,D))
    B = atype(rand(D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abd -> cd"(C,C)
        F = ein"ab,ac -> bc"(D,D)
        return safetr(E)/safetr(F)
    end 
    @time @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10
    A = atype(rand(dtype,D,d,D))
    
    AL, = leftorth(A)
    _, AR = rightorth(A)

    S = atype(rand(D,d,D,D,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _,FL = leftenv(AL, M)
        A = ein"γcη,ηcγαaβ,daα -> βd"(FL,S,FL)
        B = ein"γcη,ηca -> γa"(FL,FL)
        return safetr(A)/safetr(B)
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FR = rightenv(AR, M)
        A = ein"γcη,ηcγαaβ,daα -> βd"(FR,S,FR)
        B = ein"γcη,ηca -> γa"(FR,FR)
        return safetr(A)/safetr(B)
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))
    
    AL,C = leftorth(A)
    λL,FL = leftenv(AL, M)
    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)
    AC = ein"asc,cb -> asb"(AL,C)

    S = atype(rand(D,d,D,D,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _, AC = ACenv(AC, FL, M, FR)
        A = ein"γcη,ηcγαaβ,daα -> βd"(AC,S,AC)
        B = ein"γcη,ηca -> γa"(AC,AC)
        return safetr(A)/safetr(B)
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    S = atype(rand(D,D,D,D))
    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FL = leftenv(AL, M)
        _,FR = rightenv(AR, M)
        _, C = Cenv(C, FL, FR)
        A = ein"γη,ηγαa,bα -> ab"(C,S,C)
        B = ein"ab,ac -> bc"(C,C)
        return safetr(A)/safetr(B)
    end
    @time @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    A = atype(rand(dtype,D,d,D))

    AL, = leftorth(A)
    _, AR = rightorth(A)

    S = atype(rand(D,d,d,D,D,d,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _,FL4 = bigleftenv(AL, M)
        A = ein"abcd,ibcdefgh,efgh -> ai"(FL4,S,FL4)
        B = ein"abcd,ebcd -> ae"(FL4,FL4)
        return safetr(A)/safetr(B)
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    S = atype(rand(D,d,d,D,D,d,d,D))
    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FR4 = bigrightenv(AR, M)
        A = ein"abcd,ibcdefgh,efgh -> ai"(FR4,S,FR4)
        B = ein"abcd,ebcd -> ae"(FR4,FR4)
        return safetr(A)/safetr(B)
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "vumps with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    β,D = 0.5,10
    foo1 = β -> -log(Z(vumps_env(Ising(),β,D; atype = atype, verbose = true)))
    @test Zygote.gradient(foo1,β)[1] ≈ energy(vumps_env(Ising(),β,D;atype = atype),Ising(), β) atol = 1e-6
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    foo2 = β -> magnetisation(vumps_env(Ising(),β,D; atype = atype), Ising(), β)
    @test isapprox(num_grad(foo2,β), magofdβ(Ising(),β), atol = 1e-3)
    @test isapprox(Zygote.gradient(foo2,β)[1], magofdβ(Ising(),β), atol = 1e-6)
end