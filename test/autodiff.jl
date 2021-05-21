using ADVUMPS
using ADVUMPS:qrpos,lqpos,num_grad,leftorth,rightorth,leftenv,rightenv
using ChainRulesCore
using ChainRulesTestUtils
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote

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
    @time @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
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
    dL -= ein"ab,ab -> "(L,dL)[] * L
    @test ein"ab,ab -> "(L,dL)[] ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> ein"abcd,cd -> ab"(工,R), dL, -λL, 1)
    @test ein"ab,ab -> "(ξL,L)[] ≈ 0 atol = 1e-9

    dR = atype(rand(D,D))
    dR -= ein"ab,ab -> "(R,dR)[] .* R
    @test ein"ab,ab -> "(R,dR)[] ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> ein"ab,abcd -> cd"(L,工), dR, -λR, 1)
    @test ein"ab,ab -> "(ξR,R)[] ≈ 0 atol = 1e-9
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))
    
    AL, = leftorth(A)
    λL,FL = leftenv(AL, M)

    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)

    S = atype(rand(D,d,D,D,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _,FL = leftenv(AL, M)
        @show ein"γcη,ηcγαaβ,βaα -> "(FL,S,FL)[]/ein"γcη,ηcγ -> "(FL,FL)[]
        return ein"γcη,ηcγαaβ,βaα -> "(FL,S,FL)[]/ein"γcη,ηcγ -> "(FL,FL)[]
    end 
    function foo2(β)
        B = A * β
        return ein"abc,abc -> "(B,B)[1]
    end 
    @show Zygote.gradient(foo2, 1)[1]
    # @show Zygote.gradient(foo1, 1)[1]
    # @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    # function foo2(β)
    #     M = atype(model_tensor(Ising(),β))
    #     _,FR = rightenv(AR, M)
    #     return ein"γcη,ηcγαaβ,βaα -> "(FR,S,FR)[]/ein"γcη,ηcγ -> "(FR,FR)[]
    # end
    # @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "vumps unit test with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)

    d = 2
    D = 2

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))
    
    AL,C = leftorth(A)
    λL,FL = leftenv(AL, M)
    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)
    
    S = rand(D,d,D,D,d,D)
    function foo4(β)
        M = model_tensor(Ising(),β)
        AC = ein"asc,cb -> asb"(AL,C)
        μ1, AC = ACenv(AC, FL, M, FR)
        return ein"γcη,ηcγαaβ,βaα -> "(AC,S,AC)[]/ein"γcη,ηcγ -> "(AC,AC)[]
    end
    @test isapprox(Zygote.gradient(foo4, 0.1)[1],num_grad(foo4, 0.1), atol = 1e-9)

    S = rand(D,D,D,D)
    function foo5(β)
        M = model_tensor(Ising(),β)    
        λL,FL = leftenv(AL, M)
        λR,FR = rightenv(AR, M)
        μ1, C = Cenv(C, FL, FR)
        return ein"γη,ηγαβ,βα -> "(C,S,C)[]/ein"γη,ηγ -> "(C,C)[]
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1],num_grad(foo5, 1), atol = 1e-9)
end

@testset "vumps" begin
    β,D = 0.5,10
    foo1 = β -> -log(Z(vumps_env(Ising(),β,D)))
    @test isapprox(Zygote.gradient(foo1,β)[1], energy(vumps_env(Ising(),β,D),Ising(), β), atol = 1e-6)
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-9)

    foo2 = β -> magnetisation(vumps_env(Ising(),β,D), Ising(), β)
    @test isapprox(num_grad(foo2,β), magofdβ(Ising(),β), atol = 1e-3)
    @test isapprox(Zygote.gradient(foo2,β)[1], magofdβ(Ising(),β), atol = 1e-6)
end