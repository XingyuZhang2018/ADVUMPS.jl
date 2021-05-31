using ADVUMPS
using ADVUMPS:qrpos,lqpos,num_grad,Ising
using BenchmarkTools
using ChainRulesCore
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote

@testset "Zygote" begin
    a = randn(10,10)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(Float64[x 2x; 3x 4x])
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    function foo3(x)
        x = 2*x + x
        return x
    end
    @test Zygote.gradient(foo3, 1)[1] ≈ num_grad(foo3, 1)
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

@testset "Zygote OMEinsum function" begin
    工 = rand(2,2,2,2)
    c = rand(2,2)
    ↄ = rand(2,2)
    function c工ↄ(c,工,ↄ)
        ein"ab,abcd,cd ->"(c,工,ↄ)[]
    end
    
    fc(c) = c工ↄ(c,工,ↄ)
    function dc(c,工,ↄ)
        fc(c) = c工ↄ(c,工,ↄ)
        Zygote.gradient(fc, c)[1]
    end
    @test dc(c,工,ↄ) ≈ num_grad(fc, c)

    王 = rand(2,2,2,2,2,2)
    E = rand(2,2,2)
    Ǝ = rand(2,2,2)
    function 田(E,王,Ǝ)
        ein"abc,abcdef,def ->"(E,王,Ǝ)[]
    end

    fE(E) = 田(E,王,Ǝ)

    function dE(E,王,Ǝ)
        f(E) = 田(E,王,Ǝ)
        Zygote.gradient(f, E)[1]
    end
    @test dE(E,王,Ǝ) ≈ num_grad(fE, E)
end

@testset "QR factorization" begin
    M = rand(10,10)
    function foo5(x)
        A = M.*x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1], num_grad(foo5, 1), atol = 1e-5)
end

@testset "LQ factorization" begin
    M = rand(10,10)
    function foo6(x)
        A = M .*x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
end

@testset "eigsolve and linsolve" begin
    Random.seed!(100)
    N = 1000
    A = rand(N,N)
    λLs, Ls, info = eigsolve(L -> ein"a,ab -> b"(L,A), rand(N), 2, :LM)
    λL = real(λLs[1])
    L = real(Ls[1])
    # @show λLs[1] λLs[2]
    λRs, Rs, info = eigsolve(R -> ein"ab,b -> a"(A,R), rand(N), 2, :LM)
    λR = real(λRs[1])
    R = real(Rs[1])
    @test λL ≈ λR 
    @test ein"a,a -> "(L,L)[] ≈ 1 
    @test ein"a,a -> "(R,R)[] ≈ 1 

    dL = rand(N)
    dL -= ein"a,a -> "(L,dL)[] .* L
    @test isapprox(ein"a,a -> "(L,dL)[], 0, atol = 1e-9)
    ξL, info = linsolve(R -> ein"ab,b -> a"(A,R), dL, -λL, 1)
    @test isapprox(ein"a,a -> "(ξL,L)[], 0, atol = 1e-9)

    dR = rand(N)
    dR -= ein"a,a -> "(R,dR)[] .* R
    @test isapprox(ein"a,a -> "(R,dR)[], 0, atol = 1e-9)
    ξR, info = linsolve(L -> ein"a, ab -> b"(L,A), dR, -λR, 1)
    @test isapprox(ein"a,a -> "(ξR,R)[], 0, atol = 1e-9)
end