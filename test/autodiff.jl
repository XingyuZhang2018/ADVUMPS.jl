using ADVUMPS
using ADVUMPS:eig,leftorth,lefteig,qrpos,lqpos
using Test
using Zygote
using KrylovKit
using LinearAlgebra
using OMEinsum
using ChainRulesCore
using ChainRulesTestUtils

@testset "autodiff" begin
    a = randn(10,10)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(Float64[x 2x; 3x 4x])
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

end

@testset "nonsymmetric eigsolve" begin
    function foo2(x)
        A = [1. 2;3 1] .*x
        λ, l = lefteig(A, l -> ein"a,ab -> b"(l,A), 
                            r -> ein"ab,b -> a"(A,r), rand(2))
        return norm(l) + real(λ)
    end

    function foo3(x)
        A = [1. 2;3 1] .*x
        λ, r = righteig(A, l -> ein"a,ab -> b"(l,A), 
                            r -> ein"ab,b -> a"(A,r), rand(2))
        return norm(r) + real(λ)
    end

    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
    @test Zygote.gradient(foo3, 1)[1] ≈ num_grad(foo3, 1)
end

@testset "symmetric eigsolve" begin
    function foo4(x)
        A = [1. 2 3;2 2 3;3 3 3] .*x
        λ, v = eig(A,v -> ein"ab,b -> a"(A,v), rand(3))
        norm(v) + real(λ)
    end
    @test Zygote.gradient(foo4, 1)[1] ≈ num_grad(foo4, 1)
end

@testset "QR factorization" begin
    function foo5(x)
        A = [1. + 1im 2 3;2 2 3;3 3 3] .*x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo5, 1)[1] ≈ num_grad(foo5, 1)
    test_rrule(qrpos,rand(10,10))
end

@testset "LQ factorization" begin
    function foo6(x)
        A = [1. + 1im 2 3;2 2 3;3 3 3] .*x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test Zygote.gradient(foo6, 1)[1] ≈ num_grad(foo6, 1)
    test_rrule(lqpos,rand(10,10))
end