using ADTensor
using Test
using Zygote
using KrylovKit
using LinearAlgebra: norm

@testset "autodiff" begin
    a = randn(10,10)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo = x -> sum(Float64[x 2x; 3x 4x])
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1)

    function foo2(x)
        h = [x x;x x]
        _, As, _ = eigsolve(x->h*x,rand(2,2),1, :LM; ishermitian = false)
        A = abs(sum(As[1]))
    end
    println(foo2(1),"  ",num_grad(foo2, 1))
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end