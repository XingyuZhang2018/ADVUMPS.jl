using ADVUMPS
using BenchmarkTools
using KrylovKit
using LinearAlgebra
using Random
using Test
using TupleTools
using OMEinsum
using OMEinsum: get_size_dict
using Zygote

@testset "Zygote OMEinsum function" begin
    Random.seed!(100)
    D = 2
    d = 2
    E = rand(D,d,D)
    ┬ = rand(D,d,D)
    ┼ = rand(d,d,d,d)
    ┴ = rand(D,d,D)
    Ǝ = rand(D,d,D)

    ctal = ein"abc, adf, begd, ceh, fgh -> "
    function einsum_grad(::EinCode{ixs, iy}, xs, i) where {ixs, iy}
        nixs = TupleTools.deleteat(ixs, i)
        nxs = TupleTools.deleteat(xs, i)
        niy = ixs[i]
        einsum(EinCode(nixs, niy), nxs)
    end
    
    function 田(E, ┬, ┼, ┴, Ǝ)
        ctal(E, ┬, ┼, ┴, Ǝ)[]
    end

    function d田(E, ┬, ┼, ┴, Ǝ, i)
        einsum_grad(ctal, (E, ┬, ┼, ┴, Ǝ), i)
    end
    

    function ⨮(┬, ┼, ┴, Ǝ)
        ein"adf,begd,ceh, fgh -> abc"(┬, ┼, ┴, Ǝ)
    end

    @time d田(E, ┬, ┼, ┴, Ǝ, 1)
    @time ⨮(┬, ┼, ┴, Ǝ)
    # @test dE(E, ┬, ┼, ┴, Ǝ) ≈ ⨮(┬, ┼, ┴, Ǝ)
    @time λs, FLs, info = eigsolve(Ǝ -> d田(E, ┬, ┼, ┴, Ǝ, 1), rand(D,d,D), 1, :LM; ishermitian = false)
    # @show λs[1],info 
    @time λs, FLs, info = eigsolve(Ǝ -> ⨮(┬, ┼, ┴, Ǝ), rand(D,d,D), 1, :LM; ishermitian = false)
    # @show λs[1],info
end