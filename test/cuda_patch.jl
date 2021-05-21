using ADVUMPS
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
using BenchmarkTools
using TensorOperations
CUDA.allowscalar(false)

@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray, Array], dtype in [Float64]
    Random.seed!(100)
    d = 9
    D = 50
    AL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    FL = atype(rand(dtype, D, d, D))
    @time ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [CuArray, Array], dtype in [Float64]
    Random.seed!(100)
    D = 2^20
    function A(x)
        x .- sum(x)
    end
    rhs = atype(rand(dtype, D))
    @time linsolve(A, rhs)

    d = 16
    D = 50
    FL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    AL = atype(rand(dtype, D, d, D))
    @time eigsolve(FL -> ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
end
 