using ADTensor
using Test

@testset "ADTensor.jl" begin
    @testset "autodiff" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end

    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "example tensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "vumps" begin
        println("vumps tests running...")
        include("vumps.jl")
    end
end
