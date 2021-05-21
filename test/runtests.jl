using ADVUMPS
using Test

@testset "ADVUMPS.jl" begin
    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "example tensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "fixedpoint" begin
        println("fixedpoint tests running...")
        include("fixedpoint.jl")
    end

    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "vumpsruntime" begin
        println("vumpsruntime tests running...")
        include("vumpsruntime.jl")
    end

    @testset "autodiff" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end

    @testset "variationalipeps" begin
        println("variationalipeps tests running...")
        include("variationalipeps.jl")
    end
end
