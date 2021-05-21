using Test
using ADVUMPS
using Random

@testset "vumpsruntime" begin
    @test SquareLattice <: AbstractLattice

    M = rand(2,2,2,2)
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
    env = vumps(rt; tol=1e-10, maxiter=100)
    @test env !== nothing
end