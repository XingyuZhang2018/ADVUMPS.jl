using ADVUMPS

@testset "hamiltonianmodels" begin
    @test Ising() isa HamiltonianModel
end