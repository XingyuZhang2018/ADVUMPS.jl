using ADTensor

@testset "hamiltonianmodels" begin
    @test Ising() isa HamiltonianModel
end