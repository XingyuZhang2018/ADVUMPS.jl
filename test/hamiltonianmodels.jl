using ADVUMPS
using ADVUMPS:Ising,HamiltonianModel

@testset "hamiltonianmodels" begin
    @test Ising() isa HamiltonianModel
end