using ADVUMPS
using ADVUMPS:Ising,HamiltonianModel

@testset "hamiltonianmodels" begin
    @test Ising() isa HamiltonianModel
    @test TFIsing(1.0) isa HamiltonianModel
end