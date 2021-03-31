using ADVUMPS
using ADVUMPS: model_tensor, tensorfromclassical
using Test
using Zygote, OMEinsum

@testset "exampletensor" begin
    β = rand()
    @test model_tensor(Ising(),β) ≈ tensorfromclassical([β -β; -β β])
end
