using ADTensor
using Test
using Zygote, OMEinsum
using ADTensor: model_tensor, tensorfromclassical

@testset "exampletensor" begin
    β = rand()
    @test model_tensor(Ising(),β) ≈ tensorfromclassical([β -β; -β β])
end
