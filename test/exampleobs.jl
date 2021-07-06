using ADVUMPS
using ADVUMPS: magnetisation,Z,energy,Zofβ,magofβ,eneofβ,magofdβ,num_grad,obs_env
using CUDA
using Random
using Test

@testset "up and dowm vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(1000)
    model = Ising()
    for β = 0.4
        @show β
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = 10, tol = 1e-20, maxiter = 10, verbose = true, savefile = true)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-3)
    end
end