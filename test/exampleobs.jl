using ADVUMPS
using ADVUMPS: magnetisation,Z,energy,Zofβ,magofβ,eneofβ,magofdβ,num_grad,obs_env
using CUDA
using Random
using Test

@testset "vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    model = Ising()
    for β = 0:0.2:0.8
        env = vumps_env(Ising(),β,10; tol = 1e-20, maxiter = 20, verbose = true)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
    end
end

@testset "up and dowm vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    model = Ising()
    for β = 0.2:0.2:0.8
        @show β
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = 10, tol = 1e-20, maxiter = 20, verbose = true, savefile = false)
        # @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-10)
        # @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-6)
    end
end