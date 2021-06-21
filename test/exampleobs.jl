using ADVUMPS
using ADVUMPS: magnetisation,Z,energy,Zofβ,magofβ,eneofβ,magofdβ,num_grad,obs_env
using CUDA
using Random
using Test

@testset "vumps with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    for β = 0.6
        env = vumps_env(Ising(),β,10; atype = atype)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-10)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-5)
    end
end

@testset "up and dowm vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(00)
    model = Ising()
    for β = 0.2:0.2:1.0
        @show β
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = 20, tol = 1e-10, maxiter = 20, verbose = true, savefile = true)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-5)
    end
end