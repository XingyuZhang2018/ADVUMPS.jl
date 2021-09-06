using ADVUMPS
using ADVUMPS: magnetisation,Z,energy,Zofβ,magofβ,eneofβ,magofdβ,num_grad,obs_env,norm_FL,norm_FR
using CUDA
using OMEinsum
using Plots
using Random
using Test

@testset "vumps with $atype" for atype in [Array]
    Random.seed!(100)
    model = Ising()
    for β = 0.2:0.2:0.8
        M = atype(model_tensor(model, β))
        env = vumps_env(model, M; χ = 10, tol=1e-20, maxiter=10, verbose = true, savefile = false, atype = atype)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-3)
    end
end

@testset "up and dowm vumps with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    model = Ising()
    for β = 0.2:0.2:0.4
        @show β
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = 10, tol = 1e-20, maxiter = 10, verbose = true, savefile = false)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-3)
    end
end