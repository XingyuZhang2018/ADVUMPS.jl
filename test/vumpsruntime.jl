using Test
using ADVUMPS
using ADVUMPS: ACenv, Cenv, ACCtoALAR, leftenv, rightenv
using Random
using OMEinsum
using Zygote
using Optim, LineSearches
# @testset "vumpsruntime" begin
#     @test SquareLattice <: AbstractLattice

#     M = rand(ComplexF64,2,2,2,2)
#     rt = SquareVUMPSRuntime(M, Val(:random), 2)
#     env = vumps(rt; tol=1e-10, maxiter=100)
#     @test env !== nothing
# end

@testset "AD err" begin
    β = log(1+sqrt(2))/2
    M = model_tensor(Ising(), β)
    D = 10
    rt = SquareVUMPSRuntime(M, Val(:random), D)
    M,AL,C,AR,FL,FR = rt.M,rt.AL,rt.C,rt.AR,rt.FL,rt.FR
    AC = ein"abc,cd -> abd"(AL,C)
    # for _ in 1:100
    #     AC = ein"abc,cd -> abd"(AL,C)
    #     _, ACp = ACenv(AC, FL, M, FR)
    #     _, Cp = Cenv(C, FL, FR)
    #     ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    #     _, FL = leftenv(AL, ALp, M, FL)
    #     _, FR = rightenv(AR, ARp, M, FR)
    #     _, AC = ACenv(ACp, FL, M, FR)
    #     _, C = Cenv(Cp, FL, FR)
    #     AL, AR, errL, errR = ACCtoALAR(AC, C)
    #     @show errL, errR, ADVUMPS.error(ALp,Cp,ARp,FL,M,FR)
    # end
    function loss(x)
        AC = reshape(x[1:2*D^2], (D,2,D))
        C = reshape(x[2*D^2+1:2*D^2+D^2], (D,D))
        ACp = ein"((adf,abc),dgeb),ceh -> fgh"(FL,AC,M,FR)
        Cp = ein"(acd,ab),bce -> de"(FL,C,FR)
        ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
        _, FL = leftenv(AL, ALp, M, FL)
        _, FR = rightenv(AR, ARp, M, FR)
        _, AC = ACenv(ACp, FL, M, FR)
        _, C = Cenv(Cp, FL, FR)
        _, _, errL, errR = ACCtoALAR(AC, C)
        return errL+errR
    end
    x0 = [reshape(AC,(2*D^2)) ; reshape(C,(D^2))]
    @show loss(x0)
    g(x) = Zygote.gradient(loss, x)[1]
    res = optimize(loss, g, 
    x0, LBFGS(m = 20) ,inplace = false,
    Optim.Options(f_tol=1e-6, iterations=10,
    extended_trace=true),
    )
    @show res
end

