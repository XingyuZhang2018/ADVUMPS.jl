using ADVUMPS
using ADVUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,magnetisation,Z,energy,magofβ,magofdβ,num_grad
using LinearAlgebra
using Random
using Test
using OMEinsum
using Zygote

@testset "qr" begin
    A = rand(ComplexF64,4,4)
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq" begin
    A = rand(ComplexF64,4,4)
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth" begin
    d = 2
    D = 10
    A = rand(Float64,D,d,D)
    AL, C, λ = leftorth(A)

    M = ein"cda,cdb -> ab"(AL,conj(AL))
    # @tensor M[-1, -2] := AL[1,2,-1]*conj(AL[1,2,-2])
    @test (Array(M) ≈ I(D))

    CA = reshape(C * reshape(A, D, d*D), d*D, D)
    ALC = reshape(AL, d*D, D) * C * λ
    @test (Array(ALC) ≈ Array(CA))
end

@testset "rightorth" begin 
    d = 2
    D = 10
    A = rand(Float64,D,d,D)
    C, AR, λ = rightorth(A)

    M = ein"acd,bcd -> ab"(AR,conj(AR))
    # @tensor M[-1, -2] := AR[-1, 1, 2]*conj(AR[-2, 1, 2])
    @test (Array(M) ≈ I(D))

    AC = reshape(reshape(A, d*D, D)*C, D, d*D)
    CAR = C * reshape(AR, D, d*D) * λ
    @test (Array(CAR) ≈ Array(AC))
end

@testset "leftenv" begin
    Random.seed!(100)
    d = 2
    D = 2

    β = rand()
    # A = rand(ComplexF64,D,d,D)
    A = rand(D,d,D)
    M = model_tensor(Ising(),β)
    
    AL, = leftorth(A)
    λL,FL = leftenv(AL, M)

    @test λL * FL ≈ ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))

    S = rand(D,d,D,D,d,D)
    function foo2(β)
        M = model_tensor(Ising(),β)
        λL,FL = leftenv(AL, M)
        return ein"γcη,ηcγαaβ,βaα -> "(FL,S,FL)[]/ein"γcη,ηcγ -> "(FL,FL)[]
    end 
    @test isapprox(Zygote.gradient(foo2, 1)[1],num_grad(foo2, 1), atol = 1e-10)
end

@testset "rightenv" begin
    Random.seed!(100)
    d = 2
    D = 2

    β = rand()
    A = rand(ComplexF64,D,d,D)
    M = model_tensor(Ising(),β)

    C, AR = rightorth(A)
    λR,FR = rightenv(AR, M)

    @test λR * FR ≈ ein"αpγ,γcη,ascp,βsη -> αaβ"(AR,FR,M,conj(AR))

    S = rand(D,d,D,D,d,D)
    function foo3(β)
        M = model_tensor(Ising(),β)
        λL,FR = rightenv(AR, M)
        return ein"γcη,ηcγαaβ,βaα -> "(FR,S,FR)[]/ein"γcη,ηcγ -> "(FR,FR)[]
    end
    @test isapprox(Zygote.gradient(foo3, 1)[1],num_grad(foo3, 1), atol = 1e-10)
end

@testset "vumps unit test" begin
    Random.seed!(100)

    d = 2
    D = 2

    β = rand()
    A = rand(ComplexF64,D,d,D)
    M = model_tensor(Ising(),β)

    AL,C = leftorth(A)
    λL,FL = leftenv(AL, M)
    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)

    λ, AL, C, AR, errL, errR = ACCtoALAR(AL, C, AR, M, FL, FR)
    @test isapprox(errL,0,atol = 1e-1)
    @test isapprox(errR,0,atol = 1e-1)
    @test isapprox(ein"asc,cb -> asb"(AL,C),ein"ac,csb -> asb"(C,AR),atol = 1e-1)
    
    S = rand(D,d,D,D,d,D)
    function foo4(β)
        M = model_tensor(Ising(),β)
        AC = ein"asc,cb -> asb"(AL,C)
        μ1, AC = ACenv(AC, FL, M, FR)
        return ein"γcη,ηcγαaβ,βaα -> "(AC,S,AC)[]/ein"γcη,ηcγ -> "(AC,AC)[]
    end
    @test isapprox(Zygote.gradient(foo4, 0.1)[1],num_grad(foo4, 0.1), atol = 1e-9)

    S = rand(D,D,D,D)
    function foo5(β)
        M = model_tensor(Ising(),β)    
        λL,FL = leftenv(AL, M)
        λR,FR = rightenv(AR, M)
        μ1, C = Cenv(C, FL, FR)
        return ein"γη,ηγαβ,βα -> "(C,S,C)[]/ein"γη,ηγ -> "(C,C)[]
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1],num_grad(foo5, 1), atol = 1e-9)
end

@testset "vumps" begin
    Random.seed!(100)
    @test isapprox(magnetisation(Ising(), 0,2), magofβ(Ising(),0), atol=1e-6)
    @test isapprox(magnetisation(Ising(), 0.2,2), magofβ(Ising(),0.2), atol=1e-6)
    @test isapprox(magnetisation(Ising(), 0.4,2), magofβ(Ising(),0.4), atol=1e-6)
    @test isapprox(magnetisation(Ising(), 0.6,3), magofβ(Ising(),0.6), atol=1e-6)
    @test isapprox(magnetisation(Ising(), 0.8,2), magofβ(Ising(),0.8), atol=1e-6)

    foo1 = x -> -log(Z(Ising(), x, 3))
    @test isapprox(Zygote.gradient(foo1,0.5)[1], energy(Ising(), 0.5, 3), atol = 1e-6)
    @test isapprox(Zygote.gradient(foo1,0.5)[1], num_grad(foo1,0.5), atol = 1e-6)

    foo2 = x -> magnetisation(Ising(), x, 3)
    @test isapprox(num_grad(foo2,0.5), magofdβ(Ising(),0.5), atol = 1e-3)
    @test isapprox(Zygote.gradient(foo2,0.5)[1], num_grad(foo2,0.5), atol = 1e-6)
end