using ADVUMPS
using ADVUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,magnetisation,Z,energy,magofβ,magofdβ,num_grad,bigleftenv,bigrightenv
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
    @test isapprox(Zygote.gradient(foo2, 1)[1],num_grad(foo2, 1), atol = 1e-9)
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
    @test isapprox(Zygote.gradient(foo3, 1)[1],num_grad(foo3, 1), atol = 1e-9)
end

@testset "bigleftenv" begin
    Random.seed!(100)
    d = 2
    D = 2

    β = rand()
    # A = rand(ComplexF64,D,d,D)
    A = rand(D,d,D)
    M = model_tensor(Ising(),β)
    
    AL, = leftorth(A)
    λL,FL4 = bigleftenv(AL, M)

    @test λL * FL4 ≈ ein"dcba,def,ckge,bjhk,aji -> fghi"(FL4,AL,M,M,conj(AL))

    S = rand(D,d,d,D,D,d,d,D)
    function foo(β)
        M = model_tensor(Ising(),β)
        λL,FL4 = bigleftenv(AL, M)
        return ein"abcd,abcdefgh,efgh -> "(FL4,S,FL4)[]/ein"abcd,abcd -> "(FL4,FL4)[]
    end 
    @test isapprox(Zygote.gradient(foo, 1)[1],num_grad(foo, 1), atol = 1e-9)
end

@testset "bigrightenv" begin
    Random.seed!(100)
    d = 2
    D = 2

    β = rand()
    A = rand(ComplexF64,D,d,D)
    M = model_tensor(Ising(),β)

    C, AR = rightorth(A)
    λR,FR4 = bigrightenv(AR, M)

    @test λR * FR4 ≈ ein"fghi,def,ckge,bjhk,aji -> dcba"(FR4,AR,M,M,conj(AR))

    S = rand(D,d,d,D,D,d,d,D)
    function foo(β)
        M = model_tensor(Ising(),β)
        λR,FR4 = bigrightenv(AR, M)
        return ein"abcd,abcdefgh,efgh -> "(FR4,S,FR4)[]/ein"abcd,abcd -> "(FR4,FR4)[]
    end
    @test isapprox(Zygote.gradient(foo, 1)[1],num_grad(foo, 1), atol = 1e-9)
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
    for β = 0:0.2:0.8
        @test isapprox(magnetisation(vumps_env(Ising(),β,2),Ising(),β), magofβ(Ising(),β), atol=1e-5)
    end

    β,D = 0.5,10
    foo1 = β -> -log(Z(vumps_env(Ising(),β,D)))
    @test isapprox(Zygote.gradient(foo1,β)[1], energy(vumps_env(Ising(),β,D),Ising(), β), atol = 1e-6)
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-9)

    foo2 = β -> magnetisation(vumps_env(Ising(),β,D), Ising(), β)
    @test isapprox(num_grad(foo2,β), magofdβ(Ising(),β), atol = 1e-3)
    @test isapprox(Zygote.gradient(foo2,β)[1], magofdβ(Ising(),β), atol = 1e-6)
end