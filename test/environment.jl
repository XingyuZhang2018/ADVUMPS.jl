using ADVUMPS
using ADVUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,bigleftenv,bigrightenv
using LinearAlgebra
using Random
using Test
using OMEinsum
using Zygote
using BenchmarkTools
using CUDA
using KrylovKit

@testset "qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = @time qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "eigsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    D,d = 10,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    @test imag(λL) ≈ 0
    @test ein"ab,ab -> "(L,L)[] ≈ 1 
    @test λL * L ≈ ein"ab,abcd -> cd"(L,工)

    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]
    @test imag(λR) ≈ 0
    @test ein"ab,ab -> "(R,R)[] ≈ 1 
    @test λR * R ≈ ein"abcd,cd -> ab"(工,R)
    @test λL ≈ λR
end

@testset "leftorth with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10
    A = atype(rand(dtype,D,d,D))
    AL, C, λ = leftorth(A)

    M = ein"cda,cdb -> ab"(AL,conj(AL))
    @test (Array(M) ≈ I(D))

    CA = reshape(C * reshape(A, D, d*D), d*D, D)
    ALC = reshape(AL, d*D, D) * C * λ
    @test (Array(ALC) ≈ Array(CA))
end

@testset "rightorth with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10
    A = atype(rand(dtype,D,d,D))
    C, AR, λ = rightorth(A)

    M = ein"acd,bcd -> ab"(AR,conj(AR))
    @test (Array(M) ≈ I(D))

    AC = reshape(reshape(A, d*D, D)*C, D, d*D)
    CAR = C * reshape(AR, D, d*D) * λ
    @test (Array(CAR) ≈ Array(AC))
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))
    
    AL, = leftorth(A)
    λL,FL = leftenv(AL, M)
    @test λL * FL ≈ ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))

    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)
    @test λR * FR ≈ ein"αpγ,γcη,ascp,βsη -> αaβ"(AR,FR,M,conj(AR))
    # S = rand(D,d,D,D,d,D)
    # function foo2(β)
    #     M = model_tensor(Ising(),β)
    #     λL,FL = leftenv(AL, M)
    #     return ein"γcη,ηcγαaβ,βaα -> "(FL,S,FL)[]/ein"γcη,ηcγ -> "(FL,FL)[]
    # end 
    # @test isapprox(Zygote.gradient(foo2, 1)[1],num_grad(foo2, 1), atol = 1e-9)

    
    # S = rand(D,d,D,D,d,D)
    # function foo3(β)
    #     M = model_tensor(Ising(),β)
    #     λL,FR = rightenv(AR, M)
    #     return ein"γcη,ηcγαaβ,βaα -> "(FR,S,FR)[]/ein"γcη,ηcγ -> "(FR,FR)[]
    # end
    # @test isapprox(Zygote.gradient(foo3, 1)[1],num_grad(foo3, 1), atol = 1e-9)
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))

    AL,C = leftorth(A)
    λL,FL = leftenv(AL, M)

    _, AR = rightorth(A)
    λR,FR = rightenv(AR, M)

    AC = ein"asc,cb -> asb"(AL,C)
    λAC, AC = ACenv(AC, FL, M, FR)
    @test λAC * AC ≈ ein"αaγ,γpη,asbp,ηbβ -> αsβ"(FL,AC,M,FR)

    λC, C = Cenv(C, FL, FR)
    @test λC * C ≈ ein"αaγ,γη,ηaβ -> αβ"(FL,C,FR)
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))

    AL, = leftorth(A)
    λL,FL4 = bigleftenv(AL, M)
    @test λL * FL4 ≈ ein"dcba,def,ckge,bjhk,aji -> fghi"(FL4,AL,M,M,conj(AL))

    C, AR = rightorth(A)
    λR,FR4 = bigrightenv(AR, M)
    @test λR * FR4 ≈ ein"fghi,def,ckge,bjhk,aji -> dcba"(FR4,AR,M,M,conj(AR))

    # S = rand(D,d,d,D,D,d,d,D)
    # function foo(β)
    #     M = model_tensor(Ising(),β)
    #     λL,FL4 = bigleftenv(AL, M)
    #     return ein"abcd,abcdefgh,efgh -> "(FL4,S,FL4)[]/ein"abcd,abcd -> "(FL4,FL4)[]
    # end 
    # @test isapprox(Zygote.gradient(foo, 1)[1],num_grad(foo, 1), atol = 1e-9)

    # S = rand(D,d,d,D,D,d,d,D)
    # function foo(β)
    #     M = model_tensor(Ising(),β)
    #     λR,FR4 = bigrightenv(AR, M)
    #     return ein"abcd,abcdefgh,efgh -> "(FR4,S,FR4)[]/ein"abcd,abcd -> "(FR4,FR4)[]
    # end
    # @test isapprox(Zygote.gradient(foo, 1)[1],num_grad(foo, 1), atol = 1e-9)
end

