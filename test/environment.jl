using ADVUMPS
using ADVUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,bigleftenv,bigrightenv,norm_FL,norm_FR
using BenchmarkTools
using CUDA
using KrylovKit
using LinearAlgebra
using Random
using Test
using OMEinsum
using OMEinsum: optimize_greedy
using Zygote

@testset "qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
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
    @test Array(ein"ab,ab -> "(L,L))[] ≈ 1 
    @test λL * L ≈ ein"ab,abcd -> cd"(L,工)

    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]
    @test imag(λR) ≈ 0
    @test Array(ein"ab,ab -> "(R,R))[] ≈ 1 
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
    
    ALo, = leftorth(A)
    ALn, = leftorth(A)
    λL,FL = leftenv(ALo, ALn, M)
    @test λL * FL ≈ ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,ALo,M,ALn)
    _, ARo = rightorth(A)
    _, ARn = rightorth(A)
    λR,FR = rightenv(ARo, ARn, M)
    @test λR * FR ≈ ein"αpγ,γcη,ascp,βsη -> αaβ"(ARo,FR,M,ARn)
end

@testset "normalization leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    A = atype(rand(dtype,D,d,D))
    
    AL, = leftorth(A)
    λL,FL = norm_FL(AL, AL)
    @test λL * FL ≈ ein"(ad,acb), dce -> be"(FL,AL,AL)
    _, AR = rightorth(A)
    λR,FR = norm_FR(AR, AR)
    @test λR * FR ≈ ein"(be,acb), dce -> ad"(FR,AR,AR)
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

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))

    ALu, = leftorth(A)
    ALd, = leftorth(A)
    λL,FL4 = bigleftenv(ALu, ALd, M)
    @test λL * FL4 ≈ ein"(((dcba,def),ckge),bjhk),aji -> fghi"(FL4,ALu,M,M,ALd)

    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    λR,FR4 = bigrightenv(ARu, ARd, M)
    @test λR * FR4 ≈ ein"(((fghi,def),ckge),bjhk),aji -> dcba"(FR4,ARu,M,M,ARd)
end

