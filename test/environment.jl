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
CUDA.allowscalar(false)

@testset "qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "eigsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D,d = 3,2                                        # a───┬───c 
    ┬ = atype(rand(dtype, D,d,D))                    # │   b   │ 
    工 = ein"abc,dbe -> acde"(┬,conj(┬))             # d───┴───e 
    λcs, cs, info = eigsolve(c -> ein"ad,acde -> ce"(c,工), atype(rand(dtype, D,D)), 1, :LM)
    λc, c = λcs[1], cs[1]
    @test imag(λc) ≈ 0 atol = 1e-12
    @test Array(ein"ab,ab -> "(c,conj(c)))[] ≈ 1 
    @test λc * c ≈ ein"ad,acde -> ce"(c,工)

    λↄs, ↄs, info = eigsolve(ↄ -> ein"acde,ce -> ad"(工,ↄ), atype(rand(dtype, D,D)), 1, :LM)
    λↄ, ↄ = λↄs[1], ↄs[1]
    @test imag(λↄ) ≈ 0 atol = 1e-12
    @test Array(ein"ab,ab -> "(ↄ,conj(ↄ)))[] ≈ 1 
    @test λↄ * ↄ ≈ ein"acde,ce -> ad"(工,ↄ)
    @test λc ≈ λↄ
end

@testset "leftorth and rightorth with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3
    A = atype(rand(dtype,D,d,D))            # a───┬───c 
    AL, L, λL = leftorth(A)                 # │   b   │ 
    R, AR, λR = rightorth(A)                # d───┴───e 

    @test Array(ein"abc,abe -> ce"(AL,conj(AL))) ≈ Array(I(D))
    @test Array(ein"abc,dbc -> ad"(AR,conj(AR))) ≈ Array(I(D))

    LA = reshape(L * reshape(A, D, d*D), d*D, D)
    ALL = reshape(AL, d*D, D) * L * λL
    @test ALL ≈ LA

    A_R = reshape(reshape(A, d*D, D)*R, D, d*D)
    RAR = R * reshape(AR, D, d*D) * λR
    @test RAR ≈ A_R
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3

    A = atype(rand(dtype,D,d,D))                     #  a ────┬──── c
    M = atype(rand(dtype,d,d,d,d))                   #  │     b     │
                                                     #  ├─ d ─┼─ e ─┤
    ALu, = leftorth(A)                               #  │     g     │
    ALd, = leftorth(A)                               #  f ────┴──── h
    λL,FL = leftenv(ALu, ALd, M)
    @test λL * FL ≈ ein"((adf,abc),dgeb),fgh -> ceh"(FL,ALu,M,conj(ALd))
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    λR,FR = rightenv(ARu, ARd, M)
    @test λR * FR ≈ ein"((ceh,abc),dgeb),fgh -> adf"(FR,ARu,M,conj(ARd))
end

@testset "normalization leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3
                                       # a───┬───c 
    A = atype(rand(dtype,D,d,D))       # │   b   │       
                                       # d───┴───e 
    ALu, = leftorth(A)
    ALd, = leftorth(A)
    λL,FL = norm_FL(ALu, ALd)
    @test λL * FL ≈ ein"(ad,abc), dbe -> ce"(FL,ALu,conj(ALd))

    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    λR,FR = norm_FR(ARu, ARd)
    @test λR * FR ≈ ein"(ce,abc), dbe -> ad"(FR,ARu,conj(ARd))
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3                                      #    a ────┬──── c
                                               #    │     b     │                                      
    A = atype(rand(dtype,D,d,D))               #    ├─ d ─┼─ e ─┤
    M = atype(rand(dtype,d,d,d,d))             #    │     f     │
                                               #    ├─ g ─┼─ h ─┤
    ALu, = leftorth(A)                         #    │     j     │
    ALd, = leftorth(A)                         #    i ────┴──── k 
    λL,FL4 = bigleftenv(ALu, ALd, M)
    @test λL * FL4 ≈ ein"(((adgi,abc),dfeb),gjhf),ijk -> cehk"(FL4,ALu,M,M,conj(ALd))

    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    λR,FR4 = bigrightenv(ARu, ARd, M)
    @test λR * FR4 ≈ ein"(((cehk,abc),dfeb),gjhf),ijk -> adgi"(FR4,ARu,M,M,conj(ARd))
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3

    A = atype(rand(dtype,D,d,D))                     #  a ────┬──── c
    M = atype(rand(dtype,d,d,d,d))                   #  │     b     │
                                                     #  ├─ d ─┼─ e ─┤
    ALu,Cu = leftorth(A)                             #  │     g     │
    ALd, = leftorth(A)                               #  f ────┴──── h
    _, FL = leftenv(ALu, ALd, M)

    _, ARu = rightorth(A)                            #  a ─── b
    _, ARd = rightorth(A)                            #  │     │
    _, FR = rightenv(ARu, ARd, M)                    #  ├─ c ─┤
                                                     #  │     │
    ACu = ein"abc,cd -> abd"(ALu,Cu)                 #  d ─── e
    λACu, ACu = ACenv(ACu, FL, M, FR)
    @test λACu * ACu ≈ ein"((adf,abc),dgeb),ceh -> fgh"(FL,ACu,M,FR)

    λCu, Cu = Cenv(Cu, FL, FR)
    @test λCu * Cu ≈ ein"(acd,ab),bce -> de"(FL,Cu,FR)
end



