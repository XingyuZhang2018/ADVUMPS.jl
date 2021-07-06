using ADVUMPS
using ADVUMPS: num_grad
using ADVUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,ACCtoALAR,bigleftenv,bigrightenv
using ADVUMPS: energy,magofdβ
using ChainRulesCore
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using OMEinsum: get_size_dict, optimize_greedy, MinSpaceOut, MinSpaceDiff
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    a = atype(randn(2,2))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(atype(Float64[x 2x; 3x 4x]))
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "Zygote.@ignore" begin
    function foo2(x)
        return x^2
    end
    function foo3(x)
        return x^2 + Zygote.@ignore x^3
    end
    @test foo2(1) != foo3(1)
    @test Zygote.gradient(foo2,1)[1] ≈ Zygote.gradient(foo3,1)[1]
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo5(x)
        A = M .* x
        Q, R = qrpos(A)
        return norm(Q)
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1], num_grad(foo5, 1), atol = 1e-5)
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo6(x)
        A = M .*x
        L, Q = lqpos(A)
        return  norm(Q)
    end
    @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
end

@testset "linsolve with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    D,d = 2^2,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]

    dL = atype(rand(D,D))
    dL -= ein"ab,ab -> "(L,dL)[] * L
    @test ein"ab,ab ->  "(L,dL)[] ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> ein"abcd,cd -> ab"(工,R), dL, -λL, 1)
    @test ein"ab,ab -> "(ξL,L)[] ≈ 0 atol = 1e-9

    dR = atype(rand(D,D))
    dR -= ein"ab,ab -> "(R,dR)[] * R
    @test ein"ab,ab -> "(R,dR)[] ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> ein"ab,abcd -> cd"(L,工), dR, -λR, 1)
    @test ein"ab,ab -> "(ξR,R)[] ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    D = 10
    A = atype(rand(D,D,D))
    B = atype(rand(D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)
        F = ein"ab,ab -> "(D,D)
        return Array(E)[]/Array(F)[]
    end 
    Zygote.gradient(foo, 1)[1]
end

@testset "einsum optimize with $atype" for atype in [Array]
    D = 16
    χ = 30
    FL = atype(rand(χ,D,χ))
    AL = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))
    C = atype(rand(χ,χ))
    ap = atype(rand(D,D,D,D,2,2))
    AR = atype(rand(χ,D,χ))
    FR = atype(rand(χ,D,χ))
    FL4 = atype(rand(χ,D,D,χ))
    # 王 = atype(rand(χ,D,χ,χ,D,χ))

    function getoc(ein::EinCode{ixs, iy}, xs) where {ixs, iy}
        @time begin
            sd = get_size_dict(ixs, xs);
            oc = optimize_greedy(ein, sd; method = MinSpaceDiff())
        end
        display(oc)
        return oc
    end

    ein = ein"abc,cde,afi,bfgj,dghl,ijk,klm -> ehm"
    xs = (AL,AL,FL,M,M,AL,AL)
    # oc = getoc(ein, xs)
    # @btime $oc($AL,$AL,$FL,$M,$M,$AL,$AL)
    function foo(FL,AL,M)
        FL = ein"((afi,abc),bfgj),ijk -> cgk"(FL,AL,M,AL)
        ein"((afi,abc),bfgj),ijk -> cgk"(FL,AL,M,AL)
    end
    # @btime $foo($FL,$AL,$M)

    ein = ein"asd,cpd,cpb -> asb"
    xs = (AL,AL,AL)
    # oc = getoc(ein, xs)
    # @btime $oc($AL,$AL,$AL)
    # @btime ein"(asd,cpd),cpb -> asb"($AL,$AL,$AL)

    ein = ein"dcba,def,ckge,bjhk,aji -> fghi"
    xs = (FL4,AL,M,M,AL)
    # oc = getoc(ein, xs)

    ein = ein"αcβ,βsη,cpds,ηdγ,αpγ -> "
    xs = (FL,AL,M,FR,AL)
    # oc = getoc(ein, xs)
    
    ein = ein"αcβ,βη,ηcγ,αγ -> "
    xs = (AL,C,AL,C)
    # oc = getoc(ein, xs)

    ein = ein"γcη,ηpβ,γsα,βaα -> csap"
    xs = (AL,AL,AL,AL)
    # oc = getoc(ein, xs)
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10
    A = atype(rand(dtype,D,d,D))
    
    ALu, = leftorth(A)
    ALd, = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)

    S = atype(rand(D,d,D,D,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _,FL = leftenv(ALu, ALd, M)
        A = ein"(γcη,ηcγαaβ),βaα -> "(FL,S,FL)
        B = ein"γcη,ηcγ -> "(FL,FL)
        return Array(A)[]/Array(B)[]
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FR = rightenv(ARu, ARd, M)
        A = ein"(γcη,ηcγαaβ),βaα -> "(FR,S,FR)
        B = ein"γcη,ηcγ -> "(FR,FR)
        return Array(A)[]/Array(B)[]
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    A = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(),β))
    
    AL,C = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    _, AR = rightorth(A)
    λR,FR = rightenv(AR, AR, M)
    AC = ein"asc,cb -> asb"(AL,C)

    S = atype(rand(D,d,D,D,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _, AC = ACenv(AC, FL, M, FR)
        A = ein"γcη,ηcγαaβ,βaα -> "(AC,S,AC)
        B = ein"γcη,ηcγ -> "(AC,AC)
        return Array(A)[]/Array(B)[]
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    S = atype(rand(D,D,D,D))
    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FL = leftenv(AL, AL, M)
        _,FR = rightenv(AR, AR, M)
        _, C = Cenv(C, FL, FR)
        A = ein"γη,ηγαa,aα -> "(C,S,C)
        B = ein"ab,ab -> "(C,C)
        return Array(A)[]/Array(B)[]
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "ACCtoALAR with $atype{$dtype}" for atype in [Array], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 5, 2

    A = atype(rand(dtype, D, d, D))
    S1 = atype(rand(dtype, D, d, D, D, d, D))
    S2 = atype(rand(dtype, D, D, D, D))

    ALu,Cp = leftorth(A)
    ALd,Cp = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    M = atype(model_tensor(Ising(),1))
    _, FL = leftenv(ALu, ALd, M)
    _, FR = rightenv(ARu, ARd, M)

    ACp = ein"asc,cb -> asb"(ALu,Cp)

    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _, AC = ACenv(ACp, FL, M, FR)
        _, C = Cenv(Cp, FL, FR)
        AL, AR = ACCtoALAR(AC, C)
        s = 0
        A = ein"(γcη,ηcγαaβ),βaα -> "(AC, S1, AC)
        B = ein"γcη,γcη -> "(AC, AC)
        s += Array(A)[]/Array(B)[]
        A = ein"(γcη,ηcγαaβ),βaα -> "(AL, S1, AL)
        B = ein"γcη,γcη -> "(AL, AL)
        s += Array(A)[]/Array(B)[]
        A = ein"(γcη,ηcγαaβ),βaα -> "(AR, S1, AR)
        B = ein"γcη,γcη -> "(AR, AR)
        s += Array(A)[]/Array(B)[]
        A = ein"(γη,ηγαβ),βα -> "(C, S2, C)
        B = ein"γη,γη -> "(C, C)
        s += Array(A)[]/Array(B)[]
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-4)
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    A = atype(rand(dtype,D,d,D))

    ALu, = leftorth(A)
    ALd, = leftorth(A)
    _, ARu = rightorth(A)
    _, ARd = rightorth(A)
    S = atype(rand(D,d,d,D,D,d,d,D))
    function foo1(β)
        M = atype(model_tensor(Ising(),β))
        _,FL4 = bigleftenv(ALu, ALd, M)
        A = ein"abcd,abcdefgh,efgh -> "(FL4,S,FL4)
        B = ein"abcd,abcd -> "(FL4,FL4)
        return Array(A)[]/Array(B)[]
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    S = atype(rand(D,d,d,D,D,d,d,D))
    function foo2(β)
        M = atype(model_tensor(Ising(),β))
        _,FR4 = bigrightenv(ARu, ARd, M)
        A = ein"abcd,abcdefgh,efgh -> "(FR4,S,FR4)
        B = ein"abcd,abcd -> "(FR4,FR4)
        return Array(A)[]/Array(B)[]
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    χ = 10
    model = Ising()
    function foo1(β)
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = χ, tol = 1e-10, maxiter = 20, verbose = true, savefile = false)
        magnetisation(env,Ising(),β)
    end
    # for β = 0.5
    #     @show num_grad(foo1,β)
    #     @test Zygote.gradient(foo1, β)[1] ≈ magofdβ(model,β) atol = 1e-8
    # end

    function foo2(β) 
        magnetisation(vumps_env(Ising(),β,χ; tol = 1e-20, maxiter = 20, verbose = true), Ising(), β)
    end
    for β = 0.4
        @show num_grad(foo2,β)
        @test Zygote.gradient(foo2, β)[1] ≈ magofdβ(model,β) atol = 1e-8
    end
end