using ADVUMPS
using BenchmarkTools
using CUDA
using KrylovKit
using LinearAlgebra
using Random
using Test
using OMEinsum
using Plots
using Zygote
CUDA.allowscalar(false)

@testset "OMEinsum" begin
    Random.seed!(100)
    function testime(d,D,atype)
        AL = atype(rand(D, d, D))
        M = atype(rand(d, d, d, d))
        FL = atype(rand(D, d, D))
        @elapsed ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))
        # @belapsed ein"γcη,ηpβ,csap,γsα -> αaβ"($(FL),$(AL),$(M),$(conj(AL)))
    end

    timeplot = plot()
    for (atype,plotstyle) in [(Array,:dashdotdot), (CuArray,:solid)], d in [2,3,4,5,6].^2
        t = []
        for D = 10:5:50
            @show atype,d,D
            t = [t;testime(d,D,atype)]
        end
        D = 10:5:50
        plot!(timeplot,D,t,linestyle=plotstyle,title = "time-D", label = "$(atype) d=$(d) ", lw = 3, legend = :topleft)
    end
    xlabel!("D")
    ylabel!("time/s")
    savefig(timeplot,"./plot/OMEinsum_timeplot.svg")
end

@testset "OMEinsum acceleration ratio" begin
    Random.seed!(100)
    function testime(d,D,atype)
        AL = atype(rand(D, d, D))
        M = atype(rand(d, d, d, d))
        FL = atype(rand(D, d, D))
        # @elapsed ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL))
        @belapsed ein"γcη,ηpβ,csap,γsα -> αaβ"($(FL),$(AL),$(M),$(conj(AL)))
    end

    timeplot = plot()
    for d in [2,3].^2
        t = []
        for D = 10:5:50
            @show d,D
            t = [t;testime(d,D,Array)/testime(d,D,CuArray)]
        end
        D = 10:5:50
        plot!(timeplot,D,t,linestyle=:dashdotdot,title = "acceleration ratio-D", label = "d=$(d) ", lw = 3, legend = :topleft)
    end
    xlabel!("D")
    ylabel!("acceleration ratio/s")
    savefig(timeplot,"./plot/OMEinsum_CPUGPUd=4to9_timeplot.svg")
end

@testset "KrylovKit" begin
    Random.seed!(100)
    function testime(d,D,atype)
        AL = atype(rand(D, d, D))
        M = atype(rand(d, d, d, d))
        FL = atype(rand(D, d, D))
        t1 = @elapsed λs, FLs, _ = eigsolve(FL -> ein"γcη,ηpβ,csap,γsα -> αaβ"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)

        λl,FL = real(λs[1]),real(FLs[1])
        dFL = atype(rand(D, d, D))
        t2 = @elapsed linsolve(FR -> ein"ηpβ,βaα,csap,γsα -> ηcγ"(AL, FR, M, conj(AL)), permutedims(dFL, (3, 2, 1)), -λl, 1)
        return t1 + t2
    end

    timeplot = plot()
    for (atype,plotstyle) in [(Array,:dashdotdot), (CuArray,:solid)], d in [4].^2
        t = []
        for D = 15:5:50
            @show atype,d,D
            t = [t;testime(d,D,atype)]
        end
        D = 15:5:50
        plot!(timeplot,D,t,linestyle=plotstyle,title = "time-D", label = "$(atype) d=$(d) ", lw = 3, legend = :topleft)
    end
    xlabel!("D")
    ylabel!("time/s")
    savefig(timeplot,"./plot/KrylovKit_timeplot.svg")
end

@testset "heisenberg " begin
    Random.seed!(100)
    model = Heisenberg(1.0,1.0,1.0)
    function testime(D,χ,atype)
        ipeps, key = init_ipeps(model; D=D, χ=χ, tol=0, maxiter=1)
        # @belapsed optimiseipeps($(ipeps), $(key); f_tol = 1e-6, opiter = 0, atype = $(atype))
        @elapsed optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 0, atype = atype, verbose = true)
    end

    # @show testime(2,10,CuArray)
    timeplot = plot()
    D = 4
    for atype in [Array, CuArray]   
        t = []
        for χ = 10:5:20
            @show atype,χ
            t = [t;testime(D,χ,atype)]
        end

        χ = 10:5:20
        plot!(timeplot,χ,t,seriestype = :path,title = "time-χ", label = "$(atype)", lw = 3)
    end
    xlabel!("χ")
    ylabel!("time/s")
    savefig(timeplot,"./plot/timeplot_D$(D).svg")
end

@testset "tr" begin
    Random.seed!(100)
    a = rand(ComplexF64, 100,100)
    b = rand(ComplexF64, 100,100)
    @btime tr(($(a))'*$(b))
    @btime ein"ab, ab ->"(conj($(a)), $(b))
end
