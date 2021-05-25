using ADVUMPS
using BenchmarkTools
using CUDA
using KrylovKit
using LinearAlgebra
using Random
using Test
using Plots
using Zygote

CUDA.allowscalar(false)

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