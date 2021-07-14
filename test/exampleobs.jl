using ADVUMPS
using ADVUMPS: magnetisation,Z,energy,Zofβ,magofβ,eneofβ,magofdβ,num_grad,obs_env,norm_FL,norm_FR
using CUDA
using OMEinsum
using Plots
using Random
using Test

@testset "up and dowm vumps with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(1001)
    model = Ising()
    for β = 0.8
        @show β
        M = atype(model_tensor(model, β))
        env = obs_env(model, M; atype = atype, D = 2, χ = 10, tol = 1e-20, maxiter = 17, verbose = true, savefile = false)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-5)
        # @test isapprox(magnetisation(env,Ising(),β), magofβ(Ising(),β), atol=1e-5)
        # @test isapprox(energy(env,Ising(),β), eneofβ(Ising(),β), atol=1e-3)
    end
end

@testset "Z&O-iterate with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    model = Ising()
    Zplot = plot()
    magnetisationplot = plot()
    energyplot = plot()
    Zerrplot = plot()
    magnetisationerrplot = plot()
    energyerrplot = plot()
    overlapplot = plot()
    for β = 0.4
        @show β
        M = atype(model_tensor(model, β))
        x = []
        yZ = []
        ymag = []
        yene = []
        yZerr = []
        ymagerr = []
        yenerr = []
        overlaps = []
        iter = 1:30
        for iterate = iter
            Random.seed!(1001)
            x = [x; iterate]
            env = obs_env(model, M; atype = atype, D = 2, χ = 10, tol = 1e-20, maxiter = iterate, verbose = true, savefile = false)
            Mu, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR = env
            _, FL_n = norm_FL(ALu, ALd)
            _, FR_n = norm_FR(ARu, ARd)
            overlaps = [overlaps; ein"((ae,adb),bc),((edf,fg),cg) ->"(FL_n,ALu,Cu,ALd,Cd,FR_n)[]/ein"ac,ab,bd,cd ->"(FL_n,Cu,FR_n,Cd)[]]
            yZ = [yZ; Z(env)]
            ymag = [ymag; magnetisation(env,Ising(),β)]
            yene = [yene; energy(env,Ising(),β)]
            yZerr = [yZerr; Z(env)-Zofβ(Ising(),β)]
            ymagerr = [ymagerr; magnetisation(env,Ising(),β)-magofβ(Ising(),β)]
            yenerr = [yenerr; energy(env,Ising(),β)-eneofβ(Ising(),β)]
        end
        plot!(Zplot, x, yZ, seriestype = :scatter, title = "Z", label = "Z", lw = 3)
        plot!(Zplot, x, [Zofβ(Ising(),β) for _ =iter], title = "Z", label = "β = $(β) exact", lw = 3)
        plot!(magnetisationplot, x, ymag, seriestype = :scatter, title = "magnetisation", label = "magnetisation", lw = 3)
        plot!(magnetisationplot, x, [magofβ(Ising(),β) for _ =iter], title = "magnetisation", label = "β = $(β) exact", lw = 3)
        plot!(energyplot, x, yene, seriestype = :scatter, title = "energy", label = "energy", lw = 3)
        plot!(energyplot, x, [eneofβ(Ising(),β) for _ =iter], title = "energy", label = "β = $(β) exact", lw = 3)
        plot!(Zerrplot, x, yZerr, seriestype = :scatter, title = "Z", label = "Z", ylabel = "error", lw = 3)
        plot!(magnetisationerrplot, x, ymagerr, seriestype = :scatter, title = "magnetisation", label = "magnetisation", ylabel = "error", lw = 3)
        plot!(energyerrplot, x, yenerr, seriestype = :scatter, title = "energy", label = "energy", ylabel = "error", lw = 3)
        plot!(overlapplot, x, overlaps, seriestype = :scatter, title = "overlap", label = "overlap", ylabel = "overlap", lw = 3)
        obs = plot(Zplot, Zerrplot, magnetisationplot, magnetisationerrplot, energyplot, energyerrplot, layout = (3,2), xlabel="iterate", size = [1000, 1000])
        lo = @layout [a; b{0.2h}]
        p = plot(obs, overlapplot, layout = lo, xlabel="iterate", size = [1000, 1000])
        savefig(p,"./plot/β$(β)_Ising_Z&O-iterate.svg")
    end
end