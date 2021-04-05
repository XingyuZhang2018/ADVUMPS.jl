using ADVUMPS
using ADVUMPS:magofβ,magofdβ,num_grad
using Plots
using Zygote
using JLD2
using FileIO

# magnetisation
begin
    magplot = plot()
    for D = [2,4,8,16,32]
        mag = []
        for β = 0.41:0.0025:0.48
            printstyled("D = $(D), β = $(β) \n"; bold=true, color=:red)
            
            env = vumps_env(Ising(),β,D)
            mag = [mag;magnetisation(env, Ising(), β)]

            chkp_file = "./data/Ising()_β$(round(β+0.0025,digits=4))_D$(D).jld2"
            if isfile(chkp_file) == false
                save(chkp_file, "env", env)
            end
        end
        β = 0.41:0.0025:0.48
        magplot = plot!(β,mag,seriestype = :scatter,title = "magnetisation", label = "VUMPS D = $(D)", lw = 3)
    end

    tmag = []
    for β = 0.41:0.001:0.48
        tmag = [tmag;magofβ(Ising(), β)]
    end
    β = 0.41:0.001:0.48
    magplot = plot!(β,tmag,label = "exact", lw = 2)
    xlabel!("β")
    ylabel!("mag")
    savefig(magplot,"./plot/2Disingmag.svg")
end

# dmag/dβ
begin
    dmagplot = plot()
    for D = [2,4,8]
        dmag = []
        for β = 0.1:0.05:1.0
            printstyled("D = $(D), β = $(β) \n"; bold=true, color=:red)
            function foo(β)
                env = vumps_env(Ising(),β,D)
                Zygote.@ignore begin
                    chkp_file = "./data/Ising()_β$(round(β+0.05,digits=2))_D$(D).jld2"
                    if isfile(chkp_file) == false
                        save(chkp_file, "env", env)
                    end
                end
                magnetisation(env,Ising(),β)
            end
            dmag = [dmag;Zygote.gradient(foo,β)[1]]
        end
        β = 0.1:0.05:1.0
        dmagplot = plot!(β,dmag,seriestype = :scatter,title = "dmag/dβ", label = "ADVUMPS D = $(D)", lw = 2)
    end
    tdmag = []
    for β = 0.1:0.01:1.0
        tdmag = [tdmag;magofdβ(Ising(), β)]
    end
    β = 0.1:0.01:1.0
    dmagplot = plot!(β,tdmag, label = "exact", lw = 2)
    xlabel!("β")
    ylabel!("dmag/dβ")
    savefig(dmagplot,"./plot/2Disingdmag.svg")
end

# energy
begin
    engplot = plot()
    for D = [2,4,8]
        eng = []
        for β = 0.1:0.05:1.0
            printstyled("D = $(D), β = $(β) \n"; bold=true, color=:red)
            function foo(β)
                env = vumps_env(Ising(),β,D)
                Zygote.@ignore begin
                    chkp_file = "./data/Ising()_β$(round(β+0.05,digits=2))_D$(D).jld2"
                    if isfile(chkp_file) == false
                        save(chkp_file, "env", env)
                    end
                end
                -log(Z(env))
            end
            eng = [eng;Zygote.gradient(foo,β)[1]]
        end
        β = 0.1:0.05:1.0
        engplot = plot!(β,eng,seriestype = :scatter,title = "-dlog(Z)/dβ", label = "ADVUMPS D = $(D)", lw = 2)
    end
    β = 0.1:0.01:1.0
    tengmag = []
    for β = 0.1:0.01:1.0
        env = vumps_env(Ising(),β,8)
        chkp_file = "./data/Ising()_β$(round(β+0.01,digits=2))_D$(8).jld2"
        if isfile(chkp_file) == false
            save(chkp_file, "env", env)
        end
        tengmag = [tengmag;energy(env,Ising(),β)]
    end
    engplot = plot!(β,tengmag, label = "VUMPS D = 8", lw = 2)
    xlabel!("β")
    ylabel!("energy")
    savefig(engplot,"./plot/2Disingene.svg")
end