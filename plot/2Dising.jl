using ADVUMPS
using ADVUMPS:magofβ,magofdβ,num_grad
using Plots
using Zygote

# magnetisation
magplot = plot()
for χ = [2,4,8]
    T = 2.1:0.02:2.4
    mag = []
    for T = 2.1:0.02:2.4
        mag = [mag;magnetisation(Ising(), 1/T, χ)]
    end
    magplot = plot!(T,mag,seriestype = :scatter,title = "magnetisation", label = "VUMPS χ = $(χ)", lw = 3)
end

T = 2.1:0.001:2.4
tmag = []
for T = 2.1:0.001:2.4
    tmag = [tmag;magofβ(Ising(), 1/T)]
end
magplot = plot!(T,tmag,label = "exact", lw = 2)
xlabel!("T")
ylabel!("mag")
savefig(magplot,"./plot/2Disingmag.svg")

# dmag/dβ
dmagplot = plot()
for χ = [2,4,8]
    β = 0.1:0.05:1.0
    dmag = []
    foo = x -> magnetisation(Ising(), x, χ)
    for β = 0.1:0.05:1.0
        dmag = [dmag;Zygote.gradient(foo,β)[1]]
    end
    dmagplot = plot!(β,dmag,seriestype = :scatter,title = "dmag/dβ", label = "ADVUMPS χ = $(χ)", lw = 2)
end
β = 0.1:0.01:1.0
tdmag = []
for β = 0.1:0.01:1.0
    tdmag = [tdmag;magofdβ(Ising(), β)]
end
dmagplot = plot!(β,tdmag, label = "exact", lw = 2)
xlabel!("β")
ylabel!("dmag/dβ")
savefig(dmagplot,"./plot/2Disingdmag.svg")

# energy
engplot = plot()
for χ = [2,4,8]
    β = 0.1:0.05:1.0
    eng = []
    foo = x -> -log(Z(Ising(), x, 3))
    for β = 0.1:0.05:1.0
        eng = [eng;Zygote.gradient(foo,β)[1]]
    end
    engplot = plot!(β,eng,seriestype = :scatter,title = "-dlog(Z)/dβ", label = "ADVUMPS χ = $(χ)", lw = 2)
end
β = 0.1:0.01:1.0
tengmag = []
for β = 0.1:0.01:1.0
    tengmag = [tengmag;energy(Ising(), β, 8)]
end
engplot = plot!(β,tengmag, label = "VUMPS χ = 8", lw = 2)
xlabel!("β")
ylabel!("energy")
savefig(engplot,"./plot/2Disingene.svg")