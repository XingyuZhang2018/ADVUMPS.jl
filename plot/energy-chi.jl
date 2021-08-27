using ADVUMPS
using ADVUMPS: energy, optcont,IPEPS
using CUDA
using Plots

function energy_χ(ipeps::IPEPS{LT}, key, χ) where LT
    model, atype, D, _, tol, maxiter = key
    key = (model, atype, D, χ, tol, maxiter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    energy(h, IPEPS{LT}(atype(ipeps.bulk)), oc, key; verbose = true)
end

model = Heisenberg(1.0,1.0,1.0)
ipeps, key = init_ipeps(model;atype = Array, D=2, χ=20, tol=1e-10, maxiter=10)
x = 20
yenergy = []
for χ in x
    yenergy = [yenergy; energy_χ(ipeps, key, χ)]
end
# energyplot = plot()
# plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)
@show yenergy