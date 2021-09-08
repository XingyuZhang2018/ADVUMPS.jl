using ADVUMPS
using ADVUMPS: energy, num_grad, diaglocal, optcont
using CUDA
using LinearAlgebra: svd, norm
using LineSearches, Optim
using OMEinsum
using Random
using Zygote

CUDA.allowscalar(false)
Random.seed!(100)
model = Heisenberg(1.0,1.0,1.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADVUMPS.jl/"
ipeps, key = init_ipeps(model;atype = Array, folder = folder, D=2, χ=20, tol=1e-10, maxiter=10)
res = optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 0, verbose = true)