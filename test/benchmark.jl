using ADVUMPS
using ADVUMPS:lefteig,righteig
using BenchmarkTools
using KrylovKit
using LinearAlgebra
using Zygote
# d = 2
# D = 10

# β = rand()
# A = rand(D,d,D)
# M = model_tensor(Ising(),β)

# AL, = leftorth(A)

# function foo2(β)
#     M = model_tensor(Ising(),β)
#     λL,FL = leftenv(AL, M)
#     return norm(FL) + real(λL)
# end 

# # @test isapprox(Zygote.gradient(foo2, 1)[1],num_grad(foo2, 1), atol = 1e-2)
foo = x -> magnetisation(Ising(), x, 10)
# @benchmark magnetisation(Ising(), 0.5,2)
@benchmark Zygote.gradient(foo,0.5)[1]