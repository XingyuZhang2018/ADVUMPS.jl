using ADVUMPS
using BenchmarkTools
using KrylovKit
using LinearAlgebra
using Zygote
using Random
using CUDA
Random.seed!(100)
# foo = x -> magnetisation(Ising(), x, 20)
# @benchmark foo(0.5)
# @benchmark Zygote.gradient(foo,0.5)[1]
A = CuArray(rand(1000,1))
B = rand()
function foo()
    A .* B
end
@benchmark foo()