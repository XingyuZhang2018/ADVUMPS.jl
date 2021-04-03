using ADVUMPS
using BenchmarkTools
using KrylovKit
using LinearAlgebra
using Zygote

foo = x -> magnetisation(Ising(), x, 20)
@benchmark foo(0.5)
@benchmark Zygote.gradient(foo,0.5)[1]