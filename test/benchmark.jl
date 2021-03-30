using ADTensor
using ADTensor:eig
using IterativeSolvers
using BenchmarkTools
using KrylovKit
using LinearAlgebra

A = [1 2 3;3 2 1;2 2 1]
λ,l,r = eig(A,ones(3,3))
dl,dr = rand(3,1),rand(3,1)

function foo1()
    ξl = (A .- λ) \ ((1 .- r*l')*dl)
    ξr = (A' .- λ) \ ((1 .- l*r')*dr)
end

function foo2()
    ξl,ξr = rand(3),rand(3)
    gmres!(ξl, A .- λ, (1 .- r*l')*dl)
    gmres!(ξr, A' .- λ, (1 .- l*r')*dr)
end

@benchmark foo1()
@benchmark foo2()

