# ADVUMPS.jl

[![Build Status](https://travis-ci.com/XingyuZhang2018/ADVUMPS.jl.svg?branch=dev)](https://travis-ci.com/XingyuZhang2018/ADVUMPS.jl)
[![Coverage](https://codecov.io/gh/XingyuZhang2018/ADVUMPS.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/XingyuZhang2018/ADVUMPS.jl)

This is a julia package to realise Automatic Differential(AD) for Variational Uniform Matrix product states(VUMPS). 

In this package we implemented the algorithms described in [Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650), but in another contraction method namely VUMPS.
demonstrating two applications:
- Gradient based optimization of iPEPS
- Direct calculation of energy densities in iPEPS via derivatives of the _free energy_

The key point to implement AD for VUMPS is to get adjoint of eigsolve, which have been solved in [Automatic differentiation of dominant eigensolver and its applications in quantum physics](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.245139).

This package used [TensorNetworkAD.jl](https://github.com/under-Peter/TensorNetworkAD.jl) for reference.
## install
```shell
> git clone https://github.com/XingyuZhang2018/ADVUMPS.jl
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.7) pkg> activate .
Activating environment at `..\ADVUMPS\Project.toml`

(ADVUMPS) pkg> instantiate
```
To get back to the Julia REPL, press `backspace` or `ctrl+C`. Then Precompile `ADVUMPS`
```julia
julia> using ADVUMPS
[ Info: Precompiling ADVUMPS [1846a12a-bf3c-41d7-8528-153c1dad55cd]
```
## Example
If you want to learn deeply into this package, I highly recommend to run each single test in `/test/runtests` in sequence.
### Free Energy of the 2D Classical Ising Model

We start by constructing the tensor for the tensor network representation of the 2d classical Ising Model.
This tensor can be constructed using the `model_tensor`-function that takes a `model`-parameter - in our case `Ising()` - and an inverse temperature `β` (e.g. at `β=0.5`).

```julia
julia> a = model_tensor(Ising(), 0.5)
2×2×2×2 Array{Float64,4}:
[:, :, 1, 1] =
 2.53434  0.5    
 0.5      0.18394

[:, :, 2, 1] =
 0.5      0.18394
 0.18394  0.5    

[:, :, 1, 2] =
 0.5      0.18394
 0.18394  0.5    

[:, :, 2, 2] =
 0.18394  0.5    
 0.5      2.53434
```
Then get VUMPS environment using the `vumps_env`-function that takes a `model`-parameter (in our case `Ising()`), an inverse temperature `β` (e.g. at `β=0.5`) and environment tenosr non-physical index dimension `D` (e.g. `D=2`)
```julia
julia> env = vumps_env(Ising(),0.5,2; verbose=true);
random initial vumps environment-> vumps done@step: 16, error=3.344781147740731e-11

julia> typeof(env)
SquareVUMPSRuntime{Float64, Array{Float64, 4}, Array{Float64, 3}, Matrix{Float64}} (alias for VUMPSRuntime{SquareLattice, Float64, 4, Array{Float64, 4}, Array{Float64, 3}, Array{Float64, 2}})
```
Using the `Z` function, we can calculate the partition function of the model per site:
```julia
julia> Z(env)
2.7893001925286596
```
Given the partition function, we get the free energy as the first derivative with respect to `β` times `-1`.
With Zygote, this is straightforward to calculate:
```julia

julia> e = β -> -log(Z(vumps_env(Ising(),β,2;verbose=true)))
#1 (generic function with 1 method)

julia> using Zygote

julia> Zygote.gradient(e,0.5)[1]
random initial vumps environment-> vumps done@step: 14, error=7.015494617183392e-11
-1.7456736441068634
```
more result is ploted by `/plot/2Dising.jl`

<div align="center"><img src="./plot/2Disingmag.svg" width="300px" alt="2Disingmag" div><img src="./plot/2Disingdmag.svg" width="300px" alt="2Disingmag" div><img src="./plot/2Disingene.svg" width="300px" alt="2Disingmag" div></div>

### Finding the Ground State of infinite 2D Heisenberg model

The other algorithm variationally minimizes the energy of a Heisenberg model on a two-dimensional infinite lattice using a form of gradient descent.

First, we need the hamiltonian as a tensor network operator
```
julia> h = hamiltonian(Heisenberg())
2×2×2×2 Array{Float64,4}:
[:, :, 1, 1] =
 -0.5  0.0
  0.0  0.5

[:, :, 2, 1] =
  0.0  0.0
 -1.0  0.0

[:, :, 1, 2] =
 0.0  -1.0
 0.0   0.0

[:, :, 2, 2] =
 0.5   0.0
 0.0  -0.5
```
where we get the `Heisenberg`-hamiltonian with default parameters `Jx = Jy = Jz = 1.0`.
Next we initialize an ipeps-tensor and calculate the energy of that tensor and the hamiltonian:
```julia
julia> ipeps, key = init_ipeps(Heisenberg(); D=2, χ=4, tol=1e-10, maxiter=20);
random initial iPEPS

julia> energy(h, Heisenberg(), ipeps; χ=4, tol=1e-6, maxiter=20, verbose=true)
random initial vumps environment-> vumps done@step: 2, error=7.229086826298654e-7
-0.5002176649083763
```
where the initial energy is random.

To minimise it, we combine `Optim` and `Zygote` under the hood to provide the `optimiseipeps` function. The `key` is used to save `.log` file and finial `ipeps` `.jld2` file.
```julia
julia> using Optim

julia> res = optimiseipeps(ipeps, key; f_tol=1e-6);
0.0   0   -0.5002176912673604   0.008382963041473053
2.56   1   -0.5833740684415737   0.036353323796568957
4.06   2   -0.6032224271322908   0.11180248756665447
6.14   3   -0.6412611745937832   0.08263739617395853
7.04   4   -0.6507187692219533   0.021762124786359978
8.05   5   -0.657878638331537   0.02118473585803891
8.98   6   -0.6593281299354107   0.01682222598631693
9.84   7   -0.6598795451002519   0.0029678925459859914
10.9   8   -0.6600140519479119   0.002229621321593708
11.44   9   -0.6600503541103068   0.001724388896082868
12.22   10   -0.6600780541170598   0.0034572055646887173
13.33   11   -0.6601597903538863   0.0010594850274929899
14.11   12   -0.6601713118969584   0.00037764468229989703
15.17   13   -0.66019933476685   0.001559732147205589
15.96   14   -0.6602144819385728   0.0005130015753926326
16.77   15   -0.6602187163533778   0.0005579122480867849
17.56   16   -0.66022143422884   0.0003642683231177844
18.6   17   -0.6602303218605757   0.00011867378702528263
19.4   18   -0.6602307629374299   4.215107085494706e-5
20.44   19   -0.6602310645788612   5.4946148767987396e-5
```
where our final value for the energy `e = -0.6602` agrees with the value found in the paper.

## For more
More Extension of VUMPS is available in package [VUMPS.jl](https://github.com/XingyuZhang2018/VUMPS.jl), including:

- Complex number forward and backward propagation
- `NixNj` Big Unit Cell 
- U1-symmmetry and Z2-symmmetry