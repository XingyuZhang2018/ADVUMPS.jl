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
> git clone https://github.com/XingyuZhang2018/ADVUMPS
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.5) pkg> activate .
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
julia> env = vumps_env(Ising(),0.5,2)
random initial -> vumps done@step: 12, error=2.7568658004257003e-11
VUMPSRuntime{SquareLattice,Float64,4,Array{Float64,4},Array{Float64,3},Array{Float64,2}}([2.5343421078733233 0.4999999999999999; 0.4999999999999999 0.1839397205857211]       

[0.4999999999999999 0.18393972058572117; 0.18393972058572117 0.4999999999999999]

[0.4999999999999999 0.18393972058572117; 0.18393972058572117 0.4999999999999999]

[0.1839397205857211 0.4999999999999999; 0.4999999999999999 2.5343421078733233], [0.9636875569130792 0.2645270601158087; -0.005746618071896477 0.03603752902356838]

[-0.243495242690347 0.8206059735819796; 0.10946842384361864 0.5053044301111299], [-0.9665897652707997 -0.2521389201547527; -0.017917438412920428 0.04253417470288124], [0.844355900146621 0.4945477837633195; 0.44016190368628316 -0.823484609823201]

[0.20241327907002685 -0.03891616525941448; 0.22880008060963664 0.2752837064568502], [0.8353899271065874 0.23247291736259634; -0.041338388733611835 0.151232686670938]

[-0.0413383887336118 0.15123268667093798; 0.09588003412655502 0.43557623401198353], [-0.765320673936658 -0.32158114057618753; -0.22049051693208743 0.18173944466903963]       

[-0.2204905169320874 0.1817394446690396; -0.1659492872970294 -0.34646801079650336])
```
Using the `Z` function, we can calculate the partition function of the model per site:
```
julia> Z(env)
2.7893001925286596
```
Given the partition function, we get the free energy as the first derivative with respect to `β` times `-1`.
With Zygote, this is straightforward to calculate:
```julia

julia> e = β -> -log(Z(vumps_env(Ising(),β,2)))
#1 (generic function with 1 method)

julia> using Zygote

julia> Zygote.gradient(e,0.5)[1]
random initial -> vumps done@step: 12, error=4.676523961729668e-11
-1.7455677143228514
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

julia> ADVUMPS.energy(h, ipeps, χ=4, tol=1e-6, maxiter=20)
random initial vumps environment-> vumps done@step: 21, error=6.787978724709051e-6
-0.506219066603934
```
where the initial energy is random.

To minimise it, we combine `Optim` and `Zygote` under the hood to provide the `optimiseipeps` function. The `key` is used to save `.log` file and finial `ipeps` `.jld2` file.
```julia
julia> using Optim

julia> res = optimiseipeps(ipeps, h, key; f_tol=1e-6);
random initial vumps environment-> vumps done@step: 4, error=1.0489914354483037e-12
random initial vumps environment-> vumps done@step: 4, error=1.2732519803957594e-13
0.0s   0   -0.50059619356879   0.014102449838274057
random initial vumps environment-> vumps done@step: 3, error=9.154139619711965e-11
random initial vumps environment-> vumps done@step: 4, error=9.430263945021394e-13
random initial vumps environment-> vumps done@step: 4, error=3.979956712626124e-12
random initial vumps environment-> vumps done@step: 4, error=5.9929398056604074e-12
random initial vumps environment-> vumps done@step: 10, error=7.652370287706936e-11
random initial vumps environment-> vumps done@step: 10, error=2.4101525463803696e-11
random initial vumps environment-> vumps done@step: 16, error=9.381476715773855e-11
random initial vumps environment-> vumps done@step: 15, error=5.937732510044041e-11
1.65s   1   -0.6385044099206219   0.10899289517449176
random initial vumps environment-> vumps done@step: 9, error=9.951902201226095e-11
random initial vumps environment-> vumps done@step: 11, error=6.641486783443332e-11
random initial vumps environment-> vumps done@step: 11, error=6.015094120795308e-11
random initial vumps environment-> vumps done@step: 11, error=9.276293045844143e-11
2.65s   2   -0.6545898839653035   0.032162408113119406
random initial vumps environment-> vumps done@step: 8, error=5.124716627105218e-11
random initial vumps environment-> vumps done@step: 9, error=1.92449129673697e-11
random initial vumps environment-> vumps done@step: 6, error=2.3698881225420333e-11
random initial vumps environment-> vumps done@step: 6, error=5.797919341685305e-11
random initial vumps environment-> vumps done@step: 7, error=1.5401397944317312e-11
random initial vumps environment-> vumps done@step: 6, error=2.2618857635151177e-11
3.75s   3   -0.6573375734169976   0.039561569212385586
random initial vumps environment-> vumps done@step: 5, error=2.549137854523396e-11
random initial vumps environment-> vumps done@step: 5, error=6.68406692580882e-11
random initial vumps environment-> vumps done@step: 17, error=6.366884458126922e-11
random initial vumps environment-> vumps done@step: 17, error=9.934533464593135e-11
random initial vumps environment-> vumps done@step: 6, error=5.883791256334852e-11
random initial vumps environment-> vumps done@step: 6, error=5.83406096151748e-12
5.12s   4   -0.6596136233294728   0.015573329394085754
random initial vumps environment-> vumps done@step: 5, error=3.7539203414861e-12
random initial vumps environment-> vumps done@step: 5, error=6.877548650353064e-12
random initial vumps environment-> vumps done@step: 5, error=1.616342689952346e-11
random initial vumps environment-> vumps done@step: 5, error=1.2055035755836548e-11
5.7s   5   -0.6599632054097899   0.005715266393653051
random initial vumps environment-> vumps done@step: 4, error=7.192744864619857e-11
random initial vumps environment-> vumps done@step: 5, error=2.5455979927330197e-11
random initial vumps environment-> vumps done@step: 5, error=6.344939627625178e-12
random initial vumps environment-> vumps done@step: 5, error=1.201023390861985e-11
random initial vumps environment-> vumps done@step: 5, error=1.1552809545998206e-11
random initial vumps environment-> vumps done@step: 5, error=8.895594361977555e-12
6.52s   6   -0.6600382797191856   0.004535057733191031
random initial vumps environment-> vumps done@step: 5, error=5.2951673597294826e-12
random initial vumps environment-> vumps done@step: 5, error=3.787365760472808e-12
random initial vumps environment-> vumps done@step: 4, error=9.749135107967033e-11
random initial vumps environment-> vumps done@step: 5, error=5.25082198278789e-12
random initial vumps environment-> vumps done@step: 6, error=1.0974055538302087e-11
random initial vumps environment-> vumps done@step: 6, error=3.9295189593627114e-11
random initial vumps environment-> vumps done@step: 5, error=1.9693125758675934e-12
random initial vumps environment-> vumps done@step: 5, error=5.144487725615619e-12
7.66s   7   -0.6601481008317902   0.0018887172494824567
random initial vumps environment-> vumps done@step: 5, error=5.667439944354684e-12
random initial vumps environment-> vumps done@step: 5, error=3.2772515949642094e-12
random initial vumps environment-> vumps done@step: 5, error=2.123867676057697e-12
random initial vumps environment-> vumps done@step: 5, error=1.5835486025099777e-11
random initial vumps environment-> vumps done@step: 5, error=8.213707034045823e-12
random initial vumps environment-> vumps done@step: 5, error=2.664839932267421e-12
8.51s   8   -0.6602134912910089   0.002071590543300587
random initial vumps environment-> vumps done@step: 5, error=3.015076741488207e-12
random initial vumps environment-> vumps done@step: 5, error=2.7480885164944934e-12
random initial vumps environment-> vumps done@step: 5, error=3.5444812888396305e-12
random initial vumps environment-> vumps done@step: 5, error=3.012376398925062e-12
random initial vumps environment-> vumps done@step: 5, error=4.232969341834662e-12
random initial vumps environment-> vumps done@step: 5, error=1.2860570439812016e-12
9.38s   9   -0.6602213793592919   0.00032942396847243606
random initial vumps environment-> vumps done@step: 5, error=2.7681478003954194e-12
random initial vumps environment-> vumps done@step: 5, error=4.713808518097429e-12
random initial vumps environment-> vumps done@step: 5, error=2.589606449290332e-12
random initial vumps environment-> vumps done@step: 5, error=3.457230578419805e-12
random initial vumps environment-> vumps done@step: 5, error=2.723682876267463e-12
random initial vumps environment-> vumps done@step: 5, error=5.23086868748299e-12
10.27s   10   -0.6602221482821853   0.00036791238293452775
random initial vumps environment-> vumps done@step: 5, error=6.923202995208043e-12
random initial vumps environment-> vumps done@step: 5, error=7.491563083179787e-12
random initial vumps environment-> vumps done@step: 5, error=3.574972371996455e-12
random initial vumps environment-> vumps done@step: 5, error=1.3405389879914542e-12
random initial vumps environment-> vumps done@step: 4, error=8.022341320900762e-11
random initial vumps environment-> vumps done@step: 5, error=1.7423508186029293e-12
random initial vumps environment-> vumps done@step: 5, error=1.151866438861155e-11
random initial vumps environment-> vumps done@step: 5, error=5.746381904160193e-13
11.38s   11   -0.6602278618403974   0.0005036918984882396
random initial vumps environment-> vumps done@step: 5, error=5.634813509709247e-12
random initial vumps environment-> vumps done@step: 5, error=2.1792647028045365e-12
random initial vumps environment-> vumps done@step: 5, error=1.757038676455262e-12
random initial vumps environment-> vumps done@step: 5, error=1.195586351550817e-12
random initial vumps environment-> vumps done@step: 5, error=3.736149941173614e-12
random initial vumps environment-> vumps done@step: 5, error=1.7166160043399526e-12
12.25s   12   -0.6602296412719777   0.0001672531647296988
random initial vumps environment-> vumps done@step: 5, error=5.80563553155269e-12
random initial vumps environment-> vumps done@step: 5, error=2.2725647645770175e-12
random initial vumps environment-> vumps done@step: 5, error=3.925135356312597e-12
random initial vumps environment-> vumps done@step: 5, error=1.8290561684937775e-12
random initial vumps environment-> vumps done@step: 5, error=1.5082826189334574e-12
random initial vumps environment-> vumps done@step: 5, error=3.1896435447993355e-12
random initial vumps environment-> vumps done@step: 5, error=3.1692368278357683e-12
random initial vumps environment-> vumps done@step: 5, error=4.068355689126351e-12
13.38s   13   -0.6602304380655973   0.00017266087486525548
random initial vumps environment-> vumps done@step: 4, error=7.8959721400154e-11
random initial vumps environment-> vumps done@step: 5, error=3.1588355718029277e-12
random initial vumps environment-> vumps done@step: 5, error=3.797164129868554e-12
random initial vumps environment-> vumps done@step: 5, error=3.540769958525511e-12
random initial vumps environment-> vumps done@step: 5, error=3.455155843639284e-12
random initial vumps environment-> vumps done@step: 5, error=4.270376389093168e-12
14.23s   14   -0.6602310478803142   3.24337247078898e-5
random initial vumps environment-> vumps done@step: 5, error=2.808204269321628e-12
random initial vumps environment-> vumps done@step: 5, error=1.0854196484758662e-11
random initial vumps environment-> vumps done@step: 5, error=2.5279854183494677e-12
random initial vumps environment-> vumps done@step: 5, error=4.717683859453325e-12
random initial vumps environment-> vumps done@step: 5, error=9.165939324482338e-12
random initial vumps environment-> vumps done@step: 5, error=3.8268423259016495e-12
15.09s   15   -0.660231052548369   3.0683534939298076e-5
```
where our final value for the energy `e = -0.6602` agrees with the value found in the paper.

## to do 

* complex iPEPS and MPS

For complex situation `A` and `A'` are different independent variables, so must to input two variables.
