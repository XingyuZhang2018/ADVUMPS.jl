# ADVUMPS

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

<div align="center"><img src="./plot/2Disingmag.svg" width="120px" alt="2Disingmag" div><img src="./plot/2Disingdmag.svg" width="120px" alt="2Disingmag" div><img src="./plot/2Disingene.svg" width="120px" alt="2Disingmag" div></div>

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
julia> ipeps = SquareIPEPS(rand(2,2,2,2,2));

julia> ipeps = ADVUMPS.indexperm_symmetrize(ipeps);

julia> ADVUMPS.energy(h,ipeps, χ=4, tol=1e-6,maxit=20)
random initial -> vumps done@step: 2, error=6.27524507817513e-9
-0.5085713947063569
```
where the initial energy is random.

To minimise it, we combine `Optim` and `Zygote` under the hood to provide the `optimiseipeps` function.
```julia
julia> using Optim

julia> res = optimiseipeps(ipeps, h; χ=4, tol=1e-6, maxit=20,
                      optimargs = (Optim.Options(f_tol=1e-6, show_trace=true),));
random initial -> vumps done@step: 2, error=5.932061250059548e-9
random initial -> vumps done@step: 2, error=2.022505856713312e-9
Iter     Function value   Gradient norm 
     0    -5.085714e-01     6.099997e-02
 * time: 0.0
random initial -> vumps done@step: 2, error=4.52244313510551e-8
random initial -> vumps done@step: 2, error=3.511023537714646e-8
random initial -> vumps done@step: 3, error=2.0284469254437323e-7
random initial -> vumps done@step: 3, error=1.4716273360000774e-7
random initial -> vumps done@step: 3, error=9.593628522795993e-8
random initial -> vumps done@step: 3, error=3.796127248639449e-8
     1    -6.503229e-01     7.509995e-02
 * time: 1.4280002117156982
random initial -> vumps done@step: 4, error=2.1152357602901073e-7
random initial -> vumps done@step: 4, error=7.015830531044039e-8
random initial -> vumps done@step: 3, error=8.022106942289034e-8
random initial -> vumps done@step: 3, error=8.496580545351456e-8
     2    -6.582497e-01     8.913537e-03
 * time: 1.880000114440918
random initial -> vumps done@step: 3, error=9.44635875771351e-8
random initial -> vumps done@step: 3, error=6.764041739661678e-7
random initial -> vumps done@step: 3, error=2.43068917446062e-7
random initial -> vumps done@step: 4, error=2.0027949072605566e-8
random initial -> vumps done@step: 3, error=2.309852948441025e-7
random initial -> vumps done@step: 3, error=2.239364706295199e-7
     3    -6.586733e-01     1.915860e-02
 * time: 2.509000062942505
random initial -> vumps done@step: 3, error=1.6655328377726225e-7
random initial -> vumps done@step: 3, error=1.4492548980404085e-7
random initial -> vumps done@step: 3, error=3.911268768912821e-8
random initial -> vumps done@step: 3, error=3.009603348277458e-7
random initial -> vumps done@step: 3, error=1.799590487630004e-7
random initial -> vumps done@step: 3, error=8.5636102238974e-8
     4    -6.596982e-01     5.175987e-03
 * time: 3.128000020980835
random initial -> vumps done@step: 3, error=2.1413458365186302e-8
random initial -> vumps done@step: 3, error=1.3002714391992936e-7
random initial -> vumps done@step: 3, error=1.0312123492155217e-7
random initial -> vumps done@step: 3, error=9.935514996707667e-8
     5    -6.597726e-01     6.784199e-03
 * time: 3.5440001487731934
random initial -> vumps done@step: 4, error=1.915184058568321e-8
random initial -> vumps done@step: 3, error=1.3162312424862742e-7
random initial -> vumps done@step: 3, error=1.4392184658493842e-8
random initial -> vumps done@step: 3, error=1.1862937069190404e-7
random initial -> vumps done@step: 3, error=9.787483969427185e-8
random initial -> vumps done@step: 3, error=1.9317236128447688e-7
     6    -6.599181e-01     7.809312e-03
 * time: 4.182000160217285
random initial -> vumps done@step: 3, error=3.15660813772062e-8
random initial -> vumps done@step: 3, error=1.951174832536094e-8
random initial -> vumps done@step: 3, error=3.0578525850776337e-7
random initial -> vumps done@step: 3, error=1.9269067746273535e-7
random initial -> vumps done@step: 3, error=8.730911196439428e-8
random initial -> vumps done@step: 3, error=7.590004434192511e-8
     7    -6.601239e-01     2.521885e-03
 * time: 4.808000087738037
random initial -> vumps done@step: 3, error=8.098518662100826e-8
random initial -> vumps done@step: 3, error=5.337379647462432e-7
random initial -> vumps done@step: 3, error=6.26108823454905e-8
random initial -> vumps done@step: 3, error=3.809274705545814e-7
random initial -> vumps done@step: 3, error=2.542028733719036e-7
random initial -> vumps done@step: 3, error=3.971765905109794e-8
     8    -6.601608e-01     6.511008e-04
 * time: 5.424000024795532
random initial -> vumps done@step: 3, error=2.75081615517135e-8
random initial -> vumps done@step: 3, error=1.423437221509443e-8
random initial -> vumps done@step: 3, error=3.247063768880895e-7
random initial -> vumps done@step: 3, error=1.9243884768549126e-7
random initial -> vumps done@step: 3, error=3.045285385840119e-8
random initial -> vumps done@step: 3, error=1.612849287178833e-7
     9    -6.601807e-01     1.156388e-03
 * time: 6.041000127792358
random initial -> vumps done@step: 3, error=1.7985185077555728e-8
random initial -> vumps done@step: 3, error=1.0074368737552882e-7
random initial -> vumps done@step: 3, error=4.133804593003857e-8
random initial -> vumps done@step: 3, error=1.9941662828969343e-8
random initial -> vumps done@step: 3, error=1.043322374394592e-8
random initial -> vumps done@step: 3, error=2.859635101088892e-8
    10    -6.602058e-01     6.074573e-04
 * time: 6.656000137329102
random initial -> vumps done@step: 3, error=1.604554445669967e-7
random initial -> vumps done@step: 3, error=2.240933841146041e-7
random initial -> vumps done@step: 3, error=3.078486569886867e-8
random initial -> vumps done@step: 3, error=6.27216663694798e-8
random initial -> vumps done@step: 3, error=1.1897165037243467e-8
random initial -> vumps done@step: 3, error=2.200553122431604e-8
    11    -6.602150e-01     5.023356e-04
 * time: 7.26800012588501
random initial -> vumps done@step: 3, error=6.708030454243216e-8
random initial -> vumps done@step: 3, error=9.99614376907481e-9
random initial -> vumps done@step: 3, error=1.7906553908410097e-7
random initial -> vumps done@step: 3, error=5.0993333410340515e-8
random initial -> vumps done@step: 3, error=4.381896611190456e-8
random initial -> vumps done@step: 3, error=1.9346864723150772e-7
    12    -6.602300e-01     3.530335e-04
 * time: 7.882000207901001
random initial -> vumps done@step: 3, error=2.1421856634584414e-8
random initial -> vumps done@step: 3, error=8.544935028820756e-8
random initial -> vumps done@step: 3, error=9.9190967749313e-8
random initial -> vumps done@step: 3, error=2.747433045515976e-7
    13    -6.602303e-01     1.737444e-04
 * time: 8.296000003814697
random initial -> vumps done@step: 3, error=4.4313614221677196e-8
random initial -> vumps done@step: 3, error=2.1860287600194127e-8
random initial -> vumps done@step: 3, error=8.561316114773221e-8
random initial -> vumps done@step: 3, error=1.1682909198113247e-7
random initial -> vumps done@step: 3, error=5.429999806923313e-8
random initial -> vumps done@step: 3, error=4.337186261571496e-8
    14    -6.602306e-01     4.633383e-05
 * time: 8.91100001335144
```
where our final value for the energy `e = -0.6602` agrees with the value found in the paper.

## to do 

* complex IPEPS and MPS
