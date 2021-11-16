# contains some utils for Fermionic Tensor Network Construction
export parity_conserving,swapgate,fdag,bulk,ipeps_enviroment,square_ipeps_contraction
using OMEinsum,LinearAlgebra,BitBasis,ChainRulesCore,CUDA,TimerOutputs
export SpinfulFermions,SpinlessFermions,HamiltonianModel

include("contractrules.jl")

"""
    parity_conserving(T::Array)

Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form

# example

```julia
julia> T = rand(2,2,2)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.863822  0.133604
 0.865495  0.371586

[:, :, 2] =
 0.581621  0.819325
 0.197463  0.801167

julia> parity_conserving(T)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.863822  0.0
 0.0       0.371586

[:, :, 2] =
 0.0       0.819325
 0.197463  0.0
```
"""
function parity_conserving(T::Union{Array,CuArray}) where V<:Real
	s = size(T)
	@assert prod(size(T))%2 == 0
	T = reshape(T,[2 for i = 1:Int(log2(prod(s)))]...)
	p = zeros(size(T))
	for index in CartesianIndices(T)
		if mod(sum([i for i in Tuple(index)].-1),2) == 0
			p[index] = 1
		end
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

"""
    function swapgate(n1::Int,n2::Int)

Generate a tensor which represent swapgate in Fermionic Tensor Network. n1,n2 should be power of 2.
The generated tensor have 4 indices. (ijkl).
S(ijkl) = delta(ik)*delta(jl)*parity(gate)

# example
```
julia> swapgate(2,4)
2×4×2×4 Array{Int64, 4}:
[:, :, 1, 1] =
 1  0  0  0
 0  0  0  0

[:, :, 2, 1] =
 0  0  0  0
 1  0  0  0

[:, :, 1, 2] =
 0  1  0  0
 0  0  0  0

[:, :, 2, 2] =
 0   0  0  0
 0  -1  0  0

[:, :, 1, 3] =
 0  0  1  0
 0  0  0  0

[:, :, 2, 3] =
 0  0   0  0
 0  0  -1  0

[:, :, 1, 4] =
 0  0  0  1
 0  0  0  0

[:, :, 2, 4] =
 0  0  0  0
 0  0  0  1
```
"""
function swapgate(n1::Int,n2::Int)
	S = ein"ij,kl->ikjl"(Matrix{Float64}(I,n1,n1),Matrix{Float64}(I,n2,n2))
	for i = 1:n1, j = 1:n2
        if sum(bitarray(i-1,Int(ceil(log(2, n1)))))%2 !=0 && sum(bitarray(j-1,Int(ceil(log(2, n2)))))%2 !=0
            S[i,j,:,:] .= -S[i,j,:,:]
        end
	end
	return S
end
@non_differentiable swapgate(n1::Int,n2::Int)

"""
	function paritygate(n::Int)
		return a parity gate.(Matrix)

Example:
julia> paritygate(4)
4×4 Matrix{Float64}:
 1.0   0.0   0.0  0.0
 0.0  -1.0   0.0  0.0
 0.0   0.0  -1.0  0.0
 0.0   0.0   0.0  1.0
"""
function paritygate(n::Int)
	S = Matrix{Float64}(I,n,n)
	for i = 1:n
		if sum(bitarray(i-1,Int(ceil(log(n)/log(2)))))%2 !=0 
			S[i,i] = -1
		end
	end
	return S
end
@non_differentiable paritygate(n::Int)


"""
	each bond exist a bond which is responsible for exchange of virtual complex fermions.
		
		julia> bondgate(2)
		4×4 Matrix{Float64}:
		 1.0  0.0  0.0   0.0
		 0.0  1.0  0.0   0.0
		 0.0  0.0  1.0   0.0
		 0.0  0.0  0.0  -1.0
		
		julia> bondgate(3)
		8×8 Matrix{Float64}:
		 1.0  0.0  0.0   0.0  0.0   0.0   0.0   0.0
		 0.0  1.0  0.0   0.0  0.0   0.0   0.0   0.0
		 0.0  0.0  1.0   0.0  0.0   0.0   0.0   0.0
		 0.0  0.0  0.0  -1.0  0.0   0.0   0.0   0.0
		 0.0  0.0  0.0   0.0  1.0   0.0   0.0   0.0
		 0.0  0.0  0.0   0.0  0.0  -1.0   0.0   0.0
		 0.0  0.0  0.0   0.0  0.0   0.0  -1.0   0.0
		 0.0  0.0  0.0   0.0  0.0   0.0   0.0  -1.0
"""
function bondgate(Nv::Int)
	ind = CartesianIndices(Tuple([1:2 for i =1:Nv]))
    n = zeros(Int,Nv) # store n_i
    p = zeros([2 for i =1:Nv]...)
    for index in ind
        for i = 1:Nv
            n[i] = Tuple(index)[i]
        end
        n = n.-1
        p[index] = fsign(n)
    end
    return Array(Diagonal(p[:]))
end
@non_differentiable bondgate(Nv::Int)

"""
	fsign(n) = n1*(n2+n3+n4...) + n2(n3+n4...)
	coming from exchange complex fermions on each bond.
"""
function fsign(n::Array{Int})
    result = 0
    for i = 2:length(n)
        result += n[i]*sum(n[1:i-1])
    end
    return (-1)^mod(result,2)
end
@non_differentiable fsign(n::Array)

"""
	function add_bondgate(T::Array,dim::N,Nv::Int)	

		add an bondgate for T at dim with Nv
"""
function add_bondgate(T::Array,dim::Int,Nv::Int)
	s = size(T)
	@assert s[dim]==2^Nv
	
	perm = collect(1:length(s))
	perm[dim] = 1
	perm[1] = dim

	T = permutedims(T,perm)
	T = reshape(T,s[dim],:)
	T = permutedims(reshape(ein"ij,io->oj"(T,bondgate(Nv)),s[perm]),perm)
	return T
end



"""
    function fdag(T::Array{V,5}) where V<:Number

Obtain dag tensor for local peps tensor in Fermionic Tensor Network(by inserting swapgates). The input tensor has indices which labeled by (lurdf)
legs are counting from f and clockwisely.

input legs order: ulfdr
output legs order: ulfdr
"""
function fdag(T::Union{Array{V,5},CuArray{V,5}}) where V<:Number
	nu, nl, _, nd, nr = size(T)
	Tdag = conj(T)
	
	Tdag = ein"ulfdr,luij,rdpq->jifqp"(Tdag,_arraytype(T)(swapgate(nl,nu)),_arraytype(T)(swapgate(nr,nd)))
	return Tdag	
end


"""
    function bulk(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulk(T::Union{Array{V,5},CuArray{V,5}}) where V<:Number
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T)
	# u l s d r
	# eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,10,11),(4,9,12,13)),(1,11,10,7,8,12,13,5))
	eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,10,11),(4,9,12,13)),(11,1,7,10,8,12,13,5))
	S1 = _arraytype(T)(swapgate(nl,nu))
	S2 = _arraytype(T)(swapgate(nd,nr))
	return	_arraytype(T)(reshape(einsum(eincode,(T,Tdag,S1,S2)),nu^2,nl^2,nd^2,nr^2))
end

"""
    function op_bulk(T::Union{Array{V,5},CuArray{V,5}},op::Union{Array{V,5},CuArray{V,5}}) where V<:Number
    
	This is a bulk tensor, but insert a operator on-site
"""
function op_bulk(T::Union{Array{V,5},CuArray{V,5}},op::Union{Array{Vo,2},CuArray{Vo,2}}) where {V<:Number,Vo<:Number}
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T)
	# u l s d r
	# eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,10,11),(4,9,12,13)),(1,11,10,7,8,12,13,5))
	eincode = EinCode(((1,2,3,4,5),(3,14),(6,7,14,8,9),(2,6,10,11),(4,9,12,13)),(11,1,7,10,8,12,13,5))
	S1 = _arraytype(T)(swapgate(nl,nu))
	S2 = _arraytype(T)(swapgate(nd,nr))
	o = _arraytype(T)(op)
	return	_arraytype(T)(reshape(einsum(eincode,(T,o,Tdag,S1,S2)),nu^2,nl^2,nd^2,nr^2))
end

using ADVUMPS:obs_env
"""
	calculate enviroment (E1...E6)
	a ────┬──── c 
	│     b     │ 
	├─ d ─┼─ e ─┤ 
	│     g     │ 
	f ────┴──── h 
	order: adf,abc,dgeb,fgh,ceh
"""
function ipeps_enviroment(T::AbstractArray;χ=20,maxiter=20,show_every=Inf,infile=nothing,outfile=nothing)
	b = permutedims(bulk(T),(2,3,4,1))
	_, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FL, FR = obs_env(b;χ=χ,maxiter=maxiter,verbose=true,downfromup=false,show_every=show_every,tol=1E-10,infile=infile,outfile=outfile);

	E1 = FLo
	E2 = ein"ijk,kp->ijp"(ALu,Cu)
	E3 = ARu
	E4 = FRo
	E5 = ein"ijk,kp->ijp"(ALd,Cd)
	E6 = ARd
	E7 = FL
	E8 = FR

	(E1,E2,E3,E4,E5,E6,E7,E8) = map(_arraytype(T),(E1,E2,E3,E4,E5,E6,E7,E8))
	return (E1,E2,E3,E4,E5,E6,E7,E8)
end

function bc_ipeps_enviroment(T::AbstractArray;χ=20,maxiter=20,show_every=Inf)
	b = permutedims(bulk(T),(2,3,4,1))
end


@non_differentiable Matrix{Float64}(I::UniformScaling{Bool},x,y)
function ipeps_energy(T::Array,t::Real,Δ::Real,μ::Real;χ=40,maxiter=20)
	T = parity_conserving(T)
	enviroments = ipeps_enviroment(T,χ=χ,maxiter=maxiter)
	E = square_ipeps_contraction_horizontal(T,enviroments,observables=hamiltonian(SpinfulFermions(t,Δ,μ)))
	n = square_ipeps_contraction_horizontal(T,enviroments,observables=Matrix{Float64}(I,16,16))
	return E/n
end

function vertical_ipeps_energy(T::Array,t::Real,Δ::Real,μ::Real;χ=40,maxiter=20)
	T = parity_conserving(T)
	enviroments = ipeps_enviroment(T,χ=χ,maxiter=maxiter)
	E = square_ipeps_contraction_vertical(T,enviroments,observables=hamiltonian(SpinfulFermions(t,Δ,μ)))
	n = square_ipeps_contraction_vertical(T,enviroments,observables=Matrix{Float64}(I,16,16))
	return E/n
end


"""
	Rotate ipeps local tensor.
	from ulfdr to rufld
"""
function rotate_ipeps(T::Array)
	nu,nl,nf,nd,nr = size(T)
	Tr = permutedims(T,(5,1,3,2,4)) # add r^2 + fl parity
	Tr = ein"rufld,rj->jufld"(Tr,paritygate(size(Tr)[1])) # add gate for r^2 term
	Tr = ein"rufld,flij->ruijd"(Tr,swapgate(nf,nl))
	return Tr
end

"""
	Rotate ipeps local tensor.
	from ulfdr to rufld
"""
function rotate_ipeps_withoutbond(T::Array)
	nu,nl,nf,nd,nr = size(T)
	Tr = permutedims(T,(5,1,3,2,4)) # add r^2 + fl parity
	# Tr = ein"rufld,rj->jufld"(Tr,paritygate(size(Tr)[1])) # add gate for r^2 term
	Tr = ein"rufld,flij->ruijd"(Tr,swapgate(nf,nl))
	return Tr
end

"""
	Rotate T naivelys; comparing to rotate_ipeps, this function does not add any swapgate.
"""
function naive_rotate_ipeps(T::Array)
	Tr = permutedims(T,(5,1,3,2,4)) # add r^2 + fl parity
	return Tr
end

function double_ipeps_energy(T::Union{Array,CuArray},t::Real,Δ1X::Real,Δ1Y::Real,μ::Real;χ=80,maxiter=20,show_every=Inf,in_enviroment_file=nothing,out_enviroment_file=nothing,timer=TimerOutput(),infile=nothing,outfile=nothing)
	T = parity_conserving(T)

	@timeit timer "Obtain Enviroment" begin
		enviroments = ipeps_enviroment(T,χ=χ,maxiter=MaxIter,show_every=5;infile=infile,outfile=outfile)
	end
	
	@timeit timer "Horizontal Contraction" begin
		ρ = square_ipeps_contraction_horizontal(T,map(_arraytype(T),enviroments),observables=_arraytype(T)(hamiltonian(SpinfulFermions(t,Δ1X,μ))))

		n = square_ipeps_contraction_horizontal(T,map(_arraytype(T),enviroments),observables=_arraytype(T)(Matrix{Float64}(I,16,16)))
		e1 = E/n
	end

	@timeit timer "Vertical Contraction" begin
		E = square_ipeps_contraction_vertical(T,map(_arraytype(T),enviroments),observables=_arraytype(T)(hamiltonian(SpinfulFermions(t,Δ1Y,μ))))
		n = square_ipeps_contraction_vertical(T,map(_arraytype(T),enviroments),observables=_arraytype(T)(Matrix{Float64}(I,16,16)))
		e2 = E/n
	end

	print("VH=$(e1);VE=$(e2)\n")
	return e1+e2
end

function square_ipeps_contraction_vertical(T,env)
	nu, _, nf, _, _ = size(T)
	χ = size(env[1])[1]
	(E1,_,E3,E4,_,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

    swapgatedD = swapgate(nf,nu)
    swapgateDD = swapgate(nu,nu)

	result = VERTICAL_RULES(T,fdag(T),swapgateDD,swapgatedD,
	swapgatedD,swapgateDD,T,fdag(T),swapgateDD,
	swapgatedD,swapgatedD,swapgateDD,E3,E8,E4,E6,E1,E7)
	return CUDA.@allowscalar result[1]
end

function square_ipeps_contraction_horizontal(T,env;observables=Matrix{Float64}(I,16,16))
	nu,nl,nf,nd,nr = size(T)
	χ = size(env[1])[1]
	(E1,E2,E3,E4,E5,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

	optcode(x) = HORIZONTAL_RULES(map(_arraytype(T),x)...)
	result = optcode([T,swapgatedD,
	fdag(T),swapgatedD,swapgateDD,
	swapgateDD,
	fdag(T),swapgatedD,
	swapgatedD,swapgateDD,T,
	swapgateDD,
	reshape(observables,4,4,4,4),
	E1,E2,E3,E4,E5,E6])
	return CUDA.@allowscalar result[1]
end