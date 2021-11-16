using OMEinsum,OMEinsumContractionOrders,Random

function generate_vertical_rules(;Nv=2,χ=800)
	eincode = EinCode(((1,2,3,4,5),# T1
	(6,7,8,9,10),#T2 (dag)
	(2,6,11,12), #swapgate(nl,nu)
	(8,10,14,13), #swapgate(nf,nu)
	(3,13,15,16), #swapgate(nf,nr)
	(4,16,17,18), #swapgate(nl,nu)
	(17,19,20,21,22), #T4
	(23,24,25,26,27), #T3 (dag)
	(21,27,28,29),#swapgate(nl,nu)
	(20,19,30,31),#swapgate(nf,nr)
	(25,31,32,33),#swapgate(nf,nr)
	(23,33,9,34), #swapgate(nl,nu)
	(15,30,14,32), #hamiltonian (ij di dj)
	(35,12,1,36), # Eup: E3
	(36,18,5,37), # FRu: E8
	(37,29,22,38), # FRo: E4
	(38,26,28,39),# Edn: E6
	(39,24,34,40), # FLo: E1
	(40,7,11,35) # FLu: E7
	),())
		
	size_dict = [2^Nv for i = 1:40]
	size_dict[[3;8;14;15;30;32;20;25]] .= 4
	size_dict[35:40] .= χ
	sd = Dict(i=>size_dict[i] for i = 1:40)

	# for seed =40:100
	seed = 60
	Random.seed!(60)
	optcode = optimize_tree(eincode,sd; sc_target=28, βs=0.1:0.1:10, ntrials=2, niters=100, sc_weight=3.0)


	print("Vertical Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(optcode,sd),"\n") 
	# You would better try some times to make it optimal (By the for-end iteration...)
	# end
	
	return optcode
end
@non_differentiable generate_vertical_rules()
const VERTICAL_RULES = generate_vertical_rules()

function generate_horizontal_rules(;Nv=2,χ=800)
    eincode = EinCode(((1,2,3,4,5),# T1
    (3,4,21,22),#swapgate(nf,nu)

	(6,7,8,9,10),#T2 (dag)
    (8,22,23,27),#swapgate(nf,nu)
    (10,40,17,27), #swapgate(nl,nu)

    (2,6,29,30), #swapgate(nl,nu)

    (16,17,18,19,20),# T4(dag)
    (18,26,25,16),#swapgate(nf,nu)

    (13,28,24,26),#swapgate(nf,nu)
    (12,28,5,33),#swapgate(nl,nu)
    (11,12,13,14,15),# T3

    (20,32,31,14),#swapgate(nl,nu)

    (21,24,23,25), #hamiltonian (ij di dj)
    (38,7,29,39), #E1 FLo
    (39,30,1,34), #E2 ALu
    (34,33,11,35), #E3 ACu
    (35,31,15,36), #E4 FRo
    (36,19,32,37), #E5 ARd
    (37,9,40,38), #E6 ACd
	),())

    size_dict = [2^Nv for i = 1:40]
	size_dict[[3;8;13;18;21;23;24;25]] .= 4
	size_dict[34:39] .= χ
	sd = Dict(i=>size_dict[i] for i = 1:40)
	
	# for seed = 1:100
	seed = 4
	Random.seed!(seed)
	optcode = optimize_tree(eincode,sd; sc_target=28, βs=0.1:0.1:10, ntrials=3, niters=100, sc_weight=4.0)
	print("Horizontal Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(optcode,sd),"\n")
	# end

	return optcode
end
@non_differentiable generate_horizontal_rules()
const HORIZONTAL_RULES = generate_horizontal_rules()
