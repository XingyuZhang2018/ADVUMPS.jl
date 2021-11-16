using BitBasis
using LinearAlgebra
using ITensors
using OMEinsum


function swapgate(n1::Int,n2::Int)
	S = ein"ij,kl->ikjl"(Matrix{Float64}(I,n1,n1),Matrix{Float64}(I,n2,n2))
	for i = 1:n1
		for j = 1:n2
			if sum(bitarray(i-1,Int(ceil(log(n1)/log(2)))))%2 !=0 && sum(bitarray(j-1,Int(ceil(log(n2)/log(2)))))%2 !=0
				S[i,j,:,:] .= -S[i,j,:,:]
			end
		end
	end
	return S
end

function new_swapgate(n1::Int,n2::Int)
    S = zeros(n1,n2,n1,n2)
	for i = 1:n1, j = 1:n2
        if sum(bitarray(i-1,Int(log(2,n1))))%2 != 0 && sum(bitarray(j-1,Int(log(2,n2))))%2 != 0
            S[i,j,i,j] = -1
        else
            S[i,j,i,j] = 1
        end
	end
	return S
end

A = swapgate(8,4)
B = new_swapgate(8,4)
@show A == B