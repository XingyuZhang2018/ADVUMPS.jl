using LinearAlgebra
using KrylovKit

#https://github.com/JuliaGPU/CuArrays.jl/issues/283
safesign(x::Number) = iszero(x) ? one(x) : sign(x)
CUDA.@cufunc safesign(x::CublasFloat) = iszero(x) ? one(x) : x/abs(x)

"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A))
    Q = mattype(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A'))
    Q = mattype(mattype(F.Q)')
    L = mattype(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

"""
    leftorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
```
    ┌─AL─      ┌──      
    │ │     =  │                 
    ┕─AL─      ┕──    
```

"""
function leftorth(A, C = _mattype(A){eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, maxiter = 100, kwargs...)
    _, ρs, info = eigsolve(C'*C, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do ρ
        ρE = ein"(cd,dsb),csa -> ab"(ρ, A, conj(A))
        return ρE
    end
    ρ = ρs[1] + ρs[1]'
    ρ ./= tr(ρ)
    # C = cholesky!(ρ).U
    # If ρ is not exactly positive definite, cholesky will fail
    F = svd!(ρ)
    C = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
    _, C = qrpos!(C)

    D, d, = size(A)
    Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    rmul!(R, 1/λ)
    numiter = 1
    while norm(C-R) > tol && numiter < maxiter
        # C = R
        _, Cs, info = eigsolve(R, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do X
            Y = ein"(cd,dsb),csa -> ab"(X,A,conj(AL))
            return Y
        end
        _, C = qrpos!(Cs[1])
        # The previous lines can speed up the process when C is still very far from the correct
        # gauge transform, it finds an improved value of C by finding the fixed point of a
        # 'mixed' transfer matrix composed of `A` and `AL`, even though `AL` is also still not
        # entirely correct. Therefore, we restrict the number of iterations to be 1 and don't
        # check for convergence
        Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        rmul!(R, 1/λ)
        numiter += 1
    end
    C = R
    return real(AL), real(C), λ
end

"""
    rightorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a gauge transform C, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that `λ C AR^s = A^s C`, where an initial guess for `C` can be
provided.
````
    ─ AR─┐     ──┐  
      │  │  =    │  
    ─ AR─┘     ──┘  
````
"""
function rightorth(A, C = _mattype(A){eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, maxiter = 100, kwargs...)
    AL, C, λ = leftorth(permutedims(A,(3,2,1)), permutedims(C,(2,1)); tol = tol,maxiter = maxiter, kwargs...)
    return permutedims(C,(2,1)), permutedims(AL,(3,2,1)), λ
end

"""
    leftenv(A, M, FL; kwargs)

Compute the left environment tensor for MPS `AL` and MPO `M`, by finding the left fixed point
of `AL - M - conj(AL)` contracted along the physical dimension.
```
┌── AL─       ┌──         
│   │         │             
FL─ M ─  = λL FL─         
│   │         │             
┕── AL─       ┕──        
```
"""
function leftenv(AL, M, FL = _arraytype(AL)(rand(eltype(AL), size(AL,1), size(M,1), size(AL,1))); kwargs...)
    λs, FLs, info = eigsolve(FL -> ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false, kwargs...)
    # if length(λs) > 1 && norm(real(λs[1]) - real(λs[2])) < 1e-12
    #     @show λs
    #     if real(λs[1]) > 0
    #         return real(λs[1]), real(FLs[1])
    #     else
    #         return real(λs[2]), real(FLs[2])
    #     end
    # end
    return real(λs[1]), real(FLs[1])
end

"""
    rightenv(A, M, FR; kwargs...)

Compute the right environment tensor for MPS `AR` and MPO `M`, by finding the right fixed point
of `AR - M - conj(AR)`` contracted along the physical dimension.
```
 ─ AR──┐         ──┐   
   │   │           │   
 ─ M ──FR   = λR ──FR  
   │   │           │   
 ─ AR──┘         ──┘  
```
"""
function rightenv(AR, M, FR = _arraytype(AR)(randn(eltype(AR), size(AR,1), size(M,3), size(AR,1))); kwargs...)
    λs, FRs, info = eigsolve(FR -> ein"((αpγ,γcη),ascp),βsη -> αaβ"(AR,FR,M,conj(AR)), FR, 1, :LM; ishermitian = false, kwargs...)
    # if length(λs) > 1 && norm(real(λs[1]) - real(λs[2])) < 1e-12
    #     @show λs
    #     if real(λs[1]) > 0
    #         return real(λs[1]), real(FRs[1])
    #     else
    #         return real(λs[2]), real(FRs[2])
    #     end
    # end
    return real(λs[1]), real(FRs[1])
end

"""
    λ, FL = obs_FL(ALu, ALd, M, FL; kwargs...)

Compute the left environment tensor for observable, by finding the left fixed point
of `ALu - M - ALd` contracted along the physical dimension.
```
┌── ALu─      ┌──         
│   │         │             
FL─ M ─  = λL FL─         
│   │         │             
┕── ALd─      ┕──        
```
"""
function obs_FL(ALu, ALd, M, FL = _arraytype(ALu)(rand(eltype(ALu), size(ALu,1), size(M,1), size(ALd,1))); kwargs...)
    λs, FLs, info = eigsolve(FL -> ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,ALu,M,ALd), FL, 1, :LM; ishermitian = false, kwargs...)
    # println("obs_FL $(λs)") 
    return real(λs[1]), real(FLs[1])
end

"""
    λ, FR = obs_FR(ARu, ARd, M, FR; kwargs...)

Compute the right environment tensor for observable, by finding the right fixed point
of `ARu - M - ARd`` contracted along the physical dimension.
```
 ─ ARu──┐         ──┐   
    │   │           │   
 ─  M ──FR   = λR ──FR  
    │   │           │   
 ─ ARd──┘         ──┘  
```
"""
function obs_FR(ARu, ARd, M, FR = _arraytype(ARu)(randn(eltype(ARu), size(ARu,3), size(M,3), size(ARd,3))); kwargs...)
    λs, FRs, info = eigsolve(FR -> ein"((αpγ,γcη),ascp),βsη -> αaβ"(ARu,FR,M,ARd), FR, 1, :LM; ishermitian = false, kwargs...)
    # println("obs_FR $(λs)") 
    return real(λs[1]), real(FRs[1])
end

"""
    λ, FL = norm_FL(ALu, ALd, FL; kwargs...)

Compute the left environment tensor for normalization, by finding the left fixed point
of `ALu - ALd` contracted along the physical dimension.
```
┌──ALu─      ┌──         
FL  │  =  λL FL        
┕──ALd─      ┕──  
```
"""
function norm_FL(ALu, ALd, FL = _arraytype(ALu)(rand(eltype(ALu), size(ALu,1), size(ALd,1))); kwargs...)
    λs, FLs, info = eigsolve(FL -> ein"(ad,acb), dce -> be"(FL,ALu,ALd), FL, 1, :LM; ishermitian = false, kwargs...)
    # println("norm_FL $(λs)") 
    return real(λs[1]), real(FLs[1])
end

"""
    λ, FR = norm_FR(ARu, ARd, FR; kwargs...)

Compute the right environment tensor for normalization, by finding the right fixed point
of `ARu - ARd` contracted along the physical dimension.
```
 ─ AR──┐       ──┐   
   │   FR  = λR  FR   
 ─ AR──┘       ──┘ 
```
"""
function norm_FR(ARu, ARd, FR = _arraytype(ARu)(randn(eltype(ARu), size(ARu,3), size(ARd,3))); kwargs...)
    λs, FRs, info = eigsolve(FR -> ein"(be,acb), dce -> ad"(FR,ARu,ARd), FR, 1, :LM; ishermitian = false, kwargs...)
    # println("norm_FR $(λs)") 
    return real(λs[1]), real(FRs[1])
end

"""
    λ, FL4 = bigleftenv(AL, M, FL4 = rand(eltype(AL), size(AL,1), size(M,1), size(M,1), size(AL,1)); kwargs...)

Compute the left environment tensor for MPS `AL` and MPO `M`, by finding the left fixed point
of `AL - M - M - conj(AL)` contracted along the physical dimension.
```
┌── AL─       ┌──         
│   │         │             
│── M ─       │──       
FL4 │    = λL FL4  
│── M ─       │──
│   │         │
┕── AL─       ┕──        
```
"""
function bigleftenv(AL, M, FL4 = _arraytype(AL)(rand(eltype(AL), size(AL,1), size(M,1), size(M,1), size(AL,1))); kwargs...)
    λFL4s, FL4s, info = eigsolve(FL4 -> ein"(((dcba,def),ckge),bjhk),aji -> fghi"(FL4,AL,M,M,conj(AL)), FL4, 1, :LM; ishermitian = false, kwargs...)
    # @show λFL4s
    return real(λFL4s[1]), real(FL4s[1])
end

"""
    λ, FR4 = bigrightenv(AR, M, FR4 = randn(eltype(AR), size(AR,1), size(M,3), size(M,3), size(AR,1)); kwargs...)

Compute the right environment tensor for MPS `AR` and MPO `M`, by finding the right fixed point
of `AR - M - conj(AR)`` contracted along the physical dimension.
```
 ─ AR──┐         ──┐ 
   │   │           │ 
 ─ M ──│         ──│ 
   │  FR4   = λR  FR4
 ─ M ──│         ──│ 
   │   │           │ 
 ─ AR──┘         ──┘ 
```
"""
function bigrightenv(AR, M, FR4 = _arraytype(AR)(randn(eltype(AR), size(AR,1), size(M,3), size(M,3), size(AR,1))); kwargs...)
    λFR4s, FR4s, info = eigsolve(FR4 -> ein"(((fghi,def),ckge),bjhk),aji -> dcba"(FR4,AR,M,M,conj(AR)), FR4, 1, :LM; ishermitian = false, kwargs...)
    # @show λFR4s
    return real(λFR4s[1]), real(FR4s[1])
end


"""
Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
    of `FL - M - FR` contracted along the physical dimension.
````
┌── AC──┐         
│   │   │           ┌── AC──┐ 
FL─ M ──FR  =  λAC  │   │   │ 
│   │   │         
        
````
"""
function ACenv(AC, FL, M, FR;kwargs...)
    λs, ACs, _ = eigsolve(AC -> ein"((αaγ,γpη),asbp),ηbβ -> αsβ"(FL,AC,M,FR), AC, 1, :LM; ishermitian = false, kwargs...)
    # if length(λs) > 1 && norm(real(λs[1]) - real(λs[2])) < 1e-12
    #     @show λs
    #     if real(λs[1]) > 0
    #         return real(λs[1]), real(ACs[1])
    #     else
    #         return real(λs[2]), real(ACs[2])
    #     end
    # end
    # println("ACenv $(λs)") 
    return real(λs[1]), real(ACs[1])
end

"""
Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
````
┌──C──┐         
│     │          ┌──C──┐ 
FL─── FR  =  λC  │     │ 
│     │         
        
````
"""
function Cenv(C, FL, FR;kwargs...)
    λs, Cs, _ = eigsolve(C -> ein"(αaγ,γη),ηaβ -> αβ"(FL,C,FR), C, 1, :LM; ishermitian = false, kwargs...)
    # if length(λs) > 1 && norm(real(λs[1]) - real(λs[2])) < 1e-12
    #     @show λs
    #     if real(λs[1]) > 0
    #         return real(λs[1]), real(Cs[1])
    #     else
    #         return real(λs[2]), real(Cs[2])
    #     end
    # end
    return real(λs[1]), real(Cs[1])
end

"""
    AL, AR, errL, errR = ACCtoALAR(AC, C) 

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──AL──C──  =  ──AC──  = ──C──AR──
  │             │            │   
````
"""
function ACCtoALAR(AC, C)
    D, d, = size(AC)

    QAC, RAC = qrpos(reshape(AC,(D*d, D)))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', (D, d, D))
    errL = norm(RAC-RC)
    # @show errL

    LAC, QAC = lqpos(reshape(AC,(D, d*D)))
    LC, QC = lqpos(C)
    AR = reshape(QC'*QAC, (D, d, D))
    errR = norm(LAC-LC)
    # @show errR
    return AL, AR, errL, errR
end

"""
    err = error(AL,C,FL,M,FR)

Compute the error through all environment `AL,C,FL,M,FR`

````
        ┌── AC──┐         
        │   │   │           ┌── AC──┐ 
MAC1 =  FL─ M ──FR  =  λAC  │   │   │ 
        │   │   │         

        ┌── AC──┐         
        │   │   │           ┌──C──┐ 
MAC2 =  FL─ M ──FR  =  λAC  │     │ 
        │   │   │         
        ┕── AL─     
        
── MAC1 ──    ≈    ── AL ── MAC2 ── 
    │                 │
````
"""
function error(AL,C,FL,M,FR)
    AC = ein"asc,cb -> asb"(AL,C)
    MAC = ein"((αaγ,γpη),asbp),ηbβ -> αsβ"(FL,AC,M,FR)
    MAC -= ein"asd,(cpd,cpb) -> asb"(AL,conj(AL),MAC)
    norm(MAC)
end