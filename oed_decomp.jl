#!/bin/env julia

# Todo: make more like get-r in that it should accept args and save to disk; far easier to parallelise that way
# (MSessions don't play nicely with threads, apparently)

# Set up environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Serialization
using MATLAB
using TensorDecompositions: CANDECOMP
import JSON
import SparseArrays

#Gs = deserialize("/home/olie/.dropbox-work/OED/adj-mats/synonymy.jls")
Gs = deserialize("/home/olie/.dropbox-work/OED/adj-mats/semantic.jls")

mat"""
addpath decomp/matlab
addpath decomp/matlab/tensor_toolbox
"""
function matlab_load_sptensor(adj_mats)
    sparsefloat(am1) =
        convert(SparseArrays.SparseMatrixCSC{Float64,Int64}, am1)
    adj_mats = map(sparsefloat, adj_mats)
    mat"spt = sparse_matrix_list_to_sptensor($adj_mats)"
end

"Convert dict from matlab to CANDECOMP struct"
function _matlab_dict_to_cp(D, r)
    if r == 1
        lmbda = Array{Float64}(undef, r)
        lmbda .= D["lambda"]
        # Make it a 10x1 array, not a 10x0 array
        factors = map(permutedims ∘ permutedims, D["u"]|>Tuple)
        CANDECOMP(factors, lmbda)
    else
        CANDECOMP(D["u"]|>Tuple, D["lambda"])
    end
end

"""
    ncp(r, method="apg")

'apg' is the 'apg-tf' method from Xu and Yin 2013.

"""
function matlab_ncp_loaded_spt(r, method="apg")
    if method == "apg"
        mat"$D = ncp_apg(spt, $r, {});"
    end
    _matlab_dict_to_cp(D, r)
end

function relerror_loaded(D)
    D = Dict(("lambda" => D.lambdas, "u" => Any[D.factors...]))
    mat"relerror(spt, $D)"
end

matlab_nncp_loaded_spt(r) = matlab_ncp_loaded_spt(r, "apg")

s = size.(Gs)
matlab_load_sptensor([g[1:s[i][1],1:s[i][2]] for (i,g) in enumerate(Gs)])

Gs = nothing

components = parse(Int,get(ARGS,1,"3"))
repeats = parse(Int,get(ARGS,2,"1"))

print("Starting $components components with $repeats repeats")
Ds = [matlab_nncp_loaded_spt(components) for i in 1:repeats]

serialize("/home/olie/.dropbox-work/OED/decomps/SemanticR$components-repeats-$repeats.jls",Ds)
