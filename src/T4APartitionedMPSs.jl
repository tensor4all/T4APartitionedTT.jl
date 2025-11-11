module T4APartitionedMPSs

import OrderedCollections: OrderedSet, OrderedDict
using EllipsisNotation
using LinearAlgebra: LinearAlgebra

import ITensors: ITensors, Index, ITensor, dim, inds, qr, commoninds, uniqueinds, hasplev
using T4AITensorCompat: TensorTrain
import T4AITensorCompat: siteinds, findsites, findsite, maxlinkdim, dist, isortho, orthocenter, contract, truncate, fit
import T4AITensorCompat: T4AITensorCompat
import ITensors.TagSets: hastag, hastags
import ITensors: Algorithm, @Algorithm_str

using Distributed
using Base.Threads

default_cutoff() = 1e-25
default_maxdim() = typemax(Int)

include("util.jl")
include("projector.jl")
include("subdomainmps.jl")
include("partitionedmps.jl")
include("patching.jl")
include("contract.jl")
include("adaptivemul.jl")
include("automul.jl")

end
