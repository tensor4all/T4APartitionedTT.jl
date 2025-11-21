module T4APartitionedTT

import OrderedCollections: OrderedSet, OrderedDict
using EllipsisNotation
using LinearAlgebra: LinearAlgebra

import ITensors: ITensors, Index, ITensor, dim, inds, qr, commoninds, uniqueinds, hasplev
import T4AITensorCompat: TensorTrain
import T4AITensorCompat:
    siteinds,
    findsites,
    findsite,
    maxlinkdim,
    dist,
    isortho,
    orthocenter,
    contract,
    truncate,
    fit,
    default_cutoff,
    default_maxdim
import T4AITensorCompat: T4AITensorCompat
import ITensors.TagSets: hastag, hastags
import ITensors: Algorithm, @Algorithm_str

using Distributed
using Base.Threads

include("util.jl")
include("projector.jl")
include("subdomaintt.jl")
include("projector_tree_types.jl")
include("partitionedtt.jl")
include("patching.jl")
include("contract.jl")
include("adaptivemul.jl")
include("automul.jl")
include("projector_tree.jl")

# Re-export default parameters from T4AITensorCompat
export default_cutoff, default_maxdim

# Define default_abs_cutoff if not available from T4AITensorCompat
if isdefined(T4AITensorCompat, :default_abs_cutoff)
    const default_abs_cutoff = T4AITensorCompat.default_abs_cutoff
    export default_abs_cutoff
else
    default_abs_cutoff() = 0.0
    export default_abs_cutoff
end

end
