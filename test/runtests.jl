using T4APartitionedTT: T4APartitionedTT
using Random
using ITensors
using Test

include("_util.jl")

include("projector_tests.jl")
include("subdomaintt_tests.jl")
include("partitionedtt_tests.jl")
include("contract_tests.jl")
include("patching_tests.jl")
include("util_tests.jl")
include("automul_tests.jl")
include("projector_tree_tests.jl")
# include("bak/conversion_tests.jl")  # bak directory was removed
# include("adaptivemul_tests.jl")  # Disabled: depends on TCIAlgorithms and FMPOC
