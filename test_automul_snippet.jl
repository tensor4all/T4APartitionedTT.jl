using T4APartitionedTT
using T4AITensorCompat
using ITensors
using Random
using LinearAlgebra

include("test/_util.jl")

Random.seed!(1234)

# Setup from the test (lines 22-30)
N = 5
L = 10 # Bond dimension
d = 2 # Local dimension 

sites = [Index(d, "Qubit, n=$n") for n in 1:N]
sites_vec = [[x] for x in sites]

Ψ = _random_mpo(sites_vec; linkdims=L)
dummy_subdtt = T4APartitionedTT.SubDomainTT(Ψ)

proj_lev_l = 2 # Max projected index left tensor 
proj_lev_r = 3 # Max projected index right tensor

proj_l = vec([
    Dict(zip(sites, combo)) for combo in Iterators.product((1:d for _ in 1:proj_lev_l)...)
])

proj_r = vec([
    Dict(zip(sites, combo)) for combo in Iterators.product((1:d for _ in 1:proj_lev_r)...)
])

partΨ_l = T4APartitionedTT.PartitionedTT(T4APartitionedTT.project.(Ref(Ψ), proj_l))
partΨ_r = T4APartitionedTT.PartitionedTT(T4APartitionedTT.project.(Ref(Ψ), proj_r))

diag_dummy_l = T4APartitionedTT.makesitediagonal(dummy_subdtt, sites; baseplev=1)
diag_dummy_r = T4APartitionedTT.makesitediagonal(dummy_subdtt, sites; baseplev=0)

elemmul_dummy = T4APartitionedTT.extractdiagonal(
    T4APartitionedTT.contract(diag_dummy_l, diag_dummy_r; alg="zipup"), sites
)

element_prod = T4APartitionedTT.elemmul(partΨ_l, partΨ_r)
mps_element_prod = T4AITensorCompat.MPS(element_prod)

test_points = [[rand(1:d) for __ in 1:N] for _ in 1:1000]

result = [_evaluate(mps_element_prod, sites, p) for p in test_points]
reference = [_evaluate(Ψ, sites, p)^2 for p in test_points]

# The specific lines from the test (66-69)
max_diff = maximum(abs.(result - reference))
max_ref = maximum(abs.(reference))
atol_val = sqrt(T4APartitionedTT.default_cutoff())

@show max_diff
@show max_ref
@show atol_val
println("max_diff < atol_val: ", max_diff < atol_val)

# Check which elements fail
diffs = abs.(result - reference)
failed_indices = findall(x -> x > atol_val, diffs)
if !isempty(failed_indices)
    println("Failed indices: ", failed_indices[1:min(10, length(failed_indices))])
    println("Failed diffs: ", diffs[failed_indices[1:min(10, length(failed_indices))]])
    println(
        "Failed result values: ", result[failed_indices[1:min(10, length(failed_indices))]]
    )
    println(
        "Failed reference values: ",
        reference[failed_indices[1:min(10, length(failed_indices))]],
    )
else
    println("All differences are within atol")
end

# Check array properties
println("\nArray properties:")
println("length(result) = ", length(result))
println("length(reference) = ", length(reference))
println("any(isnan, result) = ", any(isnan, result))
println("any(isnan, reference) = ", any(isnan, reference))
println("any(isinf, result) = ", any(isinf, result))
println("any(isinf, reference) = ", any(isinf, reference))

# Check with rtol as well (isapprox uses both atol and rtol by default)
rtol_default = 1e-9  # Default rtol for isapprox
println("\nChecking with rtol: ", rtol_default)
println("max_diff / max_ref = ", max_diff / max_ref)
println("max_diff / max_ref < rtol: ", max_diff / max_ref < rtol_default)

# Try element-wise comparison
println("\nElement-wise comparison:")
all_close = all(isapprox.(result, reference; atol=atol_val, rtol=0.0))
println("all(isapprox.(result, reference; atol=atol_val, rtol=0.0)) = ", all_close)

# Check norm-based comparison (isapprox for arrays uses norm)
using LinearAlgebra
diff_norm = norm(result - reference)
ref_norm = norm(reference)
println("\nNorm-based comparison:")
println("norm(result - reference) = ", diff_norm)
println("norm(reference) = ", ref_norm)
println("diff_norm <= atol_val: ", diff_norm <= atol_val)
println("diff_norm <= rtol_default * ref_norm: ", diff_norm <= rtol_default * ref_norm)

println(
    "\nTest result with atol only: ", isapprox(result, reference; atol=atol_val, rtol=0.0)
)
println("Test result with default rtol: ", isapprox(result, reference; atol=atol_val))
