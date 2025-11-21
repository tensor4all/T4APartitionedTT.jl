using T4APartitionedTT
using T4AITensorCompat
using ITensors
using Random

Random.seed!(1234)

# Create a simple tensor train
N = 3
sitesx = [Index(2, "x=$n") for n in 1:N]
sitesy = [Index(2, "y=$n") for n in 1:N]
sites = collect(collect.(zip(sitesx, sitesy)))

# Create a random tensor train
function _random_mpo(
    sites::AbstractVector{<:AbstractVector{Index{T}}}; linkdims::Int=1
) where {T}
    sites_ = collect(Iterators.flatten(sites))
    Ψ = T4AITensorCompat.random_mps(Random.default_rng(), sites_; linkdims)
    tensors = Vector{ITensor}(undef, length(sites))
    pos = 1
    for i in 1:length(sites)
        tensors[i] = prod(Ψ[pos:(pos + length(sites[i]) - 1)])
        pos += length(sites[i])
    end
    return T4AITensorCompat.TensorTrain(tensors)
end

mpo = _random_mpo(sites; linkdims=5)
subdtt = T4APartitionedTT.SubDomainTT(mpo)

# This should trigger the error
println("Calling truncate with abs_cutoff...")
try
    result = T4APartitionedTT.truncate(subdtt; cutoff=1e-15, maxdim=10, abs_cutoff=1e-12)
    println("Success!")
catch e
    println("Error occurred:")
    showerror(stdout, e, catch_backtrace())
end
