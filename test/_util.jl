using ITensors
using Random
import T4AITensorCompat: TensorTrain, random_mps

function _random_mpo(
    rng::AbstractRNG, sites::AbstractVector{<:AbstractVector{Index{T}}}; linkdims::Int=1
) where {T}
    sites_ = collect(Iterators.flatten(sites))
    Ψ = random_mps(rng, sites_; linkdims)
    tensors = Vector{ITensor}(undef, length(sites))
    pos = 1
    for i in 1:length(sites)
        tensors[i] = prod(Ψ[pos:(pos + length(sites[i]) - 1)])
        pos += length(sites[i])
    end
    return TensorTrain(tensors)
end

function _random_mpo(
    sites::AbstractVector{<:AbstractVector{Index{T}}}; linkdims::Int=1
) where {T}
    return _random_mpo(Random.default_rng(), sites; linkdims)
end

function _evaluate(Ψ::TensorTrain, sites, index::Vector{Int})
    return only(reduce(*, Ψ[n] * onehot(sites[n] => index[n]) for n in 1:(length(Ψ))))
end
