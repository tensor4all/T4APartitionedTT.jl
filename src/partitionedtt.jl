
"""
PartitionedTT is a structure that holds multiple TensorTrains (SubDomainTT) that are associated with different non-overlapping projectors.
"""
mutable struct PartitionedTT
    data::OrderedDict{Projector,SubDomainTT}
    tree::Union{Nothing,ProjectorTreeNode}
    all_sites::Union{Nothing,Vector{Index}}

    function PartitionedTT(data::AbstractVector{SubDomainTT})
        sites_all = [siteinds(subdtt) for subdtt in data]
        for n in 2:length(data)
            Set(sites_all[n]) == Set(sites_all[1]) || error("Sitedims mismatch")
        end
        isdisjoint([subdtt.projector for subdtt in data]) || error("Projectors are overlapping")

        dict_ = OrderedDict{Projector,SubDomainTT}(
            data[i].projector => data[i] for i in 1:length(data)
        )

        # Build tree and cache all_sites
        tree = nothing
        all_sites = nothing
        if !isempty(dict_)
            all_sites = collect(Iterators.flatten(sites_all[1]))
            tree = build_projector_tree(new(dict_, nothing, nothing), all_sites)
        end

        return new(dict_, tree, all_sites)
    end
end

PartitionedTT(data::SubDomainTT) = PartitionedTT([data])

PartitionedTT() = PartitionedTT(SubDomainTT[])

function Base.show(io::IO, obj::PartitionedTT)
    n = length(obj)
    return print(io, "PartitionedTT with $n SubDomainTT$(n == 1 ? "" : "es")")
end

projectors(obj::PartitionedTT) = collect(keys(obj))

function Base.append!(a::PartitionedTT, b::PartitionedTT)
    if !isdisjoint(vcat(projectors(a), projectors(b)))
        error(
            "Projectors are overlapping or identical. Resum of patches could be necessary."
        )
    end
    for (k, v) in b.data
        a.data[k] = v
    end
    # Rebuild tree after modification
    if !isempty(a.data)
        sites = siteinds(a)
        a.all_sites = collect(Iterators.flatten(sites))
        a.tree = build_projector_tree(a, a.all_sites)
    else
        a.tree = nothing
        a.all_sites = nothing
    end
    return a
end

function Base.append!(a::PartitionedTT, b::AbstractVector{SubDomainTT})
    return append!(a, PartitionedTT(b))
end

"""
Return the site indices of the PartitionedTT.
The site indices are returned as a vector of sets, where each set corresponds to the site indices at each site.
"""
function siteindices(obj::PartitionedTT)
    return [Set(x) for x in ITensors.siteinds(first(values(obj.data)))]
end

siteinds(obj::PartitionedTT) = siteinds(first(values(obj.data)))

ITensors.siteinds(obj::PartitionedTT) = siteindices(obj)

"""
Get the number of the data in the PartitionedTT.
This is NOT the number of sites in the PartitionedTT.
"""
Base.length(obj::PartitionedTT) = length(obj.data)

"""
Indexing for PartitionedTT. This is deprecated and will be removed in the future.
"""
function Base.getindex(parttt::PartitionedTT, i::Integer)::SubDomainTT
    @warn "Indexing for PartitionedTT is deprecated. Use getindex(parttt, p::Projector) instead."
    return first(Iterators.drop(values(parttt.data), i - 1))
end

Base.getindex(obj::PartitionedTT, p::Projector) = obj.data[p]

function Base.iterate(parttt::PartitionedTT, state)
    return iterate(parttt.data, state)
end

function Base.iterate(parttt::PartitionedTT)
    return iterate(parttt.data)
end

"""
Return the keys, i.e., projectors of the PartitionedTT.
"""
function Base.keys(obj::PartitionedTT)
    return keys(obj.data)
end

"""
Return the values, i.e., SubDomainTT of the PartitionedTT.
"""
function Base.values(obj::PartitionedTT)
    return values(obj.data)
end

"""
Rearrange the site indices of the PartitionedTT according to the given order.
If nessecary, tensors are fused or split to match the new order.
"""
function rearrange_siteinds(obj::PartitionedTT, sites)
    return PartitionedTT([rearrange_siteinds(subdtt, sites) for subdtt in values(obj)])
end

function prime(Ψ::PartitionedTT, args...; kwargs...)
    return PartitionedTT([prime(subdtt, args...; kwargs...) for subdtt in values(Ψ.data)])
end

function noprime(Ψ::PartitionedTT, args...; kwargs...)
    return PartitionedTT([noprime(subdtt, args...; kwargs...) for subdtt in values(Ψ.data)])
end

"""
Return the norm of the PartitionedTT.
"""
function LinearAlgebra.norm(M::PartitionedTT)
    return sqrt(reduce(+, (x^2 for x in LinearAlgebra.norm.(values(M)))))
end

"""
Add two PartitionedTT objects.

If the two projects have the same projectors in the same order, the resulting PartitionedTT will have the same projectors in the same order.
By default, we use `directsum` algorithm to compute the sum and no truncation is performed.
"""
function Base.:+(
    parttts::PartitionedTT...;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    coeffs=ones(length(parttts)),
    kwargs...,
)::PartitionedTT
    result = PartitionedTT()
    for (coeff, parttt) in zip(coeffs, parttts)
        result = +(result, coeff * parttt; alg, cutoff, maxdim, kwargs...)
    end
    return result
end

function Base.:+(
    a::PartitionedTT,
    b::PartitionedTT;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    coeffs=(1.0, 1.0),
    kwargs...,
)::PartitionedTT
    result = PartitionedTT()
    return add!(result, a, b; alg, cutoff, maxdim, coeffs, kwargs...)
end

function add!(
    result::PartitionedTT,
    a::PartitionedTT,
    b::PartitionedTT;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    overwrite=true,
    coeffs=(1.0, 1.0),
    kwargs...,
)::PartitionedTT
    length(coeffs) == 2 || error("coeffs must be a tuple of length 2")
    data = SubDomainTT[]
    for k in unique(vcat(collect(keys(a)), collect(keys(b)))) # preserve order
        if k ∈ keys(result) && !overwrite
            continue
        end
        if k ∈ keys(a) && k ∈ keys(b)
            a[k].projector == b[k].projector || error("Projectors mismatch at $(k)")
            push!(
                data, +(coeffs[1] * a[k], coeffs[2] * b[k]; alg, cutoff, maxdim, kwargs...)
            )
        elseif k ∈ keys(a)
            push!(data, coeffs[1] * a[k])
        elseif k ∈ keys(b)
            push!(data, coeffs[2] * b[k])
        else
            error("Something went wrong")
        end
    end
    return append!(result, data)
end

function Base.:*(a::PartitionedTT, b::Number)::PartitionedTT
    return PartitionedTT([a[k] * b for k in keys(a)])
end

function Base.:*(a::Number, b::PartitionedTT)::PartitionedTT
    return b * a
end

function Base.:-(obj::PartitionedTT)::PartitionedTT
    return -1 * obj
end

"""
Truncate a PartitionedTT object piecewise.

Each SubDomainTT in the PartitionedTT is truncated independently,
but the cutoff is adjusted according to the norm of each SubDomainTT.
The total error is the sum of the errors in each SubDomainTT.
"""
function truncate(
    obj::PartitionedTT;
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    use_adaptive_weight=true,
    maxrefinement=4,
    kwargs...,
)::PartitionedTT
    norm2 = [LinearAlgebra.norm(v)^2 for v in values(obj)]
    total_norm2 = sum(norm2)
    weights = [total_norm2 / norm2_v for norm2_v in norm2] # Initial weights (FIXME: better choice?)

    compressed = obj

    for _ in 1:maxrefinement
        compressed = PartitionedTT([
            truncate(v; cutoff=cutoff * w, maxdim, kwargs...) for
            (v, w) in zip(values(obj), weights)
        ])
        actual_error = dist(obj, compressed)^2 / sum(norm2)
        if actual_error < cutoff || !use_adaptive_weight
            break
        end

        weights .*= min(cutoff / actual_error, 0.5) # Adjust weights
    end

    return compressed
end

# Only for debug
# Convert PartitionedTT to MPS/MPO (TensorTrain)
# Since MPS and MPO are both TensorTrain (type aliases), we use a single helper function
function _to_tensortrain(
    obj::PartitionedTT; cutoff=default_cutoff(), maxdim=default_maxdim()
)
    return reduce(
        (x, y) -> T4AITensorCompat.truncate(+(x, y; alg="directsum"); cutoff, maxdim),
        values(obj.data),
    ).data # direct sum
end

# TensorTrain conversion - returns TensorTrain
function TensorTrain(
    obj::PartitionedTT; cutoff=default_cutoff(), maxdim=default_maxdim()
)::TensorTrain
    return _to_tensortrain(obj; cutoff=cutoff, maxdim=maxdim)
end

# MPS conversion - returns TensorTrain
function MPS(
    obj::PartitionedTT; cutoff=default_cutoff(), maxdim=default_maxdim()
)::TensorTrain
    return _to_tensortrain(obj; cutoff=cutoff, maxdim=maxdim)
end

# MPO conversion - returns TensorTrain
function MPO(
    obj::PartitionedTT; cutoff=default_cutoff(), maxdim=default_maxdim()
)::TensorTrain
    # Convert to TensorTrain (same as MPS since both are TensorTrain)
    return _to_tensortrain(obj; cutoff=cutoff, maxdim=maxdim)
end

"""
Make the PartitionedTT diagonal for a given site index `s` by introducing a dummy index `s'`.
"""
function makesitediagonal(
    obj::PartitionedTT, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    return PartitionedTT([
        makesitediagonal(subdtt, sites; baseplev=baseplev) for subdtt in values(obj)
    ])
end

function makesitediagonal(obj::PartitionedTT, site::Index{IndsT}; baseplev=0) where {IndsT}
    return PartitionedTT([
        makesitediagonal(subdtt, site; baseplev=baseplev) for subdtt in values(obj)
    ])
end

"""
Extract diagonal of the PartitionedTT for `s`, `s'`, ... for a given site index `s`,
where `s` must have a prime level of 0.
"""
function extractdiagonal(obj::PartitionedTT, sites)
    return PartitionedTT([extractdiagonal(subdtt, sites) for subdtt in values(obj)])
end

function dist(a::PartitionedTT, b::PartitionedTT)
    return sqrt(sum(dist(TensorTrain(a[k]), TensorTrain(b[k]))^2 for k in keys(a)))
end

function _findallsiteinds_by_tag(parttt::PartitionedTT; tag=tag)
    return findallsiteinds_by_tag(only.(ITensors.siteinds(parttt)); tag=tag)
end

"""
    (parttt::PartitionedTT)(multiindex::Vector{Int})

Evaluate the PartitionedTT at the given MultiIndex.

This function finds the corresponding SubDomainTT using a tree-based search,
then evaluates it at the given MultiIndex.

# Arguments
- `multiindex::Vector{Int}`: The MultiIndex (vector of site index values) to evaluate at

# Returns
- The evaluation value of the corresponding SubDomainTT at the given MultiIndex

# Throws
- `ArgumentError` if no matching SubDomainTT is found for the given MultiIndex
"""
function (parttt::PartitionedTT)(multiindex::Vector{Int})
    # Use cached all_sites and tree
    if parttt.all_sites === nothing || parttt.tree === nothing
        throw(ArgumentError("PartitionedTT is empty"))
    end

    all_sites = parttt.all_sites
    tree = parttt.tree

    # Check that multiindex has the correct length
    if length(multiindex) != length(all_sites)
        throw(
            ArgumentError(
                "MultiIndex length $(length(multiindex)) does not match number of site indices $(length(all_sites))",
            ),
        )
    end

    # Create a Projector from the MultiIndex
    sites = siteinds(parttt)
    projector_dict = Dict{Index,Int}()
    idx_pos = 1
    for site_group in sites
        for site in site_group
            projector_dict[site] = multiindex[idx_pos]
            idx_pos += 1
        end
    end
    target_projector = Projector(projector_dict)

    # Search for matching SubDomainTT using cached tree
    result = find_in_tree(tree, target_projector, all_sites)

    if result === nothing
        # No matching SubDomainTT found
        throw(ArgumentError("No matching SubDomainTT found for the given MultiIndex"))
    end

    # Evaluate the SubDomainTT using function call syntax
    return result(multiindex, all_sites)
end
