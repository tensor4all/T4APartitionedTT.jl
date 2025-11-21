"""
Add multiple SubDomainTT objects on the same projector.

If the bond dimension of the result reaches `maxdim`,
perform patching recursively to reduce the bond dimension.
"""
function _add_patching(
    subdtts::AbstractVector{SubDomainTT};
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    alg="fit",
    patchorder=Index[],
)::Vector{SubDomainTT}
    if length(unique([sudtt.projector for sudtt in subdtts])) != 1
        error("All SubDomainTT objects must have the same projector.")
    end

    # First perform addition upto given maxdim
    # TODO: Early termination if the bond dimension reaches maxdim
    sum_approx = _add(subdtts...; alg, cutoff, maxdim)

    # If the bond dimension is less than maxdim, return the result
    maxbonddim(sum_approx) < maxdim && return [sum_approx]

    # @assert maxbonddim(sum_approx) == maxdim

    nextprjidx = _next_projindex(subdtts[1].projector, patchorder)

    nextprjidx === nothing && return [sum_approx]

    blocks = SubDomainTT[]
    for prjval in 1:ITensors.dim(nextprjidx)
        prj_ = subdtts[1].projector & Projector(nextprjidx => prjval)
        blocks =
            blocks ∪ _add_patching(
                [project(sudtt, prj_) for sudtt in subdtts]; cutoff, maxdim, alg, patchorder
            )
    end

    return blocks
end

"""
Return the next index to be projected.
"""
function _next_projindex(prj::Projector, patchorder)::Union{Nothing,Index}
    idx = findfirst(x -> !isprojectedat(prj, x), patchorder)
    if idx === nothing
        return nothing
    else
        return patchorder[idx]
    end
end

"""
Add multiple PartitionedTT objects.
"""
function add_patching(
    parttt::AbstractVector{PartitionedTT};
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    alg=Algorithm"fit"(),
    patchorder=Index[],
)::PartitionedTT
    result = _add_patching(
        union(values(x) for x in parttt); cutoff, maxdim, alg, patchorder
    )
    return PartitionedTT(result)
end

"""
Adaptive patching

Do patching recursively to reduce the bond dimension.
If the bond dimension of a SubDomainTT exceeds `maxdim`, perform patching.
"""
function _patch(
    subdtt::SubDomainTT,
    patchorder;
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    abs_cutoff=default_abs_cutoff(),
)::Vector{SubDomainTT}
    nextprjidx = _next_projindex(subdtt.projector, patchorder)
    if nextprjidx === nothing
        return [subdtt]
    end

    children = SubDomainTT[]
    for prjval in 1:ITensors.dim(nextprjidx)
        prj_ = subdtt.projector & Projector(nextprjidx => prjval)
        subdtt_ = truncate(project(subdtt, prj_); cutoff, maxdim, abs_cutoff)
        push!(children, subdtt_)
    end
    return children
end

"""
Adaptive patching

Do patching recursively to reduce the bond dimension.
If the bond dimension of a SubDomainTT exceeds `maxdim`, perform patching.

# Truncation Scheme

Each patch is truncated using `abs_cutoff = cutoff * total_norm2`, where `total_norm2` is the sum of squared norms of all patches in the PartitionedTT.
This means each patch uses the same absolute cutoff value based on the total norm of the entire PartitionedTT.

**Note**: This truncation scheme can lead to the total relative error exceeding `cutoff` when there are many patches, 
since the errors from individual patches accumulate. The total error is approximately bounded by `cutoff * number_of_patches` 
in the worst case, rather than `cutoff`.

# Arguments
- `prjtts::PartitionedTT`: The partitioned MPS to perform adaptive patching on
- `patchorder::AbstractVector{<:Index}`: The order of indices to use for patching

# Keyword Arguments
- `cutoff`: Relative cutoff threshold (default: `default_cutoff()`)
- `maxdim`: Maximum bond dimension (default: `default_maxdim()`)

# Returns
- `PartitionedTT`: A new PartitionedTT with patches that have bond dimensions ≤ `maxdim`
"""
function adaptive_patching(
    prjtts::PartitionedTT,
    patchorder::AbstractVector{<:Index};
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
)::PartitionedTT
    #ptt = collect(values(prjtts.data))

    norm2 = [LinearAlgebra.norm(subdtt)^2 for subdtt in values(prjtts.data)]
    total_norm2 = sum(norm2)
    abs_cutoff = cutoff * total_norm2

    data = OrderedDict{Projector,SubDomainTT}(key => subdtt for (key, subdtt) in prjtts)

    conv_data = OrderedDict{Projector,SubDomainTT}()
    for i in 1:100
        new_data = OrderedDict{Projector,SubDomainTT}()
        updated = false
        for (key, subdtt) in data
            if maxbonddim(subdtt) < maxdim
                conv_data[key] = subdtt
                continue
            end
            updated = true
            children::Vector{SubDomainTT} = _patch(
                subdtt, patchorder; cutoff=0.0, abs_cutoff, maxdim=typemax(Int)
            )
            for c in children
                p = deepcopy(c.projector)
                new_data[p] = c
            end
        end
        if !updated
            break
        end
        data = new_data
    end
    return PartitionedTT(collect(values(conv_data)))
end
