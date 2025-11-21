# just for backward compatibility...
_alg_map = Dict(
    ITensors.Algorithm(alg) => alg for alg in ["directsum", "densitymatrix", "fit", "naive"]
)
""" 
Contraction of two SubDomainTTs. 
Only if the shared projected indices overlap the contraction is non-vanishing.
"""
function contract(
    M1::SubDomainTT, M2::SubDomainTT; alg, kwargs...
)::Union{SubDomainTT,Nothing}
    # If the SubDomainTT don't overlap they cannot be contracted.
    if !hasoverlap(M1.projector, M2.projector)
        return nothing
    end
    proj, _ = _projector_after_contract(M1, M2)

    alg_str::String = alg isa String ? alg : _alg_map[alg]
    Ψ = contract(M1.data, M2.data; alg=Algorithm(alg_str), kwargs...)
    return project(SubDomainTT(Ψ), proj)
end

# Figure out `projector` after contracting SubDomainTT objects
function _projector_after_contract(M1::SubDomainTT, M2::SubDomainTT)
    sites1 = _allsites(M1)
    sites2 = _allsites(M2)

    external_sites = setdiff(union(sites1, sites2), intersect(sites1, sites2))
    # If the SubDomainTT don't overlap they cannot be contracted -> no final projector
    if !hasoverlap(M1.projector, M2.projector)
        return nothing, external_sites
    end

    proj = deepcopy(M1.projector.data)
    empty!(proj)

    for s in external_sites
        if isprojectedat(M1, s)
            proj[s] = M1.projector[s]
        end
        if isprojectedat(M2, s)
            proj[s] = M2.projector[s]
        end
    end

    return Projector(proj), external_sites
end

# Check for newly projected sites to be only external sites.
function _is_externalsites_compatible_with_projector(external_sites, projector)
    for s in keys(projector)
        if !(s ∈ external_sites)
            return false
        end
    end
    return true
end

"""
Project two SubDomainTT objects to `proj` before contracting them.
"""
function projcontract(
    M1::SubDomainTT,
    M2::SubDomainTT,
    proj::Projector;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    kwargs...,
)::Union{Nothing,SubDomainTT}
    # Project M1 and M2 to `proj` before contracting
    M1 = project(M1, proj)
    M2 = project(M2, proj)
    if M1 === nothing || M2 === nothing
        return nothing
    end

    _, external_sites = _projector_after_contract(M1, M2)

    if !_is_externalsites_compatible_with_projector(external_sites, proj)
        error("The projector contains projection onto a site that is not an external site.")
    end

    r = contract(M1, M2; alg, cutoff, maxdim, kwargs...)
    return r
end

"""
Project SubDomainTT vectors to `proj` before computing all possible pairwise contractions of the elements.
The results are summed or patch-summed if belonging to the same patch.
"""
function projcontract(
    M1::AbstractVector{SubDomainTT},
    M2::AbstractVector{SubDomainTT},
    proj::Projector;
    alg="zipup",
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    kwargs...,
)::Union{Nothing,Vector{SubDomainTT}}
    results = SubDomainTT[]

    for m1 in M1, m2 in M2
        r = projcontract(m1, m2, proj; alg, cutoff, maxdim, kwargs...)
        if r !== nothing
            push!(results, r)
        end
    end

    if isempty(results)
        return nothing
    end

    if length(results) == 1
        return results
    end

    res = if length(patchorder) > 0
        _add_patching(results; cutoff, maxdim, patchorder, kwargs...)
    else
        [_add(results...; alg=alg_sum, cutoff, maxdim, kwargs...)]
    end

    return res
end

# Function to add a new patch to the result patched contraction. Only if the patch is non-overlapping with any 
# of the already present ones it is added, otherwise it is fused. 
function add_result_patch!(
    dict::Dict{Projector,Vector{Tuple{SubDomainTT,SubDomainTT}}}, proj::Projector
)
    # Iterate over a copy of keys (patches) to avoid modifications while looping.
    for existing_proj in collect(keys(dict))
        if hasoverlap(existing_proj, proj)
            fused_proj = existing_proj | proj
            # Save the subdtts of the overlapping patch.
            subdtts = dict[existing_proj]
            # Remove the old projector (this deletes also its associated subdtts).
            delete!(dict, existing_proj)
            # Recursively update with the fused projector.
            new_proj = add_result_patch!(dict, fused_proj)
            # If new_proj is already present, merge the subdtts; otherwise, insert the saved subdtts.
            if haskey(dict, new_proj)
                append!(dict[new_proj], subdtts)
            else
                dict[new_proj] = subdtts
            end
            return new_proj
        end
    end
    # If no overlapping proj is found, then ensure proj is in the dictionary (sanity passage). 
    if !haskey(dict, proj)
        dict[proj] = Vector{Tuple{SubDomainTT,SubDomainTT}}()
    end
    return proj
end

# Preprocessing of the patches to obtain all the contraction tasks from two T4APartitionedTTs
function _contraction_tasks(
    M1::PartitionedTT,
    M2::PartitionedTT;
    M::PartitionedTT=PartitionedTT(),
    overwrite=true,
)::Vector{Tuple{Projector,SubDomainTT,SubDomainTT}}
    final_patches = Dict{Projector,Vector{Tuple{SubDomainTT,SubDomainTT}}}()
    # Add a new patch only if the two subdtt are compatible (overlapping internal projected
    # sites) and the new patch is non-overlapping with all the existing ones.
    for m1 in values(M1), m2 in values(M2)
        tmp_prj = _projector_after_contract(m1, m2)[1]
        if tmp_prj !== nothing
            patch = add_result_patch!(final_patches, tmp_prj)
            if haskey(final_patches, patch)
                push!(final_patches[patch], (m1, m2))
            else
                final_patches[patch] = (m1, m2)
            end
        end
    end

    # Sanity check
    for p1 in keys(final_patches), p2 in keys(final_patches)
        if p1 != p2 && hasoverlap(p1, p2)
            error("After contraction, projectors must not overlap.")
        end
    end

    # Flatten the result to create contraction tasks
    tasks = Vector{Tuple{Projector,SubDomainTT,SubDomainTT}}()
    for (proj, subdtt_pairs) in final_patches
        if haskey(M.data, proj) && !overwrite
            continue
        end
        for (subdtt1, subdtt2) in subdtt_pairs
            push!(tasks, (proj, project(subdtt1, proj), project(subdtt2, proj)))
        end
    end

    return tasks
end

"""
Contract two T4APartitionedTTs MPS objects.

At each site, the objects must share at least one site index.
"""
function contract(
    M1::PartitionedTT,
    M2::PartitionedTT;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    parallel::Symbol=:serial,
    kwargs...,
)::Union{PartitionedTT}
    M = PartitionedTT()
    return contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, parallel, kwargs...)
end

"""
Contract two PartitionedTT objects.

Existing patches `M` in the resulting PartitionedTT will be overwritten if `overwrite=true`.
"""
function contract!(
    M::PartitionedTT,
    M1::PartitionedTT,
    M2::PartitionedTT;
    alg="zipup",
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    parallel::Symbol=:serial,
    overwrite=true,
    kwargs...,
)::PartitionedTT
    # Builds contraction tasks 
    tasks = _contraction_tasks(M1, M2; M=M, overwrite=overwrite)

    # Helper contraction function
    function contract_task(task; alg, cutoff, maxdim, kwargs...)
        proj, M1_subs, M2_subs = task
        return projcontract(M1_subs, M2_subs, proj; alg, cutoff, maxdim, kwargs...)
    end

    # Serial or distributed contraction
    if parallel == :serial
        contr_results = map(
            task -> contract_task(task; alg, cutoff, maxdim, kwargs...), tasks
        )
    elseif parallel == :distributed
        contr_results = pmap(
            task -> contract_task(task; alg, cutoff, maxdim, kwargs...), tasks
        )
    else
        error("Symbol $(parallel) not recongnized.")
    end

    # Sanity check
    all(r -> r !== nothing, contr_results) ||
        error("Some contraction returned `nothing`. Faulty preprocessing of patches...")

    ## Resum SubDomainTTs projected on the same final patch
    # Group together patches to resum 
    patch_group = Dict{Projector,Vector{SubDomainTT}}()
    for subdtt in contr_results
        if haskey(patch_group, subdtt.projector)
            push!(patch_group[subdtt.projector], subdtt)
        else
            patch_group[subdtt.projector] = [subdtt]
        end
    end

    # Helper sum function
    function sum_task(group; patchorder, alg_sum, cutoff, maxdim, kwargs...)
        if length(group) == 1
            return [group[1]]
        else
            res = if length(patchorder) > 0
                _add_patching(group; cutoff, maxdim, patchorder, kwargs...)
            else
                [_add(group...; alg=alg_sum, cutoff, maxdim, kwargs...)]
            end
            return res
        end
    end

    if parallel == :serial
        summed_patches = map(
            group -> sum_task(
                group;
                patchorder=patchorder,
                alg_sum=alg_sum,
                cutoff=cutoff,
                maxdim=maxdim,
                kwargs...,
            ),
            collect(values(patch_group)),
        )
    elseif parallel == :distributed
        summed_patches = pmap(
            group -> sum_task(
                group;
                patchorder=patchorder,
                alg_sum=alg_sum,
                cutoff=cutoff,
                maxdim=maxdim,
                kwargs...,
            ),
            collect(values(patch_group)),
        )
    end

    # Assembling the PartitionedTT
    for subdtts in summed_patches
        if subdtts !== nothing
            append!(M, vcat(subdtts))
        end
    end

    return M
end
