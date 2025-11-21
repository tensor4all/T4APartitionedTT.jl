"""
    count_site_projections(parttt::PartitionedTT, all_sites::Vector{Index})::Dict{Index, Int}

Count how many patches project at each site.

# Arguments
- `parttt::PartitionedTT`: The PartitionedTT to analyze
- `all_sites::Vector{Index}`: All site indices in order

# Returns
- `Dict{Index, Int}`: Map from site index to count of patches that project at that site
"""
function count_site_projections(parttt::PartitionedTT, all_sites::AbstractVector{<:Index})::Dict{Index, Int}
    counts = Dict{Index, Int}()
    
    # Initialize all sites to 0
    for site in all_sites
        counts[site] = 0
    end
    
    # Count projections for each patch
    for (proj, subdtt) in parttt.data
        for site in keys(proj)
            if site in all_sites
                counts[site] = get(counts, site, 0) + 1
            end
        end
    end
    
    return counts
end

"""
    build_projector_tree(parttt::PartitionedTT, all_sites::Vector{Index})::ProjectorTreeNode

Build a tree structure for efficient lookup of SubDomainTT by projector.

The tree is built by selecting sites with the highest projection frequency at each level,
creating a decision tree that minimizes the average search depth.

# Arguments
- `parttt::PartitionedTT`: The PartitionedTT to build a tree for
- `all_sites::Vector{Index}`: All site indices in order

# Returns
- `ProjectorTreeNode`: Root of the projector tree
"""
function build_projector_tree(parttt::PartitionedTT, all_sites::AbstractVector{<:Index})::ProjectorTreeNode
    # Count projection frequency for each site
    site_counts = count_site_projections(parttt, all_sites)
    
    # Sort sites by projection frequency (descending)
    # Sites with higher frequency are used for branching earlier
    sorted_sites = sort(collect(all_sites), by=site -> get(site_counts, site, 0), rev=true)
    
    # Build tree recursively
    return _build_tree_recursive(parttt, sorted_sites, Dict{Projector, SubDomainTT}(parttt.data))
end

"""
    _build_tree_recursive(
        parttt::PartitionedTT,
        remaining_sites::Vector{Index},
        remaining_patches::Dict{Projector, SubDomainTT}
    )::ProjectorTreeNode

Recursively build the projector tree.

# Arguments
- `parttt::PartitionedTT`: The PartitionedTT (used for reference)
- `remaining_sites::Vector{Index}`: Sites not yet used for branching
- `remaining_patches::Dict{Projector, SubDomainTT}`: Patches not yet assigned to leaf nodes

# Returns
- `ProjectorTreeNode`: A tree node (internal or leaf)
"""
function _build_tree_recursive(
    parttt::PartitionedTT,
    remaining_sites::AbstractVector{<:Index},
    remaining_patches::Dict{Projector, SubDomainTT}
)::ProjectorTreeNode
    
    # Base case: if only one patch remains, create a leaf node
    if length(remaining_patches) == 1
        (proj, subdtt) = first(remaining_patches)
        node = ProjectorTreeNode(nothing)
        node.leaf = (proj, subdtt)
        return node
    end
    
    # Base case: if no sites remain, create a leaf node with the first matching patch
    # (This should not happen for non-overlapping projectors, but handle it gracefully)
    if isempty(remaining_sites)
        (proj, subdtt) = first(remaining_patches)
        node = ProjectorTreeNode(nothing)
        node.leaf = (proj, subdtt)
        return node
    end
    
    # Find the best site to branch on
    # Select the first site in remaining_sites that has at least one projection
    best_site = nothing
    for site in remaining_sites
        # Check if any remaining patch projects at this site
        has_projection = any(site in keys(proj) for (proj, _) in remaining_patches)
        if has_projection
            best_site = site
            break
        end
    end
    
    # If no site has projections, create a leaf node
    if best_site === nothing
        (proj, subdtt) = first(remaining_patches)
        node = ProjectorTreeNode(nothing)
        node.leaf = (proj, subdtt)
        return node
    end
    
    # Create an internal node branching on best_site
    node = ProjectorTreeNode(best_site)
    
    # Group remaining patches by their projection value at best_site
    grouped_patches = Dict{Union{Int, Nothing}, Dict{Projector, SubDomainTT}}()
    
    for (proj, subdtt) in remaining_patches
        proj_value = haskey(proj, best_site) ? proj[best_site] : nothing
        if !haskey(grouped_patches, proj_value)
            grouped_patches[proj_value] = Dict{Projector, SubDomainTT}()
        end
        grouped_patches[proj_value][proj] = subdtt
    end
    
    # Recursively build children for each group
    remaining_sites_next = filter(s -> s != best_site, remaining_sites)
    
    for (proj_value, patches_group) in grouped_patches
        if length(patches_group) == 1
            # Single patch in this group - create leaf
            (proj, subdtt) = first(patches_group)
            child_node = ProjectorTreeNode(nothing)
            child_node.leaf = (proj, subdtt)
            node.children[proj_value] = child_node
        else
            # Multiple patches - recurse
            child_node = _build_tree_recursive(parttt, remaining_sites_next, patches_group)
            node.children[proj_value] = child_node
        end
    end
    
    return node
end

"""
    find_in_tree(
        tree::ProjectorTreeNode,
        target_projector::Projector,
        all_sites::Vector{Index}
    )::Union{Nothing, SubDomainTT}

Find the matching SubDomainTT in the tree for the given target_projector.

# Arguments
- `tree::ProjectorTreeNode`: Root of the projector tree
- `target_projector::Projector`: The projector to search for
- `all_sites::Vector{Index}`: All site indices in order

# Returns
- `Union{Nothing, SubDomainTT}`: The matching SubDomainTT, or nothing if not found
"""
function find_in_tree(
    tree::ProjectorTreeNode,
    target_projector::Projector,
    all_sites::AbstractVector{<:Index}
)::Union{Nothing, SubDomainTT}
    
    # If this is a leaf node, check if it matches
    if tree.leaf !== nothing
        (proj, subdtt) = tree.leaf
        if target_projector <= proj
            return subdtt
        else
            return nothing
        end
    end
    
    # If this is an internal node, follow the branch
    if tree.site === nothing
        return nothing
    end
    
    # Get the projection value at this site
    proj_value = haskey(target_projector, tree.site) ? target_projector[tree.site] : nothing
    
    # If target_projector doesn't project at this site, we need to check all children
    # (This can happen if the tree structure doesn't perfectly match)
    if proj_value === nothing
        # Check all children
        for child in values(tree.children)
            result = find_in_tree(child, target_projector, all_sites)
            if result !== nothing
                return result
            end
        end
        return nothing
    end
    
    # Follow the branch for this projection value
    if haskey(tree.children, proj_value)
        return find_in_tree(tree.children[proj_value], target_projector, all_sites)
    else
        return nothing
    end
end

