"""
    ProjectorTreeNode

A tree node for efficient lookup of SubDomainTT by projector.

# Fields
- `site::Union{Index, Nothing}`: The site index used for branching at this node (Nothing for leaf nodes)
- `children::Dict{Union{Int, Nothing}, ProjectorTreeNode}`: Map from projection value to child node
- `leaf::Union{Nothing, Tuple{Projector, SubDomainTT}}`: Leaf node data (projector and SubDomainTT)
"""
mutable struct ProjectorTreeNode
    site::Union{Index,Nothing}
    children::Dict{Union{Int,Nothing},ProjectorTreeNode}
    leaf::Union{Nothing,Tuple{Projector,SubDomainTT}}

    function ProjectorTreeNode(site::Union{Index,Nothing}=nothing)
        return new(site, Dict{Union{Int,Nothing},ProjectorTreeNode}(), nothing)
    end
end
