"""
A TensorTrain with a projector.
"""
struct SubDomainTT
    data::TensorTrain
    projector::Projector

    function SubDomainTT(data::TensorTrain, projector::Projector)
        _iscompatible(projector, data) || error(
            "Incompatible projector and data. Even small numerical noise can cause this error.",
        )
        projector = _trim_projector(data, projector)
        return new(TensorTrain([x for x in data]), projector)
    end
end

siteinds(obj::SubDomainTT) = siteinds(obj.data)

ITensors.siteinds(obj::SubDomainTT) = siteinds(obj.data)

_allsites(Ψ::TensorTrain) = collect(Iterators.flatten(siteinds(Ψ)))
_allsites(Ψ::SubDomainTT) = _allsites(Ψ.data)

maxlinkdim(Ψ::SubDomainTT) = maxlinkdim(Ψ.data)
maxbonddim(Ψ::SubDomainTT) = maxlinkdim(Ψ.data)

function _trim_projector(obj::TensorTrain, projector)
    sites = Set(_allsites(obj))
    newprj = deepcopy(projector)
    for (k, v) in newprj.data
        if !(k in sites)
            delete!(newprj.data, k)
        end
    end
    return newprj
end

function SubDomainTT(Ψ::TensorTrain)
    return SubDomainTT(Ψ, Projector())
end

# Conversion Functions
# Conversion to TensorTrain
TensorTrain(projΨ::SubDomainTT) = projΨ.data

function project(tensor::ITensor, projector::Projector)
    slice = Union{Int,Colon}[
        isprojectedat(projector, idx) ? projector[idx] : Colon() for
        idx in ITensors.inds(tensor)
    ]
    data_org = Array(tensor, ITensors.inds(tensor)...)
    data_trim = zero(data_org)
    if all(broadcast(x -> x isa Integer, slice))
        data_trim[slice...] = data_org[slice...]
    else
        data_trim[slice...] .= data_org[slice...]
    end
    return ITensor(data_trim, ITensors.inds(tensor)...)
end

function project(projΨ::SubDomainTT, projector::Projector)::Union{Nothing,SubDomainTT}
    if !hasoverlap(projector, projΨ.projector)
        return nothing
    end

    return SubDomainTT(
        TensorTrain([project(projΨ.data[n], projector) for n in 1:length(projΨ.data)]),
        projector,
    )
end

function project(
    projΨ::SubDomainTT, pairs::Vararg{Pair{Index{T},Int}}
)::Union{Nothing,SubDomainTT} where {T}
    return project(projΨ, Projector(pairs...))
end

function project(
    Ψ::TensorTrain, pairs::Vararg{Pair{Index{T},Int}}
)::Union{Nothing,SubDomainTT} where {T}
    return project(Ψ, Projector(pairs...))
end

function project(Ψ::TensorTrain, projector::Projector)::Union{Nothing,SubDomainTT}
    return project(SubDomainTT(Ψ), projector)
end

function project(
    projΨ::SubDomainTT, projector::Dict{InsT,Int}
)::Union{Nothing,SubDomainTT} where {InsT}
    return project(projΨ, Projector(projector))
end

function project(
    Ψ::TensorTrain, projector::Dict{InsT,Int}
)::Union{Nothing,SubDomainTT} where {InsT}
    return project(SubDomainTT(Ψ), Projector(projector))
end

function _iscompatible(projector::Projector, tensor::ITensor)
    # Check compatibility by directly projecting the ITensor (not TensorTrain)
    # This is safe because project(tensor::ITensor, projector) returns ITensor, not SubDomainTT
    projected = project(tensor, projector)
    return ITensors.norm(projected - tensor) == 0.0
end

function _iscompatible(projector::Projector, Ψ::TensorTrain)
    # Check each tensor individually to avoid infinite recursion
    # We check each ITensor in the TensorTrain, not the TensorTrain itself
    # This avoids calling project(Ψ, projector) which would create SubDomainTT
    for x in Ψ
        if !_iscompatible(projector, x)
            return false
        end
    end
    return true
end

function rearrange_siteinds(subdtt::SubDomainTT, sites)
    tt_rearranged = rearrange_siteinds(TensorTrain(subdtt), sites)
    return project(SubDomainTT(tt_rearranged), subdtt.projector)
end

# Miscellaneous Functions
function Base.show(io::IO, obj::SubDomainTT)
    return print(io, "SubDomainTT projected on $(obj.projector.data)")
end

function prime(Ψ::SubDomainTT, plinc=1; kwargs...)
    return SubDomainTT(
        ITensors.prime(TensorTrain(Ψ), plinc; kwargs...),
        T4APartitionedTT.prime(Ψ.projector, plinc; kwargs...),
    )
end

function noprime(Ψ::SubDomainTT, args...; kwargs...)
    if :inds ∈ keys(kwargs)
        targetsites = kwargs[:inds]
    else
        targetsites = nothing
    end

    return SubDomainTT(
        ITensors.noprime(TensorTrain(Ψ), args...; kwargs...),
        T4APartitionedTT.noprime(Ψ.projector; targetsites),
    )
end

function Base.isapprox(x::SubDomainTT, y::SubDomainTT; kwargs...)
    return Base.isapprox(x.data, y.data, kwargs...)
end

function isprojectedat(obj::SubDomainTT, ind::IndsT)::Bool where {IndsT}
    return isprojectedat(obj.projector, ind)
end

function _fitsum(
    input_states::AbstractVector{T},
    init::T;
    coeffs::AbstractVector{<:Number}=ones(Int, length(input_states)),
    kwargs...,
) where {T}
    if !(:nsweeps ∈ keys(kwargs))
        kwargs = merge(Dict(kwargs), Dict(:nsweeps => 1))
    end
    # input_states and init are already TensorTrain, so we can use them directly
    Ψs = [x for x in input_states]  # Already TensorTrain, no need to convert
    init_Ψ = init  # Already TensorTrain, no need to convert
    res = fit(Ψs, init_Ψ; coeffs=coeffs, kwargs...)
    # res is already TensorTrain, so we can return it directly
    return res
end

function _add(ψ::TensorTrain...; alg="fit", cutoff=1e-15, maxdim=typemax(Int), kwargs...)
    if alg == "directsum"
        return +(ITensors.Algorithm(alg), ψ...)
    elseif alg == "densitymatrix"
        if cutoff < 1e-15
            @warn "Cutoff is very small, it may suffer from numerical round errors. 
                    The densitymatrix algorithm squares the singular values of the reduce density matrix. 
                    Please consider increasing it or using fit algorithm."
        end
        return +(ITensors.Algorithm("densitymatrix"), ψ...; cutoff, maxdim, kwargs...)
    elseif alg == "fit"
        function f(x, y)
            return T4AITensorCompat.truncate(
                +(ITensors.Algorithm("directsum"), x, y); cutoff, maxdim, kwargs...
            )
        end
        res_dm = reduce(f, ψ)
        res = _fitsum([x for x in ψ], res_dm; cutoff, maxdim, kwargs...)
        return res
    else
        error("Unknown algorithm $(alg) for addition!")
    end
end

function Base.:+(
    Ψ::SubDomainTT...; alg="fit", cutoff=0.0, maxdim=typemax(Int), kwargs...
)::SubDomainTT
    return _add(Ψ...; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs...)
end

function _add(
    Ψ::SubDomainTT...; alg="fit", cutoff=0.0, maxdim=typemax(Int), kwargs...
)::SubDomainTT
    return project(
        _add([x.data for x in Ψ]...; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs...),
        reduce(|, [x.projector for x in Ψ]),
    )
end

function Base.:*(a::SubDomainTT, b::Number)::SubDomainTT
    return SubDomainTT(a.data * b, a.projector)
end

function Base.:*(a::Number, b::SubDomainTT)::SubDomainTT
    return SubDomainTT(b.data * a, b.projector)
end

function Base.:-(obj::SubDomainTT)::SubDomainTT
    return SubDomainTT(-1 * obj.data, obj.projector)
end

function truncate(
    obj::SubDomainTT;
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    abs_cutoff=default_abs_cutoff(),
    kwargs...,
)::SubDomainTT
    # T4AITensorCompat.truncate handles abs_cutoff internally and converts it to adjusted cutoff
    # abs_cutoff is already a keyword argument, so it won't be in kwargs
    return project(
        SubDomainTT(
            T4AITensorCompat.truncate(
                obj.data; cutoff=cutoff, maxdim=maxdim, abs_cutoff=abs_cutoff, kwargs...
            ),
        ),
        obj.projector,
    )
end

function LinearAlgebra.norm(M::SubDomainTT)
    return LinearAlgebra.norm(TensorTrain(M))
end

function _makesitediagonal(
    obj::SubDomainTT, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    M_ = deepcopy(TensorTrain(obj))
    for site in sites
        target_site::Int = only(findsites(M_, site))
        M_[target_site] = _asdiagonal(M_[target_site], site; baseplev=baseplev)
    end

    newproj = deepcopy(obj.projector)
    for s in sites
        if isprojectedat(obj.projector, s)
            newproj.data[ITensors.prime(s, baseplev + 1)] = newproj.data[s]
            if baseplev != 0
                newproj.data[ITensors.prime(s, baseplev)] = newproj.data[s]
                delete!(newproj.data, s)
            end
        end
    end

    return project(M_, newproj)
end

function makesitediagonal(obj::SubDomainTT, site::Index{IndsT}; baseplev=0) where {IndsT}
    return _makesitediagonal(obj, [site]; baseplev=baseplev)
end

function makesitediagonal(
    obj::SubDomainTT, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    return _makesitediagonal(obj, sites; baseplev=baseplev)
end

function makesitediagonal(obj::SubDomainTT, tag::String)
    tt_diagonal = makesitediagonal(TensorTrain(obj), tag)
    SubDomainTT_diagonal = SubDomainTT(tt_diagonal)

    target_sites = findallsiteinds_by_tag(
        unique(ITensors.noprime.(Iterators.flatten(siteinds(obj)))); tag=tag
    )

    newproj = deepcopy(obj.projector)
    for s in target_sites
        if isprojectedat(obj.projector, s)
            newproj[ITensors.prime(s)] = newproj[s]
        end
    end

    return project(SubDomainTT_diagonal, newproj)
end

function extractdiagonal(
    obj::SubDomainTT, sites::AbstractVector{Index{IndsT}}
) where {IndsT}
    tensors = Vector{ITensor}(collect(obj.data))
    for i in eachindex(tensors)
        for site in intersect(sites, ITensors.inds(tensors[i]))
            sitewithallplevs = _find_site_allplevs(tensors[i], site)
            tensors[i] = if length(sitewithallplevs) > 1
                tensors[i] = _extract_diagonal(tensors[i], sitewithallplevs...)
            else
                tensors[i]
            end
        end
    end

    newD = Dict{Index,Int}()
    # Duplicates of keys are discarded
    for (k, v) in obj.projector.data
        newk = ITensors.noprime(k)
        newD[newk] = v
    end
    return SubDomainTT(TensorTrain(tensors), Projector(newD))
end

function extractdiagonal(obj::SubDomainTT, tag::String)::SubDomainTT
    targetsites = findallsiteinds_by_tag(unique(ITensors.noprime.(_allsites(obj))); tag=tag)
    return extractdiagonal(obj, targetsites)
end

function extractdiagonal(subdtt::SubDomainTT, site::Index{IndsT}) where {IndsT}
    return extractdiagonal(subdtt, [site])
end

"""
    (subdtt::SubDomainTT)(multiindex::Vector{Int}, all_sites::AbstractVector{<:Index})

Evaluate the SubDomainTT at the given MultiIndex.

# Arguments
- `multiindex::Vector{Int}`: The MultiIndex (vector of site index values) to evaluate at
- `all_sites::AbstractVector{<:Index}`: All site indices in the order corresponding to multiindex

# Returns
- The evaluation value of the SubDomainTT at the given MultiIndex
"""
function (subdtt::SubDomainTT)(multiindex::Vector{Int}, all_sites::AbstractVector{<:Index})
    # Get site indices of the SubDomainTT
    subdtt_sites = siteinds(subdtt)

    # Create a mapping from all_sites indices to MultiIndex positions
    idx_map = Dict{Index,Int}()
    for (pos, idx) in enumerate(all_sites)
        idx_map[idx] = pos
    end

    # Build index values for each site in SubDomainTT
    site_index_vals = Vector{Int}[]
    for site_group in subdtt_sites
        push!(site_index_vals, [multiindex[idx_map[idx]] for idx in site_group])
    end

    # Evaluate using ITensors onehot (similar to _evaluate function)
    tt = TensorTrain(subdtt)
    return only(
        reduce(
            *,
            [
                tt[n] * reduce(
                    *,
                    [
                        ITensors.onehot(idx => val) for
                        (idx, val) in zip(subdtt_sites[n], site_index_vals[n])
                    ],
                ) for n in 1:length(tt)
            ],
        ),
    )
end
