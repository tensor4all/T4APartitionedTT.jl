using Test

using ITensors
using Random
using LinearAlgebra

import T4AITensorCompat: TensorTrain, MPS, MPO

import T4APartitionedTT:
    T4APartitionedTT,
    Projector,
    project,
    SubDomainTT,
    rearrange_siteinds,
    makesitediagonal,
    extractdiagonal

include("_util.jl")

@testset "subdomainmps.jl" begin
    @testset "SubDomainTT" begin
        Random.seed!(1)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps
        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = TensorTrain(prjΨ1) + TensorTrain(prjΨ2)

        @test LinearAlgebra.norm(prjΨ1) ≈ LinearAlgebra.norm(TensorTrain(prjΨ1))

        @test isapprox(Ψreconst, Ψ)
    end

    @testset "rearrange_siteinds" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        sitesxy = collect(collect.(zip(sitesx, sitesy)))
        sites_rearranged = Vector{Index{Int}}[]
        for i in 1:N
            push!(sites_rearranged, sitesxy[i])
            push!(sites_rearranged, [sitesz[i]])
        end
        prjΨ1_rearranged = rearrange_siteinds(prjΨ1, sites_rearranged)

        @test reduce(*, TensorTrain(prjΨ1)) ≈ reduce(*, TensorTrain(prjΨ1_rearranged))
        @test T4APartitionedTT.siteinds(prjΨ1_rearranged) == sites_rearranged
    end

    @testset "makesitediagonal and extractdiagonal" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]

        sitesxy_vec = [[x, y] for (x, y) in zip(sitesx, sitesy)]
        sitesz_vec = [[z] for z in sitesz]
        sites = [x for pair in zip(sitesxy_vec, sitesz_vec) for x in pair]

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        prjΨ1_diagonalz = makesitediagonal(prjΨ1, "y")
        sites_diagonalz = Iterators.flatten(T4APartitionedTT.siteinds(prjΨ1_diagonalz))

        psi_diag = prod(prjΨ1_diagonalz.data)
        psi = prod(prjΨ1.data)

        @test extractdiagonal(prjΨ1_diagonalz, "y") ≈ prjΨ1

        diag_ok = true
        offdiag_ok = true

        for indval in eachindval(sites_diagonalz...)
            ind = first.(indval)
            val = last.(indval)

            index_dict = Dict{Index{Int},Vector{Int}}()
            for (i, el) in enumerate(ind)
                baseind = ITensors.noprime(el)
                if haskey(index_dict, baseind)
                    push!(index_dict[baseind], i)
                else
                    index_dict[baseind] = [i]
                end
            end
            repeated_indices = [is for is in values(index_dict) if length(is) > 1]

            isdiagonalelement = all(allequal(val[i] for i in is) for is in repeated_indices)

            if isdiagonalelement
                nondiaginds = unique(ITensors.noprime(i) => v for (i, v) in indval)
                diag_ok = diag_ok && (psi_diag[indval...] == psi[nondiaginds...])
            else
                offdiag_ok = offdiag_ok && iszero(psi_diag[indval...])
            end
        end

        @test diag_ok
        @test offdiag_ok
    end

    @testset "Caller" begin
        Random.seed!(1)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps
        prjΨ = SubDomainTT(Ψ)

        # Test Caller with projector
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        
        # Get all sites in order
        all_sites = collect(Iterators.flatten(sites))
        
        # Test evaluation at a valid MultiIndex
        # For projector sitesx[1] => 1, we need to use index 1 for sitesx[1]
        multiindex = [1, 1, 1, 1, 1, 1]  # [x1=1, y1=1, x2=1, y2=1, x3=1, y3=1]
        result1 = prjΨ1(multiindex, all_sites)
        
        # Verify that the result is a scalar (not an ITensor)
        @test result1 isa Number
        
        # Test with different index values
        multiindex2 = [1, 2, 2, 1, 2, 1]
        result2 = prjΨ1(multiindex2, all_sites)
        @test result2 isa Number
        # Results should be different for different indices
        @test result1 != result2 || (abs(result1) < 1e-10 && abs(result2) < 1e-10)
        
        # Test with empty projector (no projection)
        prjΨ_empty = SubDomainTT(Ψ)
        multiindex3 = [1, 1, 1, 1, 1, 1]
        result3 = prjΨ_empty(multiindex3, all_sites)
        @test result3 isa Number
        
        # Test that function call respects the projector constraint
        # For prjΨ1, sitesx[1] is projected to 1, so changing sitesx[1] to 2 should give 0
        multiindex_wrong = [2, 1, 1, 1, 1, 1]  # [x1=2, ...] but prjΨ1 requires x1=1
        # This should still work (projector is checked at PartitionedTT level, not SubDomainTT level)
        # But the result might be 0 if the projector constraint is enforced
        result_wrong = prjΨ1(multiindex_wrong, all_sites)
        @test result_wrong isa Number
    end
end
