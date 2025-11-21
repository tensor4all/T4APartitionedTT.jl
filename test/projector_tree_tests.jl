using Test

using ITensors
using Random

import T4AITensorCompat: TensorTrain, MPS, MPO

import T4APartitionedTT:
    T4APartitionedTT,
    Projector,
    project,
    SubDomainTT,
    PartitionedTT,
    build_projector_tree,
    count_site_projections,
    find_in_tree,
    ProjectorTreeNode

include("_util.jl")

@testset "projector_tree.jl" begin
    @testset "count_site_projections" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        # Create non-overlapping projectors
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 2))
        prjΨ3 = project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 1))
        prjΨ4 = project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 2))

        parttt = PartitionedTT([prjΨ1, prjΨ2, prjΨ3, prjΨ4])
        
        all_sites = collect(Iterators.flatten(sites))
        counts = count_site_projections(parttt, all_sites)
        
        # sitesx[1] should be projected in all 4 patches
        @test counts[sitesx[1]] == 4
        # sitesx[2] should be projected in all 4 patches
        @test counts[sitesx[2]] == 4
        # sitesx[3] should not be projected
        @test counts[sitesx[3]] == 0
        # sitesy should not be projected
        for site in sitesy
            @test counts[site] == 0
        end
    end

    @testset "build_projector_tree and find_in_tree" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        # Create non-overlapping projectors
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 2))
        prjΨ3 = project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 1))
        prjΨ4 = project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 2))

        parttt = PartitionedTT([prjΨ1, prjΨ2, prjΨ3, prjΨ4])
        
        all_sites = collect(Iterators.flatten(sites))
        tree = build_projector_tree(parttt, all_sites)
        
        # Test finding each patch
        # Find prjΨ1 (sitesx[1] => 1, sitesx[2] => 1)
        target1 = Projector(Dict(sitesx[1] => 1, sitesx[2] => 1))
        result1 = find_in_tree(tree, target1, all_sites)
        @test result1 !== nothing
        @test result1.projector == prjΨ1.projector
        
        # Find prjΨ2 (sitesx[1] => 1, sitesx[2] => 2)
        target2 = Projector(Dict(sitesx[1] => 1, sitesx[2] => 2))
        result2 = find_in_tree(tree, target2, all_sites)
        @test result2 !== nothing
        @test result2.projector == prjΨ2.projector
        
        # Find prjΨ3 (sitesx[1] => 2, sitesx[2] => 1)
        target3 = Projector(Dict(sitesx[1] => 2, sitesx[2] => 1))
        result3 = find_in_tree(tree, target3, all_sites)
        @test result3 !== nothing
        @test result3.projector == prjΨ3.projector
        
        # Find prjΨ4 (sitesx[1] => 2, sitesx[2] => 2)
        target4 = Projector(Dict(sitesx[1] => 2, sitesx[2] => 2))
        result4 = find_in_tree(tree, target4, all_sites)
        @test result4 !== nothing
        @test result4.projector == prjΨ4.projector
        
        # Test finding non-existent projector
        target_none = Projector(Dict(sitesx[1] => 3))  # Invalid value
        result_none = find_in_tree(tree, target_none, all_sites)
        @test result_none === nothing
    end

    @testset "build_projector_tree with single patch" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        parttt = PartitionedTT([prjΨ1])
        
        all_sites = collect(Iterators.flatten(sites))
        tree = build_projector_tree(parttt, all_sites)
        
        # Tree should be a leaf node
        @test tree.leaf !== nothing
        @test tree.site === nothing
        
        # Should find the patch
        target = Projector(Dict(sitesx[1] => 1))
        result = find_in_tree(tree, target, all_sites)
        @test result !== nothing
        @test result.projector == prjΨ1.projector
    end

    @testset "build_projector_tree with multiple dimensions" begin
        Random.seed!(1234)
        N = 4
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        # Create patches with different projection patterns (non-overlapping)
        patches = [
            project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 1, sitesx[3] => 1)),
            project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 1, sitesx[3] => 2)),
            project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 2, sitesx[3] => 1)),
            project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 2, sitesx[3] => 2)),
            project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 1, sitesx[3] => 1)),
            project(prjΨ, Dict(sitesx[1] => 2, sitesx[2] => 1, sitesx[3] => 2)),
        ]

        parttt = PartitionedTT(patches)
        
        all_sites = collect(Iterators.flatten(sites))
        tree = build_projector_tree(parttt, all_sites)
        
        # Test finding each patch
        for patch in patches
            target = patch.projector
            result = find_in_tree(tree, target, all_sites)
            @test result !== nothing
            @test result.projector == patch.projector
        end
    end
end

