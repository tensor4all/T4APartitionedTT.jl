using Test

using ITensors
using Random

import T4AITensorCompat: TensorTrain, MPS, MPO, siteinds

import T4APartitionedTT:
    T4APartitionedTT,
    Projector,
    project,
    SubDomainTT,
    PartitionedTT,
    prime,
    noprime,
    dist,
    siteinds

include("_util.jl")

@testset "partitionedmps.jl" begin
    @testset "two blocks" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        @test_throws ErrorException PartitionedTT([prjΨ, prjΨ1])
        @test_throws ErrorException PartitionedTT([prjΨ1, prjΨ1])

        # Iterator and length
        @test length(PartitionedTT(prjΨ1)) == 1
        @test length([(k, v) for (k, v) in PartitionedTT(prjΨ1)]) == 1

        Ψreconst = PartitionedTT(prjΨ1) + PartitionedTT(prjΨ2)
        @test Ψreconst[Projector(sitesx[1] => 1)] ≈ prjΨ1
        @test Ψreconst[Projector(sitesx[1] => 2)] ≈ prjΨ2
        @test TensorTrain(Ψreconst) ≈ Ψ
        @test ITensors.norm(Ψreconst) ≈ ITensors.norm(TensorTrain(Ψreconst))

        # Summation
        coeffs = (1.1, 0.9)
        @test TensorTrain(+(PartitionedTT(prjΨ1), PartitionedTT(prjΨ2); coeffs=coeffs)) ≈
            coeffs[1] * TensorTrain(prjΨ1) + coeffs[2] * TensorTrain(prjΨ2)
    end

    @testset "two blocks (general key)" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        a = PartitionedTT(prjΨ1)
        b = PartitionedTT(prjΨ2)

        @test TensorTrain(2 * a) ≈ 2 * TensorTrain(a) rtol = 1e-13
        @test TensorTrain(a * 2) ≈ 2 * TensorTrain(a) rtol = 1e-13
        @test TensorTrain((a + b) + 2 * (b + a)) ≈ 3 * Ψ rtol = 1e-13
        @test TensorTrain((a + b) + 2 * (b + a)) ≈ 3 * Ψ rtol = 1e-13
    end

    @testset "add" begin
        Random.seed!(1234)
        N = 3
        d = 10
        sites = [Index(d, "x=$n") for n in 1:N]
        Ψ_mps = _random_mpo([[s] for s in sites])
        Ψ = Ψ_mps

        projectors = [Projector(sites[1] => d_) for d_ in 1:d]
        coeffs = rand(length(projectors))
        prjΨs = [project(Ψ, p) for p in projectors]

        @test TensorTrain(+([PartitionedTT(x) for x in prjΨs]...; coeffs=coeffs)) ≈
            +([c * TensorTrain(x) for (c, x) in zip(coeffs, prjΨs)]...; alg="directsum")
    end

    @testset "truncate" begin
        for seed in [1, 2, 3, 4, 5]
            Random.seed!(seed)
            N = 10
            D = 10 # Bond dimension
            d = 10 # local dimension
            cutoff_global = 1e-4

            sites = [[Index(d, "n=$n")] for n in 1:N]

            Ψ_mps = 100 * _random_mpo(sites; linkdims=D)
            Ψ = Ψ_mps

            parttt = PartitionedTT([project(Ψ, Dict(sites[1][1] => d_)) for d_ in 1:d])
            parttt_truncated = T4APartitionedTT.truncate(parttt; cutoff=cutoff_global)

            diff =
                dist(TensorTrain(parttt_truncated), TensorTrain(parttt))^2 /
                norm(TensorTrain(parttt))^2
            @test diff < cutoff_global
        end
    end

    @testset "prime" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = PartitionedTT(prjΨ1) + PartitionedTT(prjΨ2)

        Ψreconst_x3prime = prime(Ψreconst, 3; inds=sitesx)

        @test Set(Iterators.flatten(ITensors.siteinds(Ψreconst_x3prime))) ==
            Set(union(ITensors.prime.(sitesx, 3), sitesy))

        @test Set(keys(Ψreconst_x3prime)) == Set([
            Projector(ITensors.prime(sitesx[1], 3) => 1),
            Projector(ITensors.prime(sitesx[1], 3) => 2),
        ])

        Ψreconst_x3prime_yprime = prime(Ψreconst_x3prime; plev=0)

        @test Set(Iterators.flatten(ITensors.siteinds(Ψreconst_x3prime_yprime))) ==
            Set(union(ITensors.prime.(sitesx, 3), ITensors.prime.(sitesy, 1)))
    end

    @testset "siteinds" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = PartitionedTT(prjΨ1) + PartitionedTT(prjΨ2)

        # Test that siteinds(PartitionedTT) returns the same as siteinds(SubDomainTT)
        @test siteinds(Ψreconst) == siteinds(prjΨ1)
        @test siteinds(Ψreconst) == siteinds(prjΨ2)

        # Test that siteinds(PartitionedTT) matches ITensors.siteinds
        @test siteinds(Ψreconst) == ITensors.siteinds(prjΨ1)
        @test siteinds(PartitionedTT(prjΨ1)) == siteinds(prjΨ1)

        # Test with single SubDomainTT
        parttt_single = PartitionedTT(prjΨ1)
        @test siteinds(parttt_single) == siteinds(prjΨ1)
    end

    @testset "Caller" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        parttt = PartitionedTT(prjΨ1) + PartitionedTT(prjΨ2)

        # Test successful evaluation with matching SubDomainTT
        # For prjΨ1 (sitesx[1] => 1), use multiindex with sitesx[1] = 1
        all_sites_flat = collect(Iterators.flatten(sites))
        multiindex1 = [1, 1, 1, 1, 1, 1]  # [x1=1, y1=1, x2=1, y2=1, x3=1, y3=1]
        result1 = parttt(multiindex1)

        # Should match evaluation of prjΨ1 using function call directly
        expected1 = prjΨ1(multiindex1, all_sites_flat)
        @test result1 ≈ expected1

        # Test with prjΨ2 (sitesx[1] => 2)
        multiindex2 = [2, 1, 1, 1, 1, 1]  # [x1=2, y1=1, x2=1, y2=1, x3=1, y3=1]
        result2 = parttt(multiindex2)

        expected2 = prjΨ2(multiindex2, all_sites_flat)
        @test result2 ≈ expected2

        # Test error when MultiIndex length doesn't match
        @test_throws ArgumentError parttt([1, 1, 1])  # Too short
        @test_throws ArgumentError parttt([1, 1, 1, 1, 1, 1, 1])  # Too long

        # Test error when no matching SubDomainTT is found
        # Create a PartitionedTT with only prjΨ1 (sitesx[1] => 1)
        parttt_single = PartitionedTT(prjΨ1)

        # Try to evaluate with sitesx[1] = 2, which doesn't match prjΨ1
        multiindex_no_match = [2, 1, 1, 1, 1, 1]  # [x1=2, ...] but prjΨ1 requires x1=1
        @test_throws ArgumentError parttt_single(multiindex_no_match)

        # Test with a more complex case: multiple projectors
        prjΨ3 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 1))
        prjΨ4 = project(prjΨ, Dict(sitesx[1] => 1, sitesx[2] => 2))
        parttt_multi = PartitionedTT(prjΨ3) + PartitionedTT(prjΨ4)

        # Should match prjΨ3
        all_sites_flat_multi = collect(Iterators.flatten(sites))
        multiindex3 = [1, 1, 1, 1, 1, 1]  # [x1=1, y1=1, x2=1, y2=1, x3=1, y3=1]
        result3 = parttt_multi(multiindex3)
        expected3 = prjΨ3(multiindex3, all_sites_flat_multi)
        @test result3 ≈ expected3

        # Should match prjΨ4
        multiindex4 = [1, 1, 2, 1, 1, 1]  # [x1=1, y1=1, x2=2, y2=1, x3=1, y3=1]
        result4 = parttt_multi(multiindex4)
        expected4 = prjΨ4(multiindex4, all_sites_flat_multi)
        @test result4 ≈ expected4

        # Should not match any projector (x2=3 doesn't exist)
        multiindex_no_match2 = [1, 1, 3, 1, 1, 1]  # [x1=1, y1=1, x2=3, ...]
        @test_throws ArgumentError parttt_multi(multiindex_no_match2)
    end
end
