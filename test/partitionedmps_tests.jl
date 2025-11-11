using Test

using ITensors
using Random

import T4AITensorCompat: TensorTrain, MPS, MPO, siteinds

import T4APartitionedMPSs:
    T4APartitionedMPSs, Projector, project, SubDomainMPS, PartitionedMPS, prime, noprime, dist, siteinds

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

        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        @test_throws ErrorException PartitionedMPS([prjΨ, prjΨ1])
        @test_throws ErrorException PartitionedMPS([prjΨ1, prjΨ1])

        # Iterator and length
        @test length(PartitionedMPS(prjΨ1)) == 1
        @test length([(k, v) for (k, v) in PartitionedMPS(prjΨ1)]) == 1

        Ψreconst = PartitionedMPS(prjΨ1) + PartitionedMPS(prjΨ2)
        @test Ψreconst[Projector(sitesx[1] => 1)] ≈ prjΨ1
        @test Ψreconst[Projector(sitesx[1] => 2)] ≈ prjΨ2
        @test TensorTrain(Ψreconst) ≈ Ψ
        @test ITensors.norm(Ψreconst) ≈ ITensors.norm(TensorTrain(Ψreconst))

        # Summation
        coeffs = (1.1, 0.9)
        @test TensorTrain(+(PartitionedMPS(prjΨ1), PartitionedMPS(prjΨ2); coeffs=coeffs)) ≈
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

        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        a = PartitionedMPS(prjΨ1)
        b = PartitionedMPS(prjΨ2)

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

        @test TensorTrain(+([PartitionedMPS(x) for x in prjΨs]...; coeffs=coeffs)) ≈ +([c * TensorTrain(x) for (c, x) in zip(coeffs, prjΨs)]...; alg="directsum")
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

            partmps = PartitionedMPS([project(Ψ, Dict(sites[1][1] => d_)) for d_ in 1:d])
            partmps_truncated = T4APartitionedMPSs.truncate(partmps; cutoff=cutoff_global)

            diff =
                dist(TensorTrain(partmps_truncated), TensorTrain(partmps))^2 /
                norm(TensorTrain(partmps))^2
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

        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = PartitionedMPS(prjΨ1) + PartitionedMPS(prjΨ2)

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
end
