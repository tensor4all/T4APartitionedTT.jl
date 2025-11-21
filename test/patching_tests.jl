using Test
import T4APartitionedTT:
    T4APartitionedTT, Projector, project, SubDomainTT, adaptive_patching, PartitionedTT
import T4AITensorCompat: TensorTrain
using ITensors
using Random

include("_util.jl")

@testset "patching.jl" begin
    @testset "adaptive_patching" begin
        Random.seed!(1234)

        R = 3
        sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]

        sites = collect(collect.(zip(sitesx, sitesy)))

        mpo = _random_mpo(sites; linkdims=20)
        subdtt = SubDomainTT(mpo)

        sites_ = collect(Iterators.flatten(sites))
        parttt = adaptive_patching(PartitionedTT(subdtt), sites_; maxdim=10, cutoff=1e-25)

        @test length(values((parttt))) > 1

        @test TensorTrain(parttt) ≈ TensorTrain(subdtt) rtol = 1e-12
    end
end
