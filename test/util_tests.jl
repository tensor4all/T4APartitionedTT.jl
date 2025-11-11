using Test

using ITensors
import T4AITensorCompat: TensorTrain, MPS, MPO

import T4APartitionedMPSs:
    T4APartitionedMPSs,
    Projector,
    project,
    SubDomainMPS,
    projcontract,
    PartitionedMPS,
    rearrange_siteinds,
    makesitediagonal,
    extractdiagonal

include("_util.jl")

@testset "util.jl" begin
    @testset "rearrange_siteinds" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

        Ψ_mps = _random_mpo(sites)
        Ψ = Ψ_mps

        prjΨ = SubDomainMPS(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        sitesxy = collect(collect.(zip(sitesx, sitesy)))
        sites_rearranged = Vector{Index{Int}}[]
        for i in 1:N
            push!(sites_rearranged, sitesxy[i])
            push!(sites_rearranged, [sitesz[i]])
        end
        prjΨ1_rearranged = rearrange_siteinds(prjΨ1, sites_rearranged)

        @test reduce(*, TensorTrain(prjΨ1)) ≈ reduce(*, TensorTrain(prjΨ1_rearranged))
        @test T4APartitionedMPSs.siteinds(prjΨ1_rearranged) == sites_rearranged
    end
end
