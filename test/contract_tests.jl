using Test
import T4APartitionedMPSs:
    T4APartitionedMPSs, PartitionedMPS, Projector, project, SubDomainMPS, projcontract
import T4AITensorCompat: TensorTrain, contract

@testset "contract.jl" begin
    @testset "contract (xk-y-z)" begin
        R = 3
        sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
        sitesk = [Index(2, "Qubit,k=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]
        sitesz = [Index(2, "Qubit,z=$n") for n in 1:R]

        sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
        sitesb = collect(collect.(zip(sitesy, sitesz)))

        a_mpo = _random_mpo(sitesa)
        b_mpo = _random_mpo(sitesb)
        p1 = project(SubDomainMPS(a_mpo), Projector(Dict(sitesx[1] => 1)))
        p2 = project(SubDomainMPS(b_mpo), Projector(Dict(sitesz[1] => 1)))

        p12 = T4APartitionedMPSs.contract(p1, p2; alg="naive")

        @test p12.projector == Projector(Dict(sitesx[1] => 1, sitesz[1] => 1))

        proj_subset = Projector(Dict(sitesx[1] => 1, sitesz[1] => 1, sitesk[1] => 1))
        p12_2 = projcontract(p1, p2, proj_subset; alg="naive")

        @test p12_2.projector == proj_subset
    end

    @testset "contract (2x2)" for patching in [true, false]
        R = 10
        cutoff = 1e-10
        linkdims = 5

        sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]
        sitesz = [Index(2, "Qubit,z=$n") for n in 1:R]

        sitesa = collect(collect.(zip(sitesx, sitesy)))
        sitesb = collect(collect.(zip(sitesy, sitesz)))

        a = _random_mpo(sitesa; linkdims=linkdims)
        b = _random_mpo(sitesb; linkdims=linkdims)

        proj_a = [
            project(SubDomainMPS(a), Projector(Dict(sitesx[1] => i, sitesy[1] => j))) for
            i in 1:2, j in 1:2
        ]
        proj_b = [
            project(SubDomainMPS(b), Projector(Dict(sitesy[1] => i, sitesz[1] => j))) for
            i in 1:2, j in 1:2
        ]

        for x in [1, 2], y in [1, 2]
            res = projcontract(
                vec(proj_a),
                vec(proj_b),
                Projector(Dict(sitesx[1] => x, sitesz[1] => y));
                cutoff=cutoff,
            )
            ref = reduce(
                (x, y) -> +(x, y; alg="directsum"),
                [
                    contract(
                        proj_a[x, k].data, proj_b[k, y].data; alg=ITensors.Algorithm"naive"()
                    ) for k in 1:2
                ],
            )
            @test res[1].data ≈ ref
        end

        patchorder = patching ? collect(Iterators.flatten(zip(sitesx, sitesz))) : Index[]
        maxdim_ = patching ? linkdims^2 : typemax(Int)
        ab = T4APartitionedMPSs.contract(
            PartitionedMPS(vec(proj_a)),
            PartitionedMPS(vec(proj_b));
            alg="fit",
            cutoff,
            maxdim=maxdim_,
            patchorder,
        )
        @test TensorTrain(ab; cutoff) ≈ contract(a, b; alg=ITensors.Algorithm"naive"()) rtol =
            10 * sqrt(cutoff)
    end
end
