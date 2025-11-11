using T4APartitionedMPSs
using Documenter

DocMeta.setdocmeta!(
    T4APartitionedMPSs, :DocTestSetup, :(using T4APartitionedMPSs); recursive=true
)

makedocs(;
    modules=[T4APartitionedMPSs],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="T4APartitionedMPSs.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4APartitionedMPSs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/tensor4all/T4APartitionedMPSs.jl.git", devbranch="main")
