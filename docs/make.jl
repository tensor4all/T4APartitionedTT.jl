using T4APartitionedTT
using Documenter

DocMeta.setdocmeta!(
    T4APartitionedTT, :DocTestSetup, :(using T4APartitionedTT); recursive=true
)

makedocs(;
    modules=[T4APartitionedTT],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="T4APartitionedTT.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4APartitionedTT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/tensor4all/T4APartitionedTT.jl.git", devbranch="main")
