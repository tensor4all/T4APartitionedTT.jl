# T4APartitionedTT

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/T4APartitionedTT.jl/dev)
[![CI](https://github.com/tensor4all/T4APartitionedTT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/T4APartitionedTT.jl/actions/workflows/CI.yml)

## General Information
This library provides an implementation of partitioned tensor trains (TTs) in Julia. A `PartitionedTT` (short for PartitionedTensorTrain) object is a collection of `SubDomainTT` (short for SubDomain Tensor Train) objects, each of which is a TT projected on a subdomain of the whole index set.
This library also provides a block-wise multiplication of partitioned TTs.


## Usage

Please refer to the [documentation](https://tensor4all.github.io/T4APartitionedTT.jl/) for usage.