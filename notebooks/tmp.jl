using Revise
using Random
import QuanticsGrids as QG

# Make sure to load T4AITensorCompat first
# before importing other T4A packages
# to activate the ITensor-related extension
import ITensors: ITensors, Index
import T4AITensorCompat as T4AIT
import T4ATensorCI as TCI
import T4ATensorCI
import T4AQuanticsTCI as QTCI
import T4APartitionedTT as T4AP
import T4APartitionedTT: PartitionedTT, SubDomainTT, adaptive_patching, Projector
using CairoMakie
using Test

Random.seed!(1234)

gaussian(x, y) = exp(- ((x-5)^2 + (y-5)^2)) + exp(- ((x+5)^2 + (y+5)^2))

R = 20
xmax = 10.0
grid = QG.DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax); unfoldingscheme=:interleaved)
tci_tolerance = 1e-7
qtci, ranks, errors = QTCI.quanticscrossinterpolate(Float64, gaussian, grid; verbosity=1, maxbonddim=100, loginterval=1, tolerance=tci_tolerance)


# Covert to ITensor-compatible TT format
sitesx = [Index(2, "x=$i") for i in 1:R]
sitesy = [Index(2, "y=$i") for i in 1:R]
all_sites = collect(Iterators.flatten(zip(sitesx, sitesy)))
itt = T4AIT.TensorTrain(TCI.TensorTrain(qtci.tci); sites=all_sites)
;
# Split into patches
maxdim_patch = 15
cutoff_patch = 0.1 * tci_tolerance^2
patchorder = collect(Iterators.flatten(T4AP.siteinds(itt)))
patched_tt = adaptive_patching(PartitionedTT(SubDomainTT(deepcopy(itt))), patchorder; maxdim=maxdim_patch, cutoff=cutoff_patch)
