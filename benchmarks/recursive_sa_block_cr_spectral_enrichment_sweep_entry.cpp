// Entry wrapper for the block-CR spectral reference benchmark.
// Keep the benchmark source isolated while preserving the repository's
// canonical ReferenceMultilevelResult type name.
#include "gfss/multilevel_reference.hpp"
namespace gfss {
using ReferenceMultilevelSolveResult = ReferenceMultilevelResult;
}
#include "recursive_sa_block_cr_spectral_enrichment_sweep_bench.cpp"
