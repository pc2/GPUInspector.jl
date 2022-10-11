using TestItemRunner
using GPUInspector
using CUDA

if !GPUInspector.functional()
    error("Can't run testsuite since CUDA/GPU not present or not functional!")
end
if Threads.nthreads() == 1 || (Threads.nthreads() < length(CUDA.devices()) + 1)
    # we should have at least one thread per gpu + one monitoring thread
    @warn(
        "You should run the tests with at least $(length(CUDA.devices()) + 1) Julia threads.",
        Threads.nthreads(),
        length(CUDA.devices())
    )
end

@run_package_tests

include("utility_tests.jl")
include("stresstest_tests.jl")
include("bandwidth_tests.jl")
include("peakflops_tests.jl")
include("gpuinfo_tests.jl")
