"""
    stresstest(; kwargs...)

Run a GPU stress test (matrix multiplication) on one or multiple GPU devices, as specified by the positional argument.
If no argument is provided (only) the currently active GPU will be used.

**Keyword arguments:**

Choose one of the following (or none):
* `duration`: stress test will take about the given time in seconds. (StressTestBatched)
* `enforced_duration`: stress test will take almost precisely the given time in seconds. (StressTestEnforced)
* `approx_duration`: stress test will hopefully take approximately the given time in seconds. No promises made! (StressTestFixedIter)
* `niter`: stress test will run the given number of matrix-multiplications, however long that will take. (StressTestFixedIter)
* `mem`: number (`<:Real`) between 0 and 1, indicating the fraction of the available GPU memory that should be used, or a `<:UnitPrefixedBytes` indicating an absolute memory limit. (StressTestStoreResults)

General settings:
* `devices` (default: e.g. `[CUDA.device()]`): GPU devices to be included in the stress test
* `dtype` (default: `Float32`): element type of the matrices
* `monitoring` (default: `false`): enable automatic monitoring, in which case a [`MonitoringResults`](@ref) object is returned.
* `size` (default: `2048`): matrices of size `(size, size)` will be used
* `verbose` (default: `true`): toggle printing of information
* `parallel` (default: `true`): If `true`, will (try to) run each GPU test on a different Julia thread. Make sure to have enough Julia threads.
* `threads` (default: `nothing`): If `parallel == true`, this argument may be used to specify the Julia threads to use.
* `clearmem` (default: `false`): If `true`, we call `clear_all_gpus_memory` after the stress test.
* `io` (default: `stdout`): set the stream where the results should be printed.

When `duration` is specifiec (i.e. `StressTestEnforced`) there is also:
* `batch_duration` (default: `ceil(Int, duration/10)`): desired duration of one batch of matmuls.
"""
stresstest(; kwargs...) = stresstest(backend(); kwargs...)
stresstest(::Backend; kwargs...) = not_implemented_yet()
