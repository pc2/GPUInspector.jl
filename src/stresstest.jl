"""
    stresstest(device_or_devices)

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
* `dtype` (default: `Float32`): element type of the matrices
* `monitoring` (default: `false`): enable automatic monitoring, in which case a [`MonitoringResults`](@ref) object is returned.
* `size` (default: `2048`): matrices of size `(size, size)` will be used
* `verbose` (default: `true`): toggle printing of information
* `parallel` (default: `true`): If `true`, will (try to) run each GPU test on a different Julia thread. Make sure to have enough Julia threads.
* `threads` (default: `nothing`): If `parallel == true`, this argument may be used to specify the Julia threads to use.
* `clearmem` (default: `false`): If `true`, we call [`clear_all_gpus_memory`](@ref) after the stress test.

When `duration` is specifiec (i.e. [`StressTestEnforced`](@ref)) there is also:
* `batch_duration` (default: `ceil(Int, duration/10)`): desired duration of one batch of matmuls.
"""
function stresstest(
    devices;
    mem=nothing,
    dtype=Float32,
    verbose=true,
    size=_default_matsize_stresstest(),
    duration=nothing,
    enforced_duration=nothing,
    approx_duration=nothing,
    niter=nothing,
    clearmem=false,
    monitoring=false,
    batch_duration=nothing,
    kwargs...,
)
    if eltype(devices) != CuDevice
        throw(ArgumentError("Unknown devices iterator / collection?!"))
    end
    if all(isnothing, [enforced_duration, duration, approx_duration, niter, mem])
        duration = 60 # default
    end
    if (monitoring || ismonitoring()) && Threads.nthreads() < length(devices) + 1
        ngpus = length(devices)
        error(
            "To test $ngpus GPUs while monitoring requires $(ngpus + 1) Julia threads. Only $(Threads.nthreads()) available.",
        )
    end

    if !isnothing(duration)
        # StressTestBatched
        verbose && @info("Will run for about $(duration) seconds on each GPU!")
        ts = [
            StressTestBatched(dev; duration, dtype, verbose, size, batch_duration) for
            dev in devices
        ]
    elseif !isnothing(niter)
        # StressTestFixedIter
        verbose && @info("Will run $(niter) iterations on each GPU.")
        ts = [StressTestFixedIter(dev; niter, dtype, verbose, size) for dev in devices]
    elseif !isnothing(approx_duration)
        # StressTestFixedIter
        verbose &&
            @info("Will try to run for $(duration) seconds on each GPU, but no promises!")
        ts = [
            StressTestFixedIter(dev; approx_duration, dtype, verbose, size) for
            dev in devices
        ]
    elseif !isnothing(enforced_duration)
        # StressTestEnforced
        verbose &&
            @info("Will run for almost precisely $(enforced_duration) seconds on each GPU.")
        ts = [
            StressTestEnforced(dev; enforced_duration, dtype, verbose, size) for
            dev in devices
        ]
    elseif !isnothing(mem)
        # StressTestStoreResults (gpu-burn-like)
        verbose && @info("Will run a `StressTestStoreResults`.")
        ts = [StressTestStoreResults(dev; mem, dtype, verbose, size) for dev in devices]
    end
    monitoring && monitoring_start(; devices=devices, verbose)
    Δt = @elapsed _run_stresstests(ts; verbose, kwargs...)
    if clearmem
        verbose && @info("Clearing GPU memory.")
        clear_all_gpus_memory(devices)
    end
    verbose && @info("Took $(round(Δt; digits=2)) seconds to run the tests.")
    if monitoring
        results = monitoring_stop(; verbose)
        return results
    else
        return nothing
    end
end
stresstest(device::CuDevice=CUDA.device(); kwargs...) = stresstest([device]; kwargs...)

function _default_threads(ntests)
    nthreads = Threads.nthreads()
    monitoring = ismonitoring()
    if nthreads >= ntests + monitoring + 1
        # if there are enough "worker" threads
        # use them and skip the main thread (1).
        return collect(2:(ntests + 1))
    elseif nthreads >= ntests + monitoring
        return collect(1:ntests)
    else
        # not enough threads
        @warn("There are too few Julia threads to run all tests in parallel!")
        return [mod1(i, nthreads - monitoring) for i in 1:ntests] # round-robin distribution
    end
end

function _run_stresstests(
    tests; parallel=true, threads=_default_threads(length(tests)), verbose=true
)
    if parallel == true
        if length(threads) != length(tests)
            throw(ArgumentError("length(threads) != length(tests)"))
        end
        @sync for (i, test) in enumerate(tests)
            @tspawnat threads[i] test(; verbose=verbose)
        end
    else
        for test in tests
            test(; verbose=verbose)
        end
    end
    return nothing
end

# function stresstest_on_worker(devices; verbose=true, worker=first(workers()), duration=nothing, kwargs...)
#     verbose && @info("Using worker $worker.")
#     # pid = withenv("JULIA_NUM_THREADS"=>length(devices)) do
#     #     pid, = addprocs(1, exeflags = "--project=$(Base.active_project())")
#     #     return pid
#     # end

#     t = @spawnat worker stresstest(devices; verbose=false, duration, kwargs...)
#     monitor_temperature(duration+5; liveplot=true)
#     return fetch(t)
# end
