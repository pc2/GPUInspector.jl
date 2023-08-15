function stresstest(
    ::NVIDIABackend;
    devices=[CUDA.device()],
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
    io::IO=stdout,
    kwargs...,
)
    logger = ConsoleLogger(io)

    Base.with_logger(logger) do
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
            verbose && @info(
                "Will try to run for $(duration) seconds on each GPU, but no promises!"
            )
            ts = [
                StressTestFixedIter(dev; approx_duration, dtype, verbose, size) for
                dev in devices
            ]
        elseif !isnothing(enforced_duration)
            # StressTestEnforced
            verbose && @info(
                "Will run for almost precisely $(enforced_duration) seconds on each GPU."
            )
            ts = [
                StressTestEnforced(dev; enforced_duration, dtype, verbose, size) for
                dev in devices
            ]
        elseif !isnothing(mem)
            # StressTestStoreResults (gpu-burn-like)
            verbose && @info("Will run a `StressTestStoreResults`.")
            ts = [StressTestStoreResults(dev; mem, dtype, verbose, size) for dev in devices]
        end
        monitoring && monitoring_start(NVIDIABackend(); devices=devices, verbose)
        Δt = @elapsed _run_stresstests(ts; verbose, kwargs...)
        if clearmem
            verbose && @info("Clearing GPU memory.")
            clear_all_gpus_memory(devices)
        end
        verbose && @info("Took $(round(Δt; digits=2)) seconds to run the tests.")
        if monitoring
            results = monitoring_stop(NVIDIABackend(); verbose)
            return results
        else
            return nothing
        end
    end
end

function _stresstest_default_threads(ntests)
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
    tests; parallel=true, threads=_stresstest_default_threads(length(tests)), verbose=true
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
