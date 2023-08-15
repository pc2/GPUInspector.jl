_stresstest_cpu_size_default(dtype) = floor(Int, sqrt(L2_cachesize() / sizeof(dtype)))

"""
    stresstest_cpu(core_or_cores)

Run a CPU stress test (matrix multiplication) on one or multiple CPU cores, as specified by the positional argument.
If no argument is provided (only) the currently active CPU core will be used.

**Keyword arguments:**

* `duration`: stress test will take about the given time in seconds.
* `dtype` (default: `Float64`): element type of the matrices
* `size` (default: `floor(Int, sqrt(L2_cachesize() / sizeof(dtype)))`): matrices of size `(size, size)` will be used
* `verbose` (default: `true`): toggle printing of information
* `parallel` (default: `true`): If `true`, will (try to) run each CPU core test on a different Julia thread. Make sure to have enough Julia threads.
* `threads` (default: `nothing`): If `parallel == true`, this argument may be used to specify the Julia threads to use.
"""
function stresstest_cpu(
    cores;
    duration=2,
    dtype=Float64,
    size=_stresstest_cpu_size_default(dtype),
    verbose=true,
    kwargs...,
)
    if !all(c -> 0 ≤ c < Sys.CPU_THREADS, cores)
        throw(ArgumentError("CPU core IDs must all be ≥ 0 and ≤ Sys.CPU_THREADS."))
    end
    if BLAS.get_num_threads() != 1 && verbose
        @info("Setting BLAS.set_num_threads(1)!")
        BLAS.set_num_threads(1)
    end
    verbose && @info("Will run for about $(duration) seconds on each CPU core!")
    Δt = @elapsed _run_stresstests_cpu(cores; verbose, duration, size, kwargs...)
    verbose && @info("Took $(round(Δt; digits=2)) seconds to run the tests.")
    return nothing
end
stresstest_cpu(core::Integer=getcpuid(); kwargs...) = stresstest_cpu([core]; kwargs...)

function _stresstest_cpu_default_threads(ncores)
    nthreads = Threads.nthreads()
    if nthreads >= ncores + 1
        # if there are enough "worker" threads
        # use them and skip the main thread (1).
        return collect(2:(ncores + 1))
    elseif nthreads == ncores
        return collect(1:ncores)
    elseif nthreads < ncores
        # not enough threads
        @warn("There are too few Julia threads to run all tests in parallel!")
        return [mod1(i, nthreads) for i in 1:ncores] # round-robin distribution
    end
end

function _run_stresstests_cpu(
    cores;
    parallel=true,
    threads=_stresstest_cpu_default_threads(length(cores)),
    verbose=true,
    kwargs...,
)
    if parallel == true
        if length(threads) != length(cores)
            throw(ArgumentError("length(threads) != length(cores)"))
        end
        @sync for (i, core) in enumerate(cores)
            if verbose
                @info("Julia thread $(threads[i]) runs test on CPU core $(core).")
            end
            @tspawnat threads[i] begin
                core_before = getcpuid()
                pinthread(core)
                _stresstest_cpu_kernel(; verbose, kwargs...)
                pinthread(core_before)
            end
        end
    else
        for core in cores
            core_before = getcpuid()
            pinthread(core)
            _stresstest_cpu_kernel(; verbose, kwargs...)
            pinthread(core_before)
        end
    end
    return nothing
end

function _stresstest_cpu_kernel(;
    duration, dtype=Float64, size=_stresstest_cpu_size_default(dtype), verbose=true
)
    A = rand(dtype, size, size)
    B = rand(dtype, size, size)
    C = zeros(dtype, size, size)

    t_start = time()
    while (time() - t_start) < duration
        mul!(C, A, B)
    end
    return nothing
end
