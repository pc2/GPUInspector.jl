"""
    memory_bandwidth_saxpy_scaling() -> sizes, bandwidths
Measures the memory bandwidth (via [`memory_bandwidth_saxpy`](@ref)) as a function of vector length.
If `verbose=true` (default), displays a unicode plot. Returns the considered lengths and GiB/s.
For further options, see [`memory_bandwidth_saxpy`](@ref).
"""
function memory_bandwidth_saxpy_scaling(;
    device=CUDA.device(), sizes=[2^20 * i for i in 10:10:300], verbose=true, io::IO=stdout, kwargs...
)
    # sizes = [2^20 * i for i in 8:128] # V100
    bandwidths = zeros(length(sizes))
    for (i, s) in enumerate(sizes)
        bandwidths[i] = memory_bandwidth_saxpy(;
            device=device, size=s, verbose=false, kwargs...
        )
        clear_gpu_memory(device)
    end
    if verbose
        peak_val, idx = findmax(bandwidths)
        peak_size = sizes[idx]
        p = UnicodePlots.lineplot(
            sizes,
            bandwidths;
            xlabel="vector length",
            ylabel="GiB/s",
            title=string(
                "Peak: ", round(peak_val; digits=2), " GiB/s (size = $(bytes(peak_size)))"
            ),
            xscale=:log2,
        )
        UnicodePlots.lineplot!(p, [peak_size, peak_size], [0.0, peak_val]; color=:red)
        println(io) # top margin
        println(io,p)
        println(io) # bottom margin
    end
    return sizes, bandwidths
end

"""
Tries to estimate the peak memory bandwidth of a GPU in GiB/s by measuring the time
it takes to perform a SAXPY, i.e. `a * x[i] + y[i]`.

**Keyword arguments:**
* `device` (default: `CUDA.device()`): CUDA device to be used.
* `dtype` (default: `Float32`): element type of the vectors.
* `size` (default: `2^20 * 10`): length of the vectors.
* `nbench` (default: `5`): number of measurements to be performed the best of which is used for the GiB/s computation.
* `verbose` (default: `true`): toggle printing.
* `cublas` (default: `true`): toggle between `CUDA.axpy!` and a custom `saxpy_gpu_kernel!`.
* `io` (default: `stdout`): set the stream where the results should be printed.

See also: [`memory_bandwidth_saxpy_scaling`](@ref).
"""
function memory_bandwidth_saxpy(;
    device=CUDA.device(),
    size=2^20 * 10,
    nbench=10,
    dtype=Float32,
    cublas=true,
    verbose=true, 
    io::IO=stdout
)
    device!(device) do
        a = dtype(pi)
        x = CUDA.rand(dtype, size)
        y = CUDA.rand(dtype, size)
        z = CUDA.zeros(dtype, size)

        nthreads = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        nblocks = cld(size, nthreads)
        t = Inf
        for _ in 1:nbench
            if cublas
                Δt = CUDA.@elapsed CUBLAS.axpy!(size, a, x, y)
            else
                Δt = CUDA.@elapsed @cuda(
                    threads = nthreads, blocks = nblocks, saxpy_gpu_kernel!(z, a, x, y)
                )
            end
            t = min(t, Δt)
        end

        bandwidth = 3.0 * sizeof(dtype) * size * (1024)^(-3) / t
        if verbose
            printstyled(io,"Memory Bandwidth (GiB/s):\n"; bold=true)
            print(io," └ max: ")
            printstyled(io,round(bandwidth; digits=2), "\n"; color=:green, bold=true)
        end
        return bandwidth
    end
end

function saxpy_gpu_kernel!(z, a, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(z)
        @inbounds z[i] = a * x[i] + y[i]
    end
    return nothing
end
