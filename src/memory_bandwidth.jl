"""
    theoretical_memory_bandwidth(; device::CuDevice=CUDA.device(); verbose=true)
Estimates the theoretical maximal GPU memory bandwidth in GiB/s.
"""
function theoretical_memory_bandwidth(dev::CuDevice=CUDA.device(); verbose=true)
    max_mem_clock_rate =
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) * 1000 # in Hz
    max_mem_bus_width =
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8.0 # in bytes
    max_bw = 2.0 * max_mem_clock_rate * max_mem_bus_width * 2^(-30)
    if verbose
        printstyled("Theoretical Maximal Memory Bandwidth (GiB/s):\n"; bold=true)
        print(" └ max: ")
        printstyled(round(max_bw; digits=1), "\n"; color=:green, bold=true)
    end
    return max_bw
end

"""
    memory_bandwidth([memsize; kwargs...])
Tries to estimate the peak memory bandwidth of a GPU in GiB/s by measuring the time
it takes to perform a memcpy of a certain amount of data (as specified by `memsize`).

**Keyword arguments:**
* `device` (default: `CUDA.device()`): CUDA device to be used.
* `dtype` (default: `Cchar`): element type of the vectors.
* `verbose` (default: `true`): toggle printing.

See also: [`memory_bandwidth_scaling`](@ref).
"""
function memory_bandwidth(
    memsize::UnitPrefixedBytes=GiB(0.5);
    dtype=Cchar,
    verbose=true,
    DtoDfactor=true,
    device=CUDA.device(),
    kwargs...,
)
    device!(device) do
        N = Int(bytes(memsize) ÷ sizeof(dtype))
        mem_gpu = CUDA.rand(dtype, N)
        mem_gpu2 = CUDA.rand(dtype, N)

        # if verbose
        #     gpu = device(mem_gpu)
        #     println("Memsize: $(Base.format_bytes(sizeof(mem_gpu)))")
        #     println("GPU: ", gpu, " - ", name(gpu), "\n")
        # end

        return _perform_memcpy(
            mem_gpu, mem_gpu2; title="Memory", DtoDfactor, verbose, kwargs...
        )
    end
end

"""
    memory_bandwidth_scaling() -> datasizes, bandwidths
Measures the memory bandwidth (via [`memory_bandwidth`](@ref)) as a function of data size.
If `verbose=true` (default), displays a unicode plot. Returns the considered data sizes and GiB/s.
For further options, see [`memory_bandwidth`](@ref).
"""
function memory_bandwidth_scaling(;
    device=CUDA.device(), sizes=logspace(1, exp2(30), 10), verbose=true, kwargs...
)
    bandwidths = zeros(length(sizes))
    for (i, s) in enumerate(sizes)
        bandwidths[i] = memory_bandwidth(B(s); device=device, verbose=false, kwargs...)
        clear_gpu_memory(device)
    end
    if verbose
        peak_val, idx = findmax(bandwidths)
        peak_size = sizes[idx]
        p = UnicodePlots.lineplot(
            sizes,
            bandwidths;
            xlabel="data size",
            ylabel="GiB/s",
            title=string(
                "Peak: ", round(peak_val; digits=2), " GiB/s (size = $(bytes(peak_size)))"
            ),
            xscale=:log2,
        )
        UnicodePlots.lineplot!(p, [peak_size, peak_size], [0.0, peak_val]; color=:red)
        println() # top margin
        display(p)
        println() # bottom margin
    end
    return sizes, bandwidths
end
