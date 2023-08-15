"""
    theoretical_memory_bandwidth(; device, verbose)
Estimates the theoretical maximal GPU memory bandwidth in GiB/s.
"""
function theoretical_memory_bandwidth(; kwargs...)
    return theoretical_memory_bandwidth(backend(); kwargs...)
end
theoretical_memory_bandwidth(::Backend; kwargs...) = not_implemented_yet()

"""
    memory_bandwidth(; kwargs...)
Tries to estimate the peak memory bandwidth of a GPU in GiB/s by measuring the time
it takes to perform a memcpy of a certain amount of data (as specified by `memsize`).

**Keyword arguments:**
* `memsize` (default: `GiB(0.5)`): memory size to be used
* `device` (default: e.g. `CUDA.device()`): GPU device to be used.
* `dtype` (default: `Cchar`): element type of the vectors.
* `verbose` (default: `true`): toggle printing.
* `io` (default: `stdout`): set the stream where the results should be printed.

See also: [`memory_bandwidth_scaling`](@ref).
"""
memory_bandwidth(; kwargs...) = memory_bandwidth(backend(); kwargs...)
memory_bandwidth(::Backend; kwargs...) = not_implemented_yet()

"""
    memory_bandwidth_scaling() -> datasizes, bandwidths
Measures the memory bandwidth (via [`memory_bandwidth`](@ref)) as a function of data size.
If `verbose=true` (default), displays a unicode plot. Returns the considered data sizes and GiB/s.
For further options, see [`memory_bandwidth`](@ref).
"""
memory_bandwidth_scaling(; kwargs...) = memory_bandwidth_scaling(backend(); kwargs...)
memory_bandwidth_scaling(::Backend; kwargs...) = not_implemented_yet()

"""
Tries to estimate the peak memory bandwidth of a GPU in GiB/s by measuring the time
it takes to perform a SAXPY, i.e. `a * x[i] + y[i]`.

**Keyword arguments:**
* `device` (default: e.g. `CUDA.device()`): GPU device to be used.
* `dtype` (default: `Float32`): element type of the vectors.
* `size` (default: `2^20 * 10`): length of the vectors.
* `nbench` (default: `5`): number of measurements to be performed the best of which is used for the GiB/s computation.
* `verbose` (default: `true`): toggle printing.
* `io` (default: `stdout`): set the stream where the results should be printed.

See also: [`memory_bandwidth_saxpy_scaling`](@ref).
"""
memory_bandwidth_saxpy(; kwargs...) = memory_bandwidth_saxpy(backend(); kwargs...)
memory_bandwidth_saxpy(::Backend; kwargs...) = not_implemented_yet()

"""
    memory_bandwidth_saxpy_scaling() -> sizes, bandwidths
Measures the memory bandwidth (via [`memory_bandwidth_saxpy`](@ref)) as a function of vector length.
If `verbose=true` (default), displays a unicode plot. Returns the considered lengths and GiB/s.
For further options, see [`memory_bandwidth_saxpy`](@ref).
"""
function memory_bandwidth_saxpy_scaling(; kwargs...)
    return memory_bandwidth_saxpy_scaling(backend(); kwargs...)
end
memory_bandwidth_saxpy_scaling(::Backend; kwargs...) = not_implemented_yet()
