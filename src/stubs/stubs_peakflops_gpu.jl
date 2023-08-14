"""
Estimates the theoretical peak performance of a GPU device in TFLOP/s.

**Keyword arguments:**
* `verbose` (default: `true`): toggle printing of information
* `device` (default: e.g. `CUDA.device()`): GPU device to be analyzed
* `dtype` (default: `Float32`): element type of the matrices
* `io` (default: `stdout`): set the stream where the results should be printed.
"""
theoretical_peakflops_gpu(; kwargs...) = theoretical_peakflops_gpu(backend(); kwargs...)
theoretical_peakflops_gpu(::Backend; kwargs...) = not_implemented_yet()

"""
    peakflops_gpu(; kwargs...)
Tries to estimate the peak performance of a GPU in TFLOP/s
"""
peakflops_gpu(; kwargs...) = peakflops_gpu(backend(); kwargs...)
peakflops_gpu(::Backend; kwargs...) = not_implemented_yet()
