"""
Estimates the theoretical peak performance of a CUDA device in TFLOP/s.

**Keyword arguments:**
* `tensorcores` (default: `hastensorcores()`): toggle usage of tensore cores. If `false`, cuda cores will be used.
* `verbose` (default: `true`): toggle printing of information
* `device` (default: `device()`): CUDA device to be analyzed
* `dtype` (default: `tensorcores ? Float16 : Float32`): element type of the matrices
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
