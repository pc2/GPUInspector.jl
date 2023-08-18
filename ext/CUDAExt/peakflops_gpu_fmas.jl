# Based on https://github.com/JuliaGPU/CUDA.jl/blob/master/examples/peakflops.jl

_kernel_fma_nfmas()::Int = 100_000
_kernel_fma_N()::Int = Int((_kernel_fma_nfmas() - 1) ÷ 3)

using CUDA: i32
"Dummy kernel doing `_kernel_fma_nfmas()` many FMAs (default: `100_000`)."
function _kernel_fma(a, b, c, out)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    @inbounds if i <= length(out)
        a_val = a[i]
        b_val = b[i]
        c_val = c[i]

        # We suggest LLVM to unroll the loop. For large values, however,
        # there might still be a loop (with a large loop body of ~1000 fmas).
        # Investigate with `@device_code_ptx @cuda kernel_fma(d_a, d_b, d_c, d_out)`.
        @unroll for j in 1:_kernel_fma_N()
            a_val = CUDA.fma(a_val, b_val, c_val)
            b_val = CUDA.fma(a_val, b_val, c_val)
            c_val = CUDA.fma(a_val, b_val, c_val)
        end

        out[i] = CUDA.fma(a_val, b_val, c_val)
    end

    return nothing
end

"""
    _peakflops_gpu_fmas(; size::Integer=5_000_000, dtype=Float32, nbench=5, nkernel=5, device=CUDA.device(), verbose=true)
Tries to estimate the peak performance of a GPU in TFLOP/s by measuring the time
it takes to perform `_kernel_fma_nfmas() * size` many FMAs on CUDA cores.

**Keyword arguments:**
* `device` (default: `CUDA.device()`): CUDA device to be used.
* `dtype` (default: `Float32`): element type of the matrices.
* `size` (default: `5_000_000`): length of vectors.
* `nkernel` (default: `5`): number of kernel calls that make up one benchmarking sample.
* `nbench` (default: `5`): number of measurements to be performed the best of which is used for the TFLOP/s computation.
* `verbose` (default: `true`): toggle printing.
* `io` (default: `stdout`): set the stream where the results should be printed.
"""
function _peakflops_gpu_fmas(;
    size::Integer=5_000_000,
    dtype=Float32,
    nbench=5,
    nkernel=5,
    device::CuDevice=CUDA.device(),
    verbose=true,
    io=getstdout(),
)
    device!(device) do
        d_a = CUDA.rand(dtype, size)
        d_b = CUDA.rand(dtype, size)
        d_c = CUDA.rand(dtype, size)
        d_out = CUDA.zeros(dtype, size)

        kernel = @cuda launch = false _kernel_fma(d_a, d_b, d_c, d_out)
        config = launch_configuration(kernel.fun)
        threads = min(size, config.threads)
        blocks = cld(size, threads)

        # warm-up
        CUDA.@elapsed kernel(d_a, d_b, d_c, d_out)

        t = Inf
        for _ in 1:nbench
            Δt = CUDA.@elapsed begin
                for _ in 1:nkernel
                    kernel(d_a, d_b, d_c, d_out; threads=threads, blocks=blocks)
                end
            end
            t = min(t, Δt)
        end
        t /= nkernel

        flopcount = 2 * _kernel_fma_nfmas() * size
        flops = (flopcount * 1e-12) / t

        if verbose
            printstyled(io, "Peakflops (TFLOP/s):\n"; bold=true)
            print(io, " └ max: ")
            printstyled(io, round(flops; digits=2), "\n"; color=:green, bold=true)
        end
        return flops
    end
end
