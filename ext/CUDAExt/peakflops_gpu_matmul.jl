"""
    peakflops_gpu_matmul_scaling(peakflops_func = peakflops_gpu_matmul; verbose=true) -> sizes, flops
Asserts the scaling of the given `peakflops_func`tion (defaults to [`peakflops_gpu_matmul`](@ref))
with increasing matrix size. If `verbose=true` (default), displays a unicode plot. Returns
the considered sizes and TFLOP/s. For further options, see [`peakflops_gpu_matmul`](@ref).
"""
function peakflops_gpu_matmul_scaling(
    peakflops_func::F=peakflops_gpu_matmul;
    device=CUDA.device(),
    verbose=true,
    sizes=2 .^ (10:15),
    io=getstdout(),
    kwargs...,
) where {F}
    flops = zeros(length(sizes))
    for (i, s) in enumerate(sizes)
        flops[i] = peakflops_func(; device=device, size=s, verbose=false, kwargs...)
        GPUInspector.clear_gpu_memory(; device=device)
    end
    if verbose
        peak_val, idx = findmax(flops)
        peak_size = sizes[idx]
        p = UnicodePlots.lineplot(
            sizes,
            flops;
            xlabel="matrix size",
            ylabel="TFLOP/s",
            title=string(
                "Peak: ", round(peak_val; digits=2), " TFLOP/s (size = $(peak_size))"
            ),
            xscale=:log2,
        )
        UnicodePlots.lineplot!(p, [peak_size, peak_size], [0.0, peak_val]; color=:red)
        println(io) # top margin
        show(io, "text/plain", p)
        println(io) # bottom margin
        println(io) # bottom margin
    end
    return sizes, flops
end

_flopcount_per_matmul(n) = Float64(n)^3

"""
    peakflops_gpu_matmul(; device, dtype=Float32, size=2^14, nmatmuls=5, nbench=5, verbose=true)
Tries to estimate the peak performance of a GPU in TFLOP/s by measuring the time
it takes to perform `nmatmuls` many (in-place) matrix-matrix multiplications.

**Keyword arguments:**
* `device` (default: `CUDA.device()`): CUDA device to be used.
* `dtype` (default: `Float32`): element type of the matrices.
* `size` (default: `2^14`): matrices will have dimensions `(size, size)`.
* `nmatmuls` (default: `5`): number of matmuls that will make up the kernel to be timed.
* `nbench` (default: `5`): number of measurements to be performed the best of which is used for the TFLOP/s computation.
* `verbose` (default: `true`): toggle printing.
* `io` (default: `stdout`): set the stream where the results should be printed.

See also: [`peakflops_gpu_matmul_scaling`](@ref), [`peakflops_gpu_matmul_graphs`](@ref).
"""
function peakflops_gpu_matmul(;
    device=CUDA.device(),
    dtype=Float32,
    size=2^14,
    nmatmuls=5,
    nbench=5,
    verbose=true,
    io=getstdout(),
)
    device!(device) do
        C = CUDA.zeros(dtype, size, size)
        A = CUDA.rand(dtype, size, size)
        B = CUDA.rand(dtype, size, size)

        CUDA.@elapsed mul!(C, A, B) # warmup

        t = Inf
        NVTX.@range "peakflops_gpu: bench loop" begin
            for i in 1:nbench
                NVTX.@range "peakflops_gpu: kernel" begin
                    Δt = CUDA.@elapsed for _ in 1:nmatmuls
                        mul!(C, A, B)
                        # cublasGemmEx_wrapper!('N','N',A,B,C)
                    end
                end
                t = min(t, Δt)
            end
        end

        flops = (_flopcount_per_matmul(size) * nmatmuls * 1e-12) / t
        if verbose
            printstyled(io, "Peakflops (TFLOP/s):\n"; bold=true)
            print(io, " └ max: ")
            printstyled(io, round(flops; digits=2), "\n"; color=:green, bold=true)
        end
        return flops
    end
end

"""
Same as [`peakflops_gpu_matmul`](@ref) but uses CUDA's graph API to define and launch the kernel.

See also: [`peakflops_gpu_matmul_scaling`](@ref).
"""
function peakflops_gpu_matmul_graphs(;
    device=CUDA.device(),
    dtype=Float32,
    size=2^14,
    nmatmuls=5,
    nbench=5,
    verbose=true,
    io=getstdout(),
)
    device!(device) do
        C = CUDA.zeros(dtype, size, size)
        A = CUDA.rand(dtype, size, size)
        B = CUDA.rand(dtype, size, size)

        t = Inf
        for i in 1:nbench
            graph = CUDA.capture() do
                for _ in 1:nmatmuls
                    mul!(C, A, B)
                    # cublasGemmEx_wrapper!('N','N',A,B,C)
                end
            end
            exec = instantiate(graph)
            Δt = CUDA.@elapsed CUDA.launch(exec)
            t = min(t, Δt)
        end

        flops = (_flopcount_per_matmul(size) * nmatmuls * 1e-12) / t
        if verbose
            printstyled(io, "Peakflops (TFLOP/s):\n"; bold=true)
            print(io, " └ max: ")
            printstyled(io, round(flops; digits=2), "\n"; color=:green, bold=true)
        end
        return flops
    end
end
