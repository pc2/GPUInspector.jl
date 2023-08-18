# function theoretical_memory_bandwidth(
#     ::NVIDIABackend; device::CuDevice=CUDA.device(), verbose=true, io=getstdout()
# )
#     max_mem_clock_rate =
#         CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) * 1000 # in Hz
#     max_mem_bus_width =
#         CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8.0 # in bytes
#     max_bw = 2.0 * max_mem_clock_rate * max_mem_bus_width * 2^(-30)
#     if verbose
#         printstyled(io, "Theoretical Maximal Memory Bandwidth (GiB/s):\n"; bold=true)
#         print(io, " └ max: ")
#         printstyled(io, round(max_bw; digits=1), "\n"; color=:green, bold=true)
#     end
#     return max_bw
# end

function GPUInspector.memory_bandwidth(
    ::AMDBackend;
    memsize::UnitPrefixedBytes=GiB(0.5),
    dtype=Cchar,
    verbose=true,
    DtoDfactor=true,
    device=AMDGPU.device(),
    io=getstdout(),
    kwargs...,
)::Float64
    AMDGPU.device!(device) do
        N = Int(bytes(memsize) ÷ sizeof(dtype))
        mem_gpu = AMDGPU.rand(dtype, N)
        mem_gpu2 = AMDGPU.rand(dtype, N)

        return _perform_memcpy(
            mem_gpu, mem_gpu2; title="Memory", DtoDfactor, verbose, io=io, kwargs...
        )
    end
end

function GPUInspector.memory_bandwidth_scaling(
    ::AMDBackend;
    device=AMDGPU.device(),
    sizes=logspace(1, exp2(30), 10),
    verbose=true,
    io=getstdout(),
    kwargs...,
)
    bandwidths = zeros(length(sizes))
    for (i, s) in enumerate(sizes)
        bandwidths[i] = memory_bandwidth(
            AMDBackend(); memsize=B(s), device=device, verbose=false, kwargs...
        )
        clear_gpu_memory(AMDBackend(); device=device)
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
        println(io) # top margin
        println(io, p)
        println(io) # bottom margin
    end
    return (sizes=sizes, bandwidths=bandwidths)
end

# """
# Extra keyword arguments:
# * `cublas` (default: `true`): toggle between `CUDA.axpy!` and a custom `_saxpy_gpu_kernel!`.

# (This method is from the NVIDIA Backend.)
# """
# function memory_bandwidth_saxpy(
#     ::NVIDIABackend;
#     device=CUDA.device(),
#     size=2^20 * 10,
#     nbench=10,
#     dtype=Float32,
#     cublas=true,
#     verbose=true,
#     io=getstdout(),
# )::Float64
#     device!(device) do
#         a = dtype(pi)
#         x = CUDA.rand(dtype, size)
#         y = CUDA.rand(dtype, size)
#         z = CUDA.zeros(dtype, size)

#         nthreads = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
#         nblocks = cld(size, nthreads)
#         t = Inf
#         for _ in 1:nbench
#             if cublas
#                 Δt = CUDA.@elapsed CUBLAS.axpy!(size, a, x, y)
#             else
#                 Δt = CUDA.@elapsed @cuda(
#                     threads = nthreads, blocks = nblocks, _saxpy_gpu_kernel!(z, a, x, y)
#                 )
#             end
#             t = min(t, Δt)
#         end

#         bandwidth = 3.0 * sizeof(dtype) * size * (1024)^(-3) / t
#         if verbose
#             printstyled(io, "Memory Bandwidth (GiB/s):\n"; bold=true)
#             print(io, " └ max: ")
#             printstyled(io, round(bandwidth; digits=2), "\n"; color=:green, bold=true)
#         end
#         return bandwidth
#     end
# end

# function _saxpy_gpu_kernel!(z, a, x, y)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if i <= length(z)
#         @inbounds z[i] = a * x[i] + y[i]
#     end
#     return nothing
# end

# function memory_bandwidth_saxpy_scaling(
#     ::NVIDIABackend;
#     device=CUDA.device(),
#     sizes=[2^20 * i for i in 10:10:300],
#     verbose=true,
#     io=getstdout(),
#     kwargs...,
# )
#     # sizes = [2^20 * i for i in 8:128] # V100
#     bandwidths = zeros(length(sizes))
#     for (i, s) in enumerate(sizes)
#         bandwidths[i] = memory_bandwidth_saxpy(
#             NVIDIABackend(); device=device, size=s, verbose=false, kwargs...
#         )
#         clear_gpu_memory(AMDBackend(); device=device)
#     end
#     if verbose
#         peak_val, idx = findmax(bandwidths)
#         peak_size = sizes[idx]
#         p = UnicodePlots.lineplot(
#             sizes,
#             bandwidths;
#             xlabel="vector length",
#             ylabel="GiB/s",
#             title=string(
#                 "Peak: ", round(peak_val; digits=2), " GiB/s (size = $(bytes(peak_size)))"
#             ),
#             xscale=:log2,
#         )
#         UnicodePlots.lineplot!(p, [peak_size, peak_size], [0.0, peak_val]; color=:red)
#         println(io) # top margin
#         println(io, p)
#         println(io) # bottom margin
#     end
#     return (sizes=sizes, bandwidths=bandwidths)
# end
