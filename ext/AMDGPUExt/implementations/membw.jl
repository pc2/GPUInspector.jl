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
        bandwidths[i] = GPUInspector.memory_bandwidth(
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

function GPUInspector.memory_bandwidth_saxpy(
    ::AMDBackend;
    device=AMDGPU.device(),
    size=2^26,
    nbench=10,
    dtype=Float32,
    verbose=true,
    io=getstdout(),
)::Float64
    device!(device) do
        a = dtype(pi)
        x = AMDGPU.rand(dtype, size)
        y = AMDGPU.rand(dtype, size)
        z = AMDGPU.zeros(dtype, size)

        kernel = @roc launch = false _saxpy_gpu_kernel!(z, a, x, y)
        occupancy = AMDGPU.launch_configuration(kernel)
        t = Inf
        for _ in 1:nbench
            Δt = AMDGPU.@elapsed @roc(
                groupsize = occupancy.groupsize, _saxpy_gpu_kernel!(z, a, x, y)
            )
            t = min(t, Δt)
        end

        bandwidth = 3.0 * sizeof(dtype) * size / t / (1024)^3
        if verbose
            printstyled(io, "Memory Bandwidth (GiB/s):\n"; bold=true)
            print(io, " └ max: ")
            printstyled(io, round(bandwidth; digits=2), "\n"; color=:green, bold=true)
        end
        return bandwidth
    end
end

function _saxpy_gpu_kernel!(z, a, x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i <= length(z)
        @inbounds z[i] = a * x[i] + y[i]
    end
    return nothing
end

function GPUInspector.memory_bandwidth_saxpy_scaling(
    ::AMDBackend;
    device=AMDGPU.device(),
    sizes=[2^20 * i for i in 10:10:300],
    verbose=true,
    io=getstdout(),
    kwargs...,
)
    # sizes = [2^20 * i for i in 8:128] # V100
    bandwidths = zeros(length(sizes))
    for (i, s) in enumerate(sizes)
        bandwidths[i] = GPUInspector.memory_bandwidth_saxpy(
            AMDBackend(); device=device, size=s, verbose=false, kwargs...
        )
        clear_gpu_memory(AMDBackend(); device=device)
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
                "Peak: ", round(peak_val; digits=2), " GiB/s (vector size = $(bytes(peak_size)))"
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
