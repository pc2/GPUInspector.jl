function GPUInspector.host2device_bandwidth(
    ::AMDBackend;
    memsize::UnitPrefixedBytes=GiB(0.5),
    dtype=Cchar,
    DtoDfactor=true,
    verbose=true,
    io=getstdout(),
    kwargs...,
)
    N = Int(bytes(memsize) ÷ sizeof(dtype))
    mem_host = rand(dtype, N)
    # mem_host_pinned = Mem.pin(rand(dtype, N)) # TODO
    mem_gpu = AMDGPU.rand(dtype, N)

    _perform_memcpy(mem_host, mem_gpu; title="Host <-> Device", verbose, io=io, kwargs...)
    verbose && println(io)
    # _perform_memcpy(
    #     mem_host_pinned,
    #     mem_gpu;
    #     title="Host (pinned) <-> Device",
    #     verbose,
    #     io=io,
    #     kwargs...,
    # )
    # verbose && println()
    # _perform_memcpy(mem_gpu, mem_gpu2; title="Device <-> Device (same device)", DtoDfactor, verbose, kwargs...)
    return nothing
end

function _perform_memcpy(
    mem1,
    mem2;
    title="",
    nbench=10,
    times=false,
    stats=false,
    DtoDfactor=false,
    verbose=true,
    io=getstdout(),
)
    sizeof(mem1) == sizeof(mem2) || error("sizeof(mem1) != sizeof(mem2)")
    ts = zeros(nbench)

    @inbounds for i in 1:nbench
        if i % 2 == 0
            ts[i] = AMDGPU.@elapsed copyto!(mem1, mem2)
        else
            ts[i] = AMDGPU.@elapsed copyto!(mem2, mem1)
        end
    end

    t_min = minimum(ts)
    t_max = maximum(ts)
    t_avg = mean(ts)

    actual_memsize_GiB = sizeof(mem1) * 2^(-30)
    if DtoDfactor
        actual_memsize_GiB *= 2 # must count both the read and the write here (taken from p2pBandwidthLatencyTest cuda sample....)
    end
    bws = actual_memsize_GiB ./ ts
    bw_min = minimum(bws)
    bw_max = maximum(bws)
    bw_avg = mean(bws)

    if verbose
        if times
            println(io, "t_min: $t_min")
            println(io, "t_max: $t_max")
            println(io, "t_avg: $t_avg")
        end
        printstyled(io, "$(title) Bandwidth (GiB/s):\n"; bold=true)
        if stats
            print(io, " ├ max: ")
            printstyled(io, round(bw_max; digits=2), "\n"; color=:green, bold=true)
            println(io, " ├ min: ", round(bw_min; digits=2))
            println(io, " ├ avg: ", round(bw_avg; digits=2))
            print(io, " └ std_dev: ")
            printstyled(io, round(std(bws); digits=2), "\n"; color=:yellow, bold=true)
        else
            print(io, " └ max: ")
            printstyled(io, round(bw_max; digits=2), "\n"; color=:green, bold=true)
        end
    end
    return bw_max
end
