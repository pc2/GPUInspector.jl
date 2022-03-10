"""
    host2device_bandwidth([memsize::UnitPrefixedBytes=GiB(0.5)]; kwargs...)

Performs a host-to-device memory copy benchmark (time measurement) and
returns the host-to-device bandwidth estimate (in GiB/s) derived from it.

**Keyword arguments:**
* `nbench` (default: `10`): number of time measurements (i.e. p2p memcopies)
* `verbose` (default: `true`): set to false to turn off any printing.
* `stats` (default: `false`): when `true` shows statistical information about the benchmark.
* `times` (default: `false`): toggle printing of measured times.
* `dtype` (default: `Cchar`): used data type.

**Examples:**
```julia
host2device_bandwidth()
host2device_bandwidth(MiB(1024))
host2device_bandwidth(KiB(20_000); dtype=Int32)
```
"""
function host2device_bandwidth(
    memsize::UnitPrefixedBytes=GiB(0.5);
    dtype=Cchar,
    DtoDfactor=true,
    verbose=true,
    kwargs...,
)
    N = Int(bytes(memsize) ÷ sizeof(dtype))
    mem_host = rand(dtype, N)
    mem_host_pinned = Mem.pin(rand(dtype, N))
    mem_gpu = CUDA.rand(dtype, N)
    # mem_gpu2 = CUDA.rand(dtype, N)

    # if verbose
    #     gpu = device(mem_gpu)
    #     println("Memsize: $(Base.format_bytes(sizeof(mem_gpu)))")
    #     println("GPU: ", gpu, " - ", name(gpu), "\n")
    # end

    _perform_memcpy(mem_host, mem_gpu; title="Host <-> Device", verbose, kwargs...)
    verbose && println()
    _perform_memcpy(
        mem_host_pinned, mem_gpu; title="Host (pinned) <-> Device", verbose, kwargs...
    )
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
)
    NVTX.@range "host2dev: $title" begin
        sizeof(mem1) == sizeof(mem2) || error("sizeof(mem1) != sizeof(mem2)")
        ts = zeros(nbench)
        NVTX.@range "host2dev: bench loop" begin
            @inbounds for i in 1:nbench
                if i % 2 == 0
                    ts[i] = CUDA.@elapsed copyto!(mem1, mem2)
                else
                    ts[i] = CUDA.@elapsed copyto!(mem2, mem1)
                end
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
                println("t_min: $t_min")
                println("t_max: $t_max")
                println("t_avg: $t_avg")
            end
            printstyled("$(title) Bandwidth (GiB/s):\n"; bold=true)
            if stats
                print(" ├ max: ")
                printstyled(round(bw_max; digits=2), "\n"; color=:green, bold=true)
                println(" ├ min: ", round(bw_min; digits=2))
                println(" ├ avg: ", round(bw_avg; digits=2))
                print(" └ std_dev: ")
                printstyled(round(std(bws); digits=2), "\n"; color=:yellow, bold=true)
            else
                print(" └ max: ")
                printstyled(round(bw_max; digits=2), "\n"; color=:green, bold=true)
            end
        end
    end
    return bw_max
end
