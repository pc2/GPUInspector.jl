function p2p_bandwidth(
    ::CUDABackend;
    memsize::UnitPrefixedBytes=B(40_000_000),
    nbench=5,
    verbose=true,
    hist=false,
    times=false,
    alternate=false,
    dtype=Float32,
    src=0,
    dst=1,
    io::IO=stdout,
)
    if ngpus() < 2
        error("At least 2 GPUs are needed for the P2P benchmark.")
    end
    mem_src, mem_dst = alloc_mem(memsize; devs=(src, dst), dtype)
    actual_memsize = sizeof(mem_src)
    ts = zeros(nbench)

    NVTX.@range "p2p: nbench loop" begin
        @inbounds for i in 1:nbench
            if mod(i, alternate ? 2 : 1) == 0
                # ts[i] = _time_cuda_elapsed(copyto!, mem_dst, mem_src)
                ts[i] = _time_cuda_elapsed(
                    (x, y) -> copyto!(x, y, CUDA.stream()), mem_dst, mem_src
                )
            else
                # ts[i] = _time_cuda_elapsed(copyto!, mem_src, mem_dst)
                ts[i] = _time_cuda_elapsed(
                    (x, y) -> copyto!(x, y, CUDA.stream()), mem_dst, mem_src
                )
            end
        end
    end

    t_min = minimum(ts)
    t_max = maximum(ts)
    t_avg = mean(ts)

    actual_memsize_GiB = actual_memsize * 2^(-30)
    bws = actual_memsize_GiB ./ ts
    bw_min = minimum(bws)
    bw_max = maximum(bws)
    bw_avg = mean(bws)

    if verbose
        # println("Memsize: $(Base.format_bytes(actual_memsize))\n")
        if times
            println(io, "t_min: $t_min")
            println(io, "t_max: $t_max")
            println(io, "t_avg: $t_avg")
        end
        printstyled(io, "Bandwidth (GiB/s):\n"; bold=true)
        print(io, " ├ max: ")
        printstyled(io, round(bw_max; digits=2), "\n"; color=:green, bold=true)
        println(io, " ├ min: ", round(bw_min; digits=2))
        println(io, " ├ avg: ", round(bw_avg; digits=2))
        print(io, " └ std_dev: ")
        printstyled(io, round(std(bws); digits=2), "\n"; color=:yellow, bold=true)
        if hist
            println(io, UnicodePlots.histogram(bws; title="Bandwidths (GiB/s)", nbins=5))
        end
    end

    return bw_max
end

function p2p_bandwidth_all(::CUDABackend; io::IO=stdout, verbose=false, kwargs...)
    ngpus = length(CUDA.devices())
    if ngpus < 2
        error("At least 2 GPUs are needed for the P2P benchmark.")
    end
    return [
        if src == dst
            nothing
        else
            p2p_bandwidth(
                CUDABackend(); src=src, dst=dst, io=io, verbose=verbose, kwargs...
            )
        end for src in 0:(ngpus - 1), dst in 0:(ngpus - 1)
    ]
end

function p2p_bandwidth_bidirectional(
    ::CUDABackend;
    memsize::UnitPrefixedBytes=B(40_000_000),
    nbench=20,
    verbose=true,
    hist=false,
    times=false,
    dtype=Float32,
    dev1=0,
    dev2=1,
    repeat=100,
    io::IO=stdout,
)
    if ngpus() < 2
        error("At least 2 GPUs are needed for the P2P benchmark.")
    end
    mem_dev1, mem_dev2 = alloc_mem(memsize; dtype, devs=(dev1, dev2))
    actual_memsize = sizeof(mem_dev1)
    ts = zeros(nbench)

    NVTX.@range "p2p: nbench loop" begin
        @inbounds for i in 1:nbench
            ts[i] = _perform_p2p_memcpy_bidirectional(
                dev1, mem_dev1, dev2, mem_dev2; repeat
            )
        end
    end

    t_min = minimum(ts)
    t_max = maximum(ts)
    t_avg = mean(ts)

    actual_memsize_GiB = (actual_memsize) / (1024^3)
    bws = (2 * actual_memsize_GiB * repeat) ./ ts
    bw_min = minimum(bws)
    bw_max = maximum(bws)
    bw_avg = mean(bws)

    if verbose
        # println("Memsize: $(Base.format_bytes(actual_memsize))\n")
        if times
            println(io, "t_min: $t_min")
            println(io, "t_max: $t_max")
            println(io, "t_avg: $t_avg")
        end
        printstyled(io, "Bandwidth (GiB/s):\n"; bold=true)
        print(io, " ├ max: ")
        printstyled(io, round(bw_max; digits=2), "\n"; color=:green, bold=true)
        println(io, " ├ min: ", round(bw_min; digits=2))
        println(io, " ├ avg: ", round(bw_avg; digits=2))
        print(io, " └ std_dev: ")
        printstyled(io, round(std(bws); digits=2), "\n"; color=:yellow, bold=true)
        if hist
            println(io, UnicodePlots.histogram(bws; title="Bandwidths (GiB/s)", nbins=5))
        end
    end

    return bw_max
end

function p2p_bandwidth_bidirectional_all(::CUDABackend; kwargs...)
    ngpus = length(CUDA.devices())
    if ngpus < 2
        error("At least 2 GPUs are needed for the P2P benchmark.")
    end
    return [
        if src == dst
            nothing
        else
            p2p_bandwidth_bidirectional(
                CUDABackend(); dev1=src, dev2=dst, verbose=false, kwargs...
            )
        end for src in 0:(ngpus - 1), dst in 0:(ngpus - 1)
    ]
end

function _perform_p2p_memcpy_bidirectional(dev1, mem_dev1, dev2, mem_dev2; repeat=5)
    NVTX.@range "p2p: perform" begin
        start = Vector{CuEvent}(undef, 2)
        stop = Vector{CuEvent}(undef, 2)

        device!(dev1)
        stream1 = CuStream()
        device!(dev2)
        stream2 = CuStream()

        for (i, devid) in pairs((dev1, dev2))
            device!(devid)
            start[i] = CuEvent()
            stop[i] = CuEvent()
        end

        device!(dev1)
        synchronize(stream1)
        synchronize(stream2)

        # Block the stream until all the work is queued up
        # delay_flag = Ref(false)
        # device!(dev1)
        # @cuda stream=stream1 _delay(delay_flag)

        # Force stream2 not to start until stream1 does, in order to ensure
        # the events on stream1 fully encompass the time needed for all
        # operations
        record(start[1], stream1)
        CUDA.wait(start[1], stream2)

        NVTX.@range "p2p: copyto!'s" begin
            for k in 1:repeat
                copyto!(mem_dev2, mem_dev1, stream2)
                copyto!(mem_dev1, mem_dev2, stream1)
            end
        end

        # Notify stream1 that stream2 is complete and record the time of
        # the total transaction
        record(stop[2], stream2)
        CUDA.wait(stop[2], stream1)
        record(stop[1], stream1)

        # Release the queued operations
        # delay_flag[] = true
        synchronize(stream1)
        synchronize(stream2)

        t = elapsed(start[1], stop[1])
    end
    return t
end

# TODO: Maybe move stuff below somewhere else?
@inline function _time_cuda_elapsed(kernel::F, mem_dst, mem_src) where {F}
    t = CUDA.context!(context(mem_src)) do
        CUDA.@elapsed begin
            NVTX.@range "p2p: kernel call" begin
                kernel(mem_dst, mem_src)
            end
        end
    end
    return t
end

## delay kernel (Warning: not sure the ref flag is working as intended!)
function _delay(delay_flag, timeout_clocks=10_000_000)
    # Wait until the application notifies us that it has completed queuing up the
    # experiment, or timeout and exit, allowing the application to make progress
    start_clock = CUDA.clock(UInt64)

    while !delay_flag[]
        sample_clock = CUDA.clock(UInt64)

        if (sample_clock - start_clock) > timeout_clocks
            break
        end
    end
end

function Base.copyto!(
    dest::DenseCuArray{T}, src::DenseCuArray{T}, stream::CuStream; async=true
) where {T}
    # based on https://github.com/JuliaGPU/CUDA.jl/blob/5413070c13be8809916900205376f6c062c55a67/src/array.jl#L421
    doffs = 1
    soffs = 1
    n = length(src)
    context!(context(src)) do
        GC.@preserve src dest begin
            unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async, stream)
            if Base.isbitsunion(T)
                unsafe_copyto!(
                    typetagdata(dest, doffs), typetagdata(src, soffs), n; async, stream
                )
            end
        end
    end
    return dest
end
