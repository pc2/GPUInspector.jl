# --------- Defaults & Helpers
_default_matsize_stresstest() = 2048 # linear size of matrices; 2048 x 2048 should be efficiently implemented in CUBLAS
_default_use_mem_stresstest() = 0.9 # use 90% of available (i.e. free) memory

function _estimate_t_matmul(C, A, B)
    t_matmul = device!(device(C)) do
        nsamples = 5
        CUDA.@elapsed mul!(C, A, B) # warmup
        t_matmul = Inf
        for _ in 1:nsamples
            t_sample = CUDA.@elapsed for _ in 1:5
                mul!(C, A, B)
            end
            t_matmul = min(t_sample / 5, t_matmul)
        end
        # CUDA.@sync mul!(C, A, B) # warmup
        # t_matmul = Inf
        # for _ in 1:nsamples
        #     t_sample = @elapsed CUDA.@sync for _ in 1:5 mul!(C, A, B) end
        #     t_matmul = min(t_sample/5, t_matmul)
        # end
        return t_matmul
    end
    return t_matmul
end

# --------- AbstractStressTest
abstract type AbstractStressTest{T} end

_eltype(::Type{AbstractStressTest{T}}) where {T} = T
_eltype(::AbstractStressTest{T}) where {T} = T

# --------- StressTestBatched
"""
GPU stress test (matrix multiplications) in which we try to run for a given time period.
We try to keep the CUDA stream continously busy with matmuls at any point in time.
Concretely, we submit batches of matmuls and, after half of them, we record a CUDA event.
On the host, after submitting a batch, we (non-blockingly) synchronize on, i.e. wait for, the CUDA event
and, if we haven't exceeded the desired duration already, submit another batch.
"""
struct StressTestBatched{T,S} <: AbstractStressTest{T}
    device::CuDevice
    C::S
    A::S
    B::S
    duration::Int64
    size::Int64
    batchsize::Int64
    batch_duration::Int64

    function StressTestBatched(
        device::CuDevice=device();
        duration=60,
        batch_duration=nothing,
        dtype=Float32,
        verbose=true,
        size=_default_matsize_stresstest(),
    )
        C, A, B = device!(device) do
            C = CUDA.zeros(dtype, size, size)
            A = CUDA.rand(dtype, size, size)
            B = CUDA.rand(dtype, size, size)
            return C, A, B
        end
        if isnothing(batch_duration)
            batch_duration = ceil(Int, duration / 10) # reasonable default?
        end
        t_matmul = _estimate_t_matmul(C, A, B)
        batchsize = ceil(Int, batch_duration / t_matmul)
        verbose && @info(
            "Chose batch size such that a batch takes approx. $(batch_duration) seconds.",
            t_matmul,
            batchsize,
        )
        return new{dtype,typeof(A)}(
            device, C, A, B, duration, size, batchsize, batch_duration
        )
    end
end

function (st::StressTestBatched)(; verbose=false)
    verbose && @info(
        "Running StressTestBatched{$(_eltype(st))} on Julia thread $(Threads.threadid()) and $(st.device)."
    )
    counter = device!(st.device) do
        C, A, B, duration, batchsize = st.C, st.A, st.B, st.duration, st.batchsize
        event_halfbatch = CuEvent()
        i_record = Int(batchsize ÷ 2)
        counter = 0
        t = time()
        CUDA.@sync @inbounds while ((time() - t) < duration)
            for i in 1:batchsize
                mul!(C, A, B)
                counter += 1
                i == i_record && CUDA.record(event_halfbatch)
            end
            synchronize(event_halfbatch)
            # alternative:
            # while !CUDA.isdone(event_halfbatch)
            #     nothing
            #     # println("not done.")
            # end
            # println("Batch half done.")
        end
        return counter
    end
    verbose && @info("Ran $counter iterations on $(st.device).")
    return nothing
end

# --------- StressTestEnforced
"""
GPU stress test (matrix multiplications) in which we run almost precisely for a given time period (duration is enforced).
"""
struct StressTestEnforced{T,S} <: AbstractStressTest{T}
    device::CuDevice
    C::S
    A::S
    B::S
    enforced_duration::Int64
    size::Int64

    function StressTestEnforced(
        device::CuDevice=device();
        enforced_duration=5,
        dtype=Float32,
        verbose=true,
        size=_default_matsize_stresstest(),
    )
        C, A, B = device!(device) do
            C = CUDA.zeros(dtype, size, size)
            A = CUDA.rand(dtype, size, size)
            B = CUDA.rand(dtype, size, size)
            return C, A, B
        end
        return new{dtype,typeof(A)}(device, C, A, B, enforced_duration, size)
    end
end

function (st::StressTestEnforced)(; verbose=false)
    verbose && @info(
        "Running StressTestEnforced{$(_eltype(st))} on Julia thread $(Threads.threadid()) and $(st.device)."
    )
    counter = 0
    device!(st.device) do
        C, A, B, enforced_duration = st.C, st.A, st.B, st.enforced_duration
        t = time()
        @inbounds while (time() - t) < enforced_duration
            CUDA.@sync mul!(C, A, B)
            counter += 1
        end
    end
    verbose && @info("Ran $counter iterations on $(st.device).")
    return nothing
end

# --------- StressTestStoreResults (gpu-burn-like)
"""
GPU stress test (matrix multiplications) in which we store all matmul results and try to
run as many iterations as possible for a certain memory limit (default: 90% of free memory).

This stress test is somewhat inspired by [gpu-burn](https://github.com/wilicc/gpu-burn) by Ville Timonen.
"""
struct StressTestStoreResults{T,S} <: AbstractStressTest{T}
    device::CuDevice
    C::Vector{S}
    A::S
    B::S
    niters::Int64
    size::Int64

    function StressTestStoreResults(
        device::CuDevice=device();
        mem=nothing,
        dtype=Float32,
        verbose=true,
        size=_default_matsize_stresstest(),
    )
        max_use_bytes = device!(() -> _memarg2maxusebytes(mem), device)
        mat_sizeof = sizeof(dtype) * size * size
        niters = Int((max_use_bytes - 2 * mat_sizeof) ÷ mat_sizeof) # we substract sizeof A and B
        use_bytes = niters * mat_sizeof
        if verbose
            mem_free, mem_total = device!(CUDA.Mem.info, device)
            @info(
                "Allocating $(bytes(use_bytes)) of memory on $(device) for results. ($(bytes(mem_free)) free, $(bytes(mem_total)) total)"
            )
            @info("Will be able to run $niters iterations.")
        end
        if use_bytes < 3 * mat_sizeof
            error("Low memory for result. Aborting.")
        end
        C, A, B = device!(device) do
            C = [CUDA.zeros(dtype, size, size) for _ in 1:niters]
            A = CUDA.rand(dtype, size, size)
            B = CUDA.rand(dtype, size, size)
            return C, A, B
        end
        return new{dtype,typeof(A)}(device, C, A, B, niters, size)
    end
end

function (st::StressTestStoreResults)(; verbose=false)
    verbose && @info(
        "Running StressTestStoreResults{$(_eltype(st))} on Julia thread $(Threads.threadid()) and $(st.device)."
    )
    device!(st.device) do
        C, A, B, niters = st.C, st.A, st.B, st.niters
        @inbounds for i in 1:niters
            mul!(C[i], A, B)
        end
    end
    return nothing
end

_containertype(::Type{StressTestStoreResults{T,S}}) where {T,S} = S
_containertype(::StressTestStoreResults{T,S}) where {T,S} = S

_memarg2maxusebytes(mem::Nothing) = CUDA.available_memory() * _default_use_mem_stresstest()
_memarg2maxusebytes(mem::UnitPrefixedBytes) = bytes(mem)
function _memarg2maxusebytes(mem::Real)
    if 0 < mem < 1
        return CUDA.available_memory() * mem
    else
        throw(
            ArgumentError(
                "Provided `mem` argument is < 0 or > 1 and thus not a valid fraction."
            ),
        )
    end
end

# --------- StressTestFixedIter
"""
GPU stress test (matrix multiplications) in which we run for a given number of iteration,
or try to run for a given time period (with potentially high uncertainty!).
In the latter case, we estimate how long a synced matmul takes and set `niter` accordingly.
"""
struct StressTestFixedIter{T,S} <: AbstractStressTest{T}
    device::CuDevice
    C::S
    A::S
    B::S
    size::Int64
    niter::Int64

    function StressTestFixedIter(
        device::CuDevice=device();
        approx_duration=60,
        niter=nothing,
        dtype=Float32,
        verbose=true,
        size=_default_matsize_stresstest(),
    )
        C, A, B = device!(device) do
            C = CUDA.zeros(dtype, size, size)
            A = CUDA.rand(dtype, size, size)
            B = CUDA.rand(dtype, size, size)
            return C, A, B
        end
        if isnothing(niter)
            t_matmul = _estimate_t_matmul(C, A, B)
            niter = ceil(Int, approx_duration / t_matmul)
            verbose && @info(
                "Estimated required number of iterations for desired duration.",
                t_matmul,
                niter
            )
        end
        return new{dtype,typeof(A)}(device, C, A, B, size, niter)
    end
end

function (st::StressTestFixedIter)(; verbose=false)
    verbose && @info(
        "Running StressTestFixedIter{$(_eltype(st))} on Julia thread $(Threads.threadid()) and $(st.device)."
    )
    device!(st.device) do
        C, A, B, niter = st.C, st.A, st.B, st.niter
        CUDA.@sync @inbounds for _ in 1:niter
            mul!(C, A, B)
        end
    end
    verbose && @info("Ran $(st.niter) iterations on $(st.device).")
    return nothing
end
