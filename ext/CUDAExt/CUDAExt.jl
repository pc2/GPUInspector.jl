module CUDAExt

using GPUInspector
using CUDA

# stdlibs etc.
using Base: UUID
using Statistics
using Logging
using LinearAlgebra

# pkgs
using UnicodePlots

# for usage in CUDAExt
using GPUInspector:
    logspace,
    ismonitoring,
    _monitoring!,
    _set_monitoring_task,
    _get_monitoring_task,
    MonitoringResults,
    _defaultylims,
    @unroll

# import stubs to implement them
import GPUInspector: backendinfo
# gpuinfo
import GPUInspector: ngpus, gpuinfo, gpuinfo_p2p_access, gpus
# p2p bw
import GPUInspector:
    p2p_bandwidth,
    p2p_bandwidth_all,
    p2p_bandwidth_bidirectional,
    p2p_bandwidth_bidirectional_all
# host2device bw
import GPUInspector: host2device_bandwidth
# membw
import GPUInspector:
    theoretical_memory_bandwidth,
    memory_bandwidth,
    memory_bandwidth_scaling,
    memory_bandwidth_saxpy,
    memory_bandwidth_saxpy_scaling
# stresstest
import GPUInspector: stresstest
# monitoring
import GPUInspector:
    monitoring_start,
    monitoring_stop,
    livemonitor_something,
    livemonitor_powerusage,
    livemonitor_temperature
# peakflops_gpu
import GPUInspector: peakflops_gpu, theoretical_peakflops_gpu

# for convenience
const BFloat16 = CUDA.BFloat16

include("cuda_wrappers.jl")
include("utility.jl")
include("stresstests.jl")
include("peakflops_gpu_fmas.jl")
include("peakflops_gpu_wmmas.jl")
include("peakflops_gpu_matmul.jl")
include("implementations/gpuinfo.jl")
include("implementations/p2p_bandwidth.jl")
include("implementations/host2device_bandwidth.jl")
include("implementations/membw.jl")
include("implementations/stresstest.jl")
include("implementations/monitoring.jl")
include("implementations/peakflops_gpu.jl")

function __init__()
    GPUInspector.CUDAJL_LOADED[] = true
    GPUInspector.backend!(:cuda)
    GPUInspector.CUDAExt = Base.get_extension(GPUInspector, :CUDAExt)

    # by default, use CUDA.FAST_MATH
    if CUDA.functional()
        toggle_tensorcoremath(true; verbose=false)
    end
    return nothing
end

function backendinfo(::CUDABackend)
    # somewhat crude way to figure out which API functions are implemented :)
    funcs = String[]
    impl_dir = joinpath(@__DIR__, "implementations/")
    for f in readdir(impl_dir)
        lines = readlines(joinpath(impl_dir, f))
        func_lines = filter(startswith("function"), lines)
        for fl in func_lines
            fname = strip(split(split(fl, "function")[2], "(")[1])
            if startswith(fname, "_") || startswith(fname, "Base")
                continue
            end
            if fname in funcs # avoid duplicates
                continue
            end
            push!(funcs, fname)
        end
    end
    println("Implementend API functions for CUDABackend:")
    for f in funcs
        println("\t", f)
    end
    return nothing
end

end # module
