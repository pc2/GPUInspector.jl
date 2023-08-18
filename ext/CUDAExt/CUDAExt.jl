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
using NVTX
using ThreadPinning

# for usage in CUDAExt
using GPUInspector:
    logspace,
    ismonitoring,
    _monitoring!,
    _set_monitoring_task,
    _get_monitoring_task,
    MonitoringResults,
    _defaultylims,
    @unroll,
    NVIDIABackend,
    getstdout

# for convenience
const BFloat16 = CUDA.BFloat16

include("cuda_wrappers.jl")
include("utility.jl")
include("stresstests.jl")
include("peakflops_gpu_fmas.jl")
include("peakflops_gpu_wmmas.jl")
include("peakflops_gpu_matmul.jl")
include("implementations/general.jl")
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

function backendinfo(::NVIDIABackend)
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
    println("Implementend API functions for NVIDIABackend:")
    for f in funcs
        println("\t", f)
    end
    return nothing
end

end # module
