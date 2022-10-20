module GPUInspector

# stdlibs
using Statistics
using Random
using LinearAlgebra
using Libdl
using Distributed: addprocs, rmprocs, @everywhere, workers
using Base: UUID
using Pkg: Pkg
using Logging

# external
using Requires
using Reexport
@reexport using CUDA
@reexport using ThreadPinning
using CpuId: cachesize
using HDF5: h5open
using UnicodePlots: UnicodePlots
using Glob: glob

# export BFloat16 for convenience
const BFloat16 = CUDA.BFloat16

include("UnitPrefixedBytes.jl")
include("cuda_wrappers.jl")
include("utility.jl")
include("utility_unroll.jl")
include("monitoring.jl")
include("workers.jl")
include("gpuinfo.jl")
include("p2p_bandwidth.jl")
include("host2device_bandwidth.jl")
include("stresstest_tests.jl")
include("stresstest.jl")
include("stresstest_cpu.jl")
include("peakflops_gpu.jl")
include("peakflops_gpu_matmul.jl")
include("peakflops_gpu_fmas.jl")
include("peakflops_gpu_wmmas.jl")
include("memory_bandwidth.jl")
include("memory_bandwidth_saxpy.jl")
include("hdf5.jl")

function __init__()
    @require CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0" include(
        "requires/cairomakie.jl"
    )

    if CUDA.functional()
        toggle_tensorcoremath(true; verbose=false) # by default, use CUDA.FAST_MATH
    end
end

export BFloat16
export UnitPrefixedBytes, B, KB, MB, GB, TB, KiB, MiB, GiB, TiB
export bytes, simplify, change_base, value
export get_temperatures, get_power_usages, get_gpu_utilizations
export clear_gpu_memory,
    clear_all_gpus_memory,
    cublasGemmEx_wrapper!,
    cublasGemmEx_wrapper_wrapper!,
    toggle_tensorcoremath,
    hastensorcores,
    MultiLogger,
    multi_log
export get_cpusocket_temperatures, get_cpu_utilizations, get_cpu_utilization
export logspace
export MonitoringResults,
    monitoring_start,
    monitoring_stop,
    plot_monitoring_results,
    savefig_monitoring_results,
    livemonitor_temperature,
    livemonitor_powerusage
export @worker, @worker_create, @worker_killall
export gpuinfo, gpuinfo_p2p_access, gpus
export p2p_bandwidth,
    p2p_bandwidth_all, p2p_bandwidth_bidirectional, p2p_bandwidth_bidirectional_all
export host2device_bandwidth
export stresstest, stresstest_cpu
export peakflops_gpu,
    peakflops_gpu_fmas,
    peakflops_gpu_wmmas,
    peakflops_gpu_matmul,
    peakflops_gpu_matmul_graphs,
    peakflops_gpu_matmul_scaling
export theoretical_peakflops_gpu, theoretical_peakflops_gpu_tensorcores
export memory_bandwidth,
    memory_bandwidth_saxpy,
    memory_bandwidth_scaling,
    memory_bandwidth_saxpy_scaling,
    theoretical_memory_bandwidth
export save_monitoring_results, load_monitoring_results

end
