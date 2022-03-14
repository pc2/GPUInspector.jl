module GPUInspector

# stdlibs
using Statistics
using Random
using LinearAlgebra
using Libdl
using Distributed: addprocs, rmprocs, @everywhere, workers
using Base: UUID

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
export BFloat16

include("UnitPrefixedBytes.jl")
export UnitPrefixedBytes, B, KB, MB, GB, TB, KiB, MiB, GiB, TiB
export bytes, simplify, change_base, value
include("cuda_wrappers.jl")
export get_temperatures, get_power_usages, get_gpu_utilizations
include("utility.jl")
include("utility_unroll.jl")
export clear_gpu_memory,
    clear_all_gpus_memory,
    cublasGemmEx_wrapper!,
    cublasGemmEx_wrapper_wrapper!,
    toggle_tensorcoremath,
    hastensorcores
export get_cpusocket_temperatures, get_cpu_utilizations, get_cpu_utilization
export logspace
include("monitoring.jl")
export MonitoringResults,
    monitoring_start,
    monitoring_stop,
    plot_monitoring_results,
    savefig_monitoring_results,
    livemonitor_temperature,
    livemonitor_powerusage
include("workers.jl")
export @worker, @worker_create, @worker_killall

include("gpuinfo.jl")
export gpuinfo, gpuinfo_p2p_access, gpus
include("p2p_bandwidth.jl")
export p2p_bandwidth,
    p2p_bandwidth_all, p2p_bandwidth_bidirectional, p2p_bandwidth_bidirectional_all
include("host2device_bandwidth.jl")
export host2device_bandwidth
include("stresstest_tests.jl")
include("stresstest.jl")
include("stresstest_cpu.jl")
export stresstest, stresstest_cpu
include("peakflops_gpu.jl")
include("peakflops_gpu_matmul.jl")
include("peakflops_gpu_fmas.jl")
include("peakflops_gpu_wmmas.jl")
export peakflops_gpu,
    peakflops_gpu_fmas,
    peakflops_gpu_wmmas,
    peakflops_gpu_matmul,
    peakflops_gpu_matmul_graphs,
    peakflops_gpu_matmul_scaling
export theoretical_peakflops_gpu, theoretical_peakflops_gpu_tensorcores
include("memory_bandwidth.jl")
include("memory_bandwidth_saxpy.jl")
export memory_bandwidth,
    memory_bandwidth_saxpy,
    memory_bandwidth_scaling,
    memory_bandwidth_saxpy_scaling,
    theoretical_memory_bandwidth

include("hdf5.jl")
export save_monitoring_results, load_monitoring_results

function __init__()
    @require CairoMakie="13f3f980-e62b-5c42-98c6-ff1f3baf88f0" include("requires/cairomakie.jl")

    if CUDA.functional()
        toggle_tensorcoremath(true; verbose=false) # by default, use CUDA.FAST_MATH
    end
end

end
