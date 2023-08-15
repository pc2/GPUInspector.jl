module GPUInspector

# stdlibs
using Statistics
using Random
using LinearAlgebra
using Libdl
using Base: UUID
using Pkg: Pkg

# external
using Reexport
@reexport using ThreadPinning
using DocStringExtensions
using UnicodePlots
using CpuId: cachesize
using HDF5: h5open
using Glob: glob

include("backends.jl")
include("UnitPrefixedBytes.jl")
include("utility.jl")
include("utility_unroll.jl")
include("stresstest_cpu.jl")
include("monitoring.jl")
include("monitoring_io.jl")

function not_implemented_yet()
    return error(
        "Not implemented yet. You either haven't loaded a backend (like CUDA.jl) yet, or" *
        " the loaded backend doesn't provide this functionality.",
    )
end
include("stubs/stubs_general.jl")
include("stubs/stubs_gpuinfo.jl")
include("stubs/stubs_p2p_bandwidth.jl")
include("stubs/stubs_host2device_bandwidth.jl")
include("stubs/stubs_membw.jl")
include("stubs/stubs_stresstest.jl")
include("stubs/stubs_monitoring.jl")
include("stubs/stubs_peakflops_gpu.jl")

# backends
export Backend, NoBackend, NVIDIABackend, AMDBackend, backend, backend!, backendinfo
export CUDAExt

# monitoring io+plotting
export plot_monitoring_results, load_monitoring_results, save_monitoring_results

# utilities
export UnitPrefixedBytes,
    B, KB, MB, GB, TB, KiB, MiB, GiB, TiB, bytes, simplify, change_base, value
export logspace

# Let's currently not export the CPU tests. After all, this is GPUInspector.jl :)
# export stresstest_cpu

# stubs gpuinfo
export ngpus, gpuinfo, gpuinfo_p2p_access, gpus
# stubs p2p bandwidth
export p2p_bandwidth,
    p2p_bandwidth_all, p2p_bandwidth_bidirectional, p2p_bandwidth_bidirectional_all
# stubs p2p bandwidth
export host2device_bandwidth
# stubs memory bandwidth
export theoretical_memory_bandwidth,
    memory_bandwidth,
    memory_bandwidth_scaling,
    memory_bandwidth_saxpy,
    memory_bandwidth_saxpy_scaling
# stubs stresstest
export stresstest
# stubs monitoring
export MonitoringResults,
    monitoring_start,
    monitoring_stop,
    savefig_monitoring_results,
    livemonitor_powerusage,
    livemonitor_something,
    livemonitor_temperature
# stubs peakflops_gpu
export peakflops_gpu, theoretical_peakflops_gpu

end
