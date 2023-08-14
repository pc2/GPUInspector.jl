module CUDAExt

using GPUInspector
using GPUInspector: logspace
using CUDA
using Statistics
using UnicodePlots

# import stubs to implement them
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

# ...
include("cuda_wrappers.jl")
include("utility.jl")
include("implementations/gpuinfo.jl")
include("implementations/p2p_bandwidth.jl")
include("implementations/host2device_bandwidth.jl")
include("implementations/membw.jl")

# export BFloat16 for convenience
const BFloat16 = CUDA.BFloat16

function __init__()
    if CUDA.functional()
        toggle_tensorcoremath(true; verbose=false) # by default, use CUDA.FAST_MATH
    end
    GPUInspector.CUDAJL_LOADED[] = true
    GPUInspector.backend!(:cuda)
    return GPUInspector.CUDAExt = Base.get_extension(GPUInspector, :CUDAExt)
end

end
