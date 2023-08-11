module CUDAExt

using GPUInspector
using CUDA
using Statistics

# import stubs to implement them
import GPUInspector: ngpus, gpuinfo, gpuinfo_p2p_access, gpus
import GPUInspector:
    p2p_bandwidth,
    p2p_bandwidth_all,
    p2p_bandwidth_bidirectional,
    p2p_bandwidth_bidirectional_all
import GPUInspector: host2device_bandwidth

# ...
include("cuda_wrappers.jl")
include("utility.jl")
include("impl_gpuinfo.jl")
include("impl_p2p_bandwidth.jl")
include("impl_host2device_bandwidth.jl")

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
