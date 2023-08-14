"""
Returns the number of available GPUs for the given/current backend
"""
ngpus(; kwargs...) = ngpus(backend(); kwargs...)
ngpus(::Backend; kwargs...) = not_implemented_yet()

"""
List the available GPUs.
"""
gpus(; kwargs...) = gpus(backend(); kwargs...)
gpus(::Backend; kwargs...) = not_implemented_yet()

"""
  gpuinfo([device]; kwargs...)
Print out detailed information about the GPU with the given device id.

**Note:** Device ids start at zero!
"""
gpuinfo(; kwargs...) = gpuinfo(backend(); kwargs...)
gpuinfo(deviceid::Integer; kwargs...) = gpuinfo(backend(), deviceid; kwargs...)
gpuinfo(device; kwargs...) = gpuinfo(backend(), device; kwargs...)
gpuinfo(::Backend, device; kwargs...) = not_implemented_yet()

"""
Query peer-to-peer (i.e. inter-GPU) access support.
"""
gpuinfo_p2p_access(; kwargs...) = gpuinfo_p2p_access(backend(); kwargs...)
gpuinfo_p2p_access(::Backend; kwargs...) = not_implemented_yet()
