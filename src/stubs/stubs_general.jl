"""
Check if GPUInspector, and its GPU backend (e.g. CUDA.jl), is available and functional.
If not, print some hopefully useful debug information (or turn it off with `verbose=false`).
"""
functional(; kwargs...) = functional(backend(); kwargs...)
functional(::Backend; kwargs...) = not_implemented_yet()


"""
    clear_gpu_memory([device]; kwargs...)
Reclaim the unused memory of the given GPU (default: currently active GPU).
"""
clear_gpu_memory(; kwargs...) = clear_gpu_memory(backend(); kwargs...)
clear_gpu_memory(device; kwargs...) = clear_gpu_memory(backend(), device; kwargs...)
clear_gpu_memory(::Backend, args...; kwargs...) = not_implemented_yet()


"""
    clear_all_gpus_memory([devices]; kwargs...)
Reclaim the unused memory of all available GPUs.
"""
clear_all_gpus_memory(; kwargs...) = clear_all_gpus_memory(backend(); kwargs...)
clear_all_gpus_memory(devices; kwargs...) = clear_all_gpus_memory(backend(), devices; kwargs...)
clear_all_gpus_memory(::Backend, args...; kwargs...) = not_implemented_yet()
