"""
Check if GPUInspector, and its GPU backend (e.g. CUDA.jl), is available and functional.
If not, print some hopefully useful debug information (or turn it off with `verbose=false`).
"""
functional(; kwargs...) = functional(backend(); kwargs...)
functional(::Backend; kwargs...) = not_implemented_yet()

"""
    clear_gpu_memory(; device, gc)

Reclaim the unused memory of a GPU
"""
clear_gpu_memory(; kwargs...) = clear_gpu_memory(backend(); kwargs...)
clear_gpu_memory(::Backend; kwargs...) = not_implemented_yet()

"Return the current device of the active backend."
device() = device(backend())
device(::Backend) = not_implemented_yet()

"Return the devices of the active backend."
devices() = devices(backend())
devices(::Backend) = not_implemented_yet()
