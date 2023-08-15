"""
Check if GPUInspector, and its GPU backend (e.g. CUDA.jl), is available and functional.
If not, print some hopefully useful debug information (or turn it off with `verbose=false`).
"""
functional(; kwargs...) = functional(backend(); kwargs...)
functional(::Backend; kwargs...) = not_implemented_yet()
