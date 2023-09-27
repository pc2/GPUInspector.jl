"Abstract Backend for GPUInspector"
abstract type Backend end
"Indicates the absence of any GPU backends"
struct NoBackend <: Backend end
"Represents the NVIDIA backend (CUDA.jl)"
struct NVIDIABackend <: Backend end
"Represents the AMD backend (AMDGPU.jl)"
struct AMDBackend <: Backend end

const BACKEND = Ref{Backend}(NoBackend())

"Returns the currently active GPU backend"
backend() = BACKEND[]
"""
$(TYPEDSIGNATURES)
Set the GPU backend (manually). Note that the corresponding backend package (e.g. CUDA.jl)
must already be loaded in the active Julia session (otherwise an exception is thrown).
"""
function backend!(b::Backend)
    check_backendpkg_loaded(b)
    BACKEND[] = b
    return nothing
end
function backend!(b::Symbol)
    if b in (:nvidia, :NVIDIA, :cuda, :CUDA)
        backend!(NVIDIABackend())
    elseif b in (:roc, :rocm, :ROC, :ROCM, :amd, :AMD)
        backend!(AMDBackend())
    else
        throw(ArgumentError("Can't set unknown backend."))
    end
end

const CUDAJL_LOADED = Ref{Bool}(false)
const AMDGPUJL_LOADED = Ref{Bool}(false)
is_cuda_loaded() = CUDAJL_LOADED[]
is_amdgpu_loaded() = AMDGPUJL_LOADED[]
is_backend_loaded(::NVIDIABackend) = is_cuda_loaded()
is_backend_loaded(::AMDBackend) = is_amdgpu_loaded()
is_backend_loaded(::NoBackend) = true

_backend2jlpkg(::NVIDIABackend) = "CUDA.jl"
_backend2jlpkg(::AMDBackend) = "AMDGPU.jl"

function check_backendpkg_loaded(b::Backend)
    if !is_backend_loaded(b)
        error("Can't set backend because $(_backend2jlpkg(b)) doesn't seem to be loaded.")
    end
end

_check_backend(b::Backend) = backend() == b
function check_backend(b::Backend)
    if !_check_backend(b)
        error("Wrong backend. Only supported by $b.")
    end
end

CUDAExt::Union{Nothing,Module} = nothing
AMDGPUExt::Union{Nothing,Module} = nothing

"""
Query information about a specific backend, e.g., what functionality the backend currently
supports.
"""
backendinfo() = backendinfo(backend())
backendinfo(::Backend) = not_implemented_yet()
