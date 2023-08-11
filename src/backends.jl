abstract type Backend end
struct NoBackend <: Backend end
struct CUDABackend <: Backend end
struct ROCBackend <: Backend end

const BACKEND = Ref{Backend}(NoBackend())
backend() = BACKEND[]
function backend!(b::Backend)
    check_backendpkg_loaded(b)
    return BACKEND[] = b
end
function backend!(backend::Symbol)
    if backend in (:cuda, :CUDA)
        backend!(CUDABackend())
    elseif backend in (:roc, :rocm, :ROC, :ROCM, :amd, :AMD)
        backend!(ROCBackend())
    else
        throw(ArgumentError("Can't set unknown backend."))
    end
end

const CUDAJL_LOADED = Ref{Bool}(false)
const AMDGPUJL_LOADED = Ref{Bool}(false)
is_cuda_loaded() = CUDAJL_LOADED[]
is_amdgpu_loaded() = AMDGPUJL_LOADED[]
is_backend_loaded(::CUDABackend) = is_cuda_loaded()
is_backend_loaded(::ROCBackend) = is_amdgpu_loaded()
is_backend_loaded(::NoBackend) = true

_backend2jlpkg(::CUDABackend) = "CUDA.jl"
_backend2jlpkg(::ROCBackend) = "AMDGPU.jl"

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

CUDAExt::Union{Nothing, Module} = nothing
