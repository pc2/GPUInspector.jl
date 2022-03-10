# ------------------------- Theoretical -------------------------
"""
Estimates the theoretical peak performance of a CUDA device in TFLOP/s.

**Keyword arguments:**
* `tensorcores` (default: `hastensorcores()`): toggle usage of tensore cores. If `false`, cuda cores will be used.
* `verbose` (default: `true`): toggle printing of information
* `device` (default: `device()`): CUDA device to be analyzed
* `dtype` (default: `tensorcores ? Float16 : Float32`): element type of the matrices
"""
function theoretical_peakflops_gpu(;
    device=CUDA.device(),
    tensorcores=hastensorcores(),
    dtype=tensorcores ? Float16 : Float32,
    verbose=true,
)
    if tensorcores
        max_peakflops = _theoretical_peakflops_gpu_tensorcores(; device, dtype)
    else
        max_peakflops = _theoretical_peakflops_gpu_cudacores(; device, dtype)
    end

    if verbose
        printstyled(
            "Theoretical Peakflops ($(Symbol(dtype) == :Int8 ? "TOP" : "TFLOP")/s):\n";
            bold=true,
        )
        if hastensorcores()
            print(" ├ tensorcores: ")
            printstyled(tensorcores, "\n"; color=:magenta, bold=true)
        end
        print(" ├ dtype: ")
        printstyled(Symbol(dtype), "\n"; color=:yellow, bold=true)
        print(" └ max: ")
        printstyled(round(max_peakflops; digits=1), "\n"; color=:green, bold=true)
    end
    return max_peakflops
end

function _theoretical_peakflops_gpu_cudacores(; device, dtype)
    max_clock_rate = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) # in kHz
    num_cuda_cores = ncudacores(device)
    max_peakflops = max_clock_rate * num_cuda_cores * 1e-9 # in TFLOP/s
    if dtype == Float32
        max_peakflops *= 2
    elseif dtype == Float64
        max_peakflops *= 1
    else
        throw(ArgumentError("Unsupported dtype."))
    end
    return max_peakflops
end

function _theoretical_peakflops_gpu_tensorcores(;
    device=CUDA.device(), dtype=Float16, verbose=true
)
    cap = CUDA.capability(device)
    if cap == v"8.0.0"
        devtype = :A100
    elseif cap == v"7.0.0"
        devtype = :V100
    else
        error("Unsupported compute capability / device generation.")
    end
    max_clock_rate = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) # in kHz
    num_tensor_cores = ntensorcores(device)
    max_peakflops = max_clock_rate * num_tensor_cores * 1e-9 # in TFLOP/s
    if devtype == :A100
        if Symbol(dtype) == :Float16
            # matrix dimensions 8x8x4, factor 2 for nflops in A*B+C
            # see e.g. https://peerj.com/articles/cs-330.pdf
            max_peakflops *= 2 * 8 * 8 * 4
        elseif Symbol(dtype) in (:Float32, :TensorFloat32, :TF32)
            max_peakflops *= 2 * 4 * 8 * 4
        elseif Symbol(dtype) == :Float64
            max_peakflops *= 2 * 4 * 2 * 2
        elseif Symbol(dtype) == :Int8
            max_peakflops *= 2 * 2 * 8 * 8 * 4
        else
            throw(ArgumentError("Unsupported dtype."))
        end
    elseif devtype == :V100
        if Symbol(dtype) == :Float16
            max_peakflops *= 2 * 4 * 4 * 4
        else
            throw(ArgumentError("Unsupported dtype."))
        end
    end
    return max_peakflops
end

# ------------------------- Empirical -------------------------
"""
    peakflops_gpu(; tensorcores=hastensorcores(), kwargs...)
Tries to estimate the peak performance of a GPU in TFLOP/s by measuring the time
it takes to perform

* `_kernel_fma_nfmas() * size` many FMAs on CUDA cores (if `tensorcores == false`)
* `_kernel_wmma_nwmmas()` many WMMAs on Tensor Cores (if `tensorcores == true`)

For more keyword argument options see [`peakflops_gpu_fmas`](@ref) and [`peakflops_gpu_wmmas`](@ref).
"""
function peakflops_gpu(;
    tensorcores=hastensorcores(),
    verbose=true,
    dtype=tensorcores ? Float16 : Float32,
    kwargs...,
)
    if tensorcores
        flops = peakflops_gpu_wmmas(; verbose=false, dtype, kwargs...)
    else
        flops = peakflops_gpu_fmas(; verbose=false, dtype, kwargs...)
    end
    if verbose
        printstyled(
            "Peakflops ($(Symbol(dtype) == :Int8 ? "TOP" : "TFLOP")/s):\n"; bold=true
        )
        if hastensorcores()
            print(" ├ tensorcores: ")
            printstyled(tensorcores, "\n"; color=:magenta, bold=true)
        end
        print(" ├ dtype: ")
        printstyled(Symbol(dtype), "\n"; color=:yellow, bold=true)
        print(" └ max: ")
        printstyled(round(flops; digits=1), "\n"; color=:green, bold=true)
    end
    return flops
end
