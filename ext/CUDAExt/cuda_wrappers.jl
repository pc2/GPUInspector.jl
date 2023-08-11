function supports_get_temperature(nvml_device::NVML.Device)
    temp = Ref{UInt32}()
    nvml_return = NVML.unsafe_nvmlDeviceGetTemperature(
        nvml_device, CUDA.NVML.NVML_TEMPERATURE_GPU, temp
    )
    return nvml_return == NVML.NVML_SUCCESS
end
function supports_get_temperature(dev::CuDevice)
    return supports_get_temperature(_cudevice2nvmldevice(dev))
end

"""
    get_temperature(device=CUDA.device())
Get current temperature of the given CUDA device in degrees Celsius.
"""
function get_temperature(device::CuDevice=CUDA.device())
    return get_temperature(_cudevice2nvmldevice(device))
end
function get_temperature(nvml_device::NVML.Device)
    temp = Ref{UInt32}()
    NVML.nvmlDeviceGetTemperature(nvml_device, CUDA.NVML.NVML_TEMPERATURE_GPU, temp)
    return Int(temp[])
end

"""
    get_temperatures(devices=CUDA.devices())
Get current temperature of the given CUDA devices in degrees Celsius.
"""
get_temperatures(devices=CUDA.devices()) = [get_temperature(dev) for dev in devices]

function supports_get_power_usage(nvml_device::NVML.Device)
    power = Ref{UInt32}()
    nvml_return = NVML.unsafe_nvmlDeviceGetPowerUsage(nvml_device, power)
    return nvml_return == NVML.NVML_SUCCESS
end
function supports_get_power_usage(dev::CuDevice)
    return supports_get_power_usage(_cudevice2nvmldevice(dev))
end

"""
    get_power_usage(device=CUDA.device())
Get current power usage of the given CUDA device in Watts.
"""
function get_power_usage(nvml_device::NVML.Device)
    power = Ref{UInt32}()
    NVML.nvmlDeviceGetPowerUsage(nvml_device, power)
    return round(power[] * 1e-3; digits=2)
end
get_power_usage(dev::CuDevice=CUDA.device()) = get_power_usage(_cudevice2nvmldevice(dev))

"""
    get_power_usages(devices=CUDA.devices())
Get current power usage of the given CUDA devices in Watts.
"""
get_power_usages(devices=CUDA.devices()) = [get_power_usage(dev) for dev in devices]

function supports_get_gpu_utilization(nvml_device::NVML.Device)
    util = Ref{NVML.nvmlUtilization_t}()
    nvml_return = NVML.unsafe_nvmlDeviceGetUtilizationRates(nvml_device, util)
    return nvml_return == NVML.NVML_SUCCESS
end
function supports_get_gpu_utilization(dev::CuDevice)
    return supports_get_gpu_utilization(_cudevice2nvmldevice(dev))
end

"""
    get_gpu_utilization(device=CUDA.device())
Get the current utilization of the given CUDA device in percent.
"""
function get_gpu_utilization(device::CuDevice=CUDA.device())
    return get_gpu_utilization(_cudevice2nvmldevice(device))
end
function get_gpu_utilization(nvml_device::NVML.Device)
    util = Ref{NVML.nvmlUtilization_t}()
    NVML.nvmlDeviceGetUtilizationRates(nvml_device, util)
    return (compute=Int(util[].gpu), mem=Int(util[].memory))
end

"""
    get_gpu_utilizations(devices=CUDA.devices())
Get the current utilization of the given CUDA devices in percent.
"""
get_gpu_utilizations(devices=CUDA.devices()) = [get_gpu_utilization(dev) for dev in devices]

# function get_cublas_math_mode(; handle=CUBLAS.handle())
#     # https://docs.nvidia.com/cuda/cublas/#cublasmath_t
#     # https://github.com/JuliaGPU/CUDA.jl/blob/64c5ca8f9d3b90e2c1420e39d0a252265f07b548/lib/cublas/CUBLAS.jl#L34
#     x = Ref{UInt32}(0)
#     CUBLAS.cublasGetMathMode(handle, x)
#     return x[]
# end

"""
Get GPU index of the given device.

**Note:** GPU indices start with zero.
"""
gpuid(dev::CuDevice=CUDA.device()) = dev.handle # TODO: Better way?
function gpuid(nvml_device::NVML.Device)
    idx = Ref{UInt32}()
    NVML.nvmlDeviceGetIndex(nvml_device, idx)
    return Int(idx[])
end

function _cudevice2nvmldevice(device::CuDevice)
    # TODO: Better way?
    for ndev in NVML.devices()
        if gpuid(ndev) == gpuid(device)
            return ndev
        end
    end
    return error("Couldn't find device...")
end

function _nvmldevice2cudevice(nvml_device::NVML.Device)
    # TODO: Better way?
    for cudev in CUDA.devices()
        if gpuid(cudev) == gpuid(nvml_device)
            return cudev
        end
    end
    return error("Couldn't find device...")
end

function cublasGemmEx_wrapper_wrapper!(;
    dtype::DataType,
    computeType::Union{Nothing,CUBLAS.cublasComputeType_t},
    size=2048,
    verbose=false,
)
    A = CUDA.rand(dtype, size, size)
    B = CUDA.rand(dtype, size, size)
    C = CUDA.zeros(dtype, size, size)
    return CUDA.@elapsed cublasGemmEx_wrapper!('N', 'N', A, B, C; computeType, verbose)
end

function cublasGemmEx_wrapper!(
    transA::Char,
    transB::Char,
    A::StridedCuVecOrMat,
    B::StridedCuVecOrMat,
    C::StridedCuVecOrMat;
    algo::CUBLAS.cublasGemmAlgo_t=CUBLAS.CUBLAS_GEMM_DEFAULT,
    computeType::Union{Nothing,CUBLAS.cublasComputeType_t}=nothing,
    verbose=false,
)
    alpha = true
    beta = false
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch(""))
    end
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldc = max(1, stride(C, 2))

    if isnothing(computeType)
        computeType = CUBLAS.gemmExComputeType(eltype(A), eltype(B), eltype(C), m, k, n)
        isnothing(computeType) && throw(
            ArgumentError("gemmEx does not support $(eltype(C))=$(eltype(A))*$(eltype(B))"),
        )
    end
    computeT = CUBLAS.juliaStorageType(eltype(C), computeType)
    if CUDA.version() < v"11.0"
        # with CUDA 11, the compute type encodes the math mode.
        # before CUDA 11, it was a plain cudaDataType.
        computeType = convert(CUDA.cudaDataType, computeT)
    end
    if verbose
        @show computeType
        @show computeT
    end
    CUBLAS.cublasGemmEx(
        CUBLAS.handle(),
        transA,
        transB,
        m,
        n,
        k,
        Ref{computeT}(alpha),
        A,
        eltype(A),
        lda,
        B,
        eltype(B),
        ldb,
        Ref{computeT}(beta),
        C,
        eltype(C),
        ldc,
        computeType,
        algo,
    )
    return C
end
