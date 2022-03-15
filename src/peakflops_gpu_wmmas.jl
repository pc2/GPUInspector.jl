_kernel_wmma_nwmmas()::Int = 10_000

function kernel_wmma_f16(a_dev, b_dev, c_dev, d_dev)
    conf16 = WMMA.Config{16,16,16,Float16}
    conf32 = WMMA.Config{16,16,16,Float32}

    a_frag = WMMA.load_a(pointer(a_dev), 16, WMMA.RowMajor, conf16)
    b_frag = WMMA.load_b(pointer(b_dev), 16, WMMA.ColMajor, conf16)
    c_frag = WMMA.load_c(pointer(c_dev), 16, WMMA.RowMajor, conf32)

    for _ in 1:_kernel_wmma_nwmmas()
        c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf32)
    end

    WMMA.store_d(pointer(d_dev), c_frag, 16, WMMA.RowMajor, conf32)
    return nothing
end

function kernel_wmma_f16_lowlevel(a_dev, b_dev, c_dev, d_dev)
    a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_global_stride_f16(pointer(a_dev), 16)
    b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_global_stride_f16(pointer(b_dev), 16)
    c_frag = WMMA.llvm_wmma_load_c_col_m16n16k16_global_stride_f32(pointer(c_dev), 16)

    for _ in 1:_kernel_wmma_nwmmas()
        c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_f32_f32(a_frag, b_frag, c_frag)
    end

    WMMA.llvm_wmma_store_d_col_m16n16k16_global_stride_f32(pointer(d_dev), c_frag, 16)
    return nothing
end

function kernel_wmma_int8_lowlevel(a_dev, b_dev, c_dev, d_dev)
    a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_global_stride_s8(pointer(a_dev), 16)
    b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_global_stride_s8(pointer(b_dev), 16)
    c_frag = WMMA.llvm_wmma_load_c_col_m16n16k16_global_stride_s32(pointer(c_dev), 16)

    for _ in 1:_kernel_wmma_nwmmas()
        c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_s8(a_frag, b_frag, c_frag)
    end

    WMMA.llvm_wmma_store_d_col_m16n16k16_global_stride_s32(pointer(d_dev), c_frag, 16)
    return nothing
end

function kernel_wmma_tf32_lowlevel(a_dev, b_dev, c_dev, d_dev)
    a_frag = WMMA.llvm_wmma_load_a_col_m16n16k8_global_stride_tf32(pointer(a_dev), 16)
    b_frag = WMMA.llvm_wmma_load_b_col_m16n16k8_global_stride_tf32(pointer(b_dev), 8)
    c_frag = WMMA.llvm_wmma_load_c_col_m16n16k8_global_stride_f32(pointer(c_dev), 16)

    for _ in 1:_kernel_wmma_nwmmas()
        c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k8_tf32(a_frag, b_frag, c_frag)
    end

    WMMA.llvm_wmma_store_d_col_m16n16k8_global_stride_f32(pointer(d_dev), c_frag, 16)
    return nothing
end

function kernel_wmma_bf16_lowlevel(a_dev, b_dev, c_dev, d_dev)
    a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_global_stride_bf16(pointer(a_dev), 16)
    b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_global_stride_bf16(pointer(b_dev), 16)
    c_frag = WMMA.llvm_wmma_load_c_col_m16n16k16_global_stride_f32(pointer(c_dev), 16)

    for _ in 1:_kernel_wmma_nwmmas()
        c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_bf16(a_frag, b_frag, c_frag)
    end

    WMMA.llvm_wmma_store_d_col_m16n16k16_global_stride_f32(pointer(d_dev), c_frag, 16)
    return nothing
end

"""
    peakflops_gpu_wmmas()
Tries to estimate the peak performance of a GPU in TFLOP/s by measuring the time
it takes to perform `_kernel_wmma_nwmmas()` many WMMAs on Tensor Cores.

**Keyword arguments:**
* `device` (default: `CUDA.device()`): CUDA device to be used.
* `dtype` (default: `Float16`): element type of the matrices. We currently only support `Float16` (`Int8`, `:TensorFloat32`, `:BFloat16`, and `Float64` might or might not work).
* `nkernel` (default: `10`): number of kernel calls that make up one benchmarking sample.
* `nbench` (default: `5`): number of measurements to be performed the best of which is used for the TFLOP/s computation.
* `threads` (default: max. threads per block): how many threads to use per block (part of the kernel launch configuration).
* `blocks` (default: `2048`): how many blocks to use (part of the kernel launch configuration).
* `verbose` (default: `true`): toggle printing.
"""
function peakflops_gpu_wmmas(;
    device::CuDevice=CUDA.device(),
    blocks=2048,
    threads=CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
    nbench=5,
    nkernel=10,
    verbose=true,
    dtype=Float16,
)
    device!(device) do
        if Symbol(dtype) == :Float16
            m = n = k = 16
            dtype_a = dtype_b = Float16
            dtype_c = dtype_d = Float32
        elseif Symbol(dtype) == :Float16_lowlevel
            m = n = k = 16
            dtype_a = dtype_b = Float16
            dtype_c = dtype_d = Float32
        elseif Symbol(dtype) == :Int8
            if pkgversion(CUDA) < v"3.8.6"
                error("At the time of writing, CUDA#master is required for Int8 WMMA support.")
            end
            m = n = k = 16
            dtype_a = dtype_b = Int8
            dtype_c = dtype_d = Int32
        elseif Symbol(dtype) in (:Float32, :TensorFloat32, :TF32)
            # requires CUDA.jl PR 1419
            if VERSION < v"1.8.0-"
                error("Julia >= 1.8.0 required for `:TensorFloat32`.")
            end
            @warn(
                "Not officially supported by GPUInspector.jl (yet). Expect errors! Requires https://github.com/JuliaGPU/CUDA.jl/pull/1419."
            )
            m = n = 16
            k = 8
            dtype_a = dtype_b = Float32
            dtype_c = dtype_d = Float32
        elseif Symbol(dtype) == :Float64
            # requires CUDA.jl PR 1426
            if VERSION < v"1.8.0-"
                error("Julia >= 1.8.0 required for `Float64`.")
            end
            @warn(
                "Not officially supported by GPUInspector.jl (yet). Expect errors! Requires https://github.com/JuliaGPU/CUDA.jl/pull/1426."
            )
            m = n = 8
            k = 4
            dtype_a = dtype_b = Float64
            dtype_c = dtype_d = Float64
        elseif Symbol(dtype) in (:BFloat16, :BF16)
            # requires CUDA.jl PR 1425
            if VERSION < v"1.8.0-"
                error("Julia >= 1.8.0 required for `:BFloat16`.")
            end
            @warn(
                "Not officially supported by GPUInspector.jl (yet). Expect errors! Requires https://github.com/JuliaGPU/CUDA.jl/pull/1425."
            )
            m = n = k = 16
            dtype_a = dtype_b = BFloat16
            dtype_c = dtype_d = Float32
        else
            throw(ArgumentError("Unsupported dtype."))
        end
        d_a = CUDA.rand(dtype_a, m, k)
        d_b = CUDA.rand(dtype_b, k, n)
        d_c = CUDA.rand(dtype_c, m, n)
        d_d = CUDA.zeros(dtype_d, m, n)

        if Symbol(dtype) == :Float16
            kernel = @cuda launch = false kernel_wmma_f16(d_a, d_b, d_c, d_d)
        elseif Symbol(dtype) == :Float16_lowlevel
            kernel = @cuda launch = false kernel_wmma_f16_lowlevel(d_a, d_b, d_c, d_d)
        elseif Symbol(dtype) == :Int8
            kernel = @cuda launch = false kernel_wmma_int8_lowlevel(d_a, d_b, d_c, d_d)
        elseif Symbol(dtype) in (:Float32, :TensorFloat32, :TF32)
            kernel = @cuda launch = false kernel_wmma_tf32_lowlevel(d_a, d_b, d_c, d_d)
        elseif Symbol(dtype) in (:BFloat16, :BF16)
            kernel = @cuda launch = false kernel_wmma_bf16_lowlevel(d_a, d_b, d_c, d_d)
        else
            throw(ArgumentError("Unsupported dtype."))
        end
        warpsize = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        # @show threads
        # @show blocks
        # @show warpsize

        # warm-up
        CUDA.@elapsed kernel(d_a, d_b, d_c, d_d; threads=threads, blocks=blocks)

        t = Inf
        for _ in 1:nbench
            Δt = CUDA.@elapsed begin
                for _ in 1:nkernel
                    # kernel(d_a, d_b, d_c, d_d; threads=threads, blocks=blocks)
                    kernel(d_a, d_b, d_c, d_d; threads=threads, blocks=blocks)
                end
            end
            t = min(t, Δt)
        end
        t /= nkernel

        nwarps = threads / warpsize
        flopcount = 2.0 * m * n * k * blocks * nwarps * _kernel_wmma_nwmmas()
        flops = (flopcount * 1e-12) / t

        if verbose
            printstyled(
                "Peakflops ($(Symbol(dtype) == :Int8 ? "TOP" : "TFLOP")/s):\n"; bold=true
            )
            print(" └ max: ")
            printstyled(round(flops; digits=2), "\n"; color=:green, bold=true)
        end
        return flops
    end
end
