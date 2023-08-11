ngpus(::CUDABackend) = length(CUDA.devices())

function gpus(::CUDABackend; io::IO=stdout)
    # Based on https://github.com/JuliaGPU/CUDA.jl/blob/ca77d1828f3bc0df34501de848c7a13f1df0b1fe/src/utilities.jl#L69
    devs = devices()
    if isempty(devs)
        println(io, "No CUDA-capable devices.")
    elseif length(devs) == 1
        println(io, "1 device:")
    else
        println(io, length(devs), " devices:")
    end
    for (i, dev) in enumerate(devs)
        if has_nvml()
            mig = uuid(dev) != parent_uuid(dev)
            nvml_gpu = NVML.Device(parent_uuid(dev))
            nvml_dev = NVML.Device(uuid(dev); mig)

            str = NVML.name(nvml_dev)
            cap = NVML.compute_capability(nvml_gpu)
            mem = NVML.memory_info(nvml_dev)
        else
            str = name(dev)
            cap = capability(dev)
            mem = device!(dev) do
                # this requires a device context, so we prefer NVML
                (free=available_memory(), total=total_memory())
            end
        end
        println(
            io,
            "  $(i-1): $str (sm_$(cap.major)$(cap.minor), $(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available)",
        )
    end
end

"""
    gpuinfo(deviceid::Integer)

Print out detailed information about the NVIDIA GPU with the given `deviceid`.

Heavily inspired by the CUDA sample "deviceQueryDrv.cpp".
"""
function gpuinfo(::CUDABackend, deviceid::Integer; io::IO=stdout)
    0 <= deviceid <= ngpus() - 1 || throw(ArgumentError("Invalid device id."))
    return gpuinfo(CuDevice(deviceid); io)
end
function gpuinfo(::CUDABackend, dev::CuDevice=CUDA.device(); io::IO=stdout)
    # query
    mp = nmultiprocessors(dev)
    cores = ncudacores(dev)
    max_clock_rate = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) ÷ 1000
    mem_clock_rate = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) ÷ 1000
    mem_bus_width = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    l2cachesize = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    maxTex1D = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    maxTex2D_width = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    maxTex2D_height = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    maxTex3D_width = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    maxTex3D_height = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    maxTex3D_depth = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    maxTex1DLayered_width = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
    )
    maxTex1DLayered_layers = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
    )
    maxTex2DLayered_width = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    )
    maxTex2DLayered_height = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    )
    maxTex2DLayered_layers = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    )
    total_constant_mem = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    shared_mem_per_block = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    )
    regs_per_block = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    warpsize = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    max_threads_per_mp = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
    )
    max_threads_per_block = CUDA.attribute(
        dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    )
    blockdim_x = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    blockdim_y = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    blockdim_z = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    griddim_x = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    griddim_y = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    griddim_z = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    texture_align = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
    max_mem_pitch = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_PITCH)
    async_engine_count = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)
    gpu_overlap = Bool(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP))
    kernel_exec_timeout_enabled = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
    )
    integrated = Bool(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_INTEGRATED))
    can_map_host_mem = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)
    )
    concurrent_kernels = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    )
    surface_alignment = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT) > 0
    ecc_enabled = Bool(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_ECC_ENABLED))
    unified_addressing = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
    )
    managed_memory = Bool(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY))
    compute_preemption = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
    )
    cooperative_launch = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)
    )
    cooperative_multi_dev_launch = Bool(
        CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH)
    )
    pci_domainid = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
    pci_busid = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    pci_deviceid = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    compute_mode = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
    comp_modes = [
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
    ]

    # printing
    println(io, "Device: ", name(dev), " ($dev)")
    println(
        io, "Total amount of global memory: ", Base.format_bytes(Int(CUDA.totalmem(dev)))
    )
    println(io, "Number of CUDA cores: ", cores)
    println(io, "Number of multiprocessors: ", mp, " ($(cores ÷ mp) CUDA cores each)")
    println(io, "GPU max. clock rate: ", max_clock_rate, " MHz")
    println(io, "Memory clock rate: ", mem_clock_rate, " MHz")
    println(io, "Memory bus width: ", mem_bus_width, "-bit")
    println(io, "L2 cache size: ", Base.format_bytes(l2cachesize))
    println(io, "Max. texture dimension sizes (1D): $maxTex1D")
    println(io, "Max. texture dimension sizes (2D): $maxTex2D_width, $maxTex2D_height")
    println(
        io,
        "Max. texture dimension sizes (3D): $maxTex3D_width, $maxTex3D_height, $maxTex3D_depth",
    )
    println(
        io,
        "Max. layered 1D texture size: $(maxTex1DLayered_width) ($(maxTex1DLayered_layers) layers)",
    )
    println(
        io,
        "Max. layered 2D texture size: $(maxTex2DLayered_width), $(maxTex2DLayered_height) ($(maxTex2DLayered_layers) layers)",
    )
    println(io, "Total amount of constant memory: ", Base.format_bytes(total_constant_mem))
    println(
        io,
        "Total amount of shared memory per block: ",
        Base.format_bytes(shared_mem_per_block),
    )
    println(io, "Total number of registers available per block: ", regs_per_block)
    println(io, "Warp size: ", warpsize)
    println(io, "Max. number of threads per multiprocessor: ", max_threads_per_mp)
    println(io, "Max. number of threads per block: ", max_threads_per_block)
    println(
        io,
        "Max. dimension size of a thread block (x,y,z): $(blockdim_x), $(blockdim_y), $(blockdim_z)",
    )
    println(
        io,
        "Max. dimension size of a grid size (x,y,z): $(griddim_x), $(griddim_y), $(griddim_z)",
    )
    println(io, "Texture alignment: ", Base.format_bytes(texture_align))
    println(io, "Maximum memory pitch: ", Base.format_bytes(max_mem_pitch))
    println(
        io,
        "Concurrent copy and kernel execution: ",
        gpu_overlap ? "Yes" : "No",
        " with $(async_engine_count) copy engine(s)",
    )
    println(io, "Run time limit on kernels: ", kernel_exec_timeout_enabled ? "Yes" : "No")
    println(io, "Integrated GPU sharing host memory: ", integrated ? "Yes" : "No")
    println(
        io, "Support host page-locked memory mapping: ", can_map_host_mem ? "Yes" : "No"
    )
    println(io, "Concurrent kernel execution: ", concurrent_kernels ? "Yes" : "No")
    println(io, "Alignment requirement for surfaces: ", surface_alignment ? "Yes" : "No")
    println(io, "Device has ECC support: ", ecc_enabled ? "Yes" : "No")
    println(
        io, "Device supports Unified Addressing (UVA): ", unified_addressing ? "Yes" : "No"
    )
    println(io, "Device supports managed memory: ", managed_memory ? "Yes" : "No")
    println(io, "Device supports compute preemption: ", compute_preemption ? "Yes" : "No")
    println(io, "Supports cooperative kernel launch: ", cooperative_launch ? "Yes" : "No")
    println(
        io,
        "Supports multi-device co-op kernel launch: ",
        cooperative_multi_dev_launch ? "Yes" : "No",
    )
    println(
        io,
        "Device PCI domain ID / bus ID / device ID: $(pci_domainid) / $(pci_busid) / $(pci_deviceid)",
    )
    println(io, "Compute mode: ", comp_modes[compute_mode + 1])

    return nothing
end

function gpuinfo_p2p_access(::CUDABackend; io::IO=stdout)
    # check p2p access
    ndevs = ngpus()
    if ndevs <= 1
        error("Only a single GPU available.")
    else
        mat_p2p_access_supported = Matrix{Bool}(undef, ndevs, ndevs)
        mat_p2p_can_access = Matrix{Bool}(undef, ndevs, ndevs)
        mat_p2p_atomic_supported = Matrix{Bool}(undef, ndevs, ndevs)
        for i in 1:ndevs
            dev_i = CuDevice(i - 1)
            for j in 1:ndevs
                dev_j = CuDevice(j - 1)
                if i != j
                    p2p_access_supported = Bool(
                        CUDA.p2p_attribute(
                            dev_i, dev_j, CUDA.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
                        ),
                    )
                    p2p_can_access = Bool(CUDA.can_access_peer(dev_i, dev_j))
                    p2p_atomic_supported = Bool(
                        CUDA.p2p_attribute(
                            dev_i,
                            dev_j,
                            CUDA.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                        ),
                    )
                    mat_p2p_atomic_supported[i, j] = p2p_atomic_supported
                    mat_p2p_access_supported[i, j] = p2p_access_supported
                    mat_p2p_can_access[i, j] = p2p_can_access
                    # p2p_performance_rank = CUDA.p2p_attribute(dev_i, dev_j, CUDA.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK)
                else
                    mat_p2p_atomic_supported[i, i] = false
                    mat_p2p_access_supported[i, i] = false
                    mat_p2p_can_access[i, j] = false
                end
            end
        end

        printstyled(io, "P2P Access Supported:\n"; bold=true)
        show(io, "text/plain", mat_p2p_access_supported)
        println(io)
        println(io)
        if mat_p2p_access_supported != mat_p2p_can_access
            printstyled(io, "P2P Can Access:\n"; bold=true)
            show(io, "text/plain", mat_p2p_can_access)
            println(io)
            println(io)
        end
        printstyled(io, "P2P Atomic Supported:\n"; bold=true)
        show(io, "text/plain", mat_p2p_atomic_supported)
        println(io)
        println(io)
    end
    return nothing
end
