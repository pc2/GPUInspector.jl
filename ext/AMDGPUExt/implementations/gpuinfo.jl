function GPUInspector.ngpus(::AMDBackend)
    return length(AMDGPU.devices())
end

function GPUInspector.gpus(::AMDBackend; io::IO=stdout)
    # Based on https://github.com/JuliaGPU/CUDA.jl/blob/ca77d1828f3bc0df34501de848c7a13f1df0b1fe/src/utilities.jl#L69
    devs = AMDGPU.devices()
    if isempty(devs)
        println(io, "No AMD devices found.")
    elseif length(devs) == 1
        println(io, "1 device:")
    else
        println(io, length(devs), " devices:")
    end
    for (i, dev) in enumerate(devs)
        mem_free, mem_tot = AMDGPU.device!(dev) do
            AMDGPU.Runtime.Mem.info()
        end
        println(
            io,
            "  $(_gpuid(dev)): ",
            repr(dev),
            " ($(Base.format_bytes(mem_free)) / $(Base.format_bytes(mem_tot)) available)",
        )
    end
end

"""
    gpuinfo(deviceid::Integer)

Print out detailed information about the AMD GPU with the given `deviceid`.

(This method is from the AMD backend.)
"""
function GPUInspector.gpuinfo(::AMDBackend, deviceid::Integer; io::IO=stdout)
    0 <= deviceid <= ngpus(AMDBackend()) - 1 || throw(ArgumentError("Invalid device id."))
    return gpuinfo(HIPDevice(deviceid); io)
end
function GPUInspector.gpuinfo(::AMDBackend, dev::HIPDevice=AMDGPU.device(); io::IO=stdout)
    # printing
    println(io, "Device: $dev \n")
    show(io, AMDGPU.HIP.properties(dev))
    return nothing
end

function GPUInspector.gpuinfo_p2p_access(::AMDBackend; io::IO=stdout)
    # check p2p access
    ndevs = ngpus(AMDBackend())
    if ndevs <= 1
        error("Only a single GPU available.")
    else
        devs = AMDGPU.devices()
        mat_p2p_can_access = Matrix{Bool}(undef, ndevs, ndevs)
        for i in 1:ndevs
            for j in 1:ndevs
                if i != j
                    mat_p2p_can_access[i, j] = Bool(AMDGPU.HIP.can_access_peer(devs[i], devs[j]))
                else
                    mat_p2p_can_access[i, j] = false
                end
            end
        end

        printstyled(io, "P2P Can Access:\n"; bold=true)
        show(io, "text/plain", mat_p2p_can_access)
        println(io)
        println(io)
    end
    return nothing
end
