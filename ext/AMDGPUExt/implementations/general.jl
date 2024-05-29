function GPUInspector.functional(::AMDBackend; verbose=true)
    if AMDGPU.functional()
        verbose && @info("AMDGPU.jl is functional.")
        working = true
    else
        verbose && @info("AMDGPU.jl not functional.")
        working = false
    end
    return working
end

function GPUInspector.clear_gpu_memory(::AMDBackend; device=AMDGPU.device(), gc=true)
    device!(device) do
        gc && GC.gc()
        AMDGPU.HIP.reclaim()
    end
    return nothing
end

GPUInspector.device(::AMDBackend) = AMDGPU.device()
GPUInspector.devices(::AMDBackend) = AMDGPU.devices()
