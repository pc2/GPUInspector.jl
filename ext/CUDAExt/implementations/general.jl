function GPUInspector.functional(::NVIDIABackend; verbose=true)
    if CUDA.functional()
        verbose && @info("CUDA/GPU available.")
        hascuda = true
    else
        verbose && @info("No CUDA/GPU found.")
        hascuda = false
        if verbose
            # debug information
            @show Libdl.find_library("libcuda")
            @show filter(contains("cuda"), lowercase.(Libdl.dllist()))
            try
                @info("CUDA.versioninfo():")
                CUDA.versioninfo()
                @info("Successful!")
            catch ex
                @warn("Unsuccessful!", ex)
                println()
            end
        end
    end
    return hascuda
end

function GPUInspector.clear_gpu_memory(::NVIDIABackend; device=CUDA.device(), gc=true)
    device!(device) do
        gc && GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

GPUInspector.device(::NVIDIABackend) = CUDA.device()
GPUInspector.devices(::NVIDIABackend) = CUDA.devices()
