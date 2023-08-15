function functional(::NVIDIABackend; verbose=true)
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
