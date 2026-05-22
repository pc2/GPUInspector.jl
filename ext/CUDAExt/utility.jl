"""
    alloc_mem(memsize::UnitPrefixedBytes; devs=(CUDA.device(),), dtype=Float32)
Allocates memory on the devices whose IDs are provided via `devs`.
Returns a vector of memory handles (i.e. `CuArray`s).

**Examples:**
```julia
alloc_mem(MiB(1024)) # allocate on the currently active device
alloc_mem(B(40_000_000); devs=(0,1)) # allocate on GPU0 and GPU1
```
"""
function alloc_mem(memsize::UnitPrefixedBytes; devs=(CUDA.device(),), dtype=Float32)
    all(map(isequal, devs, unique(devs))) ||
        throw(ArgumentError("No duplicates in `devs` allowed."))
    N = Int(bytes(memsize) ÷ sizeof(dtype))

    mem_handles = Vector{CuArray{dtype,1,CUDA.Mem.DeviceBuffer}}(undef, length(devs))
    for (i, dev) in pairs(devs)
        device!(dev)
        mem_handles[i] = CUDA.rand(N)
    end
    return mem_handles
end

"""
    toggle_tensorcoremath([enable::Bool; verbose=true])
Switches the `CUDA.math_mode` between `CUDA.FAST_MATH` (`enable=true`) and `CUDA.DEFAULT_MATH` (`enable=false`).
For matmuls of `CuArray{Float32}`s, this should have the effect of using/enabling and not using/disabling tensor cores.
Of course, this only works on supported devices and CUDA versions.

If no arguments are provided, this functions toggles between the two math modes.
"""
function toggle_tensorcoremath(
    enable::Bool=CUDA.math_mode() == CUDA.DEFAULT_MATH; verbose=true
)
    if enable
        CUDA.math_mode!(CUDA.FAST_MATH)
        verbose && @info(
            "Tensor cores should be ACTIVE for matmuls (only involving `Float32` elements).",
            CUDA.math_mode()
        )
    else
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
        verbose && @info(
            "Tensor cores should be NOT ACTIVE for matmuls (only involving `Float32` elements).",
            CUDA.math_mode()
        )
    end
    return nothing
end

_device2string(dev::CuDevice) = "GPU $(gpuid(dev)): $(CUDA.name(dev))"
_device2string(dev::NVML.Device) = _device2string(_nvmldevice2cudevice(dev))

"""
Checks whether the given `CuDevice` has Tensor Cores.
"""
hastensorcores(dev::CuDevice=CUDA.device()) = ntensorcores(dev) > 0

function nmultiprocessors(dev::CuDevice=CUDA.device())
    return CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
end
ncudacores(deviceid::Integer) = ncudacores(CuDevice(deviceid))
function ncudacores(device::CuDevice=CUDA.device())
    capver = CUDA.capability(device)
    return ncudacores(capver.major, capver.minor, nmultiprocessors(device))
end
function ncudacores(major, minor, mp)
    # based on https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
    # helper_cuda_drvapi provides something like https://github.com/LinkedInAttic/datacl/blob/master/approxalgos/GPU_Work_Final2/bussAnal/filter/lib/helper_cuda_drvapi.h#L82 but is header only
    cores = 0
    err_msg = "Unknown device type / compute capability version (major $major, minor $minor)"
    if major == 2 # Fermi
        if minor == 1
            cores = mp * 48
        else
            cores = mp * 32
        end
    elseif major == 3 # Kepler
        cores = mp * 192
    elseif major == 5 # Maxwell
        cores = mp * 128
    elseif major == 6 # Pascal
        if (minor == 1) || (minor == 2)
            cores = mp * 128
        elseif minor == 0
            cores = mp * 64
        else
            error(err_msg)
        end
    elseif major == 7 # Volta and Turing
        if (minor == 0) || (minor == 5)
            cores = mp * 64
        else
            error(err_msg)
        end
    elseif major == 8 # Ampere and Ada Lovelace
        if minor == 0
            cores = mp * 64
        elseif minor == 6
            cores = mp * 128
        elseif minor == 9
            cores = mp * 128
        else
            error(err_msg)
        end
    elseif major == 9 # Hopper
        if minor == 0
            cores = mp * 128
        else
            error(err_msg)
        end
    else
        error(err_msg)
    end
    return cores
end
function ntensorcores(device::CuDevice=CUDA.device())
    capver = CUDA.capability(device)
    return ntensorcores(capver.major, capver.minor, nmultiprocessors(device))
end
function ntensorcores(major, minor, mp)
    # based on https://en.wikipedia.org/wiki/CUDA
    err_msg = "Unknown device type / compute capability version (major $major, minor $minor)"
    if major == 7
        if minor in (0, 2, 5)
            return 8 * mp
        else
            error(err_msg)
        end
    elseif major == 8 # Ampere and Ada Lovelace
        if minor in (0, 6, 7, 9)
            return 4 * mp
        else
            error(err_msg)
        end
    elseif major == 9 # Hopper
        if minor == 0
            return 4 * mp
        else
            error(err_msg)
        end
    elseif major < 7
        return 0
    else
        error(err_msg)
    end
end
