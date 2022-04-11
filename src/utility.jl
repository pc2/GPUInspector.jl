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

"Reclaim the unused memory of the currently active GPU (i.e. `device()`)."
function clear_gpu_memory(device::CuDevice=CUDA.device(); gc=true)
    device!(device) do
        gc && GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

"Reclaim the unused memory of all available GPUs."
function clear_all_gpus_memory(devices=CUDA.devices(); gc=true)
    gc && GC.gc()
    for dev in devices
        clear_gpu_memory(dev; gc=false)
    end
    return nothing
end

"""
Check if CUDA/GPU is available and functional.
If not, print some (hopefully useful) debug information.
"""
function functional(verbose=true)
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
                @warn("Unsuccessful!")
                @warn(ex)
                println()
            end
        end
    end
    return hascuda
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

L2_cachesize() = cachesize()[2]

"""
Tries to get the temperatures of the available CPUs (sockets not cores) in degrees Celsius.

Based on `cat /sys/class/thermal/thermal_zone*/temp`.
"""
function get_cpusocket_temperatures()
    temp_strs = split(
        read(`cat $(glob("thermal_zone*", "/sys/class/thermal"))/temp`, String)
    )
    return parse.(Float64, temp_strs) ./ 1000
end

"""
Get information about all cpu cores. Returns a vector of vectors. The outer index corresponds
to cpu cores. The inner vector contains the following information (in that order):

`user nice system idle iowait irq softirq steal guest ?`

See [proc(5)](https://man7.org/linux/man-pages/man5/proc.5.html) for more information.
"""
function get_cpu_stats()
    st = read("/proc/stat", String)
    cpulines = filter(l -> startswith(l, "cpu") && !startswith(l, "cpu "), split(st, '\n'))
    stats = Vector{Vector{Int64}}(undef, length(cpulines))
    for (i, line) in enumerate(cpulines)
        stats[i] = @views parse.(Int64, split(line)[2:end])
    end
    # user nice system idle iowait irq softirq steal guest ?
    return stats
end

"""
    get_cpu_utilizations(cores=0:Sys.CPU_THREADS-1; Δt=0.01)
Get the utilization (in percent) of the given cpu `cores` over a certain time interval `Δt`.

Based on [this](https://www.idnt.net/en-US/kb/941772).
"""
function get_cpu_utilizations(cores=0:(Sys.CPU_THREADS - 1); Δt=0.01)
    stats1 = get_cpu_stats()
    sleep(Δt)
    stats2 = get_cpu_stats()
    coreidcs = cores .+ 1 # + 1 for coreids -> coreindex mapping
    idles1, totals1 = _stats2idles_and_totals(stats1[coreidcs])
    idles2, totals2 = _stats2idles_and_totals(stats2[coreidcs])
    Δtotals = totals2 .- totals1
    Δidles = idles2 .- idles1
    return 100.0 .* (Δtotals .- Δidles) ./ Δtotals
end

"""
    get_cpu_utilization(core=getcpuid(); Δt=0.01)
Get the utilization (in percent) of the given cpu `core` over a certain time interval `Δt`.
"""
function get_cpu_utilization(core=getcpuid(); kwargs...)
    return only(get_cpu_utilizations([core]; kwargs...))
end

function _stats2idles_and_totals(stats=get_cpu_stats())
    idles = zeros(length(stats))
    totals = zeros(length(stats))
    for i in eachindex(stats)
        vals = stats[i]
        idles[i] = vals[4] + vals[5] # idle + iowait
        nonidle = vals[1] + vals[2] + vals[3] + vals[6] + vals[7] + vals[8] # user nice system irq softirq steal
        totals[i] = idles[i] + nonidle
    end
    return idles, totals
end

"""
Checks whether the given `CuDevice` has Tensor Cores.
"""
function hastensorcores(dev::CuDevice=CUDA.device())
    cap = CUDA.capability(dev)
    if cap == v"8.0.0"
        # A100
        return true
    elseif cap == v"7.0.0"
        # V100
        return true
    else
        return false
    end
end

function logspace(start, stop, length)
    return exp2.(range(log2(start), log2(stop); length=length))
end

function pkgversion(pkg::Module)
    return Pkg.Types.read_project(joinpath(Base.pkgdir(pkg), "Project.toml")).version
end


"""
MultiLogger struct which combines normal and error output streams. 
Useful if you want your normal and error output that is printed to the terminal to also be saved to different files.

**Arguments:**
* `normal_file_name`: Path to normal output file.
* `error_file_name`: Path to error output file.
"""
struct MultiLogger
    normal_io::IO
    error_io::IO
    normal_logger::ConsoleLogger
    error_logger::ConsoleLogger
    
    function MultiLogger(normal_file_name::String, error_file_name::String)
        normal_io = open(normal_file_name, "w+")
        normal_logger = ConsoleLogger(normal_io)

        error_io = open(error_file_name, "w+")
        error_logger = ConsoleLogger(error_io)

        new(normal_io, error_io, normal_logger, error_logger)
    end
end

"""
Logging function for MultiLogger. Use this in combination with MultiLogger if you want 
your normal and error output that is printed to the terminal to also be saved to different files.

**Arguments:**
* `MultiLogger`: Instance of MultiLogger struct.
* `text`: Text to be printed to terminal and written to file.
* `is_error` (default: `false`): Flag which decides whether `text` should be written into normal or error stream.
"""
function multi_log(multilogger::MultiLogger, text::String, is_error::Bool=false)

      @info(text)
      println("")

      if !is_error
          with_logger(multilogger.normal_logger) do
            @info(text)
            println("")
          end
          flush(multilogger.normal_io)
      else
          with_logger(multilogger.error_logger) do
            @error(text)
            println("")
          end
          flush(multilogger.error_io)
      end
end