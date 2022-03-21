const MONITORING = Ref{Bool}(false)
const MONITORING_TASK = Ref{Union{Nothing,Task}}(nothing)

"Checks if we are currently monitoring."
ismonitoring() = MONITORING[]
_monitoring!(val::Bool) = MONITORING[] = val
_set_monitoring_task(t::Task) = MONITORING_TASK[] = t
_get_monitoring_task() = MONITORING_TASK[]

struct MonitoringResults
    times::Vector{Float64}
    devices::Vector{Tuple{String,UUID}}
    results::Dict{Symbol,Vector{Vector{Float64}}}
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, r::MonitoringResults)
    println(io, "MonitoringResults")
    return print(io, "└ Quantities: ", collect(keys(r.results)))
end
function Base.show(io::IO, r::MonitoringResults)
    return print(io, "MonitoringResults(", collect(keys(r.results)), ")")
end

function Base.getproperty(r::MonitoringResults, s::Symbol)
    if hasfield(typeof(r), s)
        return getfield(r, s)
    else
        return r.results[s]
    end
end

function Base.:(==)(r1::MonitoringResults, r2::MonitoringResults)
    equal = true
    for field in fieldnames(MonitoringResults)
        if getfield(r1, field) != getfield(r2, field)
            equal = false
        end
    end
    return equal
end

"""
    monitoring_start(; devices=CUDA.devices(), verbose=true)
Start monitoring of GPU temperature, utilization, power usage, etc.

**Keyword arguments:**
* `freq` (default: `1`): polling rate in Hz.
* `devices` (default: `CUDA.devices()`): `CuDevice`s or `NVML.Device`s to monitor.
* `thread` (default: `Threads.nthreads()`): id of the Julia thread that should run the monitoring.
* `verbose` (default: `true`): toggle verbose output.

See also [`monitoring_stop`](@ref).
"""
function monitoring_start(;
    freq=1, devices=CUDA.devices(), thread=Threads.nthreads(), verbose=true
)
    if ismonitoring()
        error("We are already monitoring.")
    end
    if thread <= 0 || thread > Threads.nthreads()
        throw(
            ArgumentError(
                "Invalid value for `thread` (must be `1 ≤ thread ≤ Threads.nthreads()`)."
            ),
        )
    end
    if Threads.nthreads() < length(devices) + 1
        @warn(
            "Notice that there aren't enough Julia threads to test all available GPUs and monitoring them at the same time."
        )
    end
    if eltype(devices) == CuDevice
        devs = _cudevice2nvmldevice.(devices)
    else
        devs = devices
    end
    verbose && @info("Spawning monitoring on Julia thread $thread.")
    # check which quantities we can monitor
    func_symbol_vec = Tuple{Function,Symbol}[]
    if all(supports_get_temperature(dev) for dev in devs)
        push!(func_symbol_vec, (get_temperatures, :temperature))
    end
    if all(supports_get_power_usage(dev) for dev in devs)
        push!(func_symbol_vec, (get_power_usages, :power))
    end
    if all(supports_get_gpu_utilization(dev) for dev in devs)
        push!(
            func_symbol_vec,
            (devs -> getproperty.(get_gpu_utilizations(devs), :compute), :compute),
        )
        push!(
            func_symbol_vec, (devs -> getproperty.(get_gpu_utilizations(devs), :mem), :mem)
        )
    end

    # spawn monitoring
    _monitoring!(true)
    t = @tspawnat thread _monitor(func_symbol_vec; freq, devices=devs)
    _set_monitoring_task(t)
    return nothing
end

function _monitor(func_symbol_vec; freq, devices)
    nfuncs = length(func_symbol_vec)
    t_freq = 1.0 / freq
    # init result dict
    d = Dict{Symbol,Vector{Vector{Float64}}}()
    for i in 1:nfuncs
        f, s = func_symbol_vec[i]
        d[s] = Vector{Float64}[f(devices)]
    end
    times = [0.0]
    # monitor
    t = time()
    while ismonitoring()
        sleep(t_freq)
        push!(times, time() - t)
        for i in 1:nfuncs
            f, s = func_symbol_vec[i]
            push!(d[s], f(devices))
        end
    end
    device_info = Vector{Tuple{String,UUID}}(undef, length(devices))
    for (i, nvmldev) in enumerate(devices)
        dev = _nvmldevice2cudevice(nvmldev)
        device_info[i] = (_device2string(dev), uuid(dev))
    end
    return MonitoringResults(times, device_info, d)
end

"""
    monitoring_stop(; verbose=true) -> results
Stops the GPU monitoring and returns the measured values.
Specifically, `results` is a named tuple with the following keys:
* `time`: the (relative) times at which we measured
* `temperature`, `power`, `compute`, `mem`

See also [`monitoring_start`](@ref) and [`plot_monitoring_results`](@ref).
"""
function monitoring_stop(; verbose=true)::MonitoringResults
    if ismonitoring()
        verbose && @info("Stopping monitoring and fetching results...")
        _monitoring!(false)
        results = fetch(_get_monitoring_task())
        return results
    else
        error("We aren't monitoring, so can't stop it...")
    end
end

"""
    plot_monitoring_results(r::MonitoringResults, symbols=keys(r.results))
Plot the quantities specified through `symbols` of a `MonitoringResults` object.
Will generate a textual in-terminal / in-logfile plot using UnicodePlots.jl.
"""
function plot_monitoring_results(r::MonitoringResults, symbols=keys(r.results))
    for s in symbols
        display(plot_monitoring_results(r, s))
    end
    return nothing
end

function plot_monitoring_results(r::MonitoringResults, s::Symbol)
    println() # top margin
    times = r.times
    values = r.results[s]
    title, ylabel = _symbol2title_and_label(s)
    ylims = _defaultylims(values)
    device_labels = [str for (str, uuid) in r.devices]

    p = UnicodePlots.lineplot(
        times,
        getindex.(values, 1);
        title=title,
        xlabel="Time [s]",
        ylabel=ylabel,
        name=device_labels[1],
        ylim=ylims,
    )
    for i in 2:length(first(values))
        UnicodePlots.lineplot!(p, times, getindex.(values, i); name=device_labels[i])
    end
    return p
end

"""
    livemonitor_temperature(duration) -> times, temperatures

Monitor the temperature of GPU(s) over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the temperatures as a `Vector{Vector{Float64}}`.

For general keyword arguments, see [`livemonitor_something`](@ref).
"""
function livemonitor_temperature(duration; kwargs...)
    return livemonitor_something(
        get_temperatures, duration; title="GPU Temperatures", ylabel="T [C]", kwargs...
    )
end

"""
    livemonitor_powerusage(duration) -> times, powerusage

Monitor the power usage of GPU(s) (in Watts) over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the power usage as a `Vector{Vector{Float64}}`.

For general keyword arguments, see [`livemonitor_something`](@ref).
"""
function livemonitor_powerusage(duration; kwargs...)
    return livemonitor_something(
        get_power_usages, duration; title="GPU Power Usage", ylabel="P [W]", kwargs...
    )
end

"""
    livemonitor_something(f, duration) -> times, values

Monitor some property of GPU(s), as specified through the function `f`, over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the temperatures as a `Vector{Vector{Float64}}`.

The function `f` will be called on a vector of devices (`CuDevice`s or `NVML.Device`s) and should return a vector of `Float64` values.

**Keyword arguments:**
* `freq` (default: `1`): polling rate in Hz.
* `devices` (default: `NVML.devices()`): `CuDevice`s or `NVML.Device`s to consider.
* `plot` (default: `false`): Create a unicode plot after the monitoring.
* `liveplot` (default: `false`): Create and update a unicode plot during the monitoring. Use optional `ylims` to specify fixed y limits.
* `title` (default: `""`): Title used in unicode plots.
* `ylabel` (default: `"Values"`): y label used in unicode plots.

See: [`livemonitor_temperature`](@ref), [`livemonitor_powerusage`](@ref)
"""
function livemonitor_something(
    f::F,
    duration;
    freq=1,
    devices=NVML.devices(),
    plot=false,
    liveplot=true,
    ylims=nothing,
    title="",
    ylabel="Values",
) where {F}
    t_freq = 1.0 / freq
    values = Vector{Float64}[f(devices)]
    times = Float64[0.0]

    t = time()
    while (time() - t) < duration
        sleep(t_freq)
        push!(times, time() - t)
        push!(values, f(devices))
        if liveplot
            print("\033c")
            display(
                _plot_something(
                    values,
                    times;
                    devices,
                    ylims=isnothing(ylims) ? _defaultylims(values) : ylims,
                    title,
                    ylabel,
                ),
            )
        end
    end

    plot && display(
        _plot_something(
            values,
            times;
            devices,
            ylims=isnothing(ylims) ? _defaultylims(values) : ylims,
            title,
            ylabel,
        ),
    )
    return times, values
end

function _plot_something(
    values,
    times=1:length(values);
    devices=nothing,
    ylims=_defaultylims(values),
    title="",
    ylabel="Values",
)
    if isnothing(devices)
        device_labels = ["" for _ in 0:(length(first(values)) - 1)]
    else
        device_labels = [_device2string(dev) for dev in devices]
    end

    p = lineplot(
        times,
        getindex.(values, 1);
        title=title,
        xlabel="Time [s]",
        ylabel=ylabel,
        name=device_labels[1],
        ylim=ylims,
    )
    for i in 2:length(first(values))
        lineplot!(p, times, getindex.(values, i); name=device_labels[i])
    end
    return p
end

function _defaultylims(values)
    total_max = maximum(maximum(vs) for vs in values)
    total_min = minimum(minimum(vs) for vs in values)
    return (floor(Int, total_min * 0.95), ceil(Int, total_max * 1.05))
end

function _symbol2title_and_label(s::Symbol)
    if s == :temperature
        return "GPU Temperature", "T [C]"
    elseif s == :power
        return "GPU Power Usage", "P [W]"
    elseif s == :compute
        return "GPU Utilization (Compute)", "U [%]"
    elseif s == :mem
        return "GPU Utilization (Memory)", "U [%]"
    else
        return "", "Values"
    end
end

"""
    savefig_monitoring_results(r::MonitoringResults, symbols=keys(r.results))
Save plots of the quantities specified through `symbols` of a `MonitoringResults` object to disk.
**Note:** Only available if CairoMakie.jl is loaded next to GPUInspector.jl.
"""
function savefig_monitoring_results(r::MonitoringResults, symbols=keys(r.results); ext=:pdf)
    return error("You need to load CairoMakie.jl first.")
end