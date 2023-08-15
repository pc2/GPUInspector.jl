"""
    monitoring_start(; devices, kwargs...)
Start monitoring of GPU temperature, utilization, power usage, etc.

**Keyword arguments:**
* `freq` (default: `1`): polling rate in Hz.
* `devices` (default: e.g. `CUDA.devices()`): GPU devices to monitor.
* `thread` (default: `Threads.nthreads()`): id of the Julia thread that should run the monitoring.
* `verbose` (default: `true`): toggle verbose output.

See also [`monitoring_stop`](@ref).
"""
monitoring_start(; kwargs...) = monitoring_start(backend(); kwargs...)
monitoring_start(::Backend; kwargs...) = not_implemented_yet()

"""
    monitoring_stop(; verbose=true) -> results
Stops the GPU monitoring and returns the measured values.

See also [`monitoring_start`](@ref) and [`plot_monitoring_results`](@ref).
"""
monitoring_stop(; kwargs...) = monitoring_stop(backend(); kwargs...)
monitoring_stop(::Backend; kwargs...) = not_implemented_yet()

# TODO: livemonitor_... (currently only in NVIDIABackend)
"""
    livemonitor_something(f, duration) -> times, values

Monitor some property of GPU(s), as specified through the function `f`, over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the temperatures as a `Vector{Vector{Float64}}`.

The function `f` will be called on a vector of devices and should return a vector of `Float64` values.

**Keyword arguments:**
* `freq` (default: `1`): polling rate in Hz.
* `devices` (default: e.g. `NVML.devices()`): Devices to monitor.
* `plot` (default: `false`): Create a unicode plot after the monitoring.
* `liveplot` (default: `false`): Create and update a unicode plot during the monitoring. Use optional `ylims` to specify fixed y limits.
* `title` (default: `""`): Title used in unicode plots.
* `ylabel` (default: `"Values"`): y label used in unicode plots.

See: [`livemonitor_temperature`](@ref), [`livemonitor_powerusage`](@ref)
"""
livemonitor_something(; kwargs...) = livemonitor_something(backend(); kwargs...)
livemonitor_something(::Backend; kwargs...) = not_implemented_yet()

"""
    livemonitor_temperature(duration) -> times, temperatures

Monitor the temperature of GPU(s) over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the temperatures as a `Vector{Vector{Float64}}`.

For general keyword arguments, see [`livemonitor_something`](@ref).
"""
function livemonitor_temperature(args...; kwargs...)
    return livemonitor_temperature(backend, args...; kwargs...)
end
livemonitor_temperature(::Backend, args...; kwargs...) = not_implemented_yet()

"""
    livemonitor_powerusage(duration) -> times, powerusage

Monitor the power usage of GPU(s) (in Watts) over a given time period, as specified by `duration` (in seconds).
Returns the (relative) times as a `Vector{Float64}` and the power usage as a `Vector{Vector{Float64}}`.

For general keyword arguments, see [`livemonitor_something`](@ref).
"""
function livemonitor_powerusage(args...; kwargs...)
    return livemonitor_powerusage(backend, args...; kwargs...)
end
livemonitor_powerusage(::Backend, args...; kwargs...) = not_implemented_yet()

"""
    savefig_monitoring_results(r::MonitoringResults, symbols=keys(r.results); ext=:pdf)
Save plots of the quantities specified through `symbols` of a `MonitoringResults` object to disk.
**Note:** Only available if CairoMakie.jl is loaded next to GPUInspector.jl.
"""
function savefig_monitoring_results(r::Any, symbols::Any=nothing; ext=:pdf)
    return error("You need to load CairoMakie.jl first.")
end
