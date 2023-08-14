const MONITORING = Ref{Bool}(false)
const MONITORING_TASK = Ref{Union{Nothing,Task}}(nothing)

"Checks if we are currently monitoring."
ismonitoring() = MONITORING[]
_monitoring!(val::Bool) = MONITORING[] = val
_set_monitoring_task(t::Task) = MONITORING_TASK[] = t
_get_monitoring_task() = MONITORING_TASK[]

"""
Struct to hold the results of monitoring.
This includes the time points (`times`), the monitored devices (`devices`), as well as
a dictionary holding the (vector-)values of different quantities (identified by symbols) at
each of the time points.
"""
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
