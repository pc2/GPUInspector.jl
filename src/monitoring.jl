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
