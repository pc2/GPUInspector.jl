"""
  p2p_bandwidth(; kwargs...)
Performs a peer-to-peer memory copy benchmark (time measurement) and
returns an inter-gpu memory bandwidth estimate (in GiB/s) derived from it.

**Keyword arguments:**
* `memsize` (default: `B(40_000_000)`): memory size to be used
* `src` (default: `0`): source device
* `dst` (default: `1`): destination device
* `nbench` (default: `5`): number of time measurements (i.e. p2p memcopies)
* `verbose` (default: `true`): set to false to turn off any printing.
* `hist` (default: `false`): when `true`, a UnicodePlots-based histogram is printed.
* `times` (default: `false`): toggle printing of measured times.
* `alternate` (default: `false`): alternate `src` and `dst`, i.e. copy data back and forth.
* `dtype` (default: `Float32`): see [`alloc_mem`](@ref).
* `io` (default: `stdout`): set the stream where the results should be printed.

**Examples:**
```julia
p2p_bandwidth()
p2p_bandwidth(MiB(1024))
p2p_bandwidth(KiB(20_000); dtype=Int32)
"""
p2p_bandwidth(; kwargs...) = p2p_bandwidth(backend(); kwargs...)
p2p_bandwidth(::Backend; kwargs...) = not_implemented_yet()

"""
    p2p_bandwidth_all(; kwargs...)
Run [`p2p_bandwidth`](@ref) for all combinations of available devices.
Returns a matrix with the p2p memory bandwidth estimates.
"""
p2p_bandwidth_all(; kwargs...) = p2p_bandwidth_all(backend(); kwargs...)
p2p_bandwidth_all(::Backend; kwargs...) = not_implemented_yet()

"""
Same as [`p2p_bandwidth`](@ref) but measures the bidirectional bandwidth (copying data back and forth).
"""
p2p_bandwidth_bidirectional(; kwargs...) = p2p_bandwidth_bidirectional(backend(); kwargs...)
p2p_bandwidth_bidirectional(::Backend; kwargs...) = not_implemented_yet()

"""
Same as [`p2p_bandwidth_all`](@ref) but measures the bidirectional bandwidth (copying data back and forth).
"""
function p2p_bandwidth_bidirectional_all(; kwargs...)
    return p2p_bandwidth_bidirectional_all(backend(); kwargs...)
end
p2p_bandwidth_bidirectional_all(::Backend; kwargs...) = not_implemented_yet()
