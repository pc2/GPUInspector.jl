"""
    host2device_bandwidth(; kwargs...)

Performs a host-to-device memory copy benchmark (time measurement) and
returns the host-to-device bandwidth estimate (in GiB/s) derived from it.

**Keyword arguments:**
* `memsize` (default: `GiB(0.5)`): memory size to be used
* `nbench` (default: `10`): number of time measurements (i.e. p2p memcopies)
* `verbose` (default: `true`): set to false to turn off any printing.
* `stats` (default: `false`): when `true` shows statistical information about the benchmark.
* `times` (default: `false`): toggle printing of measured times.
* `dtype` (default: `Cchar`): used data type.
* `io` (default: `stdout`): set the stream where the results should be printed.

**Examples:**
```julia
host2device_bandwidth()
host2device_bandwidth(MiB(1024))
host2device_bandwidth(KiB(20_000); dtype=Int32)
```
"""
host2device_bandwidth(; kwargs...) = host2device_bandwidth(backend(); kwargs...)
host2device_bandwidth(::Backend; kwargs...) = not_implemented_yet()
