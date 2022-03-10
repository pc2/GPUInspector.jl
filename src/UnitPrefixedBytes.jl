"""
Abstract type representing an amount of data, i.e. a certain number of bytes,
with a unit prefix (also "metric prefix"). Examples include the SI prefixes,
like KB, MB, and GB, but also the binary prefixes (ISO/IEC 80000), like KiB, MiB, and GiB.

See https://en.wikipedia.org/wiki/Binary_prefix for more information.
"""
abstract type UnitPrefixedBytes <: Number end

"Bytes"
struct B <: UnitPrefixedBytes
    value::Float64
end
"Kibibytes, i.e. `2^10 = 1024` bytes"
struct KiB <: UnitPrefixedBytes
    value::Float64
end
"Mebibytes, i.e. `2^20 = 1024^2` bytes"
struct MiB <: UnitPrefixedBytes
    value::Float64
end
"Gibibytes, i.e. `2^30 = 1024^3` bytes"
struct GiB <: UnitPrefixedBytes
    value::Float64
end
"Tebibytes, i.e. `2^40 = 1024^4` bytes"
struct TiB <: UnitPrefixedBytes
    value::Float64
end
"Kilobytes, i.e. `10^3 = 1000` bytes"
struct KB <: UnitPrefixedBytes
    value::Float64
end
"Megabytes, i.e. `10^6 = 1000^2` bytes"
struct MB <: UnitPrefixedBytes
    value::Float64
end
"Gigabytes, i.e. `10^9 = 1000^3` bytes"
struct GB <: UnitPrefixedBytes
    value::Float64
end
"Terabytes, i.e. `10^12 = 1000^4` bytes"
struct TB <: UnitPrefixedBytes
    value::Float64
end

# accessor
value(x::UnitPrefixedBytes) = x.value

# pretty printing
function Base.show(io::IO, x::T) where {T<:UnitPrefixedBytes}
    v = value(x)
    v_rounded = round(v; digits=2)
    if abs(v - v_rounded) > 0
        print(io, "~", v_rounded, " ", T)
    else
        print(io, v, " ", T)
    end
end

# zero
Base.zero(::Type{T}) where {T<:UnitPrefixedBytes} = T(0.0)
Base.zero(::T) where {T<:UnitPrefixedBytes} = T(0.0)

# one
Base.one(::Type{T}) where {T<:UnitPrefixedBytes} = T(1.0)
Base.one(::T) where {T<:UnitPrefixedBytes} = T(1.0)

# conversion
Base.convert(::Type{B}, x::B) = B(value(x)) # to avoid infinite recursion / stackoverflow
Base.convert(::Type{B}, x::KB) = B(value(x) * 1e3)
Base.convert(::Type{B}, x::MB) = B(value(x) * 1e6)
Base.convert(::Type{B}, x::GB) = B(value(x) * 1e9)
Base.convert(::Type{B}, x::TB) = B(value(x) * 1e12)
Base.convert(::Type{B}, x::KiB) = B(value(x) * 2^10)
Base.convert(::Type{B}, x::MiB) = B(value(x) * 2^20)
Base.convert(::Type{B}, x::GiB) = B(value(x) * 2^30)
Base.convert(::Type{B}, x::TiB) = B(value(x) * 2^40)

Base.convert(::Type{KB}, x::B) = KB(value(x) * 1e-3)
Base.convert(::Type{MB}, x::B) = MB(value(x) * 1e-6)
Base.convert(::Type{GB}, x::B) = GB(value(x) * 1e-9)
Base.convert(::Type{TB}, x::B) = TB(value(x) * 1e-12)
Base.convert(::Type{KiB}, x::B) = KiB(value(x) * 2^(-10))
Base.convert(::Type{MiB}, x::B) = MiB(value(x) * 2^(-20))
Base.convert(::Type{GiB}, x::B) = GiB(value(x) * 2^(-30))
Base.convert(::Type{TiB}, x::B) = TiB(value(x) * 2^(-40))

function Base.convert(::Type{T}, x::UnitPrefixedBytes) where {T<:UnitPrefixedBytes}
    return convert(T, convert(B, x))
end

function Base.isapprox(x::UnitPrefixedBytes, y::UnitPrefixedBytes; kwargs...)
    return isapprox(bytes(x), bytes(y); kwargs...)
end
# Base.:(==)(x::UnitPrefixedBytes, y::UnitPrefixedBytes) = bytes(x) == bytes(y) # unnecessary
# Base.:(===)(x::UnitPrefixedBytes, y::UnitPrefixedBytes) = (typeof(x) == typeof(y)) && (bytes(x) == bytes(y)) # unnecessary

# promotion
Base.promote_rule(::Type{B}, ::Type{KB}) = KB
Base.promote_rule(::Type{B}, ::Type{MB}) = MB
Base.promote_rule(::Type{B}, ::Type{GB}) = GB
Base.promote_rule(::Type{B}, ::Type{TB}) = TB
Base.promote_rule(::Type{B}, ::Type{KiB}) = KiB
Base.promote_rule(::Type{B}, ::Type{MiB}) = MiB
Base.promote_rule(::Type{B}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{B}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{KB}, ::Type{MB}) = MB
Base.promote_rule(::Type{KB}, ::Type{GB}) = GB
Base.promote_rule(::Type{KB}, ::Type{TB}) = TB
Base.promote_rule(::Type{KB}, ::Type{KiB}) = KiB
Base.promote_rule(::Type{KB}, ::Type{MiB}) = MiB
Base.promote_rule(::Type{KB}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{KB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{MB}, ::Type{GB}) = GB
Base.promote_rule(::Type{MB}, ::Type{TB}) = TB
Base.promote_rule(::Type{MB}, ::Type{KiB}) = MB
Base.promote_rule(::Type{MB}, ::Type{MiB}) = MiB
Base.promote_rule(::Type{MB}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{MB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{GB}, ::Type{TB}) = TB
Base.promote_rule(::Type{GB}, ::Type{KiB}) = GB
Base.promote_rule(::Type{GB}, ::Type{MiB}) = GB
Base.promote_rule(::Type{GB}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{GB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{TB}, ::Type{KiB}) = TB
Base.promote_rule(::Type{TB}, ::Type{MiB}) = TB
Base.promote_rule(::Type{TB}, ::Type{GiB}) = TB
Base.promote_rule(::Type{TB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{KiB}, ::Type{MiB}) = MiB
Base.promote_rule(::Type{KiB}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{KiB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{MiB}, ::Type{GiB}) = GiB
Base.promote_rule(::Type{MiB}, ::Type{TiB}) = TiB

Base.promote_rule(::Type{GiB}, ::Type{TiB}) = TiB

# basic arithmetics
Base.:+(x::T, y::T) where {T<:UnitPrefixedBytes} = T(value(x) + value(y))
Base.:-(x::T, y::T) where {T<:UnitPrefixedBytes} = T(value(x) - value(y))

Base.:*(x::Number, y::T) where {T<:UnitPrefixedBytes} = T(x * value(y))
Base.:*(x::T, y::Number) where {T<:UnitPrefixedBytes} = y * x # symmetry / commutativity

Base.:/(x::T, y::Number) where {T<:UnitPrefixedBytes} = T(value(x) / y)

# utility

"""
    bytes(x::UnitPrefixedBytes)
Return the number of bytes (without prefix) as `Float64`.
"""
bytes(x::UnitPrefixedBytes) = value(convert(B, x))

"""
    bytes(x::Number)
Returns an appropriate `UnitPrefixedBytes` object, representing the number of bytes.

**Note:** This function is type unstable by construction!

See [`simplify`](@ref) for what "appropriate" means here.
"""
bytes(x::Number) = simplify(B(x))

"""
Toggle between
* Base 10, SI prefixes, i.e. factors of 1000
* Base 2, ISO/IEC prefixes, i.e. factors of 1024

**Example:**
```julia
julia> change_base(KB(13))
~12.7 KiB

julia> change_base(KiB(13))
~13.31 KB
```
"""
change_base(x::T) where {T<:UnitPrefixedBytes} = convert(_base_partner(T), x)
_base_partner(::Type{B}) = B
_base_partner(::Type{KB}) = KiB
_base_partner(::Type{MB}) = MiB
_base_partner(::Type{GB}) = GiB
_base_partner(::Type{TB}) = TiB
_base_partner(::Type{KiB}) = KB
_base_partner(::Type{MiB}) = MB
_base_partner(::Type{GiB}) = GB
_base_partner(::Type{TiB}) = TB

"""
    simplify(x::UnitPrefixedBytes[; base])
Given a `UnitPrefixedBytes` number `x`, finds a more appropriate
`UnitPrefixedBytes` that represents the same number of bytes
but with a smaller value.

The optional keyword argument `base` can be used to switch between
base 2, i.e. ISO/IEC prefixes (default), and base 10. Allowed values are
`2`, `10`, `:SI`, `:ISO`, and `:IEC`.

**Note:** This function is type unstable by construction!

**Example:**
```julia
julia> simplify(B(40_000_000))
~38.15 MiB

julia> simplify(B(40_000_000); base=10)
40.0 MB
```
"""
function simplify(x::B; base=2)
    if base == 2 || base == :SI
        hierarchy = (B, KiB, MiB, GiB, TiB)
        factor = 1024
    elseif base == 10 || base == :IEC || base == :ISO
        hierarchy = (B, KB, MB, GB, TB)
        factor = 1000
    else
        throw(ArgumentError("Only base 2 or base 10 supported."))
    end
    v = value(x)
    for T in hierarchy
        v /= factor
        v < 1 && return convert(T, x)
    end
    return convert(last(hierarchy), x)
end
simplify(x::UnitPrefixedBytes; kwargs...) = simplify(convert(B, x); kwargs...)
