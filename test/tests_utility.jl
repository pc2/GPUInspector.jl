@testset "UnitPrefixedBytes" begin
    using InteractiveUtils: subtypes

    # general stuff
    b = B(40_000_000)
    m = MiB(38.14697265625)
    m̃ = MB(40.0)
    @test typeof(b) == B
    @test typeof(m) == MiB
    @test typeof(m̃) == MB
    @test string(b) == "4.0e7 B" || string(b) == "4.0e7 GPUInspector.B"
    @test string(m) == "~38.15 MiB" || string(m) == "~38.15 GPUInspector.MiB"
    @test string(m̃) == "40.0 MB" || string(m̃) == "40.0 GPUInspector.MB"
    @test value(b) == 40_000_000
    @test value(m) ≈ 38.14697265625
    @test value(m̃) ≈ 40.0
    @test simplify(b) ≈ m
    @test simplify(m) ≈ m
    @test simplify(m̃) ≈ m
    @test simplify(b; base=10) ≈ m̃
    @test simplify(m; base=10) ≈ m̃
    @test simplify(m̃; base=10) ≈ m̃
    @test change_base(b) == b
    @test change_base(m) == m̃
    @test change_base(m̃) == m
    @test convert(MB, b) == m̃
    @test convert(MiB, b) == m
    @test convert(MB, m) == m̃
    @test convert(MiB, m̃) == m
    @test bytes(b) == 4.0e7
    @test bytes(m) == 4.0e7
    @test bytes(m̃) == 4.0e7
    @test bytes(4.0e7) ≈ m

    # GiB <-> GB conversion because it's particularly important
    @test convert(GiB, GB(300)) ≈ GiB(279.39677238464355)
    @test change_base(GB(300)) ≈ GiB(279.39677238464355)
    @test convert(GB, GiB(300)) ≈ GB(322.12254720000004)
    @test change_base(GiB(300)) ≈ GB(322.12254720000004)

    # ≈, ==, ===
    @test b == m
    @test b == m̃
    @test m == m̃
    @test b ≈ m
    @test b ≈ m̃
    @test m ≈ m̃
    @test b === b
    @test b !== m
    @test b !== m̃
    @test m !== m̃

    # basic arithmetics
    types = subtypes(UnitPrefixedBytes)
    for T in types
        @test T(1.2) + T(2.3) ≈ T(3.5)
        @test T(3.5) - T(2.3) ≈ T(1.2)
        @test 2 * T(1.234) ≈ T(2.468)
        @test T(10) / 2 == T(5)
    end
    for T in types, S in types
        @test bytes(T(1.2) + S(2.3)) ≈ bytes(T(1.2)) + bytes(S(2.3))
        @test abs(bytes(T(1.2) - S(2.3))) ≈ abs(bytes(T(1.2)) - bytes(S(2.3)))
    end
    @test B(40_000_000) + MB(3) - 2 * KiB(2) ≈ MB(42.995904)
end
