"""
Tests for the Memspectrum package.

Run from the repository root with:

    julia --project=. test/runtests.jl

or via the package manager:

    julia --project=. -e 'using Pkg; Pkg.test()'
"""

using Test
using Random
using Statistics

# Load the package from source when running directly
if !isdefined(Main, :Memspectrum)
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
    include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
end
using .Memspectrum

@testset "Memspectrum.jl" begin

    # -----------------------------------------------------------------------
    @testset "MESA struct and solve!" begin
        Random.seed!(1)
        N  = 512
        dt = 1.0 / 256.0
        x  = randn(N)

        m = MESA()
        @test m.P    === nothing
        @test m.a_k  === nothing
        @test m.N    === nothing

        P_ret, ak_ret, opt = solve!(m, x; verbose=false)
        @test m.P   !== nothing
        @test m.a_k !== nothing
        @test m.N   == N
        @test m.p   >= 1
        @test P_ret ≈ m.P
        @test ak_ret ≈ m.a_k
        @test !isempty(opt)
    end

    # -----------------------------------------------------------------------
    @testset "spectrum / memspectrum alias" begin
        Random.seed!(2)
        N  = 1024
        dt = 1.0 / 512.0
        x  = sin.(2π * 50 .* (0:N-1) .* dt) .+ 0.1 .* randn(N)

        m = MESA()
        solve!(m, x)

        f1, psd1 = spectrum(m, dt; onesided=true)
        f2, psd2 = memspectrum(m, dt; onesided=true)

        @test f1   == f2
        @test psd1 == psd2
        @test length(f1) == N ÷ 2
        @test all(psd1 .> 0)
        @test maximum(f1) < 0.5 / dt   # top bin is below Nyquist

        # Peak should be near 50 Hz
        @test f1[argmax(psd1)] ≈ 50.0  atol=5.0
    end

    # -----------------------------------------------------------------------
    @testset "spectrum on custom frequency grid" begin
        Random.seed!(3)
        N  = 512
        dt = 1.0 / 256.0
        x  = randn(N)
        m  = MESA()
        solve!(m, x)

        f_custom = collect(LinRange(1.0, 100.0, 200))
        psd_custom = spectrum(m, dt; frequencies=f_custom)
        @test length(psd_custom) == 200
        @test all(psd_custom .> 0)
    end

    # -----------------------------------------------------------------------
    @testset "forecast" begin
        Random.seed!(4)
        N = 512
        x = randn(N)
        m = MESA()
        solve!(m, x)

        preds = forecast(m, x, 100; number_of_simulations=10, seed=42)
        @test size(preds) == (10, 100)

        # include_data
        preds_full = forecast(m, x, 100;
                              number_of_simulations=3, include_data=true)
        @test size(preds_full, 2) == m.p + 100
    end

    # -----------------------------------------------------------------------
    @testset "whiten" begin
        Random.seed!(5)
        N = 1024
        x = randn(N)
        m = MESA()
        solve!(m, x)
        w = whiten(m, x)
        # Whitened data should be shorter (trim applied) and approximately white
        @test length(w) < N
        @test abs(mean(w)) < 0.2
    end

    # -----------------------------------------------------------------------
    @testset "entropy_rate and logL" begin
        Random.seed!(6)
        N  = 512
        dt = 1.0 / 256.0
        x  = randn(N)
        m  = MESA()
        solve!(m, x)

        er = entropy_rate(m, dt)
        @test isfinite(er)

        ll = logL(m, x, dt)
        @test isfinite(ll)
    end

    # -----------------------------------------------------------------------
    @testset "save_mesa / load_mesa round-trip" begin
        Random.seed!(7)
        x = randn(256)
        m = MESA()
        solve!(m, x)

        fname = tempname() * ".txt"
        save_mesa(m, fname)
        m2 = load_mesa(fname)
        rm(fname)

        @test m2.P   ≈ m.P
        @test m2.N   == m.N
        @test m2.mu  ≈ m.mu
        @test m2.a_k ≈ m.a_k
    end

    # -----------------------------------------------------------------------
    @testset "generate_data" begin
        Random.seed!(8)
        f_tmpl = collect(LinRange(1.0, 100.0, 50))
        psd_tmpl = ones(50)
        t, ts, freqs, fs, psd_out = generate_data(f_tmpl, psd_tmpl, 1.0;
                                                   sampling_rate=256.0, seed=99)
        @test length(t)   == 256
        @test length(ts)  == 256
        @test length(freqs) > 0
    end

    # -----------------------------------------------------------------------
    @testset "mesa_spectrogram / memgram alias" begin
        Random.seed!(9)
        N  = 2048
        dt = 1.0 / 256.0
        x  = randn(N)

        t1, f1, S1 = mesa_spectrogram(x, dt; segment_length=256, overlap=0.5)
        t2, f2, S2 = memgram(x, dt; segment_length=256, overlap=0.5)

        @test t1 == t2
        @test f1 == f2
        @test S1 == S2
        @test size(S1, 1) == 256 ÷ 2
        @test all(S1 .> 0)
    end

    # -----------------------------------------------------------------------
    @testset "AR(2) spectrum peak recovery" begin
        # Generate an AR(2) time series with known resonance at ~30 Hz
        # AR coefficients: x[t] = -a1*x[t-1] - a2*x[t-2] + noise
        Random.seed!(10)
        N   = 4096
        dt  = 1.0 / 256.0
        a1, a2 = -1.5, 0.9
        x   = zeros(N)
        x[1] = randn(); x[2] = randn()
        for t in 3:N
            x[t] = -a1 * x[t-1] - a2 * x[t-2] + randn()
        end

        m = MESA()
        solve!(m, x; optimisation_method="FPE", method="Fast")
        f, psd = memspectrum(m, dt; onesided=true)

        # The spectral peak should be within ±10 Hz of the true resonance
        true_peak = acos(-a1 / (2 * sqrt(a2))) / (2π * dt)
        @test abs(f[argmax(psd)] - true_peak) < 10.0
    end

end # @testset "Memspectrum.jl"
