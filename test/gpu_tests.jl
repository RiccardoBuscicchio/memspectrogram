"""
GPU extension tests for the Memspectrum package.

Tests the MemspectrumCUDAExt extension.  Always verifies that the CPU-fallback
paths (`use_gpu=false`) work correctly when CUDA is loaded.  GPU-specific paths
(`use_gpu=true`) are only exercised when a functional CUDA device is detected at
run time; otherwise those test-sets are skipped with an informational message.

Run with CUDA available in the load path (e.g. installed in the default
environment, which is stacked on top of the project):

    julia --project=. test/gpu_tests.jl

Memspectrum must be loaded as a package (not `include`d) so that the extension
attaches to it.

Or add it to the CI matrix.
"""

using Test
using Random
using Statistics
using CUDA          # triggers automatic loading of MemspectrumCUDAExt
using Memspectrum

@test Base.get_extension(Memspectrum, :MemspectrumCUDAExt) !== nothing

@testset "MemspectrumCUDAExt" begin

    Random.seed!(42)
    N  = 512
    dt = 1.0 / 256.0
    x  = randn(N)
    m  = MESA()
    solve!(m, x; verbose=false)

    # ------------------------------------------------------------------
    # CPU-fallback paths (use_gpu=false) – always run, no GPU needed
    # ------------------------------------------------------------------
    @testset "forecast CPU fallback (use_gpu=false)" begin
        preds = forecast(m, x, 50; number_of_simulations=5, use_gpu=false)
        @test size(preds) == (5, 50)
        @test all(isfinite.(preds))
    end

    @testset "mesa_spectrogram CPU fallback (use_gpu=false)" begin
        t, f, S = mesa_spectrogram(x, dt;
                                   segment_length=128,
                                   overlap=0.5,
                                   use_gpu=false)
        @test length(t) > 0
        @test length(f) == 128 ÷ 2
        @test size(S, 1) == 128 ÷ 2
        @test all(S .> 0)
    end

    # ------------------------------------------------------------------
    # Real GPU paths (use_gpu=true) – only when a CUDA GPU is present
    # ------------------------------------------------------------------
    if CUDA.functional()
        @testset "forecast GPU (use_gpu=true)" begin
            preds = forecast(m, x, 50; number_of_simulations=8, use_gpu=true)
            @test size(preds) == (8, 50)
            @test all(isfinite.(preds))
        end

        @testset "mesa_spectrogram GPU (use_gpu=true)" begin
            t, f, S = mesa_spectrogram(x, dt;
                                       segment_length=128,
                                       overlap=0.5,
                                       use_gpu=true)
            @test length(t) > 0
            @test length(f) == 128 ÷ 2
            @test size(S, 1) == 128 ÷ 2
            @test all(S .> 0)
        end

        @testset "irregular solve! GPU matches CPU" begin
            Random.seed!(7)
            f0, r = 0.01, 0.995
            a1, a2 = -2r * cos(2π * f0), r^2
            xf = zeros(100_000)
            for t in 3:length(xf)
                xf[t] = -a1 * xf[t-1] - a2 * xf[t-2] + randn()
            end
            keep = rand(length(xf)) .< 0.05
            t = Float64.(findall(keep))
            y = xf[keep]

            for (sw, method) in ((10.0, "AICirreg"), (20.0, "AICmod"), (5.0, "Fixed"))
                m_cpu = MESA()
                solve!(m_cpu, t, y; dt=20.0, slot_width=sw,
                       optimisation_method=method, m=12)
                m_gpu = MESA()
                solve!(m_gpu, t, y; dt=20.0, slot_width=sw,
                       optimisation_method=method, m=12, use_gpu=true)
                @test m_gpu.n_products == m_cpu.n_products
                @test m_gpu.ref_coefficients ≈ m_cpu.ref_coefficients rtol=1e-10
                @test m_gpu.p == m_cpu.p
                @test m_gpu.a_k ≈ m_cpu.a_k rtol=1e-10
                @test m_gpu.P ≈ m_cpu.P rtol=1e-10
            end

            # Regular grid: GPU path also reduces to standard Burg
            x = y[1:2000] .- mean(y[1:2000])
            m_reg = MESA()
            solve!(m_reg, x; method="Standard", optimisation_method="Fixed", m=8)
            m_gpu = MESA()
            solve!(m_gpu, collect(0.0:1999), x; dt=1.0, slot_width=0.5,
                   optimisation_method="Fixed", m=8, use_gpu=true)
            @test m_gpu.ref_coefficients ≈ m_reg.ref_coefficients rtol=1e-10
        end

        @testset "irregular memgram GPU matches CPU" begin
            Random.seed!(8)
            t = sort(rand(6000) .* 6000.0)
            t[100:300] .= t[100]                      # duplicate times
            f_line = [tt < 3000 ? 0.05 : 0.15 for tt in t]
            x = sin.(2π .* f_line .* t) .+ 0.3 .* randn(length(t))
            x[t .> 5600] .= 0.0                       # all-zero tail segments
            kw = (dt=2.0, segment_duration=500.0, overlap=0.5)

            tc_c, f_c, S_c = memgram(t, x; kw...)
            tc_g, f_g, S_g = memgram(t, x; kw..., use_gpu=true)
            @test tc_g == tc_c
            @test f_g == f_c
            @test isequal(isnan.(S_g), isnan.(S_c))
            ok = .!isnan.(S_c)
            @test S_g[ok] ≈ S_c[ok] rtol=1e-8
        end
    else
        @info "No functional CUDA GPU detected; GPU-specific tests skipped."
    end

end # @testset "MemspectrumCUDAExt"
