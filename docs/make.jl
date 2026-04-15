using Documenter
using Memspectrum

# Copy example images into docs/src/assets/ so they can be referenced in the manual.
let assets_dir = joinpath(@__DIR__, "src", "assets")
    mkpath(assets_dir)
    for img in ["toy_psd_estimate.png", "toy_spectrogram.png",
                "chirp_spectrogram.png", "quadratic_chirp_spectrogram.png"]
        src = joinpath(@__DIR__, "..", "examples", img)
        dst = joinpath(assets_dir, img)
        isfile(src) && cp(src, dst; force=true)
    end
end

makedocs(
    modules  = [Memspectrum],
    sitename = "Memspectrum.jl",
    checkdocs = :exports,
    warnonly  = [:missing_docs],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home"     => "index.md",
        "API"      => "api.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(
    repo   = "github.com/RiccardoBuscicchio/memspectrogram.git",
    branch = "gh-pages",
    push_preview = true,
)
