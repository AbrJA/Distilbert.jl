using Documenter, Distilbert

makedocs(
    modules=[Distilbert],
    sitename="Distilbert.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(
    repo="github.com/AbrJA/Distilbert.jl.git",
)
