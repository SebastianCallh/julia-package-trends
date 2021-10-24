using Revise
using Dates, CSV, DataFrames, StatsPlots, Pipe, Turing, FeatureTransforms, Statistics, LazyArrays

df_raw = CSV.read("data/package_requests_by_region_by_date.csv", DataFrame)
allowed_regions = ["eu-central", "us-east", "us-west", "cn-southeast", "cn-northeast", "cn-east"]
plots_path = "plots/"
save_path(file) = joinpath(plots_path, file)
isdir(plots_path) || mkdir(plots_path)

df = @pipe df_raw |>
    sort(_, :date) |>
    filter(x -> in(x.region, allowed_regions), _,) |>
    filter(x -> !ismissing(x.client_type), _) |>
    groupby(_, [:date, :region, :client_type]) |>
    combine(_,:request_count => sum => :request_count)
df.client_type = convert.(String7, df.client_type)

module Transforms
using FeatureTransforms

struct IntegerIndex{T} <: FeatureTransforms.Transform  where T
    transform::Dict{T, Int32}
    inverse::Dict{Int32, T}
end

function IntegerIndex(x::AbstractArray{T}) where T
    trans_dict = Dict(z => i for (i, z) in enumerate(unique(x)))
    inv_dict = Dict(v => k for (k, v) in trans_dict)
    IntegerIndex{T}(trans_dict, inv_dict)
end

FeatureTransforms.cardinality(::IntegerIndex{T}) where T = FeatureTransforms.OneToOne()
FeatureTransforms._apply(x, I::IntegerIndex{T}; inverse=false, kwargs...) where T = begin
    d = inverse ? I.inverse : I.transform
    map(z -> get(d, z, missing), x)
end
end

region_trans = Transforms.IntegerIndex(df.region)
df[!,:region_index] = region_trans(df.region)

client_trans = Transforms.IntegerIndex(df.client_type)
df[!,:client_type_index] = client_trans(df.client_type)

df[!,:day] = day.(df.date)
day_transform = MeanStdScaling(df; cols=:day)
df[!,:standardized_day] = day_transform(df.day)
df[!,:log10_request_count] = log10.(df.request_count)

const x = df.standardized_day
const y = df.log10_request_count
const r = df.region_index
const t = df.client_type_index
const regions = reshape(unique(df.region), 1,: )
const client_types = reshape(unique(df.client_type), 1,: )
const R = length(unique(r))
const T = length(unique(t))

pal = get_color_palette(:auto, R)
colors = [pal[i] for i in r]
lazydist(dist, args...) = arraydist(LazyArray(@~ dist.(args...)))

#= 
x = df.standardized_day
y = df.log10_request_count
r = df.region_index
regions = reshape(unique(df.region), 1,: )
R = length(unique(r))
pal = get_color_palette(:auto, R)
colors = [pal[i] for i in r]

 
const M = 50
const N = 50
xx = mapreduce(r -> range(extrema(x)..., length=N), vcat, 1:R)
rr = mapreduce(r -> repeat([r], N), vcat, 1:R)
prediction(chain) = predict(pooled_regions(xx, missing, rr, R), chain)


function prior_predictive_plot(chain) 
    pred = prediction(chain)
    plot(
        day_transform(xx; inverse=true),
        group(pred, "y").value.data[:,:,1]',
        group=rr,
        title="Prior samples for $client_type",
        xlabel="Days",
        ylabel="Package downloads (log10)",
        label=nothing,
        color=:blues,
        alpha=.5
    )
end

function posterior_predictive_plot(chain)
    pred = prediction(chain)
    ss = summarystats(group(pred, "y"))
    plot(
        day_transform(xx; inverse=true),
        ss[:,:mean],
        ribbon=2*ss[:,:std],
        group=rr,
        title="Posterior predictive distribution for $client_type",
        xlabel="Days",
        ylabel="Package downloads (log10)",
        label=nothing,
        legend=:bottomright
    )
end

function scatter_data(plt)
    scatter!(
        plt,
        day_transform(x; inverse=true), y,
        group=r,
        color=colors,
        label=regions
    )
end
=#


#= subsample(df; nobs=100) = df[Int32.(round.(collect(range(1, nrow(df), length=nobs)))),:]
df = @pipe df |>
    groupby(_, :region) |>
    combine(subsample, _) |>
    sort(_, :date)
 =#
# @df df_date_count plot(
#     :date, log10.(:request_count),
#     group=:region,
#     legend=:bottomleft,
#     xlabel="Date",
#     ylabel="Request count (log10)",
#     title="Request count per region over time",
#     size=(800, 600)
# )
    
#= @df df plot(
    :date, log10.(:request_count),
    group=:region,
    legend=:bottomleft,
    xlabel="Date",
    ylabel="Request count (log10)",
    title="Request count per region over time",
    size=(800, 600)
)
=#
