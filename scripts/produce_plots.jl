using Revise
using Dates, CSV, DataFrames, StatsPlots, Pipe, Turing, FeatureTransforms, Statistics, LazyArrays

df_region_date = CSV.read("data/package_requests_by_region_by_date.csv", DataFrame)
client_type = "user"
out_path = "plots/$client_type"
save_path(file) = joinpath(out_path, file)
isdir(out_path) || mkdir(out_path)

df_date_count = @pipe df_region_date |> 
    sort(_, :date) |> 
    filter(x -> !in(x.region, ["sa", "in", "sg", "au"]), _,) |>
    filter(x -> !ismissing(x.client_type) && x.client_type == client_type, _) |>
    groupby(_, [:date, :region]) |>
    combine(_,
        :request_count => sum => :request_count,
        :region => identity => :region,
    )

@df df_date_count plot(
    :date, log10.(:request_count),
    group=:region,
    legend=:bottomleft,
    xlabel="Date",
    ylabel="Request count (log10)",
    title="Request count per region over time",
    size=(800, 600)
)

subsample(df) = df[Int32.(round.(collect(range(1, nrow(df), length=5000)))),:]
df_date_count_sub = @pipe df_date_count |>
    groupby(_, :region) |>
    combine(subsample, _)|>
    sort(_, :date)
    
@df df_date_count_sub plot(
    :date, log10.(:request_count),
    group=:region,
    legend=:bottomleft,
    xlabel="Date",
    ylabel="Request count (log10)",
    title="Request count per region over time",
    size=(800, 600)
)

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

region_trans = Transforms.IntegerIndex(df_date_count_sub.region)
df_date_count_sub[!,:region_index] = region_trans(df_date_count_sub.region)

df_date_count_sub[!,:day] = day.(df_date_count_sub.date)
day_transform = MeanStdScaling(df_date_count_sub; cols=:day)
df_date_count_sub[!,:standardized_day] = day_transform(df_date_count_sub.day)
df_date_count_sub[!,:log10_request_count] = log10.(df_date_count_sub.request_count)

x = df_date_count_sub.standardized_day
y = df_date_count_sub.log10_request_count
r = df_date_count_sub.region_index
regions = reshape(unique(df_date_count_sub.region), 1,: )
R = length(unique(r))
pal = get_color_palette(:auto, R)
colors = [pal[i] for i in r]

@model pooled_regions(x, y, r, R) = begin
    αμ ~ Normal(5, 5)
    ασ ~ TruncatedNormal(0, 1, 0, Inf)
    βμ ~ Normal(0, 5)
    βσ ~ TruncatedNormal(0, 1, 0, Inf)

    α ~ filldist(Normal(αμ, ασ), R)
    β ~ filldist(Normal(βμ, ασ), R)
    σ ~ TruncatedNormal(0, 1, 0, Inf)
    y ~ arraydist(LazyArray(@~ Normal.(α[r] .+ x.*β[r], σ)))
end

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
        label=nothing
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

model_pooled = pooled_regions(x, y, r, R)
prior_chain = sample(model_pooled, Prior(), 10)
plot(prior_chain)

prior_pooled_pred_plt = prior_predictive_plot(prior_chain)
scatter_data(prior_pooled_pred_plt)

post_chain = sample(model_pooled, NUTS(), 2_000; progress=false);
plot(post_chain)

post_pred_pooled_plt = posterior_predictive_plot(post_chain)
scatter_data(post_pred_pooled_plt)
savefig(post_pred_pooled_plt, save_path("package_request_posterior_predictive.svg"))

βs = group(post_chain, "β").value.data[:,:,1]
post_slope_pooled_plt = boxplot(
    βs,
    xticks=(1:R, regions),
    title="Trends in package requests per region for $client_type",
    ylabel="Change per day (log10)",
    xlabel="Region",
    label=nothing,
)
savefig(post_slope_pooled_plt, save_path("package_request_posterior_trends.svg"))

βμ = post_chain["βμ"][:]
all_regions_plt = density(
    βμ,
    title="Population mean request trend for $client_type",
    label=nothing,
    xlabel="Change per day (log10)",
    ylabel="Density",
    fill=true,
    alpha=.5
)
vline!(all_regions_plt, [mean(βμ)], label="Mean")
savefig(all_regions_plt, save_path("package_request_posterior_mean_trend.svg"))