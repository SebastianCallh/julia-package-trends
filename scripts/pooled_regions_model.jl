include("../src/common.jl")

out_path = joinpath(plots_path, "pooled-regions-model")
isdir(out_path) || mkdir(out_path)
save_path(file) = joinpath(out_path, file)

function plot_predictive_samples(chain, rr)
    pred = prediction(chain, rr)
    plot(
        day_transform(xx; inverse=true),
        group(pred, "y").value.data[:,:,1]',
        group=rr,
        title="Predictive samples",
        xlabel="Days",
        ylabel="Package downloads (log10)",
        label=nothing,
        color=:blues,
        alpha=.5
    )
end

function plot_predictive_distribution(chain, rr)
    pred = prediction(chain, rr)
    ss = summarystats(group(pred, "y"))
    plot(
        day_transform(xx; inverse=true),
        ss[:,:mean],
        ribbon=2*ss[:,:std],
        group=rr,
        title="Predictive distribution",
        xlabel="Days",
        ylabel="Package downloads (log10)",
        label=nothing,
        legend=:bottomright
    )
end

@model pooled_regions(x, y, r, R) = begin
    αμ ~ Normal(4.5, 0.5)
    ασ ~ TruncatedNormal(0, 1, 0, Inf)
    βμ ~ Normal(0, 2)
    βσ ~ TruncatedNormal(0, 1, 0, Inf)
    σ ~ TruncatedNormal(0, 0.6, 0, Inf)
    αoff ~ filldist(Normal(), R)
    βoff ~ filldist(Normal(), R)
    
    α ~ lazydist(Dirac, αμ .+ αoff .* ασ)
    β ~ lazydist(Dirac, βμ .+  βoff .* βσ)
    y ~ lazydist(Normal, α[r] .+ x.*β[r], σ)
end

const prediction(chain, t) = predict(pooled_regions(xx, missing, rr, R), chain)
model = pooled_regions(x, y, r, R)
prior_chain = sample(model, Prior(), 50; progress=false);
prior_pred_plt = plot_predictive_samples(prior_chain, rr)
scatter_data(prior_pred_plt, df)
plot(prior_chain)

@time post_chain = sample(model, NUTS(), 500);
plot(post_chain)


post_pred_pooled_plt = plot_predictive_distribution(post_chain, rr)
scatter_data(post_pred_pooled_plt, df)
savefig(post_pred_pooled_plt, save_path("package_request_posterior_predictive.svg"))

βs = group(post_chain, "β").value.data[:,:,1]
post_slope_pooled_plt = boxplot(
    βs,
    xticks=(1:R, regions),
    title="Trends in package requests per region",
    ylabel="Change per day (log10)",
    xlabel="Region",
    label=nothing,
)
savefig(post_slope_pooled_plt, save_path("package_request_posterior_trends.svg"))

βμ = post_chain["βμ"][:]
all_regions_plt = density(
    βμ,
    title="Population mean request trend",
    label=nothing,
    xlabel="Change per day (log10)",
    ylabel="Density",
    fill=true,
    alpha=.5
)
vline!(all_regions_plt, [mean(βμ)], label="Mean")
savefig(all_regions_plt, save_path("package_request_posterior_mean_trend.svg"))