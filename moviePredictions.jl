
Pkg.add("PGFPlots")
Pkg.add("Iterators")
Pkg.add("BayesNets")
Pkg.add("LightGraphs")
Pkg.add("TikzGraphs")
Pkg.add("Discretizers")
Pkg.add("RDatasets")
Pkg.add("Plots")

using Iterators
using LightGraphs
using BayesNets
using PGFPlots
using TikzGraphs
using Discretizers
using RDatasets
using Plots

movieData = readtable("movie_metadata.csv")

#variables:
#Gross (query variable)
#Budget (evidence/input variable)
#Genre (evidence/input variable)
#IMDB score (evidence/input variable)
#Number of critic reviews (evidence/input variable)
#Number of Movie facebook likes
#Total cast facebook likes
#Number of faces in movie poster
#Director facebook likes
#Content rating
#Duration
#Title year


#data discretization
nbinsLarge = 5
nbinsSmall = 3

gross_edges = binedges(DiscretizeUniformCount(nbinsLarge), movieData[:gross])
gross_discretizer = LinearDiscretizer(gross_edges);
budget_edges = binedges(DiscretizeUniformCount(nbinsLarge), movieData[:budget])
budget_discretizer = LinearDiscretizer(budget_edges);

imdbScore_edges = binedges(DiscretizeUniformCount(nbinsLarge), movieData[:imdb_score])
imdbScore_discretizer = LinearDiscretizer(imdbScore_edges);
critic_edges = binedges(DiscretizeUniformCount(nbinsLarge), movieData[:num_critic_for_reviews])
critic_discretizer = LinearDiscretizer(critic_edges);



dataUndiscretized = DataFrame(
    gross = movieData[:gross], #encode(gross_discretizer, movieData[:gross]),
    budget = movieData[:budget],#encode(budget_discretizer, movieData[:budget]),
    imdbScore = movieData[:imdb_score],#encode(imdbScore_discretizer, movieData[:imdb_score]),
    numCriticReviews = movieData[:num_critic_for_reviews],#encode(critic_discretizer, movieData[:num_critic_for_reviews]),
    numGenres = movieData[:num_genres],
)

dataDiscretized = DataFrame(
    gross = encode(gross_discretizer, movieData[:gross]),
    budget = encode(budget_discretizer, movieData[:budget]),
    imdbScore = encode(imdbScore_discretizer, movieData[:imdb_score]),
    numCriticReviews = encode(critic_discretizer, movieData[:num_critic_for_reviews]),
    numGenres = movieData[:num_genres],
)

#writetable("output.csv", data)
dataUndiscretized = dataUndiscretized[1:20,:]
display(dataUndiscretized)

#dataDiscretized = dataDiscretized[1:1000,:]
display(dataDiscretized)

#structure learning
params = K2GraphSearch([:gross, :budget, :imdbScore, :numCriticReviews, :numGenres], 
                       CategoricalCPD{Categorical{Float64}},
                       max_n_parents=4)
bn = fit(BayesNet, dataDiscretized, params)

table = rand_table_weighted(bn; nsamples=300, consistent_with=Assignment(:imdbScore=>4, :numCriticReviews=>3))

function likelihoodWeightedSampling(table, value)
    numerator = 0
    denominator = 0  
    for i = 1:length(table[:,1])
        row = table[i,:]
        if row[:gross][1] == value
            numerator = numerator + row[:p][1]
        end
        denominator = denominator + row[:p][1]
    end
    return numerator./denominator
end

function getMostLikelyClass(table,numClasses)
    highestLikelihood = 0
    mostLikelyClass = -1
    for i = 1:numClasses
        likelihood = likelihoodWeightedSampling(table,i)
        display(likelihood)
        if likelihood > highestLikelihood
            highestLikelihood = likelihood
            mostLikelyClass = i
        end
    end
    return mostLikelyClass
end

getMostLikelyClass(table,5)

#structure learning
params2 = K2GraphSearch([:gross, :budget, :imdbScore, :numCriticReviews, :numGenres], 
                       ConditionalLinearGaussianCPD,
                       max_n_parents=2)
bn2 = fit(BayesNet, dataUndiscretized, params2)

params = GreedyHillClimbing(ScoreComponentCache(dataDiscretized), max_n_parents=3, prior=UniformPrior())
bn = fit(DiscreteBayesNet, dataDiscretized, params)

bayesian_score(bn, dataDiscretized, params.prior)
