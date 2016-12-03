
Pkg.add("PGFPlots");
Pkg.add("Iterators");
Pkg.add("BayesNets");
Pkg.add("LightGraphs");
Pkg.add("TikzGraphs");
Pkg.add("Discretizers");
Pkg.add("RDatasets");
Pkg.add("Plots");

using Iterators
using LightGraphs
using BayesNets
using PGFPlots
using TikzGraphs
using Discretizers
using RDatasets
using Plots

movieData = readtable("C:\\Users\\Aaron Loh\\Documents\\Y16\\Autumn\\CS 238 - Decision Making Under Uncertainty\\Projects\\Final\\movieSuccessPredictions\\movie_metadata.csv");

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
nbinsLarge = 7
nbinsSmall = 5

gross_edges = binedges(DiscretizeUniformWidth(nbinsSmall), movieData[:gross])
gross_discretizer = LinearDiscretizer(gross_edges)

budget_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:budget])
budget_discretizer = LinearDiscretizer(budget_edges)

imdbScore_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:imdb_score])
imdbScore_discretizer = LinearDiscretizer(imdbScore_edges)

critic_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:num_critic_for_reviews])
critic_discretizer = LinearDiscretizer(critic_edges)

numMovieFacebookLikes_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:movie_facebook_likes])
numMovieFacebookLikes_discretizer = LinearDiscretizer(numMovieFacebookLikes_edges)

castMovieLikes_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:cast_total_facebook_likes])
castMovieLikes_discretizer = LinearDiscretizer(castMovieLikes_edges)

numFacesInPoster_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:facenumber_in_poster])
numFacesInPoster_discretizer = LinearDiscretizer(numFacesInPoster_edges)

directorFacebookLikes_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:director_facebook_likes])
directorFacebookLikes_discretizer = LinearDiscretizer(directorFacebookLikes_edges)

contentRating_discretizer = CategoricalDiscretizer(movieData[:content_rating])

duration_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:duration])
duration_discretizer = LinearDiscretizer(duration_edges)

titleYear_edges = binedges(DiscretizeUniformWidth(nbinsLarge), movieData[:title_year])
titleYear_discretizer = LinearDiscretizer(titleYear_edges)

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
    numGenres = movieData[:num_genres],
    imdbScore = encode(imdbScore_discretizer, movieData[:imdb_score]),
    numCriticReviews = encode(critic_discretizer, movieData[:num_critic_for_reviews]),
    numMovieFacebookLikes = encode(numMovieFacebookLikes_discretizer, movieData[:movie_facebook_likes]),
    castMovieLikes = encode(castMovieLikes_discretizer, movieData[:cast_total_facebook_likes]),
    numFacesInPoster = encode(numFacesInPoster_discretizer, movieData[:facenumber_in_poster]),
    directorFacebookLikes = encode(directorFacebookLikes_discretizer, movieData[:director_facebook_likes]),
    contentRating = encode(contentRating_discretizer, movieData[:content_rating]),
    duration = encode(duration_discretizer, movieData[:duration]),
    titleYear = encode(titleYear_discretizer, movieData[:title_year]),
);

totalSize = length(dataDiscretized[1])
percentageTrain = 0.9
lastTrainExample = Int(floor(percentageTrain*totalSize))

writetable("dataDiscretized.csv", dataDiscretized)
#dataUndiscretized = dataUndiscretized[1:20,:]
#display(dataUndiscretized)

dataDiscretizedTrain = dataDiscretized[1:lastTrainExample,:];
dataDiscretizedTest = dataDiscretized[lastTrainExample+1:totalSize,:];
#display(dataDiscretizedTrain)
#display(dataDiscretizedTest)

#structure learning
params = K2GraphSearch([:gross, :budget, :numGenres, :imdbScore, :numCriticReviews, :numMovieFacebookLikes, :castMovieLikes, :numFacesInPoster, :directorFacebookLikes, :contentRating, :duration, :titleYear],
                        DiscreteCPD,
                        max_n_parents=4);
#CategoricalCPD{Categorical{Float64}}
#DiscreteCPD

bn = fit(DiscreteBayesNet, dataDiscretizedTrain, params)

params2 = GreedyHillClimbing(ScoreComponentCache(dataDiscretized), max_n_parents=3, prior=UniformPrior())
bn2 = fit(DiscreteBayesNet, dataDiscretized, params2)


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

#getMostLikelyClass(table,5)

function getPredictionError(testDataTable)
    numMistakes = 0
    for i = 1:length(testDataTable[:,1])
        row = testDataTable[i,:]
        gross = row[:gross][1]
        budget = row[:budget][1]
        imdbScore = row[:imdbScore][1]
        numCriticReviews = row[:numCriticReviews][1]
        numGenres = row[:numGenres][1]
        println(i)
        table = rand_table_weighted(bn2; nsamples=5000, consistent_with=Assignment(:budget=>budget,:imdbScore=>imdbScore,:numCriticReviews=>numCriticReviews, :numGenres=>numGenres))
        #println("here")
        estimatedTable = estimate(table)

        predictedGrossCategory = getMostLikelyClass(estimatedTable,nbinsSmall)

        if predictedGrossCategory != gross
            numMistakes += 1
        end

    end
    return numMistakes/length(testDataTable[:,1])
end

function getMostLikelyClass(table,numClasses)
    highestLikelihood = 0
    mostLikelyClass = -1
    classProbabilities = zeros(numClasses)
    for i = 1:length(table[:,1])
        row = table[i,:]
        grossCategory = row[:gross][1]
        classProbabilities[grossCategory]+= row[:p][1]
        if classProbabilities[grossCategory] > highestLikelihood
            highestLikelihood = classProbabilities[grossCategory]
            mostLikelyClass = grossCategory
        end
    end
    return mostLikelyClass
end

error = getPredictionError(dataDiscretizedTest)

println(error)
# row = dataDiscretizedTest[34,:]
# display(row)
# display(critic_edges)
# #table(bn, :numCriticReviews)
# #count(bn, :numCriticReviews, dataDiscretized)
# table = rand_table_weighted(bn; nsamples=1000, consistent_with=Assignment(:budget=>1,:imdbScore=>6,:numCriticReviews=>4, :numGenres=>3))
# #estimatedTable = estimate(table)
#
# row = dataDiscretizedTest[2,:]
# #display(row)
# row[:numGenres][1]
# table = rand_table_weighted(bn; nsamples=100000, consistent_with=Assignment(:budget=>1,:imdbScore=>5,:numCriticReviews=>2, :numGenres=>3, :duration=>2, :contentRating=>3, :castMovieLikes=>1,:directorFacebookLikes=>1,:numFacesInPoster=>1,:numMovieFacebookLikes=>1, :titleYear=>7))
# estimatedTable = estimate(table)
#
# classProbabilities = zeros(5)
# classProbabilities[2]+=3
# classProbabilities
# length(estimatedTable[:,1])
# row = estimatedTable[2,:]
# display(row)
# row[:p][1]
# length(dataDiscretizedTest[:,1])

#structure learning
#params2 = K2GraphSearch([:gross, :budget, :imdbScore, :numCriticReviews, :numGenres],
#                       ConditionalLinearGaussianCPD,
#                       max_n_parents=2)
#bn2 = fit(BayesNet, dataUndiscretized, params2)



bayesian_score(bn2, dataDiscretizedTrain)
