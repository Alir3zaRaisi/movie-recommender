#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <queue>

// Define types for graph representation
using Node = int;
using Graph = std::unordered_map<Node, std::vector<std::pair<Node, double>>>;

// Perform random walks from a start node
std::unordered_map<Node, int> performRandomWalks(const Graph& graph, Node startNode, int walkLength, int numWalks) {
    std::unordered_map<Node, int> movieVisits;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < numWalks; ++i) {
        Node currentNode = startNode;
        for (int j = 0; j < walkLength; ++j) {
            auto neighbors = graph.find(currentNode);
            if (neighbors == graph.end() || neighbors->second.empty()) break;

            // Calculate probabilities for weighted random selection
            std::vector<double> weights;
            for (const auto& neighbor : neighbors->second) {
                weights.push_back(neighbor.second);
            }
            std::discrete_distribution<> dist(weights.begin(), weights.end());

            // Select a neighbor based on weights
            int index = dist(gen);
            currentNode = neighbors->second[index].first;

            // Count visits to movie nodes (assume movie nodes are >= 1000)
            if (currentNode >= 1000) {
                ++movieVisits[currentNode];
            }
        }
    }

    return movieVisits;
}

// Rank movies with penalties
std::vector<std::pair<Node, double>> rankMoviesWithPenalty(const Graph& graph, const std::unordered_map<Node, int>& movieVisits, 
                                                           int walkLength, char penaltyType, int totalWalks) {
    std::unordered_map<Node, double> movieScores;

    // Assume movies have IDs >= 1000
    for (const auto& [movie, visitedTimes] : movieVisits) {
        double penalty = 0;
        int unvisitedWalks = totalWalks - visitedTimes;
        int degree = graph.at(movie).size();

        // Calculate penalty based on type
        if (penaltyType == 'a') {
            penalty = walkLength;
        } else if (penaltyType == 'b') {
            penalty = (2.0 * walkLength) / (degree > 0 ? degree : 1);
        } else if (penaltyType == 'c') {
            penalty = walkLength;
        }

        // Calculate score
        double score = visitedTimes - (unvisitedWalks * penalty);
        movieScores[movie] = score;
    }

    // Sort movies by score
    std::vector<std::pair<Node, double>> sortedMovies(movieScores.begin(), movieScores.end());
    std::sort(sortedMovies.begin(), sortedMovies.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    return sortedMovies;
}

// Evaluate accuracy metrics
std::pair<double, double> evaluateAccuracy(const std::vector<std::pair<Node, double>>& recommendations, 
                                           const std::unordered_set<Node>& hiddenMovies) {
    int correctPairs = 0, totalPairs = 0;
    int topK = 10;

    std::unordered_map<Node, int> recommendedRanks;
    for (size_t i = 0; i < recommendations.size(); ++i) {
        recommendedRanks[recommendations[i].first] = i;
    }

    std::unordered_set<Node> topKMovies;
    for (int i = 0; i < std::min(topK, (int)recommendations.size()); ++i) {
        topKMovies.insert(recommendations[i].first);
    }

    // Top-K Accuracy
    int correctTopK = 0;
    for (const auto& movie : hiddenMovies) {
        if (topKMovies.find(movie) != topKMovies.end()) {
            ++correctTopK;
        }
    }
    double topKAccuracy = static_cast<double>(correctTopK) / std::min(topK, (int)hiddenMovies.size());

    // Pairwise Accuracy
    for (const auto& hiddenMovie : hiddenMovies) {
        if (recommendedRanks.find(hiddenMovie) == recommendedRanks.end()) continue;
        for (const auto& nonHiddenMovie : recommendedRanks) {
            if (hiddenMovies.find(nonHiddenMovie.first) != hiddenMovies.end()) continue;
            if (recommendedRanks[hiddenMovie] < nonHiddenMovie.second) {
                ++correctPairs;
            }
            ++totalPairs;
        }
    }

    double pairAccuracy = totalPairs > 0 ? static_cast<double>(correctPairs) / totalPairs : 0.0;

    return {topKAccuracy, pairAccuracy};
}

int main() {
    // Example graph (bipartite with user IDs < 1000 and movie IDs >= 1000)
    Graph graph;
    graph[0] = {{1000, 1.0}, {1001, 2.0}};
    graph[1] = {{1000, 3.0}, {1002, 1.5}};
    graph[1000] = {{0, 1.0}, {1, 3.0}};
    graph[1001] = {{0, 2.0}};
    graph[1002] = {{1, 1.5}};

    Node startNode = 0;
    int walkLength = 3;
    int numWalks = 5;

    auto movieVisits = performRandomWalks(graph, startNode, walkLength, numWalks);
    auto recommendations = rankMoviesWithPenalty(graph, movieVisits, walkLength, 'b', numWalks);

    std::unordered_set<Node> hiddenMovies = {1000, 1002};
    auto [topKAccuracy, pairAccuracy] = evaluateAccuracy(recommendations, hiddenMovies);

    std::cout << "Top-K Accuracy: " << topKAccuracy << "\n";
    std::cout << "Pairwise Accuracy: " << pairAccuracy << "\n";

    return 0;
}

