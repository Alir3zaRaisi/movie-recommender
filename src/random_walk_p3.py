import numpy as np
import networkx as nx
from collections import defaultdict

def perform_random_walks(graph, start_node, walk_length=3, num_walks=None):
    """
    Perform random walks from the start node.

    Parameters:
    - graph: NetworkX graph (bipartite with user and movie nodes).
    - start_node: Node to start the random walks.
    - walk_length: Length of each random walk.
    - num_walks: Number of random walks to perform (default is total edges // 3).

    Returns:
    - movie_visits: Dictionary with movie nodes as keys and visit counts as values.
    """
    # Determine number of walks
    if num_walks is None:
        num_walks = graph.number_of_nodes() // 3

    # Initialize movie visit counts
    movie_visits = defaultdict(int)

    # Perform random walks
    for _ in range(num_walks):
        current_node = start_node
        for _ in range(walk_length):
            neighbors = list(graph.neighbors(current_node))
            if not neighbors:
                break
            weights = [graph[current_node][neighbor].get("weight", 1) for neighbor in neighbors]
            probabilities = np.array(weights) / sum(weights)
            
            # Choose neighbor based on edge weights
            current_node = np.random.choice(neighbors, p=probabilities)
            if graph.nodes[current_node].get("bipartite") == 1:  # Movie nodes
                movie_visits[current_node] += 1

    return movie_visits


def rank_movies_with_penalty(graph, movie_visits, walk_length=3, penalty_type="b", total_walks=25):
    """
    Rank movies based on visit counts and penalties for missed visits in random walks.

    Parameters:
    - graph: NetworkX graph (bipartite with user and movie nodes).
    - movie_visits: Dictionary with movie nodes as keys and visit counts as values.
    - walk_length: Length of the random walks.
    - penalty_type: Penalty type ('a', 'b', or 'c').
    - total_walks: Total number of random walks performed.

    Returns:
    - sorted_movies: List of visited movies sorted by their scores (visits minus penalties).
    """
    # Estimate m̂ (number of edges in the subgraph within `walk_length` steps)
    bfs_edges = sum(1 for _ in nx.bfs_edges(graph, source=list(graph.nodes)[0], depth_limit=walk_length + 1))
    m_hat = bfs_edges  # Estimated number of edges

    # Prepare scores for visited movies only
    movie_scores = {}

    for movie, visited_times in movie_visits.items():
        # Calculate penalty based on unvisited walks
        unvisited_walks = total_walks - visited_times
        penalty = 0

        if penalty_type == "a":
            penalty = m_hat
        elif penalty_type == "b":
            degree = graph.degree[movie]
            penalty = (2 * m_hat) / degree if degree > 0 else 2 * m_hat  # Handle zero-degree cases
        elif penalty_type == "c":
            penalty = walk_length

        # Calculate the score: visits - penalty
        score = visited_times - (unvisited_walks * penalty)
        movie_scores[movie] = score

    # Sort visited movies by score in descending order
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1],reverse=True)
    return sorted_movies# Example usage

if __name__ == "__main__":
    # Load the bipartite graph
    graphml_path = "../data/ml-1m/user_movie_graph.graphml"
    graph = nx.read_graphml(graphml_path)

    # Specify a user node (e.g., 'user_1')
    user_node = 'user_2'

    # Generate movie recommendations for the user
    movie_visits = perform_random_walks(graph, start_node="user_1", walk_length=3)

# Rank movies with penalties
    sorted_movies = rank_movies_with_penalty(graph, movie_visits, walk_length=3, penalty_type="a")

# Display results
    print(sorted_movies)

