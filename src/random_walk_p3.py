import numpy as np
import networkx as nx


def personalized_pagerank(graph, start_node, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Implements Personalized PageRank (P³) on a bipartite user-movie graph.
    """
    # Create an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(graph, weight="weight")
    nodes = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Ensure the start_node exists in the graph
    if start_node not in node_idx:
        raise KeyError(f"Start node {start_node} not found in the graph!")

    # Transition matrix: Normalize adjacency matrix row-wise
    transition_matrix = adjacency_matrix.copy()
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True)  # Sum of rows
    nonzero_rows = row_sums.flatten() != 0  # Identify rows with non-zero sums
    transition_matrix[nonzero_rows] /= row_sums[nonzero_rows]

    # Personalization vector: 1 for start_node, 0 for others
    personalization = np.zeros(len(nodes))
    personalization[node_idx[start_node]] = 1

    # Initialize PageRank scores
    scores = np.copy(personalization)

    for _ in range(max_iter):
        previous_scores = np.copy(scores)
        # PageRank formula: (1-alpha) * personalization + alpha * (M.T @ scores)
        scores = (1 - alpha) * personalization + alpha * transition_matrix.T @ scores

        # Check for convergence
        if np.linalg.norm(scores - previous_scores, 1) < tol:
            break

    # Map scores back to nodes
    scores_dict = {nodes[i]: scores[i] for i in range(len(nodes))}
    return scores_dict
def recommend_movies(graph, user_node, top_k=10):
    """
    Generate movie recommendations for a user based on P³ scores.

    Parameters:
    - graph: NetworkX Graph (bipartite graph of users and movies).
    - user_node: str, the user node ID (e.g., 'user_1').
    - top_k: int, number of top movie recommendations to return (default: 10).

    Returns:
    - recommendations: list of tuples (movie_node, score), sorted by score.
    """
    # Run Personalized PageRank starting from the user node
    scores = personalized_pagerank(graph, user_node)

    # Filter scores for movie nodes only
    movie_scores = {node: score for node, score in scores.items() if node.startswith("movie_")}

    # Sort movies by score in descending order
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return recommendations


# Example usage
if __name__ == "__main__":
    # Load the bipartite graph
    graphml_path = "../data/ml-1m/user_movie_graph.graphml"
    graph = nx.read_graphml(graphml_path)

    # Specify a user node (e.g., 'user_1')
    user_node = 'user_1'

    # Generate movie recommendations for the user
    top_recommendations = recommend_movies(graph, user_node, top_k=10)

    print("Top Movie Recommendations:")
    for movie, score in top_recommendations:
        print(f"{movie}: {score:.4f}")

