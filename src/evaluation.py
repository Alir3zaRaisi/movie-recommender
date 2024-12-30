def calculate_correctly_placed_pairs(recommendations, hidden_movies):
    """
    Calculate accuracy based on correctly placed pairs.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth hidden movies.

    Returns:
    - accuracy: Fraction of correctly placed pairs.
    """
    # Extract movie rankings from recommendations
    recommended_ranks = {movie: rank for rank, (movie, _) in enumerate(recommendations)}

    # Separate hidden and non-hidden movies
    hidden_movie_ranks = {movie: recommended_ranks[movie] for movie in hidden_movies if movie in recommended_ranks}
    non_hidden_movie_ranks = {movie: recommended_ranks[movie] for movie in recommended_ranks if movie not in hidden_movies}

    # Initialize pair counts
    correct_pairs = 0
    total_pairs = 0

    # Compare all pairs of (hidden, non-hidden)
    for _, hidden_rank in hidden_movie_ranks.items():
        for _, non_hidden_rank in non_hidden_movie_ranks.items():
            # A pair is correctly placed if the hidden movie is ranked higher (lower rank value)
            if hidden_rank < non_hidden_rank:
                correct_pairs += 1
            total_pairs += 1

    # Avoid division by zero
    accuracy = (correct_pairs / total_pairs) if total_pairs > 0 else 0
    return accuracy


def calculate_top_k_accuracy(recommendations, hidden_movies, k=10):
    """
    Calculate Top-K accuracy for recommendations.

    Parameters:
    - recommendations: List of tuples (movie, score) sorted by scores (output of rank_movies_with_penalty).
    - hidden_movies: Set of ground truth movies.
    - k: Number of top recommendations to consider.

    Returns:
    - accuracy: Fraction of top-K recommendations present in hidden_movies.
    """
    # Get the top-K recommended movies
    top_k_recommendations = [movie for movie, _ in recommendations[:k]]

    # Calculate intersection with hidden movies
    correct_recommendations = set(top_k_recommendations) & hidden_movies
    # Compute accuracy
    accuracy = len(correct_recommendations) / min(len(hidden_movies),k)
    return accuracy
